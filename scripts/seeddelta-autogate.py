#!/usr/bin/env python3

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

TENSOR_KEYS = ("ffn_gate", "ffn_up", "ffn_down")


def retention_score(metrics: dict, args: argparse.Namespace) -> float | None:
    mode = args.retention_mode
    if mode == "off":
        return None
    use_proxy = mode in ("proxy", "auto")
    use_cos = mode in ("cos", "auto")
    if use_proxy and metrics.get("ffn_proxy_available", False):
        cos = read_float(metrics.get("ffn_proxy_cos_p05"))
        l2 = read_float(metrics.get("ffn_proxy_l2_p95"))
        norm = read_float(metrics.get("ffn_proxy_log_norm_ratio_p95"))
        if cos is None or l2 is None or norm is None:
            return None
        score = max(0.0, cos) / (1.0 + l2 + abs(norm))
        return max(0.0, score)
    if use_cos:
        cos_p05 = read_float(metric_value(metrics, args.gate_cos_p05_metric, "cos_p05"))
        cos_mean = read_float(metric_value(metrics, args.gate_cos_metric, "cos_mean"))
        if cos_p05 is None:
            return None
        if cos_mean is None:
            cos_mean = cos_p05
        score = max(0.0, cos_p05) * max(0.0, cos_mean)
        return max(0.0, score)
    return None


def retention_layer_score(layer_scores: dict[str, float], aggregate: str) -> float | None:
    if not layer_scores:
        return None
    vals = [v for v in layer_scores.values() if v is not None]
    if not vals:
        return None
    if aggregate == "mean":
        return sum(vals) / len(vals)
    return max(vals)


def apply_retention_adjust(val: float, retention: float | None, rank_is_higher: bool, alpha: float) -> float:
    if retention is None or alpha <= 0.0:
        return val
    penalty = alpha * (1.0 - retention)
    if rank_is_higher:
        return val * (1.0 - penalty)
    return val * (1.0 + penalty)


def parse_layers(spec: str) -> list[int]:
    layers: list[int] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            lo_s, hi_s = token.split("-", 1)
            lo = int(lo_s)
            hi = int(hi_s)
            if hi < lo:
                lo, hi = hi, lo
            layers.extend(range(lo, hi + 1))
        else:
            layers.append(int(token))
    return sorted(set(layers))


def parse_layers_set(spec: str) -> set[int] | None:
    if not spec or not spec.strip():
        return None
    return set(parse_layers(spec))


def write_policy(
    path: Path,
    layers: list[int],
    layer_tensors: dict[int, set[str]],
    k: int,
    metric: str,
    enable_down: bool,
    strip_dense: bool,
) -> None:
    ranges = []
    for layer in layers:
        allowed = set(layer_tensors.get(layer, set()))
        if not enable_down and "ffn_down" in allowed:
            allowed.remove("ffn_down")
        ranges.append(
            {
                "layers": str(layer),
                "enabled": True,
                "strip_dense": bool(strip_dense),
                "K": {"gate": k, "up": k, "down": k},
                "gating": {
                    "metric": metric,
                    "min_mean": {"gate": 0.0, "up": 0.0, "down": 0.0},
                    "min_p05": {"gate": 0.0, "up": 0.0, "down": 0.0},
                },
                "tensors": {
                    "ffn_gate": {"enabled": "ffn_gate" in allowed},
                    "ffn_up": {"enabled": "ffn_up" in allowed},
                    "ffn_down": {"enabled": "ffn_down" in allowed},
                },
                "autotune": {"enabled": False},
            }
        )
    policy = {
        "version": 1,
        "defaults": {
            "enabled": False,
            "gating": {"metric": metric, "min_mean": -1.0, "min_p05": -1.0},
            "autotune": {"enabled": False},
        },
        "ranges": ranges,
    }
    path.write_text(json.dumps(policy, indent=2) + "\n", encoding="utf-8")


def load_scan(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_tensors(layer_metrics: dict) -> dict:
    if isinstance(layer_metrics.get("tensors"), dict):
        return layer_metrics.get("tensors", {})
    return layer_metrics


def metric_value(metrics: dict, primary: str, fallback: str) -> float | None:
    val = metrics.get(primary)
    if val is None:
        return metrics.get(fallback)
    if fallback and isinstance(val, (int, float)) and val == 0.0:
        fb = metrics.get(fallback)
        if fb is not None and fb != 0.0:
            return fb
    return val


def validate_scan(layers_dict: dict) -> list[str]:
    errors: list[str] = []
    for layer_s, layer_metrics in layers_dict.items():
        tensors = extract_tensors(layer_metrics) if isinstance(layer_metrics, dict) else None
        if not isinstance(tensors, dict) or not tensors:
            errors.append(f"layer {layer_s}: missing tensors")
            continue
        for tensor_name, metrics in tensors.items():
            if not isinstance(metrics, dict):
                errors.append(f"layer {layer_s}:{tensor_name} invalid metrics")
                continue
            if "S_mean" not in metrics and "rel_l2_mean" not in metrics:
                errors.append(f"layer {layer_s}:{tensor_name} missing S_mean")
    return errors


def scan_meta_value(scan: dict, key: str) -> object:
    meta = scan.get("meta")
    if isinstance(meta, dict) and key in meta:
        return meta.get(key)
    return scan.get(key)


def validate_scan_meta(scan: dict, args: argparse.Namespace) -> None:
    mismatches: list[tuple[str, object, object]] = []

    def chk(name: str, got: object, want: object) -> None:
        if got is None or want is None:
            return
        if str(got) != str(want):
            mismatches.append((name, got, want))

    chk("scheme", scan_meta_value(scan, "scheme"), args.eval_scheme)
    if args.eval_scheme == "block":
        chk("block", scan_meta_value(scan, "block"), args.eval_block)
    chk("eval_x", scan_meta_value(scan, "eval_x"), args.eval_x)
    chk("eval_cols", scan_meta_value(scan, "eval_cols"), args.eval_cols)
    chk("K_fixed", scan_meta_value(scan, "K_fixed"), args.k)

    if mismatches:
        msg_lines = ["scan/meta mismatch:"]
        msg_lines.extend(f"  - {name}: scan={got} != args={want}" for name, got, want in mismatches)
        msg = "\n".join(msg_lines)
        if args.strict_scan_meta:
            raise SystemExit(msg)
        print(f"warn: {msg}", file=sys.stderr)

    scan_k = scan_meta_value(scan, "K_fixed")
    if scan_k is not None and int(scan_k) <= args.warn_small_k:
        print(
            f"warn: scan K_fixed={scan_k} is very small; expect quality drops unless calibrated",
            file=sys.stderr,
        )
    if args.k <= args.warn_small_k and (scan_k is None or int(args.k) != int(scan_k)):
        print(
            f"warn: policy K={args.k} is very small; consider matching scan K",
            file=sys.stderr,
        )


def read_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def calibrate_functional_thresholds(
    layers_dict: dict,
    good_layers: list[int],
    tensors: list[str],
    args: argparse.Namespace,
) -> dict[str, dict[str, float]]:
    thresholds: dict[str, dict[str, float]] = {}
    for tensor in tensors:
        cos_vals: list[float] = []
        l2_vals: list[float] = []
        norm_vals: list[float] = []
        for layer in good_layers:
            layer_metrics = layers_dict.get(str(layer), {})
            layer_tensors = extract_tensors(layer_metrics) if isinstance(layer_metrics, dict) else {}
            metrics = layer_tensors.get(tensor)
            if not isinstance(metrics, dict):
                continue
            if not metrics.get("ffn_proxy_available", False):
                continue
            cos = read_float(metrics.get("ffn_proxy_cos_p05"))
            l2 = read_float(metrics.get("ffn_proxy_l2_p95"))
            norm = read_float(metrics.get("ffn_proxy_log_norm_ratio_p95"))
            if cos is None or l2 is None or norm is None:
                continue
            cos_vals.append(cos)
            l2_vals.append(l2)
            norm_vals.append(abs(norm))
        if not cos_vals:
            continue
        cos_thr = min(cos_vals)
        if args.functional_relax_cos:
            cos_thr -= args.functional_cos_margin
        thresholds[tensor] = {
            "cos_p05": cos_thr,
            "l2_p95": max(l2_vals) * (1.0 + args.functional_l2_margin),
            "norm_p95": max(norm_vals) * (1.0 + args.functional_norm_margin),
        }
    return thresholds


def resolve_functional_thresholds(
    args: argparse.Namespace,
    tensors: list[str],
    calibrated: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    resolved: dict[str, dict[str, float]] = {}
    for tensor in tensors:
        suffix = tensor.replace("ffn_", "")
        cos_override = getattr(args, f"functional_cos_p05_threshold_{suffix}")
        l2_override = getattr(args, f"functional_l2_threshold_{suffix}")
        norm_override = getattr(args, f"functional_norm_threshold_{suffix}")

        base = calibrated.get(tensor, {})
        cos_thr = cos_override if cos_override is not None else base.get("cos_p05", args.functional_cos_p05_threshold)
        l2_thr = l2_override if l2_override is not None else base.get("l2_p95", args.functional_l2_threshold)
        norm_thr = norm_override if norm_override is not None else base.get("norm_p95", args.functional_norm_threshold)

        if cos_thr is None or l2_thr is None or norm_thr is None:
            continue

        resolved[tensor] = {
            "cos_p05": float(cos_thr),
            "l2_p95": float(l2_thr),
            "norm_p95": float(norm_thr),
        }
    return resolved


def passes_functional_gate(
    layer: int,
    metrics: dict,
    thresholds: dict[str, float] | None,
    args: argparse.Namespace,
) -> bool:
    if thresholds is None:
        return args.functional_allow_missing
    if not metrics.get("ffn_proxy_available", False):
        return args.functional_allow_missing

    cos = read_float(metrics.get("ffn_proxy_cos_p05"))
    l2 = read_float(metrics.get("ffn_proxy_l2_p95"))
    norm = read_float(metrics.get("ffn_proxy_log_norm_ratio_p95"))
    if cos is None or l2 is None or norm is None:
        return args.functional_allow_missing

    cos_thr = thresholds["cos_p05"]
    l2_thr = thresholds["l2_p95"]
    norm_thr = thresholds["norm_p95"]

    if layer < args.functional_early_layer_cutoff and not args.functional_allow_early:
        return False
    if layer < args.functional_early_layer_cutoff:
        cos_thr += args.functional_early_cos_boost
        l2_thr *= args.functional_early_l2_scale
        norm_thr *= args.functional_early_norm_scale

    return cos >= cos_thr and l2 <= l2_thr and abs(norm) <= norm_thr

def aggregate_metrics(layer_metrics: dict, tensors: list[str], metric: str, agg: str) -> float | None:
    layer_tensors = extract_tensors(layer_metrics)
    vals: list[float] = []
    for t in tensors:
        m = layer_tensors.get(t)
        if not isinstance(m, dict):
            continue
        v = m.get(metric)
        if v is None:
            continue
        vals.append(float(v))
    if not vals:
        return None
    if agg == "max":
        return max(vals)
    return sum(vals) / float(len(vals))


def run_eval(eval_script: Path, args: list[str]) -> None:
    cmd = [str(eval_script)] + args
    subprocess.run(cmd, check=True)


def parse_ppl(path: Path) -> float | None:
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("Final estimate: PPL ="):
            try:
                return float(line.split("=")[1].split()[0])
            except (IndexError, ValueError):
                return None
    return None


def parse_greedy_result(path: Path) -> bool | None:
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        if "RESULT: PASS" in line:
            return True
        if "RESULT: FAIL" in line:
            return False
    return None


def decide_allowed_tensors(
    layer: int,
    layer_metrics: dict,
    args: argparse.Namespace,
    functional_thresholds: dict[str, dict[str, float]],
    retention_scores: dict[int, dict[str, float]],
    retention_allowlist: dict[str, set[int]] | None,
) -> list[str]:
    allowed: list[str] = []
    layer_tensors = extract_tensors(layer_metrics)
    for tensor in TENSOR_KEYS:
        if tensor == "ffn_down" and not args.enable_down:
            continue
        if tensor == "ffn_up" and not args.allow_up:
            continue
        suffix = tensor.replace("ffn_", "")
        mean_override = getattr(args, f"gate_cos_threshold_{suffix}")
        p05_override = getattr(args, f"gate_cos_p05_threshold_{suffix}")
        mean_thr = mean_override if mean_override is not None else args.gate_cos_threshold
        p05_thr = p05_override if p05_override is not None else args.gate_cos_p05_threshold
        metrics = layer_tensors.get(tensor)
        if not isinstance(metrics, dict):
            continue
        cos_mean = metric_value(metrics, args.gate_cos_metric, "cos_mean")
        cos_p05 = metric_value(metrics, args.gate_cos_p05_metric, "cos_p05")
        if cos_mean is None or cos_p05 is None:
            if args.gate_allow_missing:
                allowed.append(tensor)
            continue
        if cos_mean < mean_thr or cos_p05 < p05_thr:
            continue
        if args.functional_gate:
            thresholds = functional_thresholds.get(tensor)
            if not passes_functional_gate(layer, metrics, thresholds, args):
                continue
        if args.retention_mode != "off":
            score = retention_scores.get(layer, {}).get(tensor)
            if score is None:
                if not args.retention_allow_missing:
                    continue
            else:
                if args.retention_min > 0.0 and score < args.retention_min:
                    continue
                if retention_allowlist is not None and layer not in retention_allowlist.get(tensor, set()):
                    continue
        allowed.append(tensor)
    return allowed


def main() -> int:
    ap = argparse.ArgumentParser(description="Auto-gate layer selection from scan + optional eval")
    ap.add_argument("--scan", required=True, help="Path to layer_sensitivity_scan.json")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--k", type=int, default=12, help="Fixed K for policy")
    ap.add_argument("--strict-scan-meta", action="store_true", help="Error on scan/arg mismatches")
    ap.add_argument("--warn-small-k", type=int, default=32, help="Warn when K <= this threshold")
    ap.add_argument("--metric", default="cos", help="Gating metric for policy")
    ap.add_argument("--rank-metric", default="S_mean", help="Scan metric for ranking")
    ap.add_argument("--aggregate", default="mean", choices=["mean", "max"], help="Aggregate per-layer metrics")
    ap.add_argument("--tensors", default="ffn_gate,ffn_up", help="CSV tensors to rank")
    ap.add_argument("--max-layers", type=int, default=0, help="Max layers to select (0=all)")
    ap.add_argument("--percent", type=float, default=0.0, help="Percent of layers to select (0=disabled)")
    ap.add_argument("--threshold", type=float, default=0.0, help="Score threshold (0=disabled)")
    ap.add_argument("--min-gap", type=int, default=1, help="Min gap between layers (1=no consecutive)")
    ap.add_argument("--prefer", default="1,3,5,6", help="CSV layers to prefer (bias)")
    ap.add_argument("--prefer-mult", type=float, default=0.95, help="Score multiplier for preferred layers")
    ap.add_argument("--avoid-even-only", action="store_true", help="Reject sets that are only even layers")
    ap.add_argument("--enable-down", action="store_true", help="Enable ffn_down in policy")
    ap.add_argument("--allow-up", action="store_true", help="Allow ffn_up in policy")
    ap.add_argument("--gate-cos-metric", default="cos_mean_x_w", help="Metric for mean cosine gate")
    ap.add_argument("--gate-cos-p05-metric", default="cos_p05_x_w", help="Metric for p05 cosine gate")
    ap.add_argument("--gate-cos-threshold", type=float, default=0.7, help="Min cosine mean threshold")
    ap.add_argument("--gate-cos-p05-threshold", type=float, default=0.5, help="Min cosine p05 threshold")
    ap.add_argument(
        "--gate-cos-threshold-gate",
        type=float,
        default=None,
        help="Per-tensor cosine mean threshold for ffn_gate",
    )
    ap.add_argument(
        "--gate-cos-p05-threshold-gate",
        type=float,
        default=None,
        help="Per-tensor cosine p05 threshold for ffn_gate",
    )
    ap.add_argument(
        "--gate-cos-threshold-up",
        type=float,
        default=None,
        help="Per-tensor cosine mean threshold for ffn_up",
    )
    ap.add_argument(
        "--gate-cos-p05-threshold-up",
        type=float,
        default=None,
        help="Per-tensor cosine p05 threshold for ffn_up",
    )
    ap.add_argument(
        "--gate-cos-threshold-down",
        type=float,
        default=None,
        help="Per-tensor cosine mean threshold for ffn_down",
    )
    ap.add_argument(
        "--gate-cos-p05-threshold-down",
        type=float,
        default=None,
        help="Per-tensor cosine p05 threshold for ffn_down",
    )
    ap.add_argument("--gate-allow-missing", action="store_true", help="Allow tensors with missing cos metrics")
    ap.add_argument(
        "--retention-mode",
        default="off",
        choices=["off", "proxy", "cos", "auto"],
        help="Retention score source (proxy or cos).",
    )
    ap.add_argument("--retention-min", type=float, default=0.0, help="Min retention score to keep tensor")
    ap.add_argument("--retention-topk", type=int, default=0, help="Max layers per tensor by retention score")
    ap.add_argument("--retention-aggregate", default="max", choices=["max", "mean"], help="Layer retention aggregate")
    ap.add_argument(
        "--retention-decay-alpha",
        type=float,
        default=0.0,
        help="Penalize low retention in ranking (0=disabled)",
    )
    ap.add_argument(
        "--retention-allow-missing",
        action="store_true",
        help="Allow tensors with missing retention score when retention_mode is active",
    )
    ap.add_argument(
        "--functional-gate",
        action="store_true",
        help="Enable functional proxy gate using ffn_proxy_* metrics",
    )
    ap.add_argument("--functional-good-layers", default="", help="CSV layers to calibrate thresholds")
    ap.add_argument(
        "--functional-cos-margin",
        type=float,
        default=0.01,
        help="Cos margin below min ffn_proxy_cos_p05 (requires --functional-relax-cos)",
    )
    ap.add_argument(
        "--functional-relax-cos",
        action="store_true",
        help="Allow relaxing ffn_proxy_cos_p05 below the worst good layer",
    )
    ap.add_argument(
        "--functional-l2-margin",
        type=float,
        default=0.15,
        help="Margin above max ffn_proxy_l2_p95 (calibration)",
    )
    ap.add_argument(
        "--functional-norm-margin",
        type=float,
        default=0.15,
        help="Margin above max |ffn_proxy_log_norm_ratio_p95| (calibration)",
    )
    ap.add_argument(
        "--functional-cos-p05-threshold",
        type=float,
        default=None,
        help="Min ffn_proxy_cos_p05 threshold",
    )
    ap.add_argument(
        "--functional-l2-threshold",
        type=float,
        default=None,
        help="Max ffn_proxy_l2_p95 threshold",
    )
    ap.add_argument(
        "--functional-norm-threshold",
        type=float,
        default=None,
        help="Max abs ffn_proxy_log_norm_ratio_p95 threshold",
    )
    ap.add_argument(
        "--functional-cos-p05-threshold-gate",
        type=float,
        default=None,
        help="Per-tensor min ffn_proxy_cos_p05 for ffn_gate",
    )
    ap.add_argument(
        "--functional-l2-threshold-gate",
        type=float,
        default=None,
        help="Per-tensor max ffn_proxy_l2_p95 for ffn_gate",
    )
    ap.add_argument(
        "--functional-norm-threshold-gate",
        type=float,
        default=None,
        help="Per-tensor max abs ffn_proxy_log_norm_ratio_p95 for ffn_gate",
    )
    ap.add_argument(
        "--functional-cos-p05-threshold-up",
        type=float,
        default=None,
        help="Per-tensor min ffn_proxy_cos_p05 for ffn_up",
    )
    ap.add_argument(
        "--functional-l2-threshold-up",
        type=float,
        default=None,
        help="Per-tensor max ffn_proxy_l2_p95 for ffn_up",
    )
    ap.add_argument(
        "--functional-norm-threshold-up",
        type=float,
        default=None,
        help="Per-tensor max abs ffn_proxy_log_norm_ratio_p95 for ffn_up",
    )
    ap.add_argument(
        "--functional-cos-p05-threshold-down",
        type=float,
        default=None,
        help="Per-tensor min ffn_proxy_cos_p05 for ffn_down",
    )
    ap.add_argument(
        "--functional-l2-threshold-down",
        type=float,
        default=None,
        help="Per-tensor max ffn_proxy_l2_p95 for ffn_down",
    )
    ap.add_argument(
        "--functional-norm-threshold-down",
        type=float,
        default=None,
        help="Per-tensor max abs ffn_proxy_log_norm_ratio_p95 for ffn_down",
    )
    ap.add_argument(
        "--functional-early-layer-cutoff",
        type=int,
        default=12,
        help="Early layer cutoff for stricter functional thresholds",
    )
    ap.add_argument(
        "--functional-allow-early",
        action="store_true",
        help="Allow functional gate to consider layers < early cutoff",
    )
    ap.add_argument(
        "--functional-early-cos-boost",
        type=float,
        default=0.02,
        help="Boost ffn_proxy_cos_p05 threshold for early layers",
    )
    ap.add_argument(
        "--functional-early-l2-scale",
        type=float,
        default=0.85,
        help="Scale ffn_proxy_l2_p95 threshold for early layers",
    )
    ap.add_argument(
        "--functional-early-norm-scale",
        type=float,
        default=0.85,
        help="Scale ffn_proxy_log_norm_ratio_p95 threshold for early layers",
    )
    ap.add_argument(
        "--functional-allow-missing",
        action="store_true",
        help="Allow tensors with missing ffn_proxy metrics",
    )
    ap.add_argument("--policy-out", default="", help="Output policy path (default: outdir/policy.autogen.json)")
    ap.add_argument(
        "--policy-strip-dense",
        action="store_true",
        help="Set strip_dense=true in generated policy ranges",
    )
    ap.add_argument("--forbidden-out", default="", help="Output forbidden pairs JSON (optional)")
    ap.add_argument("--scores-out", default="", help="Output scores JSON (optional)")

    ap.add_argument("--eval", action="store_true", help="Run incremental eval and rollback on failures")
    ap.add_argument("--eval-script", default="./scripts/seeddelta-policy-eval.sh", help="Eval script")
    ap.add_argument("--base", default="", help="Base GGUF for eval")
    ap.add_argument("--text", default="", help="Text file for eval PPL")
    ap.add_argument("--layers-range", default="", help="Layer range for builder (e.g. 0-31)")
    ap.add_argument("--layer-allowlist", default="", help="Layer allowlist (e.g. 12,14,16,18)")
    ap.add_argument("--layer-denylist", default="", help="Layer denylist (e.g. 0-11)")
    ap.add_argument("--eval-outdir", default="", help="Base eval outdir")
    ap.add_argument("--eval-threads", type=int, default=os.cpu_count() or 1, help="Eval threads")
    ap.add_argument("--eval-ctx", type=int, default=512, help="Eval context")
    ap.add_argument("--eval-chunks", type=int, default=16, help="Eval PPL chunks")
    ap.add_argument("--eval-scheme", default="coo", help="Eval scheme")
    ap.add_argument("--eval-block", type=int, default=32, help="Eval block size")
    ap.add_argument("--eval-x", type=int, default=32, help="Eval x")
    ap.add_argument("--eval-cols", type=int, default=64, help="Eval cols")
    ap.add_argument("--eval-greedy-pack", default="", help="Greedy pack file")
    ap.add_argument("--max-delta-ppl", type=float, default=5.0, help="Max DeltaPPL percent")
    ap.add_argument(
        "--ctx-sweep",
        default="",
        help="Comma-separated ctx values to evaluate final policy (e.g. 512,768,1024)",
    )
    args = ap.parse_args()

    scan = load_scan(Path(args.scan))
    layers_dict = scan.get("layers", {})
    if not isinstance(layers_dict, dict):
        print("error: scan.layers missing or invalid", file=sys.stderr)
        return 2

    scan_errors = validate_scan(layers_dict)
    for err in scan_errors:
        print(f"warn: scan {err}", file=sys.stderr)
    validate_scan_meta(scan, args)

    tensors = [t.strip() for t in args.tensors.split(",") if t.strip()]
    if not args.allow_up:
        tensors = [t for t in tensors if t != "ffn_up"]
    prefer_layers = set(parse_layers(args.prefer)) if args.prefer else set()
    allow_layers = parse_layers_set(args.layer_allowlist)
    deny_layers = parse_layers_set(args.layer_denylist) or set()

    functional_thresholds: dict[str, dict[str, float]] = {}
    if args.functional_gate:
        good_layers = parse_layers(args.functional_good_layers) if args.functional_good_layers else []
        calibrated: dict[str, dict[str, float]] = {}
        if good_layers:
            calibrated = calibrate_functional_thresholds(layers_dict, good_layers, tensors, args)
            if not calibrated:
                print("error: functional calibration produced no thresholds", file=sys.stderr)
                return 2
        functional_thresholds = resolve_functional_thresholds(args, tensors, calibrated)
        missing = [t for t in tensors if t not in functional_thresholds]
        if missing:
            print(f"error: functional thresholds missing for {', '.join(missing)}", file=sys.stderr)
            return 2

    retention_scores: dict[int, dict[str, float]] = {}
    for layer_s, layer_metrics in layers_dict.items():
        try:
            layer = int(layer_s)
        except ValueError:
            continue
        if allow_layers is not None and layer not in allow_layers:
            continue
        if layer in deny_layers:
            continue
        if not isinstance(layer_metrics, dict):
            continue
        layer_tensors = extract_tensors(layer_metrics)
        for tensor in TENSOR_KEYS:
            metrics = layer_tensors.get(tensor)
            if not isinstance(metrics, dict):
                continue
            score = retention_score(metrics, args)
            if score is None:
                continue
            retention_scores.setdefault(layer, {})[tensor] = float(score)

    retention_allowlist: dict[str, set[int]] | None = None
    if args.retention_topk > 0 and retention_scores:
        retention_allowlist = {tensor: set() for tensor in TENSOR_KEYS}
        for tensor in TENSOR_KEYS:
            ranked = []
            for layer, layer_scores in retention_scores.items():
                score = layer_scores.get(tensor)
                if score is None:
                    continue
                ranked.append((score, layer))
            ranked.sort(reverse=True)
            for _, layer in ranked[: args.retention_topk]:
                retention_allowlist[tensor].add(layer)

    scores: dict[int, float] = {}
    for layer_s, layer_metrics in layers_dict.items():
        try:
            layer = int(layer_s)
        except ValueError:
            continue
        if allow_layers is not None and layer not in allow_layers:
            continue
        if layer in deny_layers:
            continue
        if not isinstance(layer_metrics, dict):
            continue
        if args.rank_metric == "S_layer" and "S_layer" in layer_metrics:
            val = float(layer_metrics.get("S_layer", 0.0))
        else:
            val = aggregate_metrics(layer_metrics, tensors, args.rank_metric, args.aggregate)
        if val is None:
            continue
        if args.retention_decay_alpha > 0.0:
            layer_ret = retention_layer_score(retention_scores.get(layer, {}), args.retention_aggregate)
            val = apply_retention_adjust(val, layer_ret, args.rank_metric.startswith("cos"), args.retention_decay_alpha)
        scores[layer] = float(val)

    if not scores:
        print("error: no scores computed", file=sys.stderr)
        return 2

    rank_is_higher = args.rank_metric.startswith("cos")
    ranked = []
    for layer, val in scores.items():
        score = -val if rank_is_higher else val
        if layer in prefer_layers:
            score *= args.prefer_mult
        ranked.append((score, layer))
    ranked.sort()

    max_layers = args.max_layers if args.max_layers > 0 else None
    if args.percent and args.percent > 0.0:
        max_layers = max(1, int(math.ceil(len(ranked) * (args.percent / 100.0))))

    selected: list[int] = []
    selected_tensors: dict[int, set[str]] = {}
    forbidden_pairs: set[tuple[int, int]] = set()
    eval_steps: list[dict] = []
    eval_script = Path(args.eval_script)
    eval_outdir = Path(args.eval_outdir) if args.eval_outdir else Path(args.outdir) / "eval"
    eval_outdir.mkdir(parents=True, exist_ok=True)

    for _, layer in ranked:
        if max_layers is not None and len(selected) >= max_layers:
            break
        if args.threshold and args.threshold > 0.0:
            val = scores.get(layer, 0.0)
            if (not rank_is_higher and val > args.threshold) or (rank_is_higher and val < args.threshold):
                continue
        if args.min_gap > 0 and any(abs(layer - s) <= args.min_gap for s in selected):
            continue
        if args.avoid_even_only:
            if (not selected and layer % 2 == 0) or (all(s % 2 == 0 for s in selected) and layer % 2 == 0):
                continue
        layer_metrics = layers_dict.get(str(layer), {})
        allowed = decide_allowed_tensors(
            layer,
            layer_metrics,
            args,
            functional_thresholds,
            retention_scores,
            retention_allowlist,
        )
        if not allowed:
            continue

        candidate = selected + [layer]
        candidate.sort()
        candidate_tensors = dict(selected_tensors)
        candidate_tensors[layer] = set(allowed)

        if not args.eval:
            selected = candidate
            selected_tensors = candidate_tensors
            continue

        if not args.base or not args.text:
            print("error: --eval requires --base and --text", file=sys.stderr)
            return 2

        policy_path = eval_outdir / f"policy.step{len(eval_steps):03d}.json"
        write_policy(
            policy_path,
            candidate,
            candidate_tensors,
            args.k,
            args.metric,
            args.enable_down,
            args.policy_strip_dense,
        )

        layer_range = args.layers_range
        if not layer_range:
            layer_range = f"{candidate[0]}-{candidate[-1]}"

        run_dir = eval_outdir / f"step_{len(eval_steps):03d}"
        args_list = [
            "--base",
            args.base,
            "--policy",
            str(policy_path),
            "--layers",
            layer_range,
            "--text",
            args.text,
            "--threads",
            str(args.eval_threads),
            "--ctx",
            str(args.eval_ctx),
            "--chunks",
            str(args.eval_chunks),
            "--scheme",
            args.eval_scheme,
            "--outdir",
            str(run_dir),
            "--eval-x",
            str(args.eval_x),
            "--eval-cols",
            str(args.eval_cols),
        ]
        if args.eval_scheme == "block":
            args_list.extend(["--block", str(args.eval_block)])
        if args.eval_greedy_pack:
            args_list.extend(["--greedy-pack", args.eval_greedy_pack])

        run_eval(eval_script, args_list)

        base_ppl = parse_ppl(run_dir / "perplexity_base.log")
        sd_ppl = parse_ppl(run_dir / "perplexity_seeddelta.log")
        greedy_ok = parse_greedy_result(run_dir / "greedy_pack.log")

        delta_ok = True
        delta_pct = None
        if base_ppl and sd_ppl:
            delta_pct = (sd_ppl - base_ppl) / base_ppl * 100.0
            if delta_pct > args.max_delta_ppl:
                delta_ok = False

        pass_ok = (greedy_ok is not False) and delta_ok

        eval_steps.append(
            {
                "layer": layer,
                "candidate": candidate,
                "tensors": sorted(allowed),
                "run_dir": str(run_dir),
                "base_ppl": base_ppl,
                "seeddelta_ppl": sd_ppl,
                "delta_pct": delta_pct,
                "greedy_ok": greedy_ok,
                "pass": pass_ok,
            }
        )

        if pass_ok:
            selected = candidate
            selected_tensors = candidate_tensors
        else:
            for s in selected:
                pair = tuple(sorted((s, layer)))
                forbidden_pairs.add(pair)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    policy_out = Path(args.policy_out) if args.policy_out else outdir / "policy.autogen.json"
    write_policy(
        policy_out,
        selected,
        selected_tensors,
        args.k,
        args.metric,
        args.enable_down,
        args.policy_strip_dense,
    )

    scores_out = Path(args.scores_out) if args.scores_out else outdir / "layer_scores.json"
    scores_out.write_text(json.dumps(scores, indent=2) + "\n", encoding="utf-8")

    if args.forbidden_out or forbidden_pairs:
        forb_out = Path(args.forbidden_out) if args.forbidden_out else outdir / "forbidden_pairs.json"
        forb_out.write_text(
            json.dumps(sorted([list(p) for p in forbidden_pairs]), indent=2) + "\n",
            encoding="utf-8",
        )

    if eval_steps:
        selected_map = {str(layer): sorted(list(selected_tensors.get(layer, set()))) for layer in selected}
        report = {
            "selected_layers": selected,
            "selected_tensors": selected_map,
            "steps": eval_steps,
        }
        (outdir / "autogating_report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )

    if args.ctx_sweep:
        if not args.base or not args.text:
            print("error: --ctx-sweep requires --base and --text", file=sys.stderr)
            return 2
        ctx_list: list[int] = []
        seen: set[int] = set()
        for part in args.ctx_sweep.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                ctx_val = int(part)
            except ValueError:
                print(f"error: invalid ctx value '{part}' in --ctx-sweep", file=sys.stderr)
                return 2
            if ctx_val in seen:
                continue
            seen.add(ctx_val)
            ctx_list.append(ctx_val)
        if not ctx_list:
            print("error: --ctx-sweep has no valid ctx values", file=sys.stderr)
            return 2

        layer_range = args.layers_range
        if not layer_range and selected:
            layer_range = f"{min(selected)}-{max(selected)}"
        if not layer_range:
            print("error: --ctx-sweep needs a non-empty layer range", file=sys.stderr)
            return 2

        sweep: list[dict[str, object]] = []
        for ctx in ctx_list:
            run_dir = outdir / f"ctx_sweep_ctx{ctx}"
            args_list = [
                "--base",
                args.base,
                "--policy",
                str(policy_out),
                "--layers",
                layer_range,
                "--text",
                args.text,
                "--threads",
                str(args.eval_threads),
                "--ctx",
                str(ctx),
                "--chunks",
                str(args.eval_chunks),
                "--scheme",
                args.eval_scheme,
                "--outdir",
                str(run_dir),
                "--eval-x",
                str(args.eval_x),
                "--eval-cols",
                str(args.eval_cols),
            ]
            if args.eval_scheme == "block":
                args_list.extend(["--block", str(args.eval_block)])
            if args.eval_greedy_pack:
                args_list.extend(["--greedy-pack", args.eval_greedy_pack])
            run_eval(eval_script, args_list)

            base_ppl = parse_ppl(run_dir / "perplexity_base.log")
            sd_ppl = parse_ppl(run_dir / "perplexity_seeddelta.log")
            greedy_ok = parse_greedy_result(run_dir / "greedy_pack.log")
            delta_pct = None
            if base_ppl and sd_ppl:
                delta_pct = (sd_ppl - base_ppl) / base_ppl * 100.0

            sweep.append(
                {
                    "ctx": ctx,
                    "base_ppl": base_ppl,
                    "seeddelta_ppl": sd_ppl,
                    "delta_pct": delta_pct,
                    "greedy_ok": greedy_ok,
                }
            )

        (outdir / "ctx_sweep.json").write_text(
            json.dumps({"ctx_sweep": sweep}, indent=2) + "\n",
            encoding="utf-8",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
