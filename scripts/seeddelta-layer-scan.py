#!/usr/bin/env python3

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path


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


def write_policy(path: Path, layer: int, tensors: list[str], k: int, metric: str) -> None:
    policy = {
        "version": 1,
        "defaults": {
            "enabled": False,
            "gating": {"metric": metric, "min_mean": -1.0, "min_p05": -1.0},
            "autotune": {"enabled": False},
        },
        "ranges": [
            {
                "layers": str(layer),
                "enabled": True,
                "strip_dense": False,
                "K": {"gate": k, "up": k, "down": k},
                "gating": {
                    "metric": metric,
                    "min_mean": {"gate": 0.0, "up": 0.0, "down": 0.0},
                    "min_p05": {"gate": 0.0, "up": 0.0, "down": 0.0},
                },
                "tensors": {
                    "ffn_gate": {"enabled": "ffn_gate" in tensors},
                    "ffn_up": {"enabled": "ffn_up" in tensors},
                    "ffn_down": {"enabled": "ffn_down" in tensors},
                },
                "autotune": {"enabled": False},
            }
        ],
    }
    path.write_text(json.dumps(policy, indent=2) + "\n", encoding="utf-8")


def collect_metrics(entry: dict) -> dict:
    def read_float(key: str) -> float | None:
        val = entry.get(key)
        if val is None:
            return None
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    rel_l2_mean = read_float("rel_l2_mean") or 0.0
    rel_l2_p95 = read_float("rel_l2_p95") or 0.0
    cos_mean = read_float("cos_mean") or 0.0
    cos_p05 = read_float("cos_p05") or 0.0
    cos_mean_x_w = read_float("cos_mean_x_w") or 0.0
    cos_p05_x_w = read_float("cos_p05_x_w") or 0.0
    norm_ratio_mean = read_float("norm_ratio_mean") or 0.0
    log_norm_ratio_mean = math.log(norm_ratio_mean) if norm_ratio_mean > 0 else 0.0

    ffn_proxy_available = bool(entry.get("ffn_proxy_available", False))
    ffn_proxy_cos_mean = read_float("ffn_proxy_cos_mean")
    ffn_proxy_cos_p05 = read_float("ffn_proxy_cos_p05")
    ffn_proxy_l2_mean = read_float("ffn_proxy_l2_mean")
    ffn_proxy_l2_p95 = read_float("ffn_proxy_l2_p95")
    ffn_proxy_log_norm_ratio_mean = read_float("ffn_proxy_log_norm_ratio_mean")
    ffn_proxy_log_norm_ratio_p95 = read_float("ffn_proxy_log_norm_ratio_p95")

    return {
        "S_mean": rel_l2_mean,
        "S_p95": rel_l2_p95,
        "cos_mean": cos_mean,
        "cos_p05": cos_p05,
        "cos_mean_x_w": cos_mean_x_w,
        "cos_p05_x_w": cos_p05_x_w,
        "log_norm_ratio_mean": log_norm_ratio_mean,
        "log_norm_ratio_p95": None,
        "ffn_proxy_available": ffn_proxy_available,
        "ffn_proxy_cos_mean": ffn_proxy_cos_mean,
        "ffn_proxy_cos_p05": ffn_proxy_cos_p05,
        "ffn_proxy_l2_mean": ffn_proxy_l2_mean,
        "ffn_proxy_l2_p95": ffn_proxy_l2_p95,
        "ffn_proxy_log_norm_ratio_mean": ffn_proxy_log_norm_ratio_mean,
        "ffn_proxy_log_norm_ratio_p95": ffn_proxy_log_norm_ratio_p95,
        "raw": {
            "rel_l2_mean": rel_l2_mean,
            "rel_l2_p95": rel_l2_p95,
            "norm_ratio_mean": norm_ratio_mean,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run per-layer SeedDelta scans and emit layer_sensitivity_scan.json"
    )
    ap.add_argument("--base", required=True, help="Base GGUF model")
    ap.add_argument("--layers", required=True, help="Layer list (e.g. 0-31 or 0,2,4)")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--k", type=int, default=12, help="Fixed K for gate/up/down")
    ap.add_argument("--scheme", default="coo", choices=["coo", "block"], help="Residual scheme")
    ap.add_argument("--block", type=int, default=32, help="Block size (scheme=block)")
    ap.add_argument("--eval-cols", type=int, default=64, help="Eval columns")
    ap.add_argument("--eval-x", type=int, default=32, help="Eval x vectors")
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 1, help="Worker threads")
    ap.add_argument("--seed", type=int, default=1234, help="RNG seed")
    ap.add_argument("--metric", default="cos", help="Gating metric for policy")
    ap.add_argument("--tensors", default="ffn_gate,ffn_up,ffn_down", help="CSV tensors to scan")
    ap.add_argument("--row-scale", action="store_true", help="Enable row-scale output")
    ap.add_argument("--no-row-scale", action="store_true", help="Disable row-scale output")
    ap.add_argument("--base-fit", action="store_true", help="Enable base fit (W0)")
    ap.add_argument("--base-max-samples", type=int, default=2048, help="Base max samples")
    ap.add_argument("--base-perm-trials", type=int, default=4, help="Base perm trials")
    ap.add_argument("--strip-dense", action="store_true", help="Strip dense weights")
    ap.add_argument("--keep-gguf", action="store_true", help="Keep per-layer GGUF outputs")
    ap.add_argument("--bin-dir", default=os.environ.get("BIN_DIR", "./build/bin"), help="Binary dir")
    ap.add_argument("--imatrix", default="", help="Optional imatrix GGUF")
    args = ap.parse_args()

    base = Path(args.base)
    if not base.exists():
        print(f"error: missing base model: {base}", file=sys.stderr)
        return 2

    layers = parse_layers(args.layers)
    if not layers:
        print("error: empty --layers", file=sys.stderr)
        return 2

    bin_dir = Path(args.bin_dir)
    build_bin = bin_dir / "llama-seeddelta-build"
    if not build_bin.exists():
        print(f"error: missing binary: {build_bin}", file=sys.stderr)
        return 2

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    policy_dir = outdir / "policies"
    policy_dir.mkdir(parents=True, exist_ok=True)

    tensors = [t.strip() for t in args.tensors.split(",") if t.strip()]
    scan = {
        "version": 1,
        "input": str(base),
        "scheme": args.scheme,
        "block": args.block if args.scheme == "block" else 0,
        "K_fixed": args.k,
        "eval_cols": args.eval_cols,
        "eval_x": args.eval_x,
        "seed": args.seed,
        "base_used": bool(args.base_fit),
        "row_scale": bool(args.row_scale),
        "strip_dense": bool(args.strip_dense),
        "meta": {
            "scheme": args.scheme,
            "block": args.block if args.scheme == "block" else 0,
            "K_fixed": args.k,
            "eval_cols": args.eval_cols,
            "eval_x": args.eval_x,
            "seed": args.seed,
            "metric": args.metric,
            "tensors_scanned": list(tensors),
            "seed_delta_bin": str(build_bin),
            "threads": args.threads,
            "base_used": bool(args.base_fit),
            "row_scale": bool(args.row_scale),
            "strip_dense": bool(args.strip_dense),
            "imatrix": args.imatrix,
        },
        "layers": {},
        "errors": [],
    }

    for layer in layers:
        layer_dir = outdir / f"layer_{layer}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        policy_path = policy_dir / f"policy.layer{layer}.json"
        report_path = layer_dir / "report.json"
        model_out = layer_dir / "model_sd.gguf"

        write_policy(policy_path, layer, tensors, args.k, args.metric)

        cmd = [
            str(build_bin),
            "-i",
            str(base),
            "-o",
            str(model_out),
            "--layers",
            f"{layer}-{layer}",
            "--scheme",
            args.scheme,
            "--K",
            str(args.k),
            "--eval-cols",
            str(args.eval_cols),
            "--eval-x",
            str(args.eval_x),
            "--report-json",
            str(report_path),
            "--policy",
            str(policy_path),
            "-t",
            str(args.threads),
            "--seed",
            str(args.seed),
        ]

        if args.scheme == "block":
            cmd.extend(["--block", str(args.block)])
        if args.row_scale and not args.no_row_scale:
            cmd.append("--row-scale")
        if args.no_row_scale:
            cmd.append("--no-row-scale")
        if args.base_fit:
            cmd.extend(
                [
                    "--base",
                    "--base-max-samples",
                    str(args.base_max_samples),
                    "--base-perm-trials",
                    str(args.base_perm_trials),
                ]
            )
        if args.strip_dense:
            cmd.append("--strip-dense")
        if args.imatrix:
            cmd.extend(["--imatrix", args.imatrix])

        subprocess.run(cmd, check=True)

        if not report_path.exists():
            scan["errors"].append({"layer": layer, "error": "missing report.json"})
            continue

        rep = json.loads(report_path.read_text(encoding="utf-8"))
        weights = rep.get("weights", [])
        layer_out: dict[str, dict] = {}
        for w in weights:
            if int(w.get("layer", -1)) != layer:
                continue
            kind = str(w.get("kind", ""))
            if kind in tensors:
                layer_out[kind] = collect_metrics(w)

        if not layer_out:
            scan["errors"].append({"layer": layer, "error": "no weights matched"})
        scan["layers"][str(layer)] = layer_out

        if not args.keep_gguf and model_out.exists():
            model_out.unlink()

    scan_path = outdir / "layer_sensitivity_scan.json"
    scan_path.write_text(json.dumps(scan, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
