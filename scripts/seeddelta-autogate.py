#!/usr/bin/env python3

import argparse
import json
import math
import os
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


def write_policy(path: Path, layers: list[int], k: int, metric: str, enable_down: bool) -> None:
    layer_csv = ",".join(str(l) for l in layers)
    policy = {
        "version": 1,
        "defaults": {
            "enabled": False,
            "gating": {"metric": metric, "min_mean": -1.0, "min_p05": -1.0},
            "autotune": {"enabled": False},
        },
        "ranges": [
            {
                "layers": layer_csv,
                "enabled": True,
                "strip_dense": False,
                "K": {"gate": k, "up": k, "down": k},
                "gating": {
                    "metric": metric,
                    "min_mean": {"gate": 0.0, "up": 0.0, "down": 0.0},
                    "min_p05": {"gate": 0.0, "up": 0.0, "down": 0.0},
                },
                "tensors": {
                    "ffn_gate": {"enabled": True},
                    "ffn_up": {"enabled": True},
                    "ffn_down": {"enabled": bool(enable_down)},
                },
                "autotune": {"enabled": False},
            }
        ],
    }
    path.write_text(json.dumps(policy, indent=2) + "\n", encoding="utf-8")


def load_scan(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def aggregate_metrics(layer_metrics: dict, tensors: list[str], metric: str, agg: str) -> float | None:
    vals: list[float] = []
    for t in tensors:
        m = layer_metrics.get(t)
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


def main() -> int:
    ap = argparse.ArgumentParser(description="Auto-gate layer selection from scan + optional eval")
    ap.add_argument("--scan", required=True, help="Path to layer_sensitivity_scan.json")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--k", type=int, default=12, help="Fixed K for policy")
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
    ap.add_argument("--policy-out", default="", help="Output policy path (default: outdir/policy.autogen.json)")
    ap.add_argument("--forbidden-out", default="", help="Output forbidden pairs JSON (optional)")
    ap.add_argument("--scores-out", default="", help="Output scores JSON (optional)")

    ap.add_argument("--eval", action="store_true", help="Run incremental eval and rollback on failures")
    ap.add_argument("--eval-script", default="./scripts/seeddelta-policy-eval.sh", help="Eval script")
    ap.add_argument("--base", default="", help="Base GGUF for eval")
    ap.add_argument("--text", default="", help="Text file for eval PPL")
    ap.add_argument("--layers-range", default="", help="Layer range for builder (e.g. 0-31)")
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
    args = ap.parse_args()

    scan = load_scan(Path(args.scan))
    layers_dict = scan.get("layers", {})
    if not isinstance(layers_dict, dict):
        print("error: scan.layers missing or invalid", file=sys.stderr)
        return 2

    tensors = [t.strip() for t in args.tensors.split(",") if t.strip()]
    prefer_layers = set(parse_layers(args.prefer)) if args.prefer else set()

    scores: dict[int, float] = {}
    for layer_s, layer_metrics in layers_dict.items():
        try:
            layer = int(layer_s)
        except ValueError:
            continue
        if not isinstance(layer_metrics, dict):
            continue
        val = aggregate_metrics(layer_metrics, tensors, args.rank_metric, args.aggregate)
        if val is None:
            continue
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

        candidate = selected + [layer]
        candidate.sort()

        if not args.eval:
            selected = candidate
            continue

        if not args.base or not args.text:
            print("error: --eval requires --base and --text", file=sys.stderr)
            return 2

        policy_path = eval_outdir / f"policy.step{len(eval_steps):03d}.json"
        write_policy(policy_path, candidate, args.k, args.metric, args.enable_down)

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
        else:
            for s in selected:
                pair = tuple(sorted((s, layer)))
                forbidden_pairs.add(pair)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    policy_out = Path(args.policy_out) if args.policy_out else outdir / "policy.autogen.json"
    write_policy(policy_out, selected, args.k, args.metric, args.enable_down)

    scores_out = Path(args.scores_out) if args.scores_out else outdir / "layer_scores.json"
    scores_out.write_text(json.dumps(scores, indent=2) + "\n", encoding="utf-8")

    if args.forbidden_out or forbidden_pairs:
        forb_out = Path(args.forbidden_out) if args.forbidden_out else outdir / "forbidden_pairs.json"
        forb_out.write_text(
            json.dumps(sorted([list(p) for p in forbidden_pairs]), indent=2) + "\n",
            encoding="utf-8",
        )

    if eval_steps:
        report = {
            "selected_layers": selected,
            "steps": eval_steps,
        }
        (outdir / "autogating_report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
