#!/usr/bin/env python3

import argparse
import json
import sys


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Convert llama-seeddelta-build --report-json output into a strict-schema policy.json"
    )
    ap.add_argument("--report", required=True, help="Path to report.json from llama-seeddelta-build")
    ap.add_argument("--out", required=True, help="Output policy.json path")
    ap.add_argument(
        "--include-disabled",
        action="store_true",
        help="Include disabled tensors (emit=false) explicitly (default: only emit=true tensors)",
    )
    args = ap.parse_args()

    with open(args.report, "r", encoding="utf-8") as f:
        rep = json.load(f)

    weights = rep.get("weights", [])
    if not isinstance(weights, list):
        print("error: report.weights is not a list", file=sys.stderr)
        return 2

    policy = {
        "version": 1,
        "defaults": {
            "enabled": False,
            "strip_dense": False,
            "gating": {"metric": "cos", "min_mean": -1.0, "min_p05": -1.0},
            "autotune": {"enabled": False},
        },
        "layers": {},
    }

    layers = policy["layers"]

    for w in weights:
        if not isinstance(w, dict):
            continue

        layer = w.get("layer", None)
        kind = w.get("kind", None)
        if layer is None or kind is None:
            continue

        emit = bool(w.get("emit", False))
        strip = bool(w.get("strip_dense", False))
        k_budget = int(w.get("K_budget", 0) or 0)
        block = int(w.get("block", 0) or 0)

        if not args.include_disabled and not emit:
            continue

        layer_key = str(int(layer))
        layer_obj = layers.setdefault(layer_key, {})
        tensors = layer_obj.setdefault("tensors", {})

        tr = {"enabled": emit, "strip_dense": strip}
        if block > 0:
            tr["block"] = block
        if emit and k_budget > 0:
            tr["K"] = k_budget

        tensors[str(kind)] = tr

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(policy, f, indent=2, ensure_ascii=False)
        f.write("\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

