#!/usr/bin/env python3

import argparse
import json
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


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate a gate+down policy for E8.")
    ap.add_argument("--out", required=True, help="Output policy path")
    ap.add_argument("--layers", required=True, help="Gate layers (CSV or ranges)")
    ap.add_argument("--down", default="", help="Down layers subset (CSV or ranges)")
    ap.add_argument("--k", type=int, default=128, help="K value for gate/up/down")
    ap.add_argument("--metric", default="cos_x_w", help="Gating metric for policy")
    ap.add_argument("--strip-dense", action="store_true", help="Set strip_dense=true")
    ap.add_argument("--no-strip-dense", action="store_true", help="Set strip_dense=false")
    args = ap.parse_args()

    layers = parse_layers(args.layers)
    down_layers = set(parse_layers(args.down)) if args.down else set()
    strip_dense = args.strip_dense and not args.no_strip_dense

    policy = {
        "version": 1,
        "defaults": {
            "enabled": False,
            "gating": {"metric": args.metric, "min_mean": -1.0, "min_p05": -1.0},
            "autotune": {"enabled": False},
        },
        "ranges": [],
    }

    for layer in layers:
        tensors = {
            "ffn_gate": {"enabled": True},
            "ffn_up": {"enabled": False},
            "ffn_down": {"enabled": layer in down_layers},
        }
        policy["ranges"].append(
            {
                "layers": str(layer),
                "enabled": True,
                "strip_dense": bool(strip_dense),
                "K": {"gate": args.k, "up": args.k, "down": args.k},
                "gating": {
                    "metric": args.metric,
                    "min_mean": {"gate": 0.0, "up": 0.0, "down": 0.0},
                    "min_p05": {"gate": 0.0, "up": 0.0, "down": 0.0},
                },
                "tensors": tensors,
                "autotune": {"enabled": False},
            }
        )

    out = Path(args.out)
    out.write_text(json.dumps(policy, indent=2) + "\n", encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
