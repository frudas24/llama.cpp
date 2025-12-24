#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def parse_layers(s: str):
    out = []
    for part in (s or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a = int(a.strip())
            b = int(b.strip())
            if b < a:
                a, b = b, a
            out.extend(range(a, b + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def make_policy(layers, down_layers, k, strip_dense):
    ranges = []
    for layer in layers:
        ranges.append({
            "layers": str(layer),
            "enabled": True,
            "strip_dense": strip_dense,
            "K": {"gate": k, "up": k, "down": k},
            "gating": {
                "metric": "cos",
                "min_mean": {"gate": 0.0, "up": 0.0, "down": 0.0},
                "min_p05": {"gate": 0.0, "up": 0.0, "down": 0.0},
            },
            "tensors": {
                "ffn_gate": {"enabled": True},
                "ffn_up": {"enabled": False},
                "ffn_down": {"enabled": layer in down_layers},
            },
            "autotune": {"enabled": False},
        })
    return {
        "version": 1,
        "defaults": {"enabled": False},
        "ranges": ranges,
    }


def main():
    ap = argparse.ArgumentParser(
        description="Generate E8 gate+down policies (up-off) for a fixed layer set."
    )
    ap.add_argument("--layers", default="13,15,18,20", help="gate layer list, e.g. 13,15,18,20")
    ap.add_argument("--k", type=int, default=128, help="K budget for gate/down")
    ap.add_argument("--outdir", default="calibration/e8_policies", help="output directory")
    ap.add_argument("--strip-dense", action="store_true", help="enable strip_dense in policy ranges")
    args = ap.parse_args()

    layers = parse_layers(args.layers)
    if not layers:
        raise SystemExit("no layers parsed from --layers")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    e8a = make_policy(layers, {layers[-1]}, args.k, args.strip_dense)
    e8b = make_policy(layers, {layers[-2], layers[-1]}, args.k, args.strip_dense)
    e8c = make_policy(layers, set(layers), args.k, args.strip_dense)

    outputs = {
        "policy_e8a_down_last.json": e8a,
        "policy_e8b_down_last2.json": e8b,
        "policy_e8c_down_all.json": e8c,
    }

    for name, policy in outputs.items():
        path = outdir / name
        path.write_text(json.dumps(policy, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
