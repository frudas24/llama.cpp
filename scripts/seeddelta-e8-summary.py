#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path


def parse_ppl(path: Path) -> float | None:
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("Final estimate: PPL ="):
            try:
                return float(line.split("=")[1].split()[0])
            except Exception:
                return None
    return None


def parse_rss(path: Path) -> int | None:
    if not path.exists():
        return None
    rx = re.compile(r"Maximum resident set size \(kbytes\):\s*(\d+)")
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = rx.search(line)
        if m:
            return int(m.group(1))
    return None


def parse_result(path: Path) -> str | None:
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("RESULT:"):
            return line.strip()
    return None


def summarize(outdir: Path) -> dict:
    report = outdir / "report.json"
    emit = strip = None
    if report.exists():
        data = json.loads(report.read_text(encoding="utf-8"))
        weights = data.get("weights", [])
        emit = sum(1 for w in weights if w.get("emit"))
        strip = sum(1 for w in weights if w.get("strip_dense"))

    ppl = {}
    for p in outdir.glob("perplexity_base_ctx*.log"):
        ctx = p.stem.split("ctx")[-1]
        base = parse_ppl(p)
        sd = parse_ppl(outdir / f"perplexity_sd_ctx{ctx}.log")
        delta = None
        if base and sd:
            delta = (sd / base - 1.0) * 100.0
        ppl[str(ctx)] = {"base": base, "sd": sd, "delta_pct": delta}

    rss = {}
    for p in outdir.glob("time_base_ctx*.txt"):
        ctx = p.stem.split("ctx")[-1]
        base = parse_rss(p)
        sd = parse_rss(outdir / f"time_sd_ctx{ctx}.txt")
        delta = (sd - base) if (base and sd) else None
        rss[str(ctx)] = {"base_kb": base, "sd_kb": sd, "delta_kb": delta}

    greedy = parse_result(outdir / "greedy_pack.log")

    return {
        "outdir": str(outdir),
        "emit_count": emit,
        "strip_count": strip,
        "ppl": ppl,
        "rss": rss,
        "greedy": greedy,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize E8 results from an outdir.")
    ap.add_argument("outdir", help="Output dir produced by seeddelta-e8-run.sh")
    ap.add_argument("--json", action="store_true", help="Emit JSON summary")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    summary = summarize(outdir)
    if args.json:
        print(json.dumps(summary, indent=2))
        return 0

    print(outdir)
    print(f"emit={summary['emit_count']} strip={summary['strip_count']}")
    for ctx, vals in sorted(summary["ppl"].items(), key=lambda x: int(x[0])):
        print(f"ctx{ctx}: base={vals['base']} sd={vals['sd']} delta_pct={vals['delta_pct']}")
    for ctx, vals in sorted(summary["rss"].items(), key=lambda x: int(x[0])):
        print(f"rss ctx{ctx}: base_kb={vals['base_kb']} sd_kb={vals['sd_kb']} delta_kb={vals['delta_kb']}")
    print(f"greedy={summary['greedy']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
