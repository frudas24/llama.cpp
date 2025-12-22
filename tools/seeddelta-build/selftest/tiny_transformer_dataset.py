#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path


def rand_text(rng, min_len, max_len):
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789 "
    n = rng.randint(min_len, max_len)
    return "".join(rng.choice(alphabet) for _ in range(n)).strip()


def make_copy(rng):
    s = rand_text(rng, 8, 32)
    return "copy", f"IN: {s}", s


def make_sum(rng):
    a = rng.randint(0, 999)
    b = rng.randint(0, 999)
    return "sum", f"SUM: {a}+{b}", str(a + b)


def make_brackets(rng):
    n = rng.randint(6, 20)
    s = "".join(rng.choice("()") for _ in range(n))
    bal = 0
    ok = True
    for ch in s:
        if ch == "(":
            bal += 1
        else:
            bal -= 1
        if bal < 0:
            ok = False
            break
    if bal != 0:
        ok = False
    return "brackets", f"BAL: {s}", "OK" if ok else "BAD"


def make_sample(rng):
    tasks = [make_copy, make_sum, make_brackets]
    fn = rng.choice(tasks)
    task, inp, out = fn(rng)
    text = f"### TASK\n{task}\n### INPUT\n{inp}\n### OUTPUT\n{out}\n"
    return {"text": text, "task": task}


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--train-size", type=int, default=2000)
    ap.add_argument("--val-size", type=int, default=400)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    train = [make_sample(rng) for _ in range(args.train_size)]
    val = [make_sample(rng) for _ in range(args.val_size)]

    write_jsonl(outdir / "train.jsonl", train)
    write_jsonl(outdir / "val.jsonl", val)

    print(f"wrote {len(train)} train and {len(val)} val samples to {outdir}")


if __name__ == "__main__":
    main()
