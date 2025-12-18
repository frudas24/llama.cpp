#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

LLAMA_BIN="${LLAMA_BIN:-./build/bin/llama-cli}"

BASE=""
SD=""
PACK="calibration/greedy_zombie_pack.txt"
OUTDIR=""

THREADS="${THREADS:-16}"
CTX="${CTX:-512}"
NGEN="${NGEN:-128}"
SEED="${SEED:-1}"

usage() {
  cat <<EOF
usage: scripts/seeddelta-greedy-pack.sh --base BASE.gguf --sd SD.gguf [options]

Run a fixed greedy prompt pack (zombie detector) against a base model and a SeedΔ model.

required:
  --base BASE.gguf       base model
  --sd   SD.gguf         SeedΔ model (built from --base)

options:
  --pack FILE            prompt pack (default: calibration/greedy_zombie_pack.txt)
  --outdir DIR           output directory (default: calibration/greedy-pack-YYYYmmdd-HHMMSS)

env:
  LLAMA_BIN              (default: ./build/bin/llama-cli)
  THREADS                (default: 16)
  CTX                    (default: 512)
  NGEN                   (default: 128)
  SEED                   (default: 1)
EOF
}

while (( "$#" )); do
  case "$1" in
    --base) BASE="${2:-}"; shift 2 ;;
    --sd) SD="${2:-}"; shift 2 ;;
    --pack) PACK="${2:-}"; shift 2 ;;
    --outdir) OUTDIR="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "error: unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

[[ -n "${BASE}" && -n "${SD}" ]] || { echo "error: missing --base/--sd" >&2; usage; exit 1; }
[[ -f "${BASE}" ]] || { echo "error: base model not found: ${BASE}" >&2; exit 1; }
[[ -f "${SD}" ]] || { echo "error: SeedΔ model not found: ${SD}" >&2; exit 1; }
[[ -f "${PACK}" ]] || { echo "error: prompt pack not found: ${PACK}" >&2; exit 1; }
[[ -x "${LLAMA_BIN}" ]] || { echo "error: missing llama-cli: ${LLAMA_BIN}" >&2; exit 1; }

if [[ -z "${OUTDIR}" ]]; then
  ts="$(date +%Y%m%d-%H%M%S)"
  OUTDIR="calibration/greedy-pack-${ts}"
fi

mkdir -p "${OUTDIR}/base" "${OUTDIR}/sd" "${OUTDIR}/diff"

python3 - <<'PY' "${PACK}" "${OUTDIR}" "${BASE}" "${SD}" "${LLAMA_BIN}" "${THREADS}" "${CTX}" "${NGEN}" "${SEED}"
import sys
import re
import pathlib
import subprocess

pack, outdir, base, sd, llama_bin, threads, ctx, ngen, seed = sys.argv[1:]
outdir_p = pathlib.Path(outdir)

text = pathlib.Path(pack).read_text(encoding="utf-8", errors="replace")
blocks = re.split(r'(?m)^###\s+', text)
prompts = []
for block in blocks:
    block = block.strip()
    if not block:
        continue
    lines = block.splitlines()
    pid = lines[0].strip()
    body = "\n".join(lines[1:]).strip()
    if not body:
        continue
    prompts.append((pid, body))

def run(model: str, prompt: str, out_path: pathlib.Path) -> None:
    cmd = [
        llama_bin,
        "-m", model,
        "-t", str(threads),
        "-c", str(ctx),
        "--temp", "0",
        "--top-k", "1",
        "--seed", str(seed),
        "--single-turn",
        "--no-warmup",
        "--no-display-prompt",
        "--simple-io",
        "-n", str(ngen),
        "-p", prompt,
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out_path.write_bytes(proc.stdout)

for pid, body in prompts:
    run(base, body, outdir_p / "base" / f"{pid}.txt")
    run(sd,   body, outdir_p / "sd"   / f"{pid}.txt")

print(f"wrote {len(prompts)} prompts to {outdir_p}")
PY

for f in "${OUTDIR}/base/"*.txt; do
  bn="$(basename "${f}")"
  diff -u "${OUTDIR}/base/${bn}" "${OUTDIR}/sd/${bn}" > "${OUTDIR}/diff/${bn}.diff" || true
done

python3 - <<'PY' "${OUTDIR}"
import sys
import re
import pathlib

outdir = pathlib.Path(sys.argv[1])

def read_text(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")

def repetition_score(s: str) -> int:
    words = re.findall(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9_]+", s.lower())
    best = 1
    cur = 1
    for i in range(1, len(words)):
        if words[i] == words[i - 1]:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 1
    return best

def english_ratio_proxy(s: str) -> float:
    alpha = re.findall(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]", s)
    if not alpha:
        return 0.0
    ascii_alpha = [c for c in alpha if ("A" <= c <= "Z") or ("a" <= c <= "z")]
    return len(ascii_alpha) / len(alpha)

def strict_ok(pid: str, s: str) -> bool:
    t = s.strip().lower()
    if pid == "P03_follow_instruction_single_word":
        return t == "listo"
    if pid == "P20_sentinel":
        return t == "fin"
    return True

rows = []
for pid_file in sorted((outdir / "sd").glob("*.txt")):
    pid = pid_file.stem
    base = read_text(outdir / "base" / f"{pid}.txt")
    sd = read_text(pid_file)
    rep = repetition_score(sd)
    er = english_ratio_proxy(sd)
    ok = strict_ok(pid, sd)
    rows.append((pid, rep, er, ok))

spanish_pids = {
    "P01_greeting_es",
    "P02_concise_bullets",
    "P04_anti_loop_constraint",
    "P07_short_reasoning",
    "P10_code_explain",
    "P15_spanish_only_guard",
    "P16_programming_short",
    "P17_error_style",
    "P19_instruction_priority",
}

print("=== Greedy pack summary (SeedΔ) ===")
bad = 0
for pid, rep, er, ok in rows:
    flags = []
    if rep >= 8:
        flags.append(f"REPEATx{rep}")
    if er > 0.85 and pid in spanish_pids:
        flags.append(f"DRIFT~{er:.2f}")
    if not ok:
        flags.append("STRICT_FAIL")
    if flags:
        bad += 1
    print(f"{pid:30s} rep={rep:>2} drift={er:.2f}  {' '.join(flags)}")

total = len(rows)
result = "PASS" if bad == 0 else "FAIL"
print(f"\nPrompts flagged: {bad}/{total}")
print(f"RESULT: {result}")
print(f"Outputs in: {outdir}")
PY

