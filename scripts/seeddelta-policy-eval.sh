#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

BIN_DIR="${BIN_DIR:-./build/bin}"

BASE_MODEL=""
POLICY_FILE=""
TEXT_FILE=""
IMATRIX_FILE=""
LAYERS=""

THREADS="$(nproc)"
TB=""
CTX="512"
N_PREDICT="256"
PPL_CHUNKS="16"

SCHEME="block"
BLOCK="32"
STRIP_DENSE="1"
ROW_SCALE="1"
USE_BASE="1"
BASE_MAX_SAMPLES="2048"
BASE_PERM_TRIALS="4"

EVAL_COLS="64"
EVAL_X="16"

NO_REPACK="0"
ROUNDTRIP="0"

GREEDY_PACK=""

OUTDIR="calibration/seeddelta-policy-eval-$(date +%Y%m%d-%H%M%S)"
SD_MODEL=""
REPORT_JSON=""
POLICY_EXPORT=""
SD_MODEL_ROUNDTRIP=""
REPORT_JSON_ROUNDTRIP=""

usage() {
  cat <<'EOF'
usage: scripts/seeddelta-policy-eval.sh --base MODEL.gguf --policy policy.json --layers A-B --text TEXTFILE [options]

Builds a SeedΔ model using --policy (gating/autotune/strip) and runs a small eval:
  - greedy smoke battery (base vs seeddelta)
  - perplexity on --text
  - optional /usr/bin/time -v logs

required:
  --base MODEL.gguf
  --policy policy.json
  --layers A-B            (recommended to keep builds bounded)
  --text TEXTFILE         (e.g. wikitext-2-raw/wiki.test.raw)

options (common):
  --imatrix FILE          (recommended for cos_*_x_w gating)
  --outdir DIR            (default: calibration/seeddelta-policy-eval-YYYYmmdd-HHMMSS)
  --threads N             (default: nproc)
  --tb N                  (default: = --threads)
  --ctx N                 (default: 512)
  --n-predict N           (default: 256)
  --chunks N              perplexity chunks (default: 16)
  --no-repack             pass --no-repack to runtime (debug)
  --roundtrip             rebuild using --policy-export and verify decisions match

options (builder):
  --scheme block|coo      (default: block)
  --block N               (default: 32; only for scheme=block)
  --no-strip-dense        (default: strip enabled; policy can still disable per tensor)
  --no-row-scale          (default: row_scale enabled)
  --no-base               (default: base enabled)
  --base-max-samples N    (default: 2048)
  --base-perm-trials N    (default: 4)
  --eval-cols N           (default: 64)
  --eval-x N              (default: 16)

options (greedy pack):
  --greedy-pack FILE      run extended greedy pack (zombie detector) using FILE
                          (e.g. calibration/greedy_zombie_pack.txt)

examples:
  ./scripts/get-wikitext-2.sh
  ./scripts/seeddelta-policy-eval.sh \
    --base   /path/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
    --policy tools/seeddelta-build/policies/policy.example.qwen7b.json \
    --layers 15-20 \
    --imatrix calibration/qwen2_5_7b.imatrix.gguf \
    --text   wikitext-2-raw/wiki.test.raw \
    --threads 16 --ctx 512
EOF
}

die() {
  echo "error: $*" >&2
  exit 1
}

while (( "$#" )); do
  case "$1" in
    --base) BASE_MODEL="${2:-}"; shift 2 ;;
    --policy) POLICY_FILE="${2:-}"; shift 2 ;;
    --layers) LAYERS="${2:-}"; shift 2 ;;
    --text) TEXT_FILE="${2:-}"; shift 2 ;;
    --imatrix) IMATRIX_FILE="${2:-}"; shift 2 ;;
    --outdir) OUTDIR="${2:-}"; shift 2 ;;
    --threads) THREADS="${2:-}"; shift 2 ;;
    --tb) TB="${2:-}"; shift 2 ;;
    --ctx) CTX="${2:-}"; shift 2 ;;
    --n-predict) N_PREDICT="${2:-}"; shift 2 ;;
    --chunks) PPL_CHUNKS="${2:-}"; shift 2 ;;
    --no-repack) NO_REPACK="1"; shift ;;
    --roundtrip) ROUNDTRIP="1"; shift ;;

    --scheme) SCHEME="${2:-}"; shift 2 ;;
    --block) BLOCK="${2:-}"; shift 2 ;;
    --no-strip-dense) STRIP_DENSE="0"; shift ;;
    --no-row-scale) ROW_SCALE="0"; shift ;;
    --no-base) USE_BASE="0"; shift ;;
    --base-max-samples) BASE_MAX_SAMPLES="${2:-}"; shift 2 ;;
    --base-perm-trials) BASE_PERM_TRIALS="${2:-}"; shift 2 ;;
    --eval-cols) EVAL_COLS="${2:-}"; shift 2 ;;
    --eval-x) EVAL_X="${2:-}"; shift 2 ;;

    --greedy-pack) GREEDY_PACK="${2:-}"; shift 2 ;;

    -h|--help) usage; exit 0 ;;
    *) die "unknown arg: $1" ;;
  esac
done

[[ -n "${BASE_MODEL}" ]] || { usage; die "missing --base"; }
[[ -n "${POLICY_FILE}" ]] || { usage; die "missing --policy"; }
[[ -n "${LAYERS}" ]] || { usage; die "missing --layers"; }
[[ -n "${TEXT_FILE}" ]] || { usage; die "missing --text"; }

[[ -f "${BASE_MODEL}" ]] || die "missing base model: ${BASE_MODEL}"
[[ -f "${POLICY_FILE}" ]] || die "missing policy file: ${POLICY_FILE}"
[[ -f "${TEXT_FILE}" ]] || die "missing text file: ${TEXT_FILE}"
if [[ -n "${IMATRIX_FILE}" && ! -f "${IMATRIX_FILE}" ]]; then
  die "missing imatrix: ${IMATRIX_FILE}"
fi

TB="${TB:-${THREADS}}"

for bin in llama-seeddelta-build llama-cli llama-perplexity; do
  [[ -x "${BIN_DIR}/${bin}" ]] || die "missing binary: ${BIN_DIR}/${bin} (build first)"
done

mkdir -p "${OUTDIR}"

SD_MODEL="${SD_MODEL:-${OUTDIR}/model_sd.gguf}"
REPORT_JSON="${REPORT_JSON:-${OUTDIR}/report.json}"
POLICY_EXPORT="${POLICY_EXPORT:-${OUTDIR}/policy.exported.json}"
SD_MODEL_ROUNDTRIP="${SD_MODEL_ROUNDTRIP:-${OUTDIR}/model_sd_roundtrip.gguf}"
REPORT_JSON_ROUNDTRIP="${REPORT_JSON_ROUNDTRIP:-${OUTDIR}/report_roundtrip.json}"

BUILD_LOG="${OUTDIR}/build.log"

build_cmd=(
  "${BIN_DIR}/llama-seeddelta-build"
  -i "${BASE_MODEL}"
  -o "${SD_MODEL}"
  --layers "${LAYERS}"
  --scheme "${SCHEME}"
  -t "${THREADS}"
  --eval-cols "${EVAL_COLS}"
  --eval-x "${EVAL_X}"
  --report-json "${REPORT_JSON}"
  --policy "${POLICY_FILE}"
)

if [[ "${SCHEME}" == "block" ]]; then
  build_cmd+=( --block "${BLOCK}" )
fi

if [[ "${ROW_SCALE}" == "1" ]]; then
  build_cmd+=( --row-scale )
else
  build_cmd+=( --no-row-scale )
fi

if [[ "${USE_BASE}" == "1" ]]; then
  build_cmd+=( --base --base-max-samples "${BASE_MAX_SAMPLES}" --base-perm-trials "${BASE_PERM_TRIALS}" )
fi

if [[ "${STRIP_DENSE}" == "1" ]]; then
  build_cmd+=( --strip-dense )
fi

if [[ -n "${IMATRIX_FILE}" ]]; then
  build_cmd+=( --imatrix "${IMATRIX_FILE}" )
fi
if [[ "${ROUNDTRIP}" == "1" ]]; then
  build_cmd+=( --policy-export "${POLICY_EXPORT}" )
fi

echo "== build ==" | tee "${BUILD_LOG}"
printf 'CMD: %q ' "${build_cmd[@]}" | tee -a "${BUILD_LOG}"
echo | tee -a "${BUILD_LOG}"
"${build_cmd[@]}" 2>&1 | tee -a "${BUILD_LOG}"

if [[ "${ROUNDTRIP}" == "1" ]]; then
  RT_LOG="${OUTDIR}/roundtrip.log"
  echo "== roundtrip build ==" | tee "${RT_LOG}"
  [[ -f "${POLICY_EXPORT}" ]] || die "missing policy export: ${POLICY_EXPORT}"

  rt_cmd=(
    "${BIN_DIR}/llama-seeddelta-build"
    -i "${BASE_MODEL}"
    -o "${SD_MODEL_ROUNDTRIP}"
    --layers "${LAYERS}"
    --scheme "${SCHEME}"
    -t "${THREADS}"
    --eval-cols "${EVAL_COLS}"
    --eval-x 0
    --report-json "${REPORT_JSON_ROUNDTRIP}"
    --policy "${POLICY_EXPORT}"
    --policy-strict
  )
  if [[ "${SCHEME}" == "block" ]]; then
    rt_cmd+=( --block "${BLOCK}" )
  fi
  if [[ "${ROW_SCALE}" == "1" ]]; then
    rt_cmd+=( --row-scale )
  else
    rt_cmd+=( --no-row-scale )
  fi
  if [[ "${USE_BASE}" == "1" ]]; then
    rt_cmd+=( --base --base-max-samples "${BASE_MAX_SAMPLES}" --base-perm-trials "${BASE_PERM_TRIALS}" )
  fi
  if [[ "${STRIP_DENSE}" == "1" ]]; then
    rt_cmd+=( --strip-dense )
  fi
  if [[ -n "${IMATRIX_FILE}" ]]; then
    rt_cmd+=( --imatrix "${IMATRIX_FILE}" )
  fi

  printf 'CMD: %q ' "${rt_cmd[@]}" | tee -a "${RT_LOG}"
  echo | tee -a "${RT_LOG}"
  "${rt_cmd[@]}" 2>&1 | tee -a "${RT_LOG}"

  if ! command -v python3 >/dev/null 2>&1; then
    die "--roundtrip requires python3 to compare reports"
  fi

  REPORT_JSON="${REPORT_JSON}" REPORT_JSON_ROUNDTRIP="${REPORT_JSON_ROUNDTRIP}" python3 - <<'PY'
import json
import os
import sys

base = os.environ["REPORT_JSON"]
rt = os.environ["REPORT_JSON_ROUNDTRIP"]

def load_decisions(path: str):
    with open(path, "r", encoding="utf-8") as f:
        rep = json.load(f)
    out = {}
    for w in rep.get("weights", []):
        if not isinstance(w, dict):
            continue
        layer = w.get("layer", None)
        kind = w.get("kind", None)
        if layer is None or kind is None:
            continue
        key = (int(layer), str(kind))
        emit = bool(w.get("emit", False))
        strip = bool(w.get("strip_dense", False))
        K_budget = int(w.get("K_budget", 0) or 0)
        block = int(w.get("block", 0) or 0)

        # When a tensor is not emitted, K/block are bookkeeping and can differ
        # depending on whether the builder actually tried to fit (gating fail)
        # vs policy-disabled (no attempt). For roundtrip, require strict match
        # only on emitted tensors.
        if not emit:
            K_budget = 0
            block = 0

        out[key] = (emit, strip, K_budget, block)
    return out

a = load_decisions(base)
b = load_decisions(rt)

if a != b:
    ak = set(a.keys())
    bk = set(b.keys())
    only_a = sorted(ak - bk)[:20]
    only_b = sorted(bk - ak)[:20]
    mism = []
    for k in sorted(ak & bk):
        if a[k] != b[k]:
            mism.append((k, a[k], b[k]))
            if len(mism) >= 20:
                break
    print("roundtrip decisions mismatch", file=sys.stderr)
    if only_a:
        print("only in base report:", only_a, file=sys.stderr)
    if only_b:
        print("only in roundtrip report:", only_b, file=sys.stderr)
    for k, va, vb in mism:
        print(f"diff {k}: base={va} roundtrip={vb}", file=sys.stderr)
    sys.exit(1)

print(f"roundtrip decisions OK ({len(a)} tensors)")
PY
fi

run_cli() {
  local label="$1"
  local model="$2"
  local prompt="$3"
  local out_path="$4"
  shift 4

  local cmd=(
    "${BIN_DIR}/llama-cli"
    -m "${model}"
    -p "${prompt}"
    -n "${N_PREDICT}"
    -t "${THREADS}"
    -tb "${TB}"
    -c "${CTX}"
    --ignore-eos
    --simple-io
    --temp 0
    --top-k 1
    --seed 1
    --single-turn
    --no-warmup
    --no-display-prompt
  )

  if [[ "${label}" == "seeddelta" ]]; then
    cmd+=( --seeddelta )
  fi
  if [[ "${NO_REPACK}" == "1" ]]; then
    cmd+=( --no-repack )
  fi

  printf 'CMD: %q ' "${cmd[@]}" > "${out_path}"
  echo >> "${out_path}"
  echo "----" >> "${out_path}"

  "${cmd[@]}" "$@" < /dev/null >> "${out_path}" 2>&1 || true
}

run_ppl() {
  local label="$1"
  local model="$2"
  local out_path="$3"
  shift 3

  local cmd=(
    "${BIN_DIR}/llama-perplexity"
    -m "${model}"
    -f "${TEXT_FILE}"
    -t "${THREADS}"
    -c "${CTX}"
    --chunks "${PPL_CHUNKS}"
    --no-warmup
  )

  if [[ "${label}" == "seeddelta" ]]; then
    cmd+=( --seeddelta )
  fi
  if [[ "${NO_REPACK}" == "1" ]]; then
    cmd+=( --no-repack )
  fi

  printf 'CMD: %q ' "${cmd[@]}" > "${out_path}"
  echo >> "${out_path}"
  echo "----" >> "${out_path}"
  "${cmd[@]}" "$@" >> "${out_path}" 2>&1
}

echo "== greedy smoke ==" | tee "${OUTDIR}/smoke.log"

PROMPTS=(
  "hola"
  "Dame 3 ideas para cenar baratas en Colombia."
  "Escribe un haiku sobre lluvia."
  "What is 17*19?"
  "Explain what a mutex is in Go in one paragraph."
)

for i in "${!PROMPTS[@]}"; do
  p="${PROMPTS[$i]}"
  run_cli "base" "${BASE_MODEL}" "${p}" "${OUTDIR}/greedy_${i}_base.txt"
  run_cli "seeddelta" "${SD_MODEL}" "${p}" "${OUTDIR}/greedy_${i}_seeddelta.txt"
  diff -u "${OUTDIR}/greedy_${i}_base.txt" "${OUTDIR}/greedy_${i}_seeddelta.txt" > "${OUTDIR}/greedy_${i}.diff" || true
done

echo "== perplexity ==" | tee "${OUTDIR}/ppl.log"
run_ppl "base" "${BASE_MODEL}" "${OUTDIR}/perplexity_base.log"
run_ppl "seeddelta" "${SD_MODEL}" "${OUTDIR}/perplexity_seeddelta.log"

if command -v /usr/bin/time >/dev/null 2>&1; then
  echo "== time -v (RSS probe, n=16) ==" | tee "${OUTDIR}/time.log"

  /usr/bin/time -v "${BIN_DIR}/llama-cli" \
    -m "${BASE_MODEL}" -p "hola" -n 16 \
    -t "${THREADS}" -tb "${TB}" -c "${CTX}" \
    --ignore-eos --simple-io --temp 0 --top-k 1 --seed 1 --single-turn --no-warmup --no-display-prompt \
    $([[ "${NO_REPACK}" == "1" ]] && echo --no-repack) \
    > /dev/null 2> "${OUTDIR}/time_base.txt" || true

  /usr/bin/time -v "${BIN_DIR}/llama-cli" \
    -m "${SD_MODEL}" --seeddelta -p "hola" -n 16 \
    -t "${THREADS}" -tb "${TB}" -c "${CTX}" \
    --ignore-eos --simple-io --temp 0 --top-k 1 --seed 1 --single-turn --no-warmup --no-display-prompt \
    $([[ "${NO_REPACK}" == "1" ]] && echo --no-repack) \
    > /dev/null 2> "${OUTDIR}/time_seeddelta.txt" || true
fi

if [[ -n "${GREEDY_PACK}" ]]; then
  echo "== greedy pack (${GREEDY_PACK}) ==" | tee "${OUTDIR}/greedy_pack.log"
  if [[ ! -x "scripts/seeddelta-greedy-pack.sh" ]]; then
    echo "warning: scripts/seeddelta-greedy-pack.sh not found or not executable, skipping greedy pack" | tee -a "${OUTDIR}/greedy_pack.log"
  else
    scripts/seeddelta-greedy-pack.sh \
      --base "${BASE_MODEL}" \
      --sd   "${SD_MODEL}" \
      --pack "${GREEDY_PACK}" \
      --outdir "${OUTDIR}/greedy_pack" \
      | tee -a "${OUTDIR}/greedy_pack.log"
  fi
fi

echo "wrote results to: ${OUTDIR}"
