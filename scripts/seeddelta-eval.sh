#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

BIN_DIR="${BIN_DIR:-./build/bin}"

BASE_MODEL=""
SD_MODEL=""
TEXT_FILE=""

THREADS="$(nproc)"
CTX="4096"
N_PREDICT="128"
PPL_CHUNKS="-1"
SD_GAP="0.020"
RUN_QA="1"

OUTDIR="seeddelta-eval-$(date +%Y%m%d-%H%M%S)"

usage() {
  cat <<EOF
usage: $(basename "$0") --base MODEL.gguf --sd MODEL_sd.gguf --text TEXTFILE [options]

options:
  --threads N          (default: nproc)
  --ctx N              (default: 4096)
  --n-predict N        (default: 128)
  --chunks N           max perplexity chunks (default: -1 = all)
  --gap F              seeddelta gap (default: 0.020)
  --outdir DIR         (default: seeddelta-eval-YYYYmmdd-HHMMSS)
  --no-qa              skip QA generation (only perplexity)

examples:
  ./scripts/get-wikitext-2.sh
  ./scripts/seeddelta-eval.sh \\
    --base /path/model_base.gguf \\
    --sd   /path/model_sd.gguf \\
    --text wikitext-2-raw/wiki.test.raw \\
    --threads 16 --ctx 4096
EOF
}

while (( "$#" )); do
  case "$1" in
    --base) BASE_MODEL="${2:-}"; shift 2 ;;
    --sd|--seeddelta) SD_MODEL="${2:-}"; shift 2 ;;
    --text) TEXT_FILE="${2:-}"; shift 2 ;;
    --threads) THREADS="${2:-}"; shift 2 ;;
    --ctx) CTX="${2:-}"; shift 2 ;;
    --n-predict) N_PREDICT="${2:-}"; shift 2 ;;
    --chunks) PPL_CHUNKS="${2:-}"; shift 2 ;;
    --gap|--seeddelta-gap) SD_GAP="${2:-}"; shift 2 ;;
    --outdir) OUTDIR="${2:-}"; shift 2 ;;
    --no-qa) RUN_QA="0"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "${BASE_MODEL}" || -z "${SD_MODEL}" || -z "${TEXT_FILE}" ]]; then
  usage
  exit 1
fi

if [[ ! -x "${BIN_DIR}/llama-perplexity" || ! -x "${BIN_DIR}/llama-cli" ]]; then
  echo "missing binaries in ${BIN_DIR}; build first (e.g. cmake --build build -j)" >&2
  exit 1
fi

if [[ ! -f "${TEXT_FILE}" ]]; then
  echo "missing --text file: ${TEXT_FILE}" >&2
  exit 1
fi

mkdir -p "${OUTDIR}"

run_perplexity() {
  local label="$1"
  local model="$2"
  shift 2

  echo "== perplexity: ${label} ==" | tee "${OUTDIR}/perplexity_${label}.log"
  "${BIN_DIR}/llama-perplexity" \
    -m "${model}" \
    -f "${TEXT_FILE}" \
    -t "${THREADS}" \
    -c "${CTX}" \
    --chunks "${PPL_CHUNKS}" \
    --no-warmup \
    "$@" 2>&1 | tee -a "${OUTDIR}/perplexity_${label}.log"
  echo | tee -a "${OUTDIR}/perplexity_${label}.log"
}

run_qa() {
  local label="$1"
  local model="$2"
  local prompt="$3"
  shift 3

  "${BIN_DIR}/llama-cli" \
    -m "${model}" \
    -p "${prompt}" \
    -n "${N_PREDICT}" \
    -t "${THREADS}" \
    -c "${CTX}" \
    --simple-io \
    --temp 0 \
    --top-k 1 \
    --seed 1 \
    --single-turn \
    --no-warmup \
    --no-display-prompt \
    "$@" < /dev/null > "${OUTDIR}/qa_${label}.txt"
}

run_perplexity "base" "${BASE_MODEL}"
run_perplexity "seeddelta" "${SD_MODEL}" --seeddelta --seeddelta-gap "${SD_GAP}"

if [[ "${RUN_QA}" == "1" ]]; then
  PROMPT=$'Responde de forma concisa:\n\n1) ¿Qué es un árbol binario?\n2) Dame un ejemplo en pseudocódigo.\n'

  run_qa "base" "${BASE_MODEL}" "${PROMPT}"
  run_qa "seeddelta" "${SD_MODEL}" "${PROMPT}" --seeddelta --seeddelta-gap "${SD_GAP}"

  diff -u "${OUTDIR}/qa_base.txt" "${OUTDIR}/qa_seeddelta.txt" > "${OUTDIR}/qa.diff" || true
fi

echo "wrote results to: ${OUTDIR}"
