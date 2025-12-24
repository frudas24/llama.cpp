#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

usage() {
  cat <<EOF
usage: scripts/seeddelta-e8-run.sh --base MODEL.gguf --policy POLICY.json --outdir DIR [options]

required:
  --base MODEL.gguf
  --policy POLICY.json
  --outdir DIR

optional:
  --layers A-B            (default: 13-20)
  --imatrix FILE          (default: calibration/gemma4b.imatrix.gguf)
  --text FILE             (default: wikitext-2-raw/wiki.test.raw)
  --threads N             (default: 16)
  --ctx-list CSV          (default: 1024,2048)
  --rss-ctx-list CSV      (default: 64,128)
  --greedy-pack FILE      (default: calibration/greedy_zombie_pack.txt)
EOF
}

BASE=""
POLICY=""
OUTDIR=""
LAYERS="13-20"
IMATRIX="calibration/gemma4b.imatrix.gguf"
TEXT="wikitext-2-raw/wiki.test.raw"
THREADS="16"
CTX_LIST="1024,2048"
RSS_CTX_LIST="64,128"
GREEDY_PACK="calibration/greedy_zombie_pack.txt"

while (( "$#" )); do
  case "$1" in
    --base) BASE="${2:-}"; shift 2 ;;
    --policy) POLICY="${2:-}"; shift 2 ;;
    --outdir) OUTDIR="${2:-}"; shift 2 ;;
    --layers) LAYERS="${2:-}"; shift 2 ;;
    --imatrix) IMATRIX="${2:-}"; shift 2 ;;
    --text) TEXT="${2:-}"; shift 2 ;;
    --threads) THREADS="${2:-}"; shift 2 ;;
    --ctx-list) CTX_LIST="${2:-}"; shift 2 ;;
    --rss-ctx-list) RSS_CTX_LIST="${2:-}"; shift 2 ;;
    --greedy-pack) GREEDY_PACK="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "error: unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

[[ -n "${BASE}" && -n "${POLICY}" && -n "${OUTDIR}" ]] || { usage; exit 1; }
[[ -f "${BASE}" ]] || { echo "error: missing base model: ${BASE}" >&2; exit 1; }
[[ -f "${POLICY}" ]] || { echo "error: missing policy: ${POLICY}" >&2; exit 1; }
[[ -f "${IMATRIX}" ]] || { echo "error: missing imatrix: ${IMATRIX}" >&2; exit 1; }
[[ -f "${TEXT}" ]] || { echo "error: missing text: ${TEXT}" >&2; exit 1; }

mkdir -p "${OUTDIR}"

SD_MODEL="${OUTDIR}/model_sd.gguf"
REPORT_JSON="${OUTDIR}/report.json"

./build/bin/llama-seeddelta-build \
  -i "${BASE}" \
  -o "${SD_MODEL}" \
  --layers "${LAYERS}" \
  --scheme coo \
  -t "${THREADS}" \
  --eval-cols 64 \
  --eval-x 32 \
  --report-json "${REPORT_JSON}" \
  --policy "${POLICY}" \
  --row-scale \
  --base --base-max-samples 2048 --base-perm-trials 4 \
  --strip-dense \
  --imatrix "${IMATRIX}"

IFS=',' read -r -a ctx_vals <<< "${CTX_LIST}"
for ctx in "${ctx_vals[@]}"; do
  ./build/bin/llama-perplexity -m "${BASE}" -f "${TEXT}" -t "${THREADS}" -c "${ctx}" --chunks 4 --no-warmup \
    | tee "${OUTDIR}/perplexity_base_ctx${ctx}.log"
  ./build/bin/llama-perplexity -m "${SD_MODEL}" --seeddelta -f "${TEXT}" -t "${THREADS}" -c "${ctx}" --chunks 4 --no-warmup \
    | tee "${OUTDIR}/perplexity_sd_ctx${ctx}.log"
done

IFS=',' read -r -a rss_ctx_vals <<< "${RSS_CTX_LIST}"
for ctx in "${rss_ctx_vals[@]}"; do
  /usr/bin/time -v ./build/bin/llama-cli -m "${BASE}" -p "hola" -n 16 \
    -t "${THREADS}" -c "${ctx}" --ignore-eos --simple-io --temp 0 --top-k 1 --seed 1 --single-turn --no-warmup --no-display-prompt \
    > /dev/null 2> "${OUTDIR}/time_base_ctx${ctx}.txt" || true
  /usr/bin/time -v ./build/bin/llama-cli -m "${SD_MODEL}" --seeddelta -p "hola" -n 16 \
    -t "${THREADS}" -c "${ctx}" --ignore-eos --simple-io --temp 0 --top-k 1 --seed 1 --single-turn --no-warmup --no-display-prompt \
    > /dev/null 2> "${OUTDIR}/time_sd_ctx${ctx}.txt" || true
done

if [[ -n "${GREEDY_PACK}" && -f "${GREEDY_PACK}" ]]; then
  scripts/seeddelta-greedy-pack.sh \
    --base "${BASE}" \
    --sd "${SD_MODEL}" \
    --pack "${GREEDY_PACK}" \
    --outdir "${OUTDIR}/greedy_pack" \
    > "${OUTDIR}/greedy_pack.log" 2>&1
fi

echo "wrote results to: ${OUTDIR}"
