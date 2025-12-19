#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: seeddelta-ablate.sh --base BASE_GGUF --sd SD_GGUF --text TEXT_FILE --outdir DIR [options]

Runs quick ablations (PPL base vs SeedΔ and optional greedy pack) for fast regression checks.

Required:
  --base PATH     Base GGUF (sin SeedΔ)
  --sd PATH       GGUF con SeedΔ
  --text FILE     Texto para PPL (e.g. wikitext-2-raw/wiki.test.raw)
  --outdir DIR    Carpeta de salida (se crea si no existe)

Options:
  --pack FILE     Prompt pack para greedy smoke (usa llama-cli --single-turn)
  --threads N     Hilos (default 8)
  --ctx N         Context size (default 512)
  --chunks N      Chunks para PPL (default 8)
  --seed N        Semilla (default 1)
  --prompt STR    Prompt corto para smoke local (default "hola")

Outputs en --outdir:
  ppl_base.log, ppl_sd.log, (opcional) greedy_sd.log
EOF
}

BASE=""
SD=""
TEXT=""
OUTDIR=""
PACK=""
THREADS=8
CTX=512
CHUNKS=8
SEED=1
PROMPT="hola"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base) BASE="$2"; shift 2;;
    --sd) SD="$2"; shift 2;;
    --text) TEXT="$2"; shift 2;;
    --outdir) OUTDIR="$2"; shift 2;;
    --pack) PACK="$2"; shift 2;;
    --threads) THREADS="$2"; shift 2;;
    --ctx) CTX="$2"; shift 2;;
    --chunks) CHUNKS="$2"; shift 2;;
    --seed) SEED="$2"; shift 2;;
    --prompt) PROMPT="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1;;
  esac
done

if [[ -z "$BASE" || -z "$SD" || -z "$TEXT" || -z "$OUTDIR" ]]; then
  echo "Missing required args" >&2
  usage
  exit 1
fi

mkdir -p "$OUTDIR"

echo "== PPL base =="
./build/bin/llama-perplexity \
  -m "$BASE" \
  -f "$TEXT" \
  --chunks "$CHUNKS" -t "$THREADS" -c "$CTX" --no-warmup \
  | tee "$OUTDIR/ppl_base.log"

echo "== PPL SeedΔ =="
./build/bin/llama-perplexity \
  -m "$SD" \
  -f "$TEXT" \
  --chunks "$CHUNKS" -t "$THREADS" -c "$CTX" --no-warmup \
  --seeddelta \
  | tee "$OUTDIR/ppl_sd.log"

echo "== Smoke prompt SeedΔ =="
./build/bin/llama-cli \
  -m "$SD" \
  --seeddelta \
  -t "$THREADS" -c "$CTX" --temp 0 --top-k 1 --seed "$SEED" \
  --single-turn --no-warmup --simple-io --no-display-prompt \
  --prompt "$PROMPT" \
  | tee "$OUTDIR/smoke_sd.log"

if [[ -n "$PACK" ]]; then
  echo "== Greedy pack SeedΔ =="
  ./build/bin/llama-cli \
    -m "$SD" \
    --seeddelta \
    -t "$THREADS" -c "$CTX" --temp 0 --top-k 1 --seed "$SEED" \
    -f "$PACK" \
    --single-turn --no-warmup --no-display-prompt \
    | tee "$OUTDIR/greedy_sd.log"
fi
