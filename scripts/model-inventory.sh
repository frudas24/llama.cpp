#!/usr/bin/env bash
#
# Lists GGUF models available locally along with size and a few hints (quant inferred from filename).

set -euo pipefail

print_usage() {
  cat <<'EOF'
usage: scripts/model-inventory.sh [DIR ...]

Lists *.gguf files for each DIR (default: $HOME/.cache/llama.cpp and calibration/).
Outputs: size (GiB), quant tag (best-effort from filename), and path.

Environment:
  INVENTORY_FORMAT=table|plain   default=table
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  print_usage
  exit 0
fi

declare -a roots=()
if [[ "$#" -gt 0 ]]; then
  roots=("$@")
else
  roots+=("$HOME/.cache/llama.cpp")
  roots+=("calibration")
fi

format="${INVENTORY_FORMAT:-table}"

print_row() {
  local size_bytes="$1"
  local quant="$2"
  local path="$3"
  local size_gib
  size_gib=$(awk -v b="$size_bytes" 'BEGIN { printf "%.2f", b / (1024*1024*1024) }')
  if [[ "${format}" == "plain" ]]; then
    printf "%s GiB\t%s\t%s\n" "${size_gib}" "${quant}" "${path}"
  else
    printf "%10s GiB | %-12s | %s\n" "${size_gib}" "${quant}" "${path}"
  fi
}

infer_quant() {
  local filename="$1"
  if [[ "${filename}" =~ (Q[0-9]_[A-Z]_[A-Z]) ]]; then
    printf "%s" "${BASH_REMATCH[1]}"
  elif [[ "${filename}" =~ (Q[0-9]+_[A-Z]) ]]; then
    printf "%s" "${BASH_REMATCH[1]}"
  elif [[ "${filename}" =~ (F16|F32|BF16) ]]; then
    printf "%s" "${BASH_REMATCH[1]}"
  else
    printf "unknown"
  fi
}

for root in "${roots[@]}"; do
  if [[ ! -d "${root}" ]]; then
    >&2 echo "warn: skipping missing dir ${root}"
    continue
  fi
  [[ "${format}" == "plain" ]] || printf "\nDIR: %s\n" "${root}"
  shopt -s nullglob
  found=0
  for file in "${root}"/*.gguf; do
    found=1
    size=$(stat -c "%s" "${file}")
    quant=$(infer_quant "$(basename "${file}")")
    print_row "${size}" "${quant}" "${file}"
  done
  shopt -u nullglob
  if [[ ${found} -eq 0 ]]; then
    echo "(no gguf files)"
  fi
done
