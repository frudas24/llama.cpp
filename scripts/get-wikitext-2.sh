#!/usr/bin/env bash

set -euo pipefail

URL="https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip"
ZIP="wikitext-2-raw-v1.zip"
DIR="wikitext-2-raw"

if [[ -d "${DIR}" ]]; then
  echo "already exists: ${DIR}"
  exit 0
fi

echo "downloading ${URL}"
if command -v wget >/dev/null 2>&1; then
  wget -O "${ZIP}" "${URL}"
elif command -v curl >/dev/null 2>&1; then
  curl -L -o "${ZIP}" "${URL}"
else
  echo "error: need wget or curl to download datasets" >&2
  exit 1
fi

echo "extracting ${ZIP}"
if command -v unzip >/dev/null 2>&1; then
  unzip -q "${ZIP}"
else
  if ! command -v python3 >/dev/null 2>&1; then
    echo "error: need unzip or python3 to extract ${ZIP}" >&2
    exit 1
  fi
  python3 - <<'PY'
import zipfile

zip_path = "wikitext-2-raw-v1.zip"
with zipfile.ZipFile(zip_path, "r") as zf:
    zf.extractall(".")
print("extracted:", zip_path)
PY
fi

echo "Usage:"
echo ""
echo "  ./llama-perplexity -m model.gguf -f wikitext-2-raw/wiki.test.raw [other params]"
echo ""

exit 0
