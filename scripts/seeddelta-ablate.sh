#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN_DIR="${BIN_DIR:-${ROOT_DIR}/build/bin}"
EVAL_SCRIPT="${EVAL_SCRIPT:-${ROOT_DIR}/scripts/seeddelta-policy-eval.sh}"

usage() {
  cat <<'USAGE'
Usage: seeddelta-ablate.sh --base BASE_GGUF --policy POLICY_JSON --layers LAYERS --text TEXT_FILE --outdir DIR [options]

Runs standard ablations using seeddelta-policy-eval.sh:
  - full policy
  - down-only
  - gate/up-only
  - single layer
  - bands

Required:
  --base PATH        Base GGUF (sin SeedΔ)
  --policy PATH      Policy JSON (base policy)
  --layers L-R       Layers to apply (e.g. 10-19)
  --text FILE        Texto para PPL (e.g. wikitext-2-raw/wiki.test.raw)
  --outdir DIR       Carpeta de salida

Options:
  --imatrix FILE     Imatrix GGUF (opcional)
  --pack FILE        Prompt pack para greedy pack
  --threads N        Hilos (default: nproc)
  --ctx N            Context size (default: 512)
  --chunks N         Chunks para PPL (default: 8)
  --scheme block|coo (default: block)
  --block N          Block size (default: 32; solo scheme=block)
  --no-strip-dense   Disable strip-dense in builder/runtime
  --no-row-scale     Disable row-scale in builder/runtime
  --no-base          Disable base (W0x) in builder/runtime
  --single-layer N   Layer exacta para ablation (default: first layer in --layers)
  --bands STR        Bandas comma-separated (default: 8-14,15-20,21-27)

Outputs:
  - One subdir per ablation in --outdir
  - ablate_summary.json + ablate_summary.md
USAGE
}

BASE=""
POLICY=""
LAYERS=""
TEXT=""
OUTDIR=""
IMATRIX=""
PACK=""
THREADS=""
CTX="512"
CHUNKS="8"
SCHEME="block"
BLOCK="32"
STRIP_DENSE="1"
ROW_SCALE="1"
USE_BASE="1"
SINGLE_LAYER=""
BANDS="8-14,15-20,21-27"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base) BASE="$2"; shift 2;;
    --policy) POLICY="$2"; shift 2;;
    --layers) LAYERS="$2"; shift 2;;
    --text) TEXT="$2"; shift 2;;
    --outdir) OUTDIR="$2"; shift 2;;
    --imatrix) IMATRIX="$2"; shift 2;;
    --pack) PACK="$2"; shift 2;;
    --threads) THREADS="$2"; shift 2;;
    --ctx) CTX="$2"; shift 2;;
    --chunks) CHUNKS="$2"; shift 2;;
    --scheme) SCHEME="$2"; shift 2;;
    --block) BLOCK="$2"; shift 2;;
    --no-strip-dense) STRIP_DENSE="0"; shift;;
    --no-row-scale) ROW_SCALE="0"; shift;;
    --no-base) USE_BASE="0"; shift;;
    --single-layer) SINGLE_LAYER="$2"; shift 2;;
    --bands) BANDS="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1;;
  esac
done

if [[ -z "$BASE" || -z "$POLICY" || -z "$LAYERS" || -z "$TEXT" || -z "$OUTDIR" ]]; then
  echo "Missing required args" >&2
  usage
  exit 1
fi

if [[ -z "$THREADS" ]]; then
  THREADS="$(nproc)"
fi

[[ -x "${EVAL_SCRIPT}" ]] || { echo "missing eval script: ${EVAL_SCRIPT}" >&2; exit 1; }
[[ -f "${BASE}" ]] || { echo "missing base model: ${BASE}" >&2; exit 1; }
[[ -f "${POLICY}" ]] || { echo "missing policy: ${POLICY}" >&2; exit 1; }
[[ -f "${TEXT}" ]] || { echo "missing text: ${TEXT}" >&2; exit 1; }
if [[ -n "${IMATRIX}" && ! -f "${IMATRIX}" ]]; then
  echo "missing imatrix: ${IMATRIX}" >&2
  exit 1
fi

mkdir -p "${OUTDIR}"
POLICY_DIR="${OUTDIR}/policies"
mkdir -p "${POLICY_DIR}"
RUNS_FILE="${OUTDIR}/ablate_runs.txt"
: > "${RUNS_FILE}"

first_layer_from_layers() {
  local token="${LAYERS%%,*}"
  token="${token%%-*}"
  echo "${token}"
}

make_variant_policy() {
  local mode="$1"
  local out_path="$2"
  python3 - "${POLICY}" "${out_path}" "${mode}" <<'PY'
import json
import sys
from pathlib import Path

src_path, dst_path, mode = sys.argv[1], sys.argv[2], sys.argv[3]
obj = json.loads(Path(src_path).read_text())

def apply_tensors(rule):
    tensors = rule.setdefault("tensors", {})
    def set_enabled(name, enabled):
        entry = tensors.setdefault(name, {})
        entry["enabled"] = bool(enabled)
    if mode == "down-only":
        set_enabled("ffn_gate", False)
        set_enabled("ffn_up", False)
        set_enabled("ffn_down", True)
    elif mode == "gateup-only":
        set_enabled("ffn_gate", True)
        set_enabled("ffn_up", True)
        set_enabled("ffn_down", False)

if isinstance(obj.get("defaults"), dict):
    apply_tensors(obj["defaults"])

for rng in obj.get("ranges", []):
    if isinstance(rng, dict):
        apply_tensors(rng)

for key, rule in obj.get("layers", {}).items():
    if isinstance(rule, dict):
        apply_tensors(rule)

Path(dst_path).write_text(json.dumps(obj, indent=2) + "\n")
PY
}

run_eval() {
  local name="$1"
  local policy_path="$2"
  local layers="$3"
  local run_dir="${OUTDIR}/${name}"
  rm -rf "${run_dir}"

  local args=(
    --base "${BASE}"
    --policy "${policy_path}"
    --layers "${layers}"
    --text "${TEXT}"
    --threads "${THREADS}"
    --ctx "${CTX}"
    --chunks "${CHUNKS}"
    --scheme "${SCHEME}"
    --outdir "${run_dir}"
  )

  if [[ "${SCHEME}" == "block" ]]; then
    args+=( --block "${BLOCK}" )
  fi
  if [[ -n "${IMATRIX}" ]]; then
    args+=( --imatrix "${IMATRIX}" )
  fi
  if [[ "${STRIP_DENSE}" == "0" ]]; then
    args+=( --no-strip-dense )
  fi
  if [[ "${ROW_SCALE}" == "0" ]]; then
    args+=( --no-row-scale )
  fi
  if [[ "${USE_BASE}" == "0" ]]; then
    args+=( --no-base )
  fi
  if [[ -n "${PACK}" ]]; then
    args+=( --greedy-pack "${PACK}" )
  fi

  echo "${name}|${run_dir}" >> "${RUNS_FILE}"
  "${EVAL_SCRIPT}" "${args[@]}"
}

if [[ -z "${SINGLE_LAYER}" ]]; then
  SINGLE_LAYER="$(first_layer_from_layers)"
fi

DOWN_POLICY="${POLICY_DIR}/policy.down-only.json"
GATEUP_POLICY="${POLICY_DIR}/policy.gateup-only.json"
make_variant_policy "down-only" "${DOWN_POLICY}"
make_variant_policy "gateup-only" "${GATEUP_POLICY}"

run_eval "full" "${POLICY}" "${LAYERS}"
run_eval "down-only" "${DOWN_POLICY}" "${LAYERS}"
run_eval "gateup-only" "${GATEUP_POLICY}" "${LAYERS}"
run_eval "single-layer-${SINGLE_LAYER}" "${POLICY}" "${SINGLE_LAYER}"

IFS=',' read -r -a bands_arr <<< "${BANDS}"
for band in "${bands_arr[@]}"; do
  band_trimmed="${band//[[:space:]]/}"
  [[ -z "${band_trimmed}" ]] && continue
  run_eval "band-${band_trimmed}" "${POLICY}" "${band_trimmed}"
done

python3 - "${OUTDIR}" "${RUNS_FILE}" <<'PY'
import json
import re
import sys
from pathlib import Path

outdir = Path(sys.argv[1])
runs_file = Path(sys.argv[2])

ppl_re = re.compile(r"Final estimate: PPL = ([0-9.]+)")
result_re = re.compile(r"RESULT: (PASS|FAIL)")

def read_text(path: Path) -> str:
    try:
        return path.read_text(errors="replace")
    except FileNotFoundError:
        return ""

def parse_ppl(path: Path):
    text = read_text(path)
    m = ppl_re.findall(text)
    return float(m[-1]) if m else None

def parse_result(path: Path):
    text = read_text(path)
    m = result_re.findall(text)
    return m[-1] if m else "SKIP"

def parse_time(path: Path):
    text = read_text(path)
    m_elapsed = re.search(r"Elapsed \(wall clock\) time .*: ([0-9:.]+)", text)
    m_rss = re.search(r"Maximum resident set size \(kbytes\): ([0-9]+)", text)
    return {
        "elapsed": m_elapsed.group(1) if m_elapsed else None,
        "rss_kb": int(m_rss.group(1)) if m_rss else None,
    }

def parse_report(path: Path):
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    weights = data.get("weights", [])
    emit = [w for w in weights if w.get("emit")]
    emit_layers = sorted({w.get("layer") for w in emit if w.get("layer") is not None})
    emit_kinds = sorted({w.get("kind") for w in emit if w.get("kind")})
    strip = [w for w in emit if w.get("strip_dense")]
    emit_by_kind = {}
    for w in emit:
        emit_by_kind[w.get("kind")] = emit_by_kind.get(w.get("kind"), 0) + 1
    stack_cost_sum = 0.0
    for w in emit:
        val = w.get("stack_cost_total")
        if isinstance(val, (int, float)):
            stack_cost_sum += float(val)
    return {
        "emit_total": len(emit),
        "emit_by_kind": emit_by_kind,
        "emit_layers": emit_layers,
        "strip_total": len(strip),
        "strip_layers": sorted({w.get("layer") for w in strip if w.get("layer") is not None}),
        "stack_cost_total_sum": stack_cost_sum,
    }

runs = []
for line in runs_file.read_text().splitlines():
    if not line.strip():
        continue
    name, run_path = line.split("|", 1)
    run_dir = Path(run_path)
    entry = {
        "name": name,
        "outdir": str(run_dir),
        "ppl_base": parse_ppl(run_dir / "perplexity_base.log"),
        "ppl_sd": parse_ppl(run_dir / "perplexity_seeddelta.log"),
        "greedy_result": parse_result(run_dir / "greedy_pack.log"),
        "time_base": parse_time(run_dir / "time_base.txt"),
        "time_sd": parse_time(run_dir / "time_seeddelta.txt"),
    }
    entry.update(parse_report(run_dir / "report.json"))
    runs.append(entry)

summary = {
    "runs": runs,
}

(outdir / "ablate_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

lines = ["# SeedDelta ablations", ""]
for run in runs:
    lines.append("## {}".format(run["name"]))
    lines.append("- outdir: `{}`".format(run["outdir"]))
    lines.append("- PPL base: {} | SD: {}".format(run["ppl_base"], run["ppl_sd"]))
    lines.append("- greedy: {}".format(run["greedy_result"]))
    lines.append("- time base: {} (RSS {} KB)".format(run["time_base"].get("elapsed"), run["time_base"].get("rss_kb")))
    lines.append("- time SD: {} (RSS {} KB)".format(run["time_sd"].get("elapsed"), run["time_sd"].get("rss_kb")))
    lines.append("- emit_total: {} | emit_by_kind: {}".format(run.get("emit_total"), run.get("emit_by_kind")))
    lines.append("- strip_total: {} | strip_layers: {}".format(run.get("strip_total"), run.get("strip_layers")))
    lines.append("- stack_cost_total_sum: {}".format(run.get("stack_cost_total_sum")))
    lines.append("")

(outdir / "ablate_summary.md").write_text("\n".join(lines))
PY

echo "wrote summary: ${OUTDIR}/ablate_summary.json"
