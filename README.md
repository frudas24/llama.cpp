# SeedDelta research fork of llama.cpp

This repo is a research fork of https://github.com/ggml-org/llama.cpp focused on SeedDelta and StateCells. The goal is
to reduce model RAM/size while keeping greedy stability and PPL within a small budget.

Essentials:
- Upstream docs and build instructions live in the main llama.cpp repo.
- Most planning, experiments, and decisions are tracked in the TODO docs below.
- Primary tools: `tools/seeddelta-build/`, `scripts/seeddelta-layer-scan.py`,
  `scripts/seeddelta-autogate.py`, `scripts/seeddelta-policy-eval.sh`.

## Project evolution and pivots

We iterated through multiple approaches and kept the pivots explicit in TODO files:

1) StateCells sparse dict experiments:
   - `TODO/TODO_statecells_sparse_dict_llama.md`
2) SeedDelta FFN weight replacement (first full pass):
   - `TODO/TODO_seed_delta_weights_llama.md`
3) Multi-layer K scheduling and policy evolution:
   - `TODO/TODO_seed_delta_weights_llama_k_multicapa.md`
4) K per sublayer / tiling and policy refinement:
   - `TODO/TODO_seed_delta_weights_k_layer_sublayer.md`
5) Debug harness + toy models to isolate errors:
   - `TODO/TODO_seed_delta_debug_toy_model.md`

The key pivot was moving from "manual layer parity" to a reproducible pipeline:
per-layer scan -> tensor-aware autogate -> policy generation -> eval loop.

## Timeline of pain, pivots, and wins

This is the honest arc of the project so far, mapped to the TODOs. It’s not a
marketing timeline: it’s the “battle log” that explains why the current system
looks the way it does.

### Act I — StateCells sparse dict (exploration)
- Goal: dictionary-style compression (codes + dict) to reduce weights/RAM.
- Outcome: end-to-end pipeline worked, but “too aggressive too early” destroyed quality.
  - Reference failure (Gemma, layers 0–1, very aggressive dict config): PPL `26.7485 → 8017.7608`.
- Doc: `TODO/TODO_statecells_sparse_dict_llama.md`

### Act II — SeedΔ weights v1 (first end-to-end)
- Goal: represent `W` as procedural base + sparse residual: `W ≈ W0(seed,θ) + Δ`, and run it in runtime.
- Crucial discovery: **scale is not optional**. Without `row_scale`/amplitude control, PPL goes to space.
- Early win (Gemma 1B, layers 10–11, `K_gate=64 K_up=64 K_down=128`, `ctx=512`):
  - base PPL `≈ 1.0050`
  - SeedΔ(coo) PPL `≈ 1.0039`
  - SeedΔ(block16) PPL `≈ 1.0018`
  - (Throughput dropped; accepted as the price of RAM experiments.)
- Doc: `TODO/TODO_seed_delta_weights_llama.md`

### Act III — K scheduling / stack-safety (multi-layer reality check)
- Goal: “just compress more layers” by allocating K across layers.
- Painful lesson: **stability is not additive**. Many tensors can pass local gating and still break globally.
  - Mistral 7B Q8 baseline (wikitext-2, `ctx=512`, `chunks=4`): PPL `≈ 7.3049 ± 0.63`.
  - Aggressive policy (layers 14–23, many gate/up/down): PPL `≈ 40.94` (model becomes semantically “weird”).
  - Conservative down-only (same range): only `blk.23.ffn_down` passed; PPL `≈ 7.58` (small degradation).
- This is where “stack budget” became non-negotiable: count what you emit, cap it, and treat stacking as a first-class constraint.
- Doc: `TODO/TODO_seed_delta_weights_llama_k_multicapa.md`

### Act IV — K per sublayer / tiles (granularity and traps)
- Goal: finer control than “per-layer”: tiles/sublayers and per-tensor gating (`ffn_gate/up/down`).
- Key trap discovered (repeatedly): **greedy PASS ≠ quality**. PPL can explode while the model still “answers”.
- Another core lesson emerged here and never went away: `ffn_up` is usually the first tensor that poisons quality under current proxies.
- Doc: `TODO/TODO_seed_delta_weights_k_layer_sublayer.md`

### Act V — Debug harness + autogate (turning “magic” into engineering)
- Goal: stop guessing (parity, hand-picked layers). Measure sensitivity and generate a policy deterministically.
- Pivot: **scan → autogate → policy → eval** became the main loop.
- This act produced the first “triple win” (quality + RAM + reproducibility):
  - Gate-only + strip, policy `{13,15,18,20}` on Gemma 4B:
    - GGUF delta `≈ -85.8 MiB`
    - RSS delta `≈ -85 MiB` (ctx probes)
    - PPL improved at ctx512 and ctx2048 (greedy pack PASS as anti-collapse smoke).
- New frontier discovered: **context regime**. The same policy can improve PPL up to ~1024 ctx and flip sign beyond ~1280 unless bucketed.
- Doc: `TODO/TODO_seed_delta_debug_toy_model.md`

### Act VI — Gate+down reality check and pivot to KV procedural
- Goal: use `ffn_down` as the “RAM lever”.
- Result: strong GGUF/RSS savings but **clear PPL regression** even with conservative down:
  - Mistral 7B Q8, E8a (down only on layer 20): ΔPPL `+11.95%` (ctx1024), `+13.27%` (ctx2048), GGUF `≈ -266.8 MiB`.
  - E8b (down on 18+20): ΔPPL `+18.29%` (ctx1024), `+18.10%` (ctx2048), GGUF `≈ -324.1 MiB`.
- Conclusion: **down needs a stronger functional filter**; pausing down as default.
- Pivot: **the real RAM enemy is KV** at long ctx. We pivoted to procedural KV.
- Doc: `TODO/TODO_kv_procedural_long_context_100k.md`

**Core lessons so far**
- Gate-only can be a *regularizer* (PPL improves) but only in safe bands.
- `ffn_up` is toxic under current metrics; hard-off by default.
- Context regime is real; policies must be ctx-aware or bucketed.
- Strip must be policy-driven to see real GGUF/RSS savings.

## Selected “war stories” (numbers that changed the design)

These are the experiments that forced design decisions (not just “interesting results”):

- **StateCells can destroy quality if applied early/aggressively**:
  - PPL `26.7485 → 8017.7608` (Gemma, layers 0–1, aggressive dict).
- **Local tensor gating does not guarantee global quality**:
  - Mistral 7B Q8: baseline `≈ 7.30`, aggressive 14–23 policy `≈ 40.94`.
- **Stack budget matters more than being “right” locally**:
  - Mistral 7B Q8: down-only 14–23 emitted 1 tensor and stayed near baseline (`≈ 7.58`).
- **Greedy smoke is a safety gate, not an exact-match test**:
  - After real weight changes, diffs vs base are expected; we treat greedy as anti-collapse heuristics.
- **Policy-driven strip is the difference between theory and RAM**:
  - When strip is actually applied (confirmed in `report.json`), GGUF and RSS move together.

## Key discoveries

- `ffn_gate` is compressible with low risk under cos-based gating.
- `ffn_up` is sensitive: opening it degrades PPL; closing it restores the win.
- A simple filter using `cos_mean_x_w` and `cos_p05_x_w` captures "safe tensors"
  consistently across Q8 and F16 in the tested ranges.
- Gate-only policies can *improve* PPL while keeping greedy PASS.
- Strip must be policy-driven to see real GGUF/RSS savings; CLI flags are secondary.

## Current working configuration

- Gate-only by default:
  - `ffn_gate` allowed if `cos_mean_x_w >= 0.70` and `cos_p05_x_w >= 0.50`
  - `ffn_up` deny-by-default (only with `--allow-up`)
  - `ffn_down` tested separately with stricter thresholds
- `min_gap = 1` to avoid consecutive-layer cliffs
- Evaluate with:
  - greedy pack PASS (20/20)
  - PPL delta goal: <= 0% ideal, <= +0.5% for RAM mode
  - RSS measured with small ctx (128/256) + normal ctx (512/2k)

## Milestones (validated)

- SeedDelta scan + autogate scripts produce stable policies.
- Gate-only policy on Gemma 4B Q8/F16 lowered PPL and kept greedy PASS.
- Autogate can now be tensor-aware and deny `ffn_up` by default.
- E7c gate-only + strip on `{13,15,18,20}` reduced GGUF/RSS by ~85 MiB and
  improved PPL at ctx512/2048 with greedy PASS.
- Context regime is real: gains are strongest at ctx512-1024 and can flip
  positive past ~1280 without bucketed policies.
- E8a/E8b (gate+down) showed **RAM wins but PPL loss** at 1–2 down layers → down is now backlog.

## Current plan (post-pivot)

1) **KV procedural MVP** (window + summaries + log-mass bias):
   - Implement tiered KV (W + block summaries + outliers) with budget caps.
   - Add shadow drift metrics for safety.
2) **Multi-level consolidation**:
   - Compact summaries in LSM-style levels to keep memory sublinear.
3) **Bench long-ctx**:
   - ctx 8k/32k/100k RSS/PPL/tok/s tables with presets (W,B,r,s).
4) **SeedΔ down** stays as backlog until we have a stronger functional filter.

## E8 (gate + down, up-off) workflow

We now treat `ffn_down` as the "RAM lever" and test it only after gate-only
is stable across ctx buckets.

Helpers (single source of truth):
- `scripts/seeddelta-e8-make-policy.py`
- `scripts/seeddelta-e8-run.sh`
- `scripts/seeddelta-e8-summary.py`

Example:
```
# E8a: down only on layer 20
scripts/seeddelta-e8-make-policy.py \
  --out calibration/e8a_down20/policy_down20.json \
  --layers 13,15,18,20 --down 20 --k 128 --strip-dense

scripts/seeddelta-e8-run.sh \
  --base /models/gemma-3-4b-it-Q8_0.gguf \
  --policy calibration/e8a_down20/policy_down20.json \
  --outdir calibration/e8a_down20 \
  --imatrix calibration/gemma4b.imatrix.gguf

scripts/seeddelta-e8-summary.py calibration/e8a_down20
```

Notes:
- `--strip-dense` in the runner is optional; policy is authoritative.
- Always measure PPL on ctx1024/2048 plus RSS probes at ctx64/128.

## Operational notes

- `/models` is the canonical model root on remotes (symlinked to cache).
- Prefer `--imatrix` when using `cos_x_w` gating.
- Keep greedy pack as smoke (anti-collapse), not exact-match.

## Where to read more (living notes)

- `TODO/TODO_seed_delta_debug_toy_model.md`
- `TODO/TODO_seed_delta_v1_neuro.md`
- `TODO/TODO_kv_procedural_long_context_100k.md`
- `TODO/TODO_seed_delta_weights_k_layer_sublayer.md`
- `TODO/TODO_seed_delta_weights_llama_k_multicapa.md`
- `TODO/TODO_seed_delta_weights_llama.md`
- `TODO/TODO_statecells_sparse_dict_llama.md`
