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

## Key discoveries

- `ffn_gate` is compressible with low risk under cos-based gating.
- `ffn_up` is sensitive: opening it degrades PPL; closing it restores the win.
- A simple filter using `cos_mean_x_w` and `cos_p05_x_w` captures "safe tensors"
  consistently across Q8 and F16 in the tested ranges.
- Gate-only policies can *improve* PPL while keeping greedy PASS.

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

## Current plan (summary)

1) Gate-only coverage expansion:
   - Sweep `max_layers` (8/16/24/32) on wider ranges.
   - Track PPL, GGUF size, RSS, tok/s.
2) Gate + down (up-off):
   - Tight thresholds first; relax only if stable.
3) Strip real + RAM measurement:
   - Confirm non-zero emitted tensors and strip in logs.
   - Measure RSS with small ctx where weight savings are visible.

## Where to read more (living notes)

- `TODO/TODO_seed_delta_debug_toy_model.md`
- `TODO/TODO_seed_delta_weights_k_layer_sublayer.md`
- `TODO/TODO_seed_delta_weights_llama_k_multicapa.md`
- `TODO/TODO_seed_delta_weights_llama.md`
- `TODO/TODO_statecells_sparse_dict_llama.md`
