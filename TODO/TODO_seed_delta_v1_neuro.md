# TODO - SeedDelta v1 (neuro-inspired plan)

Goal: push SeedDelta to a stable, repeatable v1 that improves PPL in the safe ctx
band and reduces GGUF/RSS via policy-driven strip, without opening ffn_up and
without breaking greedy smoke. This is a "close the loop" plan before the KV
pivot takes over.

Success criteria:
- Greedy smoke: PASS (0 flags) on the standard pack.
- PPL: <= 0% delta in ctx512/1024; acceptable <= +0.5% in ctx2048.
- RAM: GGUF and RSS drop together when strip is enabled.

All ideas are ordered from most promising to least, based on our current data.

---

## 0) Down viability triage (E8 hypotheses)

Why now:
- E8 showed big RAM wins from ffn_down strip, but quality degraded fast.
- Before we go "neuro", we need to test the 3 concrete hypotheses that can
  explain E8 and might unlock "down" safely.

Hypotheses to attack:
- H1: K_down=128 is too low for down (needs higher fidelity).
- H2: Our current safety signal does not measure down damage (wrong metric).
- H3: Error accumulates via residuals; we need a clamp or partial strip.

Plan (minimal, executable):
- [ ] H1: sweep K_down with gate fixed at K=128 (down in 1 layer only):
      K_down = 256, 384, 512. Measure PPL ctx1024/2048 + RSS ctx64/128.
- [ ] H2: use proxy-based thresholds for down only (stricter than gate):
      cos_p05 and l2_p95 thresholds from "good" layers + tighter margin.
- [ ] H3a: add delta norm clamp for down (tau * ||W||), keep gate unchanged.
- [ ] H3b: partial strip for down (top-K rows/cols by imatrix or by |delta|),
      evaluate if RAM drops without PPL blow-up.

Exit criteria:
- If any setting gets ctx2048 delta <= +2% with meaningful RSS drop, keep down.
- If not, keep down off and move on (gate-only stays the default).

---

## 1) Consolidation + selective forgetting (hippo -> neocortex)

Neuro idea:
- The brain stores only salient deltas and lets the rest decay.

SeedDelta mapping:
- Treat Delta as "error memory" with retention/decay.
- Keep only salient deltas; apply decay on non-salient deltas.

Implementation sketch:
- Add a per-layer/tensor "retention" score from proxy metrics.
- Apply decay: delta *= (1 - alpha * (1 - ret)).
- Hard cap on deltas per layer/tensor (top-K by |delta|).

MVP steps:
- [ ] Add retention score in scan output (proxy-based).
- [ ] Add decay/trim pass in build step or policy generator.
- [ ] Run with gate-only policy {13,15,18,20}; compare ctx512/1024/2048.

Exit:
- If PPL improves or stays flat and GGUF/RSS drop, keep; otherwise back off.

---

## 2) Homeostatic scaling (norm clamp)

Neuro idea:
- Synapses scale to keep neuron activity in a stable range.

SeedDelta mapping:
- Clamp or rescale delta by norm per tensor or layer.

Implementation sketch:
- Compute delta_norm; if > tau, rescale to tau.
- Optionally track ratio W0/delta to keep balance.

MVP steps:
- [ ] Add delta_norm clamp in seeddelta-build for ffn_gate only.
- [ ] Run with current best policy; check PPL and drift at ctx2048.

Exit:
- Keep if it improves stability without killing gains.

---

## 3) Dendritic gating (conditional delta)

Neuro idea:
- Gates open only when the input signal is in a certain regime.

SeedDelta mapping:
- Apply delta only when activation signal crosses a threshold.

Implementation sketch:
- Use proxy output stats (cos/l2/log_norm_ratio) as gate.
- If activation norm is too low/high, skip delta for that batch.

MVP steps:
- [ ] Add a "conditional enable" flag in policy for ffn_gate.
- [ ] Evaluate in ctx1024/2048 and compare to base gate-only.

Exit:
- Keep if it removes the ctx>1280 flip or reduces variance.

---

## 4) Predictive coding (delta stores surprise)

Neuro idea:
- Store only the error between prediction and reality.

SeedDelta mapping:
- Train or select delta only for outlier activations, not the median case.

Implementation sketch:
- Use proxy error distribution; keep delta only above a quantile.
- Strong L1 on delta to force sparsity on "surprises".

MVP steps:
- [ ] Add error-quantile filter in autogate (per layer/tensor).
- [ ] Compare against standard cos thresholds.

Exit:
- Keep if it improves long-ctx stability without losing mid-ctx gains.

---

## 5) Competitive gating (global budget)

Neuro idea:
- Neurons compete; only a few win per context.

SeedDelta mapping:
- Enforce a global budget across layers/tensors (not just per layer).

Implementation sketch:
- Rank by safety score; keep top N total tensors.
- Add "budget_global" in policy generator.

MVP steps:
- [ ] Add global budget in autogate (max tensors total).
- [ ] Evaluate PPL vs budget size; find sweet spot.

Exit:
- Keep if it stabilizes expansion while preserving PPL gains.

---

## 6) Replay-directed retention

Neuro idea:
- Replay reinforces what improves behavior.

SeedDelta mapping:
- Keep deltas that improve a small set of critical prompts.

Implementation sketch:
- Run a short replay pack; measure delta impact.
- Increase retention for deltas that improve those prompts.

MVP steps:
- [ ] Add a "replay score" in evaluation script.
- [ ] Use it to weight retention for deltas.

Exit:
- Keep if it reduces regression rate without hurting PPL.

---

## 7) Core vs leaf layers (route separation)

Neuro idea:
- Core circuits are stable; peripheral circuits are flexible.

SeedDelta mapping:
- Maintain a denylist for early layers unless a very strong functional signal appears.

Implementation sketch:
- Keep early layer cutoff (already works).
- Add a "core/leaf" mask by sensitivity band.

MVP steps:
- [ ] Add a simple layer-band policy: allow only in a safe band unless proxy score is extreme.
- [ ] Re-evaluate context flip boundary.

Exit:
- Keep if it avoids early-layer toxicity with minimal manual curation.

---

## Minimal plan to ship SeedDelta v1

Order of execution:
1) Consolidation + forgetting (retention/decay) on top of gate-only.
2) Homeostatic clamp (if needed).
3) Conditional gating (if ctx>1280 still flips).
4) Global budget.
5) Predictive coding filter (if still unstable).
6) Replay-directed retention (optional).
7) Core/leaf band (already a fallback).

Deliverables:
- Policy preset(s) + guardrails.
- Metrics table for ctx512/1024/2048.
- Clear "down is off by default" rule until a stronger filter exists.
