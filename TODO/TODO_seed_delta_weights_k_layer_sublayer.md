# TODO — SeedΔ K por capa/subcapa (perf, data-aware y targets extra)

Este TODO agrupa el backlog **pendiente** del mecanismo SeedΔ multicapa, pero visto desde la óptica
de “K por capa/subcapa” y estabilidad a través de arquitecturas/precisiones:

- Optimización de kernels/base (`W0x`) y layout.
- Data-awareness real (activaciones, dataset por modelo).
- Multi-precisión (FP16/Q8/Q4) como fuente para SeedΔ.
- Extender SeedΔ fuera de FFN (`ffn_gate/up/down`) hacia QKV/O, embeddings y LM head.
- Harness de pruebas cruzadas (modelos pequeños vs grandes).

Documento de origen: ver también `TODO_seed_delta_weights_llama_k_multicapa.md`
para el diseño del builder/policy/gating/autotune y resultados actuales.

---

## 1) Perf / RAM (sigue siendo el cuello)

- [ ] Optimizar `W0x` (cache por token/batch; vectorizar base; reducir overhead en base+residual).
- [ ] Reducir overhead de índices/layout (u16 contiguo, mejor packing por bloque).
- [ ] Clarificar repack vs RSS: tener medición consistente (time -v + logs) y explicar picos vs steady.
- [ ] Instrumentar stack-safety: medir cuántos tensores pasan por policy vs cuántos terminan activos tras endurecer thresholds.
  - Reportar en `build.stack_budget` para correlacionar con estabilidad (greedy/PPL) en los reports.

## 2) Calidad / data-aware “de verdad”

- [ ] Mejorar dataset para imatrix por modelo (no depender de `gemma_calibration.txt` para todo).
- [ ] (Opcional) capturar activaciones reales por tensor (`X`) y optimizar `||WX - ŴX||` más allá de imatrix diagonal
      (objetivo funcional: `||WX - ŴX||` con activaciones reales en `ffn_gate/up/down`).
- [ ] Explorar construir SeedΔ desde F16/Q8 como fuente (evitar “loss-on-loss” de Q4).
- [ ] Validar gating multi-precisión: repetir builder + policy en al menos tres precisiones (FP16, Q8, Q4)
      para el mismo modelo y documentar diferencias de `cos_x_w` / `stack_budget`.
- [ ] Añadir metadata `seeddelta.source_precision` o similar al GGUF/export para rastrear con qué
      precisión se generó SeedΔ (p.ej. `fp16`, `q8_0`, `q4_k_m`).

## 3) Expandir targets más allá de FFN (Fase 5)

- [ ] Probar policy/gating en QKV/O con budgets conservadores (solo si gating indica que hay margen).
- [ ] Embeddings/LM head solo con gating ultra estricto (e.g. umbrales de cos muy altos y stack-budget mínimo).

## 4) Modelos y pruebas cruzadas

- [ ] Smoke tests obligatorios en modelos pequeños (Gemma 1B/4B) cada vez que se toque policy/gating
      para detectar regresiones temprano (PPL+greedy pack).

