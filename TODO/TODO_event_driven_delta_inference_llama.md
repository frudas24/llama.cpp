# TODO — Inferencia “event‑driven” por deltas + presupuesto (CPU)

> **Objetivo:** inspirarse en el fan‑in event‑driven (C02) para reducir cómputo por token en CPU: procesar sólo bloques/cabezas con |Δ| relevantes bajo un presupuesto temporal, imputando el resto con estado previo (staleness) y fallback seguro a camino denso si el gap sube.

---

## Índice
1) Idea y parámetros  
2) Config/flags  
3) Integración (MLP / Atención)  
4) Pseudocódigo  
5) Métricas/aceptación  
6) Roadmap  
7) Riesgos  
8) Commits sugeridos

---

### 1) Idea y parámetros
- **Delta‑coding:** por token, medir `Δx = ||x_t − x_{t−1}||` por bloque (cols de MLP, heads, tiles).  
- **Quórum/top‑K:** sólo computar los K bloques con mayor Δ hasta alcanzar convergencia o presupuesto `Tbudget`.  
- **Staleness:** para bloques no computados, reutilizar salida anterior con peso `w = β·exp(−λ·age)` y renormalizar.  
- **Gap check:** cada `N` tokens hacer cómputo full y medir `quality_gap`; si `gap>gap_tol` → fallback denso.

Parámetros típicos:
- `Tbudget_ms` (p.ej. 2–6 ms/token CPU), `quorum_frac` (0.5–0.7), `conv_eps`, `conv_J` (repeticiones de convergencia), `stale_beta`, `stale_lambda`, `delta_theta`.

### 2) Config/flags
- `--event-driven=on|off`  
- `--event-budget-ms=4`  
- `--event-quorum=0.6`  
- `--event-topk=0.4` (fracción de bloques)  
- `--event-delta-theta=0.01`  
- `--event-gap-tol=0.03`  
- Scope: `--event-target=mlp|attn|both`.

### 3) Integración
- **MLP:** dividir proyección/FFN en bloques (32/64 cols). Calcular Δ por bloque desde activación previa; computar top‑K hasta `Tbudget`; el resto se deja “stale”.  
- **Atención:** bloquear por heads o por ventanas de tokens; computar sólo heads/ventanas con alta Δ de Q/K (o energía). Los demás usan scores previos amortiguados.  
- **Fallback:** si CPU sin ganancia o gap alto, desactivar dinámicamente.

### 4) Pseudocódigo
```cpp
// per token
deadline = now + Tbudget_ms
scores = delta_norm_per_block(x_t, x_prev)
order = topk_indices(scores, K)

y = 0; wsum = 0
for b in order:
  if now > deadline: break
  y_b = compute_block(b, x_t)
  cache_y[b] = y_b; cache_age[b]=0
  y += y_b; wsum += 1

// staleness blocks
for b not computed:
  age = ++cache_age[b]
  w = stale_beta * exp(-stale_lambda * age)
  y += w * cache_y[b]; wsum += w

y /= max(eps, wsum)

// cada N tokens: gap check
if step % gap_every == 0:
  y_full = compute_dense(x_t)
  gap = |y - y_full| / |y_full|
  if gap > gap_tol: disable_event_driven()
```

### 5) Métricas/aceptación
- `event.enabled`, `event.budget_ms_used`, `event.quorum_hits`, `event.topk_kept`, `event.quality_gap`, `event.stale_pct`.  
- Aceptar si speedup CPU ≥10–15% en prompts largos con `quality_gap ≤3%` vs denso.

### 6) Roadmap
1. Infra delta_norm + cache por bloque/head + staleness weights.  
2. Hook MLP bloqueado por top‑K bajo presupuesto.  
3. Hook atención por heads/ventanas bajo presupuesto.  
4. Gap‑checker y auto‑disable; benchmarks.

### 7) Riesgos
- Overhead de ranking > ahorro: usar bloques grandes y `nth_element`/heap pequeño.  
- Drift de staleness: renormalizar pesos y limitar `age_max`.  
- Calidad sensible a tareas: exponer knobs por capa; fallback frecuente al inicio.

### Compatibilidad e interacciones
- Si `--kwta` está activo, componer: KWTA selecciona candidatos y event‑driven aplica presupuesto/top‑K final dentro de esos candidatos (una sola ordenación).  
- StatePack puede usarse como máscara previa de ceros antes de medir Δ; no ejecutar rutas approx popcount y event‑driven sobre el mismo score sin refine denso.  
- En atención, no mezclar con `--win-attn`/`--sdr-target=attn_light` en la misma capa salvo que event‑driven opere a nivel de ventanas y SDR sea solo filtro top‑K.  
- Para capas TT/StateCells sin bloques claros, event‑driven debe estar off o operar a nivel de capa completa.

### 8) Commits sugeridos
- `event: add delta-norm + block cache infra`  
- `mlp: event-driven topk blocks under budget`  
- `attn: event-driven heads/windows under budget`  
- `runtime: gap checker + metrics + auto-disable`
