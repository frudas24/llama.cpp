# TODO — Atención híbrida ventana+global ligera para CPU

> **Objetivo:** inspirarse en WGAE (TODO_N9) para crear un modo de atención CPU eficiente: atención local por ventanas + proyección global barata que mantiene contexto, reduciendo coste en secuencias largas.

---

## Índice
1) Idea y params
2) Config/flags
3) Integración
4) Pseudocódigo
5) Métricas/aceptación
6) Roadmap
7) Riesgos
8) Commits sugeridos

---

### 1) Idea y params
- Ventanas deslizantes tamaño `w` (p.ej. 64–128) con stride `s`; atención global opcional con resumen `g = P·mean(Q)` (d pequeño).
- Score final = atención local + proyección global fusionada (concat o suma).
- Params: `w`, `stride`, `d_global`, `mix` (0–1).

### 2) Config/flags
- `--win-attn=on`, `--win-size=128`, `--win-stride=64`, `--win-global-d=64`, `--win-mix=0.5`.
- Scope: capas objetivo (primeras N o alternas).

### 3) Integración
- Reemplazar atención completa por:
  - Local: atención dentro de cada ventana (block sparse).
  - Global: proyección `g` (MLP pequeña) aplicada a todos los tokens; agregar score `mix*score_global + (1-mix)*score_local`.
- Softmax por ventana; global puede usar shared key/vector para todos.

### 4) Pseudocódigo
```cpp
for each window W_t:
  Qw, Kw, Vw = slice(Q,K,V, W_t)
  local = softmax(Qw @ Kw^T / sqrt(d)) @ Vw

global_key = Pk @ mean(Q)         // d_global << d
global_val = Pv @ mean(V)
global_score = (Qg @ global_key)  // Qg = Pg @ Q

Y = local + mix * (global_score * global_val)
```

### 5) Métricas/aceptación
- `win_attn.ms`, `win_attn.speedup`, `win_attn.mix`, `win_attn.ppl_gap`.
- Aceptar si speedup CPU ≥15% en capas activas con Δppl ≤2%.

### 6) Roadmap
1. Kernel atención ventana (block-sparse) + tests.
2. Proyección global (Pk/Pv) compartida + mezcla.
3. Flags por capa y bench tkn/s + ppl gap.
4. Autotune stride/size según longitud de secuencia.

### 7) Riesgos
- Mezcla inestable: clamp `mix` y normalizar global_val.
- Stride grande → artefactos: usar overlap (stride<w) para suavizar.
- Pérdida de largo alcance: subir d_global o insertar capas densas cada M bloques.

### Compatibilidad e interacciones
- Para **capas de atención**, elegir un solo patrón entre `--win-attn`, `--sdr-target=attn_light` o `--event-driven` en atención; evitar compuestos que incrementen `gap` (event‑driven puede seguir en MLP).
- KWTA en atención puede operar por ventana/head, pero debe usar la misma segmentación `w/stride`.
- Backends de pesos (ternario/TT/procedural/statecells) siguen aplicando a proyecciones Q/K/V/O y a Pk/Pv.

### 8) Commits sugeridos
- `attn: add windowed sparse kernel + tests`
- `attn: add global projection mix path`
- `runtime: flags/metrics for window+global attention`
