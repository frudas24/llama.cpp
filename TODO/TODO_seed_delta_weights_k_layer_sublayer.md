# TODO — SeedΔ K por capa/subcapa (perf, data-aware y targets extra)

Este TODO agrupa el backlog **pendiente** del mecanismo SeedΔ multicapa, pero visto desde la óptica
de “K por capa/subcapa” y estabilidad a través de arquitecturas/precisiones:

* Optimización de kernels/base (`W0x`) y layout.
* Data-awareness real (activaciones, dataset por modelo).
* Multi-precisión (FP16/Q8/Q4) como fuente para SeedΔ.
* Extender SeedΔ fuera de FFN (`ffn_gate/up/down`) hacia QKV/O, embeddings y LM head.
* Harness de pruebas cruzadas (modelos pequeños vs grandes).

Documento de origen: ver también `TODO_seed_delta_weights_llama_k_multicapa.md`
para el diseño del builder/policy/gating/autotune y resultados actuales.

Estado rápido (marcar avances):

- [x] Schema de `report.json` extendido (metric_used/targets_used, reject_reason, tiles/K_custom, stack_cost_* placeholders).
- [x] Calcular y volcar `stack_cost_*` v1 (tau_* desde gating, coste = max(0, d_mean+d_p05), alpha=beta=1).
 - [x] Poblar k_per_tile/unique_k_count/tiles_* cuando haya tiles/K_custom; hoy quedan vacíos.
 - [x] FFN-score compuesto implementado y usado en report/gating.

---

## 0) Definiciones (para evitar ambigüedad)

Estas definiciones son **v1 (MVP)** para que “K por subcapa/tile” y “Δ por etapas” signifiquen lo mismo
cuando se implementen.

* **Tensor objetivo**: en v1, matrices FFN `ffn_gate`, `ffn_up`, `ffn_down` por capa `blk.N`.

* **Subcapa/tile (v1)**:

  * *Tiles por filas*: rangos contiguos de filas de la matriz ($W \in \mathbb{R}^{n_{out}\times n_{in}}$).
  * Cada tile $t$ define una submatriz ($W_t = W[r_t:r_{t+1}, :]$).
  * Parámetros v1:
    * `tile_rows`: filas por tile (altura del tile). Debe mantenerse fijo por run (comparabilidad).
      * Recomendación v1: `tile_rows = 1024` como default, y solo permitir presets `{512, 1024, 2048}`.
      * Si `n_out < tile_rows`, usar `tile_rows = n_out` (un solo tile).
    * `tile_rows_align`: alineación de `tile_rows` y límites $r_t$ (múltiplos de 32/64) para mantener kernels/layout cache-friendly.
      * Nota: `tile_rows_align` es solo granularidad del tile; no confundir con `--block` del scheme residual.
  * Restricción v1: los límites $r_t$ están alineados a `tile_rows_align`.
  * Ejemplo: `ffn_down` con `n_out=4096`, tiles por filas: `[0..1023]`, `[1024..2047]`, `[2048..3071]`, `[3072..4095]`.

* **K por tile (v1)**:

  * K aplica al **residual Δ** dentro de ese tile (no al tensor completo).
  * K se interpreta como **presupuesto de sparsidad del residual** *en ese tile*.
  * Definición v1 (para evitar dobles interpretaciones):

    * **Si el residual es block-sparse 2D (scheme `block`)**:
      * Un **bloque** es una submatriz 2D de tamaño $B\times B$, donde $B$ viene del parámetro `--block` del scheme residual (v1: bloques cuadrados).
      * $K_t$ = máximo número de bloques 2D activos por **row-block** (grupo de $B$ filas) dentro del tile.
    * **Si el residual es COO-like por filas**: $K_t$ = máximo número de **entradas** activas por fila dentro del tile.
  * (Nota) No mezclar definiciones: el builder debe declarar explícitamente cuál aplica según `--scheme`.
  * (Nota) Si internamente se agrupan filas en `tile_rows_align`, debe ser una optimización; no cambia la semántica de $K_t$.
  * (Nota) $K_t$ es hard-cap; el builder puede usar menos si autotune/gating no justifica el costo.
  * Ejemplo numérico (para evitar interpretaciones):
    * Scheme `block` con `--block B=32`, `tile_rows=1024`:
      * el tile tiene $1024/32=32$ row-blocks.
      * si $K_t=4$, cada row-block puede tener como máximo 4 bloques $32\times 32$ activos (en distintas posiciones de columna).
    * Scheme COO-like por filas con `tile_rows=1024`:
      * si $K_t=64$, cada fila del tile puede tener como máximo 64 entradas activas.

* **Stage (multi-stage Δ)**:

  * Definición:
    $$
    \hat W = W_0 + \Delta_1 + \Delta_2 + \cdots + \Delta_m,\quad m\ \text{pequeño}
    $$
  * Formalmente:
    $$
    R_0 = W - W_0,\quad \Delta_s \approx R_{s-1},\quad R_s = R_{s-1} - \Delta_s
    $$
  * Restricción: $\sum_s K_s \le K_{total}$ (y/o budget de ops/mem).

---

## 0.1) Criterios de aceptación (KPIs) y comparabilidad

Sin KPIs por run, el backlog se convierte en wishlist. Para cada experimento/preset, reportar:

* **Calidad numérica (termómetro):**

  * PPL en Wikitext-2 (`wikitext-2-raw/wiki.test.raw`) con `ctx`/`chunks` fijos.
  * Reportar `ΔPPL` vs base y `PPL±std` si aplica.

* **Estabilidad funcional (gate final):**

  * Greedy pack con prompts duros: `calibration/greedy_zombie_pack.txt`.
  * Runner: `scripts/seeddelta-greedy-pack.sh` o `scripts/seeddelta-policy-eval.sh --greedy-pack ...`.
  * Criterio v1: `RESULT: PASS` (0 prompts flagged).
  * Además (telemetría mínima por prompt, aunque PASS/FAIL sea el gate):

    * `flag.loop` (repetición/atasco),
    * `flag.lang_drift` (cambio de idioma),
    * `flag.instruction_break` (rompe restricciones duras).
    * Nota: `flag.lang_drift` es telemetría (v2); no debe ser gate duro mientras el detector sea heurístico.

* **Perf (prompt y gen):**

  * `tok/s` en prompt eval y gen batch=1 (p.ej. `llama-cli -n 256`) y/o `llama-perplexity` prompt time.
  * Reportar `Δtok/s` vs base y “overhead por tensor activo” si se puede aislar.

* **Memoria (steady y pico):**

  * `Maximum resident set size` (`/usr/bin/time -v`) + `llama_memory_breakdown_print` (Host/CPU_REPACK/ctx/compute).
  * Distinguir pico (repack) vs steady (host model) explícitamente.

* **Stack-safety:**

  * `stack_budget`: cuántos tensores SeedΔ activos por tipo (gate/up/down) y total.
  * `stack_cost`: coste ponderado (ver sección 7), y su correlación con greedy/PPL.

* **Trazabilidad (requisito, sin ambigüedad):**
  * Todo run debe dejar un `report.json` (o equivalente) con:
    * `accepted/emit`, `strip`, `reject_reason`, `metric_used`, métricas y thresholds/targets por tensor.
    * si hay tiled-K: `tile_rows`, `tile_rows_align`, `K_levels`, `unique_k_count`, `tiles_rounded_count` y ratio.

### Esquema mínimo `report.json` (claves fijas)

Objetivo: que scripts (ablations, A/B, dashboards) no dependan de “parsear logs” ni de nombres variables.

Por tensor (un objeto por `blk.N/<tensor_name>`):

* `accepted` (bool), `emit` (bool), `strip` (bool)
* `reject_reason` (enum fijo):
  * `gating`, `ffn_block_score`, `stack_budget`, `stack_cost`, `perf_budget`, `k_custom_limit`
* `metric_used` (enum fijo): `cos_x_w` | `ffn_cos`
* `metrics` (obj): `cos_mean`, `cos_p05`, `ffn_cos_mean`, `ffn_cos_p05`, `norm_ratio_mean`, `ffn_log_norm_ratio_mean`, `ffn_l2_mean`, etc (solo las que apliquen)
* `thresholds` (obj): umbrales usados por gating
* `targets_used` (obj): `{ tau_mean, tau_p05 }` usados por `stack_cost` (según `metric_used`)
* `stack_cost_delta` (float), `stack_cost_total` (float)

Si tiled-K:

* `tile_rows` (int), `tile_rows_align` (int)
* `k_levels` (lista de int), `k_total_per_tensor` (int)
* `unique_k_count` (int), `tiles_rounded_count` (int), `tiles_rounded_pct` (float)
* `k_per_tile` (lista densa por tile, v1): K seleccionado final (post-round) por tile, en el mismo orden de tiles por filas

Si `k_custom`:

* `k_custom_used` (bool)
* `k_requested_stats` (obj): `{min,max,mean}` + histogram opcional
* `k_selected_stats` (obj): `{min,max,mean}` + histogram opcional

Metodología multi-precisión (para conclusiones limpias):

* Para aislar “source_precision”, comparar **el mismo modelo** generado desde la misma base:

  * HF base → GGUF F16 → cuantizar a Q8/Q4 desde ese mismo F16 → construir SeedΔ en cada uno.
  * Evitar inferencias fuertes comparando fine-tunes distintos (p.ej. Kimiko FP16 vs Mistral instruct Q8).

---

## 0.2) Orden sugerido (MVP) para no dispersarse

Prioridad por ROI y por reducción de ambigüedad:

1. **Gating FFN compuesto + stack_budget/stack_cost** (secciones 5 y 7): evitar zombies antes de tocar más capas.
2. **Comparabilidad multi-precisión** (sección 2): misma base F16 → Q8/Q4, y medir headroom real.
3. **K por tile con niveles discretos (`K_levels`)** (secciones 6.2 + 6.3): high ROI / low risk.
4. **Δ por etapas** (6.1): solo si cambia de familia y es fusionable offline (alto riesgo de runtime/metadata).
5. Optimización pesada de kernels (`W0x`) solo cuando sepamos qué presets valen la pena acelerar.

---

## 0.3) Gates de avance (para no optimizar a ciegas)

Estos gates convierten el TODO en ingeniería “shippeable”. Si un gate falla, **no** avanzar a la siguiente etapa;
primero corregir métricas/criterios.

* **Gate A — Score funcional confiable:**
  * El “FFN-score compuesto” (sección 5) debe correlacionar con `ΔPPL` (Wikitext) y con fallos del greedy pack.
  * Si no correlaciona, no vale la pena optimizar tiles, stages ni kernels (estaríamos optimizando un proxy roto).

* **Gate B — Stack-safety reduce cliffs:**
  * Al comprimir “más” (más tensores/capas), `stack_budget/stack_cost` debe reducir varianza de calidad y evitar
    cliff (modelos zombis) en greedy pack.
  * Si se siguen viendo cliffs con el mismo `stack_cost`, ajustar thresholds/cost (sección 7) o revisar si el score usado
    no está midiendo la composición (p.ej. debe ser FFN-score, no tensor-score).

* **Gate C — Tiled K mejora PPL/MB sin romper throughput:**
  * “K por tile con niveles discretos” (sección 6.2) debe mejorar `PPL/MB` (o bajar `ΔPPL` a igual memoria)
    y no degradar throughput más de X% (X depende del hardware; empezar con X≈10–20% en CPU).
  * Si mejora PPL pero mata tok/s, limitar niveles, ajustar tile_rows y/o alinear mejor al layout/kernel.

---

## 0.4) Ablations obligatorias (proceso repetible)

Las ablations convierten “intuición” en evidencia (y aceleran debugging cuando un modelo se vuelve zombie).

* [x] Añadir `scripts/seeddelta-ablate.sh`:
  * genera y evalúa ablations estándar (mismo texto/ctx/threads):
    * base vs SD (policy completa),
    * `down-only`,
    * `gate/up-only`,
    * “solo 1 capa” (isolar una capa de interés),
    * “bandas” (p.ej. 8–14, 15–20, 21–27).
  * corre `llama-perplexity` (Wikitext) + greedy pack, y guarda artefactos en un outdir único.
* [x] Emitir `ablate_summary.json` (y/o `ablate_summary.md`) con KPIs comparables:
  * PPL base/SD, PASS/FAIL greedy, `stack_budget/stack_cost`, y lista de tensores/capas activas por ablation.

---

## 1) Perf / RAM (sigue siendo el cuello)

Orden recomendado (antes de micro-optimizar):

* [ ] Clarificar repack vs RSS: tener medición consistente (time -v + logs) y explicar picos vs steady.
* [ ] Medir “costo por tensor activo” en runtime:
  - cuánto sube/baja prompt/gen tok/s y cuánto cambia RSS/CPU_REPACK por cada tensor SeedΔ activado.
* [ ] Instrumentar stack-safety: medir cuántos tensores pasan por policy vs cuántos terminan activos tras endurecer thresholds.
  - Reportar `build.stack_budget` y `build.stack_cost`.
* [ ] Reducir overhead de índices/layout (u16 contiguo, mejor packing por bloque).
* [ ] Optimizar `W0x` (cache por token/batch; vectorizar base; reducir overhead en base+residual).

---

## 2) Calidad / data-aware “de verdad”

* [ ] Mejorar dataset para imatrix por modelo (no depender de `gemma_calibration.txt` para todo).
* [ ] Añadir un escalón intermedio “data-aware barato” (v1.5) antes de guardar activaciones:
  * estadísticas en streaming (normas, varianza por canal, percentiles, clipping rates),
  * sin persistir `X` completo (reduce fricción/IO y aún mejora proxies vs imatrix diagonal).
* [ ] (Opcional) capturar activaciones reales por tensor (`X`) y optimizar un objetivo funcional bien definido:
  * offline: $\min \|W X - \hat W X\|_F^2$
  * por token/posición: $e_i=\|y_i-\hat y_i\|_2$ (reportar mean/p05/p95)
  * (ideal) hacerlo a nivel de bloque FFN compuesto cuando se trate de `gate/up` (sección 5).
* [ ] Explorar construir SeedΔ desde F16/Q8 como fuente (evitar “loss-on-loss” de Q4).
* [ ] Validar gating multi-precisión: repetir builder + policy en al menos tres precisiones (FP16, Q8, Q4)
  para el mismo modelo y documentar diferencias de `cos_x_w` / `stack_budget`.
* [ ] Añadir metadata `seeddelta.source_precision` o similar al GGUF/export para rastrear con qué
  precisión se generó SeedΔ (p.ej. `fp16`, `q8_0`, `q4_k_m`).

---

## 3) Expandir targets más allá de FFN (Fase 5)

Orden de ataque recomendado (minimizar riesgo):

1) **FFN** (ya en curso): `ffn_down` suele ser más tolerante; `gate/up` requieren FFN-score.

2) **Attention “menos frágil” primero:**

* [ ] `W_o` (output projection) con budgets conservadores + gating estricto.
* [ ] `W_v` (value) si el gating lo permite.
* [ ] `W_q`/`W_k` solo con umbrales aún más estrictos (y preferiblemente con score compuesto de atención).

3) **Embeddings / LM head:**

* [ ] Solo con gating ultra estricto y stack-budget mínimo (riesgo alto de degradar vocab/logits).

---

## 4) Modelos y pruebas cruzadas

* [ ] Smoke tests obligatorios en modelos pequeños (Gemma 1B/4B) cada vez que se toque policy/gating
  para detectar regresiones temprano (PPL+greedy pack).
* [ ] Mantener un set “intermedio” llama-like (Mistral/Llama 7–8B) para validar stack-safety con más headroom que Qwen.
* [ ] Documentar por modelo: “headroom apilable” observado (cuántos tensores/capas sobreviven greedy + `ΔPPL` aceptable).

---

## 5) Gating y error a nivel bloque FFN (no solo tensor)

Motivación empírica (Qwen 7B Q4, Mistral 7B Q8, Gemma 4B Q4):

* Tensores individuales pueden “pasar” gating (`cos_x_w` decente) pero el bloque completo FFN (`gate+up+down`)
  rompe comportamiento greedy al apilar varias capas (loops, drift de idioma, fallos en instrucciones duras).
* En SwiGLU/SILU el error en `gate`/`up` es multiplicativo y selectivo: cambia qué activaciones “se abren”
  antes de la no-linealidad, y luego ese cambio se amplifica al combinarse y proyectarse con `down`.

Acciones:

* [ ] Definir un score de calidad **a nivel FFN compuesto** (medible y reproducible):

  * Definición del bloque:
    $$
    \mathrm{FFN}(x)=W_{down}\big(\mathrm{silu}(W_{gate}x)\odot (W_{up}x)\big)
    $$
    $$
    \widehat{\mathrm{FFN}}(x)=\hat W_{down}\big(\mathrm{silu}(\hat W_{gate}x)\odot (\hat W_{up}x)\big)
    $$
  * Score primario (por token/posición (i), y agregación sobre dataset):
    $$
    c_i=\cos(y_i,\hat y_i)=\frac{\langle y_i,\hat y_i\rangle}{\|y_i\|_2\,\|\hat y_i\|_2+\epsilon}
    $$
    con (y_i=\mathrm{FFN}(x_i)), (\hat y_i=\widehat{\mathrm{FFN}}(x_i)).
    Reportar: `ffn_cos_mean = mean_i(c_i)` y `ffn_cos_p05 = p05_i(c_i)`.
  * Score secundario (drift de escala):
    $$
    r_i=\frac{\|\hat y_i\|_2}{\|y_i\|_2+\epsilon}
    $$
    Reportar: `ffn_norm_ratio_mean` + `p05/p95`.
    Además, para budgets acumulativos, reportar log-ratio:
    $$
    \ell_i=\log(r_i)
    $$
    Reportar: `ffn_log_norm_ratio_mean = mean_i(\ell_i)` + `p05/p95`.
  * Error absoluto (magnitud):
    $$
    e_i=\|y_i-\hat y_i\|_2
    $$
    Reportar: `ffn_l2_mean = mean_i(e_i)` + `p05/p95` (alternativa: `ffn_mse_mean` si se normaliza).
  * Dataset/activaciones: usar activaciones reales $x_i$ **a la entrada del FFN** (post-norm del bloque, pre matmul `gate/up`)
    (mismo texto/calib usado para imatrix o un pack representativo).
  * Decisión v1:

    * permitir tocar `gate/up` **solo si** el **FFN-score** pasa (no basta con `cos_x_w` por tensor).
    * `down` puede seguir con gating por tensor mientras FFN-score no esté disponible (pero migrar a FFN-score cuando exista).

* [ ] Explorar políticas “down-only por defecto”:

  * habilitar SeedΔ en `ffn_down` más agresivamente,
  * requerir thresholds más altos o FFN-score compuesto para permitir SeedΔ en `gate/up`.

---

## 6) Δ por etapas y K por subcapa/bloque

### 6.1) Δ por etapas (multi-stage)

Idea: aproximar el residual en varias etapas pequeñas:

* $\hat W = W_0 + \Delta_1 + \Delta_2 + \cdots + \Delta_m$
* Análogo a matching pursuit/OMP: el primer top-K es miope; stages sucesivos pueden capturar estructura que el primer corte no vio.

Acciones:

* [ ] Diseñar un esquema de builder “multi-stage”:

  * número de stages (m) pequeño (p.ej. 2–3), con K decreciente o constante.
  * evaluar por stage: $\hat W^{(s)} = W_0+\sum_{j=1}^s \Delta_j$ y medir el score funcional (FFN-score o $\|W X-\hat W X\|_F^2$).
  * gating y report JSON por stage (para ver ganancia marginal de cada (\Delta_s)).

* [ ] Definir condición de stop (no “stages infinitos”):

  * permitir `stage+1` solo si mejora el score funcional > X% o reduce error > Y por costo adicional.
  * si no hay ganancia marginal clara, no agregar stages (evita matar perf/complexidad).

* [ ] Multi-stage permitido solo si cambia de “familia” (evitar 2× lo mismo):
  * Stage 1: residual sparse estándar (lo actual).
  * Stage 2 (opcional): otra familia (ejemplos):
    * corrección per-channel (scale/bias) barata,
    * lista de outliers (top-m magnitudes),
    * low-rank pequeño.
  * Preferir fusión offline a una sola representación si es posible (evitar dos streams dispersos en runtime).

* [ ] Definir un presupuesto total de compute/memoria por tensor:

  * limitar (\sum_s K_s) y el número de stages en función de la ganancia observada.

### 6.2) K por subcapa/bloque (tiles)

Idea: dividir la matriz en tiles y asignar K distinto según “importancia/dificultad”:

* tiles alineados con `tile_rows_align` (32/64) y con el layout de cache.
* pocos niveles de K (p.ej. {256, 512, 1024, 1536}) para evitar explosión de combinatoria.

Acciones:

* [ ] Definir una factorización por tiles (v1: filas) y medir curva “error vs K” por tile.
* [ ] Formular un “knapsack suave” implementable (greedy allocate):

  * definir utilidad por tile:

    * ideal: $E_t(K)=\|W_t X-\hat W_t(K) X\|_F^2$ o score funcional equivalente,
    * proxy: error ponderado por imatrix/diagonal si no hay (X).
  * medir “beneficio por +ΔK”:

    * $\Delta E_t = E_t(K_{cur}) - E_t(K_{next})$,
    * asignar K por niveles hasta consumir presupuesto total (sin solver exacto).
* [ ] Exponer en policy una forma simple:

  * `K_levels` globales + reglas por rangos de tiles (inferior/media/superior) o por “top-N tiles” según importancia.
  * **Selector TT-cross/maxvol (idea del TODO N6, explícito):**
    * Muestrear solo 3–6 tiles por capa (fibras representativas) antes de decidir K.
    * Calcular features de importancia por K (o por tile): `proxy_cos_p50/p05`, `residual_L2`, `sparsity`, `stack_cost_norm` (todas ya presentes en report/metrics).
    * Elegir {K1,(K2)} vía maxvol aproximado: K1 = argmax‖f‖; K2 = argmax‖f‖·(1–cos²(f,fK1)) para evitar colinealidad (rank-2 opcional).
    * Asignar K a cada tile por afinidad (similitud de features) con {K1,(K2)}; fallback al ciclo uniforme actual para comparabilidad.
    * Métricas en report: `kselector.rank`, `kselector.gap_vs_uniform`, `kselector.tiles_sampled`.
    * (Opcional) Autotuner seguro (bandit) que alterna selector `uniform` vs `ttcross` y rank 1/2 con recompensa `-quality_gap/lat_ms`, con hysteresis y rollback a `uniform` si empeora.

Plan realista (recuperar greedy/PPL sin humo):

* [x] Baseline controlado con Gemma 4B Q4: mismo build (base+row_scale), `greedy_pack` + `ppl` y report guardado.
* [x] Ablation por subcapa (mismo K/bloque): `down-only` vs `gate/up-only` para localizar el culpable.
* [x] Ajuste conservador: `K_down` alto + `K_gate/K_up` bajo, y comparar `block=16` vs `block=32`.
* [x] Comparar `scheme=block` vs `scheme=coo` en el caso ganador de la ablation.
* [x] Repetir el mejor preset con fuente **Q8 y F16** (misma familia de modelo) para medir “loss-on-loss”.

---

### 6.3) Flags e interfaz `K_levels` (MVP) vs `K_custom` (experimental, builder-only)

Objetivo: evitar 2 runtimes (levels vs custom) mientras mantenemos control fino y reproducible.

#### Flags (v1)

Tiles:

* `--k-tiles-enable 0|1`
* `--k-tile-rows 1024` (clamp: `tile_rows = min(tile_rows, n_out)`)
* `--k-tile-rows-align 32|64` (clamp: `tile_rows >= tile_rows_align`)
* `--k-levels 256,512,1024,1536` (2–4 niveles)
* `--k-levels-mode strict` (runtime SOLO soporta estos niveles; no hay kernel genérico para K arbitrario)

Budget:

* `--k-total-per-tensor N` (hard-cap; misma unidad que `K_t`)
  * Semántica v1: se aplica como cap por tile (p.ej. `K_selected = min(K_selected, N)`), no como sumatoria global.

`K_custom` (builder-only, v1 fijo a round):

* `--allow-k-custom 0|1`
* `--k-custom-file path/to/k_custom.yaml`
* `--k-custom-mode round`
* `--k-custom-rounding nearest` (empates hacia arriba)
* `--k-custom-max-unique 8`
* `--k-custom-max-tiles-pct 10`
* `--k-custom-warn 1` (warning agregado por tensor, no por tile)

**Unidades (importante):**

* scheme `block`: `K_t` y `k-total-per-tensor` están en “# bloques 2D activos por row-block (B filas)”.
* scheme `coo`: están en “# entradas activas por fila”.

#### Formato mínimo `k_custom.yaml` (v1)

```yaml
version: 1
defaults:
  max_unique_k: 8
  max_tiles_pct: 10
rules:
  - name: boost_down_mid
    priority: 100
    match:
      layers: "8..23"
      tensor: "ffn_down"
      tiles:
        row_ranges:
          - [0, 1024]
          - [2048, 3072]
    set:
      k: 768

  - name: gate_force_level
    priority: 90
    match:
      layers: "0..31"
      tensor: "ffn_gate"
    set:
      k_level: 3
```

Semántica v1:

* `set.k` se redondea a `K_levels` (round-only). El report deja `k_requested` y `k_selected`.
* `set.k_level` siempre fast-path (índice 0-based en `K_levels`).
* Rangos:
  * `match.layers: "L..R"` es inclusivo (0-based) y debe referir a `blk.N` tal como aparece en nombres GGUF.
  * `row_ranges: [row_lo, row_hi)` es half-open y está en índices de fila de $W$ (dimensión $n_{out}$).
    * Recomendación v1: los límites deben coincidir con bordes de tile (`tile_rows`) para mantener trazabilidad simple.
* Límites y política determinista (v1):
  * `max_tiles_pct` limita el % de tiles con override (`k_custom_used`) por tensor.
  * `max_unique_k` limita `unique_k_count` (post-round) y debe ser `<= len(k_levels)` (con 2–4 niveles suele ser redundante, pero deja el contrato listo).
  * Si se excede un límite: aplicar overrides en orden de `priority` (mayor primero) y descartar el resto hasta cumplir; registrar `reject_reason = k_custom_limit` + contadores (`tiles_dropped_count`, `tiles_dropped_pct`).

---

## 7) Control global de stack-safety (presupuesto de error acumulado)

Razonamiento:

* El error no se suma linealmente, se **encadena** a través de capas y no-linealidades.
* Un gating por tensor puede aceptar muchos deltas “borderline” que, al apilarse, degradan PPL y greedy.

Acciones:

* [ ] Definir un “presupuesto de stack” global por modelo/config:

  * máximo número de tensores SeedΔ activos por tipo (`gate/up/down`) y por bloque (FFN/attn),
  * reglas más estrictas (umbrales más altos) cuando el stack_budget potencial crece.

* [ ] Hacer el budget más inteligente con un “coste ponderado” (no solo conteo):

  * intuición: tensores borderline (cola p05 baja) consumen más presupuesto que tensores muy buenos.
  * definir un coste v1 (clamp + estable):

    * clamp: `cos := clamp(cos, -1, 1)`
    * $$
      cost = \max\big(0,\ \alpha(\tau_{mean}-\bar c) + \beta(\tau_{p05}-c_{p05})\big)
      $$
    * Definición de métricas para `\bar c` y `c_{p05}`:
      * mientras no exista FFN-score: usar `cos_mean_x_w` y `cos_p05_x_w` del tensor.
      * cuando exista FFN-score: para `gate/up` usar `ffn_cos_mean` y `ffn_cos_p05` (sección 5).
    * Targets `\tau_*` separados por métrica (no mezclar escalas):
      * `tau_tensor_mean/tau_tensor_p05` para `cos_x_w_*` (proxy por tensor).
      * `tau_ffn_mean/tau_ffn_p05` para `ffn_cos_*` (bloque compuesto).
      * En `report.json`, por tensor: `metric_used` ∈ `{cos_x_w, ffn_cos}` y `targets_used = {tau_mean,tau_p05}`.
    * Definición de targets (para que tensores “muy buenos” no paguen coste):
      * v1 (configurable, por tipo): targets más altos para `gate/up` que para `down` dentro de cada set (tensor vs ffn).
      * Ejemplo inicial (solo ilustrativo): para `cos_x_w_*` en CPU: `gate/up`: `\tau_{mean}=0.70`, `\tau_{p05}=0.55`; `down`: `\tau_{mean}=0.60`, `\tau_{p05}=0.45`.
    * Agregación v1 (sin ambigüedad):
      * `stack_cost_total = Σ_tensors cost(tensor)` (solo tensores SeedΔ emitidos/activos).
      * v2: permitir “agregación por bloque” (FFN-score) cuando exista telemetría compuesta estable.
    * (opcional v2) penalizar drift de escala si está disponible (preferir log-ratio):
      $$
      cost = cost + \gamma\,|\bar \ell|
      $$
      donde $\bar \ell$ es `ffn_log_norm_ratio_mean` (y si no existe, aproximar con `log(norm_ratio_mean)`).
  * permitir más tensores si son “muy buenos” y cortar rápido si son borderline.

* [ ] Integrar stack-safety en policy:

  * campos tipo `stack_budget.max_tensors`, `stack_budget.max_gate_up`, etc.
  * decisiones claras en report JSON cuando un tensor no se activa por exceder presupuesto global.
* [ ] Convertir stack-safety en regla explícita (no solo telemetría):
  * hard-cap o soft-cap por run (p.ej. “si stack_cost supera T, no emitir más”),
  * registrar el motivo de rechazo (budget) en el report.

### Notas de pruebas

* 2025-12-21: el greedy pack fallaba porque el script evaluaba stderr/logs junto con stdout (base vs base daba 11/20). Fix: capturar stdout, guardar stderr aparte, y recortar respuesta (después del prompt, antes de líneas `llama_`). Ahora base vs base pasa 0/20.
