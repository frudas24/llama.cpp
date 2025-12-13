# TODO — StateCells / Diccionario k‑esparso para comprimir pesos LLM

> **Objetivo:** adaptar tu tecnología “StateCells + SparseAE (Oja + k‑WTA)” para LLMs en llama.cpp: representar matrices grandes como **combinaciones k‑esparsas de un diccionario** aprendido. Meta: **reducir RAM de pesos** de forma más agresiva que Q‑quants, manteniendo calidad con fallback seguro.

---

## Índice
0) Contexto (Devstral 24B, RAM/KV)  
1) Idea central  
2) Matemática esencial  
3) Config/flags  
4) Integración (offline + runtime)  
5) Pseudocódigo  
6) Métricas/aceptación  
7) Roadmap  
8) Riesgos/mitigación  
9) Commits sugeridos
10) Research/prior art (para “hacerlo bien”)

---

## Estado actual (✅ implementado en este repo)
- **Runtime (llama-cli/server):**
  - Flags: `--statecells` y `--statecells-gap` (global).
  - Loader opcional de `blk.N.ffn_{gate,up,down}.{dict,codes,vals,row_scale}` desde GGUF.
  - Hook en grafo: intercepta matmuls de FFN vía `build_lora_mm`.
  - Kernel v1 (rápido): `p = Dᵀx` con `ggml_mul_mat` + custom op “gather/sum k‑esparso” sobre `codes` (+ `vals` y/o `row_scale` opcionales).
- **Offline:**
  - Tool `llama-statecells-build` genera `*.dict` + `*.codes` (+ `*.row_scale`) y escribe `out.gguf` (✅ progreso detallado por layer/iter/sample/encoding; **no** abre “interfaz” tipo `llama-cli`).
  - Encoding acelerado (✅): calcula `Y = Dᵀ·Wblk` en bloques (`B=64`) usando `ggml_mul_mat` + threadpool CPU (reusa threads; evita “1 dot por columna”).
  - Coeficientes por code `*.vals` (✅ opcional): `--vals` genera `blk.N.*.vals` (fp16) para mejorar calidad.
  - Builder “data-aware” v1 (✅): `--imatrix FILE` pondera entrenamiento/encoding con imatrix GGUF (aprox diagonal). Reporta métricas ponderadas `*_w` en `--eval-cols/--report-json`.
  - Iteración rápida: `--resume` (reanuda sobre `-o` existente), `--checkpoint-every N` (checkpoint por capas), `--eval-cols/--report-json` (gap report).
- **Pendiente inmediato (para “velocidad de la luz”):**
  - Calibración data‑aware “real” (minimizar `||W X − Ŵ X||` con activaciones reales), gating offline por capa/tensor (sin fallback caliente) y kernels más vectorizados.

---

## 0) Contexto (Devstral‑Small‑2 24B, 32 GB RAM)
**Geometría (aprox):** `n_layers=40`, `n_embd=5120`, `n_ff=32768`, `n_heads=40`, `n_kv_heads=8`, `head_dim=128`.

**Por qué FFN/MLP primero (ROI):** los pesos FFN (gate+up+down) dominan el total de parámetros; si StateCells funciona “bien” en FFN, el impacto en RAM es masivo.

**Trampa real de RAM en contexto largo (KV cache):**
- Aunque bajemos mucho **weights**, en contextos grandes la RAM se la come KV.
- Para Devstral, el KV en FP16 crece ~lineal por token; con `ctx` grande puede romper 32 GB.
- En llama.cpp hay mitigación práctica vía **KV cache quantization**: `-ctk` / `-ctv` (p.ej. `q8_0`).

**Implicación de diseño:** StateCells debe apuntar a **weights**; para “resolver RAM de una vez por todas” hay que reportar y tunear **weights + KV** juntos (no solo weights).

### 1) Idea central
- En lugar de guardar W densa/quant completa, se aprende un **diccionario** `D ∈ R^{d×M}` (átomos) y cada fila/bloque de W se codifica como un **SparseCode** con `k` átomos activos:
  - `W_row ≈ Σ_{j∈code(row)} s_j · D[:,j]`, con `s_j ∈ {‑1,0,+1}` o `fp16` pequeño.
- El diccionario se aprende con **Oja** (normaliza y evita explosión) + opcional **anti‑Hebb lateral** (decorrelación), tal como en `TODO_all_in_neuron_statecells_journal_boot.md`.
- Resultado: memoria ≈ `|D| + |codes|` donde `|codes| ≪ |W|`.

### 2) Matemática esencial
- Codificación por matching pursuit / k‑WTA:
  1. Para una fila `w`, computa correlaciones `y = Dᵀ w`.
  2. Selecciona top‑k índices `I = TopK(|y|, k)` (k‑WTA).
  3. Coeficientes `s_i = sign(y_i)` (o `s_i = y_i` cuantizado).
  4. Reconstrucción `ŵ = Σ_{i∈I} s_i D[:,i]`.
- Aprendizaje diccionario (Oja por átomo):
  \[
  \Delta d_j = \eta\, y_j (w - y_j d_j)
  \]
  con normalización/clipping; anti‑Hebb opcional para reducir colinealidad.

### 3) Config/flags
- Offline:
  - Tool `llama-statecells-build -i in.gguf -o out.gguf` con flags:
    - `--dict-M` tamaño diccionario por peso (p.ej. 512–4096).
    - `--dict-M-gate`, `--dict-M-up`, `--dict-M-down` overrides por matriz (Devstral: `down` con `M` bajo).
    - `--dict-k` átomos activos por columna/fila (p.ej. 16–48).
    - `--dict-k-gate`, `--dict-k-up`, `--dict-k-down` overrides por matriz.
    - `--dict-eta`, `--dict-iters`, `--dict-max-samples`, `--layers A-B`, `--dict-type f16|f32`.
    - `-t, --threads N` threads CPU para encoding/row_scale (el encoding usa `ggml_mul_mat` en bloques, no 1 dot por columna).
    - `--vals` / `--no-vals` escribe `..._vals` (coeficientes fp16 por code).
    - `--resume` reanuda sobre un `-o` ya creado (salta pesos con `dict+codes`).
    - `--checkpoint-every N` escribe `-o` cada N capas (I/O grande; útil para no perder progreso).
    - `--row-scale` / `--no-row-scale` emite `..._row_scale` (fp16) por salida (default: on).
    - `--eval-cols N` calcula métricas de gap sobre N salidas por peso (sampling).
    - `--report-json FILE` exporta métricas por peso/capa a JSON.
    - `--imatrix FILE` pondera entrenamiento/encoding con un imatrix GGUF (`llama-imatrix`) para mejorar calidad “funcional” (aprox diagonal).
    - `--imatrix-eps F` clamp mínimo antes de `sqrt` (evita ceros/NaNs).
    - `--imatrix-power F` exponente de la ponderación (1.0 = default).
    - (futuro) `--scheme sign|sign+row_scale|fp16` para controlar coeficientes de forma explícita.
    - (futuro) `--calib-*` para data‑aware “real” con activaciones (ver sección 4).
  - Esquemas:
    - `sign` (actual): coeficiente implícito ±1 dentro de `codes` I16.
    - `sign+row_scale` (✅ implementado): `codes` ±1 + escala por salida (`..._row_scale` fp16) para recuperar magnitud “casi gratis”.
    - `sign+vals` (✅ implementado, opcional): tensor adicional `..._vals` con coeficientes fp16 por code (más calidad, más RAM).
    - `fp16` (futuro): vals fp16 “full” + calibración data‑aware (CoSpaDi‑style).
- Runtime:
  - `--statecells` habilita backend si el GGUF trae dict/codes.
  - `--statecells-gap 0.02` tolerancia de fallback (por ahora global).
  - (futuro) flags per‑layer/target cuando extendamos más allá de FFN.
  - Recomendado para Devstral en 32 GB con ctx grande: usar también `-ctk`/`-ctv` (KV quant) porque StateCells no reduce KV.

### 4) Integración
**4.1 Offline (compresión)**
- Nuevo tool que:
  1) lee pesos GGUF (por capa/bloque),
  2) entrena `D` por Oja (streaming sobre filas),
  3) codifica cada fila/bloque en SparseCode top‑k,
  4) escribe un GGUF nuevo con tensores:
     - `blk.N.ffn_gate.dict`  `[n_embd, M]` tipo F16/F32  
     - `blk.N.ffn_gate.codes` `[k, n_ff]`   tipo I16 con signo embebido  
     - `blk.N.ffn_up.dict`    `[n_embd, M]` tipo F16/F32  
     - `blk.N.ffn_up.codes`   `[k, n_ff]`   tipo I16  
     - `blk.N.ffn_down.dict`  `[n_ff,  M]`  tipo F16/F32  
     - `blk.N.ffn_down.codes` `[k, n_embd]` tipo I16  
     - (✅ opcional) `blk.N.*.vals` `[k, d_out]` fp16 para esquema `sign+vals`.

  **Interpretación codes (sign‑scheme):**
  - `codes[t, r] = ±(atom+1)` con `atom∈[0,M)`; `0` = slot vacío.

  **Metadatos GGUF (modelo):**
  - `statecells.enabled` bool
  - `statecells.dict.M`, `statecells.dict.k`, `statecells.dict.eta`, `statecells.dict.iters`
  - (futuro) `statecells.scheme` (0 sign, 1 fp16), `statecells.version`, `statecells.default_gap_tol`.

  **Nota de disco/RAM:** el GGUF resultante puede conservar pesos densos para compatibilidad; con `mmap` no deberían cargar en RAM si el runtime usa StateCells.

**4.1.1 Builder “data‑aware” (CoSpaDi‑style) — prioridad para calidad**
- Riesgo principal del builder weight‑only: minimizar `||W − Ŵ||` no garantiza minimizar el error funcional de la capa.
- Objetivo recomendado para preservar calidad: minimizar `||W X − Ŵ X||` con un `X` pequeño de **activaciones reales** (calibration set).
- Plan incremental (sin rehacer todo):
  1) **Recolectar activaciones** `X` por capa/peso objetivo (`ffn_gate/up/down`) con un dataset corto (p.ej. Wikitext2 o prompts QA).
  2) En `statecells-build`, en vez de entrenar el diccionario con filas `w`, entrenar con el objetivo funcional:
     - usar `y = (W X)` como “target” y `ŷ = (Ŵ X)` como predicción,
     - ajustar `D` y/o coeficientes para reducir `||y − ŷ||`.
  3) Mantener `--calib-max-samples` pequeño (2k–8k) para iterar rápido.

**4.1.2 Schedule por matriz (M/k no global) — prioridad para RAM+speed**
- En Devstral:
  - `ffn_gate/up`: `d_in=5120`, `d_out=32768` → toleran `M` más alto.
  - `ffn_down`: `d_in=32768`, `d_out=5120` → `M` debe ser **mucho menor** (si no, el `Dᵀx` cuesta demasiado y no ahorra RAM).
- Heurística inicial (por tensor): `M ≈ d_out/8` (redondear a potencia de 2).
  - gate/up: `M≈4096`, `k=16..32`
  - down: `M≈512..640`, `k=16..32`
- Builder debe permitir overrides por peso (`--dict-M-down`, etc) y reportar memoria estimada por tensor.

**4.1.3 Coeficientes sin inflar RAM — prioridad para calidad**
- `sign` puro es agresivo (puede subir PPL y empuja a aumentar `M/k`).
- Antes de `..._vals` completo, implementar:
  - `..._row_scale` (fp16) por fila (o por bloque de filas) y mantener `codes` ±1.
  - Esto mejora magnitud con overhead mínimo (≈2 bytes por fila).

**4.2 Runtime (inferencia)**
Dos rutas:
1. **Sin reconstruir W:**  
   - Para input `x`, computa proyecciones base `p_j = dot(D[:,j], x)` para todos `j=1..M`.  
   - Para cada fila `r`: `y_r = Σ_{(j,s)∈code(r)} s · p_j`.  
   - Complejidad ≈ `O(M·d + k·rows)` (si `M ≪ rows`, ahorro grande).
   - (futuro) **reuso de `p`**: si `ffn_gate` y `ffn_up` comparten diccionario `D`, se calcula `p = Dᵀx` una vez y se usa para ambos (ahorro directo).
2. **Reconstrucción bajo demanda + cache:**  
   - Reconstruye bloques `ŵ_block` solo cuando se usan, con LRU en RAM.  
   - Útil si además hay sparsidad KWTA/event‑driven que evita usar todos los bloques.

### 5) Pseudocódigo
```cpp
// OFFLINE: aprender diccionario y codes
init D[d][M] random small
for iter in 1..I:
  for each row w in W:
    y = D^T w
    I = topk(|y|, k)
    // Oja update solo en activos
    for j in I:
      D[:,j] += eta * y[j] * (w - y[j]*D[:,j])
    normalize_columns(D)
// codificar
for each row w:
  y = D^T w
  I = topk(|y|, k)
  code_idx[row]=I
  code_sign[row]=sign(y[I])

// RUNTIME sin reconstrucción
p = zeros(M)
for j in 0..M-1:
  p[j] = dot(D[:,j], x)
for row in 0..R-1:
  y[row]=0
  for t in 0..k-1:
    j = code_idx[row][t]
    y[row] += code_sign[row][t] * p[j]
```

### 6) Métricas/aceptación
- `statecells.enabled`, `dict.M`, `dict.k`, `statecells.mem_saved_gb`, `statecells.speedup`, `statecells.gap_ppl`.
- Aceptar si:
  - Memoria de pesos ↓ ≥30–50% vs Q5_K_M (misma capa objetivo),
  - Speedup CPU ≥10–20% en prompt+gen,
  - Δppl ≤2% (o menos con scheme=fp16).
  - Para Devstral en 32 GB: reportar también **KV cache RAM** (según `-ctk/-ctv` y `ctx`), porque puede dominar el total.

### 7) Roadmap
1. ✅ Tool offline `llama-statecells-build` (leer GGUF, entrenar D, emitir `*.dict/*.codes`, escribir GGUF).  
2. ✅ Formato GGUF extendido + loader/hook runtime opt‑in en llama.cpp (`--statecells`).  
3. ✅ Kernel runtime rápido “sin reconstrucción” (precompute `p = Dᵀx`, sumar `k` coeficientes por salida).  
4. ✅ Schedule por matriz (`M/k` por `gate/up/down`).  
5. ✅ Scheme `sign+row_scale` (`...row_scale` fp16).  
6. ✅ Scheme `sign+vals` (`*.vals`) (opcional vía `--vals`).  
7. ✅ Builder data‑aware v1 con imatrix (diagonal) (`--imatrix`) + métricas `*_w`.  
8. ⏳ Builder data‑aware v2 (objetivo funcional real `||W X − Ŵ X||` con activaciones) + auto‑tune por capa/tensor.  
9. ⏳ Decisión **offline** de capas/tensores (emitir StateCells solo si pasa gap); runtime solo ejecuta lo que el GGUF trae (sin “comparar baseline” en caliente).  
10. ⏳ Integración con KWTA/event‑driven para explotar máscaras y cachear trabajo.  
11. ⏳ Bench `llama-bench` + `llama-perplexity` A/B vs Q4/Q5 (script: `scripts/statecells-eval.sh`).

### 8) Riesgos/mitigación
- **M demasiado grande** → no hay ahorro: auto‑tune M por capa (target M≈rows/8).  
- **k muy bajo** → sube ppl: empezar con k=16–32 y bajar gradualmente.  
- **Costo p_j** puede dominar: vectorizar dot(D,x) y reusar para varias proyecciones en la capa.  
- **No todas las capas separables**: permitir fallback per‑layer con `gap_tol`.
- **Tool offline lento** en pesos grandes: limitar `--dict-max-samples`, restringir `--layers`, y preferir correr el builder en Ubuntu real con suficiente RAM (evitar swap/WSL).
- **Sign‑only puede ser muy restrictivo**: priorizar `row_scale` o `vals` antes de aumentar `M` sin control.
- **KV cache domina en ctx largo**: mitigar con `-ctk/-ctv` y medir el total (weights+KV) para no optimizar “la mitad equivocada”.

### Compatibilidad e interacciones
- **Backend de pesos exclusivo por capa:** no combinar StateCells con ternario/TT/proc‑sparse en la misma matriz en primera iteración.  
- Sinergia con KWTA/event‑driven: usando la ruta “reconstrucción bajo demanda”, las máscaras reducen reconstrucciones y aumentan el ahorro real.  
- Nan‑Guardian debe cubrir reescalados/renorm y la salida reconstruida antes de mezclarse con staleness.

### 9) Commits sugeridos
- `statecells: add offline dict+code builder for GGUF`  
- `gguf: add dict/codes tensor types + metadata`  
- `kernels: add statecells matvec path (mlp)`  
- `runtime: flags + gap fallback + metrics`

---

## 10) Research/prior art (para “hacerlo bien”)
- Esto cae en la familia **Sparse Dictionary Learning / Sparse Coding** (K‑SVD/OMP) y variantes recientes para LLMs.
- Dos ideas “robables” que encajan perfecto aquí:
  - **Calibración funcional (CoSpaDi‑style):** optimizar `||W X − Ŵ X||` con activaciones reales para preservar PPL/QA.
  - **Kernels tipo codebook (AQLM‑style):** el valor no está solo en el formato; está en el **micro‑kernel cache‑friendly** (y en decidir `M/k` por tensor).
- Conclusión: la integración actual es la base correcta; para ganar contra Q‑quants en serio, el siguiente salto es **data‑aware builder + coeficientes baratos (row_scale) + schedule por tensor**.

---

## 11) Ejemplos de uso (Devstral 24B Q5 → StateCells)

### 11.1 Construir `devstral_sc.gguf` (quick smoke → expandir)

> Nota: esto **no** abre una UI; es un tool offline. Con el encoding acelerado deberías ver `(... thr, B=64)` y tiempos de **segundos** en `encoding codes` (si no lo ves, estás usando un binario viejo o sin rebuild).

```bash
IN="/home/frudas/.cache/llama.cpp/bartowski_mistralai_Devstral-Small-2-24B-Instruct-2512-GGUF_mistralai_Devstral-Small-2-24B-Instruct-2512-Q5_K_M.gguf"
OUT="/home/frudas/.cache/llama.cpp/devstral_sc.gguf"

# 1) quick smoke (capas 6-8) para validar rápido
./llama.cpp/build/bin/llama-statecells-build \
  -i "$IN" \
  -o "$OUT" \
  -t 16 \
  --layers 6-8 \
  --dict-M-gate 4096 --dict-M-up 4096 --dict-M-down 512 \
  --dict-k 32 \
  --dict-iters 2 \
  --dict-max-samples 2048 \
  --vals \
  --row-scale \
  --eval-cols 64 \
  --report-json /tmp/devstral_sc_6-8.json \
  --checkpoint-every 1

# 2) expandir a capas medias (6-25)
./llama.cpp/build/bin/llama-statecells-build \
  -i "$IN" \
  -o "$OUT" \
  --resume \
  -t 16 \
  --layers 6-25 \
  --dict-M-gate 4096 --dict-M-up 4096 --dict-M-down 512 \
  --dict-k 32 \
  --dict-iters 2 \
  --dict-max-samples 2048 \
  --vals \
  --row-scale \
  --eval-cols 64 \
  --report-json /tmp/devstral_sc_6-25.json \
  --checkpoint-every 1
```

### 11.2 Reanudar/continuar (sin perder horas)

Si `OUT` ya existe y quieres seguir agregando capas (o completar `row_scale` si faltaba), usa `--resume`:

```bash
IN="/home/frudas/.cache/llama.cpp/bartowski_mistralai_Devstral-Small-2-24B-Instruct-2512-GGUF_mistralai_Devstral-Small-2-24B-Instruct-2512-Q5_K_M.gguf"
OUT="/home/frudas/.cache/llama.cpp/devstral_sc.gguf"

./llama.cpp/build/bin/llama-statecells-build \
  -i "$IN" \
  -o "$OUT" \
  --resume \
  -t 16 \
  --layers 26-39 \
  --dict-M-gate 4096 --dict-M-up 4096 --dict-M-down 512 \
  --dict-k 32 --dict-iters 2 \
  --vals \
  --row-scale \
  --eval-cols 64 --report-json /tmp/devstral_sc_26-39.json \
  --checkpoint-every 1
```

### 11.3 Ejecutar con `llama-cli` (StateCells + KV quant)

```bash
OUT="/home/frudas/.cache/llama.cpp/devstral_sc.gguf"

./llama.cpp/build/bin/llama-cli \
  -m "$OUT" \
  --statecells --statecells-gap 0.02 \
  -t 16 -tb 16 -c 4096 -fa on \
  -ctk q8_0 -ctv q8_0 \
  -p "hola" -n 128 \
  --single-turn < /dev/null
```

### 11.4 Evaluación rápida A/B (perplexity + QA)

```bash
IN="/home/frudas/.cache/llama.cpp/bartowski_mistralai_Devstral-Small-2-24B-Instruct-2512-GGUF_mistralai_Devstral-Small-2-24B-Instruct-2512-Q5_K_M.gguf"
OUT="/home/frudas/.cache/llama.cpp/devstral_sc.gguf"

./llama.cpp/scripts/statecells-eval.sh \
  --base "$IN" \
  --sc "$OUT" \
  --text /ruta/a/wikitext-2-raw/wiki.test.raw \
  --threads 16 --ctx 4096
```

### 11.5 (Recomendado) Builder “data-aware” con `--imatrix`

Para mejorar la calidad, `llama-statecells-build` puede ponderar el entrenamiento/encoding usando un **imatrix** (GGUF) generado por `llama-imatrix`.

1) Genera el imatrix con un dataset de calibración (texto plano):

```bash
IN="/home/frudas/.cache/llama.cpp/bartowski_mistralai_Devstral-Small-2-24B-Instruct-2512-GGUF_mistralai_Devstral-Small-2-24B-Instruct-2512-Q5_K_M.gguf"
CAL="./llama.cpp/calibration/devstral_calibration.txt"
IM="./llama.cpp/calibration/devstral_ctx512_chunks16.imatrix.gguf"

./llama.cpp/build/bin/llama-imatrix \
  -m "$IN" \
  -f "$CAL" \
  -o "$IM" \
  -t 16 -c 512 \
  --no-ppl --chunks 16 \
  -lv 2 --output-frequency 1
```

2) Usa ese imatrix durante el build:

```bash
IN="/home/frudas/.cache/llama.cpp/bartowski_mistralai_Devstral-Small-2-24B-Instruct-2512-GGUF_mistralai_Devstral-Small-2-24B-Instruct-2512-Q5_K_M.gguf"
OUT="/home/frudas/.cache/llama.cpp/devstral_sc.gguf"
IM="./llama.cpp/calibration/devstral_ctx512_chunks16.imatrix.gguf"

./llama.cpp/build/bin/llama-statecells-build \
  -i "$IN" \
  -o "$OUT" \
  -t 16 \
  --imatrix "$IM" \
  --imatrix-eps 1e-8 --imatrix-power 1.0 \
  --layers 6-8 \
  --dict-M-gate 4096 --dict-M-up 4096 --dict-M-down 512 \
  --dict-k 32 --dict-iters 2 --dict-max-samples 2048 \
  --vals --row-scale \
  --eval-cols 64 --report-json /tmp/devstral_sc_6-8.json \
  --checkpoint-every 1
```

Nota: si usas `--imatrix`, el JSON incluye métricas ponderadas `*_w` (aproximación “funcional” con covarianza diagonal).

### 11.6 Smoke test rápido con Gemma (pipeline end‑to‑end)

Útil para validar que `llama-imatrix` + `llama-statecells-build --imatrix` + `llama-cli --statecells` funcionan, sin esperar horas.

```bash
IN="/home/frudas/.cache/llama.cpp/ggml-org_gemma-3-1b-it-GGUF_gemma-3-1b-it-Q4_K_M.gguf"
CAL="/tmp/gemma_calibration.txt"
IM="/tmp/gemma_ctx512_chunks8.imatrix.gguf"
OUT="/tmp/gemma_sc_imatrix.gguf"

for i in $(seq 1 8000); do echo "hola mundo esto es calibracion"; done > "$CAL"

./llama.cpp/build/bin/llama-imatrix \
  -m "$IN" -f "$CAL" -o "$IM" \
  -t 16 -c 512 \
  --no-ppl --chunks 8 \
  -lv 2 --output-frequency 1

./llama.cpp/build/bin/llama-statecells-build \
  -i "$IN" -o "$OUT" \
  -t 16 \
  --imatrix "$IM" \
  --layers 0-1 \
  --dict-M 256 --dict-k 16 --dict-iters 1 --dict-max-samples 512 \
  --vals --row-scale \
  --eval-cols 16 --report-json /tmp/gemma_sc_imatrix.json

./llama.cpp/build/bin/llama-cli \
  -m "$OUT" \
  --statecells \
  -t 16 -c 1024 \
  -p "hola" -n 64 \
  --single-turn < /dev/null

# (opcional) Comparar calidad rápido (perplexity + QA) en wikitext-2 (solo 32 chunks)
TEXT_DIR="/tmp/wikitext-2-raw"
TEXT_FILE="$TEXT_DIR/wiki.test.raw"

if [ ! -f "$TEXT_FILE" ]; then
  wget -q -O /tmp/wikitext-2-raw-v1.zip https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip
  bsdtar -xf /tmp/wikitext-2-raw-v1.zip -C /tmp
fi

./llama.cpp/scripts/statecells-eval.sh \
  --base "$IN" \
  --sc "$OUT" \
  --text "$TEXT_FILE" \
  --threads 16 --ctx 512 \
  --chunks 32 \
  --outdir /tmp/statecells-eval-gemma

rg -n "Final estimate" /tmp/statecells-eval-gemma/perplexity_*.log

# Resultado de referencia (con los settings de arriba):
#   perplexity_base      : Final estimate: PPL = 26.7485 +/- 1.09312
#   perplexity_statecells: Final estimate: PPL = 8017.7608 +/- 339.39514
#
# Interpretación: el pipeline end‑to‑end funciona, pero esta configuración
# (muy agresiva: M=256,k=16 y encima en capas tempranas 0‑1) destruye calidad.
# Para “calidad real”, subir M/k/iters y preferir capas medias primero +
# calibración más fuerte (imatrix mejor / objetivo funcional).
```
