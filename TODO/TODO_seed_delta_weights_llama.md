# TODO — SeedΔ Weights: pesos implícitos (seed) + residual esparso para reducir RAM “a lo bestia” en llama.cpp

> **Objetivo:** representar matrices grandes `W` como:
>
> [
> W ;=; W_0(\text{seed}, \theta) ;+; \Delta
> ]
>
> donde `W0` se **genera al vuelo** (sin almacenar `W`) y `Δ` es un **residual pequeño** (sparse / block-sparse / codebook) que corrige lo importante.
> Meta: **reducir RAM de pesos más allá de quants** manteniendo calidad con **fallback seguro**.

---

## 0) Base teórica (corta, útil)

* En un LLM preentrenado, no puedes reemplazar `W` por una seed “random” y esperar que funcione: necesitas que `W0 + Δ ≈ W` **muy bien**.
* El truco de memoria real es:

  * **No almacenar** `W` (ni repack/dense),
  * **generar** una base `W0` determinista barata,
  * almacenar solo un **residual** compacto `Δ` (deltas).
* Esto es tu patrón biológico: **Naturaleza (seed)** + **Crianza (deltas)**, pero aplicado a pesos de un modelo ya entrenado con un esquema de compresión *post-hoc*.

---

## 1) Objetivo + no-objetivos

### Objetivo

* Permitir que `llama.cpp` ejecute ciertas matrices (primero FFN) sin cargar sus pesos densos, usando:

  1. **Base procedural** `W0(seed, θ)`
  2. **Residual** `Δ` compacto (sparse / block-sparse / codebook)
  3. **Escalado** correcto (row/col scale) + métricas + fallback

### No-objetivos (primera iteración)

* No re-entrenar el modelo completo.
* No mezclar múltiples backends sobre la misma matriz (ternario/TT/statecells/proc-sparse) en v1.
* No optimizar GPU aún (primero CPU correctness + AVX2).

---

## 2) Matemática mínima requerida

### 2.1 Descomposición

[
W = W_0(\text{seed},\theta) + \Delta
]
y en inferencia:
[
y = W x = W_0 x + \Delta x
]

### 2.2 Base `W0` (opciones)

**Opción A (v0 correctness):** `W0` pseudoaleatorio denso generado por PRNG por bloque

* Fácil de implementar, pero puede ser caro si lo haces elemento-a-elemento.

**Opción B (v1 performance, genérica):** `W0` estructurado tipo transformada rápida

* Ejemplo estilo “Fast Hadamard / Fastfood-like”:

  * `W0 x` se computa en `O(d log d)` (sin materializar `W0`),
  * requiere padding/bloques a potencias de 2 para FHT.

> **Regla práctica:** si `W0` cuesta ~como un matmul denso, no sirve. `W0` debe ser “barato” o al menos **cacheable por tokens**.

**Opción B.1 (recomendada, concreta):** `W0(seed, θ)` = *Stacked Fastfood / ACDC‑style* (diagonales + permutación + Hadamard)

Esta familia resuelve los 2 puntos que hacen que la idea viva o muera: (1) `W0x` barato y (2) manejar matrices **rectangulares** tipo LLM sin materializar `W0`.

**Definición (por bloque):**
- Elegir `L = next_pow2(n_in)` (p.ej. 5120 → 8192).
- Definir un operador rápido `F(θ): R^L → R^L` tipo:
  - `F(x) = D3 · H · P2 · D2 · H · P1 · D1 · x`
  - donde `H` es Hadamard (FHT), `D*` son diagonales (fp16/int8) y `P*` permutaciones (seed o array).

**Rectangularidad — caso “tall” (FFN gate/up):**
- `W ∈ R^{m×n}` con `m=32768`, `n=5120`.
- `L=8192`, `B = ceil(m/L)=4`.
- Para cada bloque `b=0..B-1`, se guarda `θ_b` y se computa:
  1) `x̂ = pad(x, L)`
  2) `y_b = F_b(x̂)` (en `R^L`)
  3) `y = concat(y_0..y_{B-1})[:m]`

**Rectangularidad — caso “wide” (FFN down):**
- `W ∈ R^{m×n}` con `m=5120`, `n=32768`.
- Partir `x` en `B=4` chunks de `L=8192` y acumular:
  1) `ŷ = 0 ∈ R^L`
  2) `for b: ŷ += F_b(x_b)`   (misma familia, ahora “reduce/suma”)
  3) `y = ŷ[:m]`

**Costo (orden de magnitud, gate/up Devstral):**
- `B * L * log2(L)` ≈ `4 * 8192 * 13` ≈ `425k` ops (+ diagonales/perms lineales),
- vs denso `m*n` ≈ `32768*5120` ≈ `167M` ops por token.

**Parámetros θ (memoria)**
- Por bloque `b`: 2–4 diagonales de longitud `L` + seeds/permutaciones.
- Ejemplo: 3 diagonales fp16 → `3 * 8192 * 2B = 48 KB` por bloque; `B=4` → `~192 KB` por matriz gate/up.
- Si falta capacidad, usar suma de pocos términos: `W0 = Σ_{r=1..R} W0_r` con `R` pequeño (2–8) y diagonales compartidas o separadas.

**Opción C (v2 expresividad):** `W0` tipo *butterfly / factorized transforms*

* Familia más expresiva que “solo Hadamard+diagonales”, pero con el mismo espíritu:
  - producto de factores estructurados (tipo mariposa/FFT‑like),
  - costo ~`O(d log d)` o `O(d log^2 d)` según parametrización,
  - se puede adaptar a rectangular igual que en la opción B.1 (stacking/tiling).

### 2.3 Residual `Δ` (compacto)

Tres esquemas (escalonados):

**(1) Sparse por fila (COO top-K):**

* Guardas para cada salida `o` los `K` índices de entrada `i` con mayor residual:
  [
  \Delta_{o,i} \neq 0 ;; \text{solo en top-K}
  ]
* Costo inferencia: `O(K * n_out)` por token.

**(2) Block-sparse (bloques 32/64):**

* Residual guardado por bloques contiguos para vectorizar.
* Costo: menos overhead que COO, mejor para AVX2.

**(3) Residual comprimido por codebook (StateCells como residual):**

* `Δ_row ≈ Σ s_j D[:,j]` pero **solo** para el residual.
* Esto evita el colapso visto cuando intentas que el codebook sea “todo W”.

### 2.4 Escalado (crítico)

Para evitar colapso por escala:

* Row-scale (por salida):
  [
  y_o = \alpha_o \cdot (W_0x + \Delta x)_o
  ]
* o block-scale si usas block-sparse.

> Esto es obligatorio en v1. Sin amplitud/escala, la PPL puede irse al espacio.

### 2.5 “Data-aware” (imatrix)

Si hay `imatrix`, optimizar **error ponderado**:
[
| (W - \hat W), \Sigma_x^{1/2}|_F^2
]
donde `Σx` viene aproximada por imatrix (importancia por dimensión).
Esto guía:

* selección de top-K del residual,
* búsqueda de seed/θ para `W0`.

**Nota importante (para que Δ sea realmente “pequeño”):** el objetivo debe ser **funcional** (data‑aware), no solo `||W-Ŵ||` en weight‑space:
[
\min_{\theta,\Delta}\ \mathbb{E}_{x\sim\mathcal{D}}\ \| (W - W_0(\theta) - \Delta)\,x \|_2^2
]
Con imatrix lo aproximas como covarianza diagonal:
[
\min\ \| (W - W_0(\theta) - \Delta)\,\Sigma_x^{1/2} \|_F^2
,\quad \Sigma_x \approx \mathrm{diag}(\mathrm{imatrix})
]

**¿De dónde sale `x` / `Σx`?**
- v1: imatrix (diagonal) ya da una señal útil para priorizar dimensiones “importantes”.
- v2 (mejor): recolectar **activaciones reales** `X` en los puntos exactos (entrada a `ffn_gate/up/down`) con el mismo texto de calibración, y optimizar `||WX - ŴX||` directamente. (Esto evita “ganar en W‑space y perder en PPL”.)

---

## 3) Formato GGUF propuesto

Por tensor objetivo (ej. `blk.N.ffn_gate.weight`):

### 3.1 Metadatos (KV)

* `seeddelta.enabled` (bool)
* `seeddelta.version` (u32)
* `seeddelta.scheme` (u32: 0=coo, 1=block, 2=codebook_resid)
* `seeddelta.base.kind` (u32: 0=prng_block, 1=hadamard_acdc_stack, 2=butterfly)
* `seeddelta.base.seed` (u64)
* `seeddelta.base.scale` (f32) (+ opcional mean/var)
* `seeddelta.base.L` (u32) (tamaño interno, pow2)
* `seeddelta.base.B` (u32) (número de bloques/tiles)
* `seeddelta.base.depth` (u32) (número de etapas / diagonales en `F`)
* `seeddelta.base.R` (u32) (número de términos en suma `Σ_r W0_r`)
* `seeddelta.resid.K` (u32) o `seeddelta.resid.block` (u32)
* `seeddelta.row_scale` (bool) + tipo (f16/f32)
* imatrix provenance si aplica (`...imatrix.file`, `...power`, etc.)

### 3.2 Tensores GGUF

**Base (mínimo):**

* `blk.N.ffn_gate.base_seed` (U64) *(o en KV si prefieres)*
* `blk.N.ffn_gate.base_d1` `[L, B]` (F16/F32) *(hadamard_acdc_stack)*
* `blk.N.ffn_gate.base_d2` `[L, B]` (F16/F32) *(opcional)*
* `blk.N.ffn_gate.base_d3` `[L, B]` (F16/F32) *(opcional)*
* `blk.N.ffn_gate.base_perm1` `[L, B]` (I16/I32) *(opcional; si no, se deriva de seed al cargar)*
* `blk.N.ffn_gate.base_perm2` `[L, B]` (I16/I32) *(opcional)*
* `blk.N.ffn_gate.d_row_scale` `[n_out]` (F16/F32) *(si habilitado)*

**Residual COO (scheme=0):**

* `blk.N.ffn_gate.d_idx` `[K, n_out]` (I16/I32)
* `blk.N.ffn_gate.d_val` `[K, n_out]` (F16/F32)
* `blk.N.ffn_gate.d_row_scale` `[n_out]` (F16/F32) *(opcional)*

**Residual block-sparse (scheme=1):**

* `blk.N.ffn_gate.b_idx` `[n_blocks, n_out]` (I16/I32) (índice de bloque en `[0, n_in/block)`)
* `blk.N.ffn_gate.b_val` `[block, n_blocks, n_out]` (F16/F32)
* `blk.N.ffn_gate.d_row_scale` `[n_out]` (F16/F32) *(opcional)*

**Residual codebook (scheme=2, opcional):**

* `blk.N.ffn_gate.r_dict` `[n_in, M]` (F16)
* `blk.N.ffn_gate.r_codes` `[k, n_out]` (I16)
* `blk.N.ffn_gate.r_vals` `[k, n_out]` (F16) *(para amplitud)*

---

## 4) Integración en llama.cpp

### 4.1 Offline builder: `llama-seeddelta-build`

Inputs:

* `-i in.gguf -o out.gguf`
* `--layers A-B`
* `--scheme coo|block` *(v1: `codebook_resid` queda para v2)*
* `--block N` *(solo si `--scheme block`)*
* `--K N` + overrides `--K-gate/--K-up/--K-down`
* `--idx-type i16|i32 --val-type f16|f32`
* `--row-scale` / `--no-row-scale`
* `--imatrix file --imatrix-power p --imatrix-eps eps`
* `--base --base-max-samples N --base-perm-trials N`
* `-t/--threads N`
* `--eval-cols N --eval-x N --report-json path`

Outputs:

* GGUF extendido con tensores `seeddelta.*`
* JSON con métricas por tensor:

  * `rel_l2`, `cos`, y versiones ponderadas `*_w`
  * `*_x` (gap funcional en output-space: `||WX - ŴX||`, requiere `--eval-x`)
  * `norm_ratio` (importantísimo)
  * estimación de costo compute: `ops_*`, `ops_ratio`

### 4.2 Runtime

* Loader: registrar “payload” para cada tensor base `w` similar a tu `statecells_ctx.weights`.
* Graph hook (igual que hiciste en `build_lora_mm`):

  * si `seeddelta` está disponible → `llama_seeddelta_mul_mat(ctx0, cur, payload...)`
  * si no → `ggml_mul_mat(ctx0, w, cur)`
* Kernel custom ggml op:

  * compute: `y = row_scale * (W0x + Δx)`
  * multi-thread por `n_out`.

### 4.3 Gap / fallback seguro

* `--seeddelta-gap` (tolerancia)
* “gap check” barato:

  * offline guardar `||W||`, `||Ŵ||`, `||W-Ŵ||` (ponderado y no ponderado)
  * runtime: si tensor marcado “unsafe” o si `nan_guard` dispara → fallback a denso.

---

## 5) Roadmap por fases (entregables escalonados)

### Fase 0 — Spec + harness (1–2 días de ingeniería)

**Objetivo:** no perderte en teoría; tener mediciones reproducibles.

**Entregables**

* [x] `tools/seeddelta-eval/` o modo `--report-json` en builder
* [x] Métricas por tensor: `rel_l2`, `cos`, `norm_ratio`, y `_w` si imatrix
* [x] Script A/B: `llama-perplexity base vs seeddelta` (chunks cortos) → `scripts/seeddelta-eval.sh`
* [ ] Unit test pequeño: compara `W0x + Δx` vs `Wx` en un tensor (tolerancias)

---

### Fase 1 — Builder v0 (correctness first)

**Base-kind:** `prng_block` (solo harness) → `hadamard_acdc_stack` (camino real)
**Scheme:** `coo` con `d_val=fp16` + `row_scale`

**Entregables**

* [x] `llama-seeddelta-build` lee GGUF y escribe GGUF extendido
* [x] Generación de `W0` por bloque sin materializar todo W *(v1: base Hadamard XOR-circulant, `base_d1/d2/d3`)*
* [x] Residual top-K por fila con `fp16 vals` (COO)
* [x] `row_scale` por fila (mínimo) *(nota: en COO puro suele quedar ~1.0)*
* [x] JSON report por tensor (`--report-json`)

**Aceptación**

* PPL no explota (ΔPPL razonable) en 1–2 capas FFN de Gemma/1B

---

### Fase 1b — Builder v1 (base rápida real)

**Base-kind:** `hadamard_acdc_stack` con `L=next_pow2(n_in)` y stacking rectangular (B.1)
**Scheme:** `block` (preferido) / `coo` (smoke)

**Entregables**

* [x] Emisión de tensores base (`base_d1/base_d2/base_d3/base_perm1`, `base.depth/R/*`) en GGUF *(base_perm2 pendiente)*
* [x] Kernel offline para `W0x` y para residual `Δ` (para evaluar `||WX - ŴX||`) vía `--eval-x`
* [x] Report de costo estimado por tensor (`ops_dense/base/delta/total`, `ops_ratio`)
* [x] Overrides por matriz para residual budget (`--K-gate/--K-up/--K-down`) + metadata/JSON

---

### Fase 2 — Runtime v0 (correctness)

**Objetivo:** que corra end-to-end en `llama-cli` con `--seeddelta`.

**Entregables**

* [x] `src/llama-seeddelta.{h,cpp}` + `llama_seeddelta_context`
* [x] Loader opcional de tensores GGUF seeddelta (COO + block residual)
* [x] Hook en `build_lora_mm` igual a StateCells
* [x] Custom op ggml: `y = Δx` (COO + block)
* [x] Custom op ggml: `y = W0x + Δx` (base + COO + block)
* [ ] Nan-Guardian integrado para fallback

**Aceptación**

* Genera texto sin NaNs
* A/B perplexity con chunks pequeños funciona

---

### Fase 3 — Performance (hacerlo “real”)

**Objetivo:** que el costo compute no mate el beneficio.

**Entregables**

* [x] Scheme block-sparse (bloques 32/64) end-to-end (builder+runtime) para reemplazar COO (COO es solo smoke; es muy malo para CPU)
* [ ] `W0` optimizado (cache por token/batch, layout-friendly)
* [ ] Reducir overhead de indices (U16 cuando posible)
* [ ] Bench: `llama-cli` (prompt+gen) + `llama-perplexity` tok/s *(opcional: extender `llama-bench` con `--seeddelta`)*

**Aceptación**

* Speed regression <= X% vs base (o mejor) en CPU
* RAM de pesos baja de forma medible (y no sube por repack)

---

### Fase 4 — Data-aware (imatrix) + autotune

**Objetivo:** calidad por dólar de memoria.

**Entregables**

* [ ] Residual selection ponderada por imatrix (`*_w`)
* [ ] Seed search (pocos trials) para minimizar error ponderado
* [ ] Autotune por capa: decide `K/block` con budget global
* [ ] Report consolidado: “MB saved vs ΔPPL”

**Aceptación**

* Curvas “mem vs ppl” por modelo / target layer-range

---

### Fase 5 — Expandir targets

**Objetivo:** ir más allá de FFN.

**Entregables**

* [ ] QKV/O (attention projections) si es viable
* [ ] Embeddings / LM head (solo si no revienta calidad)
* [ ] Per-layer policy: enable/disable automático por gap

---

## 6) Métricas de éxito (gates)

* **Memoria pesos (host):** ↓ ≥ 30–60% (según layers cubiertas) sin “pagar” repack
* **Speed (gen, batch=1):** no peor que base más de un margen tolerable (o mejorar)
* **Speed (prompt, batch>1):** puede empeorar al principio; no usarlo como único gate mientras no exista kernel block-sparse/GEMM-friendly
* **Calidad:** ΔPPL controlada (por target)
* **Estabilidad:** 0 NaNs, fallback funciona

---

## 7) Riesgos + mitigaciones

* **W0 caro** → usar `fht_struct`/bloques y cache; si no, el plan muere.
* **Residual sin escala** → PPL se va al espacio: `vals` y `row_scale` son obligatorios.
* **Top-K muy agresivo** → colapso: empezar con K moderado y autotune.
* **No todas las matrices se dejan** → policy por capa + fallback.
* **Prompt phase engañosa** → medir gen (batch=1) por separado; prompt usa GEMM denso ultra-tuneado y puede “castigar” custom ops.

---

## 8) Compatibilidad e interacciones

* Backend exclusivo por matriz: no mezclar con TT/ternario/statecells en la misma matriz v1.
* StateCells queda como **opción para comprimir residual** (scheme=2), no como reemplazo total.
* KWTA/event-driven/StatePack pueden ayudar después (para skip de bloques), pero no en v1.

---

## 9) Commits sugeridos

* `seeddelta: add GGUF schema + metadata for seed+residual weights`
* `tools: add llama-seeddelta-build (base+residual exporter)`
* `runtime: wire seeddelta context + CLI flags`
* `kernels: add seeddelta mul_mat custom op (correctness)`
* `kernels: add block-sparse residual + avx2 path`
* `eval: add imatrix-weighted metrics + seed search + autotune`

---

Nota: el “Patch-Ready Plan” anterior se removió para evitar drift. La fuente de verdad ahora es:
- Builder: `tools/seeddelta-build/seeddelta-build.cpp` (`./build/bin/llama-seeddelta-build --help`)
- Runtime: `src/llama-seeddelta.{h,cpp}` + flags `--seeddelta/--seeddelta-gap` en `common/arg.cpp`.

---

## 10) Smoke test end-to-end (Gemma 3 1B)

> Objetivo: validar que `W0x + Δx` corre en runtime (`llama-perplexity`) sin explotar la PPL.

### 10.1 Build SeedΔ GGUF (2 capas para iterar rápido)

```bash
IN="/home/frudas/.cache/llama.cpp/ggml-org_gemma-3-1b-it-GGUF_gemma-3-1b-it-Q4_K_M.gguf"
IM="llama.cpp/calibration/gemma.imatrix.gguf"
OUT="llama.cpp/calibration/gemma_sd_base.gguf"

./llama.cpp/build/bin/llama-seeddelta-build \
  -i "$IN" -o "$OUT" \
  --layers 0-1 \
  --scheme block --block 16 \
  --K-gate 64 --K-up 64 --K-down 128 \
  --base --base-max-samples 2048 --base-perm-trials 4 \
  --row-scale \
  --imatrix "$IM" \
  -t 16 \
  --eval-cols 64 --eval-x 16 \
  --report-json llama.cpp/calibration/gemma_sd_0-1.json
```

### 10.2 Comparar PPL (A/B)

```bash
./llama.cpp/scripts/seeddelta-eval.sh \
  --base "$IN" \
  --sd   "$OUT" \
  --text "/home/frudas/synapp2/llama.cpp/calibration/gemma_calibration.txt" \
  --threads 16 --ctx 512 --chunks 16 --no-qa
```

### 10.3 Resultados esperados (sanity)

* La PPL de `--seeddelta` no debe “explotar” vs base (este harness usa `gemma_calibration.txt`, no wikitext).
* Nota: con residual **COO** el prompt eval puede verse peor; el gate real de performance es *gen batch=1* y requiere scheme **block-sparse**.

Ejemplo real (misma máquina, `ctx=512`, `chunks=16`, capas 10–11, `K_gate=64 K_up=64 K_down=128`):
* base: `PPL ≈ 1.0050`, prompt `≈ 161 tok/s`
* seeddelta(coo): `PPL ≈ 1.0039`, prompt `≈ 114 tok/s`
* seeddelta(block16): `PPL ≈ 1.0018`, prompt `≈ 123 tok/s`

Interpretación:
* `block16` suele recuperar calidad mejor que `block32` con el mismo budget.
* `block32` con budget bajo puede perder calidad; para recuperarla suele requerir más budget (p.ej. `K_gate/up=128`, `K_down=256`).

### 10.4 Capas medias (10–11) + comparación COO vs block

```bash
IN="/home/frudas/.cache/llama.cpp/ggml-org_gemma-3-1b-it-GGUF_gemma-3-1b-it-Q4_K_M.gguf"
IM="calibration/gemma.imatrix.gguf"
TEXT="calibration/gemma_calibration.txt"

./build/bin/llama-seeddelta-build \
  -i "$IN" -o calibration/gemma_sd_mid_10-11_coo.gguf \
  --layers 10-11 \
  --scheme coo \
  --K-gate 64 --K-up 64 --K-down 128 \
  --base --base-max-samples 2048 --base-perm-trials 4 \
  --row-scale --imatrix "$IM" \
  -t 16 --eval-cols 64 --eval-x 16 \
  --report-json calibration/gemma_sd_mid_10-11_coo.json

./build/bin/llama-seeddelta-build \
  -i "$IN" -o calibration/gemma_sd_mid_10-11_block32.gguf \
  --layers 10-11 \
  --scheme block --block 32 \
  --K-gate 64 --K-up 64 --K-down 128 \
  --base --base-max-samples 2048 --base-perm-trials 4 \
  --row-scale --imatrix "$IM" \
  -t 16 --eval-cols 64 --eval-x 16 \
  --report-json calibration/gemma_sd_mid_10-11_block32.json

./scripts/seeddelta-eval.sh \
  --base "$IN" \
  --sd   calibration/gemma_sd_mid_10-11_coo.gguf \
  --text "$TEXT" \
  --threads 16 --ctx 512 --chunks 16 --no-qa

./scripts/seeddelta-eval.sh \
  --base "$IN" \
  --sd   calibration/gemma_sd_mid_10-11_block32.gguf \
  --text "$TEXT" \
  --threads 16 --ctx 512 --chunks 16 --no-qa

# Variante de calidad (block32 + más budget):
./build/bin/llama-seeddelta-build \
  -i "$IN" -o calibration/gemma_sd_mid_10-11_block32_k128.gguf \
  --layers 10-11 \
  --scheme block --block 32 \
  --K-gate 128 --K-up 128 --K-down 256 \
  --base --base-max-samples 2048 --base-perm-trials 4 \
  --row-scale --imatrix "$IM" \
  -t 16 --eval-cols 64 --eval-x 16 \
  --report-json calibration/gemma_sd_mid_10-11_block32_k128.json

./scripts/seeddelta-eval.sh \
  --base "$IN" \
  --sd   calibration/gemma_sd_mid_10-11_block32_k128.gguf \
  --text "$TEXT" \
  --threads 16 --ctx 512 --chunks 16 --no-qa

# Variante recomendada (block16 + mismo budget):
./build/bin/llama-seeddelta-build \
  -i "$IN" -o calibration/gemma_sd_mid_10-11_block16_k64.gguf \
  --layers 10-11 \
  --scheme block --block 16 \
  --K-gate 64 --K-up 64 --K-down 128 \
  --base --base-max-samples 2048 --base-perm-trials 4 \
  --row-scale --imatrix "$IM" \
  -t 16 --eval-cols 64 --eval-x 16 \
  --report-json calibration/gemma_sd_mid_10-11_block16_k64.json

./scripts/seeddelta-eval.sh \
  --base "$IN" \
  --sd   calibration/gemma_sd_mid_10-11_block16_k64.gguf \
  --text "$TEXT" \
  --threads 16 --ctx 512 --chunks 16 --no-qa
```

### 10.5 Prueba intermedia (Gemma 3 4B IT) antes de Devstral

> Objetivo: validar que el pipeline (imatrix → seeddelta-build → runtime) funciona en un modelo más grande.

#### 10.5.1 Generar imatrix (Gemma 4B)

```bash
IN="/home/frudas/.cache/llama.cpp/ggml-org_gemma-3-4b-it-GGUF_gemma-3-4b-it-Q4_K_M.gguf"
CAL="llama.cpp/calibration/gemma_calibration.txt"
IM="llama.cpp/calibration/gemma4b.imatrix.gguf"

./llama.cpp/build/bin/llama-imatrix \
  -m "$IN" -f "$CAL" -o "$IM" \
  -t 16 -c 512 --no-ppl --chunks 8
```

#### 10.5.2 Build SeedΔ (capas medias 16–17)

```bash
IN="/home/frudas/.cache/llama.cpp/ggml-org_gemma-3-4b-it-GGUF_gemma-3-4b-it-Q4_K_M.gguf"
IM="llama.cpp/calibration/gemma4b.imatrix.gguf"
OUT="llama.cpp/calibration/gemma4b_sd_mid_16-17_block16_k64.gguf"

./llama.cpp/build/bin/llama-seeddelta-build \
  -i "$IN" -o "$OUT" \
  --layers 16-17 \
  --scheme block --block 16 \
  --K-gate 64 --K-up 64 --K-down 128 \
  --base --base-max-samples 2048 --base-perm-trials 4 \
  --row-scale --imatrix "$IM" \
  -t 16 --eval-cols 64 --eval-x 16 \
  --report-json llama.cpp/calibration/gemma4b_sd_mid_16-17_block16_k64.json
```

#### 10.5.3 Comparar PPL (A/B)

```bash
./llama.cpp/scripts/seeddelta-eval.sh \
  --base "$IN" \
  --sd   "$OUT" \
  --text "$CAL" \
  --threads 16 --ctx 512 --chunks 16 --no-qa \
  --outdir llama.cpp/calibration/seeddelta-eval-gemma4b-mid_16-17-block16_k64
```

#### 10.5.4 Resultado esperado (sanity)

* Este harness usa `gemma_calibration.txt` (placeholder), así que la PPL absoluta no es un gate; solo sirve para ver que **no explota**.
* En mi corrida (misma máquina, `ctx=512`, `chunks=16`):
  * base: `PPL ≈ 1.0001`, prompt `≈ 61.7 tok/s`
  * seeddelta(block16): `PPL ≈ 1.0022`, prompt `≈ 51.8 tok/s`
* Nota: en esta etapa el GGUF output aún incluye pesos densos + tensores SeedΔ (sirve para iterar/fallback). El ahorro de RAM real requiere la fase “strip/skip dense”.
