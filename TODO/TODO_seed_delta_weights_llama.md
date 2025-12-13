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

**Opción B (v1 performance):** `W0` estructurado tipo transformada rápida

* Ejemplo estilo “Fast Hadamard / Fastfood-like”:

  * `W0 x` se computa en `O(d log d)` (sin materializar `W0`),
  * requiere padding/bloques a potencias de 2 para FHT.

> **Regla práctica:** si `W0` cuesta ~como un matmul denso, no sirve. `W0` debe ser “barato” o al menos **cacheable por tokens**.

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

---

## 3) Formato GGUF propuesto

Por tensor objetivo (ej. `blk.N.ffn_gate.weight`):

### 3.1 Metadatos (KV)

* `seeddelta.enabled` (bool)
* `seeddelta.version` (u32)
* `seeddelta.scheme` (u32: 0=coo, 1=block, 2=codebook_resid)
* `seeddelta.base.kind` (u32: 0=prng_block, 1=fht_struct)
* `seeddelta.base.seed` (u64)
* `seeddelta.base.scale` (f32) (+ opcional mean/var)
* `seeddelta.resid.K` (u32) o `seeddelta.resid.block` (u32)
* `seeddelta.row_scale` (bool) + tipo (f16/f32)
* imatrix provenance si aplica (`...imatrix.file`, `...power`, etc.)

### 3.2 Tensores GGUF

**Base (mínimo):**

* `blk.N.ffn_gate.base_seed` (U64) *(o en KV si prefieres)*
* `blk.N.ffn_gate.row_scale` `[n_out]` (F16/F32) *(si habilitado)*

**Residual COO (scheme=0):**

* `blk.N.ffn_gate.d_idx` `[K, n_out]` (U16/U32)
* `blk.N.ffn_gate.d_val` `[K, n_out]` (F16 o I8 + escala)

**Residual block-sparse (scheme=1):**

* `blk.N.ffn_gate.b_map` `[n_out, n_blocks_kept]` (U16) (índices de bloque)
* `blk.N.ffn_gate.b_val` `[n_out, n_blocks_kept, block]` (I8/F16)
* `blk.N.ffn_gate.b_scale` `[n_out, n_blocks_kept]` (F16/F32) *(si int8)*

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
* `--targets ffn_gate,ffn_up,ffn_down`
* `--base-kind prng_block|fht_struct`
* `--seed N` + `--seed-search NTRIALS` (opcional)
* `--scheme coo|block|codebook_resid`
* `--K 16|32|64` o `--block 32|64`
* `--vals fp16|int8` + escalas
* `--row-scale`
* `--imatrix file --imatrix-power p --imatrix-eps eps`
* `--eval-cols N --report-json path`

Outputs:

* GGUF extendido con tensores `seeddelta.*`
* JSON con métricas por tensor:

  * `rel_l2`, `cos`, y versiones ponderadas `*_w`
  * `norm_ratio` (importantísimo)
  * estimación de RAM y costo compute

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

* [ ] `tools/seeddelta-eval/` o modo `--report-json` en builder
* [ ] Métricas por tensor: `rel_l2`, `cos`, `norm_ratio`, y `_w` si imatrix
* [ ] Script A/B: `llama-perplexity base vs seeddelta` (chunks cortos)
* [ ] Unit test pequeño: compara `W0x + Δx` vs `Wx` en un tensor (tolerancias)

---

### Fase 1 — Builder v0 (correctness first)

**Base-kind:** `prng_block` (sin FHT todavía)
**Scheme:** `coo` con `d_val=fp16` + `row_scale`

**Entregables**

* [ ] `llama-seeddelta-build` lee GGUF y escribe GGUF extendido
* [ ] Generación de `W0` por bloque sin materializar todo W
* [ ] Residual top-K por fila con `fp16 vals`
* [ ] `row_scale` por fila (mínimo)
* [ ] JSON report por tensor

**Aceptación**

* PPL no explota (ΔPPL razonable) en 1–2 capas FFN de Gemma/1B

---

### Fase 2 — Runtime v0 (correctness)

**Objetivo:** que corra end-to-end en `llama-cli` con `--seeddelta`.

**Entregables**

* [ ] `src/llama-seeddelta.{h,cpp}` + `llama_seeddelta_context`
* [ ] Loader opcional de tensores GGUF seeddelta
* [ ] Hook en `build_lora_mm` igual a StateCells
* [ ] Custom op ggml: `y = W0x + Δx`
* [ ] Nan-Guardian integrado para fallback

**Aceptación**

* Genera texto sin NaNs
* A/B perplexity con chunks pequeños funciona

---

### Fase 3 — Performance (hacerlo “real”)

**Objetivo:** que el costo compute no mate el beneficio.

**Entregables**

* [ ] Scheme block-sparse (bloques 32/64) + AVX2 para `Δx`
* [ ] `W0` optimizado (cache por token/batch, layout-friendly)
* [ ] Reducir overhead de indices (U16 cuando posible)
* [ ] Bench `llama-bench` prompt+gen

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
* **Speed:** no peor que base más de un margen tolerable (o mejorar)
* **Calidad:** ΔPPL controlada (por target)
* **Estabilidad:** 0 NaNs, fallback funciona

---

## 7) Riesgos + mitigaciones

* **W0 caro** → usar `fht_struct`/bloques y cache; si no, el plan muere.
* **Residual sin escala** → PPL se va al espacio: `vals` y `row_scale` son obligatorios.
* **Top-K muy agresivo** → colapso: empezar con K moderado y autotune.
* **No todas las matrices se dejan** → policy por capa + fallback.

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

aquí te dejo el **plan de implementación** en formato "patch-ready", con los detalles exactos de los **nombres de flags**, **structs C++**, puntos de integración y el **plan de migración** de **StateCells → residual-codebook**. Este plan estará organizado para que puedas integrarlo directamente en tu flujo de trabajo con `llama.cpp`.

---

# Patch-Ready Plan para Implementación `SeedΔ Weights` en `llama.cpp`

## 1) **Flags de Configuración**

### 1.1 Flags para el **Builder**

* **`--seeddelta`**: habilita la descomposición de pesos en `seed` y `residual`.

  * **Ejemplo**: `--seeddelta`
  * **Acción**: Cuando este flag esté presente, el builder generará un archivo GGUF con los pesos descompuestos (semilla + residual).

* **`--seeddelta-gap`**: umbral de tolerancia para la diferencia entre el modelo original y el modelo reducido.

  * **Ejemplo**: `--seeddelta-gap 0.01`
  * **Acción**: Este parámetro ajusta cuánto se puede permitir que el residual difiera de los pesos originales.

* **`--seeddelta-scheme`**: tipo de esquema de residual.

  * **Ejemplo**: `--seeddelta-scheme coo`
  * **Opciones**:

    * `coo`: Residual sparse por fila (top-K).
    * `block`: Residual block-sparse (bloques de 32/64).
    * `codebook`: Residual basado en código de libros (Codebook para compresión).

* **`--seeddelta-vals`**: tipo de valores para el residual (e.g., `fp16`, `int8`).

  * **Ejemplo**: `--seeddelta-vals fp16`
  * **Acción**: Define el tipo de valores para el residual (si está usando `block-sparse` o `codebook`).

* **`--seeddelta-row-scale`**: habilita la escala por fila (aplicable si el residual está por fila o bloque).

  * **Ejemplo**: `--seeddelta-row-scale`
  * **Acción**: Aplica la escala por fila a los pesos de salida en la descomposición.

### 1.2 Flags para el **Runtime**

* **`--statecells`**: habilita el uso de StateCells (esto es solo un fallback para modelos que no tienen `seed-delta`).

  * **Ejemplo**: `--statecells`
  * **Acción**: Cuando este flag está presente, `llama.cpp` utilizará StateCells en lugar de `seeddelta` si no se encuentra `seeddelta` en los pesos.
* **`--seeddelta-fallback`**: habilita el fallback de `seeddelta` a denso si se detecta un problema de memoria o calidad.

  * **Ejemplo**: `--seeddelta-fallback`
  * **Acción**: Si el modelo encuentra que `seeddelta` no es viable (por ejemplo, por NaNs), el modelo usa la matriz densa original.

---

## 2) **Estructuras de Datos C++**

### 2.1 `seeddelta_ctx` — Contexto de Residuos Seed-Delta

Esta estructura contendrá los metadatos y tensores necesarios para el cálculo de los pesos.

```cpp
struct seeddelta_ctx {
    bool enabled;                 // Si el modelo usa Seed-Delta
    uint32_t scheme;              // Esquema (0 = COO, 1 = Block, 2 = Codebook)
    uint32_t residual_size;       // Tamaño de residual por fila o bloque
    float* base_seed;             // Semilla base para generar W0
    float* residual_vals;         // Valores del residual
    uint16_t* residual_indices;   // Índices de las conexiones activas
    float* row_scale;             // Escala por fila (si aplica)
    uint32_t block_size;          // Tamaño del bloque (si aplica)
    float tolerance_gap;          // Tolerancia para comparación con W0
};
```

### 2.2 Funciones para Manipular `seeddelta_ctx`

#### `seeddelta_load_weights`

* **Propósito**: Cargar la semilla y el residual desde el archivo GGUF.

```cpp
int seeddelta_load_weights(seeddelta_ctx& ctx, const char* file_path) {
    // Cargar los tensores desde el archivo GGUF
    // Configurar ctx.base_seed, ctx.residual_vals, ctx.residual_indices, etc.
}
```

#### `seeddelta_compute`

* **Propósito**: Ejecutar la operación `W0 * x + Δ * x` usando la semilla base y el residual.

```cpp
void seeddelta_compute(const seeddelta_ctx& ctx, const float* input, float* output, size_t num_elements) {
    // Realizar la multiplicación de matrices (por fila/bloque), considerando la semilla y el residual
}
```

---

## 3) **Puntos de Integración en `llama.cpp`**

### 3.1 Carga de Tensores Seed-Delta

* En la función `llama_load_weights`, añade un bloque condicional para cargar la semilla y el residual si el flag `--seeddelta` está activado.

```cpp
if (ctx.seeddelta_enabled) {
    // Llamar a seeddelta_load_weights
    seeddelta_load_weights(ctx.seeddelta_ctx, file_path);
}
```

### 3.2 Ejecución de la Multiplicación `W * x` con Seed-Delta

* En la función `llama_compute`, cuando `--seeddelta` está activado, usar `seeddelta_compute` para la multiplicación de pesos.

```cpp
if (ctx.seeddelta_enabled) {
    // Usar la función seeddelta_compute para ejecutar el modelo con Seed-Delta
    seeddelta_compute(ctx.seeddelta_ctx, input, output, num_elements);
} else {
    // Fallback a la multiplicación convencional si no se usa Seed-Delta
    ggml_mul_mat(ctx, input, output, num_elements);
}
```

---

## 4) **Plan de Migración: StateCells → Residual-Codebook**

### 4.1 **Migración de StateCells**

* **StateCells** es el enfoque actual basado en memoria externa, y se utiliza para almacenar las conexiones explícitas.

Para **migrar a residual-codebook**:

1. **StateCells → Residual**: En lugar de almacenar la matriz completa `W`, almacenamos solo los residuos `Δ` como **sparse blocks** o **codebook**.
2. **Fallback**: Si el modelo se encuentra con problemas de memoria, puede volver a usar `StateCells` mediante el flag `--statecells` como modo de compatibilidad.

### 4.2 **Pasos de Migración**:

1. **Fase 1**: Implementación de `seeddelta` básico usando `COO` o `block-sparse`.

   * **Entregables**: Implementación de la carga de semillas, cálculo de `W0` y `Δ`, integración con el sistema.

2. **Fase 2**: Sustitución de `StateCells` por la nueva estructura `seeddelta_ctx` donde los pesos se calculan dinámicamente.

   * **Entregables**: Refactorización del código que depende de `StateCells` para que utilice el nuevo esquema `seeddelta_ctx`.

3. **Fase 3**: Optimización de `seeddelta` y evaluación del impacto en memoria, velocidad y calidad de la inferencia.

   * **Entregables**: Benchmarking de `PPL` y rendimiento comparando `StateCells` vs `seeddelta`.

4. **Fase 4**: Implementación de código de libro y búsqueda automática de residuos por capa.

   * **Entregables**: Búsqueda de `Δ` más eficiente por capa utilizando técnicas de optimización.

---

## 5) **Pruebas y Evaluación**

1. **Test de Calidad (PPL)**: Compara `PPL` entre la implementación tradicional (sin `--seeddelta`) y la nueva (`--seeddelta`).
2. **Test de Memoria**: Mide el uso de memoria comparando el tamaño de los tensores y la carga de los modelos.
3. **Test de Rendimiento**: Evalúa la latencia por token con `llama-perplexity` usando entradas representativas.

---

### 6) **Commits sugeridos**

1. `seeddelta: introduce Seed+Residual weights (initial implementation)`
2. `runtime: add support for seeddelta weights computation`
3. `builder: implement --seeddelta flag and memory optimizations`
4. `kernels: add custom op for seeddelta weights calculation`
5. `test: add PPL benchmarks for seed-residual vs dense`

---

Esto debe permitirte una migración suave y escalonada hacia el uso de **Seed+Residual** como pesos implícitos, manteniendo la calidad y reduciendo la memoria en `llama.cpp` en cada fase.