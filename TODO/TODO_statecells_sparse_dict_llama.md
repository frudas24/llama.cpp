# TODO — StateCells / Diccionario k‑esparso para comprimir pesos LLM

> **Objetivo:** adaptar tu tecnología “StateCells + SparseAE (Oja + k‑WTA)” para LLMs en llama.cpp: representar matrices grandes como **combinaciones k‑esparsas de un diccionario** aprendido. Meta: **reducir RAM de pesos** de forma más agresiva que Q‑quants, manteniendo calidad con fallback seguro.

---

## Índice
1) Idea central  
2) Matemática esencial  
3) Config/flags  
4) Integración (offline + runtime)  
5) Pseudocódigo  
6) Métricas/aceptación  
7) Roadmap  
8) Riesgos/mitigación  
9) Commits sugeridos

---

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
    - `--dict-k` átomos activos por columna/fila (p.ej. 16–48).
    - `--dict-eta`, `--dict-iters`, `--dict-max-samples`, `--layers A-B`, `--dict-type f16|f32`.
  - Esquemas:
    - `sign` (actual): coeficiente implícito ±1 dentro de `codes` I16.
    - `fp16` (futuro): tensor adicional `..._vals` con coeficientes fp16.
- Runtime:
  - `--statecells` habilita backend si el GGUF trae dict/codes.
  - `--statecells-gap 0.02` tolerancia de fallback (por ahora global).
  - (futuro) flags per‑layer/target cuando extendamos más allá de FFN.

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
     - (opcional futuro) `blk.N.*.vals` `[k, d_out]` fp16 para esquema `fp16`.

  **Interpretación codes (sign‑scheme):**
  - `codes[t, r] = ±(atom+1)` con `atom∈[0,M)`; `0` = slot vacío.

  **Metadatos GGUF (modelo):**
  - `statecells.enabled` bool
  - `statecells.dict.M`, `statecells.dict.k`, `statecells.dict.eta`, `statecells.dict.iters`
  - (futuro) `statecells.scheme` (0 sign, 1 fp16), `statecells.version`, `statecells.default_gap_tol`.

  **Nota de disco/RAM:** el GGUF resultante puede conservar pesos densos para compatibilidad; con `mmap` no deberían cargar en RAM si el runtime usa StateCells.

**4.2 Runtime (inferencia)**
Dos rutas:
1. **Sin reconstruir W:**  
   - Para input `x`, computa proyecciones base `p_j = dot(D[:,j], x)` para todos `j=1..M`.  
   - Para cada fila `r`: `y_r = Σ_{(j,s)∈code(r)} s · p_j`.  
   - Complejidad ≈ `O(M·d + k·rows)` (si `M ≪ rows`, ahorro grande).
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

### 7) Roadmap
- Estado actual: backend runtime + tool `llama-statecells-build` (sign‑scheme) ya implementados; falta kernel CPU rápido, scheme fp16/vals y gap per‑layer.
1. Tool offline `statecells-build` (leer GGUF, entrenar D, emitir codes).  
2. Formato GGUF extendido + loader en llama.cpp.  
3. Kernel runtime “sin reconstrucción” para MLP primero.  
4. Integración con KWTA/event‑driven para explotar que no todas las filas/bloques se usan.  
5. Bench `llama-bench` + `llama-perplexity` A/B vs Q4/Q5.

### 8) Riesgos/mitigación
- **M demasiado grande** → no hay ahorro: auto‑tune M por capa (target M≈rows/8).  
- **k muy bajo** → sube ppl: empezar con k=16–32 y bajar gradualmente.  
- **Costo p_j** puede dominar: vectorizar dot(D,x) y reusar para varias proyecciones en la capa.  
- **No todas las capas separables**: permitir fallback per‑layer con `gap_tol`.

### Compatibilidad e interacciones
- **Backend de pesos exclusivo por capa:** no combinar StateCells con ternario/TT/proc‑sparse en la misma matriz en primera iteración.  
- Sinergia con KWTA/event‑driven: usando la ruta “reconstrucción bajo demanda”, las máscaras reducen reconstrucciones y aumentan el ahorro real.  
- Nan‑Guardian debe cubrir reescalados/renorm y la salida reconstruida antes de mezclarse con staleness.

### 9) Commits sugeridos
- `statecells: add offline dict+code builder for GGUF`  
- `gguf: add dict/codes tensor types + metadata`  
- `kernels: add statecells matvec path (mlp)`  
- `runtime: flags + gap fallback + metrics`
