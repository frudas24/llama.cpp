# TODO — Fanout hashed + top‑K (backend procedural) para sparsidad CPU

> **Objetivo:** portar el patrón procedural de Synapse (pesos base por hash + deltas top‑K y trimming) para generar matrices block‑sparse y reconstruir solo las columnas necesarias en CPU, reduciendo memoria y flops.

---

## Índice
1) Idea y parámetros  
2) Config/flags  
3) Integración (preproc, kernel)  
4) Pseudocódigo  
5) Aceptación  
6) Roadmap  
7) Riesgos  
8) Commits sugeridos

---

### 1) Idea y parámetros
- **Peso base**: función hash determinista `splitmix64(pre, post) * base_scale` (pequeño).  
- **Deltas**: guardar solo |Δ| grandes por fila (top‑K/heap) y recortar soft (eps) + hard (watermark) igual que `applyShedding`.  
- **Dominio**: limitar fanout efectivo a `K` bloques por fila (block‑sparse).

### 2) Config/flags
- `--proc-sparse=on`, `--proc-fanout=64`, `--proc-base-scale=0.05`, `--proc-watermark=4096`, `--proc-shed-eps=1e-3`, `--proc-block=32`.  
- Scope: `--proc-target=mlp|attn|both`.

### 3) Integración
- **Preproc/offline**: convertir matriz densa en (seed base + lista de deltas fuertes por fila, bloqueados). Guardar heap top‑K por fila.  
- **Runtime kernel**: para cada fila/bloque solicitado, reconstruir `weight = base_hash + delta` solo de candidatos; aplicar mask block‑sparse a GEMM.  
- **Queue shedding**: similar a `applyShedding`: si deltas > watermark, descartar por |Δ| o eps.

### 4) Pseudocódigo
```cpp
// offline: build deltas
for row in W:
  heap = topk(|w|, fanout)
  store delta_list[row] = heap

// runtime: reconstruct block-sparse matvec
acc = 0
for (block in selected_blocks(row)):      // ≤ fanout
  w_block = base_hash(row, block) * base_scale
  if delta exists: w_block += delta
  acc += dot(w_block, x_block)

// apply shedding on delta_list if len > watermark
trim_small(delta_list[row], shed_eps, watermark)
```

### 5) Aceptación
- Memoria ≤50% de densa en capas objetivo; speedup CPU ≥10% con Δppl ≤2% en micro‑bench.  
- Reconstrucción determinista (hash estable) y reproducible.

### 6) Roadmap
1. Infra hash base + builder de deltas top‑K por bloque.  
2. Rutina de shedding (soft eps + hard cap).  
3. Kernel block‑sparse que reconstruye en caliente; bench.  
4. Toggle runtime y métricas (fanout efectivo, trims).

### 7) Riesgos
-,hash noise: elegir base_scale pequeño; permitir `base_scale=0` para ablar.  
- Overhead de reconstrucción > ahorro: usar bloques fijos y vectorizar.  
- Calidad: si fanout muy bajo, sube ppl; exponer fanout por capa.

### Compatibilidad e interacciones
- **Backend de pesos exclusivo por capa:** no combinar con ternario/TT/StateCells en la misma capa en primera iteración.  
- Sinergia con `--kwta` y `--event-driven`: los bloques a reconstruir deben ser `bloques_proc ∩ máscara_kwta/event` para no pedir bloques inexistentes.  
- En atención ventana (`--win-attn`), definir fanout por ventana/head para que el patrón procedural respete la estructura.

### 8) Commits sugeridos
- `proc-sparse: hash base + delta builder (offline)`  
- `proc-sparse: shedding + fanout masks`  
- `kernels: block-sparse matvec via procedural recon`  
- `runtime: flags/metrics for procedural sparse mode`
