# TODO — Camino ternario/int8 (BitAgg) para CPU en llama.cpp

> **Objetivo:** añadir un modo Q2‑ternary inspirado en Synapse: pesos en `{‑1,0,1}` + `gamma` por bloque y activaciones int8 con `alpha` por muestra; acumular en int32 y reescalar `(alpha*gamma/127)` para bajar memoria y multiplies en CPU sin romper los modos q4/q5.

---

## Índice
1) Mat/algoritmo
2) Flags/config
3) Integración por áreas
4) Pseudocódigo
5) Criterios de aceptación
6) Roadmap corto
7) Riesgos/mitigación
8) Commits sugeridos

---

### 1) Mat/algoritmo
- Umbral ternario: `lambda = 0.7 * mean(|w|)` por bloque (fila o tile de 32/64).
- Peso cuantizado: `q = sign(w)*1 si |w|>lambda else 0`; `gamma = mean(|w| | q!=0)` (fallback a mean abs).
- Activación cuantizada: `alpha = mean(|x|)` de (x,prev,ctx) o vector de entrada; `xq = clamp(round(x/alpha*127),-127,127)`.
- Salida: `z = (alpha*gamma/127) * dot(q, xq)`; acumulación en int32.

### 2) Flags/config
- CLI: `--q2-ternary` / `--q2-ternary-layers=decoder,ffn` / `--q2-ternary-block=32|64`.
- Quant params: `lambda_k=0.7` (tunable), `store_gamma=fp16` (o fp32), packing 2‑bit.

### 3) Integración por áreas
- **Quant offline**: extiende quantizador para emitir 2‑bit weights + gamma per row/block; guarda metadatos en tensor.
- **Kernel MLP/Attn**: ruta AVX2/AVX512 que expande 2‑bit → int8, carga xq int8, usa `_mm256_maddubs_epi16` / `_mm512_maddubs_epi16` y reduce a int32; multiplica escalar final.
- **FallBack**: si no hay AVX2, usar ruta escalar compacta.
- **Mixto**: permitir capas selectivas (ej. MLP solo) para pruebas A/B.

### 4) Pseudocódigo
```cpp
// quant offline por bloque B
for row in W:
  lambda = 0.7 * mean_abs(row)
  gamma_sum = 0; nz=0
  for i in 0..B-1:
    if fabs(w[i]) > lambda: q[i]=sign(w[i]); gamma_sum+=fabs(w[i]); nz++
    else q[i]=0
  gamma = (nz>0) ? gamma_sum/nz : mean_abs(row)
  pack2bit(q, out_bits[row])
  store_gamma[row] = float16(gamma)

// kernel
alpha = mean_abs(x_block)
xq = quant_int8(x_block, alpha)
acc = 0
for each packed byte pb in out_bits:
  q_vec = unpack2bit(pb)            // {-1,0,1} int8
  x_vec = load(xq_chunk)
  acc += dot_int8(q_vec, x_vec)     // maddubs + hadd
z = (alpha * gamma / 127.0f) * acc
```

### 5) Criterios de aceptación
- Bench CPU AVX2/AVX512: +≥15% tkn/s en MLP vs q4_0 en CPU‑solo, con Δppl ≤2%.
- No regresiones en GPUs ni en otros modos quant.
- Ruta escalar functional; rutas intrínsecas pasan tests.

### 6) Roadmap corto
1. Quant offline + packing 2‑bit + gamma (fp16).
2. Kernel AVX2/AVX512 + fallback escalar; tests de exactitud vs float.
3. Hook de flags/runtime y selección por capa; micro‑bench.
4. llama-bench y reporte Δppl/Δtkns; ajusta lambda/gamma si hace falta.

### 7) Riesgos/mitigación
- **Densidad alta** → poco beneficio: ajustar `lambda_k` o block size.
- **Desbalance gamma** por fila: opcional gamma por bloque pequeño (32) en vez de fila completa.
- **Overhead de unpack**: usar LUT 4‑pesos/byte y desenrollado fijo.

### Compatibilidad e interacciones
- **Backend de pesos exclusivo por capa:** no combinar en la misma capa con `--tt`, `--proc-sparse` o `--statecells`; elegir uno y validar `gap`.
- **Componible con sparsidad runtime:** si `--kwta` o `--event-driven` están activos, usar sus máscaras para decidir qué bloques pasan por este kernel (evita doble top‑K).
- **StatePack/SDR:** `state-pack` puede usarse solo como máscara para saltar bloques con activación nula antes del dot ternario; evitar aplicar un segundo “dot aproximado popcount” encima del dot ternario en la misma operación.
- Instrumentar reescalados/sumas con `--nan-guardian` para fallback seguro.

### 8) Commits sugeridos
- `quant: add q2 ternary pack (2bit+gamma)`
- `kernels: add avx2/512 ternary int8 path + fallback`
- `runtime: flags for q2-ternary per layer + bench harness`
