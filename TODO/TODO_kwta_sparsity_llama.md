# TODO — Gating KWTA/Top‑K suave para sparsidad dinámica

> **Objetivo:** integrar una máscara KWTA (top‑K con temperatura y duty) para omitir columnas/tokens de bajo score en CPU, reduciendo GEMMs sin romper calidad.

---

## Índice
1) Esquema y params  
2) Flags/config  
3) Integración (MLP / Atención)  
4) Pseudocódigo  
5) Aceptación  
6) Roadmap  
7) Riesgos  
8) Commits sugeridos

---

### 1) Esquema y params
- `topK`: fracción o número de columnas/tokens a conservar.  
- `temp`: suaviza ranking (softmax); `duty`: aplica gating cada N pasos para evitar flicker.  
- Máscara binaria (1 pasa, 0 salta) aplicada a bloques fijos (16/32) para vectorizar.

### 2) Flags/config
- CLI: `--kwta=on`, `--kwta-k=0.4` (fracción) o `--kwta-k-abs=128`, `--kwta-temp=0.5`, `--kwta-duty=2`.  
- Scope: `--kwta-target=mlp|attn|both`.

### 3) Integración
- **MLP:** calcula score de columnas (norma de weight*act previa o activación media); aplica máscara por bloque antes de GEMM. Opcional renormalizar salida por `1/p_keep`.  
- **Atención:** filtrar queries/keys de baja energía; mantener top‑K por secuencia y reutilizar misma máscara para K/V en CPU.  
- **Metrics:** `kwta.keep_ratio`, `kwta.duty_hits`, `kwta.temp`.

### 4) Pseudocódigo
```cpp
// construir máscara
scores = l2_norm_per_block(x, block=32)
probs = softmax(scores/temp)
mask = select_topk(probs, K)         // devuelve bool per block
if (step % duty != 0) mask = 1

// aplicar en MLP
for each block b:
  if mask[b]==0: continue; // skip GEMM block
  y_block += W_block * x_block
// opcional renorm
y *= (float)(blocks_total) / (blocks_kept)
```

### 5) Aceptación
- Speedup CPU: ≥10% tkn/s en MLP/attn con `k≈0.4..0.6`, Δppl ≤2%.  
- Gating estable (sin flicker) con duty>1; métricas expuestas.

### 6) Roadmap
1. Infra de score + topK + duty; máscara por bloque.  
2. Hook MLP bloqueado (skip) + renorm opcional.  
3. Hook atención (filtra Q/K/V) + bench.  
4. Autotune simple: si Δppl alto, subir K o duty; si densidad baja, bajar K.

### 7) Riesgos
- Saltos de calidad por flicker: duty>1 y suavizado de probs.  
- Coste de ranking > ahorro: limitar a bloques y usar nth_element/partial sort.  
- Invalida vectorización si K alto variable: agrupar bloques contiguos.

### Compatibilidad e interacciones
- Compartir la misma partición de bloques con StatePack/Event‑driven/BitAgg/Proc‑sparse para que las máscaras coincidan (`block=32/64`).  
- Si `--event-driven` también está activo, usar KWTA como **prior de candidatos** y dejar que event‑driven aplique el presupuesto/top‑K final dentro de esa lista (no top‑K doble independiente).  
- En capas con sparsidad estructural (`--proc-sparse` o `--win-attn`), intersectar la máscara KWTA con los bloques realmente disponibles.  
- En capas TT/StateCells sin unidades claras de skip, dejar KWTA off inicialmente o aplicarla solo a nivel “capa completa”.

### 8) Commits sugeridos
- `kwta: add top-k+duty mask builder (CPU)`  
- `mlp: block-skip path using kwta mask`  
- `attn: optional kwta filter for q/k/v + metrics`
