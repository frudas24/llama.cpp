# TODO — Factores TT/low-rank para MLP/Attn CPU

> **Objetivo:** portar TT‑cross/maxvol (TODO_N6) como modo opt‑in para factorizar pesos grandes (MLP y proyecciones de atención) en cores bajos, reduciendo memoria y flops en CPU con control de `gap` y fallback seguro.

---

## Índice
1) Idea y parámetros  
2) Config/flags  
3) Integración (MLP, Attn)  
4) Pseudocódigo  
5) Métricas y aceptación  
6) Roadmap  
7) Riesgos  
8) Commits sugeridos

---

### 1) Idea y parámetros
- Factorizar matrices W∈R^{m×n} en TT o en 2–3 factores low-rank (TT es más estructurado): `W ≈ G1 ⊗ G2 ⊗ ... ⊗ Gd` con rangos pequeños.  
- `rank_tt`: 2–8 (pequeño); `maxvol_k`: fibras/columnas usadas para TT‑cross.  
- Controlar `gap_tol`: si error vs denso > tol, fallback.

### 2) Config/flags
- `--tt=on|off`, `--tt-rank=4`, `--tt-maxvol-k=8`, `--tt-gap=0.02`, `--tt-target=mlp|attn|both`.  
- Guardar metadatos por capa (dims factorables, rangos).

### 3) Integración
- **MLP:** factorizar W1/W2 offline a TT cores; kernel realiza multiplicaciones secuenciales (d pasos) con bloques cache‑friendly.  
- **Atención:** factorizar proyecciones Q/K/V/O o solo O; opcional TT para MHA fused.  
- **Fallback:** si CPU sin ventaja o `gap` > tol, usa camino denso existente.

### 4) Pseudocódigo
```cpp
// offline TT-cross (bosquejo)
cores = tt_cross(W, rank_tt, maxvol_k, gap_tol)
store(cores, meta)

// runtime matvec con TT cores
vec y = x
for core in cores:           // G ∈ R^{r_{k-1}×n_k×r_k}
  y = contract(core, y)      // reshape y según n_k y multiplicar
// para MLP: y = act(y) y repetir para W2
```

### 5) Métricas y aceptación
- `tt.enabled`, `tt.rank`, `tt.gap`, `tt.speedup`, `tt.mem_saved`.  
- Aceptar si speedup CPU ≥10–15% en capas factoradas y Δppl ≤2%.

### 6) Roadmap
1. Infra TT‑cross (maxvol) + export/import cores por capa.  
2. Kernel matvec TT (AVX2/AVX512) + fallback denso.  
3. Hook MLP/Attn con flags; bench y gap check.  
4. Autotune simple: subir/bajar rank si `gap` pasa tol.

### 7) Riesgos
- Factores mal condicionados → gap alto: usar maxvol y normalizar.  
- Overhead si dims no factorizables bien: permitir disable per capa.  
- Batch pequeño: agrupar tokens para amortizar contratos.

### Compatibilidad e interacciones
- **Backend de pesos exclusivo por capa:** no combinar TT con `--q2-ternary`, `--proc-sparse` o `--statecells` sobre la misma matriz (salvo futura compresión en cascada).  
- Compatible con KWTA/Event‑driven/StatePack, pero estas máscaras solo aportan si el kernel TT expone bloques “skippeables”; de inicio aplicar gating solo a capas densas.  
- Instrumentar contratos TT con Nan‑Guardian para fallback si aparece NaN/Inf.

### 8) Commits sugeridos
- `tt: add tt-cross/maxvol builder + meta`  
- `kernels: add tt matvec path (avx2/512) + fallback`  
- `runtime: flags, metrics, gap-fallback for tt layers`
