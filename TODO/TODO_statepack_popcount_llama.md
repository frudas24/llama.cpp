# TODO — Pack binario/ternario + popcount para KV/MLP

> **Objetivo:** portar `state_pack` (binary/ternary + popcount) para reducir ancho de banda y acelerar dot aproximados en CPU (KV-cache y MLP), con fallback denso intacto.

---

## Índice
1) Concepto/params  
2) Config/flags  
3) Integración (KV, MLP)  
4) Pseudocódigo  
5) Métricas y aceptación  
6) Roadmap  
7) Riesgos/anti‑patrones  
8) Commits sugeridos

---

### 1) Concepto/params
- `epsilon`: umbral de activación → bit set si `|x|>eps`.  
- Modos: `binary` (1 bitmap) o `ternary` (pos/neg).  
- `popcnt`: usar VPOPCNTDQ si disponible, fallback Kernighan.  
- `density`: monitor para decidir si activa camino approx.

### 2) Config/flags
- CLI/env: `--state-pack=binary|ternary|none`, `--state-pack-eps=1e-4`, `--state-pack-popcnt`.  
- Per‑capa: permitir overrides (ej. solo KV-cache).

### 3) Integración
- **KV-cache**: al guardar K/V, también generar bitmap + `gamma = mean_abs(chunk)`; para atención approx usar popcount(K_mask ∧ Q_mask) y reescalar por gamma antes de mezclar con score denso opcional.  
- **MLP**: usar bitmap de activación previa para skip de columnas nulas; modo agresivo opcional: si pesos empaquetados (ver ternario) usar popcount de máscara ∧ pesos binarios/ternarios.  
- **Control**: si `density>ρ` (ej. >0.15) desactivar pack en runtime.

### 4) Pseudocódigo
```cpp
// build bitmap
for i in 0..N-1:
  if fabs(x[i]) > eps: bitset[i]=1; gamma_sum += fabs(x[i])
gamma = (active>0) ? gamma_sum/active : eps

// score aproximado (binary)
overlap = popcount(bitset_q & bitset_k)
score_approx = overlap * gamma_q * gamma_k

// ternary
pos = popcount(q_pos & k_pos)
neg = popcount(q_neg & k_neg)
mix = pos - neg
score_approx = mix * gamma_q * gamma_k
```

### 5) Métricas y aceptación
- Exportar: `pack.enabled`, `pack.mode`, `pack.density`, `pack.active`, `pack.popcnt`.  
- Aceptar si: `density<0.10` en capas packeadas, speedup ≥10% en CPU KV o MLP, Δppl/quality ≤2%.

### 6) Roadmap
1. Infra de packing (binary/ternary) + popcount intrínseco + fallback.  
2. Hook KV-cache (build bitmap in parallel) + flag gating por densidad.  
3. Modo skip columnas MLP usando bitmask; medir speedup.  
4. Bench/QA y toggle dinámico por densidad.

### 7) Riesgos/anti‑patrones
- Densidad alta → nulo beneficio: ajustar `eps` o auto‑disable.  
- Popcount vectorial no disponible: fallback podría ser más lento; proteger con CPU feature check.  
- Escalado gamma pobre: cap gamma y usar EMA para evitar explosiones.

### 8) Commits sugeridos
- `pack: add binary/ternary bitmap + popcnt helpers`  
- `kv: optional bitmap cache + approx score path`  
- `mlp: sparse column skip via activation bitmap`  
- `runtime: flags + metrics for pack density`
