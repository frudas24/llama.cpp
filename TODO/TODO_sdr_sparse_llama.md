# TODO — SDR / sparsidad aproximada para rutas CPU

> **Objetivo:** explorar representaciones dispersas tipo SDR (bitsets + popcount) para decodificación/retrieval aproximado y para saltar GEMM completo en CPU cuando la precisión lo permita.

---

## Índice
1) Idea y params  
2) Config/flags  
3) Integración (salida / atención ligera)  
4) Pseudocódigo  
5) Aceptación  
6) Roadmap  
7) Riesgos  
8) Commits sugeridos

---

### 1) Idea y params
- Hash/permutación por bloque para activar ~1–5% de bits (`active_bits`).  
- Dot aproximado = `popcount(sdr_x & sdr_w)` reescalado por norma media.  
- Densidad y correlación con logits densos son monitores clave.

### 2) Config/flags
- `--sdr=on|off`, `--sdr-active-frac=0.02`, `--sdr-bits=2048`, `--sdr-hash=splitmix`.  
- Scope: `--sdr-target=lm_head|attn_light`.

### 3) Integración
- **LM head aproximada:** construir SDR de activación previa, puntuar vocab con SDR de centroides/pesos comprimidos; mezclar con logits densos (blend) o usar como filtro top‑K.  
- **Atención ligera:** para capas “light”, usar SDR de Q/K y popcount para scores aproximados, luego refinar top‑K con cálculo denso.  
- **Fallback:** si `density>0.1` o correlación < umbral, desactivar en runtime.

### 4) Pseudocódigo
```cpp
// build SDR
indices = hash_topk(x, active_frac * bits)   // determinista por bloque
sdr = bitset(bits); set_bits(sdr, indices)

// dot aproximado
score = popcount(sdr & sdr_weight) * norm_scale

// mezcla
logits_dense = W * x      // opcional
logits = blend * score + (1-blend) * logits_dense
```

### 5) Aceptación
- Speedup perceptible en CPU (≥10% en ruta objetivo) con ΔEM/QA ≤2% en micro‑bench.  
- Correlación (R) SDR vs logits densos ≥0.9 en muestras controladas.

### 6) Roadmap
1. Builder SDR (hash/permutación determinista) + popcount helper.  
2. LM head aproximada (filter top‑K o blend) + métricas de correlación/densidad.  
3. Atención ligera con refine top‑K.  
4. Bench + toggle automático por densidad/correlación.

### 7) Riesgos
- Hash pobre → colisiones y pérdida de info: usar splitmix + permutación posicional.  
- Densidad variable → beneficio incierto: clamp densidad y auto‑off.  
- Mezcla con logits densos puede duplicar costo: usar como filtro top‑K primero.

### Compatibilidad e interacciones
- Compartir utilidades de bitset/popcount con `state-pack` (ver `TODO_statepack_popcount_llama.md`).  
- Para **atención**, no acumular aproximaciones: si una capa usa `--win-attn` o `--event-driven` en atención, desactivar `--sdr-target=attn_light` en esa capa (SDR puede seguir en LM head).  
- En **LM head**, SDR es un backend alternativo de scoring; no mezclar con otros backends de LM head (ternario/TT/statecells) salvo modo “filtro top‑K + refine denso”.

### 8) Commits sugeridos
- `sdr: add sparse bitset builder + popcount helpers`  
- `lm_head: optional sdr scorer + blend`  
- `attn: light sdr scorer + dense refine`
