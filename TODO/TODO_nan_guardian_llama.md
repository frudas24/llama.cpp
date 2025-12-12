# TODO — “Nan Guardian” para kernels low‑precision en llama.cpp

> **Objetivo:** replicar la arquitectura Nan‑Guardian (C13) como watchdog liviano para los caminos experimentales (ternario, pack+popcount, KWTA, TT, event‑driven) evitando NaN/Inf silenciosos y facilitando fallback seguro sin afectar hot‑path denso.

---

## Índice
1) Idea  
2) Config/flags  
3) Integración  
4) Pseudocódigo  
5) Métricas/aceptación  
6) Roadmap  
7) Riesgos  
8) Commits sugeridos

---

### 1) Idea
- API mínima: `nan_guard(val, meta)` que:
  - Detecta `!isfinite(val)` en puntos críticos de kernels low‑precision.  
  - Incrementa contador por dominio/capa y registra (file:line, kind).  
  - Opcionalmente clamp a 0 o dispara fallback al camino denso.
- **No muta por defecto:** la política de clamp/fallback vive arriba para no contaminar lógica.

### 2) Config/flags
- `--nan-guardian=on|off` (default on para modos experimentales).  
- `--nan-guardian-policy=log|clamp|fallback`.  
- `--nan-guardian-max=16` (máx logs por capa).

### 3) Integración
- Insertar guardias solo en rutas nuevas:
  - salida de dot ternario/int8  
  - scores popcount/SDR  
  - renormalizaciones KWTA/event‑driven  
  - contratos TT  
- Si política=fallback: al primer NaN por capa, marca la capa como “dense‑only” hasta reinicio.

### 4) Pseudocódigo
```cpp
inline bool nan_guard(float v, Meta m) {
  if (isfinite(v)) return false;
  metrics_inc("nan."+m.kind+"."+m.layer);
  if (logs[m.layer] < max_logs) log_nan(m, v);
  switch(policy){
    case CLAMP: v = 0; break;
    case FALLBACK: mark_dense_only(m.layer); break;
  }
  return true;
}

// ejemplo en kernel low-precision
z = compute_lowp(...)
if (nan_guard(z, {kind:"ternary", layer:i})) return dense_fallback(...)
```

### 5) Métricas/aceptación
- `nan.enabled`, `nan.count.{kind,layer}`, `nan.fallbacks`, `nan.last_meta`.  
- Aceptar si overhead <0.5% en kernels low‑precision y se detectan NaNs inyectados en tests.

### 6) Roadmap
1. Implementar util NanGuardian + métricas.  
2. Instrumentar kernels low‑precision existentes.  
3. Añadir tests que fuerzan NaN (escala 0, overflow) y verifican logs/fallback.  
4. Documentar knobs en README/TODOs CPU.

### 7) Riesgos
- Log spam: cap por capa y “first hit wins”.  
- Coste de branch: compilar guardias solo cuando modo low‑precision activo.

### Compatibilidad e interacciones
- Feature transversal: activarla automáticamente cuando cualquier modo low‑precision/approx esté on; no compite con otras optimizaciones.  
- Si policy=fallback marca la capa “dense‑only”, también debe desactivar KWTA/event‑driven/SDR/ternario/TT/procedural/statecells en esa capa para evitar estados mixtos.

### 8) Commits sugeridos
- `nan: add guardian util + metrics`  
- `kernels: instrument lowp paths with nan_guard`  
- `tests: nan injection + fallback validation`
