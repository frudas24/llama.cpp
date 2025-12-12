# TODO — Cache persistente/semántico de prompts para acelerar CPU

> **Objetivo:** adaptar la “memoria persistente + recuperación aproximada” (C03) a llama.cpp: guardar en disco sesiones/KV/outputs y reutilizarlos por match exacto o aproximado (subset/Jaccard) para saltar inferencia repetida en CPU.

---

## Índice
1) Idea y parámetros  
2) Config/flags  
3) Integración  
4) Pseudocódigo  
5) Métricas/aceptación  
6) Roadmap  
7) Riesgos  
8) Commits sugeridos

---

### 1) Idea y parámetros
- Al finalizar una respuesta, persistir:
  - `prompt_norm` (normalizado), `tokens_prompt`, `kv_cache` (o logits finales) y `answer`.  
- Recuperación:
  - `exact`: hash de `prompt_norm`.  
  - `subset/Jaccard`: n‑grams 2–3 y score `s∈[0,1]`.  
  - Si `s>threshold`, devolver respuesta y/o cargar KV para continuar sin recomputar prefijo.
- Control de tamaño: TTL + compactación top‑K por tópico/uso.

### 2) Config/flags
- `--cache-persist=on|off`  
- `--cache-file=./snapshot/llama_cache.jsonl`  
- `--cache-mode=exact|approx`  
- `--cache-threshold=0.9`  
- `--cache-max=10000`  
- `--cache-ttl-hours=72`

### 3) Integración
- Hook en CLI/server:
  - Antes de inferir: `Lookup(prompt)` → si hit, retorna o carga KV.  
  - Después de inferir: `Remember(prompt, kv, answer)`.  
- Formato: JSONL o binario compacto (kv en blobs separados).  
- No toca cuantizadores ni kernels.

### 4) Pseudocódigo
```cpp
prompt_norm = normalize(prompt)
if cache.enable:
  hit = cache.lookup(prompt_norm)
  if hit.score >= threshold:
     if hit.kv_exists: load_kv(hit.kv_path); resume_decode()
     else return hit.answer

answer, kv = run_inference(prompt)
cache.remember(prompt_norm, answer, kv)
cache.compact_if_needed()
```

### 5) Métricas/aceptación
- `cache.enabled`, `cache.size`, `cache.hit_exact`, `cache.hit_approx`, `cache.saved_ms`, `cache.load_ms`.  
- Aceptar si en workloads repetitivos reduce latencia total ≥30% sin falsos hits >1%.

### 6) Roadmap
1. Módulo cache (Save/Load/Lookup/Remember/Compact).  
2. Hook en CLI/server + flags.  
3. Persistencia KV opcional + blobs.  
4. Bench con prompts repetidos y con variantes ligeras.

### 7) Riesgos
- Falsos hits en approx: usar umbral alto y registrar `kind/score`.  
- I/O lento: batch save y blobs separados.  
- Cambios de modelo/quant: invalidar cache por hash de config.

### Compatibilidad e interacciones
- Ortogonal a kernels/quant: compatible con todas las optimizaciones; solo requiere invalidación por hash de modelo+quant+flags activas.  
- Si se guarda KV, respetar el formato/feature‑set de esa sesión (no cargar KV de una config distinta).

### 8) Commits sugeridos
- `cache: add persistent exact/approx prompt store`  
- `cli: lookup before infer + remember after infer`  
- `cache: optional kv blobs + compaction`
