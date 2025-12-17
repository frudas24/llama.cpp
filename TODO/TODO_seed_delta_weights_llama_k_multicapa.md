# TODO — SeedΔ: mecanismo K multicapa (`--policy`) + gating + autotune (builder)

Este documento define un mecanismo **data-aware y seguro** para escalar SeedΔ a *muchas capas* sin colapsar calidad (loops tipo “now…”) usando:

- **Policy JSON** para defaults + overrides por capa + overrides por tensor (`ffn_gate`, `ffn_up`, `ffn_down`).
- **Gating post-fit** por tensor para decidir si SeedΔ es aceptable.
- **Autotune de K/bloque** (opcional) para encontrar el mínimo presupuesto que pasa gating.
- **Strip-dense por tensor**: solo strippear cuando existe reemplazo SeedΔ aprobado.

> Fuente de verdad (paths reales en este repo):
> - Builder: `tools/seeddelta-build/seeddelta-build.cpp` (`./build/bin/llama-seeddelta-build --help`)
> - Runtime: `src/llama-seeddelta.{h,cpp}` + flags `--seeddelta/--seeddelta-gap` en `common/arg.cpp`.

---

## 0) Estado actual (realidad observada)

- El builder ya soporta: `--K-*`, `--scheme block|coo`, `--block`, `--row-scale`, `--imatrix`, `--strip-dense`, `--report-json`, y emite métricas por tensor incluyendo `cos_mean_x_w`/`cos_p05_x_w` (cuando hay `--eval-x`).
- Runtime soporta GGUF con `seeddelta.strip_dense=true` (stubs para pesos densos faltantes) y ejecución end-to-end con `--seeddelta`.
- El scheduler/offload ya fue estabilizado fijando el custom op SeedΔ a CPU para evitar `cur_backend_id == -1` en configuraciones “raras”.

### Señal clave (Qwen2.5 7B)

- Al intentar 8–20 con presets agresivos (p.ej. block16/K64) el modelo **colapsa** en greedy (`temp=0 top-k=1`) y el `cos_mean_x_w` era bajo (~0.1–0.2).
- Con sweep por capa (block32, K_gate/up=256, K_down=512, imatrix), **solo** capas altas (19–20) pasan umbrales razonables y un híbrido (solo 19–20) se mantiene estable.

Esto evidencia que necesitamos **policy per-tensor/per-layer** + **gating duro** para escalar sin “zombificar” el modelo.

---

## 1) Objetivo funcional

Agregar a `llama-seeddelta-build`:

- `--policy <path.json>` para resolver parámetros por **capa/tensor**.
- Gating post-fit por tensor que controle:
  - si se **emite** SeedΔ para ese tensor
  - si se **strippea** el peso denso (solo si el tensor pasa gating)
- Autotune opcional para buscar **K mínimo** (y/o block) que pase gating.
- Report JSON extendido: policy usada + resolved + gating + attempts + decisión final por tensor.

**No-objetivo:** “arreglar” un tensor/capa intratable. Si no pasa gating, se queda denso; el win viene de cubrir el subconjunto que sí tolera aproximación.

---

## 2) Qué problema arregla (y qué no)

### Sí arregla

- Evitar colapsos catastróficos cuando se aplica SeedΔ a rangos grandes:
  - si un tensor tiene `cos_mean_x_w` bajo, el builder lo deja denso y/o no lo strippea.
- Permitir modelos híbridos reproducibles sin scripts manuales:
  - “solo capas 19–20”
  - “solo `ffn_down` en 8–20”
  - “capas altas agresivas, medias conservadoras”

### No arregla por sí solo

- Limitaciones de expresividad de la base (`xor_circulant`/Hadamard stack) o de `perm_trials/depth`.
- “Loss-on-loss”: construir SeedΔ sobre Q4 puede limitar la calidad alcanzable vs partir de F16/Q8.
- Data-aware incompleto: imatrix diagonal y dataset no representativo puede sesgar qué se considera “importante”.

---

## 3) Gating (criterio de aceptación) — recomendado

El gating debe priorizar **métricas funcionales** (con activaciones), porque es lo que correlaciona mejor con estabilidad:

- Primario: `cos_mean_x_w` (y **cola**: `cos_p05_x_w` o `cos_p10_x_w` si existe).
- Fallback si no hay `x_w`: `cos_mean_w` (menos confiable) o `cos_mean_x`.

### Por qué NO gatear con `rel_l2` como hard rule

En nuestros reportes hay tensores “malos” con `rel_l2_mean` cerca de 1 pero `cos` cerca de 0. En alta dimensión, **cos≈0** implica dirección casi aleatoria aunque la norma parezca “bien”.

Recomendación: `rel_l2_*` puede quedar como **señal secundaria** (warning/score), no como criterio duro hasta aclarar definiciones.

### Umbrales “safety-first” (para greedy `temp=0 top-k=1`)

Valores iniciales (ajustables por modelo):

- `ffn_gate`, `ffn_up`:
  - `cos_mean_x_w >= 0.45`
  - `cos_p05_x_w  >= 0.30`
- `ffn_down`:
  - `cos_mean_x_w >= 0.25`
  - `cos_p05_x_w  >= 0.15`

> Nota: para apilar muchas capas, la cola (p05/p10) suele predecir mejor “no colapsar” que solo el mean.

---

## 4) Policy JSON (schema propuesto)

Diseño: **defaults** + **ranges** (en orden) + **layer exacto** + **tensor override**.

### 4.1 Ejemplo mínimo (orientado a Qwen2.5 7B)

```json
{
  "version": 1,
  "defaults": {
    "enable": true,
    "block": 32,
    "K": { "gate": 256, "up": 256, "down": 512 },
    "strip": true,
    "gating": {
      "metric": "cos_xw",
      "min_mean": { "gate": 0.45, "up": 0.45, "down": 0.25 },
      "min_p05":  { "gate": 0.30, "up": 0.30, "down": 0.15 }
    },
    "autotune": {
      "enabled": false,
      "schedule_gate_up":  [128, 256, 384, 512],
      "schedule_down":     [256, 512, 768, 1024],
      "max_iters": 4
    }
  },
  "ranges": [
    {
      "layers": "15-20",
      "autotune": { "enabled": true }
    }
  ],
  "layers": {
    "8": { "enable": false },
    "19": { "enable": true },
    "20": { "enable": true }
  }
}
```

### 4.2 Semántica de merge (determinista)

`defaults` → aplica cada `ranges[]` que contenga la capa (en orden) → aplica `layers[L]` → aplica `layers[L].tensors[T]`.

### 4.3 IDs de tensor (v1)

- `ffn_gate`
- `ffn_up`
- `ffn_down`

Extensiones futuras: `attn_q`, `attn_k`, `attn_v`, `attn_o`, `tok_embd`, `output` (Fase 5).

---

## 5) Integración en el builder (puntos de hook reales)

Archivo objetivo: `tools/seeddelta-build/seeddelta-build.cpp`

Conceptualmente, por cada tensor `blk.N.ffn_{gate,up,down}.weight` el builder ya hace:

1) fit base (`--base-*`)  
2) fit residual (scheme block/coo, K, row_scale)  
3) eval (cols / eval_x)  
4) write GGUF + JSON

El mecanismo policy/gating/autotune se inserta alrededor de ese flujo por tensor:

- `resolve(layer, tensor)` ⇒ `{enable, block, K*, strip, gating, autotune}`
- si `enable=false` ⇒ no emitir SeedΔ y **no strip** ese tensor
- si `autotune.enabled` ⇒ probar schedule de K (y/o block) hasta pasar gating
- si falla gating ⇒ **no emitir SeedΔ** y **no strip** ese tensor
- si pasa gating ⇒ emitir SeedΔ; si `strip=true` ⇒ strip del denso para ese tensor

**Regla crítica:** `--strip-dense` global debe convertirse en “strip **solo si** este tensor tiene reemplazo SeedΔ aprobado”.

---

## 6) Autotune (K mínimo que pasa)

Objetivo: evitar “subir K a lo loco”. Para cada tensor:

- probar K en un schedule creciente
- parar en el primer K que pasa gating
- si ninguno pasa: registrar best-effort (para diagnóstico) pero dejar denso

Prioridad práctica (dado el comportamiento observado):

- autotune primero en `ffn_gate`/`ffn_up` (son los más frágiles)
- luego `ffn_down` (a veces tolera más, pero también puede requerir K alto)

---

## 7) Report JSON (extensión compatible)

El `--report-json` actual ya contiene `weights[]` con métricas. Se propone añadir:

- `build.policy_file`, `build.policy_hash`, `build.policy_version`
- `layers[N].tensors[T].resolved` (block, K, strip)
- `layers[N].tensors[T].gating` (metric, thresholds, pass/fail, picked mean/p05)
- `layers[N].tensors[T].autotune.attempts[]` (K, metrics, pass, seconds)
- `layers[N].tensors[T].decision` (`emit_delta`, `strip_applied`, `reason`)

Esto hace reproducibles los híbridos sin depender de logs.

---

## 8) Workflow recomendado (modelo nuevo)

1) Generar imatrix con dataset representativo del modelo (chat + idioma + código si aplica).  
2) Ejecutar builder con `--policy` y `autotune.enabled=true` en un rango acotado (p.ej. 15–20).  
3) Confirmar estabilidad con batería greedy (varios prompts) usando `--seeddelta --no-repack --ignore-eos`.  
4) Solo después, medir PPL (chunks pequeños) y throughput/RSS.

---

## 9) Backlog (transferido y ajustado desde `TODO_seed_delta_weights_llama.md`)

### Builder: policy/gating/autotune (Fase 4, prioridad alta)

- [ ] Implementar `--policy` en `tools/seeddelta-build/seeddelta-build.cpp` (parse + merge + resolver).
- [ ] Implementar gating por tensor usando `cos_mean_x_w` + `cos_p05_x_w` (fallbacks si faltan).
- [ ] Cambiar `--strip-dense` a comportamiento “por tensor”: solo strip si el tensor pasó gating y hay reemplazo.
- [ ] Implementar autotune de K por tensor (schedule configurable).
- [ ] Volcar resolved/gating/autotune/decision al `--report-json`.
- [ ] Añadir `policy.example.qwen7b.json` y `policy.example.gemma.json` en `tools/seeddelta-build/policies/` (o `calibration/` si preferimos).

### Robustez runtime (pendiente real)

- [ ] Nan-Guardian/fallback: si el op produce NaN/Inf o si el tensor está marcado “unsafe”, fallback claro.
- [ ] “Unsafe tensor” metadata: si builder falla gating, marcar explícitamente (para auditar sin abrir JSON).

### Perf / RAM (sigue siendo el cuello)

- [ ] Optimizar `W0x` (cache por token/batch; vectorizar base; reducir overhead en base+residual).
- [ ] Reducir overhead de índices/layout (u16 contiguo, mejor packing por bloque).
- [ ] Clarificar repack vs RSS: tener medición consistente (time -v + logs) y explicar picos vs steady.

### Calidad / data-aware “de verdad”

- [ ] Mejorar dataset para imatrix por modelo (no depender de `gemma_calibration.txt` para todo).
- [ ] (Opcional) capturar activaciones reales por tensor (`X`) y optimizar `||WX - ŴX||` más allá de imatrix diagonal.
- [ ] Explorar construir SeedΔ desde F16/Q8 como fuente (evitar “loss-on-loss” de Q4).

### Expandir targets (Fase 5)

- [ ] Probar policy/gating en QKV/O con budgets conservadores (solo si gating indica que hay margen).
- [ ] Embeddings/LM head solo con gating ultra estricto.

---

## 10) Nota práctica (lo que ya sabemos por Qwen)

- La estrategia “todo el rango 8–20 con un K fijo” es arriesgada.
- Un policy/autotune **no es un lujo**: es la única forma de escalar de forma segura sin horas de prueba manual.
- El resultado esperado inicialmente es híbrido: pocas capas/tensores pasan. Eso sigue siendo win si reduce RAM sin degradar.

