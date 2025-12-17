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

### 1.1) Modos de uso (sin duplicar código)

- **Modo global (default):** se usa el CLI actual (p.ej. `--K-gate/--K-up/--K-down`, `--block`, `--strip-dense`, etc.) de forma uniforme. Debe seguir funcionando igual que hoy.
- **Modo policy (opcional):** `--policy` define overrides por capa/tensor (y gating/autotune). Si un campo no está definido en policy, **hereda del modo global** (CLI).

Objetivo: que `--policy` sea un “patch” sobre el builder actual, no un segundo sistema paralelo.

### 1.2) Contrato de seguridad (strip-dense)

Invariantes (corner-case crítico):

- Si un tensor denso se elimina (*strip*), entonces **debe existir** SeedΔ para ese tensor y el runtime debe poder ejecutarla.
- Si `strip_dense=true` pero el gating falla ⇒ **KEEP DENSE** siempre.
- El builder **no debe producir** un GGUF donde falte el tensor denso *y* tampoco exista SeedΔ para ese tensor, salvo bajo un modo explícito “unsafe” (p.ej. `--force-strip-dense`) que debe quedar marcado en report/metadata como override.

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

### 3.1) Prerrequisitos de métricas (para evitar resultados ambiguos)

- Si el gating/autotune usa métricas `*_x*` ⇒ exigir `--eval-x > 0` (si no, error claro o fallback explícito).
- Si el gating/autotune usa métricas `*_w` ⇒ definir qué pasa sin `--imatrix`:
  - recomendado safety-first: error claro
  - alternativa (si se permite): fallback a métrica no ponderada (`*_x` o `*_w`→`*_x`) con warning fuerte y `decision.metric_used` en el report

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

> Nota: el bloque `defaults` en policy es opcional. En modo policy, el “baseline” es el CLI global; la policy solo sobreescribe lo que declare.

### 4.1 Keys canónicas (schema v1) + unknown keys

Para evitar typos y drift, el schema v1 debe declarar keys canónicas y comportamiento ante keys desconocidas:

- Keys canónicas (v1): `enabled`, `strip_dense`, `block`, `K: {gate, up, down}`, `gating: {...}`, `autotune: {...}`.
- Default recomendado:
  - modo **lenient**: ignora keys desconocidas, emite warning y lo registra en el report.
  - `--policy-strict`: keys desconocidas ⇒ error.

### 4.2 Ejemplo mínimo (orientado a Qwen2.5 7B)

```json
{
  "version": 1,
  "defaults": {
    "enabled": true,
    "block": 32,
    "K": { "gate": 256, "up": 256, "down": 512 },
    "strip_dense": true,
    "gating": {
      "metric": "cos_x_w",
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
    "8": { "enabled": false },
    "19": { "enabled": true },
    "20": { "enabled": true }
  }
}
```

### 4.3 Semántica de merge (determinista)

1) Baseline = **CLI global** (flags actuales).  
2) Si hay `--policy`: `defaults` → aplica cada `ranges[]` que contenga la capa (en orden) → aplica `layers[L]` → aplica `layers[L].tensors[T]`.

Regla de “último gana”: si varias reglas aplican y definen el mismo campo, la regla más específica (tensor > capa > rango > defaults) y/o la última en orden debe ganar (determinista).

Prioridad formal (para evitar ambigüedad):

1. CLI global
2. policy.defaults
3. policy.ranges (primero→último)
4. policy.layers[L]
5. policy.layers[L].tensors[T]
6. Si dos reglas tienen misma especificidad, **la última en el JSON gana**.

### 4.4 IDs de tensor (v1)

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

Nota de implementación (optimización): el fit de la base `W0` típicamente **no depende de K**, así que para autotune conviene computar/guardar la base una vez por tensor y re-emitir/re-evaluar el residual para distintos K (si el layout del builder lo permite).

Contrato autotune (perf + corrección):

- Autotune no debería refitear la base si no es necesario; debe re-encodear residual + re-evaluar.
- Si ningún K pasa gating:
  - se registran `attempts[]` completos (postmortem)
  - pero el tensor queda denso (no SeedΔ, no strip), salvo modo explícito “unsafe”.

---

## 7) Report JSON (extensión compatible)

El `--report-json` actual ya contiene `weights[]` con métricas. Se propone añadir:

- `build.policy_file`, `build.policy_hash`, `build.policy_version`
- `layers[N].tensors[T].resolved` (block, K, strip)
- `layers[N].tensors[T].gating` (metric, thresholds, pass/fail, picked mean/p05)
- `layers[N].tensors[T].autotune.attempts[]` (K, metrics, pass, seconds)
- `layers[N].tensors[T].decision` (`emit_delta`, `strip_applied`, `reason`)

Esto hace reproducibles los híbridos sin depender de logs.

Campos recomendados (para forense y reproducibilidad):

- `decision.metric_used` (p.ej. `cos_x_w` / `cos_x` / `cos_w`)
- `decision.thresholds_used` (snapshot de thresholds tras merge)
- `build.policy_hash` (sha256 del JSON) y (si se puede) `build.git_commit` / version string del builder

---

## 8) Workflow recomendado (modelo nuevo)

1) Generar imatrix con dataset representativo del modelo (chat + idioma + código si aplica).  
2) Ejecutar builder con `--policy` y `autotune.enabled=true` en un rango acotado (p.ej. 15–20).  
3) Confirmar estabilidad con batería greedy (varios prompts) usando `--seeddelta --no-repack --ignore-eos`.  
4) Solo después, medir PPL (chunks pequeños) y throughput/RSS.

---

## 9) Backlog (transferido y ajustado desde `TODO_seed_delta_weights_llama.md`)

### Builder: policy/gating/autotune (Fase 4, prioridad alta)

- [x] Implementar `--policy` en `tools/seeddelta-build/seeddelta-build.cpp` (parse + merge + resolver).
- [x] Añadir `--policy-strict` (error en keys desconocidas) para evitar typos silenciosos.
- [x] (Opcional) `--policy-dump-resolved` para imprimir cfg efectivo por capa/tensor (debug rápido).
- [x] Definir contrato de métricas para gating/autotune:
  - si la policy usa `*_x*` exigir `--eval-x > 0` (y fallar claro si no)
  - si la policy usa `*_x_w` definir qué pasa sin `--imatrix` (error vs fallback + warning)
- [x] Implementar gating por tensor usando `cos_mean_x_w` + `cos_p05_x_w` (fallbacks si faltan).
- [x] Cambiar `--strip-dense` a comportamiento “por tensor”: solo strip si el tensor pasó gating y hay reemplazo.
- [x] Aclarar/implementar semántica de `seeddelta.strip_dense` con strip parcial:
  - setear el KV global si **cualquier** tensor fue strippeado (para permitir stubs)
  - opcional: metadata adicional para auditar qué tensores fueron strippeados vs mantenidos densos
- [ ] Implementar autotune de K por tensor (schedule configurable).
- [x] Volcar resolved/gating/autotune/decision al `--report-json`.
- [ ] Añadir `policy.example.qwen7b.json` y `policy.example.gemma.json` en `tools/seeddelta-build/policies/` (o `calibration/` si preferimos).
  - (Opcional) exponer gating/autotune “global” vía flags (`--autotune-k-*`, `--gating-*`) para arrancar sin policy y luego migrar.
- [ ] Resolver robusto de tensor→kind (nombres reales por arquitectura):
  - mapear `blk.N.ffn_{gate,up,down}.weight` a `ffn_gate/up/down`
  - si un modelo no tiene ese tensor, skip + dejar rastro en report (no abortar silencioso)
- [x] Iteración segura sobre modelos ya SeedΔ:
  - decidir comportamiento por defecto (skip vs overwrite) y exponer flags tipo `--skip-existing` / `--overwrite`
  - en modo policy, evitar “skipping” silencioso que invalida el resolved/gating/autotune
- [ ] Harness reproducible para policy:
  - extender `scripts/seeddelta-eval.sh` o añadir script nuevo para: build con `--policy`, guardar report, correr smoke greedy (batería) y (opcional) PPL corto
- [ ] Test mínimo del resolver/merge:
  - al menos un “self-check” (p.ej. `--policy-dump-resolved` + golden esperado) para evitar regresiones en merge order/overrides
  - incluir casos de strip seguro: `strip_dense=true` + gating fail ⇒ no strip
  - incluir caso policy “disable all” ⇒ output == input (sin deltas, sin strip)

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

---

## 11) Pruebas obligatorias (plan mínimo)

- Policy/merge (unit-ish):
  - precedencia exacta (CLI vs defaults vs ranges vs layer vs tensor)
  - lenient vs strict (unknown keys)
- Builder smoke:
  - policy disable-all ⇒ no deltas, no strip
  - policy “solo 19–20” ⇒ reproduce híbrido estable (greedy)
  - strip solicitado pero gating fail ⇒ keep dense (verificar que el tensor denso existe)
- Runtime smoke:
  - cargar GGUF híbrido y correr greedy (`--seeddelta --no-repack --ignore-eos`) en 5 prompts (es/en/código corto)
