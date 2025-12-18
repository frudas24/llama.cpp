# TODO — SeedΔ: mecanismo K multicapa (`--policy`) + gating + autotune (builder)

> Nota de pivote (2025-12-18): todo el backlog de optimización fina
> (perf `W0x`, multi-precisión, activaciones reales, targets extra como QKV/O
> y embeddings) se sigue rastreando ahora en
> `TODO_seed_delta_weights_k_layer_sublayer.md`. Este documento queda
> centrado en el builder/policy/gating/autotune y en los resultados de
> experiments multicapa.

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

### 3.2) Stack-safety (capas/tensores acumulados)

Los umbrales deben **endurecerse automáticamente** según la cantidad de tensores/capas aprobados para evitar error acumulado:

- Definir una función `stack_budget = (#tensores aprobados)` y aplicar multiplicadores conservadores.
- Ejemplo inicial:
  - Si `stack_budget <= 2`: usar thresholds base (arriba).
  - Si `3 <= stack_budget <= 8`: exigir `cos_mean_x_w` +0.05 y `cos_p05_x_w` +0.05 para gate/up.
  - Si `stack_budget > 8`: exigir `cos_mean_x_w >= 0.60` y `cos_p05_x_w >= 0.45` para gate/up; `ffn_down` sube a `0.35/0.25` respectivamente.
- Registrar los thresholds efectivos en el report (`decision.thresholds_used`) para auditar qué tan estricto fue el gating en cada caso.

El objetivo es que una política pueda declarar “quiero tocar capas 8–26” y que el builder rechace automáticamente aquellas que no alcanzan el umbral endurecido para mantener greedy estable.

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
- `build.policy_hash` (hash del JSON; actualmente `fnv1a64:<hex>`), y (si se puede) `build.git_commit` / version string del builder

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
- [x] Implementar autotune de K por tensor (schedule configurable).
- [x] Volcar resolved/gating/autotune/decision al `--report-json`.
- [x] Añadir `policy.example.qwen7b.json` y `policy.example.gemma.json` en `tools/seeddelta-build/policies/` (o `calibration/` si preferimos).
  - (Opcional) exponer gating/autotune “global” vía flags (`--autotune-k-*`, `--gating-*`) para arrancar sin policy y luego migrar.
- [x] Resolver robusto de tensor→kind (nombres reales por arquitectura):
  - mapear `blk.N.ffn_{gate,up,down}.weight` a `ffn_gate/up/down`
  - si un modelo no tiene ese tensor, skip + dejar rastro en report (no abortar silencioso)
- [x] Iteración segura sobre modelos ya SeedΔ:
  - decidir comportamiento por defecto (skip vs overwrite) y exponer flags tipo `--skip-existing` / `--overwrite`
  - en modo policy, evitar “skipping” silencioso que invalida el resolved/gating/autotune
- [x] Harness reproducible para policy:
  - script nuevo `scripts/seeddelta-policy-eval.sh`: build con `--policy`, guardar report, correr smoke greedy (batería), PPL corto y logs opcionales de `/usr/bin/time -v`
- [x] Test mínimo del resolver/merge:
  - `llama-seeddelta-build --policy-self-test` valida precedencia (CLI/defaults/ranges/layer/tensor) y strict/lenient
- [x] Export de policy desde run/report:
  - `llama-seeddelta-build --policy-export out.json` escribe policy canónica (strict-schema) desde decisiones finales
  - `scripts/seeddelta-report-to-policy.py --report report.json --out policy.json` convierte report→policy offline (sin rebuild)

### Robustez runtime (pendiente real)

- [x] Nan-Guardian/fallback: si el op produce NaN/Inf, reemplazar por un valor seguro:
  - best-effort fallback denso si `w_ref` es accesible en host
  - si no hay fallback denso disponible (p.ej. offload), clamp a `0.0f`
  - log a `stderr` solo una vez (evita spam)
- [x] “Unsafe tensor” metadata: si builder falla gating, marcar explícitamente (para auditar sin abrir JSON).
  - se escribe por tensor en KV (v1): `seeddelta.blk.N.<kind>.gating_pass`, `...enabled`, `...strip_dense`, `...K`.

### Perf / RAM, data-aware y targets extra

El backlog detallado de optimización de `W0x` / layout, mejoras data-aware (activaciones reales,
multi-precisión como fuente) y expansión a QKV/O, embeddings y LM head se mantiene ahora en:

- `TODO_seed_delta_weights_k_layer_sublayer.md`

Dentro de este documento mantenemos solo el estado del builder/policy/gating/autotune y los
resultados de los experimentos actuales; el diseño fino de K por capa/subcapa y las optimizaciones
de rendimiento/quality cross-model viven en el nuevo TODO.

### Modelos y pruebas cruzadas

- [x] Inventario de modelos locales (`~/.cache/llama.cpp/*.gguf`) con tamaño, quant y tags SeedΔ (script helper `scripts/model-inventory.sh`).
- [x] Bajar al menos un modelo “llama-like” en Q8/F16 (p.ej. Mistral 7B o Llama 3.1 8B) para validar que SeedΔ escala cuando el target es limpio. (Descargado `mistral-7b-instruct-v0.2.Q8_0.gguf` + `calibration/mistral7b.imatrix.gguf`; corrido policy-eval con varias policies y exportada una policy estable down-only 14–23).

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

---

## 12) Plan de ataque inmediato

1. **Stack-safe policy freeze**
   - Exportar (`--policy-export`) las políticas que ya sabemos estables (Gemma 1B/4B, Qwen 19–20, Mistral 7B down-only 14–23) y versionarlas.
   - Añadir multiplicadores stack-safety en el builder y revalidar que Qwen siga estable (debe seguir aprobando solo 19–20, pero ahora con thresholds explícitos) y que Mistral conserve el comportamiento de la policy exportada `tools/seeddelta-build/policies/policy.mistral7b.down14-23.exported.json`.
2. **Cross-precision baseline**
   - Para al menos un modelo llama-like (ej. Mistral 7B) comparar Q8 vs FP16: correr `scripts/seeddelta-policy-eval.sh` con la misma policy sobre ambas precisiones y medir cuántas capas/tensores pasan y cómo escala PPL.
   - Documentar en `calibration/` (JSON + eval) la diferencia entre construir SeedΔ desde Q8/FP16 vs Q4 para un mismo rango.
3. **Model inventory + small-model smoke**
   - Escribir script `scripts/model-inventory.sh` que liste gguf disponibles con quant/size y tags SeedΔ.
   - Ejecutar smoke en Gemma 1B/4B con las nuevas policies para asegurarnos de que el pipeline sigue estable cuando cambiamos thresholds.
4. **Data-aware improvements**
   - Generar nuevos imatrix específicos por modelo/dominio (no más `gemma_calibration` para Qwen).
   - Investigar captura ligera de activaciones reales para reemplazar el proxy `eval_x`.

Entrega esperada tras este plan: políticas congeladas + logs de Q8/F16 + inventario de modelos + documentación de stack_budget reales.

---

## 13) Notas de investigación 2025-12-18 (Mistral 7B Q8 y stack-safety)

### 13.1) Setup

- Modelo: `mistral-7b-instruct-v0.2.Q8_0.gguf` (`general.name = mistralai_mistral-7b-instruct-v0.2`, file_type=Q8_0, ~7.17 GiB).
- Texto de evaluación: `wikitext-2-raw/wiki.test.raw` (`scripts/get-wikitext-2.sh`).
- Imatrix: `calibration/mistral7b.imatrix.gguf` (8 chunks, ctx=512, t=8, `--no-ppl`).
- Builder: esquema block32, base xor_circulant (L=4096, B=4, depth=2, samples=2048, perm_trials=4), `--imatrix` on, `--eval-cols 64`, `--eval-x 16`.
- Runtime: CPU-only (REPACK on, Flash Attention auto), `t=8`, `ctx=512`, `chunks=4`.

Baseline PPL (sin SeedΔ):

- `llama-perplexity -m mistral-7b-instruct-v0.2.Q8_0.gguf -f wiki.test.raw --chunks 4 -t 8 -c 512 --no-warmup`
- Resultado: `PPL ≈ 7.3049 ± 0.63`.

### 13.2) Policy “base” 14–23 (gate+up+down, agresiva)

- Policy: `tools/seeddelta-build/policies/policy.mistral7b.base.json`.
- Rango: capas 14–23, block32, K_gate/up=256–640, K_down=512–1536, `cos_x_w` con thresholds medios.
- Run: `scripts/seeddelta-policy-eval.sh --base mistral.Q8 --policy policy.mistral7b.base.json --layers 14-23 --imatrix mistral7b.imatrix.gguf --text wiki.test.raw --threads 8 --ctx 512 --chunks 4 --no-strip-dense --outdir calibration/seeddelta-policy-eval-mistral-base-14-23`.
- Report:
  - `stack_pass_gate_up = 3`, `stack_pass_down = 10`.
  - `emit = true` en 13 tensores (combinación de gate/up/down en 14–23).
- PPL:
  - PPL SeedΔ: `≈ 40.94` (vs 7.30 base).
  - Greedy: texto incoherente, repetitivo y “raro” (fuerte distorsión semántica).
- Conclusión:
  - Aunque cada tensor pasa umbrales de cos_x_w “razonables” de forma aislada, apilar ~13 tensores en 10 capas **rompe el modelo**.
  - Necesario introducir límites de stack-safety (umbral por tensor + límite de cantidad).

### 13.3) Policy down-only 14–23 (conservadora)

- Policy: `tools/seeddelta-build/policies/policy.mistral7b.abl.layers14-23.down-only.json` (solo `ffn_down` enabled, thresholds duros: `cos_x_w_mean >= 0.60`, `cos_x_w_p05 >= 0.45` para down).
- Rango: 14–23, base+residual como arriba, sin strip-dense.
- Run: `scripts/seeddelta-policy-eval.sh --base mistral.Q8 --policy policy.mistral7b.abl.layers14-23.down-only.json --layers 14-23 --imatrix mistral7b.imatrix.gguf --text wiki.test.raw --threads 8 --ctx 512 --chunks 4 --no-strip-dense --outdir calibration/seeddelta-policy-eval-mistral-down-14-23`.
- Report:
  - `stack_pass_gate_up = 0`, `stack_pass_down = 1`.
  - Solo `blk.23.ffn_down` pasa gating con `cos_mean_x_w ≈ 0.64`, `cos_p05_x_w ≈ 0.43`, `ops_ratio ≈ 0.11`.
  - `emit = true` únicamente para `layer=23, kind=ffn_down` con `block=32`, `K=1536`.
- PPL:
  - PPL SeedΔ: `≈ 7.58` (vs 7.30 base; degradación pequeña).
  - Greedy: respuestas coherentes (“Hola! I'm here to help…”, sin loops ni collapse), tok/s ≈ 4 en gen.
- Policy export + roundtrip:
  - Export: `--policy-export calibration/seeddelta-policy-eval-mistral-down-14-23-roundtrip/policy.exported.json`.
  - Roundtrip: rebuild con `--policy calibration/.../policy.exported.json --policy-strict`, `--eval-x 0`.
  - Resultado: `roundtrip decisions OK (30 tensors)`.
  - Policy canonicalizada: `tools/seeddelta-build/policies/policy.mistral7b.down14-23.exported.json` (defaults disabled; solo `layer 23 ffn_down` enabled con `block=32`, `K=1536`).
- Conclusión:
  - Para Mistral 7B Q8, una única capa/tensor (23 down) con cos_x_w alto produce un modelo estable con PPL aceptable.
  - La policy exportada representa una configuración “lo más agresiva posible sin romper” dentro de este rango bajo thresholds actuales.

### 13.4) Policies 19–23 y 19–20 (apilando pocos tensores)

- Policy stack-strict 19–23:
  - `tools/seeddelta-build/policies/policy.mistral7b.stack-strict.19-23.json`.
  - Más agresiva en 19–23 (gate/up+down) con thresholds duros (`cos_x_w_mean ≥ 0.60` gate/up, `≥ 0.50` down).
  - Run: `scripts/seeddelta-policy-eval.sh ... --policy policy.mistral7b.stack-strict.19-23.json --layers 19-23 --outdir calibration/seeddelta-policy-eval-mistral-strict-19-23`.
  - Report: `stack_pass_gate_up = 2`, `stack_pass_down = 3`, `emit = true` en 5 tensores totales.
  - PPL SeedΔ: `≈ 10.74` (degradación notable vs 7.30, pero sin colapso total); greedy coherente.
- Policy ablation 19–20:
  - `tools/seeddelta-build/policies/policy.mistral7b.abl.layers19-20.only.json`.
  - Solo capas 19–20, mismos thresholds fuertes.
  - Run: `scripts/seeddelta-policy-eval.sh ... --policy policy.mistral7b.abl.layers19-20.only.json --layers 19-20 --outdir calibration/seeddelta-policy-eval-mistral-abl-19-20`.
  - Report: `stack_pass_gate_up = 1`, `stack_pass_down = 2`, `emit = true` en 3 tensores.
  - PPL SeedΔ: `≈ 8.70`; greedy coherente.
- Conclusión:
  - Incluso con cos_x_w altos, apilar 5 tensores (19–23) ya sube PPL a ~10.7.
  - Apilar 3 tensores (19–20) degrada menos (PPL ~8.7).
  - Apilar 1 tensor (solo 23 down) es el punto actual de “mejor trade-off” (PPL ~7.6).

### 13.5) Lecciones para stack-safety y generalización

- Gating por tensor con `cos_x_w` y cola (p05) funciona como filtro de *estabilidad mínima*, pero:
  - no basta para garantizar PPL baja si permitimos apilar muchos tensores,
  - el “stack budget” (número de tensores emitidos) debe entrar explícitamente en la política (cap por tipo y cap global).
- Q4 vs Q8:
  - Qwen 2.5 7B Q4: casi ninguna capa media es stack-safe; solo 19–20 se dejan tocar, y con pocos tensores.
  - Mistral 7B Q8: hay más tensores “candidatos”, pero el budget de apilamiento sigue siendo pequeño (1–3 tensores conservadores).
- Política estable exportada:
  - `tools/seeddelta-build/policies/policy.mistral7b.down14-23.exported.json` captura una configuración “safety-first” comprobada en Mistral 7B Q8.
  - Sirve como plantilla de cómo debería verse un preset “universal estable” para modelos cuantizados: pocos tensores, thresholds altos, sin strip-dense y con roundtrip verificado.

Próximos pasos derivados:

- Endurecer stack-safety en el builder:
  - añadir caps explícitos (máximo N tensores emitidos por tipo y M totales) además de thresholds por tensor,
  - exponer estos caps en la policy (`stack_budget`) y reflejarlos en el report.
- Replicar el patrón “pocos tensores de alta calidad” en otros modelos (Gemma 3 4B, Devstral 24B) antes de perseguir compresión agresiva.

---

## 14) Notas de investigación 2025-12-18 (Kimiko Mistral 7B FP16 vs Q8 y harness PPL+greedy)

### 14.1) Conversión Kimiko FP16 → GGUF y baseline

- Modelo HF: `TheBloke/Kimiko-Mistral-7B-fp16` (base Mistral 7B, FP16).
- Conversión a GGUF F16:
  - Script: `convert_hf_to_gguf.py` con venv local (`.venv`), `torch`, `transformers`, `sentencepiece`.
  - Directorio snapshot HF: `/home/devgpt/.cache/huggingface/hub/models--TheBloke--Kimiko-Mistral-7B-fp16/snapshots/<commit>`, con `config.json`, `tokenizer.json`, `pytorch_model-0000{1,2}-of-00002.bin`, `pytorch_model.bin.index.json`.
  - Comando efectivo:
    - `python convert_hf_to_gguf.py --outfile /home/devgpt/models/Kimiko-Mistral-7B-fp16.gguf --outtype f16 <snapshot_dir>`
  - Resultado:
    - GGUF FP16: `/home/devgpt/models/Kimiko-Mistral-7B-fp16.gguf` (~13.5 GiB, `file_type = F16`, 7.24B params).
- Baseline PPL (wikitext-2, ctx=512, chunks=4, t=8):
  - `llama-perplexity -m Kimiko-Mistral-7B-fp16.gguf -f wiki.test.raw --chunks 4 -t 8 -c 512 --no-warmup`
  - PPL ≈ **6.09 ± 0.45**.
  - Host RAM ≈ 13.8 GiB modelo + 256 MiB KV + 109 MiB compute ≈ 14.2 GiB.
- Comparación con Mistral Q8 instruct:
  - Q8 baseline (Mistral instruct) PPL ≈ **7.30 ± 0.63** en la misma configuración.
  - FP16 (Kimiko) es claramente mejor en PPL (como cabe esperar de una FP16 fine-tuned).

### 14.2) Imatrix FP16 y policy down-only 14–23

- Imatrix FP16:
  - `llama-imatrix -m Kimiko-Mistral-7B-fp16.gguf -f wiki.test.raw -o calibration/kimiko_mistral7b_fp16.imatrix.gguf -t 8 -c 512 --no-ppl --chunks 8`
  - Ruta: `calibration/kimiko_mistral7b_fp16.imatrix.gguf` (~4.8 MiB).
- Policy aplicada (misma que en Q8):
  - `tools/seeddelta-build/policies/policy.mistral7b.abl.layers14-23.down-only.json`
    - Solo `ffn_down` en capas 14–23.
    - Thresholds fuertes: `cos_x_w_mean ≥ 0.60`, `cos_x_w_p05 ≥ 0.45` para down.
- Build FP16:
  - `scripts/seeddelta-policy-eval.sh --base Kimiko-Mistral-7B-fp16.gguf --policy policy.mistral7b.abl.layers14-23.down-only.json --layers 14-23 --imatrix calibration/kimiko_mistral7b_fp16.imatrix.gguf --text wiki.test.raw --threads 8 --ctx 512 --chunks 4 --no-strip-dense --outdir calibration/seeddelta-policy-eval-kimiko-fp16-down-14-23`
  - Output model: `calibration/seeddelta-policy-eval-kimiko-fp16-down-14-23/model_sd.gguf` (~14.5 GiB).
  - `report.json` muestra, igual que en Q8:
    - `stack_pass_gate_up = 0`, `stack_pass_down = 1`.
    - Solo `blk.23.ffn_down` con `emit = true`, `strip_dense = false`, `block=32`, `K=1536`, `gating_pass = true`.
- PPL FP16 con SeedΔ down-only 14–23:
  - `llama-perplexity -m model_sd.gguf -f wiki.test.raw --chunks 4 -t 8 -c 512 --no-warmup --seeddelta`
  - PPL ≈ **6.37 ± 0.48** (vs 6.09 base FP16).
  - Degradación muy similar a la observada en Q8 (`7.30 → ~7.58`).
  - Host RAM prácticamente igual a base (~14.2 GiB); down-only en una capa no reduce mucho footprint FP16.

### 14.3) Policy stack-strict 19–23 en FP16 (vs Q8)

- Policy usada: `tools/seeddelta-build/policies/policy.mistral7b.stack-strict.19-23.json` (la misma que en Mistral Q8).
  - Umbrales fuertes:
    - `cos_x_w_mean ≥ 0.60`, `cos_x_w_p05 ≥ 0.45` para gate/up.
    - `cos_x_w_mean ≥ 0.50`, `cos_x_w_p05 ≥ 0.35` para down.
  - Autotune habilitado en gate/up/down, `K` creciente por schedule.
- Build FP16:
  - `scripts/seeddelta-policy-eval.sh --base Kimiko-Mistral-7B-fp16.gguf --policy policy.mistral7b.stack-strict.19-23.json --layers 19-23 --imatrix calibration/kimiko_mistral7b_fp16.imatrix.gguf --text wiki.test.raw --threads 8 --ctx 512 --chunks 4 --no-strip-dense --outdir calibration/seeddelta-policy-eval-kimiko-fp16-strict-19-23`
  - Output: `calibration/seeddelta-policy-eval-kimiko-fp16-strict-19-23/model_sd.gguf`.
  - `report.json`:
    - `stack_pass_gate_up = 3`, `stack_pass_down = 2` (5 tensores emitidos en 19–23).
    - Similar al caso Q8 (Q8 tenía `stack_pass_gate_up = 2`, `stack_pass_down = 3`).
- PPL FP16 con stack-strict 19–23:
  - `llama-perplexity -m model_sd.gguf -f wiki.test.raw --chunks 4 -t 8 -c 512 --no-warmup --seeddelta`
  - PPL ≈ **8.44 ± 0.66**.
  - Comparación:
    - FP16 base: 6.09 → strict 19–23: 8.44 (Δ ≈ +2.35).
    - Q8 base (Mistral instruct): 7.30 → strict 19–23: ~10.7 (Δ ≈ +3.4).
  - Conclusión:
    - FP16 tolera algo mejor el stack de 5 tensores (sube menos la PPL que Q8), pero sigue siendo una degradación significativa.
    - El patrón “1–3 tensores ok, 5+ tensores ya duelen” se mantiene tanto en Q8 como en FP16.

### 14.4) Harness PPL + greedy (por qué Wikitext no basta)

- Wikitext(-2) es útil pero no suficiente:
  - **Sí sirve para**:
    - regresiones rápidas y comparables entre builds (score numérico estable),
    - detectar acumulación grave de error cuando apilar deltas hace explotar la PPL,
    - A/B consistente manteniendo `ctx`, `chunks`, `seed`, etc.
  - **No basta para** el objetivo de “no romper instruct/chat”:
    - modelos instruct (Qwen, Mistral/Kimiko) están ajustados a chat/instrucciones; puedes tener PPL aceptable pero respuestas zombis (loops, cambio de idioma, violación de instrucciones).
    - el síntoma real es funcional (greedy `temp=0, top-k=1`), no solo de PPL.
- Regla práctica:
  - Si un build sube PPL poco pero degrada greedy → **greedy manda** (máximo cuidado).
  - Si greedy parece bien pero PPL explota → alerta de degradación latente (aparecerá en prompts/dominios distintos).

### 14.5) Prompt pack minimalista para detección de “zombies”

Harness recomendado (además de Wikitext) para estabilidad instruct/chat en greedy:

- Settings:
  - `--temp 0 --top-k 1 --seed 1 --single-turn -n 128`
  - No usar `--ignore-eos` aquí (facilita loops); solo si queremos forzar.
- Prompt pack sugerido (20 prompts, mezcla es/en/código/instrucciones):
  - P01_greeting_es: saludo + pregunta en español, 1 frase cada uno.
  - P02_concise_bullets: overfitting en 3 viñetas, ≤12 palabras por viñeta.
  - P03_follow_instruction_single_word: responder solo “listo”.
  - P04_anti_loop_constraint: exactamente 2 oraciones, sin repetir ninguna palabra, tema “persistencia de caché”.
  - P05_translation_control: traducción al inglés de una frase en español.
  - P06_math_exact: 17*23, solo número.
  - P07_short_reasoning: 2 razones sobre repetición vs mezcla de idiomas.
  - P08_json_strict: JSON válido con `{"task":"", "lang":"es", "steps":[...]}`.
  - P09_code_go_snippet: snippet Go corto con backoff exponencial (≤25 líneas).
  - P10_code_explain: explicación de un comando `time -v ./llama-cli ...`.
  - P11_regex: regex para 3 repeticiones consecutivas de misma palabra (case-insensitive).
  - P12_sql: query Postgres para top-10 usuarios por eventos en 24h.
  - P13_yaml: YAML con threads/ctx/temp/top_k/seed.
  - P14_micro_summary: 1 oración sobre “policies evitan zombies, apilar rompe PPL”.
  - P15_spanish_only_guard: explicar “latencia” en 2 oraciones, solo español.
  - P16_programming_short: línea sobre diferencia stack vs heap (en español).
  - P17_error_style: mensaje de error corto para “policy JSON inválida: key desconocida”.
  - P18_list_no_repeat: 8 palabras distintas relacionadas con compresión.
  - P19_instruction_priority: “qué detecta antes zombie, PPL o greedy?” (≤20 palabras, español).
  - P20_exit: responder solo `/exit`.

Uso recomendado:

- Correr el pack en greedy para base y modelo SeedΔ y verificar:
  - ausencia de loops claros (“now now now…”, repeticiones largas),
  - respeto de idioma (prompts en español no se deriven a inglés sin razón),
  - obediencia de instrucciones duras (P03, P08, P20).
- Combinar este gate con PPL en Wikitext:
  - FP16 + down-only 23: PPL sube poco y greedy se mantiene sano → aceptable como preset estable.
  - FP16 + strict 19–23: PPL ~8.44 (sube mucho) → aunque greedy no colapse, es candidato a “preset agresivo solo para experimentos”.

Implementación práctica:

- [x] Archivo de prompts: `calibration/greedy_zombie_pack.txt` (20 prompts P01–P20_sentinel, texto igual al de esta sección).
- [x] Runner dedicado: `scripts/seeddelta-greedy-pack.sh`:
  - args: `--base BASE.gguf --sd SD.gguf [--pack FILE] [--outdir DIR]`.
  - fija `--temp 0 --top-k 1 --seed 1 --single-turn -n 128`.
  - genera outputs en `OUTDIR/{base,sd,diff}` y un resumen con:
    - `rep` (repetición máxima de palabra),
    - `drift` (proxy de ratio de ASCII en prompts que deberían ser solo español),
    - `STRICT_FAIL` para P03/P20 si no se respeta exactamente “listo” / “fin”.
  - imprime `RESULT: PASS` si `Prompts flagged = 0`, si no `RESULT: FAIL`.
- [x] Integración en `scripts/seeddelta-policy-eval.sh`:
  - flag: `--greedy-pack FILE`.
  - si se pasa, al final ejecuta `scripts/seeddelta-greedy-pack.sh` con el modelo base y el SeedΔ recién construido, dejando logs en:
    - `OUTDIR/greedy_pack/` y `OUTDIR/greedy_pack.log`.

Ejemplo de uso (remoto Mistral 7B Q8):

- Comando:
  - `./scripts/seeddelta-policy-eval.sh --base ../models/mistral-7b-instruct-v0.2.Q8_0.gguf --policy tools/seeddelta-build/policies/policy.mistral7b.abl.layers14-23.down-only.json --layers 14-23 --imatrix calibration/mistral7b.imatrix.gguf --text wikitext-2-raw/wiki.test.raw --threads 8 --ctx 512 --chunks 4 --no-strip-dense --greedy-pack calibration/greedy_zombie_pack.txt --outdir calibration/seeddelta-policy-eval-mistral-q8-down-14-23-greedy`
- Resultado greedy (SeedΔ) para “down-only 14–23, Q8”:
  - Muchos prompts con `drift ~0.98–1.00` en casos que deberían ser solo español.
  - P03 (`listo`) y P20 (`fin`) marcados como `STRICT_FAIL`.
  - Resumen: `Prompts flagged: 11/20`, `RESULT: FAIL`.
- Interpretación:
  - Aunque la PPL en Wikitext solo sube moderadamente (~7.30 → ~7.58), el comportamiento instruct/chat cambia de forma fuerte (drift a inglés + fallos de instrucciones duras).
  - El harness greedy confirma que este preset “down-only 14–23” en Mistral Q8 no es aceptable como policy estable; se mantiene como preset experimental.
