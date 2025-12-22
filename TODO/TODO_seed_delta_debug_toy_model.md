# TODO - Debug SeedΔ con modelos controlados (A+B)

> Objetivo: aislar errores matematicos y de acumulacion por capas usando
> (A) matrices sinteticas predecibles y (B) un tiny transformer entrenado.
> La meta es explicar "donde se rompe" antes de gastar tiempo en modelos grandes.

---

## Teoria corta (por que hacemos esto)

- En modelos grandes, el error se mezcla entre capas y es dificil atribuir la causa.
- El pipeline SeedΔ tiene dos fuentes de error: aproximacion local (W0+Δ) y
  acumulacion por profundidad.
- Un entorno controlado permite medir el error esperado y comparar contra lo observado.

Esperamos encontrar:
- En A: errores 100% atribuibles a matematica/implementacion.
- En B: puntos de colapso por acumulacion, gating insuficiente o K mal calibrado.

---

## Plan A+B (combinado)

- A) Validar exactitud numerica del harness con matrices controladas (sin entrenamiento).
- B) Medir sensibilidad/stacking en tiny transformer con dataset determinista.
- C) Consolidar heuristicas (capas/K) antes de escalar a modelos reales.

## Progreso

- [x] Crear harness A inicial (`llama-seeddelta-toy`) para matrices sinteticas y top-K residual.
- [x] Agregar modo base (W0) en el harness A para probar residual vs base.
- [x] Definir dataset y config exacta para el tiny transformer (B).
- [x] Script de dataset y entrenamiento inicial para B.
- [x] Tokenizer con byte_fallback para evitar fallos de tokenizacion en llama-cli.

### A) Modelo controlado sin entrenamiento

Objetivo:
- Verificar exactitud de W0+Δ en casos donde la respuesta es conocida.

Diseno:
- Construir matrices W con estructura simple: identidad, diagonal, low-rank,
  circulante y bloques.
- Generar W0 y Δ que deberian aproximar W con error conocido.

Pruebas y expectativas:
- A1: W=I, W0=I, Δ=0 => salida exacta (error ~ 0).
- A2: W=diag(s), W0=diag(s), Δ=0 => salida exacta.
- A3: W=U V^T (rank-r), W0=low-rank, Δ=0 => error controlado segun r.
- A4: W=circulante, W0=base circulante, Δ=0 => salida exacta.
- A5: W=I, W0=I, Δ=COO top-K => salida exacta si Δ=0; si Δ no nula, error predecible.

Criterio de exito:
- Error numerico coincide con el esperado en >= 95% de casos.

### B) Tiny transformer entrenado

Objetivo:
- Medir acumulacion de error y detectar capas "fragiles" con comportamiento real.

Diseno:
- 6-12 capas, n_embd 128-256, n_ff 512-1024.
- Dataset determinista (copy, suma, brackets) para respuestas verificables.
- Exportar a GGUF y correr SeedΔ por capa.
  - Propuesta v1:
    - 8 capas, n_embd=192, n_ff=768, n_heads=6, vocab pequeno.
    - Dataset: copy (input==output), suma de 2 enteros, balanceo de parentesis.
    - Entrenamiento corto con seed fijo, export a GGUF fp16.

Pruebas y expectativas:
- B1: baseline sin SeedΔ => metricas estables (ppl/tok/s).
- B2: SeedΔ en 1 capa => degradacion pequena y localizada.
- B3: 2-4 capas separadas => degradacion moderada pero controlable.
- B4: capas intercaladas => buscar el primer "cliff".
- B5: policy con gating estricto => evitar colapso (greedy PASS).

Criterio de exito:
- Identificar el primer punto de colapso y correlacionarlo con metricas de gating.

---

## Entregables

- Un script A para generar matrices controladas y evaluar W x vs (W0+Δ) x.
- Un tiny model entrenado + GGUF + script de eval SeedΔ.
- Reportes por prueba: error numerico, PPL, greedy pack, stack_cost.
- Tabla de "capas fragiles" y thresholds recomendados.
- [x] `seeddelta-layer-scan.py` + `seeddelta-autogate.py` (scripts ya ejecutados en Gemma 4B) que:
  * generan `layer_sensitivity_scan.json` con métricas por capa/tensor,
  * aplican reglas (ranking, min_gap, preferencias) para crear `policy.autogen.json`,
  * soportan `--rank-metric` configurables y eval incremental.

### Schema `layer_sensitivity_scan.json` (nuevo)

Cada run debe emitir un JSON con estos campos principales:

```json
{
  "version": 1,
  "model": "<base gguf>",
  "layers": {
    "L": {
      "tensors": {
        "ffn_gate": {
          "S_rel_l2": 0.123,
          "cos_mean": 0.95,
          "cos_p05": 0.88,
          "cos_mean_x_w": 0.72,
          "cos_p05_x_w": 0.56,
          "log_norm_ratio": -0.12,
          "stack_cost": 3.4
        },
        "ffn_up": { … },
        "ffn_down": { … }
      },
      "S_layer": 0.12,      // agregado (mean/max)
      "S_layer_rank": 42.1  // número usado para ordenar
    }
  },
  "meta": {
    "scan_k": 32,
    "seed": 1234,
    "scheme": "coo"
  }
}
```

- `S_rel_l2` mide la sensibilidad relativa (norma perturbed / norma base) y sirve de señal de ranking.
- `cos_mean` / `cos_p05` (y sus variantes `_x_w`) son las métricas funcionales que el gating final utilizará para decidir si un tensor puede strippear.
- `S_layer` se calcula como el `mean` o `max` de los `S_{L,T}` y alimenta el ordenamiento del autogate.
- `stack_cost` y `log_norm_ratio` quedan disponibles para correlacionar con los cliff detectados.

### Cómo se alinea este feedback con el TODO

1. **Auto-Gating v2** ya está en marcha: el scan reemplaza la “intuición de pares/ímpares” por métricas repetibles (`S_{L,T}`) y el autogate ya ordena/filtra usando `min_gap`, `prefer`, `max_layers` y `rank_metric`.
2. **Cosine + rel_l2**: el schema captura ambos, lo que permite que el autogate use `rank_metric=S_layer` para priorizar capas seguras y luego aplicar las reglas fuertes (`cos_mean_x_w ≥ α`, `cos_p05_x_w ≥ β`) antes de activar una capa/tensor.
3. **Política híbrida**: con los scores por tensor podemos experimentar solo con `ffn_down` o solo `gate/up`, exactamente como se sugiere, sin tocar toda la capa.

### Próximo paso claro

- [x] Implementar el filtrado por tensor: el autogate ahora puede marcar `ffn_down` activo pero `gate/up` densos si `cos_*` no cumple.
- [x] Documentar thresholds `α_T`, `β_T` y `min_gap` usados para mantener seguridad.
  - Defaults autogate: `cos_mean_x_w >= 0.7`, `cos_p05_x_w >= 0.5`, `min_gap=1`.
- [x] Añadir una rutina pequeña que valide el schema + decide qué tensores pasan el gate. Por ejemplo:

```python
from pathlib import Path
import json

def validate_scan(path: Path):
    scan = json.loads(path.read_text())
    for layer, data in scan["layers"].items():
        assert "tensors" in data
        for tensor, metrics in data["tensors"].items():
            assert metrics["cos_mean_x_w"] >= 0
            assert metrics["S_rel_l2"] >= 0
        assert "S_layer" in data

def decide_policy(scan_path: Path, cos_thresh=0.7, cos_p05_thresh=0.5):
    scan = json.loads(scan_path.read_text())
    selected = []
    for layer, data in sorted(scan["layers"].items(), key=lambda kv: kv[1]["S_layer"]):
        tensors = data["tensors"]
        allow = [
            tensor
            for tensor, metrics in tensors.items()
            if metrics["cos_mean_x_w"] >= cos_thresh and metrics["cos_p05_x_w"] >= cos_p05_thresh
        ]
        if allow:
            selected.append({"layer": layer, "keep": allow})
    return selected
```

Rutina integrada en `scripts/seeddelta-autogate.py` como validación + filtrado por tensor antes de emitir `policy.autogen.json`.

### Artefactos (remote)

- `calibration/tiny_toy_data/` dataset jsonl.
- `calibration/tiny_toy_model_bf/` modelo HF + tokenizer con byte_fallback.
- `calibration/tiny_toy_model_bf/tiny.gguf` (F16) listo para pruebas.
- `calibration/tiny_toy_model_bf_1k/` modelo 1k steps (mejor output en suma simple).
- `calibration/tiny_toy_model_bf_1k/tiny.gguf` (F16) para pruebas SeedΔ.

### Autogate con filtrado cos (Gemma 4B Q8, remoto)

- Scan: `calibration/layer_scan_gemma4b_q8_10_19_v3/layer_sensitivity_scan.json`.
- Autogate: `calibration/autogate_gemma4b_q8_10_19_cosgate_v1/policy.autogen.json`.
- Policy: capas `12,14,16,18` con **solo `ffn_gate` activo** (gate/up/down filtrado por cos).
- PPL base 15.0808 → SeedΔ 13.9295 (delta -7.63%), greedy pack PASS (20/20).

### Autogate con filtrado cos (Gemma 4B F16, remoto)

- Scan: `calibration/layer_scan_gemma4b_f16_10_19_v3/layer_sensitivity_scan.json`.
- Autogate: `calibration/autogate_gemma4b_f16_10_19_cosgate_v1/policy.autogen.json`.
- Policy: capas `12,14,16,18` con **solo `ffn_gate` activo**.
- PPL base 15.0469 → SeedΔ 13.9357 (delta -7.38%), greedy pack PASS (20/20).

### Nota (pequena victoria)

- Descubrimiento: el filtrado por cos (`cos_mean_x_w` + `cos_p05_x_w`) produce una policy "segura" por tensor.
- Hecho: autogate habilito **solo `ffn_gate`** en capas `12,14,16,18` y desactivo `ffn_up/down`.
- Logrado: PPL bajo en Q8 y F16 (delta ~ -7.6% y -7.4%) con greedy PASS, confirmando estabilidad entre precisiones.

### Resultados B (tiny, 1k steps)

- Baseline PPL (val.txt): ~8789.55.
- SeedΔ layer0 (K=32 gate/up, down off): PPL ~8192.06, greedy pack PASS (6/6).
- SeedΔ even (0,2,4,6): PPL ~10048.08, greedy pack PASS (6/6).
- SeedΔ odd (1,3,5,7): PPL ~5465.42, greedy pack PASS (6/6).
- Nota: el greedy pack es heuristico; revisar outputs si hay dudas.

### Sweep por capa (K=32 gate/up, down off, coo, base on)

PPL base ~8789.55 en todos los runs.

- L0: 8192.06 (PASS)
- L1: 7301.93 (PASS)
- L2: 8737.10 (PASS)
- L3: 9939.09 (PASS)
- L4: 8312.01 (PASS)
- L5: 7065.78 (PASS)
- L6: 7716.13 (PASS)
- L7: 8117.02 (PASS)

### Combos 2 capas (K=32 gate/up, down off, coo, base on)

PPL base ~6015.29 (ctx=512) en estos runs.

- L1+L5: 3992.98 (PASS)
- L1+L6: 4090.03 (PASS)
- L5+L6: 4357.64 (PASS)

### Combo 3 capas (K=32 gate/up, down off, coo, base on)

PPL base ~6015.29 (ctx=512).

- L1+L5+L6: 3456.17 (PASS)

### Sweep K (combo L1+L5+L6, gate/up, down off, coo, base on)

PPL base ~6015.29 (ctx=512).

- K=16: 3226.41 (PASS)
- K=32: 3456.17 (PASS)
- K=64: 4034.95 (PASS)

### ffn_down on (combo L1+L5+L6, K=32, coo, base on)

PPL base ~6015.29 (ctx=512).

- down enabled: 3421.65 (PASS)

### K16 con mas capas (coo, base on)

PPL base ~6015.29 (ctx=512).

- L1+L3+L5+L6: 3073.42 (PASS)
- L0+L1+L3+L5+L6: 5450.90 (PASS)
- L1+L3+L5+L6 con down solo en L5: 3174.59 (PASS)

### K16 alterno (coo, base on)

PPL base ~6015.29 (ctx=512).

- L1+L2+L5+L6: 3708.13 (PASS)
- L0+L2+L4+L6 (pares): 15229.64 (PASS)

### K16 impares (coo, base on)

PPL base ~6015.29 (ctx=512).

- L1+L3+L5+L7: 3296.31 (PASS)

### Mix intercalado + sweep fino (coo, base on)

PPL base ~6015.29 (ctx=512).

- L0+L3+L5+L7 (mix): 7873.60 (PASS)
- L1+L3+L5+L6, K=12: 3095.68 (PASS)
- L1+L3+L5+L6, K=24: 3485.64 (PASS)

### K sweep fino (coo, base on)

PPL base ~6015.29 (ctx=512).

- L1+L3+L5+L6, K=8: 3082.56 (PASS)
- L1+L3+L5+L6, K=10: 3177.23 (PASS)
- L1+L3+L5+L6, K=14: 3080.58 (PASS)

### K sweep extremo (coo, base on)

PPL base ~6015.29 (ctx=512).

- L1+L3+L5+L6, K=6: 3160.24 (PASS)
- L1+L3+L5+L6, K=18: 3204.16 (PASS)

### Tiny 2k steps (nuevo modelo)

Modelo: `calibration/tiny_toy_model_bf_2k/tiny.gguf` (F16).

- PPL base ~18338.35 (ctx=512)
- SeedΔ L1+L3+L5+L6, K=16: 7000.89 (PASS)

### Tiny 2k steps (sweep K y pares/impares)

PPL base ~18338.35 (ctx=512).

- L1+L3+L5+L6, K=12: 7340.87 (PASS)
- L1+L3+L5+L6, K=14: 7267.77 (PASS)
- Pares (0,2,4,6), K=16: 27560.64 (PASS)
- Impares (1,3,5,7), K=16: 8049.34 (PASS)

### Tiny 2k steps (impares K y mix)

PPL base ~18338.35 (ctx=512).

- Impares (1,3,5,7), K=12: 7734.58 (PASS)
- Impares (1,3,5,7), K=14: 7666.95 (PASS)
- Mix (1,2,5,7), K=16: 9833.72 (PASS)

### Tiny 2k steps (ctx=256)

PPL base ~26081.12 (ctx=256).

- L1+L3+L5+L6, K=12: 10427.29 (PASS)
- L1+L3+L5+L6, K=14: 10385.10 (PASS)
- L1+L3+L5+L6, K=16: 9990.37 (PASS)

### Consolidado (tiny)

- Mejor set estable: L1+L3+L5+L6 con K en 8-16 (PPL ~3080-3096 en 1k; ~7-10k en 2k).
- Paridad importa: pares (0,2,4,6) degrada fuerte; impares mejor que pares, pero peor que L1+L3+L5+L6.
- K fuera de 8-16 tiende a empeorar (K=24/64).
- `ffn_down` no aporta mejora clara; mantener apagado por defecto.

### Heuristica propuesta (para modelos reales)

- [ ] Seleccionar capas candidatas por delta de PPL single-layer (mejoras o menor degradacion).
- [ ] Evitar sets solo pares; priorizar impares o mezcla con sesgo a L1/L3/L5/L6.
- [ ] Grid chico de K en {8,12,16}; elegir el mejor que pase greedy.
- [ ] Mantener `ffn_down` off; habilitar solo si el eval local de esa capa supera un umbral (cos mean >= 0.60).

---

## Notas operativas

- Mantener dataset y seeds fijos para reproducibilidad.
- Usar el mismo harness de logs (report.json + greedy pack).
- No mezclar con modelos grandes hasta cerrar A y B.

## Auto-Gating v2 - Seleccion automatica de capas basada en sensibilidad + stacking

### Motivacion

Hemos confirmado que SeedΔ puede mantener PPL/greedy cuando se aplica en capas "seguras", pero es **extremadamente sensible a donde** se aplica. Ademas, la estabilidad **no es aditiva**: dos capas que individualmente pasan pueden fallar juntas (stacking). Necesitamos un mecanismo que **mida** sensibilidad por capa y que tambien capture **interacciones**.

---

## 1) Definiciones matematicas

Sea una capa $L$ (bloque transformer) y sea $x$ la activacion de entrada al FFN (post-norm del bloque), muestreada de un batch de calibracion.

Denotemos:

* $FFN_{\\text{dense}}^{(L)}(x)$: salida del FFN de la capa $L$ usando pesos densos (baseline).
* $FFN_{\\text{seed}}^{(L)}(x)$: salida del FFN de la capa $L$ sustituyendo **solo el tensor objetivo** (gate/up/down) por su aproximacion SeedΔ (base+Δ o delta-only), dejando lo demas denso.

### 1.1 Sensibilidad por capa (metrica principal)

Definimos la sensibilidad relativa como:

$$
S_L = \\mathbb{E}_x\\left[\\frac{\\left\\lVert FFN_{\\text{dense}}^{(L)}(x) - FFN_{\\text{seed}}^{(L)}(x)\\right\\rVert_2}{\\left\\lVert FFN_{\\text{dense}}^{(L)}(x)\\right\\rVert_2 + \\epsilon}\\right]
$$

donde $\\epsilon$ evita division por cero (ej. $1e{-}8$).

### 1.2 Metricas robustas (colas y direccion)

Ademas de $S_L$, registramos metricas robustas para detectar drift en colas:

* Coseno (direccion) sobre salidas del FFN:
  $$
  \\cos_L(x) = \\frac{\\langle y, \\hat y \\rangle}{\\lVert y \\rVert_2 \\lVert \\hat y \\rVert_2 + \\epsilon}
  $$
  donde $y = FFN_{\\text{dense}}^{(L)}(x)$, $\\hat y = FFN_{\\text{seed}}^{(L)}(x)$.

* Log-ratio de norma (energia):
  $$
  r_L(x) = \\log\\left(\\frac{\\lVert \\hat y \\rVert_2 + \\epsilon}{\\lVert y \\rVert_2 + \\epsilon}\\right)
  $$

De estas, guardamos percentiles:

* $\\cos_{\\text{p05}}$ (conservador)
* $S_{\\text{p95}}$ o $L2_{\\text{p95}}$
* $r_{\\text{p95}}$ (evita explosiones sutiles)

---

## 2) Clasificacion de capas: "Estructural" vs "Redundante"

Para un umbral $\\tau$ (o percentil), la regla base es:

* Si $S_L > \\tau$  ⇒ capa "Estructural": **NO tocar** (mantener densa).
* Si $S_L \\le \\tau$ ⇒ capa "Redundante": **candidata** a SeedΔ.

**Nota:** $\\tau$ no debe ser universal. Preferible usar:

* seleccion por percentil (ej. "mejor 30% por menor $S_L$"),
* o seleccion por budget (escoger capas hasta alcanzar MB objetivo con minima degradacion).

---

## 3) Restricciones de seguridad (baratas y efectivas)

### 3.1 No consecutivas

Por defecto, prohibir comprimir capas consecutivas:
$$
L_{i+1} \\neq L_i + 1
$$
Esto fuerza espacio para "recuperacion" del residual.

### 3.2 Gap preferido

Preferir gap >= 2 si el budget lo permite (reduccion de riesgo de stacking).

---

## 4) El punto clave: stacking no aditivo ⇒ necesitamos fase de interacciones

La sensibilidad $S_L$ es **marginal** (capa sola). Para capturar colapsos tipo "10+11", hacemos construccion incremental con rollback y registro de pares prohibidos.

### 4.1 Constructor incremental (greedy-safe)

Sea un conjunto activo $A$ de capas seleccionadas. Iteramos:

1. Elegir siguiente capa $c$ del ranking (menor $S_c$).
2. Probar $A' = A \\cup \\{c\\}$ respetando "no consecutivas".
3. Evaluar aceptacion con bateria minima:

   * greedy pack: PASS
   * DeltaPPL <= presupuesto (ej. <= +5% en tiny o configurable)
   * metricas robustas: $\\cos_{p05}$ >= umbral y $r_{p95}$ <= umbral
4. Si PASS ⇒ aceptar $A \\leftarrow A'$
5. Si FAIL ⇒ rollback y registrar "forbidden interaction":

$$
\\text{forbidden}(A, c) = \\text{true}
$$

Simplificacion inicial: registrar forbidden **pairwise**:

* forbidden_pair: $(c, \\ell)$ para $\\ell \\in A$ (si el fallo aparece al anadir $c$).

---

## 5) Entregables (archivos) y formato minimo

### 5.1 Scan por capa

* `layer_sensitivity_scan.json`
  Contiene por layer y por sublayer (gate/up/down):

  * `S_mean`, `S_p95`
  * `cos_mean`, `cos_p05`
  * `log_norm_ratio_mean`, `log_norm_ratio_p95`
  * `base_used`, `scheme`, `K_fixed`
  * `eval_x_n`, `out_subsample_n`, `seed`
  * script: `scripts/seeddelta-layer-scan.py`

### 5.2 Interacciones

* `forbidden_pairs.json`
  Lista de pares (i,j) que rompen greedy/umbral cuando se activan juntos.

### 5.3 Policy autogenerada

* `policy.autogen.<model>.json`
  Incluye:

  * layers seleccionadas
  * K por subcapa (v0: fijo; v1: por capa segun sensibilidad)
  * strip settings (respetando CLI o declarados explicitamente)
  * script: `scripts/seeddelta-autogate.py`

### 5.4 Report de comparacion

* `autogating_report.md`
  Tabla:

  * baseline (PPL/greedy/RSS/tok_s)
  * policy autogen
  * manual best-known (si existe)

---

## 6) Plan de accion (checklist)

* [x] Implementar `layer_sensitivity_scan` (script/harness): recorre capa por capa, aplica SeedΔ temporalmente con $K_{\\text{fixed}}$ y produce `layer_sensitivity_scan.json`.
* [x] Definir $K_{\\text{fixed}}$ inicial (v0): recomendado 12-16 (o 32 si quieres mas senal), scheme=COO para gate/up, down opcional.
* [x] Definir dataset de calibracion `eval_x` determinista: mismo seed, mismo N, mismo ctx.
* [ ] Definir umbral $\\tau$ v0:

  * opcion A: percentil (ej. top 30% menor S)
  * opcion B: budget MB (escoger capas hasta X MB ahorrados)
* [ ] Implementar "no consecutivas" + "gap preferido".
* [x] Implementar constructor incremental con rollback + registrar `forbidden_pairs.json`.
* [x] Generar `policy.autogen.<model>.json` y correr bateria: greedy pack + PPL + RSS + tok/s.
* [ ] Comparar contra heuristic/manual: redescubre sets tipo $\\{1,3,5,6\\}$ o encuentra algo mejor?
* [ ] Integrar output en TODO principal: link a `scan.json`, `forbidden_pairs.json`, `policy.autogen`, `autogating_report.md`.

---

## 7) Criterios de exito (v0)

* Greedy pack: **PASS 0 flags**.
* DeltaPPL: tiny <= +5%, modelos reales <= +0.1% (configurable).
* RSS: reduccion medible cuando strip aplica (reportar "weights RSS" si se puede separar).
* tok/s: puede caer (aceptado), pero debe estar documentado.

## 8) Refinamiento opcional

Definir dos sensibilidades y aceptar solo si ambas estan bajo umbral:

1. $S_L^{FFN}$ (salida FFN)
2. $S_L^{Block}$ (salida del bloque despues del residual)
