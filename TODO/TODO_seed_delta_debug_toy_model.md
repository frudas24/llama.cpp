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

### Bitacora: tests de umbral (Q8, remoto)

- Prueba A (0.65 / 0.45):
  - Capas: 12,14,16,18
  - Mask: 12 gate-only; 14/16/18 gate+up; down off
  - PPL base 15.0808 → 15.5639 (delta +3.20%), greedy PASS, 49 new tensors
- Prueba B (0.65 / 0.50):
  - Capas: 12,14,16,18
  - Mask: igual a A (up sigue abierto)
  - PPL base 15.0808 → 15.5639 (delta +3.20%), greedy PASS, 49 new tensors
- Prueba up-strict (gate 0.7/0.5, up 0.85/0.70):
  - Capas: 12,14,16,18
  - Mask: gate-only en todas (up cerrado)
  - PPL base 15.0808 → 13.9295 (delta -7.63%), greedy PASS, 28 new tensors
  - Conclusión: `ffn_up` es el principal causante de degradacion en estas capas.

### Plan de ataque post-victoria (feedback consolidado)

Reglas base (congeladas por ahora):
- `ffn_up`: OFF por defecto (deny-by-default).
- `ffn_gate`: ON solo si pasa thresholds victoria (`cos_mean_x_w >= 0.70`, `cos_p05_x_w >= 0.50`).
- `ffn_down`: candidato, pero se prueba aislado de `up` y con thresholds mas estrictos.
  - En autogate: `ffn_up` solo entra si se pasa `--allow-up`.

Metricas de exito:
- greedy PASS (20/20)
- ΔPPL <= 0% ideal; aceptable <= +0.5% para "RAM mode"
- ahorro real: GGUF size y RSS (ctx pequeno 128/256 + ctx normal)
- tok/s puede degradar

Fase 1 - "marea creciente" (gate-only):
- [x] E1: autogate gate-only, rango 0-25, max_layers escalonado: 8, 16, 24 (Q8)
- [x] E1b: repetir gate-only con rango seguro (10-19), mismos thresholds
- [x] E2: repetir E1b en F16 (evitar 0-25)
- Entregable: tabla max_layers vs ΔPPL vs GGUF size vs RSS(ctx=128/256) vs tok/s

Resultados E1 (Q8, 0-25, gate-only, thresholds victoria 0.70/0.50, scheme=coo, eval-x=32, eval-cols=64):
- max_layers=8: layers=1,5,12,14,16,18,23,25
  - ctx256: base 24.0128 → SD 37.7318 (Δ +57.1%), RSS +43.5 MiB, greedy PASS, 56 new tensors
  - ctx512: base 15.0808 → SD 20.8776 (Δ +38.4%), RSS +41.9 MiB, greedy PASS, 56 new tensors
- max_layers=16: layers=1,3,5,7,9,12,14,16,18,20,23,25 (no llega a 16 por filtros)
  - ctx256: base 24.0128 → SD 101.3362 (Δ +322.0%), RSS +59.2 MiB, greedy PASS, 77 new tensors
  - ctx512: base 15.0808 → SD 59.1786 (Δ +292.4%), RSS +57.7 MiB, greedy PASS, 77 new tensors
- max_layers=24: layers iguales a m16 (mismo set por filtros)
  - ctx256: base 24.0128 → SD 101.3362 (Δ +322.0%), RSS +59.2 MiB, greedy PASS, 77 new tensors
  - ctx512: base 15.0808 → SD 59.2843 (Δ +293.1%), RSS +57.7 MiB, greedy PASS, 77 new tensors
- Nota: con rango 0-25, incluso gate-only degrada fuerte (capas tempranas). Reforzar rango "seguro" (12/14/16/18) o usar selector adicional.
- Conclusión: cos-only no basta para capas tempranas; greedy PASS no implica PPL OK. Hace falta selector funcional (FFN proxy / Δoutput) y/o prior de rango.
- [x] Implementar 2nd gate funcional en autogate (ffn_proxy_cos_p05 + l2_p95 + log_norm_ratio_p95) y extender scan con ffn_proxy_*.
- [ ] Calibrar thresholds funcionales con whitelist 12/14/16/18 y reintentar expansion con barandales.

Resultados E1b (Q8, 10-19, gate-only, thresholds victoria 0.70/0.50):
- autogate selecciona layers=10,12,14,16,18 (5 capas), greedy PASS
- ctx256: base 24.0128 → SD 32.2135 (Δ +34.15%), RSS +4.5 MiB, 35 new tensors
- ctx512: base 15.0808 → SD 25.3050 (Δ +67.80%), RSS +4.4 MiB, 35 new tensors
- Nota: incluir capa 10 rompe la mejora; siguiente intento debe ser whitelist 12/14/16/18 (sin 10) o filtro funcional.

Resultados E1b (Q8, whitelist 12/14/16/18, gate-only):
- greedy PASS, 28 new tensors
- ctx256: base 24.0128 → SD 25.0971 (Δ +4.52%), RSS +3.3 MiB
- ctx512: base 15.0808 → SD 14.7436 (Δ -2.24%), RSS +3.2 MiB
- Nota: mejora en ctx512, leve degradacion en ctx256; confirmar en F16 antes de concluir.

Resultados E2 (F16, whitelist 12/14/16/18, gate-only):
- greedy PASS, 28 new tensors
- ctx256: base 24.1772 → SD 25.1559 (Δ +4.05%), RSS +3.7 MiB
- ctx512: base 15.0469 → SD 14.7325 (Δ -2.09%), RSS +3.3 MiB
- Nota: replica el patron Q8 (mejora en ctx512, leve degradacion en ctx256).

Resultados E3 (Q8, filtro funcional + cos prefilter ajustado a scan sin base-fit):
- Scan v3 sin --base-fit (ffn_proxy disponible). Prefiltro cos ajustado a min good: mean>=0.48, p05>=0.22.
- Autogate selecciona layers=1,5,7,10,12,14,17,22 (incluye 10).
- greedy PASS, 56 new tensors.
- ctx256: base 24.0128 → SD 423.6253 (Δ +1664.16%), RSS +7.1 MiB
- ctx512: base 15.0808 → SD 192.7277 (Δ +1177.97%), RSS +5.6 MiB
- Nota: el filtro funcional auto-calibrado no excluye la capa 10; con prefilter relajado el PPL explota. Revisar criterio/umbral funcional o añadir lista negra temprana.

Resultados E4 (Q8, guardrails funcionales + K=128, scan v4 con meta):
- Autogate selecciona layers=13,15,18,20,22,25 (6 capas), 42 new tensors.
- greedy PASS (ctx256 y ctx512).
- ctx256: base 24.0128 → SD 26.9834 (Δ +12.37%)
- ctx512: base 15.0808 → SD 14.6115 (Δ -3.11%)
- Nota: con K=128 y guardrail early-layer (cutoff=12), desaparece el desastre temprano; mejora en ctx512 pero pierde en ctx256.
- ctx768: base 12.7670 → SD 12.2994 (Δ -3.66%), greedy PASS.
- ctx1024: base 16.6540 → SD 15.5520 (Δ -6.62%), greedy PASS.
- ctx1280: base 14.9195 → SD 15.0916 (Δ +1.15%), greedy PASS.
- ctx1536: base 16.3335 → SD 16.7454 (Δ +2.52%), greedy PASS.
- ctx1792: base 15.9578 → SD 18.4126 (Δ +15.39%), greedy PASS.
- ctx2048 (run1/2/3): base 15.1789 → SD 16.2703 (Δ +7.19%), greedy PASS.
- Nota: la curva no es monótona; mejora hasta ~1024, cruza a + a partir de ~1280. 2048 repetido 3x arroja la misma PPL (varianza despreciable).

Diagnostico (E4 + sweep ctx):
- Pipeline real y trazable (scan → autogate → policy → build → eval). New tensors confirma cambios.
- ffn_gate compresible; ffn_up sigue deny-by-default.
- Greedy PASS no implica calidad global (PPL es el guardián real).
- La policy actual es dependiente del regimen de ctx: mejora 512-1024 y degrada >=1280.

Matices y feedback (post-E5):
- Pipeline real: new tensors + apples-to-apples => no placebo.
- Gate-only tiene regimen: en ctx medio (512-1024) regulariza y puede bajar PPL.
- Guardrails: cutoff temprano + functional gate evitaron el infierno (E3).
- Quiebre de contexto es real: 2048 repetido con la misma PPL => no es ruido.
- E5 es oro: hay causalidad; layer 25 es el principal detonante en ctx largo.
- Matiz importante: policy_mid {13,15,18,20,22,25} no es necesariamente optima; todas las variantes leave-one-out mejoran mas en ctx1024 que la policy completa, asi que hay margen quitando 1-2 capas incluso en mid.

Siguiente ROI (ctx-aware):
- [x] E5: leave-one-out sobre {13,15,18,20,22,25} midiendo ctx1024 y ctx2048 para detectar capas toxicas en largo.
- [x] E6: policy por buckets (policy_mid <=1024, policy_long >=1536) o selector multi-ctx.
- [x] E7: verificacion de ahorro real (gate-only, up-off) con ctx 64/128/1024/2048, medir GGUF size + RSS + PPL + greedy.

Resultados E5 (leave-one-out, policy base {13,15,18,20,22,25}, gate-only):
- minus_13: ctx1024 base 16.6540 → SD 15.0719 (Δ -9.50%), ctx2048 base 15.1789 → SD 15.5016 (Δ +2.13%)
- minus_15: ctx1024 base 16.6540 → SD 15.2213 (Δ -8.60%), ctx2048 base 15.1789 → SD 15.4970 (Δ +2.10%)
- minus_18: ctx1024 base 16.6540 → SD 14.7696 (Δ -11.31%), ctx2048 base 15.1789 → SD 15.3373 (Δ +1.04%)
- minus_20: ctx1024 base 16.6540 → SD 14.8316 (Δ -10.94%), ctx2048 base 15.1789 → SD 15.4103 (Δ +1.52%)
- minus_22: ctx1024 base 16.6540 → SD 14.8465 (Δ -10.85%), ctx2048 base 15.1789 → SD 15.2537 (Δ +0.49%)
- minus_25: ctx1024 base 16.6540 → SD 15.1529 (Δ -9.01%), ctx2048 base 15.1789 → SD 14.8568 (Δ -2.12%)
- Nota: remover 25 invierte ctx2048 a mejora (unico caso negativo). Remover 22 deja ctx2048 casi neutro. Todas las variantes mejoran mas en ctx1024 que la policy completa.

Resultados E6 (policies manuales gate-only, K=128, ctx 512/1024/2048):
- P_full = {13,15,18,20,22,25}
  - ctx512: base 15.0808 → SD 14.6115 (Δ -3.11%)
  - ctx1024: base 16.6540 → SD 15.5520 (Δ -6.62%)
  - ctx2048: base 15.1789 → SD 16.2703 (Δ +7.19%)
- P_long1 = {13,15,18,20,22} (drop 25)
  - ctx512: base 15.0808 → SD 13.5248 (Δ -10.32%)
  - ctx1024: base 16.6540 → SD 15.1529 (Δ -9.01%)
  - ctx2048: base 15.1789 → SD 14.8568 (Δ -2.12%)
- P_long2 = {13,15,18,20} (drop 25+22)
  - ctx512: base 15.0808 → SD 13.4581 (Δ -10.76%)
  - ctx1024: base 16.6540 → SD 14.7227 (Δ -11.60%)
  - ctx2048: base 15.1789 → SD 14.2602 (Δ -6.05%)
- Nota: P_long2 domina en los 3 ctx (mejor PPL en mid y long). Candidato fuerte para policy_long y posiblemente policy_mid.
- Nota: report.json no expone strip_*; hay que verificar strip real por tamano GGUF/logs.
- Decision: congelar policy_long = policy_mid = P_long2 = {13,15,18,20} (mejor en ctx512/1024/2048).

Resultados E7 (gate-only, policy {13,15,18,20}):
- GGUF size (bytes): base=4,130,226,336; SD=4,151,689,600; delta=+21,463,264 (~+20.5 MiB).
- ctx64: base 15.9299 → SD 15.4288 (Δ -3.15%), greedy PASS.
- ctx128: base 18.3384 → SD 17.7740 (Δ -3.08%), greedy PASS.
- ctx1024: base 16.6540 → SD 14.7227 (Δ -11.60%), greedy PASS.
- ctx2048: base 15.1789 → SD 14.2602 (Δ -6.05%), greedy PASS.
- RSS (kB): ctx64 base=4,131,312 sd=4,153,404 delta=+22,092; ctx128 base=4,131,700 sd=4,153,608 delta=+21,908.
- RSS (kB): ctx1024 base=4,241,760 sd=4,263,764 delta=+22,004; ctx2048 base=4,322,920 sd=4,344,744 delta=+21,824.
- Nota: no se observaron campos strip_* en report.json; GGUF y RSS suben ~+21-22 MiB, sugiere que --strip-dense no esta reduciendo pesos o los deltas agregan mas de lo que se quita.

Resultados E7b (gate-only, policy {13,15,18,20} con strip_dense=true en policy, remoto, ctx1024):
- report.json: emit=4, strip_dense=4 (strip aplicado a los 4 tensores gate).
- PPL ctx1024: base 16.6540 → SD 14.7227 (Δ -11.60%).
- GGUF size (bytes): base=4,130,226,336; SD=4,040,268,224; delta=-89,958,112 (~-85.8 MiB).
- RSS (kB): base=4,242,500; SD=4,154,944; delta=-87,556 (~-85.6 MiB).
- Nota: strip en policy si reduce GGUF/RSS de forma consistente; confirma que el control debe venir de la policy (no del flag CLI).

Resultados E7c (gate-only, policy strip {13,15,18,20}, remoto, multi-ctx + RSS probe):
- ctx512: base 15.0808 → SD 13.4581 (Δ -10.76%), RSS delta -86,844 kB (~-84.8 MiB), greedy pack RESULT: PASS (0/20 flagged).
- ctx2048: base 15.1789 → SD 14.2602 (Δ -6.05%), RSS delta -87,208 kB (~-85.1 MiB), greedy pack RESULT: PASS (0/20 flagged).
- Nota: greedy pack es heuristico (anti-colapso). Los diffs exactos vs base son esperables cuando cambiamos pesos.

Resultados E8 (Mistral 7B Q8, gate+down, up-off, strip en policy):
- E8a (down solo en capa 20, policy {13,15,18,20}):
  - ctx1024: base 4.9288 → SD 5.5176 (Δ +11.95%)
  - ctx2048: base 6.3703 → SD 7.2158 (Δ +13.27%)
  - GGUF delta: -266.8 MiB, RSS delta: ~-264 MiB (ctx64/128)
  - strip aplicado: gate 4 capas + down 1 capa
- E8b (down en capas 18,20):
  - ctx1024: base 4.9288 → SD 5.8301 (Δ +18.29%)
  - ctx2048: base 6.3703 → SD 7.5232 (Δ +18.10%)
  - GGUF delta: -324.1 MiB, RSS delta: ~-322 MiB (ctx64/128)
  - strip aplicado: gate 4 capas + down 2 capas
- Conclusion: down aporta ahorro fuerte pero degrada PPL de forma clara incluso con 1-2 capas; se pausa Fase 2 hasta tener filtro funcional mas fuerte para down.

Resultados Gemma 1B (local, Q4_K_M, gate-only):
- Policy {13,15,18,20} strip_dense=true:
  - ctx512: base 1.0050 → SD 1.0072 (Δ +0.22%).
  - ctx2048: base 1.0024 → SD 1.0023 (Δ -0.01%).
  - ctx4096: base 1.0021 → SD 1.0020 (Δ -0.01%).
  - ctx8192: base 1.0023 → SD 1.0023 (Δ ~0%).
  - GGUF delta: -7.06 MiB; RSS delta: -6.8 MiB (ctx64) / -6.3 MiB (ctx128).
- Autogate {1,3,5,10} (capas tempranas):
  - ctx512: base 1.0050 → SD 1.0196 (Δ +1.45%).
  - ctx2048: base 1.0024 → SD 1.0039 (Δ +0.15%).
- Autogate con denylist 0-11 (layers {17,19,21,23}):
  - ctx512: base 1.0050 → SD 1.0076 (Δ +0.26%).
  - ctx2048: base 1.0024 → SD 1.0029 (Δ +0.05%).
- Scan ffn_proxy (sin --base-fit) + detector por ffn_proxy_score:
  - ffn_proxy_available=true (proxy solo disponible sin base-fit).
  - Autogate (rank=ffn_proxy_score, good layers 13/15/18/20) selecciona {12,19,21,23}.
  - ctx512: base 1.0050 → SD 1.0063 (Δ +0.13%).
  - ctx2048: base 1.0024 → SD 1.0031 (Δ +0.07%).

Fase 2 - "cazar al gigante" (gate+down, up-off):
- [x] D1: down estricto (ej: 0.75/0.55) con gate victoria
- [x] D2: si D1 pasa, relajar down (ej: 0.60/0.40), up sigue OFF
- [x] D3: limitar down con max_down_layers separado de max_gate_layers
- Entregable: policy gate+down, tabla capas gate vs capas down + ΔPPL + ahorro
  - Helpers: `scripts/seeddelta-e8-make-policy.py`, `scripts/seeddelta-e8-run.sh`, `scripts/seeddelta-e8-summary.py`
  - One-liners (remoto):
    - E8a (down solo capa 20):
      - `scripts/seeddelta-e8-make-policy.py --out calibration/e8a_down20/policy_down20.json --layers 13,15,18,20 --down 20 --k 128 --strip-dense`
      - `scripts/seeddelta-e8-run.sh --base ~/models/gemma-3-4b-it-Q8_0.gguf --policy calibration/e8a_down20/policy_down20.json --outdir calibration/e8a_down20`
    - E8b (down en 18,20):
      - `scripts/seeddelta-e8-make-policy.py --out calibration/e8b_down18_20/policy_down18_20.json --layers 13,15,18,20 --down 18,20 --k 128 --strip-dense`
      - `scripts/seeddelta-e8-run.sh --base ~/models/gemma-3-4b-it-Q8_0.gguf --policy calibration/e8b_down18_20/policy_down18_20.json --outdir calibration/e8b_down18_20`
    - E8c (down en 13,15,18,20):
      - `scripts/seeddelta-e8-make-policy.py --out calibration/e8c_down_all/policy_down_all.json --layers 13,15,18,20 --down 13,15,18,20 --k 128 --strip-dense`
      - `scripts/seeddelta-e8-run.sh --base ~/models/gemma-3-4b-it-Q8_0.gguf --policy calibration/e8c_down_all/policy_down_all.json --outdir calibration/e8c_down_all`

Fase 3 - STRIP real y medicion de RAM:
- [x] Confirmar en logs: no "0 new tensors", strip aplicado
- [x] Medir RSS con ctx pequeno y ctx real
- [x] Reportar GGUF size, RSS total, y breakdown si existe

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

---

## Nota temporal (2025-12-24) — estado remoto + idea “KV procedural”

### Qué estábamos corriendo (para retomar fácil)

**Remoto 64GB (root@REMOTE_64GB_HOST)**:
- Job pesado: `llama-imatrix` para **Mistral 7B Q8** con **wikitext-2 completo**.
  - Modelo (canon): `/models/mistral-7b-instruct-v0.2.Q8_0.gguf` (`/models` es symlink al cache).
  - Output: `/root/llama.cpp/calibration/mistral7b.imatrix.gguf`
  - Log: `/root/llama.cpp/calibration/mistral7b_imatrix.log`
  - ETA observado en log: ~2h21m (varía).
- Tuning OS: zram activo (25%, lz4), `vm.swappiness=1`, `vm.vfs_cache_pressure=50`.

**Remoto 32GB (devgpt@REMOTE_32GB_HOST)**:
- Baseline Mistral 7B Q8 (wikitext-2, chunks=4):
  - ctx512: PPL `7.3049 ± 0.62997`
  - ctx2048: PPL `6.8369 ± 0.27176`
- Swap físico desactivado; solo zram (prio alto) para evitar swapping “sorpresa”.
- Job en curso: “E7 equivalente” sobre Mistral 7B Q8 (gate-only + strip, policy `{13,15,18,20}`) vía `scripts/seeddelta-e8-run.sh`
  - Outdir: `calibration/mistral_e7_gateonly_k128/`
  - Mide PPL (ctx1024/2048) + RSS probes (ctx64/128) + greedy pack.

### Limpieza/consistencia lograda en tooling
- `scripts/seeddelta-e8-run.sh` ya no fuerza `--strip-dense` por CLI: el strip es **policy-driven**; CLI `--strip-dense-cli` queda opcional.
- README actualizado con timeline + “war stories” (números que forzaron decisiones).

### Idea divergente (no implementada aún): KV procedural en runtime

Objetivo: bajar RSS en ctx enormes (KV domina), aceptando penalización en tok/s.

Propuesta simple/realista: **compresión online de KV por bloques** (“summary tokens”):
- Mantener KV exacto para los últimos `W` tokens.
- Para tokens más viejos, agrupar en bloques `B` y comprimir a `r` resúmenes por bloque.
- Para cada cluster `j` (tamaño `n_j`):
  - `k̃_j = k_pivot` (evitar promediar K; RoPE/posiciones se mezclan raro)
  - `ṽ_j = mean(v_i)`
  - bias de “masa” `b_j = log(n_j)` y usar score `q·k̃_j + b_j` en softmax.
- Parámetros a explorar: `W∈{2048,4096}`, `B∈{256,512}`, `r∈{4,8}`.

Hipótesis: reduce memoria ~O(ctx) → ~O(W + (#bloques)*r) y mantiene coherencia si el “barrido” del pasado se resume bien.

---

## Nota de pivoteo (2025-12-25)

Motivo: el gate-only con strip ya entrega ahorro real de pesos y mejora PPL en Mistral (E7c), pero gate+down (E8a/E8b) degrada PPL con claridad incluso en configuraciones conservadoras. El cuello de botella de RAM ahora es el KV cache en ctx grande, no los pesos. Por eso se pivotea el foco principal al TODO de "KV procedural 100k", dejando Fase 2 (down) como backlog hasta tener un filtro funcional mas fuerte.

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
