# TODO — Debug SeedΔ con modelos controlados (A+B)

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
    - 8 capas, n_embd=192, n_ff=768, n_heads=6, vocab pequeño.
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

### Artefactos (remote)

- `calibration/tiny_toy_data/` dataset jsonl.
- `calibration/tiny_toy_model_bf/` modelo HF + tokenizer con byte_fallback.
- `calibration/tiny_toy_model_bf/tiny.gguf` (F16) listo para pruebas.
- `calibration/tiny_toy_model_bf_1k/` modelo 1k steps (mejor output en suma simple).
- `calibration/tiny_toy_model_bf_1k/tiny.gguf` (F16) para pruebas SeedΔ.

### Resultados B (tiny, 1k steps)

- Baseline PPL (val.txt): ~8789.55.
- SeedΔ layer0 (K=32 gate/up, down off): PPL ~8192.06, greedy pack PASS (6/6).
- SeedΔ even (0,2,4,6): PPL ~10048.08, greedy pack PASS (6/6).
- SeedΔ odd (1,3,5,7): PPL ~5465.42, greedy pack PASS (6/6).
- Nota: el greedy pack es heuristico; revisar outputs si hay dudas.

---

## Notas operativas

- Mantener dataset y seeds fijos para reproducibilidad.
- Usar el mismo harness de logs (report.json + greedy pack).
- No mezclar con modelos grandes hasta cerrar A y B.
