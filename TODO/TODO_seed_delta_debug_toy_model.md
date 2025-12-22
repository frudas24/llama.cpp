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

- Base PPL ~18338.35 (ctx=512)
- SeedΔ L1+L3+L5+L6, K=16: 7000.89 (PASS)

### Tiny 2k steps (sweep K y pares/impares)

Base PPL ~18338.35 (ctx=512).

- L1+L3+L5+L6, K=12: 7340.87 (PASS)
- L1+L3+L5+L6, K=14: 7267.77 (PASS)
- Pares (0,2,4,6), K=16: 27560.64 (PASS)
- Impares (1,3,5,7), K=16: 8049.34 (PASS)

### Tiny 2k steps (impares K y mix)

Base PPL ~18338.35 (ctx=512).

- Impares (1,3,5,7), K=12: 7734.58 (PASS)
- Impares (1,3,5,7), K=14: 7666.95 (PASS)
- Mix (1,2,5,7), K=16: 9833.72 (PASS)

### Tiny 2k steps (ctx=256)

Base PPL ~26081.12 (ctx=256).

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
