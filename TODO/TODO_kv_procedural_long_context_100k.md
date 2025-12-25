# TODO — KV “procedural” (online) para 100k ctx con 16–32 GB RAM

> Meta pragmática: habilitar *contexto efectivo* ~100k tokens en CPU (16–32 GB)
> con degradación baja/moderada y aceptando penalización en tok/s.
>
> Inspiración: el cerebro no mantiene el pasado completo en “working memory”.
> Usa: (1) ventana viva pequeña, (2) *chunking* y resumen, (3) índices para reactivar,
> (4) guardar principalmente “errores/outliers”, (5) jerarquía multi-escala.

---

## 0) Marco teórico (corto)

### 0.1 El enemigo: KV cache

En atención causal, por cada token guardamos *Keys* y *Values* por capa/head:

$$
\\text{Attn}(q) = \\text{softmax}(qK^\\top)\\,V
$$

El runtime guarda $K,V$ de todos los tokens previos para no recomputar.
Eso hace que la RAM crezca lineal con contexto:

$$
\\text{mem}_{KV} \\propto L\\cdot T\\cdot d_{kv}\\cdot \\text{bytes}
$$

- $L$: capas
- $T$: tokens (ctx efectivo)
- $d_{kv}$: dimensión por token (depende de `n_head_kv * head_dim`)
- `bytes`: típicamente 2 (f16/bf16), a veces menos con KV-quant

Ejemplo mental (Mistral 7B, 32 layers, n_head_kv=8, head_dim=128, f16):
- ~128 KiB/token → 2048 tokens ≈ 256 MiB (coincide con logs)
- 100k tokens ≈ 12.8 GiB solo KV

### 0.2 Qué significa “KV procedural” en la práctica

KV no es fijo (depende del prompt). “Procedural” no significa “seed”,
sino **no almacenar todo** y reconstruir/aproximar su efecto:

1) Mantener KV exacto para una ventana reciente (*working memory*)
2) Comprimir el pasado en “tokens resumen” (*gist / chunking*)
3) Conservar unos pocos outliers exactos (*error / salience*)
4) Jerarquía multi-escala (*consolidación*)

La pregunta clave:
> ¿Cómo aproximar la suma de contribuciones de muchos tokens viejos con pocos estados?

---

## 1) Objetivo y definición de éxito

### Objetivo v1 (MVP)

- Soportar “100k tokens ingresados” con un **tope de memoria** del KV:
  - KV exacto: últimos $W$ tokens
  - Pasado: resúmenes por bloques de tamaño $B$ con $r$ resúmenes por bloque

### KPI / gates

- **RSS**: reducir uso de KV significativamente (medir con ctx grande).
- **Calidad**:
  - PPL (wikitext-2) no “explota”; target inicial: ΔPPL <= +5% (ajustable).
  - Greedy pack: anti-colapso (loops / drift) como smoke, no exact-match.
- **Costo**:
  - tok/s puede caer, pero debe ser utilizable (definir target por HW).

---

## 2) Propuesta v1: Ventana viva + resúmenes por bloque (KV multi-escala)

### 2.1 Estructura de memoria (análoga al cerebro)

- Tier A (Working memory): KV exacto para los últimos $W$ tokens.
- Tier B (Episodic gist): para tokens más viejos, por cada bloque de $B$ tokens
  guardamos $r$ “summary tokens”.
- Tier C (Outliers): opcional, guardar $s$ tokens “salientes” exactos por bloque.
- Tier D (Consolidación): compresión multi-nivel (cuando Tier B crece).

Para 100k tokens, con parámetros razonables:
- $W=4096$, $B=512$, $r=8$ ⇒ pasado comprimido ~64×
- tokens efectivos ≈ $W + (T-W)/B\\cdot r$
  - para T=100k: ≈ 4096 + 96k/512*8 ≈ 5596 tokens efectivos

### 2.2 Matemática mínima de resumen con “mass bias”

En un bloque viejo tenemos pares $(k_i, v_i)$, $i=1..B$.
Queremos aproximar el aporte agregado de un cluster $C_j$ (tamaño $n_j$).

Exacto:
$$
\\sum_{i\\in C_j} \\exp(q^\\top k_i)\\,v_i
$$

Aproximación v1:
$$
\\sum_{i\\in C_j} \\exp(q^\\top k_i)\\,v_i
\\approx
\\exp(q^\\top \\tilde k_j + \\log n_j)\\,\\tilde v_j
$$

donde:
- $\\tilde k_j$ = un key “representante” (pivot) del cluster
- $\\tilde v_j$ = promedio de values del cluster
- $\\log n_j$ = bias que preserva “masa” del cluster en softmax

Esto permite usar atención normal con un ajuste simple:
$$
\\text{score}_j = q^\\top \\tilde k_j + \\log(n_j)
$$

### 2.3 Cómo construir los clusters sin matar CPU (tres niveles)

**Opción v1.0 (ultra simple, determinista):** *chunking por posición*
- dividir el bloque en $r$ sub-bloques contiguos
- $\\tilde k_j = k_{\\text{centro}}$ del sub-bloque
- $\\tilde v_j = \\text{mean}(v)$ del sub-bloque
- $n_j = B/r$ (fijo)

**Opción v1.1 (simple + semántica):** *pivots + asignación por coseno*
- seleccionar $r$ pivots (uniforme o farthest-point 1–2 pasadas)
- asignar cada $k_i$ al pivot más cercano por coseno
- $\\tilde k_j = k_{pivot_j}$ (no promediar K por RoPE/pos)
- $\\tilde v_j = mean(v_i)$
- $n_j = |C_j|$

**Opción v1.2 (robusta):** *salience residual + gist*
- igual que v1.1, pero conservar además $s$ tokens exactos por bloque (outliers)

---

## 3) Salience / outliers (guardar “errores” como el cerebro)

Objetivo: no perder detalles raros que sí importan (nombres, números, instrucciones).

Opciones de salience (baratas):

- **Norma de V**: $\\|v_i\\|_2$ alto ⇒ candidato.
- **Novelty**: $\\|k_i - k_{pivot}\\|$ alto ⇒ outlier de su cluster.
- **Entropy proxy**: si la atención se vuelve muy dispersa, subir $s$ (adaptativo).

Implementación v1:
- Por bloque guardamos los top-$s$ outliers con KV exacto.
- El resto se representa por summary tokens.

---

## 4) Consolidación multi-nivel (100k sin crecer lineal)

Cuando Tier B crece (muchos bloques), volvemos a aplicar la misma idea:

- Nivel 0: tokens exactos recientes ($W$)
- Nivel 1: summary tokens por bloque de tokens exactos viejos
- Nivel 2: summary tokens por bloque de summary tokens del nivel 1
- etc.

Regla tipo “LSM tree”:
- cuando un nivel excede $M$ summaries, compactar $M$ en $M/r$.

Esto mantiene memoria ~O(W + r·log(T)).

---

## 5) Plan de implementación (escalonado, “shippeable”)

### Fase 0 — Instrumentación y baseline

- [ ] Medir KB/token de KV en modelos target (Gemma 4B, Mistral 7B).
- [ ] Confirmar en logs el breakdown (model vs KV vs compute).
- [ ] Crear un benchmark “long prompt” reproducible:
  - construir un prompt largo con wikitext / repetición controlada
  - medir RSS y tok/s en ctx 8k/32k/100k (si cabe)

### Fase 1 — Ventana viva + resumen posicional (v1.0)

- [ ] Implementar un modo de KV cache con:
  - buffer exacto circular de tamaño $W$
  - buffer de summaries (nivel 1) con pares $(\\tilde k, \\tilde v, \\log n)$
- [ ] Modificar atención para incluir summaries:
  - concatenar K/V (exacto + summaries)
  - sumar bias $\\log n$ a scores de summaries antes del softmax
- [ ] Flags:
  - `--kv-proc` (on/off)
  - `--kv-window W`
  - `--kv-block B`
  - `--kv-r r`
- [ ] Tests:
  - `--kv-proc=off` no cambia outputs vs baseline
  - sanity en ctx pequeño (PPL similar)

### Fase 2 — Clustering simple + outliers (v1.1/v1.2)

- [ ] Reemplazar resumen posicional por pivots + asignación por coseno (opcional).
- [ ] Añadir outliers top-$s$ por bloque:
  - flags `--kv-salient s`
  - métrica inicial: novelty o norma de V

### Fase 3 — Multi-nivel (consolidación)

- [ ] Implementar niveles 2+ (compactar summaries cuando crecen).
- [ ] Definir políticas de compactación:
  - thresholds por nivel
  - cómo combinar $\\log n$ (suma de masas)

### Fase 4 — Layerwise memory (opcional, alto ROI)

- [ ] Permitir compresión solo en un subconjunto de capas:
  - ej. comprimir KV en capas bajas, mantener full en capas altas
  - flag `--kv-proc-layers lo-hi` o presets

---

## 6) Plan de evaluación (no perderse)

Gates por fase (si falla, no avanzar):

- Gate A: no colapsa (greedy smoke) en ctx grande.
- Gate B: PPL no explota (ΔPPL razonable).
- Gate C: RSS cae fuerte con ctx grande (la meta real).

Experimentos mínimos:

- [ ] `ctx=8k`: validar que la compresión no destruye corto/medio.
- [ ] `ctx=32k`: medir el primer “sweet spot” de memoria.
- [ ] `ctx=100k`: validar el objetivo mercado (RAM 16–32 GB).

---

## 7) Riesgos conocidos / notas

- RoPE/posiciones: promediar K puede ser peligroso; por eso el pivot de K.
- El bias $\\log n$ es crucial: sin él el pasado “pierde masa” y el modelo olvida demasiado.
- Greedy pack no debe ser exact-match; debe ser anti-colapso.

---

## 8) Entregables

- Código runtime: KV cache procedural (tiers + bias).
- Flags y documentación.
- Bench reproducible (long prompt).
- Tabla (por modelo): ctx vs RSS vs tok/s vs ΔPPL con presets (W,B,r,s).
