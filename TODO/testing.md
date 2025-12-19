# 1. Re-run la evaluación de PPL con el modelo Gemma 1B strip que ya tienes.
./build/bin/llama-perplexity \
  -m calibration/gemma_sd_mid_10-11_block16_k64_strip.gguf \
  -f calibration/gemma_calibration.txt \
  --chunks 16 -t 8 -c 512 --no-warmup \
  --seeddelta \
  | tee calibration/gemma_pplex_eval.log

# 2. Chequear greedy pack rápido (usa prompt pack existente y comparalo contra el modelo base).
./build/bin/llama-cli \
  -m calibration/gemma_sd_mid_10-11_block16_k64_strip.gguf \
  --seeddelta \
  -t 8 -c 512 --temp 0 --top-k 1 --seed 1 \
  --prompt-file calibration/greedy_zombie_pack.txt \
  --single-turn --no-warmup --no-display-prompt \
  | tee calibration/gemma_greedy_strip.log

# 3. Simple smoke para el modelo base (sin SeedΔ) para comparar tiempos/RSS.
./build/bin/llama-cli \
  -m calibration/gemma_sd_mid_10-11_block16_k64_strip.gguf \
  -t 8 -c 512 --temp 0 --top-k 1 --seed 1 --single-turn --no-warmup \
  --prompt "hola" >/dev/null
/usr/bin/time -v ./build/bin/llama-cli \
  -m calibration/gemma_sd_mid_10-11_block16_k64_strip.gguf \
  --seeddelta \
  -t 8 -c 512 --temp 0 --top-k 1 --seed 1 --single-turn --no-warmup \
  -p "hola" >/dev/null

# 4. Validation rápida del report JSON generado por el refactor (llegará en build).
jq '.weights[] | {kind,ffn_proxy_available,stack_cost_total}' calibration/gemma1b_ffnproxy_smoke.json




# 1) Ejecutar el script principal de evaluation (seeddelta-policy-eval.sh) contra Gemma 1B (strip) para generar reporte completo.
./scripts/seeddelta-policy-eval.sh \
  --base calibration/gemma_sd_mid_10-11_block16_k64_strip.gguf \
  --sd calibration/gemma_sd_mid_10-11_block16_k64_strip.gguf \
  --text calibration/gemma_calibration.txt \
  --threads 4 --ctx 512 --chunks 8 \
  --eval-cols 32 --eval-x 32 \
  --outdir calibration/seeddelta-policy-eval-gemma1b-smoke

# 2) Comparar los archivos JSON/gguf generados (PPL + proxy) para confirmar que salieron ok.
jq '.weights[] | {layer, kind, stack_cost_total, ffn_proxy_cos_mean, ffn_proxy_available}' calibration/seeddelta-policy-eval-gemma1b-smoke/report.json

# 3) Ejecutar el greedy pack runner con modelo strip para verificar estabilidad (prompt pack ya conocido).
./scripts/seeddelta-greedy-pack.sh \
  --base calibration/gemma_sd_mid_10-11_block16_k64_strip.gguf \
  --sd calibration/gemma_sd_mid_10-11_block16_k64_strip.gguf \
  --pack calibration/greedy_zombie_pack.txt \
  --outdir calibration/gemma_greedy_pack_eval \
  --threads 4 --ctx 512

# 4) Correlación rápida de diferentes K (opcional): re-lanzar builder con K=32 y observar reporte.
./build/bin/llama-seeddelta-build \
  -i calibration/gemma_sd_mid_10-11_block16_k64_strip.gguf \
  -o calibration/gemma1b_K32_proxy.gguf \
  --layers 10-10 --K 32 \
  --eval-cols 32 --eval-x 32 \
  --report-json calibration/gemma1b_K32_proxy.json \
  -t 4

jq '.weights[] | {kind, K, stack_cost_total, ffn_proxy_cos_mean, ffn_proxy_p05}' calibration/gemma1b_K32_proxy.json
