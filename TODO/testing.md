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


#remote server deploy and testing
'/bin/bash -lc "cd /home/frudas/synapp2/llama.cpp && ssh -i .codex/do_droplet_ed25519 -o StrictHostKeyChecking=no devgpt@REMOTE_32GB_HOST '"'"'cd llama.cpp && THREADS=8 CTX=512 ./scripts/seeddelta-policy-eval.sh --base ../models/ggml-org_gemma-3-4b-it-GGUF_gemma-3-4b-it-Q4_K_M.gguf --policy tools/seeddelta-build/policies/policy.example.gemma.json --layers 10-11 --imatrix calibration/gemma4b.imatrix.gguf --text wikitext-2-raw/wiki.test.raw --threads 8 --ctx 512 --chunks 4 --no-strip-dense --greedy-pack calibration/greedy_zombie_pack.txt --outdir calibration/seeddelta-policy-eval-gemma4b-10-11-greedy 2>&1 | tail -n 80'"'"'"'

#local

./scripts/seeddelta-policy-eval.sh --base ~/.cache/llama.cpp/ggml-org_gemma-3-4b-it-GGUF_gemma-3-4b-it-Q4_K_M.gguf --policy tools/seeddelta-build/policies/policy.example.gemma.json --layers 10-11 --imatrix calibration/gemma4b.imatrix.gguf --text wikitext-2-raw/wiki.test.raw --th
reads 8 --ctx 512 --chunks 4 --no-strip-dense --greedy-pack calibration/greedy_zombie_pack.txt --outdir calibration/seeddelta-policy-eval-gemma4b-10-11-greedy.txt


#En tamaño (líneas), los más grandes quedaron así:

tools/seeddelta-build/seeddelta-build.cpp: 1598
tools/seeddelta-build/sd_eval.cpp: 1419
src/llama-seeddelta.cpp: 1167
tools/statecells-build/sc_encode.cpp: 572
tools/statecells-build/sc_process.cpp: 323
tools/seeddelta-build/seeddelta_policy.cpp: 312
tools/statecells-build/sc_imatrix.cpp: 287
tools/seeddelta-build/sd_report.cpp: 262
tools/seeddelta-build/sd_ffn_proxy.cpp: 247
tools/seeddelta-build/seeddelta_policy_selftest.cpp: 226
src/llama-statecells.cpp: 176
tools/statecells-build/sc_utils.cpp: 176
tools/statecells-build/sc_eval.cpp: 181
tools/statecells-build/sc_build.cpp: 142