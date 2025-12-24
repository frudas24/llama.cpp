- [x] Re-run la evaluación de PPL (Gemma 1B strip): `calibration/gemma_pplex_eval.log` con `-c 512 --seeddelta`.

- [x] Greedy pack rápido (SeedΔ): usar `-f` y `-c 1024` porque con `-c 512` el prompt pack excede el contexto; log en `calibration/gemma_greedy_strip.log`.

- [x] Simple smoke modelo base (sin SeedΔ): ya no crashea en CPU; corre `llama-cli ... --prompt "hola"` en 512 ctx.
  - `/usr/bin/time` con SeedΔ ejecutado: `--seeddelta -c 512 -p "hola"` completó (RSS máx ~910 MB).
  - Nota CLI local: la UI interactiva puede quedarse colgada sin TTY; usar `--simple-io --no-display-prompt` para salida directa. Ejemplo:
    `./build/bin/llama-cli -m calibration/gemma_sd_mid_10-11_block16_k64_strip.gguf --seeddelta -t 16 -c 512 --temp 0 --top-k 1 --seed 1 -n 8 --single-turn --no-warmup --simple-io --no-display-prompt --prompt "hola"` (resp. OK, prompt ~114 t/s, gen ~38 t/s, RSS ~910 MB).

- [x] Validation rápida del report JSON: `jq '.weights[] | {kind,ffn_proxy_available,stack_cost_total}' calibration/gemma1b_ffnproxy_smoke.json`

- [x] Flags nuevos (TODO k_layer/sublayer): tiling básico para K por tile.
  - `--tile-rows N` habilita tiles por filas (default 0=off); `--tile-align N` alinea límites (default 32).
  - `--k-selector cycle|uniform|ttcross` controla asignación de K por tile (default `cycle`); `--k-selector-rank 1|2` y `--k-selector-samples N` afinan el modo `ttcross`.
  - Ejemplo smoke (remoto): `./build/bin/llama-seeddelta-build -i ../models/ggml-org_gemma-3-4b-it-GGUF_gemma-3-4b-it-Q4_K_M.gguf -o tmp-bin/gemma4b_tile.gguf --layers 10-11 --scheme block --block 16 --K 64 --tile-rows 1024 --tile-align 32 --eval-cols 16 --eval-x 4 --report-json tmp-bin/gemma4b_tile_report.json -t 16 --overwrite-existing`
  - `report.json` ahora rellena `tile_rows`, `tile_rows_align`, `k_levels`, `k_per_tile`, `unique_k_count`, `k_total_per_tensor`.




# 1) Ejecutar el script principal de evaluation (seeddelta-policy-eval.sh) contra Gemma 1B (strip) para generar reporte completo.
./scripts/seeddelta-policy-eval.sh \
  --base calibration/gemma_sd_mid_10-11_block16_k64_strip.gguf \
  --sd calibration/gemma_sd_mid_10-11_block16_k64_strip.gguf \
  --text calibration/gemma_calibration.txt \
  --threads 16 --ctx 512 --chunks 8 \
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
  --threads 16 --ctx 512

# 4) Correlación rápida de diferentes K (opcional): re-lanzar builder con K=32 y observar reporte.
./build/bin/llama-seeddelta-build \
  -i calibration/gemma_sd_mid_10-11_block16_k64_strip.gguf \
  -o calibration/gemma1b_K32_proxy.gguf \
  --layers 10-10 --K 32 \
  --eval-cols 32 --eval-x 32 \
  --report-json calibration/gemma1b_K32_proxy.json \
  -t 16

jq '.weights[] | {kind, K, stack_cost_total, ffn_proxy_cos_mean, ffn_proxy_p05}' calibration/gemma1b_K32_proxy.json


#remote server deploy and testing
'/bin/bash -lc "cd /home/frudas/synapp2/llama.cpp && ssh -i .codex/do_droplet_ed25519 -o StrictHostKeyChecking=no devgpt@REMOTE_32GB_HOST '"'"'cd llama.cpp && THREADS=16 CTX=512 ./scripts/seeddelta-policy-eval.sh --base ../models/ggml-org_gemma-3-4b-it-GGUF_gemma-3-4b-it-Q4_K_M.gguf --policy tools/seeddelta-build/policies/policy.example.gemma.json --layers 10-11 --imatrix calibration/gemma4b.imatrix.gguf --text wikitext-2-raw/wiki.test.raw --threads 16 --ctx 512 --chunks 4 --no-strip-dense --greedy-pack calibration/greedy_zombie_pack.txt --outdir calibration/seeddelta-policy-eval-gemma4b-10-11-greedy 2>&1 | tail -n 80'"'"'"'

#local

./scripts/seeddelta-policy-eval.sh --base ~/.cache/llama.cpp/ggml-org_gemma-3-4b-it-GGUF_gemma-3-4b-it-Q4_K_M.gguf --policy tools/seeddelta-build/policies/policy.example.gemma.json --layers 10-11 --imatrix calibration/gemma4b.imatrix.gguf --text wikitext-2-raw/wiki.test.raw --threads 16 --ctx 512 --chunks 4 --no-strip-dense --greedy-pack calibration/greedy_zombie_pack.txt --outdir calibration/seeddelta-policy-eval-gemma4b-10-11-greedy.txt


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
