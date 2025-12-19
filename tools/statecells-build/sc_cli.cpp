#include "include/sc_cli.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>

namespace {

[[noreturn]] void usage(const char * argv0) {
    printf("usage: %s -i in.gguf -o out.gguf [options]\n\n", argv0);
    printf("options:\n");
    printf("  --dict-M N           dictionary size (default: 4096)\n");
    printf("  --dict-M-gate N      dictionary size override for ffn_gate (default: --dict-M)\n");
    printf("  --dict-M-up N        dictionary size override for ffn_up   (default: --dict-M)\n");
    printf("  --dict-M-down N      dictionary size override for ffn_down (default: --dict-M)\n");
    printf("  --dict-k K           active atoms per column (default: 16)\n");
    printf("  --dict-k-gate K      active atoms override for ffn_gate (default: --dict-k)\n");
    printf("  --dict-k-up K        active atoms override for ffn_up   (default: --dict-k)\n");
    printf("  --dict-k-down K      active atoms override for ffn_down (default: --dict-k)\n");
    printf("  --dict-eta F         Oja learning rate (default: 0.01)\n");
    printf("  --dict-iters N       number of Oja passes (default: 3)\n");
    printf("  --dict-max-samples N max columns used to train dict (default: 2048)\n");
    printf("  -t, --threads N      worker threads for encoding (default: nproc)\n");
    printf("  --layers A-B         restrict to layer range (default: all)\n");
    printf("  --dict-type f16|f32  output dict type (default: f16)\n");
    printf("  --vals               write per-code fp16 coefficient tensor (default: off)\n");
    printf("  --no-vals            disable vals tensor output\n");
    printf("  --row-scale          write per-output row_scale tensor (default: on)\n");
    printf("  --no-row-scale       disable row_scale tensor output\n");
    printf("  --eval-cols N        evaluate reconstruction gap on N random outputs per weight (default: 0=off)\n");
    printf("  --report-json FILE   write JSON report with per-weight gap metrics (default: none)\n");
    printf("  --imatrix FILE       use importance matrix GGUF from llama-imatrix for data-aware weighting\n");
    printf("  --imatrix-eps F      clamp min imatrix value before sqrt (default: 1e-8)\n");
    printf("  --imatrix-power F    exponent for imatrix weighting (default: 1.0)\n");
    printf("  --resume             resume from existing out.gguf (skip weights that already have dict+codes)\n");
    printf("  --checkpoint-every N write out.gguf every N processed layers (default: 0=off; large I/O)\n");
    printf("  --seed N             RNG seed (default: 1234)\n");
    exit(1);
}

} // namespace

bool sc_parse_args(int argc, char ** argv, sc_args & args) {
    args.n_threads = (int) std::max(1u, std::thread::hardware_concurrency());

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if ((arg == "-i" || arg == "--input") && i + 1 < argc) { args.in_fname = argv[++i]; continue; }
        if ((arg == "-o" || arg == "--output") && i + 1 < argc) { args.out_fname = argv[++i]; continue; }
        if (arg == "--dict-M" && i + 1 < argc) { args.dict_M = std::stoll(argv[++i]); continue; }
        if (arg == "--dict-M-gate" && i + 1 < argc) { args.dict_M_gate = std::stoll(argv[++i]); continue; }
        if (arg == "--dict-M-up"   && i + 1 < argc) { args.dict_M_up   = std::stoll(argv[++i]); continue; }
        if (arg == "--dict-M-down" && i + 1 < argc) { args.dict_M_down = std::stoll(argv[++i]); continue; }
        if (arg == "--dict-k" && i + 1 < argc) { args.dict_k = std::stoi(argv[++i]); continue; }
        if (arg == "--dict-k-gate" && i + 1 < argc) { args.dict_k_gate = std::stoi(argv[++i]); continue; }
        if (arg == "--dict-k-up"   && i + 1 < argc) { args.dict_k_up   = std::stoi(argv[++i]); continue; }
        if (arg == "--dict-k-down" && i + 1 < argc) { args.dict_k_down = std::stoi(argv[++i]); continue; }
        if (arg == "--dict-eta" && i + 1 < argc) { args.dict_eta = std::stof(argv[++i]); continue; }
        if (arg == "--dict-iters" && i + 1 < argc) { args.dict_iters = std::stoi(argv[++i]); continue; }
        if (arg == "--dict-max-samples" && i + 1 < argc) { args.dict_max_samples = std::stoll(argv[++i]); continue; }
        if ((arg == "-t" || arg == "--threads") && i + 1 < argc) { args.n_threads = std::stoi(argv[++i]); continue; }
        if (arg == "--layers" && i + 1 < argc) { args.layers_range = argv[++i]; continue; }
        if (arg == "--dict-type" && i + 1 < argc) { args.dict_type_str = argv[++i]; continue; }
        if (arg == "--vals") { args.write_vals = true; continue; }
        if (arg == "--no-vals") { args.write_vals = false; continue; }
        if (arg == "--row-scale") { args.write_row_scale = true; continue; }
        if (arg == "--no-row-scale") { args.write_row_scale = false; continue; }
        if (arg == "--eval-cols" && i + 1 < argc) { args.eval_cols = std::stoll(argv[++i]); continue; }
        if (arg == "--report-json" && i + 1 < argc) { args.report_json = argv[++i]; continue; }
        if (arg == "--imatrix" && i + 1 < argc) { args.imatrix_file = argv[++i]; continue; }
        if (arg == "--imatrix-eps" && i + 1 < argc) { args.imatrix_eps = std::stof(argv[++i]); continue; }
        if (arg == "--imatrix-power" && i + 1 < argc) { args.imatrix_power = std::stof(argv[++i]); continue; }
        if (arg == "--resume") { args.resume = true; continue; }
        if (arg == "--checkpoint-every" && i + 1 < argc) { args.checkpoint_every = std::stoll(argv[++i]); continue; }
        if (arg == "--seed" && i + 1 < argc) { args.seed = std::stoi(argv[++i]); continue; }
        usage(argv[0]);
    }

    if (args.in_fname.empty() || args.out_fname.empty()) {
        usage(argv[0]);
    }

    return true;
}
