#include "include/sd_cli.h"

#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <thread>

namespace {

[[noreturn]] void usage(const char * argv0) {
    printf("usage: %s -i in.gguf -o out.gguf [options]\n\n", argv0);
    printf("options:\n");
    printf("  --layers A-B         restrict to layer range (default: all)\n");
    printf("  --scheme coo|block   residual encoding (default: coo)\n");
    printf("  --block N            block size for --scheme block (default: 32)\n");
    printf("  --K N                top-K residual entries per output (default: 32)\n");
    printf("  --K-gate N           override top-K for ffn_gate\n");
    printf("  --K-up N             override top-K for ffn_up\n");
    printf("  --K-down N           override top-K for ffn_down\n");
    printf("  --K-levels CSV       optional discrete K levels for tiling (e.g. 32,64). Falls back to K if omitted.\n");
    printf("  --idx-type i16|i32   index tensor type (default: i16)\n");
    printf("  --val-type f16|f32   value tensor type (default: f16)\n");
    printf("  --row-scale          write per-output d_row_scale tensor (default: off)\n");
    printf("  --no-row-scale       disable d_row_scale tensor output\n");
    printf("  --imatrix FILE       importance matrix GGUF from llama-imatrix for data-aware weighting\n");
    printf("  --imatrix-eps F      clamp min imatrix value before sqrt (default: 1e-8)\n");
    printf("  --imatrix-power F    exponent for imatrix weighting (default: 1.0)\n");
    printf("  --base               write Hadamard base tensors (base_d1/base_d2/base_d3/base_perm1) and store residual vs base\n");
    printf("  --base-max-samples N max sampled outputs per block for base fit (default: 2048, 0=all)\n");
    printf("  --base-perm-trials N random P1 trials per base block (default: 1)\n");
    printf("  --strip-dense        drop original dense weights for layers processed (reduces GGUF size; disables dense fallback)\n");
    printf("  --policy FILE        JSON policy for per-layer/tensor K/block/gating/autotune/strip overrides\n");
    printf("  --policy-strict      reject unknown keys in policy.json (default: warn and ignore)\n");
    printf("  --policy-dump-resolved print resolved config per tensor (debug)\n");
    printf("  --policy-export PATH write a canonical policy.json capturing final per-tensor decisions\n");
    printf("  --policy-self-test   run internal policy merge tests and exit\n");
    printf("  --overwrite-existing allow rebuilding tensors that already have SeedΔ (default: skip)\n");
    printf("  --stack-cost-cap F   hard cap for accumulated stack_cost (default: +inf => no cap)\n");
    printf("  --tile-rows N        enable tiling metadata: rows per tile (default: 0 = disabled)\n");
    printf("  --tile-align N       alignment for tile rows/boundaries (default: 32)\n");
    printf("  -t, --threads N      worker threads (default: nproc)\n");
    printf("  --eval-cols N        evaluate reconstruction gap on N random outputs per weight (default: 0=off)\n");
    printf("  --eval-x N           evaluate functional gap on N random x vectors (requires --eval-cols, default: 0=off)\n");
    printf("  --report-json PATH   write per-weight metrics JSON report\n");
    printf("  --seed N             RNG seed (default: 1234)\n");
    exit(1);
}

} // namespace

bool sd_parse_args(int argc, char ** argv, sd_args & args) {
    args.n_threads = (int) std::max(1u, std::thread::hardware_concurrency());
    args.stack_cost_cap = std::numeric_limits<double>::infinity();

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if ((arg == "-i" || arg == "--input") && i + 1 < argc) { args.in_fname = argv[++i]; continue; }
        if ((arg == "-o" || arg == "--output") && i + 1 < argc) { args.out_fname = argv[++i]; continue; }
        if (arg == "--layers" && i + 1 < argc) { args.layers_range = argv[++i]; continue; }
        if (arg == "--scheme" && i + 1 < argc) { args.scheme_str = argv[++i]; continue; }
        if (arg == "--block" && i + 1 < argc) { args.block = std::stoll(argv[++i]); continue; }
        if (arg == "--K" && i + 1 < argc) { args.K = std::stoll(argv[++i]); continue; }
        if (arg == "--K-gate" && i + 1 < argc) { args.K_gate = std::stoll(argv[++i]); continue; }
        if (arg == "--K-up"   && i + 1 < argc) { args.K_up   = std::stoll(argv[++i]); continue; }
        if (arg == "--K-down" && i + 1 < argc) { args.K_down = std::stoll(argv[++i]); continue; }
        if (arg == "--K-levels" && i + 1 < argc) {
            std::string csv = argv[++i];
            size_t pos = 0;
            while (pos < csv.size()) {
                size_t comma = csv.find(',', pos);
                std::string tok = csv.substr(pos, comma == std::string::npos ? csv.size() - pos : comma - pos);
                if (!tok.empty()) {
                    args.K_levels.push_back(std::stoll(tok));
                }
                if (comma == std::string::npos) break;
                pos = comma + 1;
            }
            continue;
        }
        if (arg == "--idx-type" && i + 1 < argc) { args.idx_type_str = argv[++i]; continue; }
        if (arg == "--val-type" && i + 1 < argc) { args.val_type_str = argv[++i]; continue; }
        if (arg == "--row-scale") { args.write_row_scale = true; continue; }
        if (arg == "--no-row-scale") { args.write_row_scale = false; continue; }
        if (arg == "--imatrix" && i + 1 < argc) { args.imatrix_file = argv[++i]; continue; }
        if (arg == "--imatrix-eps" && i + 1 < argc) { args.imatrix_eps = std::stof(argv[++i]); continue; }
        if (arg == "--imatrix-power" && i + 1 < argc) { args.imatrix_power = std::stof(argv[++i]); continue; }
        if (arg == "--base") { args.write_base = true; continue; }
        if (arg == "--base-max-samples" && i + 1 < argc) { args.base_max_samples = std::stoll(argv[++i]); continue; }
        if (arg == "--base-perm-trials" && i + 1 < argc) { args.base_perm_trials = std::max(1, std::stoi(argv[++i])); continue; }
        if (arg == "--strip-dense") { args.strip_dense = true; continue; }
        if (arg == "--policy" && i + 1 < argc) { args.policy_file = argv[++i]; continue; }
        if (arg == "--policy-strict") { args.policy_strict = true; continue; }
        if (arg == "--policy-dump-resolved") { args.policy_dump_resolved = true; continue; }
        if (arg == "--policy-export" && i + 1 < argc) { args.policy_export_file = argv[++i]; continue; }
        if (arg == "--policy-self-test") { args.policy_self_test = true; continue; }
        if (arg == "--stack-cost-cap" && i + 1 < argc) { args.stack_cost_cap = std::stod(argv[++i]); continue; }
        if (arg == "--tile-rows" && i + 1 < argc) { args.tile_rows = std::stoll(argv[++i]); continue; }
        if (arg == "--tile-align" && i + 1 < argc) { args.tile_rows_align = std::max<int64_t>(1, std::stoll(argv[++i])); continue; }
        if (arg == "--overwrite-existing") { args.overwrite_existing = true; continue; }
        if ((arg == "-t" || arg == "--threads") && i + 1 < argc) { args.n_threads = std::stoi(argv[++i]); continue; }
        if (arg == "--eval-cols" && i + 1 < argc) { args.eval_cols = std::stoll(argv[++i]); continue; }
        if (arg == "--eval-x" && i + 1 < argc) { args.eval_x = std::stoll(argv[++i]); continue; }
        if (arg == "--report-json" && i + 1 < argc) { args.report_json = argv[++i]; continue; }
        if (arg == "--seed" && i + 1 < argc) { args.seed = std::stoi(argv[++i]); continue; }
        if (arg == "-h" || arg == "--help") {
            usage(argv[0]);
        }
        fprintf(stderr, "unknown argument: %s\n", arg.c_str());
        usage(argv[0]);
    }

    if (args.in_fname.empty() || args.out_fname.empty()) {
        usage(argv[0]);
    }

    return true;
}
