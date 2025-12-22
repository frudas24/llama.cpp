#include "sd_eval.h"

#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <inttypes.h>
#include <random>
#include <string>
#include <vector>

static void usage() {
    fprintf(stderr,
            "usage: llama-seeddelta-toy [options]\n"
            "  --pattern identity|diag|lowrank|circulant|block\n"
            "  --use-base         fit W0 (xor_circulant) and encode residual vs base\n"
            "  --n-in N           (default: 64)\n"
            "  --n-out N          (default: 64)\n"
            "  --K N              top-K per column (default: 1)\n"
            "  --eval-cols N      columns to sample (default: 32)\n"
            "  --eval-x N         number of random x evals (default: 0)\n"
            "  --rank N           lowrank rank (default: 4)\n"
            "  --block N          block size for block pattern (default: 8)\n"
            "  --base-max-samples N max sampled outputs per block for base fit (default: 0=all)\n"
            "  --perm-trials N    permutation trials for base fit (default: 4)\n"
            "  --seed N           RNG seed (default: 1)\n"
            "  -h, --help\n");
}

static float * col_ptr(ggml_tensor * W, int64_t col) {
    return (float *) ((uint8_t *) W->data + col * W->nb[1]);
}

int main(int argc, char ** argv) {
    std::string pattern = "identity";
    int64_t n_in = 64;
    int64_t n_out = 64;
    int64_t K = 1;
    int64_t eval_cols = 32;
    int64_t eval_x = 0;
    int64_t rank = 4;
    int64_t block = 8;
    int64_t base_max_samples = 0;
    int perm_trials = 4;
    bool use_base = false;
    int seed = 1;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if ((arg == "-h") || (arg == "--help")) {
            usage();
            return 0;
        } else if (arg == "--pattern" && i + 1 < argc) {
            pattern = argv[++i];
        } else if (arg == "--use-base") {
            use_base = true;
        } else if (arg == "--n-in" && i + 1 < argc) {
            n_in = std::strtoll(argv[++i], nullptr, 10);
        } else if (arg == "--n-out" && i + 1 < argc) {
            n_out = std::strtoll(argv[++i], nullptr, 10);
        } else if (arg == "--K" && i + 1 < argc) {
            K = std::strtoll(argv[++i], nullptr, 10);
        } else if (arg == "--eval-cols" && i + 1 < argc) {
            eval_cols = std::strtoll(argv[++i], nullptr, 10);
        } else if (arg == "--eval-x" && i + 1 < argc) {
            eval_x = std::strtoll(argv[++i], nullptr, 10);
        } else if (arg == "--rank" && i + 1 < argc) {
            rank = std::strtoll(argv[++i], nullptr, 10);
        } else if (arg == "--block" && i + 1 < argc) {
            block = std::strtoll(argv[++i], nullptr, 10);
        } else if (arg == "--base-max-samples" && i + 1 < argc) {
            base_max_samples = std::strtoll(argv[++i], nullptr, 10);
        } else if (arg == "--perm-trials" && i + 1 < argc) {
            perm_trials = std::atoi(argv[++i]);
        } else if (arg == "--seed" && i + 1 < argc) {
            seed = std::atoi(argv[++i]);
        } else {
            fprintf(stderr, "unknown arg: %s\n", arg.c_str());
            usage();
            return 1;
        }
    }

    if (n_in <= 0 || n_out <= 0) {
        fprintf(stderr, "n-in/n-out must be > 0\n");
        return 1;
    }

    K = std::max<int64_t>(1, std::min<int64_t>(K, n_in));
    eval_cols = std::max<int64_t>(1, std::min<int64_t>(eval_cols, n_out));

    const size_t mem_size = (size_t) (n_in * n_out * sizeof(float) * 2 + 1024 * 1024);
    ggml_init_params params = { mem_size, nullptr, false };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        fprintf(stderr, "failed to init ggml context\n");
        return 1;
    }

    ggml_tensor * W = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_in, n_out);
    if (!W || !W->data) {
        fprintf(stderr, "failed to allocate W\n");
        ggml_free(ctx);
        return 1;
    }

    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> diag_vals((size_t) n_out, 1.0f);
    for (int64_t j = 0; j < n_out; ++j) {
        diag_vals[(size_t) j] = 1.0f + 0.001f * (float) j;
    }

    std::vector<float> circ((size_t) n_in, 0.0f);
    for (int64_t i = 0; i < n_in; ++i) {
        circ[(size_t) i] = dist(rng) * 0.1f;
    }

    std::vector<float> U;
    std::vector<float> V;
    if (pattern == "lowrank") {
        rank = std::max<int64_t>(1, std::min<int64_t>(rank, std::min(n_in, n_out)));
        U.resize((size_t) n_out * (size_t) rank, 0.0f);
        V.resize((size_t) n_in * (size_t) rank, 0.0f);
        for (size_t i = 0; i < U.size(); ++i) U[i] = dist(rng) * 0.1f;
        for (size_t i = 0; i < V.size(); ++i) V[i] = dist(rng) * 0.1f;
    }

    for (int64_t col = 0; col < n_out; ++col) {
        float * wcol = col_ptr(W, col);
        for (int64_t i = 0; i < n_in; ++i) wcol[i] = 0.0f;

        if (pattern == "identity") {
            if (col < n_in) wcol[col] = 1.0f;
        } else if (pattern == "diag") {
            if (col < n_in) wcol[col] = diag_vals[(size_t) col];
        } else if (pattern == "circulant") {
            for (int64_t i = 0; i < n_in; ++i) {
                const int64_t idx = (i - col) % n_in;
                const int64_t u = idx < 0 ? idx + n_in : idx;
                wcol[i] = circ[(size_t) u];
            }
        } else if (pattern == "block") {
            const int64_t bi = col / block;
            for (int64_t i = 0; i < n_in; ++i) {
                const int64_t bj = i / block;
                if (bi == bj) wcol[i] = 1.0f;
            }
        } else if (pattern == "lowrank") {
            for (int64_t i = 0; i < n_in; ++i) {
                float sum = 0.0f;
                for (int64_t r = 0; r < rank; ++r) {
                    const float u = U[(size_t) col * (size_t) rank + (size_t) r];
                    const float v = V[(size_t) i * (size_t) rank + (size_t) r];
                    sum += u * v;
                }
                wcol[i] = sum;
            }
        } else {
            fprintf(stderr, "unknown pattern: %s\n", pattern.c_str());
            ggml_free(ctx);
            return 1;
        }
    }

    std::vector<int32_t> d_idx((size_t) K * (size_t) n_out, -1);
    std::vector<float> d_val((size_t) K * (size_t) n_out, 0.0f);
    std::vector<float> w;
    std::vector<int32_t> topk;
    std::vector<float> r;
    base_fit base;

    if (use_base) {
        base = fit_base_xor_circulant(W, base_max_samples, nullptr, perm_trials, rng);
    }

    for (int64_t col = 0; col < n_out; ++col) {
        read_column_f32(W, col, w);
        const std::vector<float> * src = &w;

        if (use_base) {
            r.assign((size_t) n_in, 0.0f);
            const int64_t L = base.L;
            const bool is_tall = (n_out >= n_in);
            if (is_tall) {
                const int64_t b = col / L;
                const int64_t p = col - b * L;
                const float * h2 = base.h2.data() + (size_t) b * (size_t) L;
                const int32_t * inv = base.perm1_inv.empty() ? nullptr : base.perm1_inv.data() + (size_t) b * (size_t) L;
                for (int64_t i = 0; i < n_in; ++i) {
                    const int64_t j = inv ? (int64_t) inv[(size_t) i] : i;
                    const int64_t u = (p ^ j) & (L - 1);
                    const float base_w = h2[(size_t) u];
                    r[(size_t) i] = w[(size_t) i] - base_w;
                }
            } else {
                const int64_t p = col;
                for (int64_t b = 0; b < base.B; ++b) {
                    const float * h2 = base.h2.data() + (size_t) b * (size_t) L;
                    const int32_t * inv = base.perm1_inv.empty() ? nullptr : base.perm1_inv.data() + (size_t) b * (size_t) L;
                    const int64_t in0 = b * L;
                    const int64_t in1 = std::min<int64_t>(n_in, in0 + L);
                    for (int64_t i = in0; i < in1; ++i) {
                        const int64_t q = i - in0;
                        const int64_t j = inv ? (int64_t) inv[(size_t) q] : q;
                        const int64_t u = (p ^ j) & (L - 1);
                        const float base_w = h2[(size_t) u];
                        r[(size_t) i] = w[(size_t) i] - base_w;
                    }
                }
            }
            src = &r;
        }

        topk_abs_weighted(*src, nullptr, K, topk);
        for (int64_t rr = 0; rr < K; ++rr) {
            const int32_t ii = rr < (int64_t) topk.size() ? topk[(size_t) rr] : -1;
            d_idx[(size_t) col * (size_t) K + (size_t) rr] = ii;
            d_val[(size_t) col * (size_t) K + (size_t) rr] = (ii >= 0 && ii < (int32_t) n_in) ? (*src)[(size_t) ii] : 0.0f;
        }
    }

    std::mt19937 rng_eval(seed);
    eval_metrics em = use_base
            ? eval_seeddelta_base_residual(W, base, d_idx, d_val, nullptr, nullptr, K, eval_cols, rng_eval)
            : eval_sparse_residual(W, d_idx, d_val, nullptr, nullptr, K, eval_cols, rng_eval);

    printf("pattern=%s base=%s n_in=%" PRId64 " n_out=%" PRId64 " K=%" PRId64 "\n",
           pattern.c_str(), use_base ? "on" : "off", n_in, n_out, K);
    printf("eval cols=%" PRId64 " rel_l2 mean=%.6f p95=%.6f cos mean=%.6f p05=%.6f nr=%.6f\n",
           eval_cols, em.rel_l2_mean, em.rel_l2_p95, em.cos_mean, em.cos_p05, em.norm_ratio_mean);

    if (eval_x > 0) {
        std::mt19937 rng_x(seed + 1);
        eval_metrics emx = eval_seeddelta_x(W, use_base ? &base : nullptr, d_idx, d_val, nullptr, nullptr, K, eval_cols, eval_x, rng_x);
        printf("eval_x x=%" PRId64 " cols=%" PRId64 " rel_l2 mean=%.6f p95=%.6f cos mean=%.6f p05=%.6f nr=%.6f\n",
               eval_x, eval_cols, emx.rel_l2_mean, emx.rel_l2_p95, emx.cos_mean, emx.cos_p05, emx.norm_ratio_mean);
    }

    ggml_free(ctx);
    return 0;
}
