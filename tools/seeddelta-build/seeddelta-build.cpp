#include "common.h"
#include "ggml.h"
#include "gguf.h"
#include "seeddelta_policy.h"
#include "seeddelta_policy_export.h"
#include "seeddelta_policy_selftest.h"

#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <numeric>
#include <random>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

static void usage(const char * argv0) {
    printf("usage: %s -i in.gguf -o out.gguf [options]\n\n", argv0);
    printf("options:\n");
    printf("  --layers A-B         restrict to layer range (default: all)\n");
    printf("  --scheme coo|block   residual encoding (default: coo)\n");
    printf("  --block N            block size for --scheme block (default: 32)\n");
    printf("  --K N                top-K residual entries per output (default: 32)\n");
    printf("  --K-gate N           override top-K for ffn_gate\n");
    printf("  --K-up N             override top-K for ffn_up\n");
    printf("  --K-down N           override top-K for ffn_down\n");
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
    printf("  -t, --threads N      worker threads (default: nproc)\n");
    printf("  --eval-cols N        evaluate reconstruction gap on N random outputs per weight (default: 0=off)\n");
    printf("  --eval-x N           evaluate functional gap on N random x vectors (requires --eval-cols, default: 0=off)\n");
    printf("  --report-json PATH   write per-weight metrics JSON report\n");
    printf("  --seed N             RNG seed (default: 1234)\n");
    exit(1);
}

static std::vector<int64_t> parse_layer_range(const std::string & s, int64_t n_layer) {
    if (s.empty()) {
        std::vector<int64_t> out(n_layer);
        std::iota(out.begin(), out.end(), 0);
        return out;
    }

    std::regex re(R"(^\s*(\d+)\s*-\s*(\d+)\s*$)");
    std::smatch m;
    if (!std::regex_match(s, m, re)) {
        throw std::runtime_error("invalid --layers, expected A-B");
    }

    int64_t a = std::stoll(m[1]);
    int64_t b = std::stoll(m[2]);
    if (a > b) std::swap(a, b);
    a = std::max<int64_t>(0, a);
    b = std::min<int64_t>(n_layer - 1, b);
    std::vector<int64_t> out;
    for (int64_t i = a; i <= b; ++i) out.push_back(i);
    return out;
}

static void read_column_f32(const ggml_tensor * W, int64_t j, std::vector<float> & out) {
    const int64_t n_in = W->ne[0];
    out.resize(n_in);
    const uint8_t * base = (const uint8_t *) W->data + j * W->nb[1];
    const auto * traits = ggml_get_type_traits(W->type);

    if (!traits->is_quantized) {
        if (W->type == GGML_TYPE_F32) {
            std::memcpy(out.data(), base, n_in * sizeof(float));
            return;
        }
        if (W->type == GGML_TYPE_F16) {
            const ggml_fp16_t * v = (const ggml_fp16_t *) base;
            for (int64_t i = 0; i < n_in; ++i) out[i] = ggml_fp16_to_fp32(v[i]);
            return;
        }
    }

    GGML_ASSERT(traits->to_float && "no to_float for tensor type");
    traits->to_float(base, out.data(), n_in);
}

static void topk_abs_weighted(
        const std::vector<float> & v,
        const std::vector<float> * w_scale,
        int64_t K,
        std::vector<int32_t> & idx_out) {
    const int64_t n = (int64_t) v.size();
    K = std::min<int64_t>(K, n);
    idx_out.resize((size_t) K);

    std::vector<int32_t> idx((size_t) n);
    std::iota(idx.begin(), idx.end(), 0);
    auto score = [&](int32_t i) -> float {
        const float s = std::fabs(v[(size_t) i]);
        return w_scale ? (s * (*w_scale)[(size_t) i]) : s;
    };
    if (K < n) {
        std::nth_element(idx.begin(), idx.begin() + K, idx.end(), [&](int32_t a, int32_t b) {
            return score(a) > score(b);
        });
        idx.resize((size_t) K);
    }
    std::sort(idx.begin(), idx.end(), [&](int32_t a, int32_t b) {
        return score(a) > score(b);
    });
    idx_out = idx;
}

// Select top blocks by residual energy (sum (v[i]*w)^2 within block).
static void topk_blocks_energy_weighted(
        const std::vector<float> & v,
        const std::vector<float> * w_scale,
        int64_t block,
        int64_t n_blocks_keep,
        std::vector<int32_t> & idx_out) {
    const int64_t n = (int64_t) v.size();
    if (n <= 0 || block <= 0 || n_blocks_keep <= 0) {
        idx_out.clear();
        return;
    }

    const int64_t n_blocks = (n + block - 1) / block;
    n_blocks_keep = std::min<int64_t>(n_blocks_keep, n_blocks);

    std::vector<double> energy((size_t) n_blocks, 0.0);
    for (int64_t b = 0; b < n_blocks; ++b) {
        const int64_t i0 = b * block;
        const int64_t i1 = std::min<int64_t>(n, i0 + block);
        double e = 0.0;
        for (int64_t i = i0; i < i1; ++i) {
            const double ws = w_scale ? (double) (*w_scale)[(size_t) i] : 1.0;
            const double x = (double) v[(size_t) i] * ws;
            e += x * x;
        }
        energy[(size_t) b] = e;
    }

    std::vector<int32_t> idx((size_t) n_blocks);
    std::iota(idx.begin(), idx.end(), 0);
    auto score = [&](int32_t b) -> double {
        return energy[(size_t) b];
    };

    if (n_blocks_keep < n_blocks) {
        std::nth_element(idx.begin(), idx.begin() + n_blocks_keep, idx.end(), [&](int32_t a, int32_t b) {
            return score(a) > score(b);
        });
        idx.resize((size_t) n_blocks_keep);
    }

    std::sort(idx.begin(), idx.end(), [&](int32_t a, int32_t b) {
        return score(a) > score(b);
    });

    idx_out = idx;
}

struct eval_metrics {
    double rel_l2_mean = 0.0;
    double rel_l2_p95  = 0.0;
    double cos_mean    = 0.0;
    double cos_p05     = 0.0;
    double norm_ratio_mean = 0.0;
};

struct cost_estimate {
    int64_t L = 0;
    int64_t B = 0;
    double ops_dense = 0.0;
    double ops_base  = 0.0;
    double ops_delta = 0.0;
    double ops_row_scale = 0.0;
    double ops_total = 0.0;
    double ops_ratio = 0.0;
};

static eval_metrics eval_sparse_residual(
        const ggml_tensor * W,
        const std::vector<int32_t> & d_idx,
        const std::vector<float> & d_val,
        const std::vector<float> * d_row_scale,
        const std::vector<float> * w_scale,
        int64_t K,
        int64_t eval_cols,
        std::mt19937 & rng) {
    eval_metrics out;

    const int64_t n_in  = W->ne[0];
    const int64_t n_out = W->ne[1];

    if (eval_cols <= 0 || n_out <= 0 || n_in <= 0 || K <= 0) {
        return out;
    }

    eval_cols = std::min<int64_t>(eval_cols, n_out);

    std::vector<int64_t> idx(n_out);
    std::iota(idx.begin(), idx.end(), 0);
    std::shuffle(idx.begin(), idx.end(), rng);
    idx.resize((size_t) eval_cols);

    std::vector<float> w;
    std::vector<double> rel_l2;
    std::vector<double> cos;
    std::vector<double> norm_ratio;
    rel_l2.reserve((size_t) eval_cols);
    cos.reserve((size_t) eval_cols);
    norm_ratio.reserve((size_t) eval_cols);

    for (int64_t ci = 0; ci < eval_cols; ++ci) {
        const int64_t col = idx[(size_t) ci];

        read_column_f32(W, col, w);

        const double scale = d_row_scale ? (double) (*d_row_scale)[(size_t) col] : 1.0;

        double w_norm2  = 0.0;
        for (int64_t i = 0; i < n_in; ++i) {
            const double ws = w_scale ? (double) (*w_scale)[(size_t) i] : 1.0;
            const double wi = (double) w[(size_t) i] * ws;
            w_norm2 += wi * wi;
        }

        double wh_norm2 = 0.0;
        double dot      = 0.0;

        const int32_t * idx_col = d_idx.data() + (size_t) col * (size_t) K;
        const float *   val_col = d_val.data() + (size_t) col * (size_t) K;

        for (int64_t r = 0; r < K; ++r) {
            const int32_t ii = idx_col[r];
            if (ii < 0 || ii >= (int32_t) n_in) {
                continue;
            }
            const double ws = w_scale ? (double) (*w_scale)[(size_t) ii] : 1.0;
            const double wh = scale * (double) val_col[r];
            const double whs = wh * ws;
            wh_norm2 += whs * whs;
            dot      += (double) w[(size_t) ii] * ws * whs;
        }

        const double err2 = std::max(w_norm2 + wh_norm2 - 2.0 * dot, 0.0);

        const double denom_w = std::sqrt(std::max(w_norm2,  1e-20));
        const double denom_h = std::sqrt(std::max(wh_norm2, 1e-20));

        rel_l2.push_back(std::sqrt(err2) / denom_w);
        cos.push_back(dot / (denom_w * denom_h));
        norm_ratio.push_back(denom_h / denom_w);
    }

    auto percentile = [](std::vector<double> v, double p) -> double {
        if (v.empty()) return 0.0;
        std::sort(v.begin(), v.end());
        const double x = p * double(v.size() - 1);
        const size_t i = (size_t) x;
        const size_t j = std::min(i + 1, v.size() - 1);
        const double a = x - double(i);
        return v[i] * (1.0 - a) + v[j] * a;
    };

    double sum_rel = 0.0;
    double sum_cos = 0.0;
    double sum_nr  = 0.0;
    for (size_t i = 0; i < rel_l2.size(); ++i) sum_rel += rel_l2[i];
    for (size_t i = 0; i < cos.size();    ++i) sum_cos += cos[i];
    for (size_t i = 0; i < norm_ratio.size(); ++i) sum_nr += norm_ratio[i];

    out.rel_l2_mean = rel_l2.empty() ? 0.0 : sum_rel / double(rel_l2.size());
    out.rel_l2_p95  = percentile(rel_l2, 0.95);
    out.cos_mean    = cos.empty()    ? 0.0 : sum_cos / double(cos.size());
    out.cos_p05     = percentile(cos, 0.05);
    out.norm_ratio_mean = norm_ratio.empty() ? 0.0 : sum_nr / double(norm_ratio.size());

    return out;
}

static eval_metrics eval_block_residual(
        const ggml_tensor * W,
        const std::vector<int32_t> & b_idx,
        const std::vector<float> & b_val,
        int64_t block,
        int64_t n_blocks,
        const std::vector<float> * d_row_scale,
        const std::vector<float> * w_scale,
        int64_t eval_cols,
        std::mt19937 & rng) {
    eval_metrics out;

    const int64_t n_in  = W->ne[0];
    const int64_t n_out = W->ne[1];

    if (eval_cols <= 0 || n_out <= 0 || n_in <= 0 || block <= 0 || n_blocks <= 0) {
        return out;
    }
    if ((int64_t) b_idx.size() != n_blocks * n_out) {
        return out;
    }
    if ((int64_t) b_val.size() != block * n_blocks * n_out) {
        return out;
    }

    eval_cols = std::min<int64_t>(eval_cols, n_out);

    std::vector<int64_t> idx(n_out);
    std::iota(idx.begin(), idx.end(), 0);
    std::shuffle(idx.begin(), idx.end(), rng);
    idx.resize((size_t) eval_cols);

    std::vector<float> w;
    std::vector<double> rel_l2;
    std::vector<double> cos;
    std::vector<double> norm_ratio;
    rel_l2.reserve((size_t) eval_cols);
    cos.reserve((size_t) eval_cols);
    norm_ratio.reserve((size_t) eval_cols);

    for (int64_t ci = 0; ci < eval_cols; ++ci) {
        const int64_t col = idx[(size_t) ci];

        read_column_f32(W, col, w);

        const double scale = d_row_scale ? (double) (*d_row_scale)[(size_t) col] : 1.0;

        double w_norm2  = 0.0;
        for (int64_t i = 0; i < n_in; ++i) {
            const double ws = w_scale ? (double) (*w_scale)[(size_t) i] : 1.0;
            const double wi = (double) w[(size_t) i] * ws;
            w_norm2 += wi * wi;
        }

        double wh_norm2 = 0.0;
        double dot      = 0.0;

        const int32_t * idx_col = b_idx.data() + (size_t) col * (size_t) n_blocks;
        const float *   val_col = b_val.data() + (size_t) col * (size_t) n_blocks * (size_t) block;

        for (int64_t bi = 0; bi < n_blocks; ++bi) {
            const int32_t blk = idx_col[bi];
            if (blk < 0) {
                continue;
            }
            const int64_t in0 = (int64_t) blk * block;
            const int64_t in1 = std::min<int64_t>(n_in, in0 + block);
            for (int64_t i = in0; i < in1; ++i) {
                const int64_t t = i - in0;
                const double ws = w_scale ? (double) (*w_scale)[(size_t) i] : 1.0;
                const double wh = scale * (double) val_col[(size_t) bi * (size_t) block + (size_t) t];
                const double whs = wh * ws;
                wh_norm2 += whs * whs;
                dot      += (double) w[(size_t) i] * ws * whs;
            }
        }

        const double err2 = std::max(w_norm2 + wh_norm2 - 2.0 * dot, 0.0);

        const double denom_w = std::sqrt(std::max(w_norm2,  1e-20));
        const double denom_h = std::sqrt(std::max(wh_norm2, 1e-20));

        rel_l2.push_back(std::sqrt(err2) / denom_w);
        cos.push_back(dot / (denom_w * denom_h));
        norm_ratio.push_back(denom_h / denom_w);
    }

    auto percentile = [](std::vector<double> v, double p) -> double {
        if (v.empty()) return 0.0;
        std::sort(v.begin(), v.end());
        const double x = p * double(v.size() - 1);
        const size_t i = (size_t) x;
        const size_t j = std::min(i + 1, v.size() - 1);
        const double a = x - double(i);
        return v[i] * (1.0 - a) + v[j] * a;
    };

    double sum_rel = 0.0;
    double sum_cos = 0.0;
    double sum_nr  = 0.0;
    for (size_t i = 0; i < rel_l2.size(); ++i) sum_rel += rel_l2[i];
    for (size_t i = 0; i < cos.size();    ++i) sum_cos += cos[i];
    for (size_t i = 0; i < norm_ratio.size(); ++i) sum_nr += norm_ratio[i];

    out.rel_l2_mean = rel_l2.empty() ? 0.0 : sum_rel / double(rel_l2.size());
    out.rel_l2_p95  = percentile(rel_l2, 0.95);
    out.cos_mean    = cos.empty()    ? 0.0 : sum_cos / double(cos.size());
    out.cos_p05     = percentile(cos, 0.05);
    out.norm_ratio_mean = norm_ratio.empty() ? 0.0 : sum_nr / double(norm_ratio.size());

    return out;
}

static int64_t next_pow2_i64(int64_t v) {
    if (v <= 1) {
        return 1;
    }
    int64_t p = 1;
    while (p < v) {
        p <<= 1;
    }
    return p;
}

static void hadamard_transform(std::vector<float> & data) {
    const int64_t n = (int64_t) data.size();
    for (int64_t len = 1; len < n; len <<= 1) {
        for (int64_t i = 0; i < n; i += (len << 1)) {
            for (int64_t j = 0; j < len; ++j) {
                const float u = data[(size_t) (i + j)];
                const float v = data[(size_t) (i + j + len)];
                data[(size_t) (i + j)]       = u + v;
                data[(size_t) (i + j + len)] = u - v;
            }
        }
    }
}

static void hadamard_transform(float * data, int64_t n) {
    for (int64_t len = 1; len < n; len <<= 1) {
        for (int64_t i = 0; i < n; i += (len << 1)) {
            for (int64_t j = 0; j < len; ++j) {
                const float u = data[i + j];
                const float v = data[i + j + len];
                data[i + j]       = u + v;
                data[i + j + len] = u - v;
            }
        }
    }
}

struct base_fit {
    int64_t L = 0;
    int64_t B = 0;
    std::vector<float> d1; // [L, B]
    std::vector<float> d2; // [L, B]
    std::vector<float> d3; // [L, B]
    std::vector<float> h2; // [L, B] where h2 = H * d2 (so w0[p,q] = h2[p^q])
    std::vector<int32_t> perm1;     // [L, B] dest->src mapping for P1 (optional)
    std::vector<int32_t> perm1_inv; // [L, B] src->dest mapping (for fast w0 lookup)
};

static void apply_base_block_f32(
        const float * x_in,
        float * y_out,
        float * tmp,
        const base_fit & base,
        int64_t b) {
    const int64_t L = base.L;
    GGML_ASSERT(L > 0 && base.B > 0);
    GGML_ASSERT(b >= 0 && b < base.B);

    const float * d1 = base.d1.empty() ? nullptr : base.d1.data() + (size_t) b * (size_t) L;
    const float * d2 = base.d2.empty() ? nullptr : base.d2.data() + (size_t) b * (size_t) L;
    const float * d3 = base.d3.empty() ? nullptr : base.d3.data() + (size_t) b * (size_t) L;
    const int32_t * p1 = base.perm1.empty() ? nullptr : base.perm1.data() + (size_t) b * (size_t) L;

    // y = D1 * x
    if (d1) {
        for (int64_t i = 0; i < L; ++i) {
            y_out[i] = x_in[i] * d1[(size_t) i];
        }
    } else {
        std::memcpy(y_out, x_in, (size_t) L * sizeof(float));
    }

    // y = P1 * y
    if (p1) {
        for (int64_t i = 0; i < L; ++i) {
            int32_t src = p1[(size_t) i];
            if (src < 0 || src >= (int32_t) L) {
                src = 0;
            }
            tmp[i] = y_out[src];
        }
        std::memcpy(y_out, tmp, (size_t) L * sizeof(float));
    }

    // y = H * y
    hadamard_transform(y_out, L);

    // y = D2 * y
    if (d2) {
        for (int64_t i = 0; i < L; ++i) {
            y_out[i] *= d2[(size_t) i];
        }
    }

    // y = H * y
    hadamard_transform(y_out, L);

    // y = D3 * y
    if (d3) {
        for (int64_t i = 0; i < L; ++i) {
            y_out[i] *= d3[(size_t) i];
        }
    }
}

static cost_estimate estimate_cost(
        const base_fit * base,
        int64_t n_in,
        int64_t n_out,
        int64_t K,
        bool row_scale) {
    cost_estimate out;
    if (n_in <= 0 || n_out <= 0 || K <= 0) {
        return out;
    }

    out.ops_dense = 2.0 * (double) n_in * (double) n_out;
    out.ops_delta = 2.0 * (double) K * (double) n_out;
    out.ops_row_scale = row_scale ? (double) n_out : 0.0;

    if (base && base->L > 0 && base->B > 0) {
        out.L = base->L;
        out.B = base->B;

        const bool is_tall = n_out >= n_in;
        const double L = (double) base->L;
        const double log2L = std::log2(std::max(2.0, L));

        const double had_ops  = 2.0 * L * log2L; // 2 Hadamards
        const double diag_ops = 3.0 * L;         // D1/D2/D3 (even if identity)
        const double acc_ops  = is_tall ? 0.0 : L;

        out.ops_base = (double) base->B * (had_ops + diag_ops + acc_ops);
    }

    out.ops_total = out.ops_base + out.ops_delta + out.ops_row_scale;
    out.ops_ratio = out.ops_dense > 0.0 ? (out.ops_total / out.ops_dense) : 0.0;
    return out;
}

// Fit a simple Hadamard base for a weight tensor W (stored as [n_in, n_out] in GGUF):
//   W0 = D3 * H * D2 * H * D1, with D1=D3=I and D2 chosen so that W0[p,q] ~= h2[p^q]
// per block. This is the least-squares projection onto XOR-circulant matrices (per block),
// estimated from at most max_samples output columns.
static base_fit fit_base_xor_circulant(
        const ggml_tensor * W,
        int64_t max_samples,
        const std::vector<float> * w_scale,
        int perm_trials,
        std::mt19937 & rng) {
    GGML_ASSERT(W && ggml_n_dims(W) == 2);

    const int64_t n_in  = W->ne[0];
    const int64_t n_out = W->ne[1];
    const bool is_tall = n_out >= n_in;

    const int64_t need_L = std::min(n_in, n_out);
    const int64_t L = next_pow2_i64(need_L);
    const int64_t B = (std::max(n_in, n_out) + L - 1) / L;

    base_fit out;
    out.L = L;
    out.B = B;
    out.d1.assign((size_t) L * (size_t) B, 1.0f);
    out.d2.assign((size_t) L * (size_t) B, 0.0f);
    out.d3.assign((size_t) L * (size_t) B, 1.0f);
    out.h2.assign((size_t) L * (size_t) B, 0.0f);
    out.perm1.assign((size_t) L * (size_t) B, 0);
    out.perm1_inv.assign((size_t) L * (size_t) B, 0);

    std::vector<float> w;
    std::vector<double> sum((size_t) L);
    std::vector<double> sum2((size_t) L);
    std::vector<double> wgt((size_t) L);
    std::vector<int32_t> perm((size_t) L);
    std::vector<int32_t> inv((size_t) L);

    perm_trials = std::max(1, perm_trials);
    const bool have_w = w_scale != nullptr;

    auto init_identity_perm = [&](std::vector<int32_t> & p) {
        for (int64_t i = 0; i < L; ++i) p[(size_t) i] = (int32_t) i;
    };

    auto randomize_perm = [&](std::vector<int32_t> & p) {
        init_identity_perm(p);
        std::shuffle(p.begin(), p.end(), rng);
    };

    auto invert_perm = [&](const std::vector<int32_t> & p, std::vector<int32_t> & out_inv) {
        out_inv.assign((size_t) L, 0);
        for (int64_t i = 0; i < L; ++i) {
            const int32_t src = p[(size_t) i];
            if (src >= 0 && src < (int32_t) L) {
                out_inv[(size_t) src] = (int32_t) i;
            }
        }
    };

    auto fit_block = [&](int64_t b, const std::vector<int64_t> & cols, int64_t S,
                         std::vector<float> & h2_out, std::vector<float> & d2_out) -> double {
        std::fill(sum.begin(),  sum.end(),  0.0);
        std::fill(sum2.begin(), sum2.end(), 0.0);
        std::fill(wgt.begin(),  wgt.end(),  0.0);

        if (is_tall) {
            const int64_t o0 = b * L;
            for (int64_t si = 0; si < S; ++si) {
                const int64_t col = cols[(size_t) si];
                const int64_t p = col - o0;
                read_column_f32(W, col, w);
                for (int64_t j = 0; j < L; ++j) {
                    const int32_t src = perm[(size_t) j];
                    if (src < 0 || src >= (int32_t) n_in) {
                        continue;
                    }
                    const double ws = have_w ? (double) (*w_scale)[(size_t) src] : 1.0;
                    const double ww = ws * ws;
                    const double v = (double) w[(size_t) src];
                    const int64_t u = (p ^ j) & (L - 1);
                    sum[(size_t) u]  += ww * v;
                    sum2[(size_t) u] += ww * v * v;
                    wgt[(size_t) u]  += ww;
                }
            }
        } else {
            const int64_t in0 = b * L;
            const int64_t in1 = std::min<int64_t>(n_in, in0 + L);
            for (int64_t si = 0; si < S; ++si) {
                const int64_t col = cols[(size_t) si];
                const int64_t p = col;
                read_column_f32(W, col, w);
                for (int64_t j = 0; j < L; ++j) {
                    const int32_t src0 = perm[(size_t) j];
                    const int64_t src = in0 + (int64_t) src0;
                    if (src < in0 || src >= in1) {
                        continue;
                    }
                    const double ws = have_w ? (double) (*w_scale)[(size_t) src] : 1.0;
                    const double ww = ws * ws;
                    const double v = (double) w[(size_t) src];
                    const int64_t u = (p ^ j) & (L - 1);
                    sum[(size_t) u]  += ww * v;
                    sum2[(size_t) u] += ww * v * v;
                    wgt[(size_t) u]  += ww;
                }
            }
        }

        h2_out.assign((size_t) L, 0.0f);
        double sse = 0.0;
        for (int64_t u = 0; u < L; ++u) {
            const double ww = wgt[(size_t) u];
            if (ww > 0.0) {
                const double m = sum[(size_t) u] / ww;
                h2_out[(size_t) u] = (float) m;
                sse += std::max(sum2[(size_t) u] - (sum[(size_t) u] * sum[(size_t) u]) / ww, 0.0);
            }
        }

        d2_out = h2_out;
        hadamard_transform(d2_out);
        const float inv_L = 1.0f / (float) L;
        for (int64_t u = 0; u < L; ++u) {
            d2_out[(size_t) u] *= inv_L;
        }

        return sse;
    };

    if (is_tall) {
        for (int64_t b = 0; b < B; ++b) {
            const int64_t o0 = b * L;
            const int64_t o1 = std::min<int64_t>(n_out, (b + 1) * L);
            const int64_t n_out_blk = o1 - o0;
            if (n_out_blk <= 0) {
                continue;
            }

            std::vector<int64_t> cols((size_t) n_out_blk);
            for (int64_t i = 0; i < n_out_blk; ++i) cols[(size_t) i] = o0 + i;
            std::shuffle(cols.begin(), cols.end(), rng);

            const int64_t S = (max_samples <= 0) ? n_out_blk : std::min<int64_t>(n_out_blk, max_samples);

            double best_sse = INFINITY;
            std::vector<float> best_h2;
            std::vector<float> best_d2;
            std::vector<int32_t> best_perm;
            std::vector<int32_t> best_inv;

            for (int t = 0; t < perm_trials; ++t) {
                if (t == 0) {
                    init_identity_perm(perm);
                } else {
                    randomize_perm(perm);
                }

                std::vector<float> h2;
                std::vector<float> d2;
                const double sse = fit_block(b, cols, S, h2, d2);
                if (sse < best_sse) {
                    best_sse = sse;
                    best_h2 = std::move(h2);
                    best_d2 = std::move(d2);
                    best_perm = perm;
                }
            }

            perm = best_perm;
            invert_perm(perm, best_inv);

            std::memcpy(out.h2.data()   + (size_t) b * (size_t) L, best_h2.data(), (size_t) L * sizeof(float));
            std::memcpy(out.d2.data()   + (size_t) b * (size_t) L, best_d2.data(), (size_t) L * sizeof(float));
            std::memcpy(out.perm1.data()     + (size_t) b * (size_t) L, perm.data(),     (size_t) L * sizeof(int32_t));
            std::memcpy(out.perm1_inv.data() + (size_t) b * (size_t) L, best_inv.data(), (size_t) L * sizeof(int32_t));
        }
    } else {
        const int64_t n_out_eff = std::min<int64_t>(n_out, L);
        std::vector<int64_t> cols((size_t) n_out_eff);
        for (int64_t i = 0; i < n_out_eff; ++i) cols[(size_t) i] = i;
        std::shuffle(cols.begin(), cols.end(), rng);
        const int64_t S = (max_samples <= 0) ? n_out_eff : std::min<int64_t>(n_out_eff, max_samples);

        for (int64_t b = 0; b < B; ++b) {
            const int64_t in0 = b * L;
            const int64_t in1 = std::min<int64_t>(n_in, in0 + L);
            const int64_t n_in_blk = in1 - in0;
            if (n_in_blk <= 0) {
                continue;
            }

            double best_sse = INFINITY;
            std::vector<float> best_h2;
            std::vector<float> best_d2;
            std::vector<int32_t> best_perm;
            std::vector<int32_t> best_inv;

            for (int t = 0; t < perm_trials; ++t) {
                if (t == 0) {
                    init_identity_perm(perm);
                } else {
                    randomize_perm(perm);
                }

                std::vector<float> h2;
                std::vector<float> d2;
                const double sse = fit_block(b, cols, S, h2, d2);
                if (sse < best_sse) {
                    best_sse = sse;
                    best_h2 = std::move(h2);
                    best_d2 = std::move(d2);
                    best_perm = perm;
                }
            }

            perm = best_perm;
            invert_perm(perm, best_inv);

            std::memcpy(out.h2.data()   + (size_t) b * (size_t) L, best_h2.data(), (size_t) L * sizeof(float));
            std::memcpy(out.d2.data()   + (size_t) b * (size_t) L, best_d2.data(), (size_t) L * sizeof(float));
            std::memcpy(out.perm1.data()     + (size_t) b * (size_t) L, perm.data(),     (size_t) L * sizeof(int32_t));
            std::memcpy(out.perm1_inv.data() + (size_t) b * (size_t) L, best_inv.data(), (size_t) L * sizeof(int32_t));
        }
    }

    return out;
}

static eval_metrics eval_seeddelta_base_residual(
        const ggml_tensor * W,
        const base_fit & base,
        const std::vector<int32_t> & d_idx,
        const std::vector<float> & d_val,
        const std::vector<float> * d_row_scale,
        const std::vector<float> * w_scale,
        int64_t K,
        int64_t eval_cols,
        std::mt19937 & rng) {
    eval_metrics out;

    const int64_t n_in  = W->ne[0];
    const int64_t n_out = W->ne[1];

    if (eval_cols <= 0 || n_out <= 0 || n_in <= 0 || K <= 0 || base.L <= 0 || base.B <= 0) {
        return out;
    }

    eval_cols = std::min<int64_t>(eval_cols, n_out);

    std::vector<int64_t> idx(n_out);
    std::iota(idx.begin(), idx.end(), 0);
    std::shuffle(idx.begin(), idx.end(), rng);
    idx.resize((size_t) eval_cols);

    const int64_t L = base.L;
    const bool is_tall = n_out >= n_in;

    std::vector<float> w;
    std::vector<double> rel_l2;
    std::vector<double> cos;
    std::vector<double> norm_ratio;
    rel_l2.reserve((size_t) eval_cols);
    cos.reserve((size_t) eval_cols);
    norm_ratio.reserve((size_t) eval_cols);

    for (int64_t ci = 0; ci < eval_cols; ++ci) {
        const int64_t col = idx[(size_t) ci];
        read_column_f32(W, col, w);

        const double scale = d_row_scale ? (double) (*d_row_scale)[(size_t) col] : 1.0;

        double w_norm2 = 0.0;
        double wh_norm2_hat = 0.0;
        double dot_hat = 0.0;

        if (is_tall) {
            const int64_t b = col / L;
            const int64_t p = col - b * L;
            const float * h2 = base.h2.data() + (size_t) b * (size_t) L;
            const int32_t * inv = base.perm1_inv.empty() ? nullptr : base.perm1_inv.data() + (size_t) b * (size_t) L;

            for (int64_t i = 0; i < n_in; ++i) {
                const double ws = w_scale ? (double) (*w_scale)[(size_t) i] : 1.0;
                const double wi = (double) w[(size_t) i] * ws;
                w_norm2 += wi * wi;

                const int64_t j = inv ? (int64_t) inv[(size_t) i] : i;
                const int64_t u = (p ^ j) & (L - 1);
                const double base_w = (double) h2[(size_t) u];
                const double base_s = base_w * ws;
                wh_norm2_hat += base_s * base_s;
                dot_hat += wi * base_s;
            }
        } else {
            const int64_t p = col;
            for (int64_t i = 0; i < n_in; ++i) {
                const double ws = w_scale ? (double) (*w_scale)[(size_t) i] : 1.0;
                const double wi = (double) w[(size_t) i] * ws;
                w_norm2 += wi * wi;

                const int64_t b = i / L;
                const int64_t q = i - b * L;
                const float * h2 = base.h2.data() + (size_t) b * (size_t) L;
                const int32_t * inv = base.perm1_inv.empty() ? nullptr : base.perm1_inv.data() + (size_t) b * (size_t) L;
                const int64_t j = inv ? (int64_t) inv[(size_t) q] : q;
                const int64_t u = (p ^ j) & (L - 1);
                const double base_w = (double) h2[(size_t) u];
                const double base_s = base_w * ws;
                wh_norm2_hat += base_s * base_s;
                dot_hat += wi * base_s;
            }
        }

        const int32_t * idx_col = d_idx.data() + (size_t) col * (size_t) K;
        const float *   val_col = d_val.data() + (size_t) col * (size_t) K;

        for (int64_t r = 0; r < K; ++r) {
            const int32_t ii = idx_col[r];
            if (ii < 0 || ii >= (int32_t) n_in) {
                continue;
            }

            const double ws = w_scale ? (double) (*w_scale)[(size_t) ii] : 1.0;
            const double wi = (double) w[(size_t) ii] * ws;

            const double delta = (double) val_col[r];
            const double delta_s = delta * ws;

            double base_w = 0.0;
            if (is_tall) {
                const int64_t b = col / L;
                const int64_t p = col - b * L;
                const float * h2 = base.h2.data() + (size_t) b * (size_t) L;
                const int32_t * inv = base.perm1_inv.empty() ? nullptr : base.perm1_inv.data() + (size_t) b * (size_t) L;
                const int64_t j = inv ? (int64_t) inv[(size_t) ii] : (int64_t) ii;
                const int64_t u = (p ^ j) & (L - 1);
                base_w = (double) h2[(size_t) u];
            } else {
                const int64_t p = col;
                const int64_t b = (int64_t) ii / L;
                const int64_t q = (int64_t) ii - b * L;
                const float * h2 = base.h2.data() + (size_t) b * (size_t) L;
                const int32_t * inv = base.perm1_inv.empty() ? nullptr : base.perm1_inv.data() + (size_t) b * (size_t) L;
                const int64_t j = inv ? (int64_t) inv[(size_t) q] : q;
                const int64_t u = (p ^ j) & (L - 1);
                base_w = (double) h2[(size_t) u];
            }

            const double base_s = base_w * ws;

            dot_hat += wi * delta_s;
            wh_norm2_hat += 2.0 * base_s * delta_s + delta_s * delta_s;
        }

        const double dot = scale * dot_hat;
        const double wh_norm2 = scale * scale * wh_norm2_hat;

        const double err2 = std::max(w_norm2 + wh_norm2 - 2.0 * dot, 0.0);

        const double denom_w = std::sqrt(std::max(w_norm2,  1e-20));
        const double denom_h = std::sqrt(std::max(wh_norm2, 1e-20));

        rel_l2.push_back(std::sqrt(err2) / denom_w);
        cos.push_back(dot / (denom_w * denom_h));
        norm_ratio.push_back(denom_h / denom_w);
    }

    auto percentile = [](std::vector<double> v, double p) -> double {
        if (v.empty()) return 0.0;
        std::sort(v.begin(), v.end());
        const double x = p * double(v.size() - 1);
        const size_t i = (size_t) x;
        const size_t j = std::min(i + 1, v.size() - 1);
        const double a = x - double(i);
        return v[i] * (1.0 - a) + v[j] * a;
    };

    double sum_rel = 0.0;
    double sum_cos = 0.0;
    double sum_nr  = 0.0;
    for (size_t i = 0; i < rel_l2.size(); ++i) sum_rel += rel_l2[i];
    for (size_t i = 0; i < cos.size();    ++i) sum_cos += cos[i];
    for (size_t i = 0; i < norm_ratio.size(); ++i) sum_nr += norm_ratio[i];

    out.rel_l2_mean = rel_l2.empty() ? 0.0 : sum_rel / double(rel_l2.size());
    out.rel_l2_p95  = percentile(rel_l2, 0.95);
    out.cos_mean    = cos.empty()    ? 0.0 : sum_cos / double(cos.size());
    out.cos_p05     = percentile(cos, 0.05);
    out.norm_ratio_mean = norm_ratio.empty() ? 0.0 : sum_nr / double(norm_ratio.size());

    return out;
}

static eval_metrics eval_seeddelta_base_block_residual(
        const ggml_tensor * W,
        const base_fit & base,
        const std::vector<int32_t> & b_idx,
        const std::vector<float> & b_val,
        int64_t block,
        int64_t n_blocks,
        const std::vector<float> * d_row_scale,
        const std::vector<float> * w_scale,
        int64_t eval_cols,
        std::mt19937 & rng) {
    eval_metrics out;

    const int64_t n_in  = W->ne[0];
    const int64_t n_out = W->ne[1];

    if (eval_cols <= 0 || n_out <= 0 || n_in <= 0 || base.L <= 0 || base.B <= 0 || block <= 0 || n_blocks <= 0) {
        return out;
    }
    if ((int64_t) b_idx.size() != n_blocks * n_out) {
        return out;
    }
    if ((int64_t) b_val.size() != block * n_blocks * n_out) {
        return out;
    }

    eval_cols = std::min<int64_t>(eval_cols, n_out);

    std::vector<int64_t> idx(n_out);
    std::iota(idx.begin(), idx.end(), 0);
    std::shuffle(idx.begin(), idx.end(), rng);
    idx.resize((size_t) eval_cols);

    const int64_t L = base.L;
    const bool is_tall = n_out >= n_in;

    std::vector<float> w;
    std::vector<double> rel_l2;
    std::vector<double> cos;
    std::vector<double> norm_ratio;
    rel_l2.reserve((size_t) eval_cols);
    cos.reserve((size_t) eval_cols);
    norm_ratio.reserve((size_t) eval_cols);

    for (int64_t ci = 0; ci < eval_cols; ++ci) {
        const int64_t col = idx[(size_t) ci];
        read_column_f32(W, col, w);

        const double scale = d_row_scale ? (double) (*d_row_scale)[(size_t) col] : 1.0;

        double w_norm2 = 0.0;
        double wh_norm2_hat = 0.0;
        double dot_hat = 0.0;

        if (is_tall) {
            const int64_t b = col / L;
            const int64_t p = col - b * L;
            const float * h2 = base.h2.data() + (size_t) b * (size_t) L;
            const int32_t * inv = base.perm1_inv.empty() ? nullptr : base.perm1_inv.data() + (size_t) b * (size_t) L;

            for (int64_t i = 0; i < n_in; ++i) {
                const double ws = w_scale ? (double) (*w_scale)[(size_t) i] : 1.0;
                const double wi = (double) w[(size_t) i] * ws;
                w_norm2 += wi * wi;

                const int64_t j = inv ? (int64_t) inv[(size_t) i] : i;
                const int64_t u = (p ^ j) & (L - 1);
                const double base_w = (double) h2[(size_t) u];
                const double base_s = base_w * ws;
                wh_norm2_hat += base_s * base_s;
                dot_hat += wi * base_s;
            }
        } else {
            const int64_t p = col;
            for (int64_t i = 0; i < n_in; ++i) {
                const double ws = w_scale ? (double) (*w_scale)[(size_t) i] : 1.0;
                const double wi = (double) w[(size_t) i] * ws;
                w_norm2 += wi * wi;

                const int64_t b = i / L;
                const int64_t q = i - b * L;
                const float * h2 = base.h2.data() + (size_t) b * (size_t) L;
                const int32_t * inv = base.perm1_inv.empty() ? nullptr : base.perm1_inv.data() + (size_t) b * (size_t) L;
                const int64_t j = inv ? (int64_t) inv[(size_t) q] : q;
                const int64_t u = (p ^ j) & (L - 1);
                const double base_w = (double) h2[(size_t) u];
                const double base_s = base_w * ws;
                wh_norm2_hat += base_s * base_s;
                dot_hat += wi * base_s;
            }
        }

        const int32_t * idx_col = b_idx.data() + (size_t) col * (size_t) n_blocks;
        const float *   val_col = b_val.data() + (size_t) col * (size_t) n_blocks * (size_t) block;

        for (int64_t bi = 0; bi < n_blocks; ++bi) {
            const int32_t blk = idx_col[bi];
            if (blk < 0) {
                continue;
            }
            const int64_t in0 = (int64_t) blk * block;
            const int64_t in1 = std::min<int64_t>(n_in, in0 + block);
            for (int64_t i = in0; i < in1; ++i) {
                const int64_t t = i - in0;
                const double ws = w_scale ? (double) (*w_scale)[(size_t) i] : 1.0;
                const double wi = (double) w[(size_t) i] * ws;

                const double delta = (double) val_col[(size_t) bi * (size_t) block + (size_t) t];
                const double delta_s = delta * ws;

                double base_w = 0.0;
                if (is_tall) {
                    const int64_t b = col / L;
                    const int64_t p = col - b * L;
                    const float * h2 = base.h2.data() + (size_t) b * (size_t) L;
                    const int32_t * inv = base.perm1_inv.empty() ? nullptr : base.perm1_inv.data() + (size_t) b * (size_t) L;
                    const int64_t j = inv ? (int64_t) inv[(size_t) i] : (int64_t) i;
                    const int64_t u = (p ^ j) & (L - 1);
                    base_w = (double) h2[(size_t) u];
                } else {
                    const int64_t p = col;
                    const int64_t b = i / L;
                    const int64_t q = i - b * L;
                    const float * h2 = base.h2.data() + (size_t) b * (size_t) L;
                    const int32_t * inv = base.perm1_inv.empty() ? nullptr : base.perm1_inv.data() + (size_t) b * (size_t) L;
                    const int64_t j = inv ? (int64_t) inv[(size_t) q] : q;
                    const int64_t u = (p ^ j) & (L - 1);
                    base_w = (double) h2[(size_t) u];
                }

                const double base_s = base_w * ws;

                dot_hat += wi * delta_s;
                wh_norm2_hat += 2.0 * base_s * delta_s + delta_s * delta_s;
            }
        }

        const double dot = scale * dot_hat;
        const double wh_norm2 = scale * scale * wh_norm2_hat;

        const double err2 = std::max(w_norm2 + wh_norm2 - 2.0 * dot, 0.0);

        const double denom_w = std::sqrt(std::max(w_norm2,  1e-20));
        const double denom_h = std::sqrt(std::max(wh_norm2, 1e-20));

        rel_l2.push_back(std::sqrt(err2) / denom_w);
        cos.push_back(dot / (denom_w * denom_h));
        norm_ratio.push_back(denom_h / denom_w);
    }

    auto percentile = [](std::vector<double> v, double p) -> double {
        if (v.empty()) return 0.0;
        std::sort(v.begin(), v.end());
        const double x = p * double(v.size() - 1);
        const size_t i = (size_t) x;
        const size_t j = std::min(i + 1, v.size() - 1);
        const double a = x - double(i);
        return v[i] * (1.0 - a) + v[j] * a;
    };

    double sum_rel = 0.0;
    double sum_cos = 0.0;
    double sum_nr  = 0.0;
    for (size_t i = 0; i < rel_l2.size(); ++i) sum_rel += rel_l2[i];
    for (size_t i = 0; i < cos.size();    ++i) sum_cos += cos[i];
    for (size_t i = 0; i < norm_ratio.size(); ++i) sum_nr += norm_ratio[i];

    out.rel_l2_mean = rel_l2.empty() ? 0.0 : sum_rel / double(rel_l2.size());
    out.rel_l2_p95  = percentile(rel_l2, 0.95);
    out.cos_mean    = cos.empty()    ? 0.0 : sum_cos / double(cos.size());
    out.cos_p05     = percentile(cos, 0.05);
    out.norm_ratio_mean = norm_ratio.empty() ? 0.0 : sum_nr / double(norm_ratio.size());

    return out;
}

static eval_metrics eval_seeddelta_x(
        const ggml_tensor * W,
        const base_fit * base,
        const std::vector<int32_t> & d_idx,
        const std::vector<float> & d_val,
        const std::vector<float> * d_row_scale,
        const std::vector<float> * x_scale,
        int64_t K,
        int64_t eval_cols,
        int64_t eval_x,
        std::mt19937 & rng) {
    eval_metrics out;

    const int64_t n_in  = W->ne[0];
    const int64_t n_out = W->ne[1];

    if (eval_cols <= 0 || eval_x <= 0 || n_out <= 0 || n_in <= 0 || K <= 0) {
        return out;
    }

    eval_cols = std::min<int64_t>(eval_cols, n_out);

    std::vector<int64_t> cols(n_out);
    std::iota(cols.begin(), cols.end(), 0);
    std::shuffle(cols.begin(), cols.end(), rng);
    cols.resize((size_t) eval_cols);

    // Pre-read the sampled columns so we can reuse them across eval_x vectors.
    std::vector<float> wcols((size_t) eval_cols * (size_t) n_in);
    std::vector<float> w;
    for (int64_t ci = 0; ci < eval_cols; ++ci) {
        const int64_t col = cols[(size_t) ci];
        read_column_f32(W, col, w);
        std::memcpy(wcols.data() + (size_t) ci * (size_t) n_in, w.data(), (size_t) n_in * sizeof(float));
    }

    const bool have_base = base && base->L > 0 && base->B > 0;
    const bool is_tall = n_out >= n_in;
    const int64_t L = have_base ? base->L : 0;
    const int64_t B = have_base ? base->B : 0;

    std::vector<std::vector<int64_t>> cols_in_block;
    std::vector<int64_t> pos_in_block;
    if (have_base && is_tall) {
        cols_in_block.resize((size_t) B);
        pos_in_block.resize((size_t) eval_cols);
        for (int64_t ci = 0; ci < eval_cols; ++ci) {
            const int64_t col = cols[(size_t) ci];
            const int64_t b = col / L;
            if (b >= 0 && b < B) {
                cols_in_block[(size_t) b].push_back(ci);
                pos_in_block[(size_t) ci] = col - b * L;
            }
        }
    }

    std::normal_distribution<float> nd(0.0f, 1.0f);

    std::vector<float> x((size_t) n_in);
    std::vector<float> x_hat;
    std::vector<float> x_chunk;
    std::vector<float> v;
    std::vector<float> tmp;
    std::vector<float> y_base;

    if (have_base) {
        tmp.resize((size_t) L);
        v.resize((size_t) L);
        y_base.resize((size_t) L);
        if (is_tall) {
            x_hat.resize((size_t) L);
        } else {
            x_chunk.resize((size_t) L);
        }
    }

    std::vector<double> rel_l2;
    std::vector<double> cos;
    std::vector<double> norm_ratio;
    rel_l2.reserve((size_t) eval_x);
    cos.reserve((size_t) eval_x);
    norm_ratio.reserve((size_t) eval_x);

    std::vector<double> y_true((size_t) eval_cols);
    std::vector<double> y_hat((size_t) eval_cols);

    for (int64_t xi = 0; xi < eval_x; ++xi) {
        for (int64_t i = 0; i < n_in; ++i) {
            const float s = x_scale ? (*x_scale)[(size_t) i] : 1.0f;
            x[(size_t) i] = nd(rng) * s;
        }

        // y_true = W^T x for sampled outputs.
        for (int64_t ci = 0; ci < eval_cols; ++ci) {
            const float * wc = wcols.data() + (size_t) ci * (size_t) n_in;
            double acc = 0.0;
            for (int64_t i = 0; i < n_in; ++i) {
                acc += (double) wc[(size_t) i] * (double) x[(size_t) i];
            }
            y_true[(size_t) ci] = acc;
        }

        std::fill(y_hat.begin(), y_hat.end(), 0.0);

        // Base output (if enabled).
        if (have_base) {
            if (is_tall) {
                for (int64_t i = 0; i < n_in; ++i) x_hat[(size_t) i] = x[(size_t) i];
                for (int64_t i = n_in; i < L; ++i) x_hat[(size_t) i] = 0.0f;

                for (int64_t b = 0; b < B; ++b) {
                    if (cols_in_block[(size_t) b].empty()) {
                        continue;
                    }

                    apply_base_block_f32(x_hat.data(), v.data(), tmp.data(), *base, b);
                    for (int64_t ci : cols_in_block[(size_t) b]) {
                        const int64_t p = pos_in_block[(size_t) ci];
                        y_hat[(size_t) ci] = (p >= 0 && p < L) ? (double) v[(size_t) p] : 0.0;
                    }
                }
            } else {
                std::fill(y_base.begin(), y_base.end(), 0.0f);

                for (int64_t b = 0; b < B; ++b) {
                    const int64_t in0 = b * L;
                    const int64_t in1 = std::min<int64_t>(n_in, in0 + L);
                    for (int64_t i = 0; i < in1 - in0; ++i) x_chunk[(size_t) i] = x[(size_t) (in0 + i)];
                    for (int64_t i = in1 - in0; i < L; ++i) x_chunk[(size_t) i] = 0.0f;

                    apply_base_block_f32(x_chunk.data(), v.data(), tmp.data(), *base, b);
                    for (int64_t i = 0; i < L; ++i) y_base[(size_t) i] += v[(size_t) i];
                }

                for (int64_t ci = 0; ci < eval_cols; ++ci) {
                    const int64_t col = cols[(size_t) ci];
                    y_hat[(size_t) ci] = (col >= 0 && col < L) ? (double) y_base[(size_t) col] : 0.0;
                }
            }
        }

        // Add Δx and apply row_scale.
        for (int64_t ci = 0; ci < eval_cols; ++ci) {
            const int64_t col = cols[(size_t) ci];
            double y = y_hat[(size_t) ci];

            const int32_t * idx_col = d_idx.data() + (size_t) col * (size_t) K;
            const float *   val_col = d_val.data() + (size_t) col * (size_t) K;
            for (int64_t r = 0; r < K; ++r) {
                const int32_t ii = idx_col[r];
                if (ii < 0 || ii >= (int32_t) n_in) {
                    continue;
                }
                y += (double) val_col[r] * (double) x[(size_t) ii];
            }

            const double scale = d_row_scale ? (double) (*d_row_scale)[(size_t) col] : 1.0;
            y_hat[(size_t) ci] = y * scale;
        }

        double y_norm2 = 0.0;
        double yh_norm2 = 0.0;
        double dot = 0.0;
        for (int64_t ci = 0; ci < eval_cols; ++ci) {
            const double a = y_true[(size_t) ci];
            const double b = y_hat[(size_t) ci];
            y_norm2  += a * a;
            yh_norm2 += b * b;
            dot      += a * b;
        }

        const double err2 = std::max(y_norm2 + yh_norm2 - 2.0 * dot, 0.0);
        const double denom_y  = std::sqrt(std::max(y_norm2,  1e-20));
        const double denom_yh = std::sqrt(std::max(yh_norm2, 1e-20));

        rel_l2.push_back(std::sqrt(err2) / denom_y);
        cos.push_back(dot / (denom_y * denom_yh));
        norm_ratio.push_back(denom_yh / denom_y);
    }

    auto percentile = [](std::vector<double> v, double p) -> double {
        if (v.empty()) return 0.0;
        std::sort(v.begin(), v.end());
        const double x = p * double(v.size() - 1);
        const size_t i = (size_t) x;
        const size_t j = std::min(i + 1, v.size() - 1);
        const double a = x - double(i);
        return v[i] * (1.0 - a) + v[j] * a;
    };

    double sum_rel = 0.0;
    double sum_cos = 0.0;
    double sum_nr  = 0.0;
    for (size_t i = 0; i < rel_l2.size(); ++i) sum_rel += rel_l2[i];
    for (size_t i = 0; i < cos.size();    ++i) sum_cos += cos[i];
    for (size_t i = 0; i < norm_ratio.size(); ++i) sum_nr += norm_ratio[i];

    out.rel_l2_mean = rel_l2.empty() ? 0.0 : sum_rel / double(rel_l2.size());
    out.rel_l2_p95  = percentile(rel_l2, 0.95);
    out.cos_mean    = cos.empty()    ? 0.0 : sum_cos / double(cos.size());
    out.cos_p05     = percentile(cos, 0.05);
    out.norm_ratio_mean = norm_ratio.empty() ? 0.0 : sum_nr / double(norm_ratio.size());

    return out;
}

static eval_metrics eval_seeddelta_x_block(
        const ggml_tensor * W,
        const base_fit * base,
        const std::vector<int32_t> & b_idx,
        const std::vector<float> & b_val,
        int64_t block,
        int64_t n_blocks,
        const std::vector<float> * d_row_scale,
        const std::vector<float> * x_scale,
        int64_t eval_cols,
        int64_t eval_x,
        std::mt19937 & rng) {
    eval_metrics out;

    const int64_t n_in  = W->ne[0];
    const int64_t n_out = W->ne[1];

    if (eval_cols <= 0 || eval_x <= 0 || n_out <= 0 || n_in <= 0 || block <= 0 || n_blocks <= 0) {
        return out;
    }
    if ((int64_t) b_idx.size() != n_blocks * n_out) {
        return out;
    }
    if ((int64_t) b_val.size() != block * n_blocks * n_out) {
        return out;
    }

    eval_cols = std::min<int64_t>(eval_cols, n_out);

    std::vector<int64_t> cols(n_out);
    std::iota(cols.begin(), cols.end(), 0);
    std::shuffle(cols.begin(), cols.end(), rng);
    cols.resize((size_t) eval_cols);

    // Pre-read the sampled columns so we can reuse them across eval_x vectors.
    std::vector<float> wcols((size_t) eval_cols * (size_t) n_in);
    std::vector<float> w;
    for (int64_t ci = 0; ci < eval_cols; ++ci) {
        const int64_t col = cols[(size_t) ci];
        read_column_f32(W, col, w);
        std::memcpy(wcols.data() + (size_t) ci * (size_t) n_in, w.data(), (size_t) n_in * sizeof(float));
    }

    const bool have_base = base && base->L > 0 && base->B > 0;
    const bool is_tall = n_out >= n_in;
    const int64_t L = have_base ? base->L : 0;
    const int64_t B = have_base ? base->B : 0;

    std::vector<std::vector<int64_t>> cols_in_block;
    std::vector<int64_t> pos_in_block;
    if (have_base && is_tall) {
        cols_in_block.resize((size_t) B);
        pos_in_block.resize((size_t) eval_cols);
        for (int64_t ci = 0; ci < eval_cols; ++ci) {
            const int64_t col = cols[(size_t) ci];
            const int64_t b = col / L;
            if (b >= 0 && b < B) {
                cols_in_block[(size_t) b].push_back(ci);
                pos_in_block[(size_t) ci] = col - b * L;
            }
        }
    }

    std::normal_distribution<float> nd(0.0f, 1.0f);

    std::vector<float> x((size_t) n_in);
    std::vector<float> x_hat;
    std::vector<float> x_chunk;
    std::vector<float> v;
    std::vector<float> tmp;
    std::vector<float> y_base;

    if (have_base) {
        tmp.resize((size_t) L);
        v.resize((size_t) L);
        y_base.resize((size_t) L);
        if (is_tall) {
            x_hat.resize((size_t) L);
        } else {
            x_chunk.resize((size_t) L);
        }
    }

    std::vector<double> rel_l2;
    std::vector<double> cos;
    std::vector<double> norm_ratio;
    rel_l2.reserve((size_t) eval_x);
    cos.reserve((size_t) eval_x);
    norm_ratio.reserve((size_t) eval_x);

    std::vector<double> y_true((size_t) eval_cols);
    std::vector<double> y_hat((size_t) eval_cols);

    for (int64_t xi = 0; xi < eval_x; ++xi) {
        for (int64_t i = 0; i < n_in; ++i) {
            const float s = x_scale ? (*x_scale)[(size_t) i] : 1.0f;
            x[(size_t) i] = nd(rng) * s;
        }

        // y_true = W^T x for sampled outputs.
        for (int64_t ci = 0; ci < eval_cols; ++ci) {
            const float * wc = wcols.data() + (size_t) ci * (size_t) n_in;
            double acc = 0.0;
            for (int64_t i = 0; i < n_in; ++i) {
                acc += (double) wc[(size_t) i] * (double) x[(size_t) i];
            }
            y_true[(size_t) ci] = acc;
        }

        std::fill(y_hat.begin(), y_hat.end(), 0.0);

        // Base output (if enabled).
        if (have_base) {
            if (is_tall) {
                for (int64_t i = 0; i < n_in; ++i) x_hat[(size_t) i] = x[(size_t) i];
                for (int64_t i = n_in; i < L; ++i) x_hat[(size_t) i] = 0.0f;

                for (int64_t b = 0; b < B; ++b) {
                    if (cols_in_block[(size_t) b].empty()) {
                        continue;
                    }

                    apply_base_block_f32(x_hat.data(), v.data(), tmp.data(), *base, b);
                    for (int64_t ci : cols_in_block[(size_t) b]) {
                        const int64_t p = pos_in_block[(size_t) ci];
                        y_hat[(size_t) ci] = (p >= 0 && p < L) ? (double) v[(size_t) p] : 0.0;
                    }
                }
            } else {
                std::fill(y_base.begin(), y_base.end(), 0.0f);

                for (int64_t b = 0; b < B; ++b) {
                    const int64_t in0 = b * L;
                    const int64_t in1 = std::min<int64_t>(n_in, in0 + L);
                    for (int64_t i = 0; i < in1 - in0; ++i) x_chunk[(size_t) i] = x[(size_t) (in0 + i)];
                    for (int64_t i = in1 - in0; i < L; ++i) x_chunk[(size_t) i] = 0.0f;

                    apply_base_block_f32(x_chunk.data(), v.data(), tmp.data(), *base, b);
                    for (int64_t i = 0; i < L; ++i) y_base[(size_t) i] += v[(size_t) i];
                }

                for (int64_t ci = 0; ci < eval_cols; ++ci) {
                    const int64_t col = cols[(size_t) ci];
                    y_hat[(size_t) ci] = (col >= 0 && col < L) ? (double) y_base[(size_t) col] : 0.0;
                }
            }
        }

        // Add Δx and apply row_scale.
        for (int64_t ci = 0; ci < eval_cols; ++ci) {
            const int64_t col = cols[(size_t) ci];
            double y = y_hat[(size_t) ci];

            const int32_t * idx_col = b_idx.data() + (size_t) col * (size_t) n_blocks;
            const float *   val_col = b_val.data() + (size_t) col * (size_t) n_blocks * (size_t) block;
            for (int64_t bi = 0; bi < n_blocks; ++bi) {
                const int32_t blk = idx_col[bi];
                if (blk < 0) {
                    continue;
                }
                const int64_t in0 = (int64_t) blk * block;
                const int64_t in1 = std::min<int64_t>(n_in, in0 + block);
                for (int64_t i = in0; i < in1; ++i) {
                    const int64_t t = i - in0;
                    y += (double) val_col[(size_t) bi * (size_t) block + (size_t) t] * (double) x[(size_t) i];
                }
            }

            const double scale = d_row_scale ? (double) (*d_row_scale)[(size_t) col] : 1.0;
            y_hat[(size_t) ci] = y * scale;
        }

        double y_norm2 = 0.0;
        double yh_norm2 = 0.0;
        double dot = 0.0;
        for (int64_t ci = 0; ci < eval_cols; ++ci) {
            const double a = y_true[(size_t) ci];
            const double b = y_hat[(size_t) ci];
            y_norm2  += a * a;
            yh_norm2 += b * b;
            dot      += a * b;
        }

        const double err2 = std::max(y_norm2 + yh_norm2 - 2.0 * dot, 0.0);
        const double denom_y  = std::sqrt(std::max(y_norm2,  1e-20));
        const double denom_yh = std::sqrt(std::max(yh_norm2, 1e-20));

        rel_l2.push_back(std::sqrt(err2) / denom_y);
        cos.push_back(dot / (denom_y * denom_yh));
        norm_ratio.push_back(denom_yh / denom_y);
    }

    auto percentile = [](std::vector<double> v, double p) -> double {
        if (v.empty()) return 0.0;
        std::sort(v.begin(), v.end());
        const double x = p * double(v.size() - 1);
        const size_t i = (size_t) x;
        const size_t j = std::min(i + 1, v.size() - 1);
        const double a = x - double(i);
        return v[i] * (1.0 - a) + v[j] * a;
    };

    double sum_rel = 0.0;
    double sum_cos = 0.0;
    double sum_nr  = 0.0;
    for (size_t i = 0; i < rel_l2.size(); ++i) sum_rel += rel_l2[i];
    for (size_t i = 0; i < cos.size();    ++i) sum_cos += cos[i];
    for (size_t i = 0; i < norm_ratio.size(); ++i) sum_nr += norm_ratio[i];

    out.rel_l2_mean = rel_l2.empty() ? 0.0 : sum_rel / double(rel_l2.size());
    out.rel_l2_p95  = percentile(rel_l2, 0.95);
    out.cos_mean    = cos.empty()    ? 0.0 : sum_cos / double(cos.size());
    out.cos_p05     = percentile(cos, 0.05);
    out.norm_ratio_mean = norm_ratio.empty() ? 0.0 : sum_nr / double(norm_ratio.size());

    return out;
}

static bool string_remove_suffix(std::string & s, const std::string & suffix) {
    if (s.size() < suffix.size()) {
        return false;
    }
    if (s.compare(s.size() - suffix.size(), suffix.size(), suffix) != 0) {
        return false;
    }
    s.resize(s.size() - suffix.size());
    return true;
}

static int load_legacy_imatrix(
        const std::string & imatrix_file,
        std::vector<std::string> & imatrix_datasets,
        std::unordered_map<std::string, std::vector<float>> & imatrix_data) {
    std::ifstream in(imatrix_file.c_str(), std::ios::binary);
    if (!in) {
        fprintf(stderr, "%s: failed to open %s\n", __func__, imatrix_file.c_str());
        return -1;
    }

    int32_t n_entries = 0;
    in.read((char *) &n_entries, sizeof(n_entries));
    if (in.fail() || n_entries <= 0) {
        fprintf(stderr, "%s: no data in file %s\n", __func__, imatrix_file.c_str());
        return -1;
    }

    imatrix_data.clear();
    imatrix_data.reserve((size_t) n_entries);

    for (int i = 0; i < n_entries; i++) {
        int32_t len = 0;
        in.read((char *) &len, sizeof(len));
        if (in.fail() || len <= 0) {
            fprintf(stderr, "%s: failed reading name for entry %d from %s\n", __func__, i + 1, imatrix_file.c_str());
            return -1;
        }

        std::vector<char> name_as_vec((size_t) len);
        in.read(name_as_vec.data(), len);
        if (in.fail()) {
            fprintf(stderr, "%s: failed reading name for entry %d from %s\n", __func__, i + 1, imatrix_file.c_str());
            return -1;
        }

        std::string name(name_as_vec.begin(), name_as_vec.end());

        int32_t ncall = 0;
        in.read((char *) &ncall, sizeof(ncall));
        if (in.fail() || ncall <= 0) {
            fprintf(stderr, "%s: invalid ncall %d for entry %s\n", __func__, ncall, name.c_str());
            return -1;
        }

        int32_t nval = 0;
        in.read((char *) &nval, sizeof(nval));
        if (in.fail() || nval <= 0) {
            fprintf(stderr, "%s: invalid nval %d for entry %s\n", __func__, nval, name.c_str());
            return -1;
        }

        auto & e = imatrix_data[name];
        e.resize((size_t) nval);
        in.read((char *) e.data(), (size_t) nval * sizeof(float));
        if (in.fail()) {
            fprintf(stderr, "%s: failed reading data for entry %s\n", __func__, name.c_str());
            return -1;
        }
    }

    int m_last_call = 0;
    if (in.peek() != EOF) {
        in.read((char *) &m_last_call, sizeof(m_last_call));
        int dataset_len = 0;
        in.read((char *) &dataset_len, sizeof(dataset_len));
        if (!in.fail() && dataset_len > 0) {
            std::vector<char> dataset_as_vec((size_t) dataset_len);
            in.read(dataset_as_vec.data(), dataset_len);
            if (!in.fail()) {
                imatrix_datasets.resize(1);
                imatrix_datasets[0].assign(dataset_as_vec.begin(), dataset_as_vec.end());
                fprintf(stderr, "%s: imatrix dataset='%s'\n", __func__, imatrix_datasets[0].c_str());
            }
        }
    }

    fprintf(stderr, "%s: loaded %d importance matrix entries from %s computed on %d chunks\n",
            __func__, int(imatrix_data.size()), imatrix_file.c_str(), m_last_call);

    return m_last_call;
}

// Loads an imatrix GGUF file produced by llama-imatrix.
// Data format matches tools/quantize loader: tensors are stored as <name>.in_sum2 and <name>.counts.
static int load_imatrix(
        const std::string & imatrix_file,
        std::vector<std::string> & imatrix_datasets,
        std::unordered_map<std::string, std::vector<float>> & imatrix_data) {
    static const char * const LLM_KV_IMATRIX_DATASETS    = "imatrix.datasets";
    static const char * const LLM_KV_IMATRIX_CHUNK_COUNT = "imatrix.chunk_count";
    static const char * const LLM_KV_IMATRIX_CHUNK_SIZE  = "imatrix.chunk_size";

    struct ggml_context * ctx = nullptr;
    struct gguf_init_params meta_gguf_params = {
        /* .no_alloc = */ false, // the data is needed
        /* .ctx      = */ &ctx,
    };

    struct gguf_context * ctx_gguf = gguf_init_from_file(imatrix_file.c_str(), meta_gguf_params);
    if (!ctx_gguf) {
        fprintf(stderr, "%s: imatrix file '%s' is using old format\n", __func__, imatrix_file.c_str());
        return load_legacy_imatrix(imatrix_file, imatrix_datasets, imatrix_data);
    }

    const int32_t n_entries = gguf_get_n_tensors(ctx_gguf);
    if (n_entries < 1) {
        fprintf(stderr, "%s: no data in file %s\n", __func__, imatrix_file.c_str());
        gguf_free(ctx_gguf);
        ggml_free(ctx);
        return -1;
    }

    const int dataset_idx     = gguf_find_key(ctx_gguf, LLM_KV_IMATRIX_DATASETS);
    const int chunk_count_idx = gguf_find_key(ctx_gguf, LLM_KV_IMATRIX_CHUNK_COUNT);
    const int chunk_size_idx  = gguf_find_key(ctx_gguf, LLM_KV_IMATRIX_CHUNK_SIZE);
    if (dataset_idx < 0 || chunk_count_idx < 0 || chunk_size_idx < 0) {
        fprintf(stderr, "%s: missing imatrix metadata in file %s\n", __func__, imatrix_file.c_str());
        gguf_free(ctx_gguf);
        ggml_free(ctx);
        return -1;
    }

    const uint32_t chunk_size = gguf_get_val_u32(ctx_gguf, chunk_size_idx);
    GGML_UNUSED(chunk_size);

    const std::string sums_suffix{ ".in_sum2" };
    const std::string counts_suffix{ ".counts" };

    std::map<std::string, std::pair<struct ggml_tensor *, struct ggml_tensor *>> sums_counts_for;

    for (struct ggml_tensor * cur = ggml_get_first_tensor(ctx); cur; cur = ggml_get_next_tensor(ctx, cur)) {
        std::string name = cur->name;

        if (name.empty()) {
            continue;
        }

        if (string_remove_suffix(name, sums_suffix)) {
            sums_counts_for[std::move(name)].first = cur;
        } else if (string_remove_suffix(name, counts_suffix)) {
            sums_counts_for[std::move(name)].second = cur;
        }
    }

    imatrix_data.clear();
    imatrix_data.reserve(sums_counts_for.size());

    for (const auto & sc : sums_counts_for) {
        const        std::string & name   = sc.first;
        const struct ggml_tensor * sums   = sc.second.first;
        const struct ggml_tensor * counts = sc.second.second;

        if (!sums || !counts) {
            fprintf(stderr, "%s: mismatched sums and counts for %s\n", __func__, name.c_str());
            gguf_free(ctx_gguf);
            ggml_free(ctx);
            return -1;
        }

        const int64_t ne0 = sums->ne[0];
        const int64_t ne1 = sums->ne[1];

        auto & e = imatrix_data[name];
        e.resize((size_t) ggml_nelements(sums));

        for (int64_t j = 0; j < ne1; ++j) {
            const float count = ((const float *) counts->data)[j];
            if (count > 0.0f) {
                for (int64_t i = 0; i < ne0; ++i) {
                    e[(size_t) j * (size_t) ne0 + (size_t) i] = ((const float *) sums->data)[(size_t) j * (size_t) ne0 + (size_t) i] / count;
                }
            } else {
                // Partial imatrix data, tensor never got any input during calibration.
                for (int64_t i = 0; i < ne0; ++i) {
                    e[(size_t) j * (size_t) ne0 + (size_t) i] = 1.0f;
                }
            }
        }
    }

    const int m_last_chunk = (int) gguf_get_val_u32(ctx_gguf, chunk_count_idx);

    const int64_t n_datasets = gguf_get_arr_n(ctx_gguf, dataset_idx);
    imatrix_datasets.clear();
    imatrix_datasets.reserve((size_t) n_datasets);
    for (int64_t i = 0; i < n_datasets; ++i) {
        imatrix_datasets.push_back(gguf_get_arr_str(ctx_gguf, dataset_idx, i));
    }
    if (!imatrix_datasets.empty()) {
        fprintf(stderr, "%s: imatrix datasets=['%s'", __func__, imatrix_datasets[0].c_str());
        for (size_t i = 1; i < imatrix_datasets.size(); ++i) {
            fprintf(stderr, ", '%s'", imatrix_datasets[i].c_str());
        }
        fprintf(stderr, "]\n");
    }

    fprintf(stderr, "%s: loaded %d importance matrix entries from %s computed on %d chunks\n",
            __func__, int(imatrix_data.size()), imatrix_file.c_str(), m_last_chunk);

    gguf_free(ctx_gguf);
    ggml_free(ctx);

    return m_last_chunk;
}

static bool make_imatrix_sqrt_scale(
        const std::unordered_map<std::string, std::vector<float>> & imatrix_data,
        const std::string & weight_name,
        int64_t n_in,
        float eps,
        float power,
        std::vector<float> & scale_out) {
    const auto it = imatrix_data.find(weight_name);
    if (it == imatrix_data.end()) {
        return false;
    }

    if ((int64_t) it->second.size() < n_in) {
        fprintf(stderr, "seeddelta-build: imatrix entry %s has %" PRId64 " values, expected >= %" PRId64 "\n",
                weight_name.c_str(), (int64_t) it->second.size(), n_in);
        return false;
    }

    eps = std::max(eps, 0.0f);
    power = std::max(power, 0.0f);

    scale_out.resize((size_t) n_in);
    double sum = 0.0;
    for (int64_t i = 0; i < n_in; ++i) {
        float v = it->second[(size_t) i];
        if (!std::isfinite(v) || v < eps) {
            v = eps;
        }

        float s = 1.0f;
        if (power == 0.0f) {
            s = 1.0f;
        } else if (power == 1.0f) {
            s = std::sqrt(v);
        } else {
            s = std::pow(v, 0.5f * power);
        }

        if (!std::isfinite(s) || s <= 0.0f) {
            s = 1.0f;
        }

        scale_out[(size_t) i] = s;
        sum += (double) s;
    }

    const double mean = sum / std::max<double>(1.0, (double) n_in);
    if (mean > 0.0) {
        const float inv_mean = (float) (1.0 / mean);
        for (float & s : scale_out) {
            s *= inv_mean;
        }
    }

    return true;
}

struct report_entry {
    int64_t layer = -1;
    std::string kind;
    int64_t n_in = 0;
    int64_t n_out = 0;
    int64_t K_budget = 0;
    int64_t K = 0;
    int64_t block = 0;
    int64_t n_blocks = 0;
    eval_metrics em;
    eval_metrics em_w;
    eval_metrics em_x;
    eval_metrics em_x_w;
    cost_estimate cost;
    bool has_w = false;
    bool has_x = false;
    // Gating/decision metadata
    bool gating_enabled = false;
    bool gating_pass = true;
    std::string gating_metric_used;
    double gating_value = 0.0;
    double gating_p05 = 0.0;
    double gating_min_mean = 0.0;
    double gating_min_p05 = 0.0;
    bool emit = true;
    bool strip_applied = false;
    std::string decision_reason;

    struct autotune_attempt {
        int64_t K_budget = 0;
        int64_t K_eff = 0;
        int64_t n_blocks = 0;
        double metric_value = 0.0;
        double metric_p05 = 0.0;
        bool pass = false;
        double seconds = 0.0;
    };
    bool autotune_enabled = false;
    int64_t autotune_selected_budget = 0;
    std::vector<autotune_attempt> autotune_attempts;
};

struct pending_tensor_set {
    ggml_context * ctx = nullptr;
    std::vector<ggml_tensor *> tensors;
};

static std::string metric_kind_to_string(sd_metric_kind m) {
    switch (m) {
        case sd_metric_kind::cos_x_w: return "cos_x_w";
        case sd_metric_kind::cos_x:   return "cos_x";
        case sd_metric_kind::cos_w:   return "cos_w";
        case sd_metric_kind::cos:     return "cos";
    }
    return "cos";
}

static double pick_metric_value(const report_entry & re, sd_metric_kind m) {
    switch (m) {
        case sd_metric_kind::cos_x_w: return re.em_x_w.cos_mean;
        case sd_metric_kind::cos_x:   return re.em_x.cos_mean;
        case sd_metric_kind::cos_w:   return re.em_w.cos_mean;
        case sd_metric_kind::cos:     return re.em.cos_mean;
    }
    return re.em.cos_mean;
}

static double pick_metric_p05(const report_entry & re, sd_metric_kind m) {
    switch (m) {
        case sd_metric_kind::cos_x_w: return re.em_x_w.cos_p05;
        case sd_metric_kind::cos_x:   return re.em_x.cos_p05;
        case sd_metric_kind::cos_w:   return re.em_w.cos_p05;
        case sd_metric_kind::cos:     return re.em.cos_p05;
    }
    return re.em.cos_p05;
}

static std::string fnv1a_hex(const std::string & data) {
    uint64_t hash = 1469598103934665603ULL; // FNV-1a 64-bit offset
    for (unsigned char c : data) {
        hash ^= c;
        hash *= 1099511628211ULL;
    }
    std::ostringstream oss;
    oss << std::hex;
    oss.width(16);
    oss.fill('0');
    oss << hash;
    return oss.str();
}

static std::string slurp_file(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return {};
    }
    std::ostringstream oss;
    oss << in.rdbuf();
    return oss.str();
}

int main(int argc, char ** argv) {
    std::string in_fname;
    std::string out_fname;

    std::string layers_range;
    std::string imatrix_file;
    std::string report_json;
    std::string policy_file;
    std::string policy_export_file;

    std::string scheme_str = "coo";
    int64_t block = 32;

    int64_t K = 32;
    int64_t K_gate = -1;
    int64_t K_up   = -1;
    int64_t K_down = -1;
    std::string idx_type_str = "i16";
    std::string val_type_str = "f16";
    bool write_row_scale = false;
    bool write_base = false;
    bool strip_dense = false;
    int64_t base_max_samples = 2048;
    int base_perm_trials = 1;
    int n_threads = (int) std::max(1u, std::thread::hardware_concurrency());
    int64_t eval_cols = 0;
    int64_t eval_x = 0;
    float imatrix_eps = 1e-8f;
    float imatrix_power = 1.0f;
    int seed = 1234;
    bool policy_strict = false;
    bool policy_dump_resolved = false;
    bool policy_self_test = false;
    bool overwrite_existing = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if ((arg == "-i" || arg == "--input") && i + 1 < argc) { in_fname = argv[++i]; continue; }
        if ((arg == "-o" || arg == "--output") && i + 1 < argc) { out_fname = argv[++i]; continue; }
        if (arg == "--layers" && i + 1 < argc) { layers_range = argv[++i]; continue; }
        if (arg == "--scheme" && i + 1 < argc) { scheme_str = argv[++i]; continue; }
        if (arg == "--block" && i + 1 < argc) { block = std::stoll(argv[++i]); continue; }
        if (arg == "--K" && i + 1 < argc) { K = std::stoll(argv[++i]); continue; }
        if (arg == "--K-gate" && i + 1 < argc) { K_gate = std::stoll(argv[++i]); continue; }
        if (arg == "--K-up"   && i + 1 < argc) { K_up   = std::stoll(argv[++i]); continue; }
        if (arg == "--K-down" && i + 1 < argc) { K_down = std::stoll(argv[++i]); continue; }
        if (arg == "--idx-type" && i + 1 < argc) { idx_type_str = argv[++i]; continue; }
        if (arg == "--val-type" && i + 1 < argc) { val_type_str = argv[++i]; continue; }
        if (arg == "--row-scale") { write_row_scale = true; continue; }
        if (arg == "--no-row-scale") { write_row_scale = false; continue; }
        if (arg == "--imatrix" && i + 1 < argc) { imatrix_file = argv[++i]; continue; }
        if (arg == "--imatrix-eps" && i + 1 < argc) { imatrix_eps = std::stof(argv[++i]); continue; }
        if (arg == "--imatrix-power" && i + 1 < argc) { imatrix_power = std::stof(argv[++i]); continue; }
        if (arg == "--base") { write_base = true; continue; }
        if (arg == "--base-max-samples" && i + 1 < argc) { base_max_samples = std::stoll(argv[++i]); continue; }
        if (arg == "--base-perm-trials" && i + 1 < argc) { base_perm_trials = std::max(1, std::stoi(argv[++i])); continue; }
        if (arg == "--strip-dense") { strip_dense = true; continue; }
        if (arg == "--policy" && i + 1 < argc) { policy_file = argv[++i]; continue; }
        if (arg == "--policy-strict") { policy_strict = true; continue; }
        if (arg == "--policy-dump-resolved") { policy_dump_resolved = true; continue; }
        if (arg == "--policy-export" && i + 1 < argc) { policy_export_file = argv[++i]; continue; }
        if (arg == "--policy-self-test") { policy_self_test = true; continue; }
        if (arg == "--overwrite-existing") { overwrite_existing = true; continue; }
        if ((arg == "-t" || arg == "--threads") && i + 1 < argc) { n_threads = std::stoi(argv[++i]); continue; }
        if (arg == "--eval-cols" && i + 1 < argc) { eval_cols = std::stoll(argv[++i]); continue; }
        if (arg == "--eval-x" && i + 1 < argc) { eval_x = std::stoll(argv[++i]); continue; }
        if (arg == "--report-json" && i + 1 < argc) { report_json = argv[++i]; continue; }
        if (arg == "--seed" && i + 1 < argc) { seed = std::stoi(argv[++i]); continue; }
        if (arg == "-h" || arg == "--help") {
            usage(argv[0]);
        }
        fprintf(stderr, "unknown argument: %s\n", arg.c_str());
        usage(argv[0]);
    }

    if (policy_self_test) {
        return sd_policy_self_test();
    }

    if (in_fname.empty() || out_fname.empty()) {
        usage(argv[0]);
    }

    enum resid_scheme {
        RESID_COO   = 0,
        RESID_BLOCK = 1,
    };

    resid_scheme scheme = RESID_COO;
    if (scheme_str == "coo") {
        scheme = RESID_COO;
    } else if (scheme_str == "block") {
        scheme = RESID_BLOCK;
    } else {
        throw std::runtime_error("invalid --scheme (expected: coo|block)");
    }
    if (scheme == RESID_BLOCK) {
        if (block <= 0 || block > 4096) {
            throw std::runtime_error("invalid --block (expected: 1..4096)");
        }
    }

    ggml_type idx_type = GGML_TYPE_I16;
    if (idx_type_str == "i16") idx_type = GGML_TYPE_I16;
    else if (idx_type_str == "i32") idx_type = GGML_TYPE_I32;
    else throw std::runtime_error("invalid --idx-type");

    ggml_type val_type = GGML_TYPE_F16;
    if (val_type_str == "f16") val_type = GGML_TYPE_F16;
    else if (val_type_str == "f32") val_type = GGML_TYPE_F32;
    else throw std::runtime_error("invalid --val-type");

    ggml_context * ctx_data = nullptr;
    gguf_init_params params = { false, &ctx_data };
    gguf_context * src = gguf_init_from_file(in_fname.c_str(), params);
    if (!src || !ctx_data) {
        fprintf(stderr, "failed to load %s\n", in_fname.c_str());
        return 1;
    }

    std::vector<std::string> imatrix_datasets;
    std::unordered_map<std::string, std::vector<float>> imatrix_data;
    const bool have_imatrix = !imatrix_file.empty();
    if (have_imatrix) {
        const int rc = load_imatrix(imatrix_file, imatrix_datasets, imatrix_data);
        if (rc < 0) {
            fprintf(stderr, "seeddelta-build: failed to load imatrix %s\n", imatrix_file.c_str());
            return 1;
        }
    }

    const int64_t n_tensors = gguf_get_n_tensors(src);

    sd_policy policy;
    sd_policy * policy_ptr = nullptr;
    std::string policy_hash;
    if (!policy_file.empty()) {
        auto pres = sd_policy_load(policy_file, policy_strict, policy);
        if (!pres.ok) {
            fprintf(stderr, "seeddelta-build: %s\n", pres.error.c_str());
            return 1;
        }
        for (const auto & w : pres.warnings) {
            fprintf(stderr, "seeddelta-build: policy warning: %s\n", w.c_str());
        }
        policy_ptr = &policy;
        const std::string policy_bytes = slurp_file(policy_file);
        if (!policy_bytes.empty()) {
            policy_hash = "fnv1a64:" + fnv1a_hex(policy_bytes);
        }
    }

    // Discover number of layers from tensor names.
    int64_t max_layer_id = -1;
    std::regex re_layer(R"(blk\.(\d+)\.)");
    for (int64_t ti = 0; ti < n_tensors; ++ti) {
        const char * name = gguf_get_tensor_name(src, ti);
        std::cmatch m;
        if (std::regex_search(name, m, re_layer)) {
            max_layer_id = std::max<int64_t>(max_layer_id, std::stoll(m[1]));
        }
    }
    const int64_t n_layer = max_layer_id + 1;
    auto layers = parse_layer_range(layers_range, n_layer);

    const std::vector<std::string> kinds = { "ffn_gate", "ffn_up", "ffn_down" };

    std::mt19937 rng(seed);

    // Keep per-weight ggml contexts alive until we write the output.
    std::vector<ggml_context *> sd_contexts;

    int64_t n_added = 0;
    std::vector<report_entry> report;
    std::vector<pending_tensor_set> pending;
    std::unordered_set<std::string> strip_weights;
    bool any_strip = false;

    const int64_t K_default = std::max<int64_t>(1, K);
    const int64_t K_gate_eff = (K_gate > 0 ? K_gate : K_default);
    const int64_t K_up_eff   = (K_up   > 0 ? K_up   : K_default);
    const int64_t K_down_eff = (K_down > 0 ? K_down : K_default);
    const bool K_variable = (K_gate_eff != K_default) || (K_up_eff != K_default) || (K_down_eff != K_default);

    sd_resolved_tensor baseline_cfg;
    baseline_cfg.block = block;
    baseline_cfg.K_gate = K_gate_eff;
    baseline_cfg.K_up   = K_up_eff;
    baseline_cfg.K_down = K_down_eff;
    baseline_cfg.strip_dense = strip_dense;
    baseline_cfg.enabled = true;
    baseline_cfg.metric = sd_metric_kind::cos_x_w;
    baseline_cfg.min_mean = 0.0f;
    baseline_cfg.min_p05  = 0.0f;
    baseline_cfg.gating_enabled = (policy_ptr != nullptr);

    if (policy_ptr && eval_cols <= 0) {
        fprintf(stderr, "seeddelta-build: --policy requires --eval-cols > 0 for gating/autotune\n");
        return 1;
    }

    for (const int64_t il : layers) {
        for (const auto & kind : kinds) {
            const std::string weight_name = "blk." + std::to_string(il) + "." + kind + ".weight";
            const bool have_weight = gguf_find_tensor(src, weight_name.c_str()) != -1;

            sd_resolved_tensor baseline_kind = baseline_cfg;
            if (kind == "ffn_down") {
                baseline_kind.min_mean = 0.25f;
                baseline_kind.min_p05  = 0.15f;
            } else {
                baseline_kind.min_mean = 0.45f;
                baseline_kind.min_p05  = 0.30f;
            }

            sd_resolved_tensor cfg = sd_policy_resolve(policy_ptr, il, kind, baseline_kind);
            if (!cfg.enabled) {
                // still record later in report as disabled
            }
            if (policy_dump_resolved) {
                fprintf(stderr, "policy: blk.%" PRId64 ".%s enabled=%d strip=%d block=%" PRId64 " K(g/u/d)=%" PRId64 "/%" PRId64 "/%" PRId64 " metric=%s min_mean=%.3f min_p05=%.3f\n",
                        il, kind.c_str(), cfg.enabled ? 1 : 0, cfg.strip_dense ? 1 : 0, cfg.block,
                        cfg.K_gate, cfg.K_up, cfg.K_down,
                        metric_kind_to_string(cfg.metric).c_str(), cfg.min_mean, cfg.min_p05);
            }

            if (cfg.require_eval_x && (eval_x <= 0 || eval_cols <= 0)) {
                fprintf(stderr, "seeddelta-build: gating metric requires --eval-x and --eval-cols > 0 for %s\n", weight_name.c_str());
                return 1;
            }
            if (cfg.require_imatrix && !have_imatrix) {
                fprintf(stderr, "seeddelta-build: gating metric requires imatrix for %s\n", weight_name.c_str());
                return 1;
            }

            if (!have_weight) {
                if (policy_ptr || !report_json.empty()) {
                    report_entry re;
                    re.layer = il;
                    re.kind = kind;
                    re.n_in = 0;
                    re.n_out = 0;
                    re.emit = false;
                    re.strip_applied = false;
                    re.gating_enabled = (policy_ptr != nullptr);
                    re.gating_pass = false;
                    re.decision_reason = "missing_tensor";
                    re.gating_metric_used = metric_kind_to_string(cfg.metric);
                    re.gating_min_mean = cfg.min_mean;
                    re.gating_min_p05 = cfg.min_p05;
                    report.push_back(std::move(re));
                }
                continue;
            }

            const std::string d_idx_name      = "blk." + std::to_string(il) + "." + kind + ".d_idx";
            const std::string d_val_name      = "blk." + std::to_string(il) + "." + kind + ".d_val";
            const std::string d_row_scale_name = "blk." + std::to_string(il) + "." + kind + ".d_row_scale";
            const std::string b_idx_name      = "blk." + std::to_string(il) + "." + kind + ".b_idx";
            const std::string b_val_name      = "blk." + std::to_string(il) + "." + kind + ".b_val";

            if (!overwrite_existing && (gguf_find_tensor(src, d_idx_name.c_str()) != -1 || gguf_find_tensor(src, d_val_name.c_str()) != -1 ||
                gguf_find_tensor(src, b_idx_name.c_str()) != -1 || gguf_find_tensor(src, b_val_name.c_str()) != -1)) {
                fprintf(stderr, "seeddelta-build: %s already has delta tensors, skipping (use --overwrite-existing to rebuild)\n", weight_name.c_str());
                continue;
            }

            ggml_tensor * W = ggml_get_tensor(ctx_data, weight_name.c_str());
            if (!W || ggml_n_dims(W) != 2) {
                continue;
            }

            if (!cfg.enabled) {
                report_entry re;
                re.layer = il;
                re.kind = kind;
                re.n_in = W->ne[0];
                re.n_out = W->ne[1];
                re.emit = false;
                re.strip_applied = false;
                re.gating_enabled = (policy_ptr != nullptr);
                re.gating_pass = false;
                re.decision_reason = "disabled";
                re.gating_metric_used = metric_kind_to_string(cfg.metric);
                re.gating_min_mean = cfg.min_mean;
                re.gating_min_p05 = cfg.min_p05;
                report.push_back(std::move(re));
                continue;
            }

            const int64_t n_in  = W->ne[0];
            const int64_t n_out = W->ne[1];
            const int64_t K_kind =
                    kind == "ffn_gate" ? cfg.K_gate :
                    kind == "ffn_up"   ? cfg.K_up   :
                                        cfg.K_down;
            const int64_t K_budget = std::max<int64_t>(1, std::min<int64_t>(K_kind, n_in));
            const bool is_tall = n_out >= n_in;
            const int64_t block_here = (scheme == RESID_BLOCK) ? cfg.block : block;
            const int64_t block = block_here; // shadow global block for per-tensor overrides

            std::vector<float> w_scale;
            const bool have_w = have_imatrix && make_imatrix_sqrt_scale(imatrix_data, weight_name, n_in, imatrix_eps, imatrix_power, w_scale);
            if (cfg.require_imatrix && !have_w) {
                report_entry re;
                re.layer = il;
                re.kind = kind;
                re.n_in = n_in;
                re.n_out = n_out;
                re.emit = false;
                re.gating_enabled = (policy_ptr != nullptr);
                re.gating_pass = false;
                re.decision_reason = "missing_imatrix";
                re.gating_metric_used = metric_kind_to_string(cfg.metric);
                re.gating_min_mean = cfg.min_mean;
                re.gating_min_p05 = cfg.min_p05;
                report.push_back(std::move(re));
                continue;
            }

            base_fit base;
            if (write_base) {
                const int64_t t0 = ggml_time_us();
                base = fit_base_xor_circulant(W, base_max_samples, have_w ? &w_scale : nullptr, base_perm_trials, rng);
                const double sec = double(ggml_time_us() - t0) / 1e6;
                fprintf(stderr, "  [blk.%" PRId64 ".%s] base fit kind=xor_circulant L=%" PRId64 " B=%" PRId64 " depth=2 samples=%" PRId64 " perm_trials=%d%s (%.1fs)\n",
                        il, kind.c_str(), base.L, base.B, base_max_samples, base_perm_trials, have_w ? " imatrix=on" : "", sec);
            }

            struct trial_data {
                report_entry re;
                std::vector<int32_t> d_idx;
                std::vector<float>   d_val;
                std::vector<int32_t> b_idx;
                std::vector<float>   b_val;
                std::vector<float>   d_row_scale;
                double seconds = 0.0;
            };

            auto run_trial = [&](int64_t K_budget_trial) -> trial_data {
                trial_data t;

                const int64_t t0_total = ggml_time_us();
                const int64_t K_budget_clamped = std::max<int64_t>(1, std::min<int64_t>(K_budget_trial, n_in));

                int64_t n_blocks_keep = 0;
                int64_t K_eff = K_budget_clamped;
                if (scheme == RESID_BLOCK) {
                    const int64_t n_blocks_total = (n_in + block - 1) / block;
                    n_blocks_keep = std::max<int64_t>(1, std::min<int64_t>((K_budget_clamped + block - 1) / block, n_blocks_total));
                    K_eff = n_blocks_keep * block;
                }

                if (scheme == RESID_BLOCK) {
                    printf("seeddelta-build: layer %" PRId64 " %s [% " PRId64 " x %" PRId64 "] type=%s scheme=block block=%" PRId64 " nb=%" PRId64 " K=%" PRId64 " (budget=%" PRId64 ")%s\n",
                           il, kind.c_str(), n_in, n_out, ggml_type_name(W->type), block, n_blocks_keep, K_eff, K_budget_clamped,
                           have_w ? " imatrix=on" : (have_imatrix ? " imatrix=missing" : ""));
                } else {
                    printf("seeddelta-build: layer %" PRId64 " %s [% " PRId64 " x %" PRId64 "] type=%s scheme=coo K=%" PRId64 "%s\n",
                           il, kind.c_str(), n_in, n_out, ggml_type_name(W->type), K_eff,
                           have_w ? " imatrix=on" : (have_imatrix ? " imatrix=missing" : ""));
                }

                t.re.layer = il;
                t.re.kind = kind;
                t.re.n_in = n_in;
                t.re.n_out = n_out;
                t.re.K_budget = K_budget_clamped;
                t.re.K = K_eff;
                t.re.block = (scheme == RESID_BLOCK) ? block : 0;
                t.re.n_blocks = (scheme == RESID_BLOCK) ? n_blocks_keep : 0;
                t.re.has_w = have_w;
                t.re.cost = estimate_cost(write_base ? &base : nullptr, n_in, n_out, K_eff, write_row_scale);

                if (scheme == RESID_BLOCK) {
                    GGML_ASSERT(n_blocks_keep > 0);
                    t.b_idx.assign((size_t) n_blocks_keep * (size_t) n_out, -1);
                    t.b_val.assign((size_t) block * (size_t) n_blocks_keep * (size_t) n_out, 0.0f);
                } else {
                    t.d_idx.assign((size_t) K_eff * (size_t) n_out, -1);
                    t.d_val.assign((size_t) K_eff * (size_t) n_out, 0.0f);
                }
                if (write_row_scale) {
                    t.d_row_scale.resize((size_t) n_out, 1.0f);
                }

                std::atomic<int64_t> next_col{0};
                std::vector<std::thread> workers;
                workers.reserve((size_t) n_threads);

                const int64_t chunk = 16;

                for (int ti = 0; ti < n_threads; ++ti) {
                    workers.emplace_back([&, ti]() {
                        GGML_UNUSED(ti);

                        std::vector<float> w;
                        std::vector<float> r;
                        std::vector<int32_t> topk;
                        std::vector<int32_t> top_blocks;
                        while (true) {
                            const int64_t col0 = next_col.fetch_add(chunk);
                            if (col0 >= n_out) {
                                break;
                            }

                            const int64_t col1 = std::min<int64_t>(n_out, col0 + chunk);
                            for (int64_t col = col0; col < col1; ++col) {
                                read_column_f32(W, col, w);

                                double dot = 0.0;
                                double nn  = 0.0;

                                if (write_base) {
                                    r.resize((size_t) n_in);

                                    const int64_t L = base.L;
                                    GGML_ASSERT(L > 0);

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

                                            if (write_row_scale) {
                                                const double ws = have_w ? (double) w_scale[(size_t) i] : 1.0;
                                                dot += (double) w[(size_t) i] * (double) base_w * ws * ws;
                                                nn  += (double) base_w * (double) base_w * ws * ws;
                                            }
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

                                                if (write_row_scale) {
                                                    const double ws = have_w ? (double) w_scale[(size_t) i] : 1.0;
                                                    dot += (double) w[(size_t) i] * (double) base_w * ws * ws;
                                                    nn  += (double) base_w * (double) base_w * ws * ws;
                                                }
                                            }
                                        }
                                    }
                                }

                                if (scheme == RESID_BLOCK) {
                                    const std::vector<float> & src = write_base ? r : w;
                                    topk_blocks_energy_weighted(src, have_w ? &w_scale : nullptr, block, n_blocks_keep, top_blocks);

                                    float ss = 1.0f;
                                    if (write_row_scale) {
                                        if (write_base) {
                                            for (int64_t bi = 0; bi < n_blocks_keep; ++bi) {
                                                const int32_t blk = top_blocks[(size_t) bi];
                                                if (blk < 0) {
                                                    continue;
                                                }
                                                const int64_t in0 = (int64_t) blk * block;
                                                const int64_t in1 = std::min<int64_t>(n_in, in0 + block);
                                                for (int64_t ii = in0; ii < in1; ++ii) {
                                                    const double wi = (double) w[(size_t) ii];
                                                    const double di = (double) r[(size_t) ii];
                                                    const double ws = have_w ? (double) w_scale[(size_t) ii] : 1.0;
                                                    dot += wi * di * ws * ws;
                                                    nn  += (2.0 * wi * di - di * di) * ws * ws;
                                                }
                                            }
                                        } else {
                                            for (int64_t bi = 0; bi < n_blocks_keep; ++bi) {
                                                const int32_t blk = top_blocks[(size_t) bi];
                                                if (blk < 0) {
                                                    continue;
                                                }
                                                const int64_t in0 = (int64_t) blk * block;
                                                const int64_t in1 = std::min<int64_t>(n_in, in0 + block);
                                                for (int64_t ii = in0; ii < in1; ++ii) {
                                                    const double wi  = (double) w[(size_t) ii];
                                                    const double whi = (double) w[(size_t) ii];
                                                    const double ws = have_w ? (double) w_scale[(size_t) ii] : 1.0;
                                                    dot += wi * whi * ws * ws;
                                                    nn  += whi * whi * ws * ws;
                                                }
                                            }
                                        }

                                        if (nn > 1e-30) {
                                            ss = (float) (dot / nn);
                                        }
                                        t.d_row_scale[(size_t) col] = ss;
                                    }

                                    for (int64_t bi = 0; bi < n_blocks_keep; ++bi) {
                                        const int32_t blk = bi < (int64_t) top_blocks.size() ? top_blocks[(size_t) bi] : -1;
                                        t.b_idx[(size_t) col * (size_t) n_blocks_keep + (size_t) bi] = blk;

                                        const int64_t in0 = (blk >= 0) ? (int64_t) blk * block : 0;
                                        for (int64_t tt = 0; tt < block; ++tt) {
                                            const int64_t ii = in0 + tt;
                                            float vv = 0.0f;
                                            if (blk >= 0 && ii >= 0 && ii < n_in) {
                                                vv = write_base ? r[(size_t) ii] : w[(size_t) ii];
                                            }
                                            t.b_val[((size_t) col * (size_t) n_blocks_keep + (size_t) bi) * (size_t) block + (size_t) tt] = vv;
                                        }
                                    }
                                } else {
                                    if (write_base) {
                                        topk_abs_weighted(r, have_w ? &w_scale : nullptr, K_eff, topk);
                                    } else {
                                        topk_abs_weighted(w, have_w ? &w_scale : nullptr, K_eff, topk);
                                    }

                                    float ss = 1.0f;
                                    if (write_row_scale) {
                                        if (write_base) {
                                            for (int64_t rr = 0; rr < K_eff; ++rr) {
                                                const int32_t ii = topk[(size_t) rr];
                                                if (ii < 0 || ii >= (int32_t) n_in) {
                                                    continue;
                                                }

                                                const double wi = (double) w[(size_t) ii];
                                                const double di = (double) r[(size_t) ii];
                                                const double ws = have_w ? (double) w_scale[(size_t) ii] : 1.0;

                                                dot += wi * di * ws * ws;
                                                nn  += (2.0 * wi * di - di * di) * ws * ws;
                                            }
                                        } else {
                                            for (int64_t rr = 0; rr < K_eff; ++rr) {
                                                const int32_t ii = topk[(size_t) rr];
                                                if (ii < 0 || ii >= (int32_t) n_in) {
                                                    continue;
                                                }
                                                const double wi  = (double) w[(size_t) ii];
                                                const double whi = (double) w[(size_t) ii];
                                                const double ws = have_w ? (double) w_scale[(size_t) ii] : 1.0;
                                                dot += wi * whi * ws * ws;
                                                nn  += whi * whi * ws * ws;
                                            }
                                        }

                                        if (nn > 1e-30) {
                                            ss = (float) (dot / nn);
                                        }
                                        t.d_row_scale[(size_t) col] = ss;
                                    }

                                    for (int64_t rr = 0; rr < K_eff; ++rr) {
                                        const int32_t ii = topk[(size_t) rr];
                                        t.d_idx[(size_t) col * (size_t) K_eff + (size_t) rr] = ii;

                                        if (ii >= 0 && ii < (int32_t) n_in) {
                                            t.d_val[(size_t) col * (size_t) K_eff + (size_t) rr] = write_base ? r[(size_t) ii] : w[(size_t) ii];
                                        } else {
                                            t.d_val[(size_t) col * (size_t) K_eff + (size_t) rr] = 0.0f;
                                        }
                                    }
                                }
                            }
                        }
                    });
                }

                for (auto & th : workers) {
                    th.join();
                }

                if (t.re.cost.ops_dense > 0.0) {
                    if (write_base) {
                        fprintf(stderr, "  [blk.%" PRId64 ".%s] cost L=%" PRId64 " B=%" PRId64 " ops dense=%.3g base=%.3g delta=%.3g total=%.3g ratio=%.4f\n",
                                il, kind.c_str(),
                                t.re.cost.L, t.re.cost.B,
                                t.re.cost.ops_dense, t.re.cost.ops_base, t.re.cost.ops_delta, t.re.cost.ops_total, t.re.cost.ops_ratio);
                    } else {
                        fprintf(stderr, "  [blk.%" PRId64 ".%s] cost ops dense=%.3g delta=%.3g total=%.3g ratio=%.4f\n",
                                il, kind.c_str(),
                                t.re.cost.ops_dense, t.re.cost.ops_delta, t.re.cost.ops_total, t.re.cost.ops_ratio);
                    }
                }

                if (eval_cols > 0) {
                    const int64_t t0 = ggml_time_us();
                    eval_metrics em;
                    if (scheme == RESID_BLOCK) {
                        em = write_base
                                ? eval_seeddelta_base_block_residual(W, base, t.b_idx, t.b_val, block, n_blocks_keep, write_row_scale ? &t.d_row_scale : nullptr, nullptr, eval_cols, rng)
                                : eval_block_residual(W, t.b_idx, t.b_val, block, n_blocks_keep, write_row_scale ? &t.d_row_scale : nullptr, nullptr, eval_cols, rng);
                    } else {
                        em = write_base
                                ? eval_seeddelta_base_residual(W, base, t.d_idx, t.d_val, write_row_scale ? &t.d_row_scale : nullptr, nullptr, K_eff, eval_cols, rng)
                                : eval_sparse_residual(W, t.d_idx, t.d_val, write_row_scale ? &t.d_row_scale : nullptr, nullptr, K_eff, eval_cols, rng);
                    }
                    const double sec = double(ggml_time_us() - t0) / 1e6;
                    fprintf(stderr, "  [blk.%" PRId64 ".%s] eval cols=%" PRId64 " rel_l2 mean=%.4f p95=%.4f cos mean=%.4f p05=%.4f nr=%.4f (%.1fs)\n",
                            il, kind.c_str(), eval_cols, em.rel_l2_mean, em.rel_l2_p95, em.cos_mean, em.cos_p05, em.norm_ratio_mean, sec);

                    t.re.em = em;

                    if (have_w) {
                        eval_metrics em_w;
                        if (scheme == RESID_BLOCK) {
                            em_w = write_base
                                    ? eval_seeddelta_base_block_residual(W, base, t.b_idx, t.b_val, block, n_blocks_keep, write_row_scale ? &t.d_row_scale : nullptr, &w_scale, eval_cols, rng)
                                    : eval_block_residual(W, t.b_idx, t.b_val, block, n_blocks_keep, write_row_scale ? &t.d_row_scale : nullptr, &w_scale, eval_cols, rng);
                        } else {
                            em_w = write_base
                                    ? eval_seeddelta_base_residual(W, base, t.d_idx, t.d_val, write_row_scale ? &t.d_row_scale : nullptr, &w_scale, K_eff, eval_cols, rng)
                                    : eval_sparse_residual(W, t.d_idx, t.d_val, write_row_scale ? &t.d_row_scale : nullptr, &w_scale, K_eff, eval_cols, rng);
                        }
                        t.re.em_w = em_w;
                        fprintf(stderr, "  [blk.%" PRId64 ".%s] eval_w cols=%" PRId64 " rel_l2 mean=%.4f p95=%.4f cos mean=%.4f p05=%.4f nr=%.4f\n",
                                il, kind.c_str(), eval_cols, em_w.rel_l2_mean, em_w.rel_l2_p95, em_w.cos_mean, em_w.cos_p05, em_w.norm_ratio_mean);
                    }
                }

                if (eval_cols > 0 && eval_x > 0) {
                    const int64_t t0 = ggml_time_us();
                    eval_metrics emx;
                    if (scheme == RESID_BLOCK) {
                        emx = eval_seeddelta_x_block(W, write_base ? &base : nullptr, t.b_idx, t.b_val, block, n_blocks_keep, write_row_scale ? &t.d_row_scale : nullptr, nullptr, eval_cols, eval_x, rng);
                    } else {
                        emx = eval_seeddelta_x(W, write_base ? &base : nullptr, t.d_idx, t.d_val, write_row_scale ? &t.d_row_scale : nullptr, nullptr, K_eff, eval_cols, eval_x, rng);
                    }
                    const double sec = double(ggml_time_us() - t0) / 1e6;
                    fprintf(stderr, "  [blk.%" PRId64 ".%s] eval_x x=%" PRId64 " cols=%" PRId64 " rel_l2 mean=%.4f p95=%.4f cos mean=%.4f p05=%.4f nr=%.4f (%.1fs)\n",
                            il, kind.c_str(), eval_x, eval_cols, emx.rel_l2_mean, emx.rel_l2_p95, emx.cos_mean, emx.cos_p05, emx.norm_ratio_mean, sec);
                    t.re.em_x = emx;
                    t.re.has_x = true;

                    if (have_w) {
                        eval_metrics emx_w;
                        if (scheme == RESID_BLOCK) {
                            emx_w = eval_seeddelta_x_block(W, write_base ? &base : nullptr, t.b_idx, t.b_val, block, n_blocks_keep, write_row_scale ? &t.d_row_scale : nullptr, &w_scale, eval_cols, eval_x, rng);
                        } else {
                            emx_w = eval_seeddelta_x(W, write_base ? &base : nullptr, t.d_idx, t.d_val, write_row_scale ? &t.d_row_scale : nullptr, &w_scale, K_eff, eval_cols, eval_x, rng);
                        }
                        t.re.em_x_w = emx_w;
                        fprintf(stderr, "  [blk.%" PRId64 ".%s] eval_x_w x=%" PRId64 " cols=%" PRId64 " rel_l2 mean=%.4f p95=%.4f cos mean=%.4f p05=%.4f nr=%.4f\n",
                                il, kind.c_str(), eval_x, eval_cols, emx_w.rel_l2_mean, emx_w.rel_l2_p95, emx_w.cos_mean, emx_w.cos_p05, emx_w.norm_ratio_mean);
                    }
                }

                t.seconds = double(ggml_time_us() - t0_total) / 1e6;
                return t;
            };

            std::vector<int64_t> schedule;
            if (cfg.autotune.enabled) {
                schedule = (kind == "ffn_down") ? cfg.autotune.schedule_down : cfg.autotune.schedule_gate_up;
            }
            if (schedule.empty()) {
                schedule.push_back(K_budget);
            }

            // De-dup schedule while keeping order.
            std::unordered_set<int64_t> seen;
            std::vector<int64_t> sched_unique;
            for (int64_t v : schedule) {
                if (v <= 0) continue;
                if (seen.insert(v).second) {
                    sched_unique.push_back(v);
                }
            }

            const int max_iters = cfg.autotune.enabled && cfg.autotune.max_iters > 0
                    ? std::min<int>((int) cfg.autotune.max_iters, (int) sched_unique.size())
                    : (int) sched_unique.size();

            trial_data best_trial;
            bool have_best = false;
            double best_score = -1e30;

            trial_data selected_trial;
            bool have_selected = false;

            std::vector<report_entry::autotune_attempt> attempts;
            if (cfg.autotune.enabled) {
                attempts.reserve((size_t) max_iters);
            }

            for (int it = 0; it < max_iters; ++it) {
                const int64_t K_try = sched_unique[(size_t) it];
                trial_data td = run_trial(K_try);

                td.re.gating_enabled = cfg.gating_enabled;
                td.re.gating_metric_used = metric_kind_to_string(cfg.metric);
                td.re.gating_min_mean = cfg.min_mean;
                td.re.gating_min_p05  = cfg.min_p05;

                const bool metric_needs_x = (cfg.metric == sd_metric_kind::cos_x || cfg.metric == sd_metric_kind::cos_x_w);
                const bool metric_needs_w = (cfg.metric == sd_metric_kind::cos_w || cfg.metric == sd_metric_kind::cos_x_w);
                bool metric_available = true;
                if (cfg.gating_enabled) {
                    if (metric_needs_x && !td.re.has_x) {
                        metric_available = false;
                    }
                    if (metric_needs_w && !have_w) {
                        metric_available = false;
                    }
                }

                const double metric_val = pick_metric_value(td.re, cfg.metric);
                const double metric_p05 = pick_metric_p05(td.re, cfg.metric);
                bool gating_pass = true;
                if (cfg.gating_enabled) {
                    gating_pass = metric_available && metric_val >= cfg.min_mean && metric_p05 >= cfg.min_p05;
                }
                const bool emit = cfg.enabled && (!cfg.gating_enabled || gating_pass);

                td.re.gating_pass = (!cfg.gating_enabled) ? true : (gating_pass && metric_available);
                td.re.gating_value = metric_val;
                td.re.gating_p05 = metric_p05;
                td.re.emit = emit;
                td.re.strip_applied = emit && cfg.strip_dense;
                if (!cfg.gating_enabled) {
                    td.re.decision_reason = "no_gating";
                } else if (!metric_available) {
                    td.re.decision_reason = "metric_unavailable";
                } else if (!gating_pass) {
                    td.re.decision_reason = "gating_fail";
                } else {
                    td.re.decision_reason = "pass_gating";
                }

                if (cfg.autotune.enabled) {
                    report_entry::autotune_attempt at;
                    at.K_budget = td.re.K_budget;
                    at.K_eff = td.re.K;
                    at.n_blocks = td.re.n_blocks;
                    at.metric_value = metric_val;
                    at.metric_p05 = metric_p05;
                    at.pass = emit;
                    at.seconds = td.seconds;
                    attempts.push_back(at);
                }

                if (metric_available && metric_val > best_score) {
                    best_score = metric_val;
                    best_trial = td;
                    have_best = true;
                }

                if (emit) {
                    selected_trial = std::move(td);
                    have_selected = true;
                    break;
                }
            }

            if (!have_best) {
                best_trial.re.layer = il;
                best_trial.re.kind = kind;
                best_trial.re.n_in = n_in;
                best_trial.re.n_out = n_out;
                best_trial.re.emit = false;
                best_trial.re.gating_enabled = cfg.gating_enabled;
                best_trial.re.gating_metric_used = metric_kind_to_string(cfg.metric);
                best_trial.re.gating_min_mean = cfg.min_mean;
                best_trial.re.gating_min_p05 = cfg.min_p05;
                best_trial.re.decision_reason = "metric_unavailable";
                have_best = true;
            }

            trial_data final_td;
            if (have_selected) {
                final_td = std::move(selected_trial);
                final_td.re.autotune_enabled = cfg.autotune.enabled;
                final_td.re.autotune_selected_budget = final_td.re.K_budget;
                final_td.re.autotune_attempts = std::move(attempts);
            } else {
                final_td = std::move(best_trial);
                final_td.re.emit = false;
                final_td.re.strip_applied = false;
                final_td.re.autotune_enabled = cfg.autotune.enabled;
                final_td.re.autotune_selected_budget = 0;
                final_td.re.autotune_attempts = std::move(attempts);
                if (cfg.autotune.enabled) {
                    final_td.re.decision_reason = "autotune_failed_keep_dense";
                }
            }

            report_entry re = std::move(final_td.re);
            std::vector<int32_t> d_idx = std::move(final_td.d_idx);
            std::vector<float>   d_val = std::move(final_td.d_val);
            std::vector<int32_t> b_idx = std::move(final_td.b_idx);
            std::vector<float>   b_val = std::move(final_td.b_val);
            std::vector<float>   d_row_scale = std::move(final_td.d_row_scale);

            const int64_t n_blocks_keep = re.n_blocks;
            const int64_t K_eff = re.K;

            if (!re.emit) {
                report.push_back(std::move(re));
                continue;
            }

            if (re.strip_applied) {
                strip_weights.insert(weight_name);
                any_strip = true;
            }

            report.push_back(std::move(re));

            // Allocate a dedicated ggml context for new tensors to avoid a giant arena.
            const size_t size_idx = scheme == RESID_BLOCK
                    ? (size_t) n_blocks_keep * (size_t) n_out * ggml_type_size(idx_type)
                    : (size_t) K_eff * (size_t) n_out * ggml_type_size(idx_type);
            const size_t size_val = scheme == RESID_BLOCK
                    ? (size_t) block * (size_t) n_blocks_keep * (size_t) n_out * ggml_type_size(val_type)
                    : (size_t) K_eff * (size_t) n_out * ggml_type_size(val_type);
            const size_t size_row_scale = write_row_scale ? (size_t) n_out * sizeof(ggml_fp16_t) : 0;
            const ggml_type perm_type = write_base ? (base.L <= 32768 ? GGML_TYPE_I16 : GGML_TYPE_I32) : GGML_TYPE_I16;
            const size_t size_base_diag = write_base ? (size_t) base.L * (size_t) base.B * sizeof(ggml_fp16_t) * 3 : 0;
            const size_t size_base_perm = write_base ? (size_t) base.L * (size_t) base.B * ggml_type_size(perm_type) : 0;
            const size_t size_base = size_base_diag + size_base_perm;
            const size_t n_tensors_new = 2 + (write_row_scale ? 1 : 0) + (write_base ? 4 : 0);
            const size_t mem_size_sd = ggml_tensor_overhead() * (n_tensors_new + 4) + size_idx + size_val + size_row_scale + size_base;

            ggml_init_params sd_params = { mem_size_sd, nullptr, false };
            ggml_context * ctx_sd = ggml_init(sd_params);
            sd_contexts.push_back(ctx_sd);

            ggml_tensor * t_idx = nullptr;
            ggml_tensor * t_val = nullptr;
            if (scheme == RESID_BLOCK) {
                t_idx = ggml_new_tensor_2d(ctx_sd, idx_type, n_blocks_keep, n_out);
                ggml_set_name(t_idx, b_idx_name.c_str());
                if (idx_type == GGML_TYPE_I16) {
                    auto * dst_i16 = (int16_t *) t_idx->data;
                    for (int64_t col = 0; col < n_out; ++col) {
                        for (int64_t bi = 0; bi < n_blocks_keep; ++bi) {
                            const int32_t blk = b_idx[(size_t) col * (size_t) n_blocks_keep + (size_t) bi];
                            dst_i16[col * n_blocks_keep + bi] = (blk < -32768 || blk > 32767) ? (int16_t) -1 : (int16_t) blk;
                        }
                    }
                } else {
                    std::memcpy(t_idx->data, b_idx.data(), b_idx.size() * sizeof(int32_t));
                }

                t_val = ggml_new_tensor_3d(ctx_sd, val_type, block, n_blocks_keep, n_out);
                ggml_set_name(t_val, b_val_name.c_str());
                if (val_type == GGML_TYPE_F16) {
                    auto * dst_f16 = (ggml_fp16_t *) t_val->data;
                    for (int64_t col = 0; col < n_out; ++col) {
                        for (int64_t bi = 0; bi < n_blocks_keep; ++bi) {
                            ggml_fp32_to_fp16_row(
                                    b_val.data() + ((size_t) col * (size_t) n_blocks_keep + (size_t) bi) * (size_t) block,
                                    dst_f16 + ((size_t) col * (size_t) n_blocks_keep + (size_t) bi) * (size_t) block,
                                    (int) block);
                        }
                    }
                } else {
                    std::memcpy(t_val->data, b_val.data(), b_val.size() * sizeof(float));
                }
            } else {
                t_idx = ggml_new_tensor_2d(ctx_sd, idx_type, K_eff, n_out);
                ggml_set_name(t_idx, d_idx_name.c_str());
                if (idx_type == GGML_TYPE_I16) {
                    auto * dst_i16 = (int16_t *) t_idx->data;
                    for (int64_t col = 0; col < n_out; ++col) {
                        for (int64_t r = 0; r < K_eff; ++r) {
                            const int32_t ii = d_idx[(size_t) col * (size_t) K_eff + (size_t) r];
                            dst_i16[col * K_eff + r] = (ii < -32768 || ii > 32767) ? (int16_t) -1 : (int16_t) ii;
                        }
                    }
                } else {
                    std::memcpy(t_idx->data, d_idx.data(), d_idx.size() * sizeof(int32_t));
                }

                t_val = ggml_new_tensor_2d(ctx_sd, val_type, K_eff, n_out);
                ggml_set_name(t_val, d_val_name.c_str());
                if (val_type == GGML_TYPE_F16) {
                    auto * dst_f16 = (ggml_fp16_t *) t_val->data;
                    for (int64_t col = 0; col < n_out; ++col) {
                        ggml_fp32_to_fp16_row(d_val.data() + (size_t) col * (size_t) K_eff, dst_f16 + (size_t) col * (size_t) K_eff, (int) K_eff);
                    }
                } else {
                    std::memcpy(t_val->data, d_val.data(), d_val.size() * sizeof(float));
                }
            }

            pending_tensor_set pts;
            pts.ctx = ctx_sd;
            pts.tensors.push_back(t_idx);
            pts.tensors.push_back(t_val);
            n_added += 2;

            if (write_row_scale) {
                ggml_tensor * t_rs = ggml_new_tensor_1d(ctx_sd, GGML_TYPE_F16, n_out);
                ggml_set_name(t_rs, d_row_scale_name.c_str());
                auto * rs_f16 = (ggml_fp16_t *) t_rs->data;
                ggml_fp32_to_fp16_row(d_row_scale.data(), rs_f16, n_out);
                pts.tensors.push_back(t_rs);
                n_added += 1;
            }

            if (write_base) {
                const std::string base_d1_name = "blk." + std::to_string(il) + "." + kind + ".base_d1";
                const std::string base_d2_name = "blk." + std::to_string(il) + "." + kind + ".base_d2";
                const std::string base_d3_name = "blk." + std::to_string(il) + "." + kind + ".base_d3";
                const std::string base_perm1_name = "blk." + std::to_string(il) + "." + kind + ".base_perm1";

                ggml_tensor * t_d1 = ggml_new_tensor_2d(ctx_sd, GGML_TYPE_F16, base.L, base.B);
                ggml_set_name(t_d1, base_d1_name.c_str());
                ggml_tensor * t_d2 = ggml_new_tensor_2d(ctx_sd, GGML_TYPE_F16, base.L, base.B);
                ggml_set_name(t_d2, base_d2_name.c_str());
                ggml_tensor * t_d3 = ggml_new_tensor_2d(ctx_sd, GGML_TYPE_F16, base.L, base.B);
                ggml_set_name(t_d3, base_d3_name.c_str());
                ggml_tensor * t_p1 = ggml_new_tensor_2d(ctx_sd, perm_type, base.L, base.B);
                ggml_set_name(t_p1, base_perm1_name.c_str());

                auto * d1_f16 = (ggml_fp16_t *) t_d1->data;
                auto * d2_f16 = (ggml_fp16_t *) t_d2->data;
                auto * d3_f16 = (ggml_fp16_t *) t_d3->data;
                for (int64_t b = 0; b < base.B; ++b) {
                    ggml_fp32_to_fp16_row(base.d1.data() + (size_t) b * (size_t) base.L, d1_f16 + (size_t) b * (size_t) base.L, base.L);
                    ggml_fp32_to_fp16_row(base.d2.data() + (size_t) b * (size_t) base.L, d2_f16 + (size_t) b * (size_t) base.L, base.L);
                    ggml_fp32_to_fp16_row(base.d3.data() + (size_t) b * (size_t) base.L, d3_f16 + (size_t) b * (size_t) base.L, base.L);
                }

                pts.tensors.push_back(t_d1);
                pts.tensors.push_back(t_d2);
                pts.tensors.push_back(t_d3);
                pts.tensors.push_back(t_p1);
                if (perm_type == GGML_TYPE_I16) {
                    auto * p1_i16 = (int16_t *) t_p1->data;
                    for (size_t i = 0; i < base.perm1.size(); ++i) {
                        const int32_t v = base.perm1[i];
                        p1_i16[i] = (v < 0 || v > 32767) ? (int16_t) 0 : (int16_t) v;
                    }
                } else {
                    std::memcpy(t_p1->data, base.perm1.data(), base.perm1.size() * sizeof(int32_t));
                }
                n_added += 4;
            }

            pending.push_back(std::move(pts));
        }
    }

    if (n_added == 0) {
        fprintf(stderr, "no new SeedΔ tensors added\n");
    }

    gguf_context * dst = gguf_init_empty();
    gguf_set_kv(dst, src);

    // Add original tensors, skipping those strippeados.
    for (int64_t ti = 0; ti < n_tensors; ++ti) {
        const char * name = gguf_get_tensor_name(src, ti);
        if (any_strip && strip_weights.count(name) > 0) {
            continue;
        }
        ggml_tensor * t = ggml_get_tensor(ctx_data, name);
        if (!t) {
            fprintf(stderr, "warning: missing tensor %s in ctx_data\n", name);
            continue;
        }
        gguf_add_tensor(dst, t);
    }

    for (const auto & pts : pending) {
        for (ggml_tensor * t : pts.tensors) {
            gguf_add_tensor(dst, t);
        }
    }

    gguf_set_val_bool(dst, "seeddelta.enabled", n_added > 0);
    gguf_set_val_u32(dst, "seeddelta.version", 1);
    gguf_set_val_u32(dst, "seeddelta.scheme", (uint32_t) scheme);
    gguf_set_val_bool(dst, "seeddelta.row_scale", write_row_scale);
    gguf_set_val_u32(dst, "seeddelta.resid.K", (uint32_t) K_default);
    gguf_set_val_bool(dst, "seeddelta.strip_dense", any_strip);
    if (scheme == RESID_BLOCK) {
        gguf_set_val_u32(dst, "seeddelta.resid.block", (uint32_t) block);
    }
    gguf_set_val_bool(dst, "seeddelta.resid.K_variable", K_variable);
    if (K_variable) {
        gguf_set_val_u32(dst, "seeddelta.resid.K_gate", (uint32_t) K_gate_eff);
        gguf_set_val_u32(dst, "seeddelta.resid.K_up",   (uint32_t) K_up_eff);
        gguf_set_val_u32(dst, "seeddelta.resid.K_down", (uint32_t) K_down_eff);
    }
    gguf_set_val_bool(dst, "seeddelta.base.enabled", write_base);
    if (write_base) {
        gguf_set_val_u32(dst, "seeddelta.base.kind", 1);  // hadamard_acdc_stack
        gguf_set_val_u32(dst, "seeddelta.base.depth", 2); // D3*H*D2*H*D1
        gguf_set_val_u32(dst, "seeddelta.base.R", 1);
        gguf_set_val_u32(dst, "seeddelta.base.max_samples", (uint32_t) std::max<int64_t>(0, base_max_samples));
        gguf_set_val_u32(dst, "seeddelta.base.perm_trials", (uint32_t) std::max(1, base_perm_trials));
    }

    // Per-tensor audit metadata (minimal, per GGUF kv).
    for (const auto & re : report) {
        const std::string prefix = "seeddelta.blk." + std::to_string(re.layer) + "." + re.kind;
        gguf_set_val_bool(dst, (prefix + ".enabled").c_str(), re.emit);
        gguf_set_val_bool(dst, (prefix + ".gating_pass").c_str(), re.gating_pass);
        gguf_set_val_bool(dst, (prefix + ".strip_dense").c_str(), re.strip_applied);
        gguf_set_val_u32(dst, (prefix + ".K").c_str(), (uint32_t) std::max<int64_t>(0, re.K));
    }

    printf("writing %s with %" PRId64 " new tensors\n", out_fname.c_str(), n_added);
    if (!gguf_write_to_file(dst, out_fname.c_str(), false)) {
        fprintf(stderr, "seeddelta-build: failed to write %s\n", out_fname.c_str());
        return 1;
    }

    if (!report_json.empty()) {
        std::ofstream out(report_json);
        if (!out) {
            fprintf(stderr, "seeddelta-build: failed to open report path %s\n", report_json.c_str());
            return 1;
        }

        auto json_escape = [](const std::string & s) -> std::string {
            std::string o;
            o.reserve(s.size() + 8);
            for (char c : s) {
                switch (c) {
                    case '\\': o += "\\\\"; break;
                    case '"':  o += "\\\""; break;
                    case '\n': o += "\\n"; break;
                    case '\r': o += "\\r"; break;
                    case '\t': o += "\\t"; break;
                    default:   o += c; break;
                }
            }
            return o;
        };

        out << "{\n";
        out << "  \"input\": \"" << json_escape(in_fname) << "\",\n";
        out << "  \"output\": \"" << json_escape(out_fname) << "\",\n";
        out << "  \"imatrix\": " << (have_imatrix ? "true" : "false") << ",\n";
        out << "  \"base\": " << (write_base ? "true" : "false") << ",\n";
        out << "  \"policy_file\": \"" << (policy_file.empty() ? "" : json_escape(policy_file)) << "\",\n";
        out << "  \"policy_hash\": \"" << json_escape(policy_hash) << "\",\n";
        if (write_base) {
            out << "  \"base_kind\": \"xor_circulant\",\n";
            out << "  \"base_max_samples\": " << base_max_samples << ",\n";
            out << "  \"base_perm_trials\": " << base_perm_trials << ",\n";
        }
        if (have_imatrix) {
            out << "  \"imatrix_file\": \"" << json_escape(imatrix_file) << "\",\n";
            out << "  \"imatrix_eps\": " << imatrix_eps << ",\n";
            out << "  \"imatrix_power\": " << imatrix_power << ",\n";
            out << "  \"imatrix_datasets\": [";
            for (size_t i = 0; i < imatrix_datasets.size(); ++i) {
                if (i) out << ", ";
                out << "\"" << json_escape(imatrix_datasets[i]) << "\"";
            }
            out << "],\n";
        }
        out << "  \"resid\": {\n";
        out << "    \"scheme\": \"" << (scheme == RESID_BLOCK ? "block" : "coo") << "\",\n";
        if (scheme == RESID_BLOCK) {
            out << "    \"block\": " << block << ",\n";
        }
        out << "    \"K\": " << K_default << ",\n";
        out << "    \"K_gate\": " << K_gate_eff << ",\n";
        out << "    \"K_up\": " << K_up_eff << ",\n";
        out << "    \"K_down\": " << K_down_eff << ",\n";
        out << "    \"idx_type\": \"" << json_escape(idx_type_str) << "\",\n";
        out << "    \"val_type\": \"" << json_escape(val_type_str) << "\",\n";
        out << "    \"row_scale\": " << (write_row_scale ? "true" : "false") << "\n";
        out << "  },\n";
        out << "  \"eval_cols\": " << eval_cols << ",\n";
        out << "  \"eval_x\": " << eval_x << ",\n";
        out << "  \"weights\": [\n";

        for (size_t i = 0; i < report.size(); ++i) {
            const auto & e = report[i];
            out << "    {\n";
            out << "      \"layer\": " << e.layer << ",\n";
            out << "      \"kind\": \"" << json_escape(e.kind) << "\",\n";
            out << "      \"n_in\": " << e.n_in << ",\n";
            out << "      \"n_out\": " << e.n_out << ",\n";
            out << "      \"K_budget\": " << e.K_budget << ",\n";
            out << "      \"K\": " << e.K << ",\n";
            out << "      \"block\": " << e.block << ",\n";
            out << "      \"n_blocks\": " << e.n_blocks << ",\n";
            out << "      \"has_w\": " << (e.has_w ? "true" : "false") << ",\n";
            out << "      \"has_x\": " << (e.has_x ? "true" : "false") << ",\n";
            out << "      \"base_L\": " << e.cost.L << ",\n";
            out << "      \"base_B\": " << e.cost.B << ",\n";
            out << "      \"ops_dense\": " << e.cost.ops_dense << ",\n";
            out << "      \"ops_base\": " << e.cost.ops_base << ",\n";
            out << "      \"ops_delta\": " << e.cost.ops_delta << ",\n";
            out << "      \"ops_row_scale\": " << e.cost.ops_row_scale << ",\n";
            out << "      \"ops_total\": " << e.cost.ops_total << ",\n";
            out << "      \"ops_ratio\": " << e.cost.ops_ratio << ",\n";
            out << "      \"rel_l2_mean\": " << e.em.rel_l2_mean << ",\n";
            out << "      \"rel_l2_p95\": " << e.em.rel_l2_p95 << ",\n";
            out << "      \"cos_mean\": " << e.em.cos_mean << ",\n";
            out << "      \"cos_p05\": " << e.em.cos_p05 << ",\n";
            out << "      \"norm_ratio_mean\": " << e.em.norm_ratio_mean << ",\n";
            out << "      \"rel_l2_mean_w\": " << e.em_w.rel_l2_mean << ",\n";
            out << "      \"rel_l2_p95_w\": " << e.em_w.rel_l2_p95 << ",\n";
            out << "      \"cos_mean_w\": " << e.em_w.cos_mean << ",\n";
            out << "      \"cos_p05_w\": " << e.em_w.cos_p05 << ",\n";
            out << "      \"norm_ratio_mean_w\": " << e.em_w.norm_ratio_mean << ",\n";
            out << "      \"rel_l2_mean_x\": " << e.em_x.rel_l2_mean << ",\n";
            out << "      \"rel_l2_p95_x\": " << e.em_x.rel_l2_p95 << ",\n";
            out << "      \"cos_mean_x\": " << e.em_x.cos_mean << ",\n";
            out << "      \"cos_p05_x\": " << e.em_x.cos_p05 << ",\n";
            out << "      \"norm_ratio_mean_x\": " << e.em_x.norm_ratio_mean << ",\n";
            out << "      \"rel_l2_mean_x_w\": " << e.em_x_w.rel_l2_mean << ",\n";
            out << "      \"rel_l2_p95_x_w\": " << e.em_x_w.rel_l2_p95 << ",\n";
            out << "      \"cos_mean_x_w\": " << e.em_x_w.cos_mean << ",\n";
            out << "      \"cos_p05_x_w\": " << e.em_x_w.cos_p05 << ",\n";
            out << "      \"norm_ratio_mean_x_w\": " << e.em_x_w.norm_ratio_mean << ",\n";
            out << "      \"gating_enabled\": " << (e.gating_enabled ? "true" : "false") << ",\n";
            out << "      \"gating_metric\": \"" << json_escape(e.gating_metric_used) << "\",\n";
            out << "      \"gating_value\": " << e.gating_value << ",\n";
            out << "      \"gating_p05\": " << e.gating_p05 << ",\n";
            out << "      \"gating_min_mean\": " << e.gating_min_mean << ",\n";
            out << "      \"gating_min_p05\": " << e.gating_min_p05 << ",\n";
            out << "      \"gating_pass\": " << (e.gating_pass ? "true" : "false") << ",\n";
            out << "      \"emit\": " << (e.emit ? "true" : "false") << ",\n";
            out << "      \"strip_dense\": " << (e.strip_applied ? "true" : "false") << ",\n";
            out << "      \"decision\": \"" << json_escape(e.decision_reason) << "\",\n";
            out << "      \"autotune_enabled\": " << (e.autotune_enabled ? "true" : "false") << ",\n";
            out << "      \"autotune_selected_budget\": " << e.autotune_selected_budget << ",\n";
            out << "      \"autotune_attempts\": [";
            for (size_t ai = 0; ai < e.autotune_attempts.size(); ++ai) {
                const auto & a = e.autotune_attempts[ai];
                out << (ai == 0 ? "\n" : ",\n");
                out << "        {";
                out << "\"K_budget\": " << a.K_budget << ", ";
                out << "\"K_eff\": " << a.K_eff << ", ";
                out << "\"n_blocks\": " << a.n_blocks << ", ";
                out << "\"metric_value\": " << a.metric_value << ", ";
                out << "\"metric_p05\": " << a.metric_p05 << ", ";
                out << "\"pass\": " << (a.pass ? "true" : "false") << ", ";
                out << "\"seconds\": " << a.seconds;
                out << "}";
            }
            if (!e.autotune_attempts.empty()) {
                out << "\n      ";
            }
            out << "]\n";

            out << "    }" << (i + 1 < report.size() ? "," : "") << "\n";
        }
        out << "  ]\n";
        out << "}\n";

        fprintf(stderr, "seeddelta-build: wrote report %s\n", report_json.c_str());
    }

    if (!policy_export_file.empty()) {
        std::vector<sd_tensor_decision> decisions;
        decisions.reserve(report.size());
        for (const auto & e : report) {
            sd_tensor_decision d;
            d.layer = e.layer;
            d.kind = e.kind;
            d.enabled = e.emit;
            d.strip_dense = e.strip_applied;
            d.block = e.block;
            d.K_budget = e.K_budget;
            decisions.push_back(std::move(d));
        }

        auto pres = sd_policy_export_write_canonical(policy_export_file, decisions);
        if (!pres.ok) {
            fprintf(stderr, "seeddelta-build: failed to export policy: %s\n", pres.error.c_str());
            return 1;
        }
        fprintf(stderr, "seeddelta-build: wrote policy export %s\n", policy_export_file.c_str());
    }

    return 0;
}
