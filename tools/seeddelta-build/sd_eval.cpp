#include "sd_eval.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

static double percentile_vec(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const double x = p * double(v.size() - 1);
    const size_t i = (size_t) x;
    const size_t j = std::min(i + 1, v.size() - 1);
    const double a = x - double(i);
    return v[i] * (1.0 - a) + v[j] * a;
}

void read_column_f32(const ggml_tensor * W, int64_t j, std::vector<float> & out) {
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

void topk_abs_weighted(
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
void topk_blocks_energy_weighted(
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

eval_metrics eval_sparse_residual(
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

eval_metrics eval_block_residual(
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

void apply_base_block_f32(
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

// Fit a simple Hadamard base for a weight tensor W (stored as [n_in, n_out] in GGUF):
//   W0 = D3 * H * D2 * H * D1, with D1=D3=I and D2 chosen so that W0[p,q] ~= h2[p^q]
// per block. This is the least-squares projection onto XOR-circulant matrices (per block),
// estimated from at most max_samples output columns.
base_fit fit_base_xor_circulant(
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

eval_metrics eval_seeddelta_base_residual(
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

eval_metrics eval_seeddelta_base_block_residual(
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

eval_metrics eval_seeddelta_x(
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

eval_metrics eval_seeddelta_x_block(
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
