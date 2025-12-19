#include "sc_eval.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

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

eval_metrics eval_reconstruction_sign(
        const ggml_tensor * W,
        const std::vector<float> & D,
        int64_t M,
        int k,
        const std::vector<int16_t> & codes,
        const std::vector<ggml_fp16_t> * vals,
        const std::vector<ggml_fp16_t> * row_scale,
        int64_t eval_cols,
        std::mt19937 & rng,
        const std::vector<float> * w_scale_sqrt) {
    eval_metrics out;

    const int64_t n_in  = W->ne[0];
    const int64_t n_out = W->ne[1];

    if (eval_cols <= 0 || n_out <= 0 || n_in <= 0 || M <= 0 || k <= 0) {
        return out;
    }

    eval_cols = std::min<int64_t>(eval_cols, n_out);

    std::vector<int64_t> idx(n_out);
    std::iota(idx.begin(), idx.end(), 0);
    std::shuffle(idx.begin(), idx.end(), rng);
    idx.resize(eval_cols);

    std::vector<float> w;
    std::vector<float> w_hat(n_in, 0.0f);

    std::vector<double> rel_l2;
    std::vector<double> cos;
    rel_l2.reserve(eval_cols);
    cos.reserve(eval_cols);

    const bool do_weighted = w_scale_sqrt && (int64_t) w_scale_sqrt->size() == n_in;
    std::vector<double> rel_l2_w;
    std::vector<double> cos_w;
    if (do_weighted) {
        rel_l2_w.reserve(eval_cols);
        cos_w.reserve(eval_cols);
    }

    for (int64_t ci = 0; ci < eval_cols; ++ci) {
        const int64_t col = idx[ci];

        read_column_f32(W, col, w);
        std::fill(w_hat.begin(), w_hat.end(), 0.0f);

        const int16_t * codes_col = codes.data() + (size_t) col * (size_t) k;
        const ggml_fp16_t * vals_col = vals ? (vals->data() + (size_t) col * (size_t) k) : nullptr;
        for (int ti = 0; ti < k; ++ti) {
            const int16_t code = codes_col[ti];
            if (code == 0) {
                continue;
            }

            const float sign = code > 0 ? 1.0f : -1.0f;
            const float coef = vals_col ? ggml_fp16_to_fp32(vals_col[ti]) : 1.0f;
            const int64_t atom = (int64_t) std::abs(code) - 1;
            if (atom < 0 || atom >= M) {
                continue;
            }

            const float * dcol = D.data() + atom * n_in;
            for (int64_t i = 0; i < n_in; ++i) {
                w_hat[i] += (sign * coef) * dcol[i];
            }
        }

        const double scale = row_scale ? (double) ggml_fp16_to_fp32((*row_scale)[(size_t) col]) : 1.0;

        double w_norm2  = 0.0;
        double wh_norm2 = 0.0;
        double err2     = 0.0;
        double dot      = 0.0;

        double w_norm2_w  = 0.0;
        double wh_norm2_w = 0.0;
        double err2_w     = 0.0;
        double dot_w      = 0.0;

        for (int64_t i = 0; i < n_in; ++i) {
            const double wi  = w[i];
            const double wHi = scale * (double) w_hat[i];
            const double di  = wi - wHi;
            w_norm2  += wi  * wi;
            wh_norm2 += wHi * wHi;
            err2     += di  * di;
            dot      += wi  * wHi;

            if (do_weighted) {
                const double s = (*w_scale_sqrt)[(size_t) i];
                const double wi_w  = wi  * s;
                const double wHi_w = wHi * s;
                const double di_w  = wi_w - wHi_w;
                w_norm2_w  += wi_w  * wi_w;
                wh_norm2_w += wHi_w * wHi_w;
                err2_w     += di_w  * di_w;
                dot_w      += wi_w  * wHi_w;
            }
        }

        const double denom_w = std::sqrt(std::max(w_norm2,  1e-20));
        const double denom_h = std::sqrt(std::max(wh_norm2, 1e-20));
        rel_l2.push_back(std::sqrt(std::max(err2, 0.0)) / denom_w);
        cos.push_back(dot / (denom_w * denom_h));

        if (do_weighted) {
            const double denom_w_w = std::sqrt(std::max(w_norm2_w,  1e-20));
            const double denom_h_w = std::sqrt(std::max(wh_norm2_w, 1e-20));
            rel_l2_w.push_back(std::sqrt(std::max(err2_w, 0.0)) / denom_w_w);
            cos_w.push_back(dot_w / (denom_w_w * denom_h_w));
        }
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
    for (size_t i = 0; i < rel_l2.size(); ++i) sum_rel += rel_l2[i];
    for (size_t i = 0; i < cos.size();    ++i) sum_cos += cos[i];

    out.rel_l2_mean = rel_l2.empty() ? 0.0 : sum_rel / double(rel_l2.size());
    out.rel_l2_p95  = percentile(rel_l2, 0.95);
    out.cos_mean    = cos.empty()    ? 0.0 : sum_cos / double(cos.size());
    out.cos_p05     = percentile(cos, 0.05);

    if (do_weighted) {
        double sum_rel_w = 0.0;
        double sum_cos_w = 0.0;
        for (size_t i = 0; i < rel_l2_w.size(); ++i) sum_rel_w += rel_l2_w[i];
        for (size_t i = 0; i < cos_w.size();    ++i) sum_cos_w += cos_w[i];

        out.rel_l2_mean_w = rel_l2_w.empty() ? 0.0 : sum_rel_w / double(rel_l2_w.size());
        out.rel_l2_p95_w  = percentile(rel_l2_w, 0.95);
        out.cos_mean_w    = cos_w.empty()    ? 0.0 : sum_cos_w / double(cos_w.size());
        out.cos_p05_w     = percentile(cos_w, 0.05);
    }

    return out;
}
