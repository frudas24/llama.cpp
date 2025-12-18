#include "sd_ffn_proxy.h"
#include "sd_eval.h"

#include <algorithm>
#include <cmath>
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

bool eval_ffn_proxy_coo_replace_one(
        const std::string & kind,
        const ggml_tensor * W_gate,
        const ggml_tensor * W_up,
        const ggml_tensor * W_down,
        const std::vector<int32_t> & d_idx,
        const std::vector<float> & d_val,
        const base_fit * base,
        bool write_base,
        int64_t K,
        int64_t eval_x,
        int64_t eval_cols,
        int seed,
        ffn_proxy_metrics & out) {
    if (!W_gate || !W_up || !W_down) {
        return false;
    }
    if (eval_x <= 0) {
        return false;
    }

    if (kind == "ffn_down") {
        // v0 proxy: not supported for down (would require approximating W_down weights tile-wise).
        return false;
    }

    const int64_t n_in = W_gate->ne[0];
    const int64_t n_hidden = W_gate->ne[1];
    const int64_t n_out = W_down->ne[1];

    const int64_t n_x = std::min<int64_t>(eval_x, 128);
    const int64_t n_hidden_samp = std::min<int64_t>(n_hidden, eval_cols > 0 ? eval_cols : 128);
    const int64_t n_out_samp = std::min<int64_t>(n_out, eval_cols > 0 ? eval_cols : 128);

    const bool have_base = write_base && base && base->L > 0 && base->B > 0;
    const bool is_tall = n_out >= n_in;
    const int64_t L = have_base ? base->L : 0;
    const int64_t B = have_base ? base->B : 0;

    std::mt19937 rng(seed);
    std::vector<int64_t> hid_idx(n_hidden);
    std::iota(hid_idx.begin(), hid_idx.end(), 0);
    std::shuffle(hid_idx.begin(), hid_idx.end(), rng);
    hid_idx.resize((size_t) n_hidden_samp);

    std::vector<int64_t> out_idx(n_out);
    std::iota(out_idx.begin(), out_idx.end(), 0);
    std::shuffle(out_idx.begin(), out_idx.end(), rng);
    out_idx.resize((size_t) n_out_samp);

    std::normal_distribution<float> nd(0.0f, 1.0f);

    std::vector<float> x((size_t) n_in);
    std::vector<float> col;
    std::vector<double> y_true;
    std::vector<double> y_hat;
    y_true.reserve((size_t) n_out_samp);
    y_hat.reserve((size_t) n_out_samp);

    std::vector<double> gate_true((size_t) n_hidden_samp);
    std::vector<double> gate_hat((size_t) n_hidden_samp);
    std::vector<double> up_true((size_t) n_hidden_samp);
    std::vector<double> up_hat((size_t) n_hidden_samp);

    std::vector<double> l2_all;
    std::vector<double> cos_all;
    std::vector<double> log_nr_all;
    l2_all.reserve((size_t) n_x);
    cos_all.reserve((size_t) n_x);
    log_nr_all.reserve((size_t) n_x);

    auto silu = [](double v) -> double {
        return v / (1.0 + std::exp(-v));
    };

    auto dot_dense_col = [&](const ggml_tensor * W, int64_t col_idx, const std::vector<float> & xv) -> double {
        read_column_f32(W, col_idx, col);
        double acc = 0.0;
        for (int64_t i = 0; i < n_in; ++i) {
            acc += (double) col[(size_t) i] * (double) xv[(size_t) i];
        }
        return acc;
    };

    auto dot_delta_col = [&](int64_t col_idx, const std::vector<float> & xv) -> double {
        const int32_t * idx_col = d_idx.data() + (size_t) col_idx * (size_t) K;
        const float *   val_col = d_val.data() + (size_t) col_idx * (size_t) K;
        double acc = 0.0;
        for (int64_t r = 0; r < K; ++r) {
            const int32_t ii = idx_col[r];
            if (ii < 0 || ii >= (int32_t) n_in) continue;
            acc += (double) val_col[r] * (double) xv[(size_t) ii];
        }
        return acc;
    };

    // Base helpers for gate/up (same input dim as n_in).
    std::vector<std::vector<int64_t>> cols_in_block;
    std::vector<int64_t> pos_in_block;
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
            cols_in_block.resize((size_t) B);
            pos_in_block.resize((size_t) n_hidden_samp);
            for (int64_t hi = 0; hi < n_hidden_samp; ++hi) {
                const int64_t col_idx = hid_idx[(size_t) hi];
                const int64_t b = col_idx / L;
                if (b >= 0 && b < B) {
                    cols_in_block[(size_t) b].push_back(hi);
                    pos_in_block[(size_t) hi] = col_idx - b * L;
                }
            }
        } else {
            x_chunk.resize((size_t) L);
        }
    }

    auto base_col_dot = [&](int64_t col_idx, const std::vector<float> & xv) -> double {
        if (!have_base) return 0.0;
        if (is_tall) {
            for (int64_t i = 0; i < n_in; ++i) x_hat[(size_t) i] = xv[(size_t) i];
            for (int64_t i = n_in; i < L; ++i) x_hat[(size_t) i] = 0.0f;
            const int64_t b = col_idx / L;
            const int64_t p = col_idx - b * L;
            if (b < 0 || b >= B) return 0.0;
            apply_base_block_f32(x_hat.data(), v.data(), tmp.data(), *base, b);
            return (p >= 0 && p < L) ? (double) v[(size_t) p] : 0.0;
        } else {
            std::fill(y_base.begin(), y_base.end(), 0.0f);
            for (int64_t b = 0; b < B; ++b) {
                const int64_t in0 = b * L;
                const int64_t in1 = std::min<int64_t>(n_in, in0 + L);
                for (int64_t i = 0; i < in1 - in0; ++i) x_chunk[(size_t) i] = xv[(size_t) (in0 + i)];
                for (int64_t i = in1 - in0; i < L; ++i) x_chunk[(size_t) i] = 0.0f;
                apply_base_block_f32(x_chunk.data(), v.data(), tmp.data(), *base, b);
                for (int64_t i = 0; i < L; ++i) y_base[(size_t) i] += v[(size_t) i];
            }
            return (col_idx >= 0 && col_idx < L) ? (double) y_base[(size_t) col_idx] : 0.0;
        }
    };

    for (int64_t xi = 0; xi < n_x; ++xi) {
        for (int64_t i = 0; i < n_in; ++i) {
            x[(size_t) i] = nd(rng);
        }

        for (int64_t hi = 0; hi < n_hidden_samp; ++hi) {
            const int64_t h = hid_idx[(size_t) hi];
            const double g_true = dot_dense_col(W_gate, h, x);
            const double u_true = dot_dense_col(W_up,   h, x);
            gate_true[(size_t) hi] = g_true;
            up_true[(size_t) hi]   = u_true;

            double g_hat = g_true;
            double u_hat = u_true;
            if (kind == "ffn_gate") {
                g_hat = (have_base ? base_col_dot(h, x) : 0.0) + dot_delta_col(h, x);
            } else if (kind == "ffn_up") {
                u_hat = (have_base ? base_col_dot(h, x) : 0.0) + dot_delta_col(h, x);
            }
            gate_hat[(size_t) hi] = g_hat;
            up_hat[(size_t) hi]   = u_hat;
        }

        y_true.assign((size_t) n_out_samp, 0.0);
        y_hat.assign((size_t) n_out_samp, 0.0);

        for (int64_t oi = 0; oi < n_out_samp; ++oi) {
            const int64_t outcol = out_idx[(size_t) oi];
            read_column_f32(W_down, outcol, col);
            double yt = 0.0, yh = 0.0;
            for (int64_t hi = 0; hi < n_hidden_samp; ++hi) {
                const int64_t h = hid_idx[(size_t) hi];
                const double a_true = silu(gate_true[(size_t) hi]);
                const double a_hat  = silu(gate_hat[(size_t) hi]);
                const double up_t = up_true[(size_t) hi];
                const double up_h = up_hat[(size_t) hi];
                const double w = (h < (int64_t) col.size()) ? (double) col[(size_t) h] : 0.0;
                yt += a_true * up_t * w;
                yh += a_hat  * up_h * w;
            }
            y_true[(size_t) oi] = yt;
            y_hat[(size_t) oi]  = yh;
        }

        double dot = 0.0;
        double n1 = 0.0;
        double n2 = 0.0;
        double l2 = 0.0;
        for (size_t i = 0; i < y_true.size(); ++i) {
            const double a = y_true[i];
            const double b = y_hat[i];
            dot += a * b;
            n1  += a * a;
            n2  += b * b;
            const double d = a - b;
            l2 += d * d;
        }
        const double denom = std::sqrt(std::max(n1, 1e-20)) * std::sqrt(std::max(n2, 1e-20));
        const double cos = denom > 0.0 ? dot / denom : 0.0;
        const double l2n = std::sqrt(std::max(l2, 0.0));
        const double nr = (n1 > 0.0) ? std::sqrt(std::max(n2, 1e-20)) / std::sqrt(std::max(n1, 1e-20)) : 0.0;

        cos_all.push_back(cos);
        l2_all.push_back(l2n);
        if (nr > 0.0) {
            log_nr_all.push_back(std::log(nr));
        }
    }

    out.eval_x = n_x;
    out.eval_out = n_out_samp;
    out.cos_mean = cos_all.empty() ? 0.0 : std::accumulate(cos_all.begin(), cos_all.end(), 0.0) / (double) cos_all.size();
    out.cos_p05  = percentile_vec(cos_all, 0.05);
    out.l2_mean  = l2_all.empty() ? 0.0 : std::accumulate(l2_all.begin(), l2_all.end(), 0.0) / (double) l2_all.size();
    out.l2_p95   = percentile_vec(l2_all, 0.95);
    out.log_norm_ratio_mean = log_nr_all.empty() ? 0.0 : std::accumulate(log_nr_all.begin(), log_nr_all.end(), 0.0) / (double) log_nr_all.size();
    out.log_norm_ratio_p95  = percentile_vec(log_nr_all, 0.95);
    return true;
}
