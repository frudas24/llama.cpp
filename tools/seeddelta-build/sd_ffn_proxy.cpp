#include "sd_ffn_proxy.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <string>
#include <vector>

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
    std::string metric_used;
    double gating_value = 0.0;
    double gating_p05 = 0.0;
    double gating_min_mean = 0.0;
    double gating_min_p05 = 0.0;
    double target_tau_mean = 0.0;
    double target_tau_p05 = 0.0;
    bool emit = true;
    bool strip_applied = false;
    std::string decision_reason;
    std::string reject_reason;
    double stack_cost_delta = 0.0;
    double stack_cost_total = 0.0;

    // Tiled-K metadata (v1: may be empty if no tiles)
    int64_t tile_rows = 0;
    int64_t tile_rows_align = 0;
    int64_t k_total_per_tensor = 0;
    std::vector<int64_t> k_levels;
    std::vector<int64_t> k_per_tile; // post-round, dense per tile (row order)
    int64_t unique_k_count = 0;
    int64_t tiles_rounded_count = 0;
    double tiles_rounded_pct = 0.0;
    int64_t tiles_dropped_count = 0;
    double tiles_dropped_pct = 0.0;
    bool k_custom_used = false;
    struct k_stats {
        double min = 0.0;
        double max = 0.0;
        double mean = 0.0;
        bool has = false;
    };
    k_stats k_requested_stats;
    k_stats k_selected_stats;

    // FFN proxy metrics (v0: replace-only-current-tensor; logging only)
    bool ffn_proxy_available = false;
    std::string ffn_proxy_reason;
    std::string ffn_proxy_scope;
    bool ffn_proxy_base_used = false;
    int64_t ffn_proxy_eval_x = 0;
    int64_t ffn_proxy_eval_out = 0;
    int64_t ffn_proxy_seed = 0;
    double ffn_proxy_cos_mean = 0.0;
    double ffn_proxy_cos_p05 = 0.0;
    double ffn_proxy_l2_mean = 0.0;
    double ffn_proxy_l2_p95 = 0.0;
    double ffn_proxy_log_norm_ratio_mean = 0.0;
    double ffn_proxy_log_norm_ratio_p95 = 0.0;

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

static void finalize_report_entry(report_entry & e) {
    if (e.metric_used.empty()) {
        e.metric_used = e.gating_metric_used;
    }
    if (e.target_tau_mean == 0.0) {
        e.target_tau_mean = e.gating_min_mean;
    }
    if (e.target_tau_p05 == 0.0) {
        e.target_tau_p05 = e.gating_min_p05;
    }
    if (e.reject_reason.empty()) {
        e.reject_reason = e.decision_reason;
    }
}

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

static double percentile_vec(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const double x = p * double(v.size() - 1);
    const size_t i = (size_t) x;
    const size_t j = std::min(i + 1, v.size() - 1);
    const double a = x - double(i);
    return v[i] * (1.0 - a) + v[j] * a;
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
    double stack_cost_cap = std::numeric_limits<double>::infinity();

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
        if (arg == "--stack-cost-cap" && i + 1 < argc) { stack_cost_cap = std::stod(argv[++i]); continue; }
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
    double stack_cost_running = 0.0;

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

    stack_safety_tracker stack_guard;

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

            float gating_min_mean = cfg.min_mean;
            float gating_min_p05  = cfg.min_p05;
            if (cfg.gating_enabled) {
                stack_guard.adjust(kind, gating_min_mean, gating_min_p05);
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
                    re.metric_used = re.gating_metric_used;
                    re.gating_min_mean = gating_min_mean;
                    re.gating_min_p05 = gating_min_p05;
                    re.target_tau_mean = gating_min_mean;
                    re.target_tau_p05 = gating_min_p05;
                    re.reject_reason = re.decision_reason;
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
                re.metric_used = re.gating_metric_used;
                re.gating_min_mean = gating_min_mean;
                re.gating_min_p05 = gating_min_p05;
                re.target_tau_mean = gating_min_mean;
                re.target_tau_p05 = gating_min_p05;
                re.reject_reason = re.decision_reason;
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
                re.metric_used = re.gating_metric_used;
                re.gating_min_mean = gating_min_mean;
                re.gating_min_p05 = gating_min_p05;
                re.target_tau_mean = gating_min_mean;
                re.target_tau_p05 = gating_min_p05;
                re.reject_reason = re.decision_reason;
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
                td.re.gating_min_mean = gating_min_mean;
                td.re.gating_min_p05  = gating_min_p05;

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
                    gating_pass = metric_available && metric_val >= gating_min_mean && metric_p05 >= gating_min_p05;
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
                best_trial.re.gating_min_mean = gating_min_mean;
                best_trial.re.gating_min_p05 = gating_min_p05;
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

            // FFN proxy v0 (logging-only): replace only this tensor (coo, no base).
            if (scheme == RESID_COO && !write_base && eval_x > 0) {
                const std::string wg_name = "blk." + std::to_string(il) + ".ffn_gate.weight";
                const std::string wu_name = "blk." + std::to_string(il) + ".ffn_up.weight";
                const std::string wd_name = "blk." + std::to_string(il) + ".ffn_down.weight";
                ggml_tensor * W_gate = ggml_get_tensor(ctx_data, wg_name.c_str());
                ggml_tensor * W_up   = ggml_get_tensor(ctx_data, wu_name.c_str());
                ggml_tensor * W_down = ggml_get_tensor(ctx_data, wd_name.c_str());

                ffn_proxy_metrics fpm;
                const int proxy_seed = seed + (int) il * 101 + (int) (kind == "ffn_gate" ? 1 : (kind == "ffn_up" ? 2 : 3));
                bool proxy_ok = eval_ffn_proxy_coo_replace_one(kind, W_gate, W_up, W_down, d_idx, d_val, write_base ? &base : nullptr, write_base, K_eff, eval_x, eval_cols, proxy_seed, fpm);
                if (proxy_ok) {
                    re.ffn_proxy_available = true;
                    re.ffn_proxy_scope = "replace_only_current_tensor";
                    re.ffn_proxy_base_used = write_base;
                    re.ffn_proxy_eval_x = fpm.eval_x;
                    re.ffn_proxy_eval_out = fpm.eval_out;
                    re.ffn_proxy_seed = proxy_seed;
                    re.ffn_proxy_cos_mean = fpm.cos_mean;
                    re.ffn_proxy_cos_p05 = fpm.cos_p05;
                    re.ffn_proxy_l2_mean = fpm.l2_mean;
                    re.ffn_proxy_l2_p95 = fpm.l2_p95;
                    re.ffn_proxy_log_norm_ratio_mean = fpm.log_norm_ratio_mean;
                    re.ffn_proxy_log_norm_ratio_p95 = fpm.log_norm_ratio_p95;
                } else {
                    re.ffn_proxy_available = false;
                    re.ffn_proxy_reason = (kind == "ffn_down") ? "proxy_kind_not_supported" : "proxy_unavailable";
                }
            } else {
                re.ffn_proxy_available = false;
                re.ffn_proxy_reason = (eval_x <= 0) ? "proxy_requires_eval_x" : "proxy_requires_coo";
            }

            
