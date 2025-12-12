#include "common.h"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <random>
#include <regex>
#include <string>
#include <vector>

static void usage(const char * argv0) {
    printf("usage: %s -i in.gguf -o out.gguf [options]\n\n", argv0);
    printf("options:\n");
    printf("  --dict-M N           dictionary size (default: 4096)\n");
    printf("  --dict-k K           active atoms per column (default: 16)\n");
    printf("  --dict-eta F         Oja learning rate (default: 0.01)\n");
    printf("  --dict-iters N       number of Oja passes (default: 3)\n");
    printf("  --dict-max-samples N max columns used to train dict (default: 2048)\n");
    printf("  --layers A-B         restrict to layer range (default: all)\n");
    printf("  --dict-type f16|f32  output dict type (default: f16)\n");
    printf("  --eval-cols N        evaluate reconstruction gap on N random outputs per weight (default: 0=off)\n");
    printf("  --report-json FILE   write JSON report with per-weight gap metrics (default: none)\n");
    printf("  --resume             resume from existing out.gguf (skip weights that already have dict+codes)\n");
    printf("  --checkpoint-every N write out.gguf every N processed layers (default: 0=off; large I/O)\n");
    printf("  --seed N             RNG seed (default: 1234)\n");
    exit(1);
}

static bool file_exists(const std::string & path) {
    std::ifstream ifs(path);
    return ifs.good();
}

static std::vector<int64_t> parse_layer_range(const std::string & s, int64_t n_layer) {
    if (s.empty()) {
        std::vector<int64_t> all(n_layer);
        std::iota(all.begin(), all.end(), 0);
        return all;
    }
    int64_t a = 0, b = n_layer - 1;
    if (sscanf(s.c_str(), "%" SCNd64 "-%" SCNd64, &a, &b) != 2) {
        throw std::runtime_error("invalid --layers, expected A-B");
    }
    a = std::max<int64_t>(0, a);
    b = std::min<int64_t>(n_layer - 1, b);
    std::vector<int64_t> out;
    for (int64_t i = a; i <= b; ++i) out.push_back(i);
    return out;
}

static void normalize_columns(std::vector<float> & D, int64_t n_in, int64_t M) {
    for (int64_t j = 0; j < M; ++j) {
        float norm = 0.0f;
        const float * col = D.data() + j * n_in;
        for (int64_t i = 0; i < n_in; ++i) norm += col[i] * col[i];
        norm = std::sqrt(std::max(norm, 1e-12f));
        float * colw = D.data() + j * n_in;
        for (int64_t i = 0; i < n_in; ++i) colw[i] /= norm;
    }
}

static std::vector<int> topk_abs_indices(const std::vector<float> & y, int k) {
    std::vector<int> idx(y.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::nth_element(idx.begin(), idx.begin() + k, idx.end(), [&](int a, int b) {
        return std::fabs(y[a]) > std::fabs(y[b]);
    });
    idx.resize(k);
    std::sort(idx.begin(), idx.end(), [&](int a, int b) {
        return std::fabs(y[a]) > std::fabs(y[b]);
    });
    return idx;
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

static void train_dict_oja_kwta(
        const ggml_tensor * W,
        int64_t M,
        int k,
        float eta,
        int iters,
        int64_t max_samples,
        std::mt19937 & rng,
        std::vector<float> & D_out,
        const std::string & tag) {
    const int64_t n_in  = W->ne[0];
    const int64_t n_out = W->ne[1];

    M = std::min<int64_t>(M, n_out);
    k = std::min<int>(k, (int) M);

    std::normal_distribution<float> nd(0.0f, 0.02f);
    D_out.assign(n_in * M, 0.0f);
    for (auto & v : D_out) v = nd(rng);
    normalize_columns(D_out, n_in, M);

    const int64_t n_samples = std::min<int64_t>(n_out, max_samples);
    const int64_t progress_every = std::max<int64_t>(1, n_samples / 20);
    std::vector<int64_t> sample_idx(n_out);
    std::iota(sample_idx.begin(), sample_idx.end(), 0);
    std::shuffle(sample_idx.begin(), sample_idx.end(), rng);
    sample_idx.resize(n_samples);

    std::vector<float> w;
    std::vector<float> y(M);

    for (int iter = 0; iter < iters; ++iter) {
        const int64_t iter_t0 = ggml_time_us();
        fprintf(stderr, "  [%s] training dict iter %d/%d (%" PRId64 " samples)\n", tag.c_str(), iter + 1, iters, n_samples);
        for (int64_t s = 0; s < n_samples; ++s) {
            const int64_t col = sample_idx[s];
            read_column_f32(W, col, w);

            // y = D^T w
            for (int64_t j = 0; j < M; ++j) {
                const float * dcol = D_out.data() + j * n_in;
                float dot = 0.0f;
                for (int64_t i = 0; i < n_in; ++i) dot += dcol[i] * w[i];
                y[j] = dot;
            }

            const auto topk = topk_abs_indices(y, k);

            for (int ti = 0; ti < k; ++ti) {
                const int j = topk[ti];
                float * dcol = D_out.data() + (int64_t) j * n_in;
                const float yj = y[j];
                for (int64_t i = 0; i < n_in; ++i) {
                    dcol[i] += eta * yj * (w[i] - yj * dcol[i]);
                }
                // renorm this column
                float norm = 0.0f;
                for (int64_t i = 0; i < n_in; ++i) norm += dcol[i] * dcol[i];
                norm = std::sqrt(std::max(norm, 1e-12f));
                for (int64_t i = 0; i < n_in; ++i) dcol[i] /= norm;
            }

            if ((s + 1) % progress_every == 0 || s + 1 == n_samples) {
                const double pct = 100.0 * double(s + 1) / double(n_samples);
                const double sec = double(ggml_time_us() - iter_t0) / 1e6;
                fprintf(stderr, "\r    [%s] iter %d/%d sample %" PRId64 "/%" PRId64 " (%.1f%%) elapsed %.1fs",
                        tag.c_str(), iter + 1, iters, s + 1, n_samples, pct, sec);
                fflush(stderr);
                if (s + 1 == n_samples) {
                    fprintf(stderr, "\n");
                }
            }
        }
    }
}

static void encode_codes_sign(
        const ggml_tensor * W,
        const std::vector<float> & D,
        int64_t M,
        int k,
        std::vector<int16_t> & codes_out,
        const std::string & tag) {
    const int64_t n_in  = W->ne[0];
    const int64_t n_out = W->ne[1];
    codes_out.assign((size_t) k * n_out, 0);

    const int64_t progress_every = std::max<int64_t>(1, n_out / 20);
    const int64_t t0 = ggml_time_us();

    std::vector<float> w;
    std::vector<float> y(M);

    for (int64_t col = 0; col < n_out; ++col) {
        read_column_f32(W, col, w);

        for (int64_t j = 0; j < M; ++j) {
            const float * dcol = D.data() + j * n_in;
            float dot = 0.0f;
            for (int64_t i = 0; i < n_in; ++i) dot += dcol[i] * w[i];
            y[j] = dot;
        }

        const auto topk = topk_abs_indices(y, k);
        for (int ti = 0; ti < k; ++ti) {
            const int j = topk[ti];
            const float yj = y[j];
            const int sign = yj >= 0 ? 1 : -1;
            codes_out[(size_t) ti + (size_t) col * k] = (int16_t) (sign * (j + 1));
        }

        if ((col + 1) % progress_every == 0 || col + 1 == n_out) {
            const double pct = 100.0 * double(col + 1) / double(n_out);
            const double sec = double(ggml_time_us() - t0) / 1e6;
            fprintf(stderr, "\r  [%s] encoding codes %" PRId64 "/%" PRId64 " (%.1f%%) elapsed %.1fs",
                    tag.c_str(), col + 1, n_out, pct, sec);
            fflush(stderr);
            if (col + 1 == n_out) {
                fprintf(stderr, "\n");
            }
        }
    }
}

struct eval_metrics {
    double rel_l2_mean = 0.0;
    double rel_l2_p95  = 0.0;
    double cos_mean    = 0.0;
    double cos_p05     = 0.0;
};

static eval_metrics eval_reconstruction_sign(
        const ggml_tensor * W,
        const std::vector<float> & D,
        int64_t M,
        int k,
        const std::vector<int16_t> & codes,
        int64_t eval_cols,
        std::mt19937 & rng) {
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

    for (int64_t ci = 0; ci < eval_cols; ++ci) {
        const int64_t col = idx[ci];

        read_column_f32(W, col, w);
        std::fill(w_hat.begin(), w_hat.end(), 0.0f);

        const int16_t * codes_col = codes.data() + (size_t) col * (size_t) k;
        for (int ti = 0; ti < k; ++ti) {
            const int16_t code = codes_col[ti];
            if (code == 0) {
                continue;
            }

            const float sign = code > 0 ? 1.0f : -1.0f;
            const int64_t atom = (int64_t) std::abs(code) - 1;
            if (atom < 0 || atom >= M) {
                continue;
            }

            const float * dcol = D.data() + atom * n_in;
            for (int64_t i = 0; i < n_in; ++i) {
                w_hat[i] += sign * dcol[i];
            }
        }

        double w_norm2  = 0.0;
        double wh_norm2 = 0.0;
        double err2     = 0.0;
        double dot      = 0.0;

        for (int64_t i = 0; i < n_in; ++i) {
            const double wi  = w[i];
            const double wHi = w_hat[i];
            const double di  = wi - wHi;
            w_norm2  += wi  * wi;
            wh_norm2 += wHi * wHi;
            err2     += di  * di;
            dot      += wi  * wHi;
        }

        const double denom_w = std::sqrt(std::max(w_norm2,  1e-20));
        const double denom_h = std::sqrt(std::max(wh_norm2, 1e-20));
        rel_l2.push_back(std::sqrt(std::max(err2, 0.0)) / denom_w);
        cos.push_back(dot / (denom_w * denom_h));
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

    return out;
}

int main(int argc, char ** argv) {
    std::string in_fname;
    std::string out_fname;

    int64_t dict_M = 4096;
    int dict_k = 16;
    float dict_eta = 0.01f;
    int dict_iters = 3;
    int64_t dict_max_samples = 2048;
    std::string layers_range;
    std::string dict_type_str = "f16";
    int64_t eval_cols = 0;
    std::string report_json;
    bool resume = false;
    int64_t checkpoint_every = 0;
    int seed = 1234;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if ((arg == "-i" || arg == "--input") && i + 1 < argc) { in_fname = argv[++i]; continue; }
        if ((arg == "-o" || arg == "--output") && i + 1 < argc) { out_fname = argv[++i]; continue; }
        if (arg == "--dict-M" && i + 1 < argc) { dict_M = std::stoll(argv[++i]); continue; }
        if (arg == "--dict-k" && i + 1 < argc) { dict_k = std::stoi(argv[++i]); continue; }
        if (arg == "--dict-eta" && i + 1 < argc) { dict_eta = std::stof(argv[++i]); continue; }
        if (arg == "--dict-iters" && i + 1 < argc) { dict_iters = std::stoi(argv[++i]); continue; }
        if (arg == "--dict-max-samples" && i + 1 < argc) { dict_max_samples = std::stoll(argv[++i]); continue; }
        if (arg == "--layers" && i + 1 < argc) { layers_range = argv[++i]; continue; }
        if (arg == "--dict-type" && i + 1 < argc) { dict_type_str = argv[++i]; continue; }
        if (arg == "--eval-cols" && i + 1 < argc) { eval_cols = std::stoll(argv[++i]); continue; }
        if (arg == "--report-json" && i + 1 < argc) { report_json = argv[++i]; continue; }
        if (arg == "--resume") { resume = true; continue; }
        if (arg == "--checkpoint-every" && i + 1 < argc) { checkpoint_every = std::stoll(argv[++i]); continue; }
        if (arg == "--seed" && i + 1 < argc) { seed = std::stoi(argv[++i]); continue; }
        usage(argv[0]);
    }

    if (in_fname.empty() || out_fname.empty()) {
        usage(argv[0]);
    }

    if (!report_json.empty() && eval_cols <= 0) {
        eval_cols = 64;
    }

    const bool out_exists = file_exists(out_fname);
    std::string src_fname = in_fname;
    if (resume) {
        if (out_exists) {
            fprintf(stderr, "statecells-build: resuming from existing %s\n", out_fname.c_str());
            src_fname = out_fname;
        } else {
            fprintf(stderr, "statecells-build: --resume requested but %s not found, starting from input\n", out_fname.c_str());
        }
    }

    ggml_context * ctx_data = nullptr;
    gguf_init_params params = { false, &ctx_data };
    gguf_context * src = gguf_init_from_file(src_fname.c_str(), params);
    if (!src || !ctx_data) {
        fprintf(stderr, "failed to load %s\n", src_fname.c_str());
        return 1;
    }

    const int64_t n_tensors = gguf_get_n_tensors(src);

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

    gguf_context * dst = gguf_init_empty();
    gguf_set_kv(dst, src);

    // Add original tensors.
    for (int64_t ti = 0; ti < n_tensors; ++ti) {
        const char * name = gguf_get_tensor_name(src, ti);
        ggml_tensor * t = ggml_get_tensor(ctx_data, name);
        if (!t) {
            fprintf(stderr, "warning: missing tensor %s in ctx_data\n", name);
            continue;
        }
        gguf_add_tensor(dst, t);
    }

    ggml_type dict_type = GGML_TYPE_F16;
    if (dict_type_str == "f32") dict_type = GGML_TYPE_F32;
    if (dict_type_str != "f16" && dict_type_str != "f32") {
        throw std::runtime_error("invalid --dict-type");
    }

    std::mt19937 rng(seed);

    // Keep per-weight ggml contexts alive until we write the output.
    std::vector<ggml_context *> sc_contexts;

    struct report_row {
        int64_t layer = 0;
        std::string kind;
        int64_t n_in = 0;
        int64_t n_out = 0;
        int64_t M_eff = 0;
        int k_eff = 0;
        int64_t eval_cols = 0;
        eval_metrics em;
    };
    std::vector<report_row> report_rows;

    // Count existing StateCells payload tensors so --resume doesn't accidentally disable the feature.
    int64_t n_sc_existing = 0;
    std::regex re_sc_payload(R"(blk\.(\d+)\.(ffn_gate|ffn_up|ffn_down)\.(dict|codes)$)");
    for (int64_t ti = 0; ti < n_tensors; ++ti) {
        const char * name = gguf_get_tensor_name(src, ti);
        if (std::regex_match(name, re_sc_payload)) {
            ++n_sc_existing;
        }
    }

    int64_t n_added = 0;

    const std::vector<std::string> kinds = { "ffn_gate", "ffn_up", "ffn_down" };
    int64_t layers_since_checkpoint = 0;

    for (const int64_t il : layers) {
        bool layer_added_any = false;

        for (const auto & kind : kinds) {
            const std::string weight_name = "blk." + std::to_string(il) + "." + kind + ".weight";
            if (gguf_find_tensor(src, weight_name.c_str()) == -1) {
                continue;
            }

            const std::string dict_name  = "blk." + std::to_string(il) + "." + kind + ".dict";
            const std::string codes_name = "blk." + std::to_string(il) + "." + kind + ".codes";

            const bool have_dict  = gguf_find_tensor(src, dict_name.c_str())  != -1;
            const bool have_codes = gguf_find_tensor(src, codes_name.c_str()) != -1;
            if (have_dict || have_codes) {
                if (have_dict && have_codes) {
                    continue;
                }
                fprintf(stderr, "statecells-build: inconsistent existing tensors for %s (dict=%d codes=%d)\n",
                        weight_name.c_str(), (int) have_dict, (int) have_codes);
                return 1;
            }

            ggml_tensor * W = ggml_get_tensor(ctx_data, weight_name.c_str());
            if (!W || ggml_n_dims(W) != 2) {
                continue;
            }

            const int64_t n_in  = W->ne[0];
            const int64_t n_out = W->ne[1];

            printf("statecells-build: layer %" PRId64 " %s [% " PRId64 " x %" PRId64 "] type=%s\n",
                   il, kind.c_str(), n_in, n_out, ggml_type_name(W->type));

            std::vector<float> D;
            const std::string tag = "blk." + std::to_string(il) + "." + kind;
            train_dict_oja_kwta(W, dict_M, dict_k, dict_eta, dict_iters, dict_max_samples, rng, D, tag);

            int64_t M_eff = std::min<int64_t>(dict_M, n_out);
            int k_eff = std::min<int>(dict_k, (int) M_eff);

            std::vector<int16_t> codes;
            encode_codes_sign(W, D, M_eff, k_eff, codes, tag);

            eval_metrics em;
            if (eval_cols > 0) {
                const int64_t t0 = ggml_time_us();
                em = eval_reconstruction_sign(W, D, M_eff, k_eff, codes, eval_cols, rng);
                const double sec = double(ggml_time_us() - t0) / 1e6;
                fprintf(stderr, "  [%s] eval cols=%" PRId64 " rel_l2 mean=%.4f p95=%.4f cos mean=%.4f p05=%.4f (%.1fs)\n",
                        tag.c_str(), eval_cols, em.rel_l2_mean, em.rel_l2_p95, em.cos_mean, em.cos_p05, sec);
            }

            // Allocate a dedicated ggml context for dict+codes to avoid a giant arena.
            const size_t size_dict  = (size_t) n_in * (size_t) M_eff * ggml_type_size(dict_type);
            const size_t size_codes = (size_t) k_eff * (size_t) n_out * sizeof(int16_t);
            const size_t mem_size_sc = ggml_tensor_overhead() * 8 + size_dict + size_codes;
            ggml_init_params sc_params = { mem_size_sc, nullptr, false };
            ggml_context * ctx_sc = ggml_init(sc_params);
            sc_contexts.push_back(ctx_sc);

            // Create dict tensor.
            ggml_tensor * t_dict = ggml_new_tensor_2d(ctx_sc, dict_type, n_in, M_eff);
            ggml_set_name(t_dict, dict_name.c_str());

            if (dict_type == GGML_TYPE_F16) {
                auto * dst_f16 = (ggml_fp16_t *) t_dict->data;
                for (int64_t j = 0; j < M_eff; ++j) {
                    ggml_fp32_to_fp16_row(D.data() + j * n_in, dst_f16 + j * n_in, n_in);
                }
            } else {
                std::memcpy(t_dict->data, D.data(), (size_t) n_in * M_eff * sizeof(float));
            }

            // Create codes tensor.
            ggml_tensor * t_codes = ggml_new_tensor_2d(ctx_sc, GGML_TYPE_I16, k_eff, n_out);
            ggml_set_name(t_codes, codes_name.c_str());
            std::memcpy(t_codes->data, codes.data(), codes.size() * sizeof(int16_t));

            gguf_add_tensor(dst, t_dict);
            gguf_add_tensor(dst, t_codes);
            n_added += 2;
            layer_added_any = true;

            if (!report_json.empty() || eval_cols > 0) {
                report_rows.push_back(report_row{ il, kind, n_in, n_out, M_eff, k_eff, eval_cols, em });
            }
        }

        if (checkpoint_every > 0 && layer_added_any) {
            layers_since_checkpoint++;
            if (layers_since_checkpoint >= checkpoint_every) {
                gguf_set_val_bool(dst, "statecells.enabled", (n_sc_existing + n_added) > 0);
                gguf_set_val_u32(dst, "statecells.dict.M", (uint32_t) dict_M);
                gguf_set_val_u32(dst, "statecells.dict.k", (uint32_t) dict_k);
                gguf_set_val_f32(dst, "statecells.dict.eta", dict_eta);
                gguf_set_val_u32(dst, "statecells.dict.iters", (uint32_t) dict_iters);

                printf("checkpoint: writing %s (%" PRId64 " new tensors so far)\n", out_fname.c_str(), n_added);
                gguf_write_to_file(dst, out_fname.c_str(), false);
                layers_since_checkpoint = 0;
            }
        }
    }

    if (n_sc_existing + n_added == 0) {
        fprintf(stderr, "no FFN weights found for StateCells\n");
    } else if (n_added == 0) {
        fprintf(stderr, "no new StateCells tensors added (already present or out of range)\n");
    }

    gguf_set_val_bool(dst, "statecells.enabled", (n_sc_existing + n_added) > 0);
    gguf_set_val_u32(dst, "statecells.dict.M", (uint32_t) dict_M);
    gguf_set_val_u32(dst, "statecells.dict.k", (uint32_t) dict_k);
    gguf_set_val_f32(dst, "statecells.dict.eta", dict_eta);
    gguf_set_val_u32(dst, "statecells.dict.iters", (uint32_t) dict_iters);

    if (src_fname == out_fname && n_added == 0) {
        printf("no changes, keeping existing %s\n", out_fname.c_str());
    } else {
        printf("writing %s with %" PRId64 " new tensors\n", out_fname.c_str(), n_added);
        gguf_write_to_file(dst, out_fname.c_str(), false);
    }

    if (!report_json.empty()) {
        std::ofstream ofs(report_json);
        if (!ofs) {
            fprintf(stderr, "failed to write report to %s\n", report_json.c_str());
        } else {
            ofs << "{\n";
            ofs << "  \"input\": "   << "\"" << in_fname  << "\",\n";
            ofs << "  \"source\": "  << "\"" << src_fname << "\",\n";
            ofs << "  \"output\": "  << "\"" << out_fname << "\",\n";
            ofs << "  \"resume\": "  << (resume ? "true" : "false") << ",\n";
            ofs << "  \"dict\": {\n";
            ofs << "    \"M\": " << dict_M << ",\n";
            ofs << "    \"k\": " << dict_k << ",\n";
            ofs << "    \"eta\": " << dict_eta << ",\n";
            ofs << "    \"iters\": " << dict_iters << ",\n";
            ofs << "    \"max_samples\": " << dict_max_samples << "\n";
            ofs << "  },\n";
            ofs << "  \"eval_cols\": " << eval_cols << ",\n";
            ofs << "  \"weights\": [\n";
            for (size_t i = 0; i < report_rows.size(); ++i) {
                const auto & r = report_rows[i];
                ofs << "    {\n";
                ofs << "      \"layer\": " << r.layer << ",\n";
                ofs << "      \"kind\": " << "\"" << r.kind << "\"" << ",\n";
                ofs << "      \"n_in\": " << r.n_in << ",\n";
                ofs << "      \"n_out\": " << r.n_out << ",\n";
                ofs << "      \"M\": " << r.M_eff << ",\n";
                ofs << "      \"k\": " << r.k_eff << ",\n";
                ofs << "      \"rel_l2_mean\": " << r.em.rel_l2_mean << ",\n";
                ofs << "      \"rel_l2_p95\": "  << r.em.rel_l2_p95  << ",\n";
                ofs << "      \"cos_mean\": "    << r.em.cos_mean    << ",\n";
                ofs << "      \"cos_p05\": "     << r.em.cos_p05     << "\n";
                ofs << "    }" << (i + 1 == report_rows.size() ? "\n" : ",\n");
            }
            ofs << "  ]\n";
            ofs << "}\n";
        }
    }

    for (auto * c : sc_contexts) {
        ggml_free(c);
    }
    ggml_free(ctx_data);
    gguf_free(dst);
    gguf_free(src);

    return 0;
}
