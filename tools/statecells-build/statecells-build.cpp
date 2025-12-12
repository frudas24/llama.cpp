#include "common.h"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
    printf("  --seed N             RNG seed (default: 1234)\n");
    exit(1);
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
        if (arg == "--seed" && i + 1 < argc) { seed = std::stoi(argv[++i]); continue; }
        usage(argv[0]);
    }

    if (in_fname.empty() || out_fname.empty()) {
        usage(argv[0]);
    }

    ggml_context * ctx_data = nullptr;
    gguf_init_params params = { false, &ctx_data };
    gguf_context * src = gguf_init_from_file(in_fname.c_str(), params);
    if (!src || !ctx_data) {
        fprintf(stderr, "failed to load %s\n", in_fname.c_str());
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

    std::regex re_ffn(R"(blk\.(\d+)\.(ffn_gate|ffn_up|ffn_down)\.weight$)");

    int64_t n_added = 0;

    for (int64_t ti = 0; ti < n_tensors; ++ti) {
        const char * name = gguf_get_tensor_name(src, ti);
        std::cmatch m;
        if (!std::regex_match(name, m, re_ffn)) {
            continue;
        }

        const int64_t il = std::stoll(m[1]);
        const std::string kind = m[2];
        if (std::find(layers.begin(), layers.end(), il) == layers.end()) {
            continue;
        }

        ggml_tensor * W = ggml_get_tensor(ctx_data, name);
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

        // Allocate a dedicated ggml context for dict+codes to avoid a giant arena.
        const size_t size_dict  = (size_t) n_in * (size_t) M_eff * ggml_type_size(dict_type);
        const size_t size_codes = (size_t) k_eff * (size_t) n_out * sizeof(int16_t);
        const size_t mem_size_sc = ggml_tensor_overhead() * 8 + size_dict + size_codes;
        ggml_init_params sc_params = { mem_size_sc, nullptr, false };
        ggml_context * ctx_sc = ggml_init(sc_params);
        sc_contexts.push_back(ctx_sc);

        // Create dict tensor.
        ggml_tensor * t_dict = ggml_new_tensor_2d(ctx_sc, dict_type, n_in, M_eff);
        const std::string dict_name = "blk." + std::to_string(il) + "." + kind + ".dict";
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
        const std::string codes_name = "blk." + std::to_string(il) + "." + kind + ".codes";
        ggml_set_name(t_codes, codes_name.c_str());
        std::memcpy(t_codes->data, codes.data(), codes.size() * sizeof(int16_t));

        gguf_add_tensor(dst, t_dict);
        gguf_add_tensor(dst, t_codes);
        n_added += 2;
    }

    if (n_added == 0) {
        fprintf(stderr, "no FFN weights found for StateCells\n");
    }

    gguf_set_val_bool(dst, "statecells.enabled", n_added > 0);
    gguf_set_val_u32(dst, "statecells.dict.M", (uint32_t) dict_M);
    gguf_set_val_u32(dst, "statecells.dict.k", (uint32_t) dict_k);
    gguf_set_val_f32(dst, "statecells.dict.eta", dict_eta);
    gguf_set_val_u32(dst, "statecells.dict.iters", (uint32_t) dict_iters);

    printf("writing %s with %" PRId64 " new tensors\n", out_fname.c_str(), n_added);
    gguf_write_to_file(dst, out_fname.c_str(), false);

    for (auto * c : sc_contexts) {
        ggml_free(c);
    }
    ggml_free(ctx_data);
    gguf_free(dst);
    gguf_free(src);

    return 0;
}
