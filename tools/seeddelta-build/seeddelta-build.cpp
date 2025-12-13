#include "common.h"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <atomic>
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
#include <thread>
#include <vector>

static void usage(const char * argv0) {
    printf("usage: %s -i in.gguf -o out.gguf [options]\n\n", argv0);
    printf("options:\n");
    printf("  --layers A-B         restrict to layer range (default: all)\n");
    printf("  --K N                top-K residual entries per output (default: 32)\n");
    printf("  --idx-type i16|i32   index tensor type (default: i16)\n");
    printf("  --val-type f16|f32   value tensor type (default: f16)\n");
    printf("  --row-scale          write per-output d_row_scale tensor (default: off)\n");
    printf("  --no-row-scale       disable d_row_scale tensor output\n");
    printf("  -t, --threads N      worker threads (default: nproc)\n");
    printf("  --eval-cols N        evaluate reconstruction gap on N random outputs per weight (default: 0=off)\n");
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

static void topk_abs(const std::vector<float> & v, int64_t K, std::vector<int32_t> & idx_out) {
    const int64_t n = (int64_t) v.size();
    K = std::min<int64_t>(K, n);
    idx_out.resize((size_t) K);

    std::vector<int32_t> idx((size_t) n);
    std::iota(idx.begin(), idx.end(), 0);
    if (K < n) {
        std::nth_element(idx.begin(), idx.begin() + K, idx.end(), [&](int32_t a, int32_t b) {
            return std::fabs(v[(size_t) a]) > std::fabs(v[(size_t) b]);
        });
        idx.resize((size_t) K);
    }
    std::sort(idx.begin(), idx.end(), [&](int32_t a, int32_t b) {
        return std::fabs(v[(size_t) a]) > std::fabs(v[(size_t) b]);
    });
    idx_out = idx;
}

struct eval_metrics {
    double rel_l2_mean = 0.0;
    double rel_l2_p95  = 0.0;
    double cos_mean    = 0.0;
    double cos_p05     = 0.0;
};

static eval_metrics eval_sparse_residual(
        const ggml_tensor * W,
        const std::vector<int32_t> & d_idx,
        const std::vector<float> & d_val,
        const std::vector<float> * d_row_scale,
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
    rel_l2.reserve((size_t) eval_cols);
    cos.reserve((size_t) eval_cols);

    for (int64_t ci = 0; ci < eval_cols; ++ci) {
        const int64_t col = idx[(size_t) ci];

        read_column_f32(W, col, w);

        const double scale = d_row_scale ? (double) (*d_row_scale)[(size_t) col] : 1.0;

        double w_norm2  = 0.0;
        for (int64_t i = 0; i < n_in; ++i) {
            const double wi = w[(size_t) i];
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
            const double wh = scale * (double) val_col[r];
            wh_norm2 += wh * wh;
            dot      += (double) w[(size_t) ii] * wh;
        }

        const double err2 = std::max(w_norm2 + wh_norm2 - 2.0 * dot, 0.0);

        const double denom_w = std::sqrt(std::max(w_norm2,  1e-20));
        const double denom_h = std::sqrt(std::max(wh_norm2, 1e-20));

        rel_l2.push_back(std::sqrt(err2) / denom_w);
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

    std::string layers_range;

    int64_t K = 32;
    std::string idx_type_str = "i16";
    std::string val_type_str = "f16";
    bool write_row_scale = false;
    int n_threads = (int) std::max(1u, std::thread::hardware_concurrency());
    int64_t eval_cols = 0;
    int seed = 1234;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if ((arg == "-i" || arg == "--input") && i + 1 < argc) { in_fname = argv[++i]; continue; }
        if ((arg == "-o" || arg == "--output") && i + 1 < argc) { out_fname = argv[++i]; continue; }
        if (arg == "--layers" && i + 1 < argc) { layers_range = argv[++i]; continue; }
        if (arg == "--K" && i + 1 < argc) { K = std::stoll(argv[++i]); continue; }
        if (arg == "--idx-type" && i + 1 < argc) { idx_type_str = argv[++i]; continue; }
        if (arg == "--val-type" && i + 1 < argc) { val_type_str = argv[++i]; continue; }
        if (arg == "--row-scale") { write_row_scale = true; continue; }
        if (arg == "--no-row-scale") { write_row_scale = false; continue; }
        if ((arg == "-t" || arg == "--threads") && i + 1 < argc) { n_threads = std::stoi(argv[++i]); continue; }
        if (arg == "--eval-cols" && i + 1 < argc) { eval_cols = std::stoll(argv[++i]); continue; }
        if (arg == "--seed" && i + 1 < argc) { seed = std::stoi(argv[++i]); continue; }
        if (arg == "-h" || arg == "--help") {
            usage(argv[0]);
        }
        fprintf(stderr, "unknown argument: %s\n", arg.c_str());
        usage(argv[0]);
    }

    if (in_fname.empty() || out_fname.empty()) {
        usage(argv[0]);
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

    std::mt19937 rng(seed);

    // Keep per-weight ggml contexts alive until we write the output.
    std::vector<ggml_context *> sd_contexts;

    const std::vector<std::string> kinds = { "ffn_gate", "ffn_up", "ffn_down" };

    int64_t n_added = 0;

    for (const int64_t il : layers) {
        for (const auto & kind : kinds) {
            const std::string weight_name = "blk." + std::to_string(il) + "." + kind + ".weight";
            if (gguf_find_tensor(src, weight_name.c_str()) == -1) {
                continue;
            }

            const std::string d_idx_name      = "blk." + std::to_string(il) + "." + kind + ".d_idx";
            const std::string d_val_name      = "blk." + std::to_string(il) + "." + kind + ".d_val";
            const std::string d_row_scale_name = "blk." + std::to_string(il) + "." + kind + ".d_row_scale";

            if (gguf_find_tensor(src, d_idx_name.c_str()) != -1 || gguf_find_tensor(src, d_val_name.c_str()) != -1) {
                fprintf(stderr, "seeddelta-build: %s already has d_idx/d_val, skipping\n", weight_name.c_str());
                continue;
            }

            ggml_tensor * W = ggml_get_tensor(ctx_data, weight_name.c_str());
            if (!W || ggml_n_dims(W) != 2) {
                continue;
            }

            const int64_t n_in  = W->ne[0];
            const int64_t n_out = W->ne[1];
            const int64_t K_eff = std::max<int64_t>(1, std::min<int64_t>(K, n_in));

            printf("seeddelta-build: layer %" PRId64 " %s [% " PRId64 " x %" PRId64 "] type=%s K=%" PRId64 "\n",
                   il, kind.c_str(), n_in, n_out, ggml_type_name(W->type), K_eff);

            std::vector<int32_t> d_idx((size_t) K_eff * (size_t) n_out, -1);
            std::vector<float>   d_val((size_t) K_eff * (size_t) n_out, 0.0f);
            std::vector<float>   d_row_scale;
            if (write_row_scale) {
                d_row_scale.resize((size_t) n_out, 1.0f);
            }

            std::atomic<int64_t> next_col{0};
            std::vector<std::thread> workers;
            workers.reserve((size_t) n_threads);

            const int64_t chunk = 16;

            for (int ti = 0; ti < n_threads; ++ti) {
                workers.emplace_back([&, ti]() {
                    std::vector<float> w;
                    std::vector<int32_t> topk;
                    while (true) {
                        const int64_t col0 = next_col.fetch_add(chunk);
                        if (col0 >= n_out) {
                            break;
                        }

                        const int64_t col1 = std::min<int64_t>(n_out, col0 + chunk);
                        for (int64_t col = col0; col < col1; ++col) {
                            read_column_f32(W, col, w);
                            topk_abs(w, K_eff, topk);

                            float ss = 1.0f;
                            if (write_row_scale) {
                                double dot = 0.0;
                                double nn  = 0.0;
                                for (int64_t r = 0; r < K_eff; ++r) {
                                    const int32_t ii = topk[(size_t) r];
                                    if (ii < 0 || ii >= (int32_t) n_in) {
                                        continue;
                                    }
                                    const double wi  = (double) w[(size_t) ii];
                                    const double whi = (double) w[(size_t) ii]; // w_hat uses the stored vals
                                    dot += wi * whi;
                                    nn  += whi * whi;
                                }
                                if (nn > 1e-30) {
                                    ss = (float) (dot / nn);
                                }
                                d_row_scale[(size_t) col] = ss;
                            }

                            for (int64_t r = 0; r < K_eff; ++r) {
                                const int32_t ii = topk[(size_t) r];
                                d_idx[(size_t) col * (size_t) K_eff + (size_t) r] = ii;
                                d_val[(size_t) col * (size_t) K_eff + (size_t) r] = (ii >= 0 && ii < (int32_t) n_in) ? w[(size_t) ii] : 0.0f;
                            }
                        }
                    }
                });
            }

            for (auto & th : workers) {
                th.join();
            }

            if (eval_cols > 0) {
                const int64_t t0 = ggml_time_us();
                eval_metrics em = eval_sparse_residual(W, d_idx, d_val, write_row_scale ? &d_row_scale : nullptr, K_eff, eval_cols, rng);
                const double sec = double(ggml_time_us() - t0) / 1e6;
                fprintf(stderr, "  [blk.%" PRId64 ".%s] eval cols=%" PRId64 " rel_l2 mean=%.4f p95=%.4f cos mean=%.4f p05=%.4f (%.1fs)\n",
                        il, kind.c_str(), eval_cols, em.rel_l2_mean, em.rel_l2_p95, em.cos_mean, em.cos_p05, sec);
            }

            // Allocate a dedicated ggml context for new tensors to avoid a giant arena.
            const size_t size_idx = (size_t) K_eff * (size_t) n_out * ggml_type_size(idx_type);
            const size_t size_val = (size_t) K_eff * (size_t) n_out * ggml_type_size(val_type);
            const size_t size_row_scale = write_row_scale ? (size_t) n_out * sizeof(ggml_fp16_t) : 0;
            const size_t mem_size_sd = ggml_tensor_overhead() * 6 + size_idx + size_val + size_row_scale;

            ggml_init_params sd_params = { mem_size_sd, nullptr, false };
            ggml_context * ctx_sd = ggml_init(sd_params);
            sd_contexts.push_back(ctx_sd);

            ggml_tensor * t_idx = ggml_new_tensor_2d(ctx_sd, idx_type, K_eff, n_out);
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

            ggml_tensor * t_val = ggml_new_tensor_2d(ctx_sd, val_type, K_eff, n_out);
            ggml_set_name(t_val, d_val_name.c_str());
            if (val_type == GGML_TYPE_F16) {
                auto * dst_f16 = (ggml_fp16_t *) t_val->data;
                for (int64_t col = 0; col < n_out; ++col) {
                    ggml_fp32_to_fp16_row(d_val.data() + (size_t) col * (size_t) K_eff, dst_f16 + (size_t) col * (size_t) K_eff, K_eff);
                }
            } else {
                std::memcpy(t_val->data, d_val.data(), d_val.size() * sizeof(float));
            }

            gguf_add_tensor(dst, t_idx);
            gguf_add_tensor(dst, t_val);
            n_added += 2;

            if (write_row_scale) {
                ggml_tensor * t_rs = ggml_new_tensor_1d(ctx_sd, GGML_TYPE_F16, n_out);
                ggml_set_name(t_rs, d_row_scale_name.c_str());
                auto * rs_f16 = (ggml_fp16_t *) t_rs->data;
                ggml_fp32_to_fp16_row(d_row_scale.data(), rs_f16, n_out);
                gguf_add_tensor(dst, t_rs);
                n_added += 1;
            }
        }
    }

    if (n_added == 0) {
        fprintf(stderr, "no new SeedΔ tensors added\n");
    }

    gguf_set_val_bool(dst, "seeddelta.enabled", n_added > 0);
    gguf_set_val_u32(dst, "seeddelta.resid.K", (uint32_t) K);

    printf("writing %s with %" PRId64 " new tensors\n", out_fname.c_str(), n_added);
    gguf_write_to_file(dst, out_fname.c_str(), false);

    return 0;
}

