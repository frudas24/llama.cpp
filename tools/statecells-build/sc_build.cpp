#include "common.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include "include/sc_cli.h"
#include "include/sc_eval.h"
#include "include/sc_imatrix.h"
#include "include/sc_report.h"
#include "include/sc_build.h"

#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <numeric>
#include <random>
#include <regex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>



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
    k = std::min<int>(k, (int) idx.size());
    if (k <= 0) {
        idx.clear();
        return idx;
    }
    if (k < (int) idx.size()) {
        std::nth_element(idx.begin(), idx.begin() + k, idx.end(), [&](int a, int b) {
            return std::fabs(y[a]) > std::fabs(y[b]);
        });
        idx.resize(k);
    }
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

static void read_mat_f16_f32_to_f32(const ggml_tensor * A, std::vector<float> & out) {
    GGML_ASSERT(A != nullptr);
    GGML_ASSERT(ggml_n_dims(A) == 2);
    GGML_ASSERT(A->type == GGML_TYPE_F16 || A->type == GGML_TYPE_F32);

    const int64_t n0 = A->ne[0];
    const int64_t n1 = A->ne[1];

    out.resize((size_t) n0 * (size_t) n1);

    if (ggml_is_contiguous(A)) {
        if (A->type == GGML_TYPE_F32) {
            std::memcpy(out.data(), A->data, (size_t) n0 * (size_t) n1 * sizeof(float));
            return;
        }

        const ggml_fp16_t * src = (const ggml_fp16_t *) A->data;
        for (int64_t j = 0; j < n1; ++j) {
            ggml_fp16_to_fp32_row(src + j * n0, out.data() + j * n0, n0);
        }
        return;
    }

    const uint8_t * base = (const uint8_t *) A->data;
    for (int64_t j = 0; j < n1; ++j) {
        for (int64_t i = 0; i < n0; ++i) {
            const uint8_t * p = base + i * A->nb[0] + j * A->nb[1];
            out[(size_t) j * (size_t) n0 + (size_t) i] = A->type == GGML_TYPE_F16 ?
                    ggml_fp16_to_fp32(*(const ggml_fp16_t *) p) :
                    *(const float *) p;
        }
    }
}

static void read_mat_f16_f32_to_fp16(const ggml_tensor * A, std::vector<ggml_fp16_t> & out) {
    GGML_ASSERT(A != nullptr);
    GGML_ASSERT(ggml_n_dims(A) == 2);
    GGML_ASSERT(A->type == GGML_TYPE_F16 || A->type == GGML_TYPE_F32);

    const int64_t n0 = A->ne[0];
    const int64_t n1 = A->ne[1];

    out.resize((size_t) n0 * (size_t) n1);

    if (ggml_is_contiguous(A)) {
        if (A->type == GGML_TYPE_F16) {
            std::memcpy(out.data(), A->data, (size_t) n0 * (size_t) n1 * sizeof(ggml_fp16_t));
            return;
        }

        const float * src = (const float *) A->data;
        for (int64_t j = 0; j < n1; ++j) {
            ggml_fp32_to_fp16_row(src + j * n0, out.data() + j * n0, n0);
        }
        return;
    }

    const uint8_t * base = (const uint8_t *) A->data;
    for (int64_t j = 0; j < n1; ++j) {
        for (int64_t i = 0; i < n0; ++i) {
            const uint8_t * p = base + i * A->nb[0] + j * A->nb[1];
            const float v = A->type == GGML_TYPE_F16 ?
                    ggml_fp16_to_fp32(*(const ggml_fp16_t *) p) :
                    *(const float *) p;
            out[(size_t) j * (size_t) n0 + (size_t) i] = ggml_fp32_to_fp16(v);
        }
    }
}

static void read_vec_f16_f32_to_fp16(const ggml_tensor * A, std::vector<ggml_fp16_t> & out) {
    GGML_ASSERT(A != nullptr);
    GGML_ASSERT(ggml_n_dims(A) == 1);
    GGML_ASSERT(A->type == GGML_TYPE_F16 || A->type == GGML_TYPE_F32);

    const int64_t n = A->ne[0];
    out.resize((size_t) n);

    if (ggml_is_contiguous(A)) {
        if (A->type == GGML_TYPE_F16) {
            std::memcpy(out.data(), A->data, (size_t) n * sizeof(ggml_fp16_t));
            return;
        }

        const float * src = (const float *) A->data;
        ggml_fp32_to_fp16_row(src, out.data(), n);
        return;
    }

    const uint8_t * base = (const uint8_t *) A->data;
    for (int64_t i = 0; i < n; ++i) {
        const uint8_t * p = base + i * A->nb[0];
        const float v = A->type == GGML_TYPE_F16 ?
                ggml_fp16_to_fp32(*(const ggml_fp16_t *) p) :
                *(const float *) p;
        out[(size_t) i] = ggml_fp32_to_fp16(v);
    }
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
        const std::string & tag,
        const std::vector<float> * w_scale) {
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
            if (w_scale && (int64_t) w_scale->size() == n_in) {
                for (int64_t i = 0; i < n_in; ++i) {
                    w[i] *= (*w_scale)[(size_t) i];
                }
            }

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
        int n_threads,
        std::vector<int16_t> & codes_out,
        std::vector<ggml_fp16_t> * vals_out,
        const std::string & tag,
        const std::vector<float> * w_scale) {
    const int64_t n_in  = W->ne[0];
    const int64_t n_out = W->ne[1];

    GGML_ASSERT((int64_t) D.size() == n_in * M);
    GGML_ASSERT(M > 0);

    k = std::min<int>(k, (int) M);
    if (k <= 0 || n_out <= 0) {
        codes_out.clear();
        return;
    }

    // codes store ±(atom+1) in int16.
    GGML_ASSERT(M <= 32767);

    codes_out.assign((size_t) k * (size_t) n_out, 0);
    if (vals_out) {
        vals_out->assign((size_t) k * (size_t) n_out, ggml_fp32_to_fp16(0.0f));
    }

    // Encode with a blocked GEMM: Y = D^T * Wblk, Y: [M, B]
    // This reuses D across B columns (huge speedup vs per-column dot loops).
    n_threads = std::max(1, n_threads);
    n_threads = std::min<int>(n_threads, (int) std::max<int64_t>(1, n_out));

    const int64_t t0 = ggml_time_us();
    const int64_t progress_every = std::max<int64_t>(1, n_out / 20);

    const int64_t B = std::min<int64_t>(64, n_out);

    const size_t size_dict = (size_t) n_in * (size_t) M * sizeof(float);
    const size_t size_wblk = (size_t) n_in * (size_t) B * sizeof(float);
    const size_t size_y    = (size_t) M    * (size_t) B * sizeof(float);

    const size_t mem_size =
            ggml_tensor_overhead() * 16 +
            ggml_graph_overhead_custom(256, /*grads*/ false) +
            size_dict + size_wblk + size_y +
            1024*1024;

    ggml_init_params params = { mem_size, nullptr, false };
    ggml_context * ctx = ggml_init(params);
    GGML_ASSERT(ctx != nullptr);

    ggml_tensor * t_dict = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_in, M);
    ggml_tensor * t_wblk = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_in, B);

    std::memcpy(t_dict->data, D.data(), size_dict);

    ggml_tensor * t_y = ggml_mul_mat(ctx, t_dict, t_wblk);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, t_y);

    ggml_threadpool_params tpp = ggml_threadpool_params_default(n_threads);
    ggml_threadpool * threadpool = ggml_threadpool_new(&tpp);
    GGML_ASSERT(threadpool != nullptr);

    ggml_cplan plan = ggml_graph_plan(gf, n_threads, threadpool);
    std::vector<uint8_t> work_buffer;
    if (plan.work_size > 0) {
        work_buffer.resize(plan.work_size);
        plan.work_data = work_buffer.data();
    }

    auto * wblk_data = (float *) t_wblk->data;
    const auto * y_data = (const float *) t_y->data;

    std::vector<float> w;
    std::vector<int> idx((size_t) M);

    int64_t last_print = 0;
    for (int64_t col0 = 0; col0 < n_out; col0 += B) {
        const int64_t nb = std::min<int64_t>(B, n_out - col0);

        for (int64_t b = 0; b < nb; ++b) {
            read_column_f32(W, col0 + b, w);
            if (w_scale && (int64_t) w_scale->size() == n_in) {
                for (int64_t i = 0; i < n_in; ++i) {
                    w[i] *= (*w_scale)[(size_t) i];
                }
            }
            std::memcpy(wblk_data + b * n_in, w.data(), (size_t) n_in * sizeof(float));
        }
        for (int64_t b = nb; b < B; ++b) {
            std::memset(wblk_data + b * n_in, 0, (size_t) n_in * sizeof(float));
        }

        const enum ggml_status st = ggml_graph_compute(gf, &plan);
        GGML_ASSERT(st == GGML_STATUS_SUCCESS);

        for (int64_t b = 0; b < nb; ++b) {
            const int64_t col = col0 + b;
            const float * y_col = y_data + b * M;

            std::iota(idx.begin(), idx.end(), 0);
            const int k_eff = std::min<int>(k, (int) idx.size());
            if (k_eff > 0 && k_eff < (int) idx.size()) {
                std::nth_element(idx.begin(), idx.begin() + k_eff, idx.end(), [&](int a, int b) {
                    return std::fabs(y_col[a]) > std::fabs(y_col[b]);
                });
            }
            if (k_eff > 0) {
                std::sort(idx.begin(), idx.begin() + k_eff, [&](int a, int b) {
                    return std::fabs(y_col[a]) > std::fabs(y_col[b]);
                });
            }

            for (int ti = 0; ti < k_eff; ++ti) {
                const int j = idx[ti];
                const float yj = y_col[j];
                const int sign = yj >= 0 ? 1 : -1;
                codes_out[(size_t) ti + (size_t) col * (size_t) k] = (int16_t) (sign * (j + 1));
                if (vals_out) {
                    float v = std::fabs(yj);
                    if (!std::isfinite(v) || v < 0.0f) {
                        v = 0.0f;
                    }
                    if (v > 65504.0f) { // max finite fp16
                        v = 65504.0f;
                    }
                    (*vals_out)[(size_t) ti + (size_t) col * (size_t) k] = ggml_fp32_to_fp16(v);
                }
            }
        }

        const int64_t done = std::min<int64_t>(col0 + nb, n_out);
        if (done >= last_print + progress_every || done == n_out) {
            last_print = done;
            const double pct = 100.0 * double(done) / double(n_out);
            const double sec = double(ggml_time_us() - t0) / 1e6;
            fprintf(stderr, "\r  [%s] encoding codes %" PRId64 "/%" PRId64 " (%.1f%%) elapsed %.1fs (%d thr, B=%" PRId64 ")",
                    tag.c_str(), done, n_out, pct, sec, n_threads, B);
            fflush(stderr);
            if (done == n_out) {
                fprintf(stderr, "\n");
            }
        }
    }

    ggml_threadpool_free(threadpool);
    ggml_free(ctx);
}

static void compute_vals_sign(
        const ggml_tensor * W,
        const std::vector<float> & D,
        int64_t M,
        int k,
        const std::vector<int16_t> & codes,
        int n_threads,
        std::vector<ggml_fp16_t> & vals_out,
        const std::string & tag,
        const std::vector<float> * w_scale) {
    const int64_t n_in  = W->ne[0];
    const int64_t n_out = W->ne[1];

    GGML_ASSERT((int64_t) D.size() == n_in * M);
    GGML_ASSERT((int64_t) codes.size() == (int64_t) k * n_out);
    GGML_ASSERT(M > 0);

    k = std::min<int>(k, (int) M);
    if (k <= 0 || n_out <= 0) {
        vals_out.clear();
        return;
    }

    // codes store ±(atom+1) in int16.
    GGML_ASSERT(M <= 32767);

    vals_out.assign((size_t) k * (size_t) n_out, ggml_fp32_to_fp16(0.0f));

    n_threads = std::max(1, n_threads);
    n_threads = std::min<int>(n_threads, (int) std::max<int64_t>(1, n_out));

    const int64_t t0 = ggml_time_us();
    const int64_t progress_every = std::max<int64_t>(1, n_out / 20);

    const int64_t B = std::min<int64_t>(64, n_out);

    const size_t size_dict = (size_t) n_in * (size_t) M * sizeof(float);
    const size_t size_wblk = (size_t) n_in * (size_t) B * sizeof(float);
    const size_t size_y    = (size_t) M    * (size_t) B * sizeof(float);

    const size_t mem_size =
            ggml_tensor_overhead() * 16 +
            ggml_graph_overhead_custom(256, /*grads*/ false) +
            size_dict + size_wblk + size_y +
            1024*1024;

    ggml_init_params params = { mem_size, nullptr, false };
    ggml_context * ctx = ggml_init(params);
    GGML_ASSERT(ctx != nullptr);

    ggml_tensor * t_dict = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_in, M);
    ggml_tensor * t_wblk = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_in, B);

    std::memcpy(t_dict->data, D.data(), size_dict);

    ggml_tensor * t_y = ggml_mul_mat(ctx, t_dict, t_wblk);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, t_y);

    ggml_threadpool_params tpp = ggml_threadpool_params_default(n_threads);
    ggml_threadpool * threadpool = ggml_threadpool_new(&tpp);
    GGML_ASSERT(threadpool != nullptr);

    ggml_cplan plan = ggml_graph_plan(gf, n_threads, threadpool);
    std::vector<uint8_t> work_buffer;
    if (plan.work_size > 0) {
        work_buffer.resize(plan.work_size);
        plan.work_data = work_buffer.data();
    }

    auto * wblk_data = (float *) t_wblk->data;
    const auto * y_data = (const float *) t_y->data;

    std::vector<float> w;

    int64_t last_print = 0;
    for (int64_t col0 = 0; col0 < n_out; col0 += B) {
        const int64_t nb = std::min<int64_t>(B, n_out - col0);

        for (int64_t b = 0; b < nb; ++b) {
            read_column_f32(W, col0 + b, w);
            if (w_scale && (int64_t) w_scale->size() == n_in) {
                for (int64_t i = 0; i < n_in; ++i) {
                    w[i] *= (*w_scale)[(size_t) i];
                }
            }
            std::memcpy(wblk_data + b * n_in, w.data(), (size_t) n_in * sizeof(float));
        }
        for (int64_t b = nb; b < B; ++b) {
            std::memset(wblk_data + b * n_in, 0, (size_t) n_in * sizeof(float));
        }

        const enum ggml_status st = ggml_graph_compute(gf, &plan);
        GGML_ASSERT(st == GGML_STATUS_SUCCESS);

        for (int64_t b = 0; b < nb; ++b) {
            const int64_t col = col0 + b;
            const float * y_col = y_data + b * M;

            const int16_t * codes_col = codes.data() + (size_t) col * (size_t) k;
            ggml_fp16_t * vals_col = vals_out.data() + (size_t) col * (size_t) k;

            for (int ti = 0; ti < k; ++ti) {
                const int16_t code = codes_col[ti];
                if (code == 0) {
                    vals_col[ti] = ggml_fp32_to_fp16(0.0f);
                    continue;
                }

                const int64_t atom = (int64_t) std::abs(code) - 1;
                if (atom < 0 || atom >= M) {
                    vals_col[ti] = ggml_fp32_to_fp16(0.0f);
                    continue;
                }

                float v = std::fabs(y_col[atom]);
                if (!std::isfinite(v) || v < 0.0f) {
                    v = 0.0f;
                }
                if (v > 65504.0f) { // max finite fp16
                    v = 65504.0f;
                }
                vals_col[ti] = ggml_fp32_to_fp16(v);
            }
        }

        const int64_t done = std::min<int64_t>(col0 + nb, n_out);
        if (done >= last_print + progress_every || done == n_out) {
            last_print = done;
            const double pct = 100.0 * double(done) / double(n_out);
            const double sec = double(ggml_time_us() - t0) / 1e6;
            fprintf(stderr, "\r  [%s] computing vals %" PRId64 "/%" PRId64 " (%.1f%%) elapsed %.1fs (%d thr, B=%" PRId64 ")",
                    tag.c_str(), done, n_out, pct, sec, n_threads, B);
            fflush(stderr);
            if (done == n_out) {
                fprintf(stderr, "\n");
            }
        }
    }

    ggml_threadpool_free(threadpool);
    ggml_free(ctx);
}

static void compute_row_scale_sign(
        const ggml_tensor * W,
        const std::vector<float> & D,
        int64_t M,
        int k,
        const std::vector<int16_t> & codes,
        const std::vector<ggml_fp16_t> * vals,
        int n_threads,
        std::vector<ggml_fp16_t> & row_scale_out,
        const std::string & tag) {
    const int64_t n_in  = W->ne[0];
    const int64_t n_out = W->ne[1];

    GGML_ASSERT((int64_t) codes.size() == (int64_t) k * n_out);
    GGML_ASSERT(!vals || (int64_t) vals->size() == (int64_t) k * n_out);
    row_scale_out.assign((size_t) n_out, ggml_fp32_to_fp16(1.0f));

    n_threads = std::max(1, n_threads);
    n_threads = std::min<int>(n_threads, (int) std::max<int64_t>(1, n_out));

    const int64_t t0 = ggml_time_us();

    if (n_threads == 1) {
        const int64_t progress_every = std::max<int64_t>(1, n_out / 20);

        std::vector<float> w;
        std::vector<float> w_hat((size_t) n_in, 0.0f);

        for (int64_t col = 0; col < n_out; ++col) {
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

            double dot = 0.0;
            double norm2 = 0.0;
            for (int64_t i = 0; i < n_in; ++i) {
                const double wh = w_hat[i];
                dot   += wh * (double) w[i];
                norm2 += wh * wh;
            }

            float scale = 1.0f;
            if (norm2 > 1e-20) {
                scale = (float) (dot / norm2);
            }
            if (!std::isfinite(scale) || scale < 0.0f) {
                scale = 0.0f;
            }
            row_scale_out[(size_t) col] = ggml_fp32_to_fp16(scale);

            if ((col + 1) % progress_every == 0 || col + 1 == n_out) {
                const double pct = 100.0 * double(col + 1) / double(n_out);
                const double sec = double(ggml_time_us() - t0) / 1e6;
                fprintf(stderr, "\r  [%s] computing row_scale %" PRId64 "/%" PRId64 " (%.1f%%) elapsed %.1fs",
                        tag.c_str(), col + 1, n_out, pct, sec);
                fflush(stderr);
                if (col + 1 == n_out) {
                    fprintf(stderr, "\n");
                }
            }
        }
        return;
    }

    std::atomic<int64_t> next_col{0};
    std::atomic<int64_t> done_cols{0};

    auto worker = [&]() {
        std::vector<float> w;
        std::vector<float> w_hat((size_t) n_in, 0.0f);

        while (true) {
            const int64_t col = next_col.fetch_add(1);
            if (col >= n_out) {
                break;
            }

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

            double dot = 0.0;
            double norm2 = 0.0;
            for (int64_t i = 0; i < n_in; ++i) {
                const double wh = w_hat[i];
                dot   += wh * (double) w[i];
                norm2 += wh * wh;
            }

            float scale = 1.0f;
            if (norm2 > 1e-20) {
                scale = (float) (dot / norm2);
            }
            if (!std::isfinite(scale) || scale < 0.0f) {
                scale = 0.0f;
            }
            row_scale_out[(size_t) col] = ggml_fp32_to_fp16(scale);

            done_cols.fetch_add(1, std::memory_order_relaxed);
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(n_threads);
    for (int i = 0; i < n_threads; ++i) {
        threads.emplace_back(worker);
    }

    const int64_t progress_every = std::max<int64_t>(1, n_out / 20);
    int64_t last_print = 0;
    while (true) {
        const int64_t done = done_cols.load(std::memory_order_relaxed);
        if (done >= n_out) {
            break;
        }

        if (done >= last_print + progress_every) {
            last_print = done;
            const double pct = 100.0 * double(done) / double(n_out);
            const double sec = double(ggml_time_us() - t0) / 1e6;
            fprintf(stderr, "\r  [%s] computing row_scale %" PRId64 "/%" PRId64 " (%.1f%%) elapsed %.1fs (%d thr)",
                    tag.c_str(), done, n_out, pct, sec, n_threads);
            fflush(stderr);
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }

    for (auto & t : threads) {
        t.join();
    }

    {
        const double sec = double(ggml_time_us() - t0) / 1e6;
        fprintf(stderr, "\r  [%s] computing row_scale %" PRId64 "/%" PRId64 " (100.0%%) elapsed %.1fs (%d thr)\n",
                tag.c_str(), n_out, n_out, sec, n_threads);
    }
}



int sc_build(
        const sc_args & args,
        const sc_report_config & report_cfg,
        int imatrix_chunk_count,
        const std::unordered_map<std::string, std::vector<float>> & imatrix_data) {
    std::string in_fname = args.in_fname;
    std::string out_fname = args.out_fname;

    int64_t dict_M = args.dict_M;
    int64_t dict_M_gate = args.dict_M_gate;
    int64_t dict_M_up   = args.dict_M_up;
    int64_t dict_M_down = args.dict_M_down;
    int dict_k = args.dict_k;
    int dict_k_gate = args.dict_k_gate;
    int dict_k_up   = args.dict_k_up;
    int dict_k_down = args.dict_k_down;
    float dict_eta = args.dict_eta;
    int dict_iters = args.dict_iters;
    int64_t dict_max_samples = args.dict_max_samples;
    int n_threads = args.n_threads;
    std::string layers_range = args.layers_range;
    std::string dict_type_str = args.dict_type_str;
    bool write_vals = args.write_vals;
    bool write_row_scale = args.write_row_scale;
    int64_t eval_cols = args.eval_cols;
    std::string report_json = args.report_json;
    std::string imatrix_file = args.imatrix_file;
    float imatrix_eps = args.imatrix_eps;
    float imatrix_power = args.imatrix_power;
    int64_t checkpoint_every = args.checkpoint_every;
    int seed = args.seed;
    const bool has_imatrix = report_cfg.imatrix.enabled;
    const std::string imatrix_dataset = report_cfg.imatrix.dataset;

    if (in_fname.empty() || out_fname.empty()) {
        fprintf(stderr, "statecells-build: missing input or output filename\n");
        return 1;
    }

    const int64_t t0_total = ggml_time_us();

    const std::string src_fname = report_cfg.source.empty() ? in_fname : report_cfg.source;

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
    if (has_imatrix) {
        gguf_set_val_str(dst, "statecells.imatrix.file", imatrix_file.c_str());
        if (!imatrix_dataset.empty()) {
            gguf_set_val_str(dst, "statecells.imatrix.dataset", imatrix_dataset.c_str());
        }
        if (imatrix_chunk_count >= 0) {
            gguf_set_val_u32(dst, "statecells.imatrix.chunks", (uint32_t) imatrix_chunk_count);
        }
        gguf_set_val_f32(dst, "statecells.imatrix.eps", imatrix_eps);
        gguf_set_val_f32(dst, "statecells.imatrix.power", imatrix_power);
    }

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

    std::vector<sc_report_row> report_rows;

    const auto select_M_for_kind = [&](const std::string & kind) -> int64_t {
        if (kind == "ffn_gate" && dict_M_gate > 0) return dict_M_gate;
        if (kind == "ffn_up"   && dict_M_up   > 0) return dict_M_up;
        if (kind == "ffn_down" && dict_M_down > 0) return dict_M_down;
        return dict_M;
    };

    const auto select_k_for_kind = [&](const std::string & kind) -> int {
        if (kind == "ffn_gate" && dict_k_gate > 0) return dict_k_gate;
        if (kind == "ffn_up"   && dict_k_up   > 0) return dict_k_up;
        if (kind == "ffn_down" && dict_k_down > 0) return dict_k_down;
        return dict_k;
    };

    // Count existing StateCells payload tensors so --resume doesn't accidentally disable the feature.
    int64_t n_sc_existing = 0;
    std::regex re_sc_payload(R"(blk\.(\d+)\.(ffn_gate|ffn_up|ffn_down)\.(dict|codes|row_scale)$)");
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
            const std::string vals_name  = "blk." + std::to_string(il) + "." + kind + ".vals";
            const std::string row_scale_name = "blk." + std::to_string(il) + "." + kind + ".row_scale";

            const bool have_dict  = gguf_find_tensor(src, dict_name.c_str())  != -1;
            const bool have_codes = gguf_find_tensor(src, codes_name.c_str()) != -1;
            const bool have_vals  = gguf_find_tensor(src, vals_name.c_str())  != -1;
            const bool have_row_scale = gguf_find_tensor(src, row_scale_name.c_str()) != -1;
            if ((have_dict || have_codes || have_vals || have_row_scale) && (have_dict != have_codes || (!have_dict && (have_vals || have_row_scale)))) {
                fprintf(stderr, "statecells-build: inconsistent existing tensors for %s (dict=%d codes=%d row_scale=%d)\n",
                        weight_name.c_str(), (int) have_dict, (int) have_codes, (int) have_row_scale);
                return 1;
            }

            const bool need_dict_codes = !(have_dict && have_codes);
            const bool need_vals       = write_vals && !have_vals;
            const bool need_row_scale  = write_row_scale && !have_row_scale;
            if (!need_dict_codes && !need_vals && !need_row_scale) {
                continue;
            }

            ggml_tensor * W = ggml_get_tensor(ctx_data, weight_name.c_str());
            if (!W || ggml_n_dims(W) != 2) {
                continue;
            }

            const int64_t n_in  = W->ne[0];
            const int64_t n_out = W->ne[1];

            const int64_t dict_M_cur = select_M_for_kind(kind);
            const int dict_k_cur = select_k_for_kind(kind);

            const std::string tag = "blk." + std::to_string(il) + "." + kind;

            std::vector<float> imatrix_w_scale_sqrt;
            std::vector<float> imatrix_w_scale2;
            const std::vector<float> * w_scale_train = nullptr;
            const std::vector<float> * w_scale_encode = nullptr;

            if (!imatrix_data.empty()) {
                if (make_imatrix_sqrt_scale(imatrix_data, weight_name, n_in, imatrix_eps, imatrix_power, imatrix_w_scale_sqrt)) {
                    w_scale_train = &imatrix_w_scale_sqrt;
                    imatrix_w_scale2 = imatrix_w_scale_sqrt;
                    for (float & v : imatrix_w_scale2) {
                        v *= v;
                    }
                    w_scale_encode = &imatrix_w_scale2;
                } else {
                    fprintf(stderr, "statecells-build: warning: missing imatrix entry for %s; proceeding unweighted\n", weight_name.c_str());
                }
            }

            int64_t M_eff = 0;
            int k_eff = 0;
            std::vector<float> D;
            std::vector<int16_t> codes;
            std::vector<ggml_fp16_t> vals;

            if (need_dict_codes) {
                printf("statecells-build: layer %" PRId64 " %s [% " PRId64 " x %" PRId64 "] type=%s M=%" PRId64 " k=%d\n",
                       il, kind.c_str(), n_in, n_out, ggml_type_name(W->type), dict_M_cur, dict_k_cur);

                train_dict_oja_kwta(W, dict_M_cur, dict_k_cur, dict_eta, dict_iters, dict_max_samples, rng, D, tag, w_scale_train);

                M_eff = std::min<int64_t>(dict_M_cur, n_out);
                k_eff = std::min<int>(dict_k_cur, (int) M_eff);

                if (w_scale_train) {
                    unscale_dict_inplace(D, n_in, M_eff, *w_scale_train);
                }

                encode_codes_sign(W, D, M_eff, k_eff, n_threads, codes, need_vals ? &vals : nullptr, tag, w_scale_encode);
            } else {
                ggml_tensor * t_dict  = ggml_get_tensor(ctx_data, dict_name.c_str());
                ggml_tensor * t_codes = ggml_get_tensor(ctx_data, codes_name.c_str());
                if (!t_dict || !t_codes) {
                    fprintf(stderr, "statecells-build: missing existing dict/codes tensors for %s\n", weight_name.c_str());
                    return 1;
                }
                if (ggml_n_dims(t_dict) != 2 || ggml_n_dims(t_codes) != 2 || t_codes->type != GGML_TYPE_I16) {
                    fprintf(stderr, "statecells-build: invalid existing dict/codes tensor types for %s\n", weight_name.c_str());
                    return 1;
                }

                M_eff = t_dict->ne[1];
                k_eff = (int) t_codes->ne[0];

                if (t_dict->ne[0] != n_in || t_codes->ne[1] != n_out) {
                    fprintf(stderr, "statecells-build: mismatched shapes for %s (W=[%" PRId64 "x%" PRId64 "] dict=[%" PRId64 "x%" PRId64 "] codes=[%" PRId64 "x%" PRId64 "])\n",
                            weight_name.c_str(), n_in, n_out,
                            (int64_t) t_dict->ne[0], (int64_t) t_dict->ne[1],
                            (int64_t) t_codes->ne[0], (int64_t) t_codes->ne[1]);
                    return 1;
                }

                std::string add_what;
                if (need_vals) {
                    add_what += " vals";
                }
                if (need_row_scale) {
                    add_what += " row_scale";
                }
                printf("statecells-build: layer %" PRId64 " %s add%s only (M=%" PRId64 " k=%d)\n",
                       il, kind.c_str(), add_what.c_str(), M_eff, k_eff);

                read_mat_f16_f32_to_f32(t_dict, D);
                codes.resize((size_t) k_eff * (size_t) n_out);
                std::memcpy(codes.data(), t_codes->data, codes.size() * sizeof(int16_t));

                if (need_vals) {
                    compute_vals_sign(W, D, M_eff, k_eff, codes, n_threads, vals, tag, w_scale_encode);
                } else if (have_vals && (need_row_scale || eval_cols > 0)) {
                    ggml_tensor * t_vals = ggml_get_tensor(ctx_data, vals_name.c_str());
                    if (!t_vals || ggml_n_dims(t_vals) != 2) {
                        fprintf(stderr, "statecells-build: missing existing vals tensor for %s\n", weight_name.c_str());
                        return 1;
                    }
                    if (t_vals->type != GGML_TYPE_F16 && t_vals->type != GGML_TYPE_F32) {
                        fprintf(stderr, "statecells-build: invalid existing vals tensor type for %s\n", weight_name.c_str());
                        return 1;
                    }
                    if (t_vals->ne[0] != k_eff || t_vals->ne[1] != n_out) {
                        fprintf(stderr, "statecells-build: mismatched vals shape for %s (vals=[%" PRId64 "x%" PRId64 "] expected=[%dx%" PRId64 "])\n",
                                weight_name.c_str(), (int64_t) t_vals->ne[0], (int64_t) t_vals->ne[1], k_eff, n_out);
                        return 1;
                    }

                    read_mat_f16_f32_to_fp16(t_vals, vals);
                }
            }

            std::vector<ggml_fp16_t> row_scale;
            if (need_row_scale) {
                compute_row_scale_sign(W, D, M_eff, k_eff, codes, vals.empty() ? nullptr : &vals, n_threads, row_scale, tag);
            } else if (have_row_scale && eval_cols > 0) {
                ggml_tensor * t_rs = ggml_get_tensor(ctx_data, row_scale_name.c_str());
                if (!t_rs || ggml_n_dims(t_rs) != 1) {
                    fprintf(stderr, "statecells-build: missing existing row_scale tensor for %s\n", weight_name.c_str());
                    return 1;
                }
                if (t_rs->type != GGML_TYPE_F16 && t_rs->type != GGML_TYPE_F32) {
                    fprintf(stderr, "statecells-build: invalid existing row_scale tensor type for %s\n", weight_name.c_str());
                    return 1;
                }
                if (t_rs->ne[0] != n_out) {
                    fprintf(stderr, "statecells-build: mismatched row_scale shape for %s (row_scale=[%" PRId64 "] expected=[%" PRId64 "])\n",
                            weight_name.c_str(), (int64_t) t_rs->ne[0], n_out);
                    return 1;
                }

                read_vec_f16_f32_to_fp16(t_rs, row_scale);
            }

            eval_metrics em;
            if (eval_cols > 0) {
                const int64_t t0 = ggml_time_us();
                em = eval_reconstruction_sign(W, D, M_eff, k_eff, codes, vals.empty() ? nullptr : &vals, row_scale.empty() ? nullptr : &row_scale, eval_cols, rng, w_scale_train);
                const double sec = double(ggml_time_us() - t0) / 1e6;
                if (w_scale_train) {
                    fprintf(stderr, "  [%s] eval cols=%" PRId64 " rel_l2 mean=%.4f p95=%.4f cos mean=%.4f p05=%.4f | rel_l2_w mean=%.4f p95=%.4f cos_w mean=%.4f p05=%.4f (%.1fs)\n",
                            tag.c_str(), eval_cols,
                            em.rel_l2_mean, em.rel_l2_p95, em.cos_mean, em.cos_p05,
                            em.rel_l2_mean_w, em.rel_l2_p95_w, em.cos_mean_w, em.cos_p05_w,
                            sec);
                } else {
                    fprintf(stderr, "  [%s] eval cols=%" PRId64 " rel_l2 mean=%.4f p95=%.4f cos mean=%.4f p05=%.4f (%.1fs)\n",
                            tag.c_str(), eval_cols, em.rel_l2_mean, em.rel_l2_p95, em.cos_mean, em.cos_p05, sec);
                }
            }

            // Allocate a dedicated ggml context for dict+codes to avoid a giant arena.
            const size_t size_dict  = need_dict_codes ? (size_t) n_in * (size_t) M_eff * ggml_type_size(dict_type) : 0;
            const size_t size_codes = need_dict_codes ? (size_t) k_eff * (size_t) n_out * sizeof(int16_t) : 0;
            const size_t size_vals  = need_vals ? (size_t) k_eff * (size_t) n_out * sizeof(ggml_fp16_t) : 0;
            const size_t size_row_scale = need_row_scale ? (size_t) n_out * sizeof(ggml_fp16_t) : 0;
            const size_t mem_size_sc = ggml_tensor_overhead() * 8 + size_dict + size_codes + size_vals + size_row_scale;
            ggml_init_params sc_params = { mem_size_sc, nullptr, false };
            ggml_context * ctx_sc = ggml_init(sc_params);
            sc_contexts.push_back(ctx_sc);

            if (need_dict_codes) {
                // Create dict tensor.
                ggml_tensor * t_dict = ggml_new_tensor_2d(ctx_sc, dict_type, n_in, M_eff);
                ggml_set_name(t_dict, dict_name.c_str());

                if (dict_type == GGML_TYPE_F16) {
                    auto * dst_f16 = (ggml_fp16_t *) t_dict->data;
                    for (int64_t j = 0; j < M_eff; ++j) {
                        ggml_fp32_to_fp16_row(D.data() + j * n_in, dst_f16 + j * n_in, n_in);
                    }
                } else {
                    std::memcpy(t_dict->data, D.data(), (size_t) n_in * (size_t) M_eff * sizeof(float));
                }

                // Create codes tensor.
                ggml_tensor * t_codes = ggml_new_tensor_2d(ctx_sc, GGML_TYPE_I16, k_eff, n_out);
                ggml_set_name(t_codes, codes_name.c_str());
                std::memcpy(t_codes->data, codes.data(), codes.size() * sizeof(int16_t));

                gguf_add_tensor(dst, t_dict);
                gguf_add_tensor(dst, t_codes);
                n_added += 2;
            }

            if (need_vals) {
                ggml_tensor * t_vals = ggml_new_tensor_2d(ctx_sc, GGML_TYPE_F16, k_eff, n_out);
                ggml_set_name(t_vals, vals_name.c_str());
                std::memcpy(t_vals->data, vals.data(), vals.size() * sizeof(ggml_fp16_t));
                gguf_add_tensor(dst, t_vals);
                n_added += 1;
            }

            if (need_row_scale) {
                ggml_tensor * t_rs = ggml_new_tensor_1d(ctx_sc, GGML_TYPE_F16, n_out);
                ggml_set_name(t_rs, row_scale_name.c_str());
                std::memcpy(t_rs->data, row_scale.data(), row_scale.size() * sizeof(ggml_fp16_t));
                gguf_add_tensor(dst, t_rs);
                n_added += 1;
            }
            layer_added_any = true;

            if (!report_json.empty() || eval_cols > 0) {
                report_rows.push_back(sc_report_row{ il, kind, n_in, n_out, M_eff, k_eff, eval_cols, em, w_scale_train != nullptr });
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
        if (!sc_write_report_json(report_json, report_cfg, report_rows)) {
            fprintf(stderr, "failed to write report to %s\n", report_json.c_str());
        }
    }

    for (auto * c : sc_contexts) {
        ggml_free(c);
    }
    ggml_free(ctx_data);
    gguf_free(dst);
    gguf_free(src);

    {
        const double sec = double(ggml_time_us() - t0_total) / 1e6;
        fprintf(stderr, "statecells-build: done in %.1fs\n", sec);
    }

    return 0;
}
