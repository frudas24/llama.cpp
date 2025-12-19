#include "include/sc_encode.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "ggml-cpu.h"
#include "include/sc_utils.h"

namespace {

struct block_gemm {
    ggml_context * ctx = nullptr;
    ggml_tensor  * t_dict = nullptr;
    ggml_tensor  * t_wblk = nullptr;
    ggml_tensor  * t_y = nullptr;
    ggml_cgraph  * gf = nullptr;
    ggml_threadpool * threadpool = nullptr;
    ggml_cplan plan = {};
    std::vector<uint8_t> work;
    int64_t B = 0;

    block_gemm(int64_t n_in, int64_t M, int64_t B_in, int n_threads) : B(B_in) {
        const size_t size_dict = (size_t) n_in * (size_t) M * sizeof(float);
        const size_t size_wblk = (size_t) n_in * (size_t) B * sizeof(float);
        const size_t size_y    = (size_t) M    * (size_t) B * sizeof(float);

        const size_t mem_size =
                ggml_tensor_overhead() * 16 +
                ggml_graph_overhead_custom(256, /*grads*/ false) +
                size_dict + size_wblk + size_y +
                1024*1024;

        ggml_init_params params = { mem_size, nullptr, false };
        ctx = ggml_init(params);
        t_dict = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_in, M);
        t_wblk = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_in, B);
        t_y    = ggml_mul_mat(ctx, t_dict, t_wblk);

        gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, t_y);

        ggml_threadpool_params tpp = ggml_threadpool_params_default(n_threads);
        threadpool = ggml_threadpool_new(&tpp);

        plan = ggml_graph_plan(gf, n_threads, threadpool);
        if (plan.work_size > 0) {
            work.resize(plan.work_size);
            plan.work_data = work.data();
        }
    }

    ~block_gemm() {
        if (threadpool) ggml_threadpool_free(threadpool);
        if (ctx) ggml_free(ctx);
    }
};

} // namespace

void encode_codes_sign(
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
        if (vals_out) vals_out->clear();
        return;
    }

    GGML_ASSERT(M <= 32767);

    codes_out.assign((size_t) k * (size_t) n_out, (int16_t) 0);
    if (vals_out) {
        vals_out->assign((size_t) k * (size_t) n_out, ggml_fp32_to_fp16(0.0f));
    }

    n_threads = std::max(1, n_threads);
    n_threads = std::min<int>(n_threads, (int) std::max<int64_t>(1, n_out));

    const int64_t t0 = ggml_time_us();
    const int64_t progress_every = std::max<int64_t>(1, n_out / 20);
    const int64_t B = std::min<int64_t>(64, n_out);

    block_gemm bg(n_in, M, B, n_threads);
    std::memcpy(bg.t_dict->data, D.data(), (size_t) n_in * (size_t) M * sizeof(float));

    auto * wblk_data = (float *) bg.t_wblk->data;
    const auto * y_data = (const float *) bg.t_y->data;

    std::vector<int> idx((size_t) M);
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

        const enum ggml_status st = ggml_graph_compute(bg.gf, &bg.plan);
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
                    if (v > 65504.0f) {
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
}

void compute_vals_sign(
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

    GGML_ASSERT(M <= 32767);

    vals_out.assign((size_t) k * (size_t) n_out, ggml_fp32_to_fp16(0.0f));

    n_threads = std::max(1, n_threads);
    n_threads = std::min<int>(n_threads, (int) std::max<int64_t>(1, n_out));

    const int64_t t0 = ggml_time_us();
    const int64_t progress_every = std::max<int64_t>(1, n_out / 20);
    const int64_t B = std::min<int64_t>(64, n_out);

    block_gemm bg(n_in, M, B, n_threads);
    std::memcpy(bg.t_dict->data, D.data(), (size_t) n_in * (size_t) M * sizeof(float));

    auto * wblk_data = (float *) bg.t_wblk->data;
    const auto * y_data = (const float *) bg.t_y->data;

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

        const enum ggml_status st = ggml_graph_compute(bg.gf, &bg.plan);
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
                if (v > 65504.0f) {
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
}
