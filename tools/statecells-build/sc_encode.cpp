#include "include/sc_encode.h"

#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <thread>

#include "ggml-cpu.h"
#include "include/sc_utils.h"

void train_dict_oja_kwta(
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

    // codes store ±(atom+1) in int16.
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

void compute_row_scale_sign(
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
            const int64_t col = next_col.fetch_add(1, std::memory_order_relaxed);
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
