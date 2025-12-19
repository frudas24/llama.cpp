#include "include/sc_encode.h"

#include <atomic>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <thread>
#include <vector>

#include "ggml-cpu.h"
#include "include/sc_utils.h"

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
