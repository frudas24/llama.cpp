#include "llama-seeddelta.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

static inline float read_f16_or_f32(const ggml_tensor * t, const uint8_t * base, int64_t i0, int64_t i1) {
    const size_t off = i0 * t->nb[0] + i1 * t->nb[1];
    if (t->type == GGML_TYPE_F16) {
        const ggml_fp16_t v = *(const ggml_fp16_t *)(base + off);
        return ggml_fp16_to_fp32(v);
    }
    const float v = *(const float *)(base + off);
    return v;
}

static inline float read_f16_or_f32_1d(const ggml_tensor * t, const uint8_t * base, int64_t i0) {
    const size_t off = i0 * t->nb[0];
    if (t->type == GGML_TYPE_F16) {
        const ggml_fp16_t v = *(const ggml_fp16_t *)(base + off);
        return ggml_fp16_to_fp32(v);
    }
    const float v = *(const float *)(base + off);
    return v;
}

static inline float read_x_f16_or_f32(const ggml_tensor * x, const uint8_t * base, int64_t i0, int64_t i1) {
    const size_t off = i0 * x->nb[0] + i1 * x->nb[1];
    if (x->type == GGML_TYPE_F16) {
        const ggml_fp16_t v = *(const ggml_fp16_t *)(base + off);
        return ggml_fp16_to_fp32(v);
    }
    const float v = *(const float *)(base + off);
    return v;
}

static inline int32_t read_idx_i16_i32(const ggml_tensor * idx, const uint8_t * base, int64_t i0, int64_t i1) {
    const size_t off = i0 * idx->nb[0] + i1 * idx->nb[1];
    if (idx->type == GGML_TYPE_I16) {
        return (int32_t) *(const int16_t *)(base + off);
    }
    return *(const int32_t *)(base + off);
}

static inline float read_base_f16_or_f32(const ggml_tensor * t, const uint8_t * base, int64_t i0, int64_t i1) {
    const size_t off = i0 * t->nb[0] + i1 * t->nb[1];
    if (t->type == GGML_TYPE_F16) {
        return ggml_fp16_to_fp32(*(const ggml_fp16_t *)(base + off));
    }
    return *(const float *)(base + off);
}

static inline int32_t read_base_i16_i32(const ggml_tensor * t, const uint8_t * base, int64_t i0, int64_t i1) {
    const size_t off = i0 * t->nb[0] + i1 * t->nb[1];
    if (t->type == GGML_TYPE_I16) {
        return (int32_t) *(const int16_t *)(base + off);
    }
    return *(const int32_t *)(base + off);
}

static std::atomic_flag llama_seeddelta_nan_warned = ATOMIC_FLAG_INIT;
static std::atomic<int64_t> llama_seeddelta_nan_count{0};
static thread_local std::vector<float> llama_seeddelta_wcol_tls;
static thread_local std::vector<uint8_t> llama_seeddelta_wbuf_tls;
static thread_local std::vector<uint8_t> llama_seeddelta_xbuf_tls;
static const bool llama_seeddelta_debug_data = std::getenv("LLAMA_SEEDDELTA_DEBUG_DATA") != nullptr;
static const bool llama_seeddelta_debug_stats = std::getenv("LLAMA_SEEDDELTA_DEBUG_STATS") != nullptr;
static std::atomic<int> llama_seeddelta_debug_budget{16};
static std::atomic<int64_t> llama_seeddelta_debug_stats_budget{100000};

static int64_t llama_seeddelta_env_int64(const char * name, int64_t fallback) {
    const char * value = std::getenv(name);
    if (!value || !value[0]) {
        return fallback;
    }
    char * end = nullptr;
    const long long parsed = std::strtoll(value, &end, 10);
    if (!end || end == value) {
        return fallback;
    }
    return parsed;
}

static const int64_t llama_seeddelta_debug_stats_per_tensor =
    llama_seeddelta_env_int64("LLAMA_SEEDDELTA_DEBUG_STATS_PER_TENSOR", 1024);

struct llama_seeddelta_debug_stats_entry {
    std::string name;
    uint64_t count = 0;
    double sum_abs = 0.0;
    double sum_diff = 0.0;
    double max_abs = 0.0;
};

class llama_seeddelta_debug_stats_accum {
public:
    llama_seeddelta_debug_stats_accum(bool enabled, int64_t per_tensor_cap)
        : enabled_(enabled),
          per_tensor_cap_(per_tensor_cap > 0 ? (uint64_t) per_tensor_cap : 0) {}

    bool should_sample(const ggml_tensor * w_ref) {
        if (!enabled_ || per_tensor_cap_ == 0) {
            return false;
        }
        std::lock_guard<std::mutex> lock(mu_);
        auto & entry = entries_[w_ref];
        return entry.count < per_tensor_cap_;
    }

    void add(const ggml_tensor * w_ref, double diff) {
        if (!enabled_) {
            return;
        }
        std::lock_guard<std::mutex> lock(mu_);
        auto & entry = entries_[w_ref];
        if (entry.name.empty()) {
            const char * name = w_ref ? ggml_get_name(w_ref) : nullptr;
            entry.name = (name && name[0]) ? name : "(unnamed)";
        }
        entry.count += 1;
        const double abs_diff = std::abs(diff);
        entry.sum_abs += abs_diff;
        entry.sum_diff += diff;
        if (abs_diff > entry.max_abs) {
            entry.max_abs = abs_diff;
        }
    }

    ~llama_seeddelta_debug_stats_accum() {
        if (!enabled_) {
            return;
        }
        std::lock_guard<std::mutex> lock(mu_);
        if (entries_.empty()) {
            std::fprintf(stderr, "[seeddelta-debug-stats] no samples collected\n");
            return;
        }

        struct row_view {
            const llama_seeddelta_debug_stats_entry * entry;
        };
        std::vector<row_view> rows;
        rows.reserve(entries_.size());

        uint64_t total = 0;
        for (const auto & it : entries_) {
            rows.push_back({ &it.second });
            total += it.second.count;
        }

        std::sort(rows.begin(), rows.end(), [](const row_view & a, const row_view & b) {
            return a.entry->max_abs > b.entry->max_abs;
        });

        std::fprintf(stderr, "[seeddelta-debug-stats] tensors=%zu samples=%" PRIu64 "\n",
                     entries_.size(), total);
        std::fprintf(stderr, "[seeddelta-debug-stats] top by max_abs:\n");

        const size_t limit = std::min<size_t>(16, rows.size());
        for (size_t i = 0; i < limit; ++i) {
            const auto * e = rows[i].entry;
            const double mean_abs = e->count ? (e->sum_abs / (double) e->count) : 0.0;
            const double mean_diff = e->count ? (e->sum_diff / (double) e->count) : 0.0;
            std::fprintf(stderr,
                         "  %s count=%" PRIu64 " mean_abs=%.6g max_abs=%.6g mean_diff=%.6g\n",
                         e->name.c_str(), e->count, mean_abs, e->max_abs, mean_diff);
        }
    }

private:
    bool enabled_ = false;
    uint64_t per_tensor_cap_ = 0;
    std::mutex mu_;
    std::unordered_map<const ggml_tensor *, llama_seeddelta_debug_stats_entry> entries_;
};

static llama_seeddelta_debug_stats_accum llama_seeddelta_debug_stats_accum_instance(
    llama_seeddelta_debug_stats,
    llama_seeddelta_debug_stats_per_tensor);

enum class llama_seeddelta_fb_status {
    ok,
    no_w,
    no_x,
    no_data,
    non_host,
    dim_mismatch,
    n_in_mismatch,
    o_range,
    t_range,
    read_fail,
};

static bool llama_seeddelta_tensor_read(
        const ggml_tensor * t,
        size_t offset,
        size_t size,
        std::vector<uint8_t> & buf,
        const uint8_t ** out) {
    (void) buf;
    if (!t || !out || size == 0) {
        return false;
    }
    const ggml_tensor * base = t->view_src ? t->view_src : t;
    ggml_backend_buffer_t buffer = base->buffer;
    if (!buffer || !base->data) {
        return false;
    }
    if (!ggml_backend_buffer_is_host(buffer)) {
        return false;
    }
    if (offset + size > ggml_nbytes(t)) {
        return false;
    }
    *out = (const uint8_t *) t->data + offset;
    return true;
}

static bool llama_seeddelta_read_weight_col_f32(const ggml_tensor * w, int64_t col, std::vector<float> & out) {
    if (!w || ggml_n_dims(w) != 2) {
        return false;
    }
    if (!w->data) {
        return false;
    }

    const int64_t n_in  = w->ne[0];
    const int64_t n_out = w->ne[1];
    if (n_in <= 0 || col < 0 || col >= n_out) {
        return false;
    }

    const size_t row_bytes = (size_t) w->nb[1];
    const size_t offset = (size_t) col * w->nb[1];
    const uint8_t * w_data = nullptr;
    if (!llama_seeddelta_tensor_read(w, offset, row_bytes, llama_seeddelta_wbuf_tls, &w_data)) {
        return false;
    }

    out.resize((size_t) n_in);

    const uint8_t * base = w_data;
    const auto * traits = ggml_get_type_traits(w->type);
    if (!traits) {
        return false;
    }

    if (!traits->is_quantized) {
        if (w->type == GGML_TYPE_F32) {
            if ((size_t) w->nb[0] == sizeof(float)) {
                std::memcpy(out.data(), base, (size_t) n_in * sizeof(float));
            } else {
                for (int64_t i = 0; i < n_in; ++i) {
                    out[(size_t) i] = *(const float *)(base + i * w->nb[0]);
                }
            }
            return true;
        }
        if (w->type == GGML_TYPE_F16) {
            for (int64_t i = 0; i < n_in; ++i) {
                out[(size_t) i] = ggml_fp16_to_fp32(*(const ggml_fp16_t *)(base + i * w->nb[0]));
            }
            return true;
        }
        if (traits->to_float) {
            traits->to_float(base, out.data(), n_in);
            return true;
        }
        return false;
    }

    if (!traits->to_float) {
        return false;
    }
    traits->to_float(base, out.data(), n_in);
    return true;
}

static bool llama_seeddelta_dense_fallback(
        const ggml_tensor * w_ref,
        const ggml_tensor * x,
        int64_t o,
        int64_t t,
        float & y_out,
        llama_seeddelta_fb_status * fb_status = nullptr) {
    auto fail = [&](llama_seeddelta_fb_status st) {
        if (fb_status) *fb_status = st;
        return false;
    };
    if (!w_ref || !x) {
        return fail(llama_seeddelta_fb_status::no_w);
    }
    if (!x->data) {
        return fail(llama_seeddelta_fb_status::no_x);
    }
    {
        const ggml_tensor * w_base = w_ref->view_src ? w_ref->view_src : w_ref;
        if (w_base->buffer && !ggml_backend_buffer_is_host(w_base->buffer)) {
            return fail(llama_seeddelta_fb_status::non_host);
        }
    }
    {
        const ggml_tensor * x_base = x->view_src ? x->view_src : x;
        if (x_base->buffer && !ggml_backend_buffer_is_host(x_base->buffer)) {
            return fail(llama_seeddelta_fb_status::non_host);
        }
    }
    if (ggml_n_dims(x) < 2) {
        return fail(llama_seeddelta_fb_status::dim_mismatch);
    }
    if (ggml_n_dims(w_ref) != 2) {
        return fail(llama_seeddelta_fb_status::dim_mismatch);
    }

    const int64_t n_in = x->ne[0];
    if (w_ref->ne[0] != n_in) {
        return fail(llama_seeddelta_fb_status::n_in_mismatch);
    }
    if (o < 0 || o >= w_ref->ne[1]) {
        return fail(llama_seeddelta_fb_status::o_range);
    }
    if (t < 0 || t >= x->ne[1]) {
        return fail(llama_seeddelta_fb_status::t_range);
    }

    if (!llama_seeddelta_read_weight_col_f32(w_ref, o, llama_seeddelta_wcol_tls)) {
        return fail(llama_seeddelta_fb_status::read_fail);
    }

    const size_t row_bytes = (size_t) n_in * x->nb[0];
    const size_t x_offset = (size_t) t * x->nb[1];
    const uint8_t * x_data = nullptr;
    if (!llama_seeddelta_tensor_read(x, x_offset, row_bytes, llama_seeddelta_xbuf_tls, &x_data)) {
        return fail(llama_seeddelta_fb_status::no_data);
    }

    double acc = 0.0;
    for (int64_t i = 0; i < n_in; ++i) {
        const float xv = read_f16_or_f32_1d(x, x_data, i);
        acc += (double) llama_seeddelta_wcol_tls[(size_t) i] * (double) xv;
    }

    y_out = (float) acc;
    if (fb_status) *fb_status = llama_seeddelta_fb_status::ok;
    return std::isfinite(y_out);
}

static inline float llama_seeddelta_nan_guard(
        const ggml_tensor * w_ref,
        const ggml_tensor * x,
        int64_t o,
        int64_t t,
        float y) {
    if (std::isfinite(y)) {
        return y;
    }

    bool fallback_ok = false;
    float y_fb = 0.0f;
    if (llama_seeddelta_dense_fallback(w_ref, x, o, t, y_fb)) {
        y = y_fb;
        fallback_ok = true;
    } else {
        y = 0.0f;
    }

    llama_seeddelta_nan_count.fetch_add(1, std::memory_order_relaxed);
    if (!llama_seeddelta_nan_warned.test_and_set(std::memory_order_relaxed)) {
        const char * name = w_ref ? ggml_get_name(w_ref) : "<null>";
        std::fprintf(stderr,
                "llama-seeddelta: NaN/Inf detected in custom op output; %s for tensor '%s'\n",
                fallback_ok ? "falling back to dense" : "replacing with 0.0 (dense fallback unavailable)",
                name ? name : "<unnamed>");
    }

    return y;
}

static inline float llama_seeddelta_debug_compare(
        const ggml_tensor * w_ref,
        const ggml_tensor * x,
        int64_t o,
        int64_t t,
        float y) {
    const bool want_print = llama_seeddelta_debug_data;
    const bool want_stats = llama_seeddelta_debug_stats;
    if (!want_print && !want_stats) {
        return y;
    }
    bool do_stats = false;
    if (want_stats) {
        if (llama_seeddelta_debug_stats_accum_instance.should_sample(w_ref)) {
            const int64_t prev = llama_seeddelta_debug_stats_budget.fetch_sub(1, std::memory_order_relaxed);
            do_stats = prev > 0;
        }
    }
    if (!want_print && !do_stats) {
        return y;
    }
    int budget = llama_seeddelta_debug_budget.load(std::memory_order_relaxed);
    if (budget <= 0) {
        if (!do_stats) {
            return y;
        }
    }
    float y_dense = 0.0f;
    llama_seeddelta_fb_status fb = llama_seeddelta_fb_status::ok;
    if (llama_seeddelta_dense_fallback(w_ref, x, o, t, y_dense, &fb)) {
        const double diff = (double) y - (double) y_dense;
        if (do_stats) {
            llama_seeddelta_debug_stats_accum_instance.add(w_ref, diff);
        }
        if (want_print) {
            const int prev = llama_seeddelta_debug_budget.fetch_sub(1, std::memory_order_relaxed);
            if (prev > 0) {
                const char * name = w_ref ? ggml_get_name(w_ref) : nullptr;
                std::fprintf(stderr, "[seeddelta-debug-data] tensor=%s o=%" PRId64 " t=%" PRId64 " out=%g dense=%g diff=%g\n",
                             name ? name : "(unnamed)", o, t, y, y_dense, diff);
            }
        }
    } else {
        if (want_print) {
            const int prev = llama_seeddelta_debug_budget.fetch_sub(1, std::memory_order_relaxed);
            if (prev > 0) {
                const char * name = w_ref ? ggml_get_name(w_ref) : nullptr;
                const char * reason =
                    fb == llama_seeddelta_fb_status::no_w ? "no_w" :
                    fb == llama_seeddelta_fb_status::no_x ? "no_x" :
                    fb == llama_seeddelta_fb_status::no_data ? "no_data" :
                    fb == llama_seeddelta_fb_status::non_host ? "non_host" :
                    fb == llama_seeddelta_fb_status::dim_mismatch ? "dim_mismatch" :
                    fb == llama_seeddelta_fb_status::n_in_mismatch ? "n_in_mismatch" :
                    fb == llama_seeddelta_fb_status::o_range ? "o_range" :
                    fb == llama_seeddelta_fb_status::t_range ? "t_range" :
                    fb == llama_seeddelta_fb_status::read_fail ? "read_fail" :
                    "unknown";
                std::fprintf(stderr, "[seeddelta-debug-data] tensor=%s o=%" PRId64 " t=%" PRId64 " fallback-unavailable (%s)\n",
                             name ? name : "(unnamed)", o, t, reason);
            }
        }
    }
    return y;
}

static inline void hadamard_transform(float * data, int64_t n) {
#if defined(__AVX2__)
    // AVX2 path: vectorize the inner butterfly.
    // n is expected to be a power of two.
    for (int64_t len = 1; len < n; len <<= 1) {
        for (int64_t i = 0; i < n; i += (len << 1)) {
            int64_t j = 0;
            const int64_t vstep = 8;
            const int64_t j_end = len - (len % vstep);
            for (; j < j_end; j += vstep) {
                const __m256 u = _mm256_loadu_ps(data + i + j);
                const __m256 v = _mm256_loadu_ps(data + i + j + len);
                _mm256_storeu_ps(data + i + j,       _mm256_add_ps(u, v));
                _mm256_storeu_ps(data + i + j + len, _mm256_sub_ps(u, v));
            }
            for (; j < len; ++j) {
                const float u = data[i + j];
                const float v = data[i + j + len];
                data[i + j]       = u + v;
                data[i + j + len] = u - v;
            }
        }
    }
#else
    for (int64_t len = 1; len < n; len <<= 1) {
        for (int64_t i = 0; i < n; i += (len << 1)) {
            for (int64_t j = 0; j < len; ++j) {
                const float u = data[i + j];
                const float v = data[i + j + len];
                data[i + j]       = u + v;
                data[i + j + len] = u - v;
            }
        }
    }
#endif
}

static void apply_base_block(
        const float * x_in,
        float * y_out,
        float * tmp,
        int64_t L,
        int64_t b,
        const ggml_tensor * d1,
        const ggml_tensor * d2,
        const ggml_tensor * d3,
        const ggml_tensor * p1,
        const ggml_tensor * p2) {
    std::memcpy(y_out, x_in, (size_t) L * sizeof(float));

    const uint8_t * d1_data = d1 ? (const uint8_t *) d1->data : nullptr;
    const uint8_t * d2_data = d2 ? (const uint8_t *) d2->data : nullptr;
    const uint8_t * d3_data = d3 ? (const uint8_t *) d3->data : nullptr;
    const uint8_t * p1_data = p1 ? (const uint8_t *) p1->data : nullptr;
    const uint8_t * p2_data = p2 ? (const uint8_t *) p2->data : nullptr;

    // y = D1 * x
    for (int64_t i = 0; i < L; ++i) {
        y_out[i] *= read_base_f16_or_f32(d1, d1_data, i, b);
    }

    // y = P1 * y
    if (p1) {
        for (int64_t i = 0; i < L; ++i) {
            int32_t src = read_base_i16_i32(p1, p1_data, i, b);
            if (src < 0 || src >= (int32_t) L) {
                src = 0;
            }
            tmp[i] = y_out[src];
        }
        std::memcpy(y_out, tmp, (size_t) L * sizeof(float));
    }

    // y = H * y
    hadamard_transform(y_out, L);

    // If depth=1 (no second stage), apply D3 and return.
    if (!d2 && !p2) {
        if (d3) {
            for (int64_t i = 0; i < L; ++i) {
                y_out[i] *= read_base_f16_or_f32(d3, d3_data, i, b);
            }
        }
        return;
    }

    // y = D2 * y
    if (d2) {
        for (int64_t i = 0; i < L; ++i) {
            y_out[i] *= read_base_f16_or_f32(d2, d2_data, i, b);
        }
    }

    // y = P2 * y
    if (p2) {
        for (int64_t i = 0; i < L; ++i) {
            int32_t src = read_base_i16_i32(p2, p2_data, i, b);
            if (src < 0 || src >= (int32_t) L) {
                src = 0;
            }
            tmp[i] = y_out[src];
        }
        std::memcpy(y_out, tmp, (size_t) L * sizeof(float));
    }

    // y = H * y
    hadamard_transform(y_out, L);

    // y = D3 * y
    if (d3) {
        for (int64_t i = 0; i < L; ++i) {
            y_out[i] *= read_base_f16_or_f32(d3, d3_data, i, b);
        }
    }
}

// Compute y = Σ val[r,o] * x[idx[r,o], t] where:
//   x     : [n_in, n_tokens] (F16/F32)
//   d_idx : [K,    n_out]    (I16/I32)
//   d_val : [K,    n_out]    (F16/F32)
//   row_scale : [n_out]      optional scale per output row
//   y     : [n_out, n_tokens]
static void llama_seeddelta_coo_op(struct ggml_tensor * dst, int ith, int nth, void * userdata) {
    const ggml_tensor * w_ref = (const ggml_tensor *) userdata;

    const ggml_tensor * x        = dst->src[0];
    const ggml_tensor * d_idx    = dst->src[1];
    const ggml_tensor * d_val    = dst->src[2];
    const ggml_tensor * row_scale = dst->src[3];

    GGML_ASSERT(x->type == GGML_TYPE_F16 || x->type == GGML_TYPE_F32);
    GGML_ASSERT(d_idx->type == GGML_TYPE_I16 || d_idx->type == GGML_TYPE_I32);
    GGML_ASSERT(d_val->type == GGML_TYPE_F16 || d_val->type == GGML_TYPE_F32);

    const int64_t n_in     = x->ne[0];
    const int64_t n_tokens = x->ne[1];
    const int64_t K        = d_idx->ne[0];
    const int64_t n_out    = d_idx->ne[1];

    GGML_ASSERT(d_val->ne[0] == K);
    GGML_ASSERT(d_val->ne[1] == n_out);

    GGML_ASSERT(dst->ne[0] == n_out);
    GGML_ASSERT(dst->ne[1] == n_tokens);

    if (row_scale) {
        GGML_ASSERT(row_scale->type == GGML_TYPE_F16 || row_scale->type == GGML_TYPE_F32);
        GGML_ASSERT(row_scale->ne[0] == n_out);
    }

    const int64_t o0 = (n_out * ith) / nth;
    const int64_t o1 = (n_out * (ith + 1)) / nth;

    const bool fast_f32_contig =
            x->type == GGML_TYPE_F32 &&
            d_val->type == GGML_TYPE_F32 &&
            ggml_is_contiguous(x) &&
            ggml_is_contiguous(d_idx) &&
            ggml_is_contiguous(d_val) &&
            (!row_scale || ggml_is_contiguous(row_scale)) &&
            ggml_is_contiguous(dst);

    if (fast_f32_contig) {
        const auto * x_data  = (const float *) x->data;
        const auto * rs_f16  = row_scale && row_scale->type == GGML_TYPE_F16 ? (const ggml_fp16_t *) row_scale->data : nullptr;
        const auto * rs_f32  = row_scale && row_scale->type == GGML_TYPE_F32 ? (const float *)      row_scale->data : nullptr;
        auto * dst_data      = (float *) dst->data;

        if (d_idx->type == GGML_TYPE_I16) {
            const auto * idx_data = (const int16_t *) d_idx->data;
            const auto * val_data = (const float *)   d_val->data;

            for (int64_t o = o0; o < o1; ++o) {
                const float scale = row_scale ? (rs_f32 ? rs_f32[o] : ggml_fp16_to_fp32(rs_f16[o])) : 1.0f;
                const int16_t * idx_col = idx_data + o * K;
                const float *   val_col = val_data + o * K;
                for (int64_t t = 0; t < n_tokens; ++t) {
                    float y = 0.0f;
                    for (int64_t r = 0; r < K; ++r) {
                        const int32_t ii = (int32_t) idx_col[r];
                        if (ii < 0 || ii >= (int32_t) n_in) {
                            continue;
                        }
                        y += val_col[r] * x_data[ii + t * n_in];
                    }
                    float out = (scale != 1.0f) ? (y * scale) : y;
                    out = llama_seeddelta_nan_guard(w_ref, x, o, t, out);
                    out = llama_seeddelta_debug_compare(w_ref, x, o, t, out);
                    dst_data[o + t * n_out] = out;
                }
            }
        } else {
            const auto * idx_data = (const int32_t *) d_idx->data;
            const auto * val_data = (const float *)   d_val->data;

            for (int64_t o = o0; o < o1; ++o) {
                const float scale = row_scale ? (rs_f32 ? rs_f32[o] : ggml_fp16_to_fp32(rs_f16[o])) : 1.0f;
                const int32_t * idx_col = idx_data + o * K;
                const float *   val_col = val_data + o * K;
                for (int64_t t = 0; t < n_tokens; ++t) {
                    float y = 0.0f;
                    for (int64_t r = 0; r < K; ++r) {
                        const int32_t ii = idx_col[r];
                        if (ii < 0 || ii >= (int32_t) n_in) {
                            continue;
                        }
                        y += val_col[r] * x_data[ii + t * n_in];
                    }
                    float out = (scale != 1.0f) ? (y * scale) : y;
                    out = llama_seeddelta_nan_guard(w_ref, x, o, t, out);
                    out = llama_seeddelta_debug_compare(w_ref, x, o, t, out);
                    dst_data[o + t * n_out] = out;
                }
            }
        }

        return;
    }

    const uint8_t * x_data     = (const uint8_t *) x->data;
    const uint8_t * idx_data   = (const uint8_t *) d_idx->data;
    const uint8_t * val_data   = (const uint8_t *) d_val->data;
    const uint8_t * rs_data    = row_scale ? (const uint8_t *) row_scale->data : nullptr;
    uint8_t * dst_data         = (uint8_t *) dst->data;

    for (int64_t o = o0; o < o1; ++o) {
        const uint8_t * idx_col = idx_data + o * d_idx->nb[1];
        const uint8_t * val_col = val_data + o * d_val->nb[1];
        const float scale = row_scale ? read_f16_or_f32_1d(row_scale, rs_data, o) : 1.0f;
        for (int64_t t = 0; t < n_tokens; ++t) {
            float y = 0.0f;
            for (int64_t r = 0; r < K; ++r) {
                const int32_t ii = read_idx_i16_i32(d_idx, idx_col, r, 0);
                if (ii < 0 || ii >= (int32_t) n_in) {
                    continue;
                }
                const float vv = read_f16_or_f32(d_val, val_col, r, 0);
                const float xv = read_x_f16_or_f32(x, x_data, ii, t);
                y += vv * xv;
            }
            float out = (scale != 1.0f) ? (y * scale) : y;
            out = llama_seeddelta_nan_guard(w_ref, x, o, t, out);
            out = llama_seeddelta_debug_compare(w_ref, x, o, t, out);
            *(float *)(dst_data + o * dst->nb[0] + t * dst->nb[1]) = out;
        }
    }
}

// Compute y = Σ_{bi} dot(b_val[:,bi,o], x[blk*block:(blk+1)*block,t]) where:
//   x     : [n_in, n_tokens]         (F16/F32)
//   b_idx : [nb,   n_out]            (I16/I32) input block indices
//   b_val : [block, nb, n_out]       (F16/F32) residual block values
//   row_scale : [n_out]              optional scale per output row
//   y     : [n_out, n_tokens]
static void llama_seeddelta_block_op(struct ggml_tensor * dst, int ith, int nth, void * userdata) {
    const ggml_tensor * w_ref = (const ggml_tensor *) userdata;

    const ggml_tensor * x         = dst->src[0];
    const ggml_tensor * b_idx     = dst->src[1];
    const ggml_tensor * b_val     = dst->src[2];
    const ggml_tensor * row_scale = dst->src[3];

    GGML_ASSERT(x->type == GGML_TYPE_F16 || x->type == GGML_TYPE_F32);
    GGML_ASSERT(b_idx->type == GGML_TYPE_I16 || b_idx->type == GGML_TYPE_I32);
    GGML_ASSERT(b_val->type == GGML_TYPE_F16 || b_val->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_n_dims(b_idx) == 2);
    GGML_ASSERT(ggml_n_dims(b_val) == 3);

    const int64_t n_in     = x->ne[0];
    const int64_t n_tokens = x->ne[1];
    const int64_t nb       = b_idx->ne[0];
    const int64_t n_out    = b_idx->ne[1];
    const int64_t block    = b_val->ne[0];

    GGML_ASSERT(b_val->ne[1] == nb);
    GGML_ASSERT(b_val->ne[2] == n_out);

    GGML_ASSERT(dst->ne[0] == n_out);
    GGML_ASSERT(dst->ne[1] == n_tokens);

    if (row_scale) {
        GGML_ASSERT(row_scale->type == GGML_TYPE_F16 || row_scale->type == GGML_TYPE_F32);
        GGML_ASSERT(ggml_n_dims(row_scale) == 1);
        GGML_ASSERT(row_scale->ne[0] == n_out);
    }

    const int64_t o0 = (n_out * ith) / nth;
    const int64_t o1 = (n_out * (ith + 1)) / nth;

    const bool fast_f32_contig =
            x->type == GGML_TYPE_F32 &&
            b_val->type == GGML_TYPE_F32 &&
            ggml_is_contiguous(x) &&
            ggml_is_contiguous(b_idx) &&
            ggml_is_contiguous(b_val) &&
            (!row_scale || ggml_is_contiguous(row_scale)) &&
            ggml_is_contiguous(dst);

    if (fast_f32_contig) {
        const auto * x_data = (const float *) x->data;
        const auto * rs_f16 = row_scale && row_scale->type == GGML_TYPE_F16 ? (const ggml_fp16_t *) row_scale->data : nullptr;
        const auto * rs_f32 = row_scale && row_scale->type == GGML_TYPE_F32 ? (const float *)      row_scale->data : nullptr;
        auto * dst_data     = (float *) dst->data;

        if (b_idx->type == GGML_TYPE_I16) {
            const auto * idx_data = (const int16_t *) b_idx->data;
            const auto * val_data = (const float *)   b_val->data;
            for (int64_t o = o0; o < o1; ++o) {
                const float scale = row_scale ? (rs_f32 ? rs_f32[o] : ggml_fp16_to_fp32(rs_f16[o])) : 1.0f;
                const int16_t * idx_col = idx_data + o * nb;
                const float *   val_col = val_data + o * nb * block;
                for (int64_t t = 0; t < n_tokens; ++t) {
                    const float * x_col = x_data + t * n_in;
                    float y = 0.0f;
                    for (int64_t bi = 0; bi < nb; ++bi) {
                        const int32_t blk = (int32_t) idx_col[bi];
                        if (blk < 0) {
                            continue;
                        }
                        const int64_t in0 = (int64_t) blk * block;
                        if (in0 < 0 || in0 >= n_in) {
                            continue;
                        }
                        const int64_t len = std::min<int64_t>(block, n_in - in0);
                        const float * xv = x_col + in0;
                        const float * vv = val_col + bi * block;
                        for (int64_t j = 0; j < len; ++j) {
                            y += vv[j] * xv[j];
                        }
                    }
                    float out = (scale != 1.0f) ? (y * scale) : y;
                    out = llama_seeddelta_nan_guard(w_ref, x, o, t, out);
                    out = llama_seeddelta_debug_compare(w_ref, x, o, t, out);
                    dst_data[o + t * n_out] = out;
                }
            }
        } else {
            const auto * idx_data = (const int32_t *) b_idx->data;
            const auto * val_data = (const float *)   b_val->data;
            for (int64_t o = o0; o < o1; ++o) {
                const float scale = row_scale ? (rs_f32 ? rs_f32[o] : ggml_fp16_to_fp32(rs_f16[o])) : 1.0f;
                const int32_t * idx_col = idx_data + o * nb;
                const float *   val_col = val_data + o * nb * block;
                for (int64_t t = 0; t < n_tokens; ++t) {
                    const float * x_col = x_data + t * n_in;
                    float y = 0.0f;
                    for (int64_t bi = 0; bi < nb; ++bi) {
                        const int32_t blk = idx_col[bi];
                        if (blk < 0) {
                            continue;
                        }
                        const int64_t in0 = (int64_t) blk * block;
                        if (in0 < 0 || in0 >= n_in) {
                            continue;
                        }
                        const int64_t len = std::min<int64_t>(block, n_in - in0);
                        const float * xv = x_col + in0;
                        const float * vv = val_col + bi * block;
                        for (int64_t j = 0; j < len; ++j) {
                            y += vv[j] * xv[j];
                        }
                    }
                    float out = (scale != 1.0f) ? (y * scale) : y;
                    out = llama_seeddelta_nan_guard(w_ref, x, o, t, out);
                    out = llama_seeddelta_debug_compare(w_ref, x, o, t, out);
                    dst_data[o + t * n_out] = out;
                }
            }
        }

        return;
    }

    const bool fast_f16_contig =
            x->type == GGML_TYPE_F16 &&
            b_val->type == GGML_TYPE_F16 &&
            ggml_is_contiguous(x) &&
            ggml_is_contiguous(b_idx) &&
            ggml_is_contiguous(b_val) &&
            (!row_scale || ggml_is_contiguous(row_scale)) &&
            ggml_is_contiguous(dst);

    if (fast_f16_contig) {
        const auto * x_data   = (const ggml_fp16_t *) x->data;
        const auto * rs_f16   = row_scale && row_scale->type == GGML_TYPE_F16 ? (const ggml_fp16_t *) row_scale->data : nullptr;
        const auto * rs_f32   = row_scale && row_scale->type == GGML_TYPE_F32 ? (const float *)      row_scale->data : nullptr;
        const auto * val_data = (const ggml_fp16_t *) b_val->data;
        auto * dst_data       = (float *) dst->data;

        if (b_idx->type == GGML_TYPE_I16) {
            const auto * idx_data = (const int16_t *) b_idx->data;
            for (int64_t o = o0; o < o1; ++o) {
                const float scale = row_scale ? (rs_f32 ? rs_f32[o] : ggml_fp16_to_fp32(rs_f16[o])) : 1.0f;
                const int16_t * idx_col = idx_data + o * nb;
                const ggml_fp16_t * val_col = val_data + o * nb * block;
                for (int64_t t = 0; t < n_tokens; ++t) {
                    const ggml_fp16_t * x_col = x_data + t * n_in;
                    float y = 0.0f;
                    for (int64_t bi = 0; bi < nb; ++bi) {
                        const int32_t blk = (int32_t) idx_col[bi];
                        if (blk < 0) {
                            continue;
                        }
                        const int64_t in0 = (int64_t) blk * block;
                        if (in0 < 0 || in0 >= n_in) {
                            continue;
                        }
                        const int64_t len = std::min<int64_t>(block, n_in - in0);
                        const ggml_fp16_t * xv = x_col + in0;
                        const ggml_fp16_t * vv = val_col + bi * block;
                        for (int64_t j = 0; j < len; ++j) {
                            y += ggml_fp16_to_fp32(vv[j]) * ggml_fp16_to_fp32(xv[j]);
                        }
                    }
                    float out = (scale != 1.0f) ? (y * scale) : y;
                    out = llama_seeddelta_nan_guard(w_ref, x, o, t, out);
                    dst_data[o + t * n_out] = out;
                }
            }
        } else {
            const auto * idx_data = (const int32_t *) b_idx->data;
            for (int64_t o = o0; o < o1; ++o) {
                const float scale = row_scale ? (rs_f32 ? rs_f32[o] : ggml_fp16_to_fp32(rs_f16[o])) : 1.0f;
                const int32_t * idx_col = idx_data + o * nb;
                const ggml_fp16_t * val_col = val_data + o * nb * block;
                for (int64_t t = 0; t < n_tokens; ++t) {
                    const ggml_fp16_t * x_col = x_data + t * n_in;
                    float y = 0.0f;
                    for (int64_t bi = 0; bi < nb; ++bi) {
                        const int32_t blk = idx_col[bi];
                        if (blk < 0) {
                            continue;
                        }
                        const int64_t in0 = (int64_t) blk * block;
                        if (in0 < 0 || in0 >= n_in) {
                            continue;
                        }
                        const int64_t len = std::min<int64_t>(block, n_in - in0);
                        const ggml_fp16_t * xv = x_col + in0;
                        const ggml_fp16_t * vv = val_col + bi * block;
                        for (int64_t j = 0; j < len; ++j) {
                            y += ggml_fp16_to_fp32(vv[j]) * ggml_fp16_to_fp32(xv[j]);
                        }
                    }
                    float out = (scale != 1.0f) ? (y * scale) : y;
                    out = llama_seeddelta_nan_guard(w_ref, x, o, t, out);
                    dst_data[o + t * n_out] = out;
                }
            }
        }

        return;
    }

    const uint8_t * x_data   = (const uint8_t *) x->data;
    const uint8_t * idx_data = (const uint8_t *) b_idx->data;
    const uint8_t * val_data = (const uint8_t *) b_val->data;
    const uint8_t * rs_data  = row_scale ? (const uint8_t *) row_scale->data : nullptr;
    uint8_t * dst_data       = (uint8_t *) dst->data;

    for (int64_t o = o0; o < o1; ++o) {
        const uint8_t * idx_col = idx_data + o * b_idx->nb[1];
        const uint8_t * val_col = val_data + o * b_val->nb[2];
        const float scale = row_scale ? read_f16_or_f32_1d(row_scale, rs_data, o) : 1.0f;
        for (int64_t t = 0; t < n_tokens; ++t) {
            float y = 0.0f;
            for (int64_t bi = 0; bi < nb; ++bi) {
                const int32_t blk = read_idx_i16_i32(b_idx, idx_col, bi, 0);
                if (blk < 0) {
                    continue;
                }
                const int64_t in0 = (int64_t) blk * block;
                if (in0 < 0 || in0 >= n_in) {
                    continue;
                }
                const int64_t in1 = std::min<int64_t>(n_in, in0 + block);
                const uint8_t * val_blk = val_col + bi * b_val->nb[1];
                for (int64_t ii = in0; ii < in1; ++ii) {
                    const int64_t j = ii - in0;
                    const float vv = read_f16_or_f32_1d(b_val, val_blk, j);
                    const float xv = read_x_f16_or_f32(x, x_data, ii, t);
                    y += vv * xv;
                }
            }
            float out = (scale != 1.0f) ? (y * scale) : y;
            out = llama_seeddelta_nan_guard(w_ref, x, o, t, out);
            out = llama_seeddelta_debug_compare(w_ref, x, o, t, out);
            *(float *)(dst_data + o * dst->nb[0] + t * dst->nb[1]) = out;
        }
    }
}

ggml_tensor * llama_seeddelta_mul_mat(
        ggml_context * ctx,
        ggml_tensor  * x,
        ggml_tensor  * w_ref,
        ggml_tensor  * d_idx,
        ggml_tensor  * d_val,
        ggml_tensor  * row_scale) {
    GGML_ASSERT(d_idx != nullptr && d_val != nullptr);

    const int64_t n_out    = d_idx->ne[1];
    const int64_t n_tokens = x->ne[1];

    ggml_tensor * args[4] = { x, d_idx, d_val, row_scale };

    ggml_tensor * res = ggml_custom_4d(
            ctx, GGML_TYPE_F32,
            n_out, n_tokens, 1, 1,
            args, 4,
            llama_seeddelta_coo_op,
            GGML_N_TASKS_MAX,
            w_ref);

    return res;
}

ggml_tensor * llama_seeddelta_mul_mat_block(
        ggml_context * ctx,
        ggml_tensor  * x,
        ggml_tensor  * w_ref,
        ggml_tensor  * b_idx,
        ggml_tensor  * b_val,
        ggml_tensor  * row_scale) {
    GGML_ASSERT(b_idx != nullptr && b_val != nullptr);

    const int64_t n_out    = b_idx->ne[1];
    const int64_t n_tokens = x->ne[1];

    ggml_tensor * args[4] = { x, b_idx, b_val, row_scale };

    ggml_tensor * res = ggml_custom_4d(
            ctx, GGML_TYPE_F32,
            n_out, n_tokens, 1, 1,
            args, 4,
            llama_seeddelta_block_op,
            GGML_N_TASKS_MAX,
            w_ref);

    return res;
}

struct llama_seeddelta_scratch {
    std::vector<float> x_hat;
    std::vector<float> x_chunk;
    std::vector<float> x_full;
    std::vector<float> v;
    std::vector<float> tmp;
    std::vector<float> y_hat;
};

static thread_local llama_seeddelta_scratch llama_seeddelta_scratch_tls;

static void llama_seeddelta_base_op(struct ggml_tensor * dst, int ith, int nth, void * userdata) {
    const ggml_tensor * w_ref = (const ggml_tensor *) userdata;

    const ggml_tensor * x          = dst->src[0];
    const ggml_tensor * base_d1    = dst->src[1];
    const ggml_tensor * base_d2    = dst->src[2];
    const ggml_tensor * base_d3    = dst->src[3];
    const ggml_tensor * base_perm1 = dst->src[4];
    const ggml_tensor * base_perm2 = dst->src[5];
    const ggml_tensor * d_idx      = dst->src[6];
    const ggml_tensor * d_val      = dst->src[7];
    const ggml_tensor * row_scale  = dst->src[8];

    GGML_ASSERT(x && base_d1 && d_idx && d_val);
    GGML_ASSERT(x->type == GGML_TYPE_F16 || x->type == GGML_TYPE_F32);
    GGML_ASSERT(base_d1->type == GGML_TYPE_F16 || base_d1->type == GGML_TYPE_F32);
    GGML_ASSERT(!base_d2 || base_d2->type == GGML_TYPE_F16 || base_d2->type == GGML_TYPE_F32);
    GGML_ASSERT(!base_d3 || base_d3->type == GGML_TYPE_F16 || base_d3->type == GGML_TYPE_F32);
    GGML_ASSERT(!base_perm1 || base_perm1->type == GGML_TYPE_I16 || base_perm1->type == GGML_TYPE_I32);
    GGML_ASSERT(!base_perm2 || base_perm2->type == GGML_TYPE_I16 || base_perm2->type == GGML_TYPE_I32);

    GGML_ASSERT(d_idx->type == GGML_TYPE_I16 || d_idx->type == GGML_TYPE_I32);
    GGML_ASSERT(d_val->type == GGML_TYPE_F16 || d_val->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_n_dims(base_d1) == 2);
    GGML_ASSERT(!base_d2    || (ggml_n_dims(base_d2)    == 2 && base_d2->ne[0] == base_d1->ne[0] && base_d2->ne[1] == base_d1->ne[1]));
    GGML_ASSERT(!base_d3    || (ggml_n_dims(base_d3)    == 2 && base_d3->ne[0] == base_d1->ne[0] && base_d3->ne[1] == base_d1->ne[1]));
    GGML_ASSERT(!base_perm1 || (ggml_n_dims(base_perm1) == 2 && base_perm1->ne[0] == base_d1->ne[0] && base_perm1->ne[1] == base_d1->ne[1]));
    GGML_ASSERT(!base_perm2 || (ggml_n_dims(base_perm2) == 2 && base_perm2->ne[0] == base_d1->ne[0] && base_perm2->ne[1] == base_d1->ne[1]));

    const int64_t n_in     = x->ne[0];
    const int64_t n_tokens = x->ne[1];
    const int64_t K        = d_idx->ne[0];
    const int64_t n_out    = d_idx->ne[1];

    GGML_ASSERT(d_val->ne[0] == K && d_val->ne[1] == n_out);
    GGML_ASSERT(dst->ne[0] == n_out && dst->ne[1] == n_tokens);

    if (row_scale) {
        GGML_ASSERT(row_scale->type == GGML_TYPE_F16 || row_scale->type == GGML_TYPE_F32);
        GGML_ASSERT(row_scale->ne[0] == n_out);
    }

    const int64_t L = base_d1->ne[0];
    const int64_t B = base_d1->ne[1];

    const int64_t o0 = (n_out * ith) / nth;
    const int64_t o1 = (n_out * (ith + 1)) / nth;

    auto & x_hat   = llama_seeddelta_scratch_tls.x_hat;
    auto & x_chunk = llama_seeddelta_scratch_tls.x_chunk;
    auto & x_full  = llama_seeddelta_scratch_tls.x_full;
    auto & v       = llama_seeddelta_scratch_tls.v;
    auto & tmp     = llama_seeddelta_scratch_tls.tmp;
    auto & y_hat   = llama_seeddelta_scratch_tls.y_hat;

    v.resize((size_t) L);
    tmp.resize((size_t) L);

    const bool is_tall = n_out >= n_in;
    if (is_tall) {
        x_hat.resize((size_t) L);
    } else {
        x_chunk.resize((size_t) L);
        x_full.resize((size_t) n_in);
        y_hat.resize((size_t) L);
    }

    const uint8_t * x_data     = (const uint8_t *) x->data;
    const uint8_t * idx_data   = (const uint8_t *) d_idx->data;
    const uint8_t * val_data   = (const uint8_t *) d_val->data;
    const uint8_t * rs_data    = row_scale ? (const uint8_t *) row_scale->data : nullptr;
    uint8_t * dst_data         = (uint8_t *) dst->data;

    if (is_tall) {
        for (int64_t t = 0; t < n_tokens; ++t) {
            // pad x -> x_hat (L)
            for (int64_t i = 0; i < n_in; ++i) {
                x_hat[(size_t) i] = read_x_f16_or_f32(x, x_data, i, t);
            }
            for (int64_t i = n_in; i < L; ++i) {
                x_hat[(size_t) i] = 0.0f;
            }

            const int64_t b0 = o0 / L;
            const int64_t b1 = (o1 - 1) / L;
            for (int64_t b = b0; b <= b1 && b < B; ++b) {
                apply_base_block(x_hat.data(), v.data(), tmp.data(), L, b, base_d1, base_d2, base_d3, base_perm1, base_perm2);

                const int64_t bo0 = std::max<int64_t>(o0, b * L);
                const int64_t bo1 = std::min<int64_t>(o1, (b + 1) * L);
                for (int64_t o = bo0; o < bo1; ++o) {
                    float y = v[(size_t) (o - b * L)];

                    const uint8_t * idx_col = idx_data + o * d_idx->nb[1];
                    const uint8_t * val_col = val_data + o * d_val->nb[1];
                    for (int64_t r = 0; r < K; ++r) {
                        const int32_t ii = read_idx_i16_i32(d_idx, idx_col, r, 0);
                        if (ii < 0 || ii >= (int32_t) n_in) {
                            continue;
                        }
                        const float vv = read_f16_or_f32(d_val, val_col, r, 0);
                        const float xv = x_hat[(size_t) ii];
                        y += vv * xv;
                    }

                    const float scale = row_scale ? read_f16_or_f32_1d(row_scale, rs_data, o) : 1.0f;
                    float out = (scale != 1.0f) ? (y * scale) : y;
                    out = llama_seeddelta_nan_guard(w_ref, x, o, t, out);
                    out = llama_seeddelta_debug_compare(w_ref, x, o, t, out);
                    *(float *)(dst_data + o * dst->nb[0] + t * dst->nb[1]) = out;
                }
            }
        }
    } else {
        const int64_t t0 = (n_tokens * ith) / nth;
        const int64_t t1 = (n_tokens * (ith + 1)) / nth;
        for (int64_t t = t0; t < t1; ++t) {
            std::fill(y_hat.begin(), y_hat.end(), 0.0f);

            // y_hat = Σ_b F_b(x_chunk_b)
            for (int64_t b = 0; b < B; ++b) {
                const int64_t in0 = b * L;
                const int64_t in1 = std::min<int64_t>(n_in, in0 + L);
                for (int64_t i = 0; i < in1 - in0; ++i) {
                    const float xv = read_x_f16_or_f32(x, x_data, in0 + i, t);
                    x_chunk[(size_t) i] = xv;
                    x_full[(size_t) (in0 + i)] = xv;
                }
                for (int64_t i = in1 - in0; i < L; ++i) {
                    x_chunk[(size_t) i] = 0.0f;
                }

                apply_base_block(x_chunk.data(), v.data(), tmp.data(), L, b, base_d1, base_d2, base_d3, base_perm1, base_perm2);
                for (int64_t i = 0; i < L; ++i) {
                    y_hat[(size_t) i] += v[(size_t) i];
                }
            }

            for (int64_t o = 0; o < n_out; ++o) {
                float y = y_hat[(size_t) o];

                const uint8_t * idx_col = idx_data + o * d_idx->nb[1];
                const uint8_t * val_col = val_data + o * d_val->nb[1];
                for (int64_t r = 0; r < K; ++r) {
                    const int32_t ii = read_idx_i16_i32(d_idx, idx_col, r, 0);
                    if (ii < 0 || ii >= (int32_t) n_in) {
                        continue;
                    }
                    const float vv = read_f16_or_f32(d_val, val_col, r, 0);
                    const float xv = x_full[(size_t) ii];
                    y += vv * xv;
                }

                const float scale = row_scale ? read_f16_or_f32_1d(row_scale, rs_data, o) : 1.0f;
                float out = (scale != 1.0f) ? (y * scale) : y;
                out = llama_seeddelta_nan_guard(w_ref, x, o, t, out);
                out = llama_seeddelta_debug_compare(w_ref, x, o, t, out);
                *(float *)(dst_data + o * dst->nb[0] + t * dst->nb[1]) = out;
            }
        }
    }
}

static void llama_seeddelta_base_block_op(struct ggml_tensor * dst, int ith, int nth, void * userdata) {
    const ggml_tensor * w_ref = (const ggml_tensor *) userdata;

    const ggml_tensor * x          = dst->src[0];
    const ggml_tensor * base_d1    = dst->src[1];
    const ggml_tensor * base_d2    = dst->src[2];
    const ggml_tensor * base_d3    = dst->src[3];
    const ggml_tensor * base_perm1 = dst->src[4];
    const ggml_tensor * base_perm2 = dst->src[5];
    const ggml_tensor * b_idx      = dst->src[6];
    const ggml_tensor * b_val      = dst->src[7];
    const ggml_tensor * row_scale  = dst->src[8];

    GGML_ASSERT(x && base_d1 && b_idx && b_val);
    GGML_ASSERT(x->type == GGML_TYPE_F16 || x->type == GGML_TYPE_F32);
    GGML_ASSERT(base_d1->type == GGML_TYPE_F16 || base_d1->type == GGML_TYPE_F32);
    GGML_ASSERT(!base_d2 || base_d2->type == GGML_TYPE_F16 || base_d2->type == GGML_TYPE_F32);
    GGML_ASSERT(!base_d3 || base_d3->type == GGML_TYPE_F16 || base_d3->type == GGML_TYPE_F32);
    GGML_ASSERT(!base_perm1 || base_perm1->type == GGML_TYPE_I16 || base_perm1->type == GGML_TYPE_I32);
    GGML_ASSERT(!base_perm2 || base_perm2->type == GGML_TYPE_I16 || base_perm2->type == GGML_TYPE_I32);

    GGML_ASSERT(b_idx->type == GGML_TYPE_I16 || b_idx->type == GGML_TYPE_I32);
    GGML_ASSERT(b_val->type == GGML_TYPE_F16 || b_val->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_n_dims(base_d1) == 2);
    GGML_ASSERT(!base_d2    || (ggml_n_dims(base_d2)    == 2 && base_d2->ne[0] == base_d1->ne[0] && base_d2->ne[1] == base_d1->ne[1]));
    GGML_ASSERT(!base_d3    || (ggml_n_dims(base_d3)    == 2 && base_d3->ne[0] == base_d1->ne[0] && base_d3->ne[1] == base_d1->ne[1]));
    GGML_ASSERT(!base_perm1 || (ggml_n_dims(base_perm1) == 2 && base_perm1->ne[0] == base_d1->ne[0] && base_perm1->ne[1] == base_d1->ne[1]));
    GGML_ASSERT(!base_perm2 || (ggml_n_dims(base_perm2) == 2 && base_perm2->ne[0] == base_d1->ne[0] && base_perm2->ne[1] == base_d1->ne[1]));

    GGML_ASSERT(ggml_n_dims(b_idx) == 2);
    GGML_ASSERT(ggml_n_dims(b_val) == 3);

    const int64_t n_in     = x->ne[0];
    const int64_t n_tokens = x->ne[1];
    const int64_t nb       = b_idx->ne[0];
    const int64_t n_out    = b_idx->ne[1];
    const int64_t block    = b_val->ne[0];

    GGML_ASSERT(b_val->ne[1] == nb && b_val->ne[2] == n_out);
    GGML_ASSERT(dst->ne[0] == n_out && dst->ne[1] == n_tokens);

    if (row_scale) {
        GGML_ASSERT(row_scale->type == GGML_TYPE_F16 || row_scale->type == GGML_TYPE_F32);
        GGML_ASSERT(row_scale->ne[0] == n_out);
    }

    const int64_t L = base_d1->ne[0];
    const int64_t B = base_d1->ne[1];

    const int64_t o0 = (n_out * ith) / nth;
    const int64_t o1 = (n_out * (ith + 1)) / nth;

    auto & x_hat   = llama_seeddelta_scratch_tls.x_hat;
    auto & x_chunk = llama_seeddelta_scratch_tls.x_chunk;
    auto & x_full  = llama_seeddelta_scratch_tls.x_full;
    auto & v       = llama_seeddelta_scratch_tls.v;
    auto & tmp     = llama_seeddelta_scratch_tls.tmp;
    auto & y_hat   = llama_seeddelta_scratch_tls.y_hat;

    v.resize((size_t) L);
    tmp.resize((size_t) L);

    const bool is_tall = n_out >= n_in;
    if (is_tall) {
        x_hat.resize((size_t) L);
    } else {
        x_chunk.resize((size_t) L);
        x_full.resize((size_t) n_in);
        y_hat.resize((size_t) L);
    }

    const uint8_t * x_data   = (const uint8_t *) x->data;
    const uint8_t * idx_data = (const uint8_t *) b_idx->data;
    const uint8_t * val_data = (const uint8_t *) b_val->data;
    const uint8_t * rs_data  = row_scale ? (const uint8_t *) row_scale->data : nullptr;
    uint8_t * dst_data       = (uint8_t *) dst->data;

    if (is_tall) {
        for (int64_t t = 0; t < n_tokens; ++t) {
            // pad x -> x_hat (L)
            for (int64_t i = 0; i < n_in; ++i) {
                x_hat[(size_t) i] = read_x_f16_or_f32(x, x_data, i, t);
            }
            for (int64_t i = n_in; i < L; ++i) {
                x_hat[(size_t) i] = 0.0f;
            }

            const int64_t b0 = o0 / L;
            const int64_t b1 = (o1 - 1) / L;
            for (int64_t b = b0; b <= b1 && b < B; ++b) {
                apply_base_block(x_hat.data(), v.data(), tmp.data(), L, b, base_d1, base_d2, base_d3, base_perm1, base_perm2);

                const int64_t bo0 = std::max<int64_t>(o0, b * L);
                const int64_t bo1 = std::min<int64_t>(o1, (b + 1) * L);
                for (int64_t o = bo0; o < bo1; ++o) {
                    float y = v[(size_t) (o - b * L)];

                    const uint8_t * idx_col = idx_data + o * b_idx->nb[1];
                    const uint8_t * val_col = val_data + o * b_val->nb[2];
                    for (int64_t bi = 0; bi < nb; ++bi) {
                        const int32_t blk = read_idx_i16_i32(b_idx, idx_col, bi, 0);
                        if (blk < 0) {
                            continue;
                        }
                        const int64_t in0 = (int64_t) blk * block;
                        if (in0 < 0 || in0 >= n_in) {
                            continue;
                        }
                        const int64_t in1 = std::min<int64_t>(n_in, in0 + block);
                        const int64_t len = in1 - in0;
                        const uint8_t * val_blk = val_col + bi * b_val->nb[1];
                        const float * x_blk = x_hat.data() + in0;
                        for (int64_t j = 0; j < len; ++j) {
                            y += read_f16_or_f32_1d(b_val, val_blk, j) * x_blk[j];
                        }
                    }

                    const float scale = row_scale ? read_f16_or_f32_1d(row_scale, rs_data, o) : 1.0f;
                    float out = (scale != 1.0f) ? (y * scale) : y;
                    out = llama_seeddelta_nan_guard(w_ref, x, o, t, out);
                    out = llama_seeddelta_debug_compare(w_ref, x, o, t, out);
                    *(float *)(dst_data + o * dst->nb[0] + t * dst->nb[1]) = out;
                }
            }
        }
    } else {
        const int64_t t0 = (n_tokens * ith) / nth;
        const int64_t t1 = (n_tokens * (ith + 1)) / nth;
        for (int64_t t = t0; t < t1; ++t) {
            std::fill(y_hat.begin(), y_hat.end(), 0.0f);

            // y_hat = Σ_b F_b(x_chunk_b)
            for (int64_t b = 0; b < B; ++b) {
                const int64_t in0 = b * L;
                const int64_t in1 = std::min<int64_t>(n_in, in0 + L);
                for (int64_t i = 0; i < in1 - in0; ++i) {
                    const float xv = read_x_f16_or_f32(x, x_data, in0 + i, t);
                    x_chunk[(size_t) i] = xv;
                    x_full[(size_t) (in0 + i)] = xv;
                }
                for (int64_t i = in1 - in0; i < L; ++i) {
                    x_chunk[(size_t) i] = 0.0f;
                }

                apply_base_block(x_chunk.data(), v.data(), tmp.data(), L, b, base_d1, base_d2, base_d3, base_perm1, base_perm2);
                for (int64_t i = 0; i < L; ++i) {
                    y_hat[(size_t) i] += v[(size_t) i];
                }
            }

            for (int64_t o = 0; o < n_out; ++o) {
                float y = y_hat[(size_t) o];

                const uint8_t * idx_col = idx_data + o * b_idx->nb[1];
                const uint8_t * val_col = val_data + o * b_val->nb[2];
                for (int64_t bi = 0; bi < nb; ++bi) {
                    const int32_t blk = read_idx_i16_i32(b_idx, idx_col, bi, 0);
                    if (blk < 0) {
                        continue;
                    }
                    const int64_t in0 = (int64_t) blk * block;
                    if (in0 < 0 || in0 >= n_in) {
                        continue;
                    }
                    const int64_t in1 = std::min<int64_t>(n_in, in0 + block);
                    const int64_t len = in1 - in0;
                    const uint8_t * val_blk = val_col + bi * b_val->nb[1];
                    const float * x_blk = x_full.data() + in0;
                    for (int64_t j = 0; j < len; ++j) {
                        y += read_f16_or_f32_1d(b_val, val_blk, j) * x_blk[j];
                    }
                }

                const float scale = row_scale ? read_f16_or_f32_1d(row_scale, rs_data, o) : 1.0f;
                float out = (scale != 1.0f) ? (y * scale) : y;
                out = llama_seeddelta_nan_guard(w_ref, x, o, t, out);
                out = llama_seeddelta_debug_compare(w_ref, x, o, t, out);
                *(float *)(dst_data + o * dst->nb[0] + t * dst->nb[1]) = out;
            }
        }
    }
}

ggml_tensor * llama_seeddelta_mul_mat_base(
        ggml_context * ctx,
        ggml_tensor  * x,
        ggml_tensor  * w_ref,
        ggml_tensor  * base_d1,
        ggml_tensor  * base_d2,
        ggml_tensor  * base_d3,
        ggml_tensor  * base_perm1,
        ggml_tensor  * base_perm2,
        ggml_tensor  * d_idx,
        ggml_tensor  * d_val,
        ggml_tensor  * row_scale) {
    GGML_ASSERT(base_d1 != nullptr);
    GGML_ASSERT(d_idx  != nullptr && d_val != nullptr);

    const int64_t n_out    = d_idx->ne[1];
    const int64_t n_tokens = x->ne[1];

    ggml_tensor * args[9] = {
        x, base_d1, base_d2, base_d3, base_perm1, base_perm2, d_idx, d_val, row_scale
    };

    ggml_tensor * res = ggml_custom_4d(
            ctx, GGML_TYPE_F32,
            n_out, n_tokens, 1, 1,
            args, 9,
            llama_seeddelta_base_op,
            GGML_N_TASKS_MAX,
            w_ref);

    return res;
}

ggml_tensor * llama_seeddelta_mul_mat_base_block(
        ggml_context * ctx,
        ggml_tensor  * x,
        ggml_tensor  * w_ref,
        ggml_tensor  * base_d1,
        ggml_tensor  * base_d2,
        ggml_tensor  * base_d3,
        ggml_tensor  * base_perm1,
        ggml_tensor  * base_perm2,
        ggml_tensor  * b_idx,
        ggml_tensor  * b_val,
        ggml_tensor  * row_scale) {
    GGML_ASSERT(base_d1 != nullptr);
    GGML_ASSERT(b_idx  != nullptr && b_val != nullptr);

    const int64_t n_out    = b_idx->ne[1];
    const int64_t n_tokens = x->ne[1];

    ggml_tensor * args[9] = {
        x, base_d1, base_d2, base_d3, base_perm1, base_perm2, b_idx, b_val, row_scale
    };

    ggml_tensor * res = ggml_custom_4d(
            ctx, GGML_TYPE_F32,
            n_out, n_tokens, 1, 1,
            args, 9,
            llama_seeddelta_base_block_op,
            GGML_N_TASKS_MAX,
            w_ref);

    return res;
}
