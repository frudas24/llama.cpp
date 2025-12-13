#include "llama-seeddelta.h"

#include "ggml.h"

#include <cmath>
#include <cstdlib>

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

// Compute y = Σ val[r,o] * x[idx[r,o], t] where:
//   x     : [n_in, n_tokens] (F16/F32)
//   d_idx : [K,    n_out]    (I16/I32)
//   d_val : [K,    n_out]    (F16/F32)
//   row_scale : [n_out]      optional scale per output row
//   y     : [n_out, n_tokens]
static void llama_seeddelta_coo_op(struct ggml_tensor * dst, int ith, int nth, void * userdata) {
    GGML_UNUSED(userdata);

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
                    dst_data[o + t * n_out] = (scale != 1.0f) ? (y * scale) : y;
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
                    dst_data[o + t * n_out] = (scale != 1.0f) ? (y * scale) : y;
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
            *(float *)(dst_data + o * dst->nb[0] + t * dst->nb[1]) = (scale != 1.0f) ? (y * scale) : y;
        }
    }
}

ggml_tensor * llama_seeddelta_mul_mat(
        ggml_context * ctx,
        ggml_tensor  * x,
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
            nullptr);

    return res;
}

