#include "llama-seeddelta.h"

#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

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

static inline void hadamard_transform(float * data, int64_t n) {
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

// Compute y = Σ_{bi} dot(b_val[:,bi,o], x[blk*block:(blk+1)*block,t]) where:
//   x     : [n_in, n_tokens]         (F16/F32)
//   b_idx : [nb,   n_out]            (I16/I32) input block indices
//   b_val : [block, nb, n_out]       (F16/F32) residual block values
//   row_scale : [n_out]              optional scale per output row
//   y     : [n_out, n_tokens]
static void llama_seeddelta_block_op(struct ggml_tensor * dst, int ith, int nth, void * userdata) {
    GGML_UNUSED(userdata);

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
                    dst_data[o + t * n_out] = (scale != 1.0f) ? (y * scale) : y;
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
                    dst_data[o + t * n_out] = (scale != 1.0f) ? (y * scale) : y;
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
                    const float vv = read_f16_or_f32(b_val, val_blk, j, 0);
                    const float xv = read_x_f16_or_f32(x, x_data, ii, t);
                    y += vv * xv;
                }
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

ggml_tensor * llama_seeddelta_mul_mat_block(
        ggml_context * ctx,
        ggml_tensor  * x,
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
            nullptr);

    return res;
}

static void llama_seeddelta_base_op(struct ggml_tensor * dst, int ith, int nth, void * userdata) {
    GGML_UNUSED(userdata);

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

    std::vector<float> x_hat;
    std::vector<float> x_chunk;
    std::vector<float> v;
    std::vector<float> tmp;
    std::vector<float> y_hat;

    v.resize((size_t) L);
    tmp.resize((size_t) L);

    const bool is_tall = n_out >= n_in;
    if (is_tall) {
        x_hat.resize((size_t) L);
    } else {
        x_chunk.resize((size_t) L);
        y_hat.resize((size_t) L);
    }

    const uint8_t * x_data     = (const uint8_t *) x->data;
    const uint8_t * idx_data   = (const uint8_t *) d_idx->data;
    const uint8_t * val_data   = (const uint8_t *) d_val->data;
    const uint8_t * rs_data    = row_scale ? (const uint8_t *) row_scale->data : nullptr;
    uint8_t * dst_data         = (uint8_t *) dst->data;

    for (int64_t t = 0; t < n_tokens; ++t) {
        if (is_tall) {
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
                        const float xv = read_x_f16_or_f32(x, x_data, ii, t);
                        y += vv * xv;
                    }

                    const float scale = row_scale ? read_f16_or_f32_1d(row_scale, rs_data, o) : 1.0f;
                    *(float *)(dst_data + o * dst->nb[0] + t * dst->nb[1]) = (scale != 1.0f) ? (y * scale) : y;
                }
            }
        } else {
            std::fill(y_hat.begin(), y_hat.end(), 0.0f);

            // y_hat = Σ_b F_b(x_chunk_b)
            for (int64_t b = 0; b < B; ++b) {
                const int64_t in0 = b * L;
                const int64_t in1 = std::min<int64_t>(n_in, in0 + L);
                for (int64_t i = 0; i < in1 - in0; ++i) {
                    x_chunk[(size_t) i] = read_x_f16_or_f32(x, x_data, in0 + i, t);
                }
                for (int64_t i = in1 - in0; i < L; ++i) {
                    x_chunk[(size_t) i] = 0.0f;
                }

                apply_base_block(x_chunk.data(), v.data(), tmp.data(), L, b, base_d1, base_d2, base_d3, base_perm1, base_perm2);
                for (int64_t i = 0; i < L; ++i) {
                    y_hat[(size_t) i] += v[(size_t) i];
                }
            }

            for (int64_t o = o0; o < o1 && o < L; ++o) {
                float y = y_hat[(size_t) o];

                const uint8_t * idx_col = idx_data + o * d_idx->nb[1];
                const uint8_t * val_col = val_data + o * d_val->nb[1];
                for (int64_t r = 0; r < K; ++r) {
                    const int32_t ii = read_idx_i16_i32(d_idx, idx_col, r, 0);
                    if (ii < 0 || ii >= (int32_t) n_in) {
                        continue;
                    }
                    const float vv = read_f16_or_f32(d_val, val_col, r, 0);
                    const float xv = read_x_f16_or_f32(x, x_data, ii, t);
                    y += vv * xv;
                }

                const float scale = row_scale ? read_f16_or_f32_1d(row_scale, rs_data, o) : 1.0f;
                *(float *)(dst_data + o * dst->nb[0] + t * dst->nb[1]) = (scale != 1.0f) ? (y * scale) : y;
            }
        }
    }
}

static void llama_seeddelta_base_block_op(struct ggml_tensor * dst, int ith, int nth, void * userdata) {
    GGML_UNUSED(userdata);

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

    std::vector<float> x_hat;
    std::vector<float> x_chunk;
    std::vector<float> v;
    std::vector<float> tmp;
    std::vector<float> y_hat;

    v.resize((size_t) L);
    tmp.resize((size_t) L);

    const bool is_tall = n_out >= n_in;
    if (is_tall) {
        x_hat.resize((size_t) L);
    } else {
        x_chunk.resize((size_t) L);
        y_hat.resize((size_t) L);
    }

    const uint8_t * x_data   = (const uint8_t *) x->data;
    const uint8_t * idx_data = (const uint8_t *) b_idx->data;
    const uint8_t * val_data = (const uint8_t *) b_val->data;
    const uint8_t * rs_data  = row_scale ? (const uint8_t *) row_scale->data : nullptr;
    uint8_t * dst_data       = (uint8_t *) dst->data;

    for (int64_t t = 0; t < n_tokens; ++t) {
        if (is_tall) {
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
                        const uint8_t * val_blk = val_col + bi * b_val->nb[1];
                        for (int64_t ii = in0; ii < in1; ++ii) {
                            const int64_t j = ii - in0;
                            const float vv = read_f16_or_f32(b_val, val_blk, j, 0);
                            const float xv = read_x_f16_or_f32(x, x_data, ii, t);
                            y += vv * xv;
                        }
                    }

                    const float scale = row_scale ? read_f16_or_f32_1d(row_scale, rs_data, o) : 1.0f;
                    *(float *)(dst_data + o * dst->nb[0] + t * dst->nb[1]) = (scale != 1.0f) ? (y * scale) : y;
                }
            }
        } else {
            std::fill(y_hat.begin(), y_hat.end(), 0.0f);

            // y_hat = Σ_b F_b(x_chunk_b)
            for (int64_t b = 0; b < B; ++b) {
                const int64_t in0 = b * L;
                const int64_t in1 = std::min<int64_t>(n_in, in0 + L);
                for (int64_t i = 0; i < in1 - in0; ++i) {
                    x_chunk[(size_t) i] = read_x_f16_or_f32(x, x_data, in0 + i, t);
                }
                for (int64_t i = in1 - in0; i < L; ++i) {
                    x_chunk[(size_t) i] = 0.0f;
                }

                apply_base_block(x_chunk.data(), v.data(), tmp.data(), L, b, base_d1, base_d2, base_d3, base_perm1, base_perm2);
                for (int64_t i = 0; i < L; ++i) {
                    y_hat[(size_t) i] += v[(size_t) i];
                }
            }

            for (int64_t o = o0; o < o1 && o < L; ++o) {
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
                    const uint8_t * val_blk = val_col + bi * b_val->nb[1];
                    for (int64_t ii = in0; ii < in1; ++ii) {
                        const int64_t j = ii - in0;
                        const float vv = read_f16_or_f32(b_val, val_blk, j, 0);
                        const float xv = read_x_f16_or_f32(x, x_data, ii, t);
                        y += vv * xv;
                    }
                }

                const float scale = row_scale ? read_f16_or_f32_1d(row_scale, rs_data, o) : 1.0f;
                *(float *)(dst_data + o * dst->nb[0] + t * dst->nb[1]) = (scale != 1.0f) ? (y * scale) : y;
            }
        }
    }
}

ggml_tensor * llama_seeddelta_mul_mat_base(
        ggml_context * ctx,
        ggml_tensor  * x,
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
            nullptr);

    return res;
}

ggml_tensor * llama_seeddelta_mul_mat_base_block(
        ggml_context * ctx,
        ggml_tensor  * x,
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
            nullptr);

    return res;
}
