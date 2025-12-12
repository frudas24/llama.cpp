#include "llama-statecells.h"

#include "ggml.h"

#include <cmath>

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

// Compute y = gather_sum(p, codes) where:
//   p     : [M, n_tokens] projections, p[j,t] = dot(D[:,j], x[:,t])
//   codes : [k, n_out]    I16 signed indices, codes[r,o] = ±(atom+1), 0 = empty
//   row_scale : [n_out]   optional scale per output row
//   y     : [n_out, n_tokens]
static void llama_statecells_gather_sum_op(struct ggml_tensor * dst, int ith, int nth, void * userdata) {
    GGML_UNUSED(userdata);

    const ggml_tensor * p     = dst->src[0];
    const ggml_tensor * codes = dst->src[1];
    const ggml_tensor * row_scale = dst->src[2];

    GGML_ASSERT(p->type == GGML_TYPE_F16 || p->type == GGML_TYPE_F32);
    GGML_ASSERT(codes->type == GGML_TYPE_I16);

    const int64_t M        = p->ne[0];
    const int64_t k        = codes->ne[0];
    const int64_t n_out    = codes->ne[1];
    const int64_t n_tokens = p->ne[1];

    GGML_ASSERT(dst->ne[0] == n_out);
    GGML_ASSERT(dst->ne[1] == n_tokens);

    if (row_scale) {
        GGML_ASSERT(row_scale->type == GGML_TYPE_F16 || row_scale->type == GGML_TYPE_F32);
        GGML_ASSERT(row_scale->ne[0] == n_out);
    }

    const int64_t o0 = (n_out * ith) / nth;
    const int64_t o1 = (n_out * (ith + 1)) / nth;

    const bool fast_f32_contig =
            p->type == GGML_TYPE_F32 &&
            ggml_is_contiguous(p) &&
            ggml_is_contiguous(codes) &&
            (!row_scale || ggml_is_contiguous(row_scale)) &&
            ggml_is_contiguous(dst);

    if (fast_f32_contig) {
        const auto * p_data     = (const float *)   p->data;
        const auto * codes_data = (const int16_t *) codes->data;
        const auto * rs_f16     = row_scale && row_scale->type == GGML_TYPE_F16 ? (const ggml_fp16_t *) row_scale->data : nullptr;
        const auto * rs_f32     = row_scale && row_scale->type == GGML_TYPE_F32 ? (const float *)      row_scale->data : nullptr;
        auto * dst_data         = (float *)         dst->data;

        for (int64_t o = o0; o < o1; ++o) {
            const float scale = row_scale ? (rs_f32 ? rs_f32[o] : ggml_fp16_to_fp32(rs_f16[o])) : 1.0f;

            // init y[o, :] = 0
            for (int64_t t = 0; t < n_tokens; ++t) {
                dst_data[o + t * n_out] = 0.0f;
            }

            // y[o,t] += sign * p[atom,t]
            const int16_t * codes_col = codes_data + o * k;
            for (int64_t r = 0; r < k; ++r) {
                const int16_t code = codes_col[r];
                if (code == 0) {
                    continue;
                }

                const float sign = code > 0 ? 1.0f : -1.0f;
                const int64_t atom = (int64_t) std::abs(code) - 1;
                if (atom < 0 || atom >= M) {
                    continue;
                }

                for (int64_t t = 0; t < n_tokens; ++t) {
                    dst_data[o + t * n_out] += sign * p_data[atom + t * M];
                }
            }

            if (scale != 1.0f) {
                for (int64_t t = 0; t < n_tokens; ++t) {
                    dst_data[o + t * n_out] *= scale;
                }
            }
        }

        return;
    }

    const uint8_t * p_data     = (const uint8_t *) p->data;
    const uint8_t * codes_data = (const uint8_t *) codes->data;
    const uint8_t * rs_data    = row_scale ? (const uint8_t *) row_scale->data : nullptr;
    uint8_t * dst_data         = (uint8_t *) dst->data;

    for (int64_t o = o0; o < o1; ++o) {
        const uint8_t * codes_col = codes_data + o * codes->nb[1];
        const float scale = row_scale ? read_f16_or_f32_1d(row_scale, rs_data, o) : 1.0f;

        // init y[o, :] = 0
        for (int64_t t = 0; t < n_tokens; ++t) {
            *(float *)(dst_data + o * dst->nb[0] + t * dst->nb[1]) = 0.0f;
        }

        // y[o,t] += sign * p[atom,t]
        for (int64_t r = 0; r < k; ++r) {
            const int16_t code = *(const int16_t *)(codes_col + r * codes->nb[0]);
            if (code == 0) {
                continue;
            }

            const float sign = code > 0 ? 1.0f : -1.0f;
            const int64_t atom = (int64_t) std::abs(code) - 1;
            if (atom < 0 || atom >= M) {
                continue;
            }

            for (int64_t t = 0; t < n_tokens; ++t) {
                const float pv = read_f16_or_f32(p, p_data, atom, t);
                *(float *)(dst_data + o * dst->nb[0] + t * dst->nb[1]) += sign * pv;
            }
        }

        if (scale != 1.0f) {
            for (int64_t t = 0; t < n_tokens; ++t) {
                *(float *)(dst_data + o * dst->nb[0] + t * dst->nb[1]) *= scale;
            }
        }
    }
}

ggml_tensor * llama_statecells_mul_mat(
        ggml_context * ctx,
        ggml_tensor  * x,
        ggml_tensor * dict,
        ggml_tensor * codes,
        ggml_tensor * row_scale) {
    GGML_ASSERT(dict != nullptr && codes != nullptr);

    // Phase 1: p = D^T x, computed once per token/batch.
    // dict: [n_in, M]
    // x   : [n_in, n_tokens]
    // p   : [M,    n_tokens]
    ggml_tensor * p = ggml_mul_mat(ctx, dict, x);

    const int64_t n_out    = codes->ne[1];
    const int64_t n_tokens = x->ne[1];

    // Always pass 3 args so dst->src[2] is deterministically set (can be NULL).
    ggml_tensor * args[3] = { p, codes, row_scale };

    ggml_tensor * res = ggml_custom_4d(
            ctx, GGML_TYPE_F32,
            n_out, n_tokens, 1, 1,
            args, 3,
            llama_statecells_gather_sum_op,
            GGML_N_TASKS_MAX,
            nullptr);

    return res;
}
