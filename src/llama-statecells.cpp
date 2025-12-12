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

static void llama_statecells_mul_mat_op(struct ggml_tensor * dst, int ith, int nth, void * userdata) {
    GGML_UNUSED(userdata);

    const ggml_tensor * x     = dst->src[0];
    const ggml_tensor * dict  = dst->src[1];
    const ggml_tensor * codes = dst->src[2];

    GGML_ASSERT(dict->type == GGML_TYPE_F16 || dict->type == GGML_TYPE_F32);
    GGML_ASSERT(x->type    == GGML_TYPE_F16 || x->type    == GGML_TYPE_F32);
    GGML_ASSERT(codes->type == GGML_TYPE_I16);

    const int64_t n_in     = dict->ne[0];
    const int64_t M        = dict->ne[1];
    const int64_t k        = codes->ne[0];
    const int64_t n_out    = codes->ne[1];
    const int64_t n_tokens = x->ne[1];

    GGML_ASSERT(x->ne[0] == n_in);
    GGML_ASSERT(dst->ne[0] == n_out);
    GGML_ASSERT(dst->ne[1] == n_tokens);

    const uint8_t * x_data    = (const uint8_t *) x->data;
    const uint8_t * dict_data = (const uint8_t *) dict->data;
    const uint8_t * codes_data= (const uint8_t *) codes->data;

    float * dst_data = (float *) dst->data;

    const int64_t o0 = (n_out * ith) / nth;
    const int64_t o1 = (n_out * (ith + 1)) / nth;

    for (int64_t o = o0; o < o1; ++o) {
        const uint8_t * codes_col = codes_data + o * codes->nb[1];

        for (int64_t t = 0; t < n_tokens; ++t) {
            float y = 0.0f;

            // accumulate k atoms
            for (int64_t r = 0; r < k; ++r) {
                const int16_t code = *(const int16_t *)(codes_col + r * codes->nb[0]);
                if (code == 0) {
                    continue;
                }

                const int sign = code > 0 ? 1 : -1;
                const int64_t atom = (int64_t) std::abs(code) - 1;
                if (atom < 0 || atom >= M) {
                    continue;
                }

                const uint8_t * dict_atom = dict_data + atom * dict->nb[1];
                const uint8_t * x_tok     = x_data    + t    * x->nb[1];

                float dot = 0.0f;
                for (int64_t i = 0; i < n_in; ++i) {
                    const float dv = read_f16_or_f32(dict, dict_atom, i, 0);
                    const float xv = read_f16_or_f32(x,    x_tok,     i, 0);
                    dot += dv * xv;
                }

                y += sign * dot;
            }

            dst_data[o + t * dst->ne[0]] = y;
        }
    }
}

ggml_tensor * llama_statecells_mul_mat(
        ggml_context * ctx,
        ggml_tensor  * x,
        const ggml_tensor * dict,
        const ggml_tensor * codes) {
    GGML_ASSERT(dict != nullptr && codes != nullptr);

    const int64_t n_out    = codes->ne[1];
    const int64_t n_tokens = x->ne[1];

    ggml_tensor * args[3] = { x, (ggml_tensor *) dict, (ggml_tensor *) codes };

    ggml_tensor * res = ggml_custom_4d(
            ctx, GGML_TYPE_F32,
            n_out, n_tokens, 1, 1,
            args, 3,
            llama_statecells_mul_mat_op,
            GGML_N_TASKS_MAX,
            nullptr);

    return res;
}

