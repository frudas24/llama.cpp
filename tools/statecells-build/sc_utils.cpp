#include "include/sc_utils.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstring>
#include <numeric>
#include <stdexcept>

std::vector<int64_t> parse_layer_range(const std::string & s, int64_t n_layer) {
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

void normalize_columns(std::vector<float> & D, int64_t n_in, int64_t M) {
    for (int64_t j = 0; j < M; ++j) {
        float norm = 0.0f;
        const float * col = D.data() + j * n_in;
        for (int64_t i = 0; i < n_in; ++i) norm += col[i] * col[i];
        norm = std::sqrt(std::max(norm, 1e-12f));
        float * colw = D.data() + j * n_in;
        for (int64_t i = 0; i < n_in; ++i) colw[i] /= norm;
    }
}

std::vector<int> topk_abs_indices(const std::vector<float> & y, int k) {
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

void read_column_f32(const ggml_tensor * W, int64_t j, std::vector<float> & out) {
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

void read_mat_f16_f32_to_f32(const ggml_tensor * A, std::vector<float> & out) {
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

void read_mat_f16_f32_to_fp16(const ggml_tensor * A, std::vector<ggml_fp16_t> & out) {
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

void read_vec_f16_f32_to_fp16(const ggml_tensor * A, std::vector<ggml_fp16_t> & out) {
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
