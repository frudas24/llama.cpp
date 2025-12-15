#pragma once

#include <unordered_map>

struct ggml_context;
struct ggml_tensor;

// Minimal runtime representation for SeedΔ weights.
// v0: residual COO only (d_idx + d_val), optional row_scale; W0 assumed to be zero.
// v1: optional base W0 (Hadamard/ACDC-style) + residual COO.

struct llama_seeddelta_weight {
    ggml_tensor * d_idx = nullptr; // [K, n_out], I16/I32 input indices
    ggml_tensor * d_val = nullptr; // [K, n_out], F16/F32 residual values
    ggml_tensor * b_idx = nullptr; // [nb, n_out], I16/I32 input block indices
    ggml_tensor * b_val = nullptr; // [block, nb, n_out], F16/F32 residual block values
    ggml_tensor * row_scale = nullptr; // [n_out], F16/F32 (optional)

    // Base W0 parameters (optional). Shape convention: [L, B] tensors where ne0=L, ne1=B.
    ggml_tensor * base_d1 = nullptr;    // [L, B], F16/F32
    ggml_tensor * base_d2 = nullptr;    // [L, B], F16/F32
    ggml_tensor * base_d3 = nullptr;    // [L, B], F16/F32
    ggml_tensor * base_perm1 = nullptr; // [L, B], I16/I32 (optional)
    ggml_tensor * base_perm2 = nullptr; // [L, B], I16/I32 (optional)
};

struct llama_seeddelta_context {
    bool  enabled = false;
    float gap_tol = 0.02f;

    // Map base weight tensor -> SeedΔ payload.
    std::unordered_map<const ggml_tensor *, llama_seeddelta_weight> weights;

    const llama_seeddelta_weight * find(const ggml_tensor * w) const {
        auto it = weights.find(w);
        return it == weights.end() ? nullptr : &it->second;
    }
};

// Build a custom ggml op: y = Δ(x) using SeedΔ residual COO (W0 not included in v0).
ggml_tensor * llama_seeddelta_mul_mat(
        ggml_context * ctx,
        ggml_tensor  * x,
        ggml_tensor  * d_idx,
        ggml_tensor  * d_val,
        ggml_tensor  * row_scale);

// Build a custom ggml op: y = Δ(x) using SeedΔ residual block-sparse.
ggml_tensor * llama_seeddelta_mul_mat_block(
        ggml_context * ctx,
        ggml_tensor  * x,
        ggml_tensor  * b_idx,
        ggml_tensor  * b_val,
        ggml_tensor  * row_scale);

// Build a custom ggml op: y = W0(x) + Δ(x) using Hadamard/ACDC base + residual COO.
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
        ggml_tensor  * row_scale);

// Build a custom ggml op: y = W0(x) + Δ(x) using Hadamard/ACDC base + residual block-sparse.
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
        ggml_tensor  * row_scale);
