#pragma once

#include <unordered_map>

struct ggml_context;
struct ggml_tensor;

// Minimal runtime representation for SeedΔ weights.
// v0: residual COO only (d_idx + d_val), optional row_scale; W0 assumed to be zero.

struct llama_seeddelta_weight {
    ggml_tensor * d_idx = nullptr; // [K, n_out], I16/I32 input indices
    ggml_tensor * d_val = nullptr; // [K, n_out], F16/F32 residual values
    ggml_tensor * row_scale = nullptr; // [n_out], F16/F32 (optional)
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

