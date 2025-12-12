#pragma once

#include <unordered_map>

struct ggml_context;
struct ggml_tensor;

// Minimal runtime representation for StateCells sparse-dictionary weights.
// For now we support FFN matrices only; dict/codes are expected to be present in GGUF.

struct llama_statecells_weight {
    ggml_tensor * dict  = nullptr; // [n_in, M], F16/F32
    ggml_tensor * codes = nullptr; // [k, n_out], I16 signed atom indices
};

struct llama_statecells_context {
    bool  enabled = false;
    float gap_tol = 0.02f;

    // Map base weight tensor -> StateCells payload.
    std::unordered_map<const ggml_tensor *, llama_statecells_weight> weights;

    const llama_statecells_weight * find(const ggml_tensor * w) const {
        auto it = weights.find(w);
        return it == weights.end() ? nullptr : &it->second;
    }
};

// Build a custom ggml op: y = W(x) using StateCells dict+codes.
ggml_tensor * llama_statecells_mul_mat(
        ggml_context * ctx,
        ggml_tensor  * x,
        ggml_tensor * dict,
        ggml_tensor * codes);
