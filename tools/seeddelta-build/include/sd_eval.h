#pragma once

#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "ggml.h"

struct base_fit {
    int64_t L = 0;
    int64_t B = 0;
    std::vector<float> d1;
    std::vector<float> d2;
    std::vector<float> d3;
    std::vector<float> h2;
    std::vector<int32_t> perm1;
    std::vector<int32_t> perm1_inv;
};

struct eval_metrics {
    double rel_l2_mean = 0.0;
    double rel_l2_p95  = 0.0;
    double cos_mean    = 0.0;
    double cos_p05     = 0.0;
    double norm_ratio_mean = 0.0;
};

void read_column_f32(const ggml_tensor * W, int64_t j, std::vector<float> & out);

void topk_abs_weighted(
        const std::vector<float> & v,
        const std::vector<float> * w_scale,
        int64_t K,
        std::vector<int32_t> & idx_out);

void topk_blocks_energy_weighted(
        const std::vector<float> & v,
        const std::vector<float> * w_scale,
        int64_t block,
        int64_t n_blocks_keep,
        std::vector<int32_t> & idx_out);

void apply_base_block_f32(
        const float * x_in,
        float * y_out,
        float * tmp,
        const base_fit & base,
        int64_t b);

base_fit fit_base_xor_circulant(
        const ggml_tensor * W,
        int64_t max_samples,
        const std::vector<float> * w_scale,
        int perm_trials,
        std::mt19937 & rng);

eval_metrics eval_sparse_residual(
        const ggml_tensor * W,
        const std::vector<int32_t> & d_idx,
        const std::vector<float> & d_val,
        const std::vector<float> * d_row_scale,
        const std::vector<float> * w_scale,
        int64_t K,
        int64_t eval_cols,
        std::mt19937 & rng);

eval_metrics eval_block_residual(
        const ggml_tensor * W,
        const std::vector<int32_t> & b_idx,
        const std::vector<float> & b_val,
        int64_t block,
        int64_t n_blocks_keep,
        const std::vector<float> * d_row_scale,
        const std::vector<float> * w_scale,
        int64_t eval_cols,
        std::mt19937 & rng);

eval_metrics eval_seeddelta_base_residual(
        const ggml_tensor * W,
        const base_fit & base,
        const std::vector<int32_t> & d_idx,
        const std::vector<float> & d_val,
        const std::vector<float> * d_row_scale,
        const std::vector<float> * w_scale,
        int64_t K,
        int64_t eval_cols,
        std::mt19937 & rng);

eval_metrics eval_seeddelta_base_block_residual(
        const ggml_tensor * W,
        const base_fit & base,
        const std::vector<int32_t> & b_idx,
        const std::vector<float> & b_val,
        int64_t block,
        int64_t n_blocks_keep,
        const std::vector<float> * d_row_scale,
        const std::vector<float> * w_scale,
        int64_t eval_cols,
        std::mt19937 & rng);

eval_metrics eval_seeddelta_x(
        const ggml_tensor * W,
        const base_fit * base,
        const std::vector<int32_t> & d_idx,
        const std::vector<float> & d_val,
        const std::vector<float> * d_row_scale,
        const std::vector<float> * w_scale,
        int64_t K,
        int64_t eval_cols,
        int64_t eval_x,
        std::mt19937 & rng);

eval_metrics eval_seeddelta_x_block(
        const ggml_tensor * W,
        const base_fit * base,
        const std::vector<int32_t> & b_idx,
        const std::vector<float> & b_val,
        int64_t block,
        int64_t n_blocks_keep,
        const std::vector<float> * d_row_scale,
        const std::vector<float> * w_scale,
        int64_t eval_cols,
        int64_t eval_x,
        std::mt19937 & rng);
