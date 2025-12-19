#pragma once

#include <cstdint>
#include <random>
#include <vector>

#include "ggml.h"

struct eval_metrics {
    double rel_l2_mean = 0.0;
    double rel_l2_p95  = 0.0;
    double cos_mean    = 0.0;
    double cos_p05     = 0.0;

    double rel_l2_mean_w = 0.0;
    double rel_l2_p95_w  = 0.0;
    double cos_mean_w    = 0.0;
    double cos_p05_w     = 0.0;
};

eval_metrics eval_reconstruction_sign(
        const ggml_tensor * W,
        const std::vector<float> & D,
        int64_t M,
        int k,
        const std::vector<int16_t> & codes,
        const std::vector<ggml_fp16_t> * vals,
        const std::vector<ggml_fp16_t> * row_scale,
        int64_t eval_cols,
        std::mt19937 & rng,
        const std::vector<float> * w_scale_sqrt);
