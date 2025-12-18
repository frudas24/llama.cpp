#pragma once

#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "ggml.h"
#include "sd_eval.h"

struct ffn_proxy_metrics {
    double cos_mean = 0.0;
    double cos_p05 = 0.0;
    double l2_mean = 0.0;
    double l2_p95 = 0.0;
    double log_norm_ratio_mean = 0.0;
    double log_norm_ratio_p95 = 0.0;
    int64_t eval_x = 0;
    int64_t eval_out = 0;
};

bool eval_ffn_proxy_coo_replace_one(
        const std::string & kind,
        const ggml_tensor * W_gate,
        const ggml_tensor * W_up,
        const ggml_tensor * W_down,
        const std::vector<int32_t> & d_idx,
        const std::vector<float> & d_val,
        const base_fit * base,
        bool write_base,
        int64_t K,
        int64_t eval_x,
        int64_t eval_cols,
        int seed,
        ffn_proxy_metrics & out);
