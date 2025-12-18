#pragma once

#include <cstdint>

struct base_fit;

struct cost_estimate {
    int64_t L = 0;
    int64_t B = 0;
    double ops_dense = 0.0;
    double ops_base  = 0.0;
    double ops_delta = 0.0;
    double ops_row_scale = 0.0;
    double ops_total = 0.0;
    double ops_ratio = 0.0;
};

cost_estimate estimate_cost(
        const base_fit * base,
        int64_t n_in,
        int64_t n_out,
        int64_t K,
        bool row_scale);
