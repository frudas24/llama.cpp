#include "sd_cost.h"

#include <algorithm>
#include <cmath>

cost_estimate estimate_cost(
        const base_fit * base,
        int64_t n_in,
        int64_t n_out,
        int64_t K,
        bool row_scale) {
    cost_estimate out;
    if (n_in <= 0 || n_out <= 0 || K <= 0) {
        return out;
    }

    out.ops_dense = 2.0 * (double) n_in * (double) n_out;
    out.ops_delta = 2.0 * (double) K * (double) n_out;
    out.ops_row_scale = row_scale ? (double) n_out : 0.0;

    if (base && base->L > 0 && base->B > 0) {
        out.L = base->L;
        out.B = base->B;

        const bool is_tall = n_out >= n_in;
        const double L = (double) base->L;
        const double log2L = std::log2(std::max(2.0, L));

        const double had_ops  = 2.0 * L * log2L;
        const double diag_ops = 3.0 * L;
        const double acc_ops  = is_tall ? 0.0 : L;

        out.ops_base = (double) base->B * (had_ops + diag_ops + acc_ops);
    }

    out.ops_total = out.ops_base + out.ops_delta + out.ops_row_scale;
    out.ops_ratio = out.ops_dense > 0.0 ? (out.ops_total / out.ops_dense) : 0.0;
    return out;
}
