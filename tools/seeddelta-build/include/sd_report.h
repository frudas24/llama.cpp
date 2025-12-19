#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "sd_cost.h"
#include "sd_eval.h"
#include <seeddelta_policy.h>

struct report_entry {
    int64_t layer = -1;
    std::string kind;
    int64_t n_in = 0;
    int64_t n_out = 0;
    int64_t K_budget = 0;
    int64_t K = 0;
    int64_t block = 0;
    int64_t n_blocks = 0;
    eval_metrics em;
    eval_metrics em_w;
    eval_metrics em_x;
    eval_metrics em_x_w;
    cost_estimate cost;
    bool has_w = false;
    bool has_x = false;
    // Gating/decision metadata
    bool gating_enabled = false;
    bool gating_pass = true;
    std::string gating_metric_used;
    std::string metric_used;
    double gating_value = 0.0;
    double gating_p05 = 0.0;
    double gating_min_mean = 0.0;
    double gating_min_p05 = 0.0;
    double target_tau_mean = 0.0;
    double target_tau_p05 = 0.0;
    bool emit = true;
    bool strip_applied = false;
    std::string decision_reason;
    std::string reject_reason;
    double stack_cost_delta = 0.0;
    double stack_cost_total = 0.0;
    double ffn_score = 0.0;

    // Tiled-K metadata (v1: may be empty if no tiles)
    int64_t tile_rows = 0;
    int64_t tile_rows_align = 0;
    int64_t k_total_per_tensor = 0;
    std::vector<int64_t> k_levels;
    std::vector<int64_t> k_per_tile; // post-round, dense per tile (row order)
    int64_t unique_k_count = 0;
    int64_t tiles_rounded_count = 0;
    double tiles_rounded_pct = 0.0;
    int64_t tiles_dropped_count = 0;
    double tiles_dropped_pct = 0.0;
    bool k_custom_used = false;
    struct k_stats {
        double min = 0.0;
        double max = 0.0;
        double mean = 0.0;
        bool has = false;
    };
    k_stats k_requested_stats;
    k_stats k_selected_stats;

    // FFN proxy metrics (v0: replace-only-current-tensor; logging only)
    bool ffn_proxy_available = false;
    std::string ffn_proxy_reason;
    std::string ffn_proxy_scope;
    bool ffn_proxy_base_used = false;
    int64_t ffn_proxy_eval_x = 0;
    int64_t ffn_proxy_eval_out = 0;
    int64_t ffn_proxy_seed = 0;
    double ffn_proxy_cos_mean = 0.0;
    double ffn_proxy_cos_p05 = 0.0;
    double ffn_proxy_l2_mean = 0.0;
    double ffn_proxy_l2_p95 = 0.0;
    double ffn_proxy_log_norm_ratio_mean = 0.0;
    double ffn_proxy_log_norm_ratio_p95 = 0.0;

    struct autotune_attempt {
        int64_t K_budget = 0;
        int64_t K_eff = 0;
        int64_t n_blocks = 0;
        double metric_value = 0.0;
        double metric_p05 = 0.0;
        bool pass = false;
        double seconds = 0.0;
    };
    bool autotune_enabled = false;
    int64_t autotune_selected_budget = 0;
    std::vector<autotune_attempt> autotune_attempts;
};

namespace sd_report {

enum class seeddelta_scheme {
    coo,
    block,
};

struct resid_config {
    seeddelta_scheme scheme = seeddelta_scheme::coo;
    int64_t block = 0;
    int64_t K = 0;
    int64_t K_gate = 0;
    int64_t K_up = 0;
    int64_t K_down = 0;
    std::string idx_type;
    std::string val_type;
    bool row_scale = false;
};

struct report_config {
    std::string input;
    std::string output;
    bool write_base = false;
    int64_t base_max_samples = 0;
    int base_perm_trials = 0;
    std::string policy_file;
    std::string policy_hash;
    bool have_imatrix = false;
    std::string imatrix_file;
    double imatrix_eps = 0.0;
    double imatrix_power = 0.0;
    std::vector<std::string> imatrix_datasets;
    resid_config resid;
    int64_t eval_cols = 0;
    int64_t eval_x = 0;
    int stack_pass_gate_up = 0;
    int stack_pass_down = 0;
};

void finalize_report_entry(report_entry & e);

std::string metric_kind_to_string(sd_metric_kind m);
double pick_metric_value(const report_entry & re, sd_metric_kind m);
double pick_metric_p05(const report_entry & re, sd_metric_kind m);
double ffn_score_from_entry(const report_entry & re);

bool write_report_json(
        const std::string & path,
        const report_config & cfg,
        const std::vector<report_entry> & report);

} // namespace sd_report
