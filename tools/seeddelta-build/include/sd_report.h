#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "sd_cost.h"
#include "sd_eval.h"

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

    int64_t tile_rows = 0;
    int64_t tile_rows_align = 0;
    int64_t k_total_per_tensor = 0;
    std::vector<int64_t> k_levels;
    std::vector<int64_t> k_per_tile;
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

void finalize_report_entry(report_entry & e);

bool write_report_json(
        const std::string & path,
        const std::string & in_fname,
        const std::string & out_fname,
        bool have_imatrix,
        const std::string & imatrix_file,
        const std::vector<std::string> & imatrix_datasets,
        float imatrix_eps,
        float imatrix_power,
        bool write_base,
        int64_t base_max_samples,
        int base_perm_trials,
        const std::string & policy_file,
        const std::string & policy_hash,
        bool scheme_block,
        int64_t block,
        int64_t K_default,
        int64_t K_gate_eff,
        int64_t K_up_eff,
        int64_t K_down_eff,
        const std::string & idx_type_str,
        const std::string & val_type_str,
        bool write_row_scale,
        int64_t eval_cols,
        int64_t eval_x,
        int stack_pass_gate_up,
        int stack_pass_down,
        const std::vector<report_entry> & report);
