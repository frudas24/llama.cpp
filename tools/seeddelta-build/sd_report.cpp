#include "sd_report.h"

#include <fstream>
#include <cstdio>

namespace {

std::string json_escape(const std::string & s) {
    std::string o;
    o.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
            case '\\': o += "\\\\"; break;
            case '\"': o += "\\\""; break;
            case '\n': o += "\\n"; break;
            case '\r': o += "\\r"; break;
            case '\t': o += "\\t"; break;
            default: o += c; break;
        }
    }
    return o;
}

} // namespace

void finalize_report_entry(report_entry & e) {
    if (e.metric_used.empty()) {
        e.metric_used = e.gating_metric_used;
    }
    if (e.target_tau_mean == 0.0) {
        e.target_tau_mean = e.gating_min_mean;
    }
    if (e.target_tau_p05 == 0.0) {
        e.target_tau_p05 = e.gating_min_p05;
    }
    if (e.reject_reason.empty()) {
        e.reject_reason = e.decision_reason;
    }
}

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
        const std::vector<report_entry> & report) {
    std::ofstream out(path);
    if (!out) {
        return false;
    }

    out << "{\n";
    out << "  \"input\": \"" << json_escape(in_fname) << "\",\n";
    out << "  \"output\": \"" << json_escape(out_fname) << "\",\n";
    out << "  \"imatrix\": " << (have_imatrix ? "true" : "false") << ",\n";
    out << "  \"base\": " << (write_base ? "true" : "false") << ",\n";
    out << "  \"policy_file\": \"" << (policy_file.empty() ? "" : json_escape(policy_file)) << "\",\n";
    out << "  \"policy_hash\": \"" << json_escape(policy_hash) << "\",\n";
    if (write_base) {
        out << "  \"base_kind\": \"xor_circulant\",\n";
        out << "  \"base_max_samples\": " << base_max_samples << ",\n";
        out << "  \"base_perm_trials\": " << base_perm_trials << ",\n";
    }
    if (have_imatrix) {
        out << "  \"imatrix_file\": \"" << json_escape(imatrix_file) << "\",\n";
        out << "  \"imatrix_eps\": " << imatrix_eps << ",\n";
        out << "  \"imatrix_power\": " << imatrix_power << ",\n";
        out << "  \"imatrix_datasets\": [";
        for (size_t i = 0; i < imatrix_datasets.size(); ++i) {
            if (i) out << ", ";
            out << "\"" << json_escape(imatrix_datasets[i]) << "\"";
        }
        out << "],\n";
    }
    out << "  \"resid\": {\n";
    out << "    \"scheme\": \"" << (scheme_block ? "block" : "coo") << "\",\n";
    if (scheme_block) {
        out << "    \"block\": " << block << ",\n";
    }
    out << "    \"K\": " << K_default << ",\n";
    out << "    \"K_gate\": " << K_gate_eff << ",\n";
    out << "    \"K_up\": " << K_up_eff << ",\n";
    out << "    \"K_down\": " << K_down_eff << ",\n";
    out << "    \"idx_type\": \"" << json_escape(idx_type_str) << "\",\n";
    out << "    \"val_type\": \"" << json_escape(val_type_str) << "\",\n";
    out << "    \"row_scale\": " << (write_row_scale ? "true" : "false") << "\n";
    out << "  },\n";
    out << "  \"eval_cols\": " << eval_cols << ",\n";
    out << "  \"eval_x\": " << eval_x << ",\n";
    out << "  \"stack_pass_gate_up\": " << stack_pass_gate_up << ",\n";
    out << "  \"stack_pass_down\": " << stack_pass_down << ",\n";
    out << "  \"weights\": [\n";

    for (size_t i = 0; i < report.size(); ++i) {
        const auto & e = report[i];
        out << "    {\n";
        out << "      \"layer\": " << e.layer << ",\n";
        out << "      \"kind\": \"" << json_escape(e.kind) << "\",\n";
        out << "      \"n_in\": " << e.n_in << ",\n";
        out << "      \"n_out\": " << e.n_out << ",\n";
        out << "      \"K_budget\": " << e.K_budget << ",\n";
        out << "      \"K\": " << e.K << ",\n";
        out << "      \"block\": " << e.block << ",\n";
        out << "      \"n_blocks\": " << e.n_blocks << ",\n";
        out << "      \"has_w\": " << (e.has_w ? "true" : "false") << ",\n";
        out << "      \"has_x\": " << (e.has_x ? "true" : "false") << ",\n";
        out << "      \"base_L\": " << e.cost.L << ",\n";
        out << "      \"base_B\": " << e.cost.B << ",\n";
        out << "      \"ops_dense\": " << e.cost.ops_dense << ",\n";
        out << "      \"ops_base\": " << e.cost.ops_base << ",\n";
        out << "      \"ops_delta\": " << e.cost.ops_delta << ",\n";
        out << "      \"ops_row_scale\": " << e.cost.ops_row_scale << ",\n";
        out << "      \"ops_total\": " << e.cost.ops_total << ",\n";
        out << "      \"ops_ratio\": " << e.cost.ops_ratio << ",\n";
        out << "      \"rel_l2_mean\": " << e.em.rel_l2_mean << ",\n";
        out << "      \"rel_l2_p95\": " << e.em.rel_l2_p95 << ",\n";
        out << "      \"cos_mean\": " << e.em.cos_mean << ",\n";
        out << "      \"cos_p05\": " << e.em.cos_p05 << ",\n";
        out << "      \"norm_ratio_mean\": " << e.em.norm_ratio_mean << ",\n";
        out << "      \"rel_l2_mean_w\": " << e.em_w.rel_l2_mean << ",\n";
        out << "      \"rel_l2_p95_w\": " << e.em_w.rel_l2_p95 << ",\n";
        out << "      \"cos_mean_w\": " << e.em_w.cos_mean << ",\n";
        out << "      \"cos_p05_w\": " << e.em_w.cos_p05 << ",\n";
        out << "      \"norm_ratio_mean_w\": " << e.em_w.norm_ratio_mean << ",\n";
        out << "      \"rel_l2_mean_x\": " << e.em_x.rel_l2_mean << ",\n";
        out << "      \"rel_l2_p95_x\": " << e.em_x.rel_l2_p95 << ",\n";
        out << "      \"cos_mean_x\": " << e.em_x.cos_mean << ",\n";
        out << "      \"cos_p05_x\": " << e.em_x.cos_p05 << ",\n";
        out << "      \"norm_ratio_mean_x\": " << e.em_x.norm_ratio_mean << ",\n";
        out << "      \"rel_l2_mean_x_w\": " << e.em_x_w.rel_l2_mean << ",\n";
        out << "      \"rel_l2_p95_x_w\": " << e.em_x_w.rel_l2_p95 << ",\n";
        out << "      \"cos_mean_x_w\": " << e.em_x_w.cos_mean << ",\n";
        out << "      \"cos_p05_x_w\": " << e.em_x_w.cos_p05 << ",\n";
        out << "      \"norm_ratio_mean_x_w\": " << e.em_x_w.norm_ratio_mean << ",\n";
        out << "      \"gating_enabled\": " << (e.gating_enabled ? "true" : "false") << ",\n";
        out << "      \"gating_metric\": \"" << json_escape(e.gating_metric_used) << "\",\n";
        out << "      \"gating_value\": " << e.gating_value << ",\n";
        out << "      \"gating_p05\": " << e.gating_p05 << ",\n";
        out << "      \"gating_min_mean\": " << e.gating_min_mean << ",\n";
        out << "      \"gating_min_p05\": " << e.gating_min_p05 << ",\n";
        out << "      \"gating_pass\": " << (e.gating_pass ? "true" : "false") << ",\n";
        out << "      \"metric_used\": \"" << json_escape(e.metric_used) << "\",\n";
        out << "      \"targets_used\": {\"tau_mean\": " << e.target_tau_mean << ", \"tau_p05\": " << e.target_tau_p05 << "},\n";
        out << "      \"stack_cost_delta\": " << e.stack_cost_delta << ",\n";
        out << "      \"stack_cost_total\": " << e.stack_cost_total << ",\n";
        out << "      \"ffn_proxy_available\": " << (e.ffn_proxy_available ? "true" : "false") << ",\n";
        out << "      \"ffn_proxy_reason\": \"" << json_escape(e.ffn_proxy_reason) << "\",\n";
        out << "      \"ffn_proxy_scope\": \"" << json_escape(e.ffn_proxy_scope) << "\",\n";
        out << "      \"ffn_proxy_base_used\": " << (e.ffn_proxy_base_used ? "true" : "false") << ",\n";
        out << "      \"ffn_proxy_eval_x\": " << e.ffn_proxy_eval_x << ",\n";
        out << "      \"ffn_proxy_eval_out\": " << e.ffn_proxy_eval_out << ",\n";
        out << "      \"ffn_proxy_seed\": " << e.ffn_proxy_seed << ",\n";
        out << "      \"ffn_proxy_cos_mean\": " << e.ffn_proxy_cos_mean << ",\n";
        out << "      \"ffn_proxy_cos_p05\": " << e.ffn_proxy_cos_p05 << ",\n";
        out << "      \"ffn_proxy_l2_mean\": " << e.ffn_proxy_l2_mean << ",\n";
        out << "      \"ffn_proxy_l2_p95\": " << e.ffn_proxy_l2_p95 << ",\n";
        out << "      \"ffn_proxy_log_norm_ratio_mean\": " << e.ffn_proxy_log_norm_ratio_mean << ",\n";
        out << "      \"ffn_proxy_log_norm_ratio_p95\": " << e.ffn_proxy_log_norm_ratio_p95 << ",\n";
        out << "      \"emit\": " << (e.emit ? "true" : "false") << ",\n";
        out << "      \"strip_dense\": " << (e.strip_applied ? "true" : "false") << ",\n";
        out << "      \"decision\": \"" << json_escape(e.decision_reason) << "\",\n";
        out << "      \"reject_reason\": \"" << json_escape(e.reject_reason) << "\",\n";
        out << "      \"tile_rows\": " << e.tile_rows << ",\n";
        out << "      \"tile_rows_align\": " << e.tile_rows_align << ",\n";
        out << "      \"k_total_per_tensor\": " << e.k_total_per_tensor << ",\n";
        out << "      \"k_levels\": [";
        for (size_t ki = 0; ki < e.k_levels.size(); ++ki) {
            if (ki) out << ", ";
            out << e.k_levels[ki];
        }
        out << "],\n";
        out << "      \"unique_k_count\": " << e.unique_k_count << ",\n";
        out << "      \"tiles_rounded_count\": " << e.tiles_rounded_count << ",\n";
        out << "      \"tiles_rounded_pct\": " << e.tiles_rounded_pct << ",\n";
        out << "      \"tiles_dropped_count\": " << e.tiles_dropped_count << ",\n";
        out << "      \"tiles_dropped_pct\": " << e.tiles_dropped_pct << ",\n";
        out << "      \"k_per_tile\": [";
        for (size_t ti = 0; ti < e.k_per_tile.size(); ++ti) {
            if (ti) out << ", ";
            out << e.k_per_tile[ti];
        }
        out << "],\n";
        out << "      \"k_custom_used\": " << (e.k_custom_used ? "true" : "false") << ",\n";
        out << "      \"k_requested_stats\": {\"has\": " << (e.k_requested_stats.has ? "true" : "false") << ", \"min\": " << e.k_requested_stats.min << ", \"max\": " << e.k_requested_stats.max << ", \"mean\": " << e.k_requested_stats.mean << "},\n";
        out << "      \"k_selected_stats\": {\"has\": " << (e.k_selected_stats.has ? "true" : "false") << ", \"min\": " << e.k_selected_stats.min << ", \"max\": " << e.k_selected_stats.max << ", \"mean\": " << e.k_selected_stats.mean << "},\n";
        out << "      \"autotune_enabled\": " << (e.autotune_enabled ? "true" : "false") << ",\n";
        out << "      \"autotune_selected_budget\": " << e.autotune_selected_budget << ",\n";
        out << "      \"autotune_attempts\": [";
        for (size_t ai = 0; ai < e.autotune_attempts.size(); ++ai) {
            const auto & a = e.autotune_attempts[ai];
            out << (ai == 0 ? "\n" : ",\n");
            out << "        {";
            out << "\"K_budget\": " << a.K_budget << ", ";
            out << "\"K_eff\": " << a.K_eff << ", ";
            out << "\"n_blocks\": " << a.n_blocks << ", ";
            out << "\"metric_value\": " << a.metric_value << ", ";
            out << "\"metric_p05\": " << a.metric_p05 << ", ";
            out << "\"pass\": " << (a.pass ? "true" : "false") << ", ";
            out << "\"seconds\": " << a.seconds;
            out << "}";
        }
        if (!e.autotune_attempts.empty()) {
            out << "\n      ";
        }
        out << "]\n";

        out << "    }" << (i + 1 < report.size() ? "," : "") << "\n";
    }
    out << "  ]\n";
    out << "}\n";

    fprintf(stderr, "seeddelta-build: wrote report %s\n", path.c_str());
    return true;
}
