#include "sd_constants.h"
#include "sd_report.h"

#include <cstdio>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace {

std::string json_escape(const std::string & s) {
    std::string out;
    out.reserve(s.size());
    for (unsigned char c : s) {
        switch (c) {
            case '\\': out += "\\\\"; break;
            case '"':  out += "\\\""; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (c < 0x20) {
                    std::ostringstream oss;
                    oss << "\\u";
                    oss << std::hex << std::uppercase;
                    oss.width(4);
                    oss.fill('0');
                    oss << (int) c;
                    out += oss.str();
                } else {
                    out.push_back(static_cast<char>(c));
                }
        }
    }
    return out;
}

const char * scheme_to_string(sd_report::seeddelta_scheme scheme) {
    return (scheme == sd_report::seeddelta_scheme::block)
            ? sd_constants::scheme_block.data()
            : sd_constants::scheme_coo.data();
}

} // namespace

void sd_report::finalize_report_entry(report_entry & e) {
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

std::string sd_report::metric_kind_to_string(sd_metric_kind m) {
    switch (m) {
        case sd_metric_kind::cos_x_w: return "cos_x_w";
        case sd_metric_kind::cos_x:   return "cos_x";
        case sd_metric_kind::cos_w:   return "cos_w";
        case sd_metric_kind::cos:     return "cos";
    }
    return "cos";
}

double sd_report::pick_metric_value(const report_entry & re, sd_metric_kind m) {
    switch (m) {
        case sd_metric_kind::cos_x_w: return re.em_x_w.cos_mean;
        case sd_metric_kind::cos_x:   return re.em_x.cos_mean;
        case sd_metric_kind::cos_w:   return re.em_w.cos_mean;
        case sd_metric_kind::cos:     return re.em.cos_mean;
    }
    return re.em.cos_mean;
}

double sd_report::pick_metric_p05(const report_entry & re, sd_metric_kind m) {
    switch (m) {
        case sd_metric_kind::cos_x_w: return re.em_x_w.cos_p05;
        case sd_metric_kind::cos_x:   return re.em_x.cos_p05;
        case sd_metric_kind::cos_w:   return re.em_w.cos_p05;
        case sd_metric_kind::cos:     return re.em.cos_p05;
    }
    return re.em.cos_p05;
}

bool sd_report::write_report_json(
        const std::string & path,
        const report_config & cfg,
        const std::vector<report_entry> & report) {
    std::ofstream out(path);
    if (!out) {
        fprintf(stderr, "seeddelta-build: failed to open report %s\n", path.c_str());
        return false;
    }

    out << "{\n";
    out << "  \"input\": \"" << json_escape(cfg.input) << "\",\n";
    out << "  \"output\": \"" << json_escape(cfg.output) << "\",\n";
    out << "  \"base\": " << (cfg.write_base ? "true" : "false") << ",\n";
    out << "  \"policy_file\": \"" << (cfg.policy_file.empty() ? "" : json_escape(cfg.policy_file)) << "\",\n";
    out << "  \"policy_hash\": \"" << json_escape(cfg.policy_hash) << "\",\n";
    if (cfg.write_base) {
        out << "  \"base_kind\": \"xor_circulant\",\n";
        out << "  \"base_max_samples\": " << cfg.base_max_samples << ",\n";
        out << "  \"base_perm_trials\": " << cfg.base_perm_trials << ",\n";
    }
    if (cfg.have_imatrix) {
        out << "  \"imatrix_file\": \"" << json_escape(cfg.imatrix_file) << "\",\n";
        out << "  \"imatrix_eps\": " << cfg.imatrix_eps << ",\n";
        out << "  \"imatrix_power\": " << cfg.imatrix_power << ",\n";
        out << "  \"imatrix_datasets\": [";
        for (size_t i = 0; i < cfg.imatrix_datasets.size(); ++i) {
            if (i) out << ", ";
            out << "\"" << json_escape(cfg.imatrix_datasets[i]) << "\"";
        }
        out << "],\n";
    }
    out << "  \"resid\": {\n";
    out << "    \"scheme\": \"" << scheme_to_string(cfg.resid.scheme) << "\",\n";
    if (cfg.resid.scheme == seeddelta_scheme::block) {
        out << "    \"block\": " << cfg.resid.block << ",\n";
    }
    out << "    \"K\": " << cfg.resid.K << ",\n";
    out << "    \"K_gate\": " << cfg.resid.K_gate << ",\n";
    out << "    \"K_up\": " << cfg.resid.K_up << ",\n";
    out << "    \"K_down\": " << cfg.resid.K_down << ",\n";
    out << "    \"idx_type\": \"" << json_escape(cfg.resid.idx_type) << "\",\n";
    out << "    \"val_type\": \"" << json_escape(cfg.resid.val_type) << "\",\n";
    out << "    \"row_scale\": " << (cfg.resid.row_scale ? "true" : "false") << "\n";
    out << "  },\n";
    out << "  \"eval_cols\": " << cfg.eval_cols << ",\n";
    out << "  \"eval_x\": " << cfg.eval_x << ",\n";
    out << "  \"stack_pass_gate_up\": " << cfg.stack_pass_gate_up << ",\n";
    out << "  \"stack_pass_down\": " << cfg.stack_pass_down << ",\n";
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
