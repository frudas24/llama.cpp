#include <cstdio>
#include <string>
#include <vector>

#include "sd_report.h"

int main() {
    report_entry entry;
    entry.layer = 0;
    entry.kind = "ffn_gate";
    entry.n_in = 4;
    entry.n_out = 4;
    entry.K_budget = 8;
    entry.K = 4;
    entry.metric_used = "cos";
    entry.gating_metric_used = "cos";
    entry.target_tau_mean = 0.5;
    entry.target_tau_p05 = 0.4;
    entry.gating_min_mean = 0.5;
    entry.gating_min_p05 = 0.4;
    entry.decision_reason = "selftest";
    entry.reject_reason = "selftest";
    entry.emit = true;
    entry.stack_cost_delta = 0.1;
    entry.stack_cost_total = 0.5;
    entry.autotune_enabled = true;
    entry.autotune_selected_budget = 8;

    std::vector<report_entry> report = { entry };
    const std::string path = "selftest_report.json";
    const bool ok = write_report_json(
            path,
            "in",
            "out",
            false,
            "",
            {},
            0.0f,
            0.0f,
            false,
            0,
            0,
            "",
            "",
            false,
            0,
            0,
            0,
            0,
            "",
            "",
            false,
            0,
            0,
            0,
            0,
            report);
    if (ok) {
        std::remove(path.c_str());
    }
    return ok ? 0 : 1;
}
