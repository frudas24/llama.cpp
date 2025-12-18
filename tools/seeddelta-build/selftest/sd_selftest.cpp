#include "sd_constants.h"
#include "sd_report.h"
#include "seeddelta_policy.h"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>

namespace {

bool test_policy_merge() {
    sd_policy policy;
    policy.layers[0].enabled = false;
    policy.layers[0].strip_dense = true;
    policy.layers[0].gate_thr.min_mean = 0.56f;
    policy.layers[0].tensors["ffn_gate"].enabled = true;
    policy.layers[0].tensors["ffn_gate"].K_gate = 99;
    policy.layers[0].tensors["ffn_gate"].strip_dense = true;
    policy.layers[0].tensors["ffn_gate"].gate_thr.min_p05 = 0.42f;

    sd_resolved_tensor baseline;
    baseline.enabled = true;
    baseline.K_gate = 1;
    baseline.strip_dense = false;

    sd_resolved_tensor resolved = sd_policy_resolve(&policy, 0, "ffn_gate", baseline);
    if (!resolved.enabled) {
        std::cerr << "policy merge: expected tensor to be enabled\n";
        return false;
    }
    if (resolved.K_gate != 99) {
        std::cerr << "policy merge: expected K_gate override\n";
        return false;
    }
    if (!resolved.strip_dense) {
        std::cerr << "policy merge: expected strip_dense override\n";
        return false;
    }
    if (resolved.min_mean != 0.56f) {
        std::cerr << "policy merge: min_mean not merged\n";
        return false;
    }
    if (resolved.min_p05 != 0.42f) {
        std::cerr << "policy merge: min_p05 not merged\n";
        return false;
    }
    return true;
}

bool test_report_serialization() {
    const std::string path = "sd_selftest_report.json";
    sd_report::report_entry entry;
    entry.layer = 3;
    entry.kind = "ffn_gate";
    entry.has_w = true;
    entry.gating_metric_used = "cos_x";
    entry.gating_value = 0.7;
    entry.gating_p05 = 0.4;
    entry.decision_reason = sd_constants::decision_pass_gating;
    entry.emit = true;
    entry.metric_used.clear();
    sd_report::finalize_report_entry(entry);

    sd_report::report_config cfg;
    cfg.input = "in.gguf";
    cfg.output = "out.gguf";
    cfg.write_base = true;
    cfg.base_max_samples = 1;
    cfg.base_perm_trials = 1;
    cfg.policy_file = "policy.json";
    cfg.policy_hash = "hash";
    cfg.have_imatrix = true;
    cfg.imatrix_file = "matrix.gguf";
    cfg.imatrix_eps = 1e-8;
    cfg.imatrix_power = 1.0;
    cfg.imatrix_datasets = { "dataset" };
    cfg.resid.scheme = sd_report::seeddelta_scheme::coo;
    cfg.resid.K = 32;
    cfg.resid.K_gate = 32;
    cfg.resid.K_up = 32;
    cfg.resid.K_down = 32;
    cfg.resid.idx_type = "i16";
    cfg.resid.val_type = "f16";
    cfg.resid.row_scale = true;
    cfg.eval_cols = 1;
    cfg.eval_x = 1;
    cfg.stack_pass_gate_up = 0;
    cfg.stack_pass_down = 0;

    if (!sd_report::write_report_json(path, cfg, { entry })) {
        std::cerr << "report serialization: failed to write JSON\n";
        return false;
    }

    std::ifstream in(path);
    if (!in) {
        std::cerr << "report serialization: missing output file\n";
        return false;
    }

    std::string contents((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    std::remove(path.c_str());

    if (contents.find("\"layer\": 3") == std::string::npos) {
        std::cerr << "report serialization: missing layer\n";
        return false;
    }
    if (contents.find("\"kind\": \"ffn_gate\"") == std::string::npos) {
        std::cerr << "report serialization: missing kind\n";
        return false;
    }
    if (contents.find(sd_constants::decision_pass_gating.data()) == std::string::npos) {
        std::cerr << "report serialization: missing decision reason\n";
        return false;
    }
    return true;
}

} // namespace

int main() {
    bool ok = test_policy_merge() && test_report_serialization();
    std::fputs(ok ? "sd_selftest: ok\n" : "sd_selftest: failed\n", stdout);
    return ok ? 0 : 1;
}
