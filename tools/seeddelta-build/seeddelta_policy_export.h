// SeedDelta policy export helpers.
// Exports a strict-schema policy.json capturing final per-tensor decisions
// (enabled/strip/K/block) from a run/report.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct sd_tensor_decision {
    int64_t layer = -1;
    std::string kind;      // "ffn_gate" | "ffn_up" | "ffn_down"
    bool enabled = false;  // emit delta for this tensor
    bool strip_dense = false;
    int64_t block = 0;     // only for scheme=block (0 otherwise)
    int64_t K_budget = 0;  // top-K budget used (pre-clamp already)
};

struct sd_policy_export_result {
    bool ok = false;
    std::string error;
};

// Writes a strict-schema policy.json with defaults.enabled=false and per-layer/tensor
// overrides derived from decisions. The output is suitable for `--policy-strict`.
sd_policy_export_result sd_policy_export_write_canonical(
        const std::string & out_path,
        const std::vector<sd_tensor_decision> & decisions);

