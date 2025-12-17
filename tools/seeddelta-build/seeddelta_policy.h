// SeedDelta policy parsing and resolution.
// Provides per-layer/per-tensor overrides (K/block/strip/gating/autotune) with
// deterministic merge order: CLI baseline -> defaults -> ranges (in order) -> layer -> tensor.

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

struct sd_policy_parse_result {
    bool ok = false;
    std::string error;
    std::vector<std::string> warnings;
};

enum class sd_metric_kind {
    cos,      // cos_mean / cos_p05
    cos_x,    // cos_mean_x / cos_p05_x
    cos_w,    // cos_mean_w / cos_p05_w
    cos_x_w,  // cos_mean_x_w / cos_p05_x_w
};

struct sd_gating_thresholds {
    std::optional<float> min_mean;
    std::optional<float> min_p05;
};

struct sd_autotune_cfg {
    bool enabled = false;
    std::vector<int64_t> schedule_gate_up;
    std::vector<int64_t> schedule_down;
    int max_iters = 0; // 0 = unlimited / schedule size
};

struct sd_rule {
    std::optional<bool> enabled;
    std::optional<bool> strip_dense;
    std::optional<int64_t> block;
    std::optional<int64_t> K_gate;
    std::optional<int64_t> K_up;
    std::optional<int64_t> K_down;
    std::optional<sd_metric_kind> metric;
    sd_gating_thresholds gate_thr;
    sd_gating_thresholds up_thr;
    sd_gating_thresholds down_thr;
    std::optional<sd_autotune_cfg> autotune;

    // Overrides per tensor name: "ffn_gate", "ffn_up", "ffn_down"
    std::unordered_map<std::string, sd_rule> tensors;
};

struct sd_policy {
    int version = 1;
    sd_rule defaults;
    struct range_rule {
        std::vector<int64_t> layers;
        sd_rule rule;
    };
    std::vector<range_rule> ranges;
    std::unordered_map<int64_t, sd_rule> layers;
    std::string source_path;
    bool strict = false;
    std::vector<std::string> warnings;
};

struct sd_resolved_tensor {
    bool enabled = true;
    bool strip_dense = false;
    int64_t block = 32;
    int64_t K_gate = 32;
    int64_t K_up = 32;
    int64_t K_down = 32;
    sd_metric_kind metric = sd_metric_kind::cos_x_w;
    float min_mean = 0.0f;
    float min_p05 = 0.0f;
    bool gating_enabled = false;
    bool require_eval_x = false;
    bool require_imatrix = false;
    sd_autotune_cfg autotune;
};

// Parse policy JSON file. On error returns ok=false with message.
sd_policy_parse_result sd_policy_load(const std::string & path, bool strict, sd_policy & out);

// Resolve effective config for a given layer+tensor (kind) starting from a baseline (CLI).
sd_resolved_tensor sd_policy_resolve(
        const sd_policy * policy,          // nullptr => no policy, use baseline
        int64_t layer,
        const std::string & tensor_kind,   // "ffn_gate" | "ffn_up" | "ffn_down"
        const sd_resolved_tensor & baseline);

// Helper: parse a layer selector string ("8-20,24") into a sorted unique list.
std::vector<int64_t> sd_parse_layer_selector(const std::string & s);

