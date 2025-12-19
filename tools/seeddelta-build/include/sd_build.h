#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "ggml.h"
#include "gguf.h"
#include "sd_cli.h"
#include "sd_report.h"
#include "seeddelta_policy.h"

// Residual encoding scheme.
enum class sd_resid_scheme {
    coo = 0,
    block = 1,
};

// Guards against aggressive stacking decisions by relaxing thresholds once passes accumulate.
struct stack_safety_tracker {
    int gate_up_pass = 0;
    int down_pass = 0;

    void adjust(const std::string & kind, float & min_mean, float & min_p05) const {
        const bool is_down = (kind == "ffn_down");
        const int passed = is_down ? down_pass : gate_up_pass;
        if (passed >= 8) {
            if (is_down) {
                min_mean = std::max(min_mean, 0.35f);
                min_p05  = std::max(min_p05, 0.25f);
            } else {
                min_mean = std::max(min_mean, 0.60f);
                min_p05  = std::max(min_p05, 0.45f);
            }
        } else if (passed >= 3) {
            min_mean += 0.05f;
            min_p05  += 0.05f;
        }
    }

    void record_pass(const std::string & kind) {
        if (kind == "ffn_down") {
            ++down_pass;
        } else {
            ++gate_up_pass;
        }
    }
};

struct pending_tensor_set {
    ggml_context * ctx = nullptr;
    std::vector<ggml_tensor *> tensors;
};

struct sd_build_result {
    bool ok = true;
    int64_t n_added = 0;
    std::vector<report_entry> report;
    std::vector<pending_tensor_set> pending;
    std::unordered_set<std::string> strip_weights;
    bool any_strip = false;
    stack_safety_tracker stack_guard;
    std::vector<ggml_context *> sd_contexts;
};

// Performs the layer-by-layer SeedΔ build/eval loop. Returns report data and tensors to write.
sd_build_result sd_build_layers(
        const sd_args & args,
        gguf_context * src,
        ggml_context * ctx_data,
        const std::vector<int64_t> & layers,
        sd_resid_scheme scheme,
        ggml_type idx_type,
        ggml_type val_type,
        int64_t block,
        int64_t K_gate_eff,
        int64_t K_up_eff,
        int64_t K_down_eff,
        const sd_policy * policy_ptr,
        bool policy_dump_resolved,
        bool overwrite_existing,
        bool have_imatrix,
        const std::unordered_map<std::string, std::vector<float>> & imatrix_data,
        float imatrix_eps,
        float imatrix_power,
        double stack_cost_cap);
