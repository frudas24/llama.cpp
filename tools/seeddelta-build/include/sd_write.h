#pragma once

#include <cstdint>
#include <string>
#include <unordered_set>
#include <vector>

#include "ggml.h"
#include "sd_build.h"
#include "sd_report.h"

struct sd_write_params {
    gguf_context * src = nullptr;
    ggml_context * ctx_data = nullptr;

    sd_resid_scheme scheme = sd_resid_scheme::coo;
    ggml_type idx_type = GGML_TYPE_I16;
    ggml_type val_type = GGML_TYPE_F16;
    int64_t block = 32;
    int64_t K_default = 32;
    int64_t K_gate_eff = 32;
    int64_t K_up_eff = 32;
    int64_t K_down_eff = 32;
    bool K_variable = false;
    bool write_row_scale = false;
    bool write_base = false;
    bool strip_dense = false;

    bool any_strip = false;
    const std::unordered_set<std::string> * strip_weights = nullptr;
    const std::vector<pending_tensor_set> * pending = nullptr;
    const std::vector<ggml_context *> * sd_contexts = nullptr; // lifetime only
    const std::vector<report_entry> * report = nullptr;
    const stack_safety_tracker * stack_guard = nullptr;
    int64_t n_added = 0;

    std::string in_fname;
    std::string out_fname;
    std::string report_json;
    std::string policy_file;
    std::string policy_hash;
    std::string policy_export_file;
    bool have_imatrix = false;
    std::string imatrix_file;
    float imatrix_eps = 0.0f;
    float imatrix_power = 0.0f;
    std::vector<std::string> imatrix_datasets;
    int64_t eval_cols = 0;
    int64_t eval_x = 0;
    int64_t base_max_samples = 0;
    int base_perm_trials = 0;
};

int sd_write_output(const sd_write_params & params);
