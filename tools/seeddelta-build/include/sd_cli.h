#pragma once

#include <cstdint>
#include <string>

struct sd_args {
    std::string in_fname;
    std::string out_fname;

    std::string layers_range;
    std::string imatrix_file;
    std::string report_json;
    std::string policy_file;
    std::string policy_export_file;

    std::string scheme_str = "coo";
    int64_t block = 32;

    int64_t K = 32;
    int64_t K_gate = -1;
    int64_t K_up   = -1;
    int64_t K_down = -1;
    std::string idx_type_str = "i16";
    std::string val_type_str = "f16";
    bool write_row_scale = false;
    bool write_base = false;
    bool strip_dense = false;
    int64_t base_max_samples = 2048;
    int base_perm_trials = 1;
    int n_threads = 1;
    int64_t eval_cols = 0;
    int64_t eval_x = 0;
    float imatrix_eps = 1e-8f;
    float imatrix_power = 1.0f;
    int seed = 1234;
    bool policy_strict = false;
    bool policy_dump_resolved = false;
    bool policy_self_test = false;
    bool overwrite_existing = false;
    double stack_cost_cap = 0.0;
    int64_t tile_rows = 0;
    int64_t tile_rows_align = 32;
};

bool sd_parse_args(int argc, char ** argv, sd_args & args);
