#pragma once

#include <cstdint>
#include <string>

struct sc_args {
    std::string in_fname;
    std::string out_fname;

    int64_t dict_M = 4096;
    int64_t dict_M_gate = -1;
    int64_t dict_M_up   = -1;
    int64_t dict_M_down = -1;
    int dict_k = 16;
    int dict_k_gate = -1;
    int dict_k_up   = -1;
    int dict_k_down = -1;
    float dict_eta = 0.01f;
    int dict_iters = 3;
    int64_t dict_max_samples = 2048;
    int n_threads = 1;
    std::string layers_range;
    std::string dict_type_str = "f16";
    bool write_vals = false;
    bool write_row_scale = true;
    int64_t eval_cols = 0;
    std::string report_json;
    std::string imatrix_file;
    float imatrix_eps = 1e-8f;
    float imatrix_power = 1.0f;
    bool resume = false;
    int64_t checkpoint_every = 0;
    int seed = 1234;
};

bool sc_parse_args(int argc, char ** argv, sc_args & args);
