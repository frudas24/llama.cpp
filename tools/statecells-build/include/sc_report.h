#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "sc_eval.h"

struct sc_report_row {
    int64_t layer = 0;
    std::string kind;
    int64_t n_in = 0;
    int64_t n_out = 0;
    int64_t M_eff = 0;
    int k_eff = 0;
    int64_t eval_cols = 0;
    eval_metrics em;
    bool imatrix = false;
};

struct sc_report_config {
    std::string input;
    std::string source;
    std::string output;
    bool resume = false;
    bool write_vals = false;
    bool write_row_scale = false;
    struct {
        bool enabled = false;
        std::string file;
        std::string dataset;
        int chunks = -1;
        float eps = 0.0f;
        float power = 0.0f;
    } imatrix;
    struct {
        int64_t M = 0;
        int64_t k = 0;
        float eta = 0.0f;
        int iters = 0;
        int64_t max_samples = 0;
    } dict;
    int64_t eval_cols = 0;
};

bool sc_write_report_json(
        const std::string & path,
        const sc_report_config & cfg,
        const std::vector<sc_report_row> & rows);
