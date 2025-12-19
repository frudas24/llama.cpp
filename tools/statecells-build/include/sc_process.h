#pragma once

#include <cstdint>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "gguf.h"
#include "ggml.h"
#include "sc_cli.h"
#include "sc_report.h"

struct sc_process_result {
    int64_t n_added = 0;
    int64_t n_sc_existing = 0;
    std::vector<sc_report_row> report_rows;
    std::vector<ggml_context *> sc_contexts;
};

sc_process_result sc_process_layers(
        const sc_args & args,
        gguf_context * src,
        ggml_context * ctx_data,
        gguf_context * dst,
        const std::unordered_map<std::string, std::vector<float>> & imatrix_data,
        std::mt19937 & rng);
