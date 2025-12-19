#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "ggml.h"

std::vector<int64_t> sd_parse_layer_range(const std::string & s, int64_t n_layer);
bool sd_string_remove_suffix(std::string & s, const std::string & suffix);

int sd_load_imatrix(
        const std::string & imatrix_file,
        std::vector<std::string> & imatrix_datasets,
        std::unordered_map<std::string, std::vector<float>> & imatrix_data);

bool sd_make_imatrix_sqrt_scale(
        const std::unordered_map<std::string, std::vector<float>> & imatrix_data,
        const std::string & weight_name,
        int64_t n_in,
        float eps,
        float power,
        std::vector<float> & scale_out);
