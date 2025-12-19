#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

// Loads an imatrix GGUF file produced by llama-imatrix (or legacy format). Returns chunk count or <0 on error.
int load_imatrix(
        const std::string & imatrix_file,
        std::vector<std::string> & imatrix_datasets,
        std::unordered_map<std::string, std::vector<float>> & imatrix_data);

// Compute per-input sqrt(imatrix) scaling for a given tensor; returns false if the entry is missing.
bool make_imatrix_sqrt_scale(
        const std::unordered_map<std::string, std::vector<float>> & imatrix_data,
        const std::string & weight_name,
        int64_t n_in,
        float eps,
        float power,
        std::vector<float> & scale_out);

// Undo per-input scaling for a learned dictionary.
void unscale_dict_inplace(
        std::vector<float> & D,
        int64_t n_in,
        int64_t M,
        const std::vector<float> & w_scale);
