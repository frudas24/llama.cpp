#pragma once

#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "ggml.h"

void train_dict_oja_kwta(
        const ggml_tensor * W,
        int64_t M,
        int k,
        float eta,
        int iters,
        int64_t max_samples,
        std::mt19937 & rng,
        std::vector<float> & D_out,
        const std::string & tag,
        const std::vector<float> * w_scale);

void encode_codes_sign(
        const ggml_tensor * W,
        const std::vector<float> & D,
        int64_t M,
        int k,
        int n_threads,
        std::vector<int16_t> & codes_out,
        std::vector<ggml_fp16_t> * vals_out,
        const std::string & tag,
        const std::vector<float> * w_scale);

void compute_vals_sign(
        const ggml_tensor * W,
        const std::vector<float> & D,
        int64_t M,
        int k,
        const std::vector<int16_t> & codes,
        int n_threads,
        std::vector<ggml_fp16_t> & vals_out,
        const std::string & tag,
        const std::vector<float> * w_scale);

void compute_row_scale_sign(
        const ggml_tensor * W,
        const std::vector<float> & D,
        int64_t M,
        int k,
        const std::vector<int16_t> & codes,
        const std::vector<ggml_fp16_t> * vals,
        int n_threads,
        std::vector<ggml_fp16_t> & row_scale_out,
        const std::string & tag);
