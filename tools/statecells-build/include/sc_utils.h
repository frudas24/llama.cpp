#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "ggml.h"

std::vector<int64_t> parse_layer_range(const std::string & s, int64_t n_layer);
void normalize_columns(std::vector<float> & D, int64_t n_in, int64_t M);
std::vector<int> topk_abs_indices(const std::vector<float> & y, int k);
void read_column_f32(const ggml_tensor * W, int64_t j, std::vector<float> & out);
void read_mat_f16_f32_to_f32(const ggml_tensor * A, std::vector<float> & out);
void read_mat_f16_f32_to_fp16(const ggml_tensor * A, std::vector<ggml_fp16_t> & out);
void read_vec_f16_f32_to_fp16(const ggml_tensor * A, std::vector<ggml_fp16_t> & out);
