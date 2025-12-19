#include "common.h"
#include "ggml.h"
#include "gguf.h"
#include "seeddelta_policy.h"
#include "seeddelta_policy_export.h"
#include "seeddelta_policy_selftest.h"
#include "sd_cli.h"
#include "sd_policy_resolve.h"
#include "sd_utils.h"
#include "sd_runner.h"
#include "sd_cost.h"
#include "sd_eval.h"
#include "sd_ffn_proxy.h"
#include "sd_report.h"
#include "sd_constants.h"
#include "sd_build.h"
#include "sd_write.h"

#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

int sd_run(const sd_args & args) {
    const std::string in_fname = args.in_fname;
    const std::string out_fname = args.out_fname;
    const std::string layers_range = args.layers_range;
    const std::string imatrix_file = args.imatrix_file;
    const std::string report_json = args.report_json;
    const std::string policy_file = args.policy_file;
    const std::string policy_export_file = args.policy_export_file;

    const std::string scheme_str = args.scheme_str;
    const int64_t block = args.block;

    const int64_t K = args.K;
    const int64_t K_gate = args.K_gate;
    const int64_t K_up   = args.K_up;
    const int64_t K_down = args.K_down;
    const std::string idx_type_str = args.idx_type_str;
    const std::string val_type_str = args.val_type_str;
    const bool write_row_scale = args.write_row_scale;
    const bool write_base = args.write_base;
    const bool strip_dense = args.strip_dense;
    const int64_t base_max_samples = args.base_max_samples;
    const int base_perm_trials = args.base_perm_trials;
    int64_t eval_cols = args.eval_cols;
    const int64_t eval_x = args.eval_x;
    const float imatrix_eps = args.imatrix_eps;
    const float imatrix_power = args.imatrix_power;
    const bool policy_strict = args.policy_strict;
    const bool policy_dump_resolved = args.policy_dump_resolved;
    const bool policy_self_test = args.policy_self_test;
    const bool overwrite_existing = args.overwrite_existing;
    const double stack_cost_cap = args.stack_cost_cap;

    if (policy_self_test) {
        return sd_policy_self_test();
    }

    sd_resid_scheme scheme = sd_resid_scheme::coo;
    if (scheme_str == "coo") {
        scheme = sd_resid_scheme::coo;
    } else if (scheme_str == "block") {
        scheme = sd_resid_scheme::block;
    } else {
        throw std::runtime_error("invalid --scheme (expected: coo|block)");
    }
    if (scheme == sd_resid_scheme::block) {
        if (block <= 0 || block > 4096) {
            throw std::runtime_error("invalid --block (expected: 1..4096)");
        }
    }

    ggml_type idx_type = GGML_TYPE_I16;
    if (idx_type_str == "i16") idx_type = GGML_TYPE_I16;
    else if (idx_type_str == "i32") idx_type = GGML_TYPE_I32;
    else throw std::runtime_error("invalid --idx-type");

    ggml_type val_type = GGML_TYPE_F16;
    if (val_type_str == "f16") val_type = GGML_TYPE_F16;
    else if (val_type_str == "f32") val_type = GGML_TYPE_F32;
    else throw std::runtime_error("invalid --val-type");

    ggml_context * ctx_data = nullptr;
    gguf_init_params params = { false, &ctx_data };
    gguf_context * src = gguf_init_from_file(in_fname.c_str(), params);
    if (!src || !ctx_data) {
        fprintf(stderr, "failed to load %s\n", in_fname.c_str());
        return 1;
    }

    std::vector<std::string> imatrix_datasets;
    std::unordered_map<std::string, std::vector<float>> imatrix_data;
    const bool have_imatrix = !imatrix_file.empty();
    if (have_imatrix) {
    const int rc = sd_load_imatrix(imatrix_file, imatrix_datasets, imatrix_data);
        if (rc < 0) {
            fprintf(stderr, "seeddelta-build: failed to load imatrix %s\n", imatrix_file.c_str());
            return 1;
        }
    }

    const int64_t n_tensors = gguf_get_n_tensors(src);

    sd_policy policy;
    sd_policy * policy_ptr = nullptr;
    std::string policy_hash;
    sd_policy_state policy_state = sd_policy_load_from_file(policy_file, policy_strict);
    if (!policy_state.error.empty()) {
        fprintf(stderr, "seeddelta-build: %s\n", policy_state.error.c_str());
        return 1;
    }
    for (const auto & w : policy_state.warnings) {
        fprintf(stderr, "seeddelta-build: policy warning: %s\n", w.c_str());
    }
    if (policy_state.has_policy) {
        policy = policy_state.policy;
        policy_ptr = &policy;
        policy_hash = policy_state.hash;
    }

    // Discover number of layers from tensor names.
    int64_t max_layer_id = -1;
    std::regex re_layer(R"(blk\.(\d+)\.)");
    for (int64_t ti = 0; ti < n_tensors; ++ti) {
        const char * name = gguf_get_tensor_name(src, ti);
        std::cmatch m;
        if (std::regex_search(name, m, re_layer)) {
            max_layer_id = std::max<int64_t>(max_layer_id, std::stoll(m[1]));
        }
    }
    const int64_t n_layer = max_layer_id + 1;
    auto layers = sd_parse_layer_range(layers_range, n_layer);

    const int64_t K_default = std::max<int64_t>(1, K);
    const int64_t K_gate_eff = (K_gate > 0 ? K_gate : K_default);
    const int64_t K_up_eff   = (K_up   > 0 ? K_up   : K_default);
    const int64_t K_down_eff = (K_down > 0 ? K_down : K_default);
    const bool K_variable = (K_gate_eff != K_default) || (K_up_eff != K_default) || (K_down_eff != K_default);

    sd_build_result build_res = sd_build_layers(
            args,
            src,
            ctx_data,
            layers,
            scheme,
            idx_type,
            val_type,
            block,
            K_gate_eff,
            K_up_eff,
            K_down_eff,
            policy_ptr,
            policy_dump_resolved,
            overwrite_existing,
            have_imatrix,
            imatrix_data,
            imatrix_eps,
            imatrix_power,
            stack_cost_cap);

    if (!build_res.ok) {
        return 1;
    }

    sd_write_params wparams;
    wparams.src = src;
    wparams.ctx_data = ctx_data;
    wparams.scheme = scheme;
    wparams.idx_type = idx_type;
    wparams.val_type = val_type;
    wparams.block = block;
    wparams.K_default = K_default;
    wparams.K_gate_eff = K_gate_eff;
    wparams.K_up_eff = K_up_eff;
    wparams.K_down_eff = K_down_eff;
    wparams.K_variable = K_variable;
    wparams.write_row_scale = write_row_scale;
    wparams.write_base = write_base;
    wparams.strip_dense = strip_dense;
    wparams.any_strip = build_res.any_strip;
    wparams.strip_weights = &build_res.strip_weights;
    wparams.pending = &build_res.pending;
    wparams.sd_contexts = &build_res.sd_contexts;
    wparams.report = &build_res.report;
    wparams.stack_guard = &build_res.stack_guard;
    wparams.n_added = build_res.n_added;
    wparams.in_fname = in_fname;
    wparams.out_fname = out_fname;
    wparams.report_json = report_json;
    wparams.policy_file = policy_file;
    wparams.policy_hash = policy_hash;
    wparams.policy_export_file = policy_export_file;
    wparams.have_imatrix = have_imatrix;
    wparams.imatrix_file = imatrix_file;
    wparams.imatrix_eps = imatrix_eps;
    wparams.imatrix_power = imatrix_power;
    wparams.imatrix_datasets = imatrix_datasets;
    wparams.eval_cols = eval_cols;
    wparams.eval_x = eval_x;
    wparams.base_max_samples = base_max_samples;
    wparams.base_perm_trials = base_perm_trials;

    return sd_write_output(wparams);
}
