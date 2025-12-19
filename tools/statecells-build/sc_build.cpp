#include "common.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include "include/sc_cli.h"
#include "include/sc_eval.h"
#include "include/sc_imatrix.h"
#include "include/sc_build.h"
#include "include/sc_process.h"
#include "include/sc_report.h"
#include "include/sc_utils.h"

#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <numeric>
#include <random>
#include <regex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>



int sc_build(
        const sc_args & args,
        const sc_report_config & report_cfg,
        int imatrix_chunk_count,
        const std::unordered_map<std::string, std::vector<float>> & imatrix_data) {
    std::string in_fname = args.in_fname;
    std::string out_fname = args.out_fname;

    int64_t dict_M = args.dict_M;
    int dict_k = args.dict_k;
    float dict_eta = args.dict_eta;
    int dict_iters = args.dict_iters;
    std::string report_json = args.report_json;
    std::string imatrix_file = args.imatrix_file;
    float imatrix_eps = args.imatrix_eps;
    float imatrix_power = args.imatrix_power;
    int seed = args.seed;
    const bool has_imatrix = report_cfg.imatrix.enabled;
    const std::string imatrix_dataset = report_cfg.imatrix.dataset;

    if (in_fname.empty() || out_fname.empty()) {
        fprintf(stderr, "statecells-build: missing input or output filename\n");
        return 1;
    }

    const int64_t t0_total = ggml_time_us();

    const std::string src_fname = report_cfg.source.empty() ? in_fname : report_cfg.source;

    ggml_context * ctx_data = nullptr;
    gguf_init_params params = { false, &ctx_data };
    gguf_context * src = gguf_init_from_file(src_fname.c_str(), params);
    if (!src || !ctx_data) {
        fprintf(stderr, "failed to load %s\n", src_fname.c_str());
        return 1;
    }

    gguf_context * dst = gguf_init_empty();
    gguf_set_kv(dst, src);
    if (has_imatrix) {
        gguf_set_val_str(dst, "statecells.imatrix.file", imatrix_file.c_str());
        if (!imatrix_dataset.empty()) {
            gguf_set_val_str(dst, "statecells.imatrix.dataset", imatrix_dataset.c_str());
        }
        if (imatrix_chunk_count >= 0) {
            gguf_set_val_u32(dst, "statecells.imatrix.chunks", (uint32_t) imatrix_chunk_count);
        }
        gguf_set_val_f32(dst, "statecells.imatrix.eps", imatrix_eps);
        gguf_set_val_f32(dst, "statecells.imatrix.power", imatrix_power);
    }

    // Add original tensors.
    const int64_t n_tensors = gguf_get_n_tensors(src);
    for (int64_t ti = 0; ti < n_tensors; ++ti) {
        const char * name = gguf_get_tensor_name(src, ti);
        ggml_tensor * t = ggml_get_tensor(ctx_data, name);
        if (!t) {
            fprintf(stderr, "warning: missing tensor %s in ctx_data\n", name);
            continue;
        }
        gguf_add_tensor(dst, t);
    }

    std::mt19937 rng(seed);
    sc_process_result proc = sc_process_layers(args, src, ctx_data, dst, imatrix_data, rng);
    auto & sc_contexts = proc.sc_contexts;
    auto & report_rows = proc.report_rows;
    const int64_t n_added = proc.n_added;
    const int64_t n_sc_existing = proc.n_sc_existing;

    if (!proc.ok) {
        for (auto * c : sc_contexts) {
            ggml_free(c);
        }
        ggml_free(ctx_data);
        gguf_free(dst);
        gguf_free(src);
        return 1;
    }

    if (n_sc_existing + n_added == 0) {
        fprintf(stderr, "no FFN weights found for StateCells\n");
    } else if (n_added == 0) {
        fprintf(stderr, "no new StateCells tensors added (already present or out of range)\n");
    }

    gguf_set_val_bool(dst, "statecells.enabled", (n_sc_existing + n_added) > 0);
    gguf_set_val_u32(dst, "statecells.dict.M", (uint32_t) dict_M);
    gguf_set_val_u32(dst, "statecells.dict.k", (uint32_t) dict_k);
    gguf_set_val_f32(dst, "statecells.dict.eta", dict_eta);
    gguf_set_val_u32(dst, "statecells.dict.iters", (uint32_t) dict_iters);

    if (src_fname == out_fname && n_added == 0) {
        printf("no changes, keeping existing %s\n", out_fname.c_str());
    } else {
        printf("writing %s with %" PRId64 " new tensors\n", out_fname.c_str(), n_added);
        gguf_write_to_file(dst, out_fname.c_str(), false);
    }

    if (!report_json.empty()) {
        if (!sc_write_report_json(report_json, report_cfg, report_rows)) {
            fprintf(stderr, "failed to write report to %s\n", report_json.c_str());
        }
    }

    for (auto * c : sc_contexts) {
        ggml_free(c);
    }
    ggml_free(ctx_data);
    gguf_free(dst);
    gguf_free(src);

    {
        const double sec = double(ggml_time_us() - t0_total) / 1e6;
        fprintf(stderr, "statecells-build: done in %.1fs\n", sec);
    }

    return 0;
}
