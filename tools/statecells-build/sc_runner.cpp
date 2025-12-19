#include "include/sc_build.h"
#include "include/sc_cli.h"
#include "include/sc_imatrix.h"
#include "include/sc_report.h"
#include "include/sc_runner.h"

#include <cstdio>
#include <fstream>
#include <unordered_map>
#include <vector>

static bool file_exists(const std::string & path) {
    std::ifstream ifs(path);
    return ifs.good();
}

int sc_run(const sc_args & args) {
    sc_args run_args = args;
    if (!run_args.report_json.empty() && run_args.eval_cols <= 0) {
        run_args.eval_cols = 64;
    }

    std::vector<std::string> imatrix_datasets;
    std::unordered_map<std::string, std::vector<float>> imatrix_data;
    int imatrix_chunk_count = -1;
    if (!run_args.imatrix_file.empty()) {
        imatrix_chunk_count = load_imatrix(run_args.imatrix_file, imatrix_datasets, imatrix_data);
        if (imatrix_chunk_count < 0 || imatrix_data.empty()) {
            fprintf(stderr, "statecells-build: failed to load imatrix from %s\n", run_args.imatrix_file.c_str());
            return 1;
        }
        fprintf(stderr, "statecells-build: using imatrix (%zu entries)\n", imatrix_data.size());
    }

    std::string src_fname = run_args.in_fname;
    if (run_args.resume) {
        if (file_exists(run_args.out_fname)) {
            fprintf(stderr, "statecells-build: resuming from existing %s\n", run_args.out_fname.c_str());
            src_fname = run_args.out_fname;
        } else {
            fprintf(stderr, "statecells-build: --resume requested but %s not found, starting from input\n", run_args.out_fname.c_str());
        }
    }

    sc_report_config report_cfg;
    report_cfg.input = run_args.in_fname;
    report_cfg.source = src_fname;
    report_cfg.output = run_args.out_fname;
    report_cfg.resume = run_args.resume;
    report_cfg.write_vals = run_args.write_vals;
    report_cfg.write_row_scale = run_args.write_row_scale;
    report_cfg.imatrix.enabled = !run_args.imatrix_file.empty();
    report_cfg.imatrix.file = run_args.imatrix_file;
    if (!imatrix_datasets.empty()) {
        report_cfg.imatrix.dataset = imatrix_datasets[0];
    }
    report_cfg.imatrix.chunks = imatrix_chunk_count;
    report_cfg.imatrix.eps = run_args.imatrix_eps;
    report_cfg.imatrix.power = run_args.imatrix_power;
    report_cfg.dict.M = run_args.dict_M;
    report_cfg.dict.k = run_args.dict_k;
    report_cfg.dict.eta = run_args.dict_eta;
    report_cfg.dict.iters = run_args.dict_iters;
    report_cfg.dict.max_samples = run_args.dict_max_samples;
    report_cfg.eval_cols = run_args.eval_cols;

    return sc_build(run_args, report_cfg, imatrix_chunk_count, imatrix_data);
}
