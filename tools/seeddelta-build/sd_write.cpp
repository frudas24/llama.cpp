#include "include/sd_write.h"

#include <cstdio>
#include <string>
#include <utility>
#include <cinttypes>

#include "gguf.h"
#include "sd_constants.h"
#include "seeddelta_policy_export.h"

int sd_write_output(const sd_write_params & params) {
    const auto & report = *params.report;
    const auto & pending = *params.pending;

    if (params.n_added == 0) {
        fprintf(stderr, "no new SeedΔ tensors added\n");
    }

    gguf_context * dst = gguf_init_empty();
    gguf_set_kv(dst, params.src);

    const int64_t n_tensors = gguf_get_n_tensors(params.src);

    for (int64_t ti = 0; ti < n_tensors; ++ti) {
        const char * name = gguf_get_tensor_name(params.src, ti);
        if (params.any_strip && params.strip_weights && params.strip_weights->count(name) > 0) {
            continue;
        }
        ggml_tensor * t = ggml_get_tensor(params.ctx_data, name);
        if (!t) {
            fprintf(stderr, "warning: missing tensor %s in ctx_data\n", name);
            continue;
        }
        gguf_add_tensor(dst, t);
    }

    for (const auto & pts : pending) {
        for (ggml_tensor * t : pts.tensors) {
            gguf_add_tensor(dst, t);
        }
    }

    gguf_set_val_bool(dst, "seeddelta.enabled", params.n_added > 0);
    gguf_set_val_u32(dst, "seeddelta.version", 1);
    gguf_set_val_u32(dst, "seeddelta.scheme", (uint32_t) params.scheme);
    gguf_set_val_bool(dst, "seeddelta.row_scale", params.write_row_scale);
    gguf_set_val_u32(dst, "seeddelta.resid.K", (uint32_t) params.K_default);
    gguf_set_val_bool(dst, "seeddelta.strip_dense", params.any_strip);
    if (params.scheme == sd_resid_scheme::block) {
        gguf_set_val_u32(dst, "seeddelta.resid.block", (uint32_t) params.block);
    }
    gguf_set_val_bool(dst, "seeddelta.resid.K_variable", params.K_variable);
    if (params.K_variable) {
        gguf_set_val_u32(dst, "seeddelta.resid.K_gate", (uint32_t) params.K_gate_eff);
        gguf_set_val_u32(dst, "seeddelta.resid.K_up",   (uint32_t) params.K_up_eff);
        gguf_set_val_u32(dst, "seeddelta.resid.K_down", (uint32_t) params.K_down_eff);
    }
    gguf_set_val_bool(dst, "seeddelta.base.enabled", params.write_base);
    if (params.write_base) {
        gguf_set_val_u32(dst, "seeddelta.base.kind", 1);
        gguf_set_val_u32(dst, "seeddelta.base.depth", 2);
        gguf_set_val_u32(dst, "seeddelta.base.R", 1);
        gguf_set_val_u32(dst, "seeddelta.base.max_samples", (uint32_t) std::max<int64_t>(0, params.base_max_samples));
        gguf_set_val_u32(dst, "seeddelta.base.perm_trials", (uint32_t) std::max(1, params.base_perm_trials));
    }

    for (const auto & re : report) {
        const std::string prefix = "seeddelta.blk." + std::to_string(re.layer) + "." + re.kind;
        gguf_set_val_bool(dst, (prefix + ".enabled").c_str(), re.emit);
        gguf_set_val_bool(dst, (prefix + ".gating_pass").c_str(), re.gating_pass);
        gguf_set_val_bool(dst, (prefix + ".strip_dense").c_str(), re.strip_applied);
        gguf_set_val_u32(dst, (prefix + ".K").c_str(), (uint32_t) std::max<int64_t>(0, re.K));
    }

    printf("writing %s with %" PRId64 " new tensors\n", params.out_fname.c_str(), params.n_added);
    if (!gguf_write_to_file(dst, params.out_fname.c_str(), false)) {
        fprintf(stderr, "seeddelta-build: failed to write %s\n", params.out_fname.c_str());
        return 1;
    }

    if (!params.report_json.empty()) {
        sd_report::report_config cfg;
        cfg.input = params.in_fname;
        cfg.output = params.out_fname;
        cfg.write_base = params.write_base;
        cfg.base_max_samples = params.base_max_samples;
        cfg.base_perm_trials = params.base_perm_trials;
        cfg.policy_file = params.policy_file;
        cfg.policy_hash = params.policy_hash;
        cfg.have_imatrix = params.have_imatrix;
        cfg.imatrix_file = params.imatrix_file;
        cfg.imatrix_eps = params.imatrix_eps;
        cfg.imatrix_power = params.imatrix_power;
        cfg.imatrix_datasets = params.imatrix_datasets;
        cfg.resid.scheme = (params.scheme == sd_resid_scheme::block) ? sd_report::seeddelta_scheme::block : sd_report::seeddelta_scheme::coo;
        cfg.resid.block = params.block;
        cfg.resid.K = params.K_default;
        cfg.resid.K_gate = params.K_gate_eff;
        cfg.resid.K_up = params.K_up_eff;
        cfg.resid.K_down = params.K_down_eff;
        cfg.resid.idx_type = ggml_type_name(params.idx_type);
        cfg.resid.val_type = ggml_type_name(params.val_type);
        cfg.resid.row_scale = params.write_row_scale;
        cfg.eval_cols = params.eval_cols;
        cfg.eval_x = params.eval_x;
        cfg.stack_pass_gate_up = params.stack_guard ? params.stack_guard->gate_up_pass : 0;
        cfg.stack_pass_down = params.stack_guard ? params.stack_guard->down_pass : 0;
        if (!sd_report::write_report_json(params.report_json, cfg, report)) {
            return 1;
        }
    }

    if (!params.policy_export_file.empty()) {
        std::vector<sd_tensor_decision> decisions;
        decisions.reserve(report.size());
        for (const auto & e : report) {
            sd_tensor_decision d;
            d.layer = e.layer;
            d.kind = e.kind;
            d.enabled = e.emit;
            d.strip_dense = e.strip_applied;
            d.block = e.block;
            d.K_budget = e.K_budget;
            decisions.push_back(std::move(d));
        }

        auto pres = sd_policy_export_write_canonical(params.policy_export_file, decisions);
        if (!pres.ok) {
            fprintf(stderr, "seeddelta-build: failed to export policy: %s\n", pres.error.c_str());
            return 1;
        }
        fprintf(stderr, "seeddelta-build: wrote policy export %s\n", params.policy_export_file.c_str());
    }

    return 0;
}
