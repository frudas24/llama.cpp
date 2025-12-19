#include "common.h"
#include "ggml.h"
#include "gguf.h"
#include "seeddelta_policy.h"
#include "seeddelta_policy_export.h"
#include "seeddelta_policy_selftest.h"
#include "sd_runner.h"
#include "sd_cli.h"
#include "sd_cost.h"
#include "sd_eval.h"
#include "sd_ffn_proxy.h"
#include "sd_report.h"
#include "sd_constants.h"

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

struct stack_safety_tracker {
    int gate_up_pass = 0;
    int down_pass = 0;

    void adjust(const std::string & kind, float & min_mean, float & min_p05) const {
        const bool is_down = (kind == "ffn_down");
        const int passed = is_down ? down_pass : gate_up_pass;
        if (passed >= 8) {
            if (is_down) {
                min_mean = std::max(min_mean, 0.35f);
                min_p05  = std::max(min_p05, 0.25f);
            } else {
                min_mean = std::max(min_mean, 0.60f);
                min_p05  = std::max(min_p05, 0.45f);
            }
        } else if (passed >= 3) {
            min_mean += 0.05f;
            min_p05  += 0.05f;
        }
    }

    void record_pass(const std::string & kind) {
        if (kind == "ffn_down") {
            ++down_pass;
        } else {
            ++gate_up_pass;
        }
    }
};

static std::vector<int64_t> parse_layer_range(const std::string & s, int64_t n_layer) {
    if (s.empty()) {
        std::vector<int64_t> out(n_layer);
        std::iota(out.begin(), out.end(), 0);
        return out;
    }

    std::regex re(R"(^\s*(\d+)\s*-\s*(\d+)\s*$)");
    std::smatch m;
    if (!std::regex_match(s, m, re)) {
        throw std::runtime_error("invalid --layers, expected A-B");
    }

    int64_t a = std::stoll(m[1]);
    int64_t b = std::stoll(m[2]);
    if (a > b) std::swap(a, b);
    a = std::max<int64_t>(0, a);
    b = std::min<int64_t>(n_layer - 1, b);
    std::vector<int64_t> out;
    for (int64_t i = a; i <= b; ++i) out.push_back(i);
    return out;
}

static bool string_remove_suffix(std::string & s, const std::string & suffix) {
    if (s.size() < suffix.size()) {
        return false;
    }
    if (s.compare(s.size() - suffix.size(), suffix.size(), suffix) != 0) {
        return false;
    }
    s.resize(s.size() - suffix.size());
    return true;
}

static int load_legacy_imatrix(
        const std::string & imatrix_file,
        std::vector<std::string> & imatrix_datasets,
        std::unordered_map<std::string, std::vector<float>> & imatrix_data) {
    std::ifstream in(imatrix_file.c_str(), std::ios::binary);
    if (!in) {
        fprintf(stderr, "%s: failed to open %s\n", __func__, imatrix_file.c_str());
        return -1;
    }

    int32_t n_entries = 0;
    in.read((char *) &n_entries, sizeof(n_entries));
    if (in.fail() || n_entries <= 0) {
        fprintf(stderr, "%s: no data in file %s\n", __func__, imatrix_file.c_str());
        return -1;
    }

    imatrix_data.clear();
    imatrix_data.reserve((size_t) n_entries);

    for (int i = 0; i < n_entries; i++) {
        int32_t len = 0;
        in.read((char *) &len, sizeof(len));
        if (in.fail() || len <= 0) {
            fprintf(stderr, "%s: failed reading name for entry %d from %s\n", __func__, i + 1, imatrix_file.c_str());
            return -1;
        }

        std::vector<char> name_as_vec((size_t) len);
        in.read(name_as_vec.data(), len);
        if (in.fail()) {
            fprintf(stderr, "%s: failed reading name for entry %d from %s\n", __func__, i + 1, imatrix_file.c_str());
            return -1;
        }

        std::string name(name_as_vec.begin(), name_as_vec.end());

        int32_t ncall = 0;
        in.read((char *) &ncall, sizeof(ncall));
        if (in.fail() || ncall <= 0) {
            fprintf(stderr, "%s: invalid ncall %d for entry %s\n", __func__, ncall, name.c_str());
            return -1;
        }

        int32_t nval = 0;
        in.read((char *) &nval, sizeof(nval));
        if (in.fail() || nval <= 0) {
            fprintf(stderr, "%s: invalid nval %d for entry %s\n", __func__, nval, name.c_str());
            return -1;
        }

        auto & e = imatrix_data[name];
        e.resize((size_t) nval);
        in.read((char *) e.data(), (size_t) nval * sizeof(float));
        if (in.fail()) {
            fprintf(stderr, "%s: failed reading data for entry %s\n", __func__, name.c_str());
            return -1;
        }
    }

    int m_last_call = 0;
    if (in.peek() != EOF) {
        in.read((char *) &m_last_call, sizeof(m_last_call));
        int dataset_len = 0;
        in.read((char *) &dataset_len, sizeof(dataset_len));
        if (!in.fail() && dataset_len > 0) {
            std::vector<char> dataset_as_vec((size_t) dataset_len);
            in.read(dataset_as_vec.data(), dataset_len);
            if (!in.fail()) {
                imatrix_datasets.resize(1);
                imatrix_datasets[0].assign(dataset_as_vec.begin(), dataset_as_vec.end());
                fprintf(stderr, "%s: imatrix dataset='%s'\n", __func__, imatrix_datasets[0].c_str());
            }
        }
    }

    fprintf(stderr, "%s: loaded %d importance matrix entries from %s computed on %d chunks\n",
            __func__, int(imatrix_data.size()), imatrix_file.c_str(), m_last_call);

    return m_last_call;
}

// Loads an imatrix GGUF file produced by llama-imatrix.
// Data format matches tools/quantize loader: tensors are stored as <name>.in_sum2 and <name>.counts.
static int load_imatrix(
        const std::string & imatrix_file,
        std::vector<std::string> & imatrix_datasets,
        std::unordered_map<std::string, std::vector<float>> & imatrix_data) {
    static const char * const LLM_KV_IMATRIX_DATASETS    = "imatrix.datasets";
    static const char * const LLM_KV_IMATRIX_CHUNK_COUNT = "imatrix.chunk_count";
    static const char * const LLM_KV_IMATRIX_CHUNK_SIZE  = "imatrix.chunk_size";

    struct ggml_context * ctx = nullptr;
    struct gguf_init_params meta_gguf_params = {
        /* .no_alloc = */ false, // the data is needed
        /* .ctx      = */ &ctx,
    };

    struct gguf_context * ctx_gguf = gguf_init_from_file(imatrix_file.c_str(), meta_gguf_params);
    if (!ctx_gguf) {
        fprintf(stderr, "%s: imatrix file '%s' is using old format\n", __func__, imatrix_file.c_str());
        return load_legacy_imatrix(imatrix_file, imatrix_datasets, imatrix_data);
    }

    const int32_t n_entries = gguf_get_n_tensors(ctx_gguf);
    if (n_entries < 1) {
        fprintf(stderr, "%s: no data in file %s\n", __func__, imatrix_file.c_str());
        gguf_free(ctx_gguf);
        ggml_free(ctx);
        return -1;
    }

    const int dataset_idx     = gguf_find_key(ctx_gguf, LLM_KV_IMATRIX_DATASETS);
    const int chunk_count_idx = gguf_find_key(ctx_gguf, LLM_KV_IMATRIX_CHUNK_COUNT);
    const int chunk_size_idx  = gguf_find_key(ctx_gguf, LLM_KV_IMATRIX_CHUNK_SIZE);
    if (dataset_idx < 0 || chunk_count_idx < 0 || chunk_size_idx < 0) {
        fprintf(stderr, "%s: missing imatrix metadata in file %s\n", __func__, imatrix_file.c_str());
        gguf_free(ctx_gguf);
        ggml_free(ctx);
        return -1;
    }

    const uint32_t chunk_size = gguf_get_val_u32(ctx_gguf, chunk_size_idx);
    GGML_UNUSED(chunk_size);

    const std::string sums_suffix{ ".in_sum2" };
    const std::string counts_suffix{ ".counts" };

    std::map<std::string, std::pair<struct ggml_tensor *, struct ggml_tensor *>> sums_counts_for;

    for (struct ggml_tensor * cur = ggml_get_first_tensor(ctx); cur; cur = ggml_get_next_tensor(ctx, cur)) {
        std::string name = cur->name;

        if (name.empty()) {
            continue;
        }

        if (string_remove_suffix(name, sums_suffix)) {
            sums_counts_for[std::move(name)].first = cur;
        } else if (string_remove_suffix(name, counts_suffix)) {
            sums_counts_for[std::move(name)].second = cur;
        }
    }

    imatrix_data.clear();
    imatrix_data.reserve(sums_counts_for.size());

    for (const auto & sc : sums_counts_for) {
        const        std::string & name   = sc.first;
        const struct ggml_tensor * sums   = sc.second.first;
        const struct ggml_tensor * counts = sc.second.second;

        if (!sums || !counts) {
            fprintf(stderr, "%s: mismatched sums and counts for %s\n", __func__, name.c_str());
            gguf_free(ctx_gguf);
            ggml_free(ctx);
            return -1;
        }

        const int64_t ne0 = sums->ne[0];
        const int64_t ne1 = sums->ne[1];

        auto & e = imatrix_data[name];
        e.resize((size_t) ggml_nelements(sums));

        for (int64_t j = 0; j < ne1; ++j) {
            const float count = ((const float *) counts->data)[j];
            if (count > 0.0f) {
                for (int64_t i = 0; i < ne0; ++i) {
                    e[(size_t) j * (size_t) ne0 + (size_t) i] = ((const float *) sums->data)[(size_t) j * (size_t) ne0 + (size_t) i] / count;
                }
            } else {
                // Partial imatrix data, tensor never got any input during calibration.
                for (int64_t i = 0; i < ne0; ++i) {
                    e[(size_t) j * (size_t) ne0 + (size_t) i] = 1.0f;
                }
            }
        }
    }

    const int m_last_chunk = (int) gguf_get_val_u32(ctx_gguf, chunk_count_idx);

    const int64_t n_datasets = gguf_get_arr_n(ctx_gguf, dataset_idx);
    imatrix_datasets.clear();
    imatrix_datasets.reserve((size_t) n_datasets);
    for (int64_t i = 0; i < n_datasets; ++i) {
        imatrix_datasets.push_back(gguf_get_arr_str(ctx_gguf, dataset_idx, i));
    }
    if (!imatrix_datasets.empty()) {
        fprintf(stderr, "%s: imatrix datasets=['%s'", __func__, imatrix_datasets[0].c_str());
        for (size_t i = 1; i < imatrix_datasets.size(); ++i) {
            fprintf(stderr, ", '%s'", imatrix_datasets[i].c_str());
        }
        fprintf(stderr, "]\n");
    }

    fprintf(stderr, "%s: loaded %d importance matrix entries from %s computed on %d chunks\n",
            __func__, int(imatrix_data.size()), imatrix_file.c_str(), m_last_chunk);

    gguf_free(ctx_gguf);
    ggml_free(ctx);

    return m_last_chunk;
}

static bool make_imatrix_sqrt_scale(
        const std::unordered_map<std::string, std::vector<float>> & imatrix_data,
        const std::string & weight_name,
        int64_t n_in,
        float eps,
        float power,
        std::vector<float> & scale_out) {
    const auto it = imatrix_data.find(weight_name);
    if (it == imatrix_data.end()) {
        return false;
    }

    if ((int64_t) it->second.size() < n_in) {
        fprintf(stderr, "seeddelta-build: imatrix entry %s has %" PRId64 " values, expected >= %" PRId64 "\n",
                weight_name.c_str(), (int64_t) it->second.size(), n_in);
        return false;
    }

    eps = std::max(eps, 0.0f);
    power = std::max(power, 0.0f);

    scale_out.resize((size_t) n_in);
    double sum = 0.0;
    for (int64_t i = 0; i < n_in; ++i) {
        float v = it->second[(size_t) i];
        if (!std::isfinite(v) || v < eps) {
            v = eps;
        }

        float s = 1.0f;
        if (power == 0.0f) {
            s = 1.0f;
        } else if (power == 1.0f) {
            s = std::sqrt(v);
        } else {
            s = std::pow(v, 0.5f * power);
        }

        if (!std::isfinite(s) || s <= 0.0f) {
            s = 1.0f;
        }

        scale_out[(size_t) i] = s;
        sum += (double) s;
    }

    const double mean = sum / std::max<double>(1.0, (double) n_in);
    if (mean > 0.0) {
        const float inv_mean = (float) (1.0 / mean);
        for (float & s : scale_out) {
            s *= inv_mean;
        }
    }

    return true;
}

struct pending_tensor_set {
    ggml_context * ctx = nullptr;
    std::vector<ggml_tensor *> tensors;
};

static std::string fnv1a_hex(const std::string & data) {
    uint64_t hash = 1469598103934665603ULL; // FNV-1a 64-bit offset
    for (unsigned char c : data) {
        hash ^= c;
        hash *= 1099511628211ULL;
    }
    std::ostringstream oss;
    oss << std::hex;
    oss.width(16);
    oss.fill('0');
    oss << hash;
    return oss.str();
}

static std::string slurp_file(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return {};
    }
    std::ostringstream oss;
    oss << in.rdbuf();
    return oss.str();
}

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
    const int n_threads = args.n_threads;
    int64_t eval_cols = args.eval_cols;
    const int64_t eval_x = args.eval_x;
    const float imatrix_eps = args.imatrix_eps;
    const float imatrix_power = args.imatrix_power;
    const int seed = args.seed;
    const bool policy_strict = args.policy_strict;
    const bool policy_dump_resolved = args.policy_dump_resolved;
    const bool policy_self_test = args.policy_self_test;
    const bool overwrite_existing = args.overwrite_existing;
    const double stack_cost_cap = args.stack_cost_cap;

    if (policy_self_test) {
        return sd_policy_self_test();
    }

    enum resid_scheme {
        RESID_COO   = 0,
        RESID_BLOCK = 1,
    };

    resid_scheme scheme = RESID_COO;
    if (scheme_str == "coo") {
        scheme = RESID_COO;
    } else if (scheme_str == "block") {
        scheme = RESID_BLOCK;
    } else {
        throw std::runtime_error("invalid --scheme (expected: coo|block)");
    }
    if (scheme == RESID_BLOCK) {
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
        const int rc = load_imatrix(imatrix_file, imatrix_datasets, imatrix_data);
        if (rc < 0) {
            fprintf(stderr, "seeddelta-build: failed to load imatrix %s\n", imatrix_file.c_str());
            return 1;
        }
    }

    const int64_t n_tensors = gguf_get_n_tensors(src);

    sd_policy policy;
    sd_policy * policy_ptr = nullptr;
    std::string policy_hash;
    if (!policy_file.empty()) {
        auto pres = sd_policy_load(policy_file, policy_strict, policy);
        if (!pres.ok) {
            fprintf(stderr, "seeddelta-build: %s\n", pres.error.c_str());
            return 1;
        }
        for (const auto & w : pres.warnings) {
            fprintf(stderr, "seeddelta-build: policy warning: %s\n", w.c_str());
        }
        policy_ptr = &policy;
        const std::string policy_bytes = slurp_file(policy_file);
        if (!policy_bytes.empty()) {
            policy_hash = "fnv1a64:" + fnv1a_hex(policy_bytes);
        }
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
    auto layers = parse_layer_range(layers_range, n_layer);

    const std::vector<std::string> kinds = { "ffn_gate", "ffn_up", "ffn_down" };

    std::mt19937 rng(seed);

    // Keep per-weight ggml contexts alive until we write the output.
    std::vector<ggml_context *> sd_contexts;

    int64_t n_added = 0;
    std::vector<report_entry> report;
    std::vector<pending_tensor_set> pending;
    std::unordered_set<std::string> strip_weights;
    bool any_strip = false;
    double stack_cost_running = 0.0;

    const int64_t K_default = std::max<int64_t>(1, K);
    const int64_t K_gate_eff = (K_gate > 0 ? K_gate : K_default);
    const int64_t K_up_eff   = (K_up   > 0 ? K_up   : K_default);
    const int64_t K_down_eff = (K_down > 0 ? K_down : K_default);
    const bool K_variable = (K_gate_eff != K_default) || (K_up_eff != K_default) || (K_down_eff != K_default);

    sd_resolved_tensor baseline_cfg;
    baseline_cfg.block = block;
    baseline_cfg.K_gate = K_gate_eff;
    baseline_cfg.K_up   = K_up_eff;
    baseline_cfg.K_down = K_down_eff;
    baseline_cfg.strip_dense = strip_dense;
    baseline_cfg.enabled = true;
    baseline_cfg.metric = sd_metric_kind::cos_x_w;
    baseline_cfg.min_mean = 0.0f;
    baseline_cfg.min_p05  = 0.0f;
    baseline_cfg.gating_enabled = (policy_ptr != nullptr);

    if (policy_ptr && eval_cols <= 0) {
        fprintf(stderr, "seeddelta-build: --policy requires --eval-cols > 0 for gating/autotune\n");
        return 1;
    }

    stack_safety_tracker stack_guard;

    for (const int64_t il : layers) {
        for (const auto & kind : kinds) {
            const std::string weight_name = "blk." + std::to_string(il) + "." + kind + ".weight";
            const bool have_weight = gguf_find_tensor(src, weight_name.c_str()) != -1;

            sd_resolved_tensor baseline_kind = baseline_cfg;
            if (kind == "ffn_down") {
                baseline_kind.min_mean = 0.25f;
                baseline_kind.min_p05  = 0.15f;
            } else {
                baseline_kind.min_mean = 0.45f;
                baseline_kind.min_p05  = 0.30f;
            }

            sd_resolved_tensor cfg = sd_policy_resolve(policy_ptr, il, kind, baseline_kind);
            if (!cfg.enabled) {
                // still record later in report as disabled
            }
            if (policy_dump_resolved) {
                fprintf(stderr, "policy: blk.%" PRId64 ".%s enabled=%d strip=%d block=%" PRId64 " K(g/u/d)=%" PRId64 "/%" PRId64 "/%" PRId64 " metric=%s min_mean=%.3f min_p05=%.3f\n",
                        il, kind.c_str(), cfg.enabled ? 1 : 0, cfg.strip_dense ? 1 : 0, cfg.block,
                        cfg.K_gate, cfg.K_up, cfg.K_down,
                        sd_report::metric_kind_to_string(cfg.metric).c_str(), cfg.min_mean, cfg.min_p05);
            }

            float gating_min_mean = cfg.min_mean;
            float gating_min_p05  = cfg.min_p05;
            if (cfg.gating_enabled) {
                stack_guard.adjust(kind, gating_min_mean, gating_min_p05);
            }

            if (cfg.require_eval_x && (eval_x <= 0 || eval_cols <= 0)) {
                fprintf(stderr, "seeddelta-build: gating metric requires --eval-x and --eval-cols > 0 for %s\n", weight_name.c_str());
                return 1;
            }
            if (cfg.require_imatrix && !have_imatrix) {
                fprintf(stderr, "seeddelta-build: gating metric requires imatrix for %s\n", weight_name.c_str());
                return 1;
            }

            if (!have_weight) {
                if (policy_ptr || !report_json.empty()) {
                    report_entry re;
                    re.layer = il;
                    re.kind = kind;
                    re.n_in = 0;
                    re.n_out = 0;
                    re.emit = false;
                    re.strip_applied = false;
                    re.gating_enabled = (policy_ptr != nullptr);
                    re.gating_pass = false;
                    re.decision_reason = sd_constants::decision_missing_tensor;
                    re.gating_metric_used = sd_report::metric_kind_to_string(cfg.metric);
                    re.metric_used = re.gating_metric_used;
                    re.gating_min_mean = gating_min_mean;
                    re.gating_min_p05 = gating_min_p05;
                    re.target_tau_mean = gating_min_mean;
                    re.target_tau_p05 = gating_min_p05;
                    re.reject_reason = re.decision_reason;
                    report.push_back(std::move(re));
                }
                continue;
            }

            const std::string d_idx_name      = "blk." + std::to_string(il) + "." + kind + ".d_idx";
            const std::string d_val_name      = "blk." + std::to_string(il) + "." + kind + ".d_val";
            const std::string d_row_scale_name = "blk." + std::to_string(il) + "." + kind + ".d_row_scale";
            const std::string b_idx_name      = "blk." + std::to_string(il) + "." + kind + ".b_idx";
            const std::string b_val_name      = "blk." + std::to_string(il) + "." + kind + ".b_val";

            if (!overwrite_existing && (gguf_find_tensor(src, d_idx_name.c_str()) != -1 || gguf_find_tensor(src, d_val_name.c_str()) != -1 ||
                gguf_find_tensor(src, b_idx_name.c_str()) != -1 || gguf_find_tensor(src, b_val_name.c_str()) != -1)) {
                fprintf(stderr, "seeddelta-build: %s already has delta tensors, skipping (use --overwrite-existing to rebuild)\n", weight_name.c_str());
                continue;
            }

            ggml_tensor * W = ggml_get_tensor(ctx_data, weight_name.c_str());
            if (!W || ggml_n_dims(W) != 2) {
                continue;
            }

            if (!cfg.enabled) {
                report_entry re;
                re.layer = il;
                re.kind = kind;
                re.n_in = W->ne[0];
                re.n_out = W->ne[1];
                re.emit = false;
                re.strip_applied = false;
                re.gating_enabled = (policy_ptr != nullptr);
                re.gating_pass = false;
                re.decision_reason = sd_constants::decision_disabled;
                re.gating_metric_used = sd_report::metric_kind_to_string(cfg.metric);
                re.metric_used = re.gating_metric_used;
                re.gating_min_mean = gating_min_mean;
                re.gating_min_p05 = gating_min_p05;
                re.target_tau_mean = gating_min_mean;
                re.target_tau_p05 = gating_min_p05;
                re.reject_reason = re.decision_reason;
                report.push_back(std::move(re));
                continue;
            }

            const int64_t n_in  = W->ne[0];
            const int64_t n_out = W->ne[1];
            const int64_t K_kind =
                    kind == "ffn_gate" ? cfg.K_gate :
                    kind == "ffn_up"   ? cfg.K_up   :
                                        cfg.K_down;
            const int64_t K_budget = std::max<int64_t>(1, std::min<int64_t>(K_kind, n_in));
            const bool is_tall = n_out >= n_in;
            const int64_t block_here = (scheme == RESID_BLOCK) ? cfg.block : block;
            const int64_t block = block_here; // shadow global block for per-tensor overrides

            std::vector<float> w_scale;
            const bool have_w = have_imatrix && make_imatrix_sqrt_scale(imatrix_data, weight_name, n_in, imatrix_eps, imatrix_power, w_scale);
            if (cfg.require_imatrix && !have_w) {
                report_entry re;
                re.layer = il;
                re.kind = kind;
                re.n_in = n_in;
                re.n_out = n_out;
                re.emit = false;
                re.gating_enabled = (policy_ptr != nullptr);
                re.gating_pass = false;
                re.decision_reason = sd_constants::decision_missing_imatrix;
                re.gating_metric_used = sd_report::metric_kind_to_string(cfg.metric);
                re.metric_used = re.gating_metric_used;
                re.gating_min_mean = gating_min_mean;
                re.gating_min_p05 = gating_min_p05;
                re.target_tau_mean = gating_min_mean;
                re.target_tau_p05 = gating_min_p05;
                re.reject_reason = re.decision_reason;
                report.push_back(std::move(re));
                continue;
            }

            base_fit base;
            if (write_base) {
                const int64_t t0 = ggml_time_us();
                base = fit_base_xor_circulant(W, base_max_samples, have_w ? &w_scale : nullptr, base_perm_trials, rng);
                const double sec = double(ggml_time_us() - t0) / 1e6;
                fprintf(stderr, "  [blk.%" PRId64 ".%s] base fit kind=xor_circulant L=%" PRId64 " B=%" PRId64 " depth=2 samples=%" PRId64 " perm_trials=%d%s (%.1fs)\n",
                        il, kind.c_str(), base.L, base.B, base_max_samples, base_perm_trials, have_w ? " imatrix=on" : "", sec);
            }

            struct trial_data {
                report_entry re;
                std::vector<int32_t> d_idx;
                std::vector<float>   d_val;
                std::vector<int32_t> b_idx;
                std::vector<float>   b_val;
                std::vector<float>   d_row_scale;
                double seconds = 0.0;
            };

            auto run_trial = [&](int64_t K_budget_trial) -> trial_data {
                trial_data t;

                const int64_t t0_total = ggml_time_us();
                const int64_t K_budget_clamped = std::max<int64_t>(1, std::min<int64_t>(K_budget_trial, n_in));

                int64_t n_blocks_keep = 0;
                int64_t K_eff = K_budget_clamped;
                if (scheme == RESID_BLOCK) {
                    const int64_t n_blocks_total = (n_in + block - 1) / block;
                    n_blocks_keep = std::max<int64_t>(1, std::min<int64_t>((K_budget_clamped + block - 1) / block, n_blocks_total));
                    K_eff = n_blocks_keep * block;
                }

                if (scheme == RESID_BLOCK) {
                    printf("seeddelta-build: layer %" PRId64 " %s [% " PRId64 " x %" PRId64 "] type=%s scheme=block block=%" PRId64 " nb=%" PRId64 " K=%" PRId64 " (budget=%" PRId64 ")%s\n",
                           il, kind.c_str(), n_in, n_out, ggml_type_name(W->type), block, n_blocks_keep, K_eff, K_budget_clamped,
                           have_w ? " imatrix=on" : (have_imatrix ? " imatrix=missing" : ""));
                } else {
                    printf("seeddelta-build: layer %" PRId64 " %s [% " PRId64 " x %" PRId64 "] type=%s scheme=coo K=%" PRId64 "%s\n",
                           il, kind.c_str(), n_in, n_out, ggml_type_name(W->type), K_eff,
                           have_w ? " imatrix=on" : (have_imatrix ? " imatrix=missing" : ""));
                }

                t.re.layer = il;
                t.re.kind = kind;
                t.re.n_in = n_in;
                t.re.n_out = n_out;
                t.re.K_budget = K_budget_clamped;
                t.re.K = K_eff;
                t.re.block = (scheme == RESID_BLOCK) ? block : 0;
                t.re.n_blocks = (scheme == RESID_BLOCK) ? n_blocks_keep : 0;
                t.re.has_w = have_w;
                t.re.cost = estimate_cost(write_base ? &base : nullptr, n_in, n_out, K_eff, write_row_scale);

                if (scheme == RESID_BLOCK) {
                    GGML_ASSERT(n_blocks_keep > 0);
                    t.b_idx.assign((size_t) n_blocks_keep * (size_t) n_out, -1);
                    t.b_val.assign((size_t) block * (size_t) n_blocks_keep * (size_t) n_out, 0.0f);
                } else {
                    t.d_idx.assign((size_t) K_eff * (size_t) n_out, -1);
                    t.d_val.assign((size_t) K_eff * (size_t) n_out, 0.0f);
                }
                if (write_row_scale) {
                    t.d_row_scale.resize((size_t) n_out, 1.0f);
                }

                std::atomic<int64_t> next_col{0};
                std::vector<std::thread> workers;
                workers.reserve((size_t) n_threads);

                const int64_t chunk = 16;

                for (int ti = 0; ti < n_threads; ++ti) {
                    workers.emplace_back([&, ti]() {
                        GGML_UNUSED(ti);

                        std::vector<float> w;
                        std::vector<float> r;
                        std::vector<int32_t> topk;
                        std::vector<int32_t> top_blocks;
                        while (true) {
                            const int64_t col0 = next_col.fetch_add(chunk);
                            if (col0 >= n_out) {
                                break;
                            }

                            const int64_t col1 = std::min<int64_t>(n_out, col0 + chunk);
                            for (int64_t col = col0; col < col1; ++col) {
                                read_column_f32(W, col, w);

                                double dot = 0.0;
                                double nn  = 0.0;

                                if (write_base) {
                                    r.resize((size_t) n_in);

                                    const int64_t L = base.L;
                                    GGML_ASSERT(L > 0);

                                    if (is_tall) {
                                        const int64_t b = col / L;
                                        const int64_t p = col - b * L;
                                        const float * h2 = base.h2.data() + (size_t) b * (size_t) L;
                                        const int32_t * inv = base.perm1_inv.empty() ? nullptr : base.perm1_inv.data() + (size_t) b * (size_t) L;

                                        for (int64_t i = 0; i < n_in; ++i) {
                                            const int64_t j = inv ? (int64_t) inv[(size_t) i] : i;
                                            const int64_t u = (p ^ j) & (L - 1);
                                            const float base_w = h2[(size_t) u];
                                            r[(size_t) i] = w[(size_t) i] - base_w;

                                            if (write_row_scale) {
                                                const double ws = have_w ? (double) w_scale[(size_t) i] : 1.0;
                                                dot += (double) w[(size_t) i] * (double) base_w * ws * ws;
                                                nn  += (double) base_w * (double) base_w * ws * ws;
                                            }
                                        }
                                    } else {
                                        const int64_t p = col;

                                        for (int64_t b = 0; b < base.B; ++b) {
                                            const float * h2 = base.h2.data() + (size_t) b * (size_t) L;
                                            const int32_t * inv = base.perm1_inv.empty() ? nullptr : base.perm1_inv.data() + (size_t) b * (size_t) L;
                                            const int64_t in0 = b * L;
                                            const int64_t in1 = std::min<int64_t>(n_in, in0 + L);
                                            for (int64_t i = in0; i < in1; ++i) {
                                                const int64_t q = i - in0;
                                                const int64_t j = inv ? (int64_t) inv[(size_t) q] : q;
                                                const int64_t u = (p ^ j) & (L - 1);
                                                const float base_w = h2[(size_t) u];
                                                r[(size_t) i] = w[(size_t) i] - base_w;

                                                if (write_row_scale) {
                                                    const double ws = have_w ? (double) w_scale[(size_t) i] : 1.0;
                                                    dot += (double) w[(size_t) i] * (double) base_w * ws * ws;
                                                    nn  += (double) base_w * (double) base_w * ws * ws;
                                                }
                                            }
                                        }
                                    }
                                }

                                if (scheme == RESID_BLOCK) {
                                    const std::vector<float> & src = write_base ? r : w;
                                    topk_blocks_energy_weighted(src, have_w ? &w_scale : nullptr, block, n_blocks_keep, top_blocks);

                                    float ss = 1.0f;
                                    if (write_row_scale) {
                                        if (write_base) {
                                            for (int64_t bi = 0; bi < n_blocks_keep; ++bi) {
                                                const int32_t blk = top_blocks[(size_t) bi];
                                                if (blk < 0) {
                                                    continue;
                                                }
                                                const int64_t in0 = (int64_t) blk * block;
                                                const int64_t in1 = std::min<int64_t>(n_in, in0 + block);
                                                for (int64_t ii = in0; ii < in1; ++ii) {
                                                    const double wi = (double) w[(size_t) ii];
                                                    const double di = (double) r[(size_t) ii];
                                                    const double ws = have_w ? (double) w_scale[(size_t) ii] : 1.0;
                                                    dot += wi * di * ws * ws;
                                                    nn  += (2.0 * wi * di - di * di) * ws * ws;
                                                }
                                            }
                                        } else {
                                            for (int64_t bi = 0; bi < n_blocks_keep; ++bi) {
                                                const int32_t blk = top_blocks[(size_t) bi];
                                                if (blk < 0) {
                                                    continue;
                                                }
                                                const int64_t in0 = (int64_t) blk * block;
                                                const int64_t in1 = std::min<int64_t>(n_in, in0 + block);
                                                for (int64_t ii = in0; ii < in1; ++ii) {
                                                    const double wi  = (double) w[(size_t) ii];
                                                    const double whi = (double) w[(size_t) ii];
                                                    const double ws = have_w ? (double) w_scale[(size_t) ii] : 1.0;
                                                    dot += wi * whi * ws * ws;
                                                    nn  += whi * whi * ws * ws;
                                                }
                                            }
                                        }

                                        if (nn > 1e-30) {
                                            ss = (float) (dot / nn);
                                        }
                                        t.d_row_scale[(size_t) col] = ss;
                                    }

                                    for (int64_t bi = 0; bi < n_blocks_keep; ++bi) {
                                        const int32_t blk = bi < (int64_t) top_blocks.size() ? top_blocks[(size_t) bi] : -1;
                                        t.b_idx[(size_t) col * (size_t) n_blocks_keep + (size_t) bi] = blk;

                                        const int64_t in0 = (blk >= 0) ? (int64_t) blk * block : 0;
                                        for (int64_t tt = 0; tt < block; ++tt) {
                                            const int64_t ii = in0 + tt;
                                            float vv = 0.0f;
                                            if (blk >= 0 && ii >= 0 && ii < n_in) {
                                                vv = write_base ? r[(size_t) ii] : w[(size_t) ii];
                                            }
                                            t.b_val[((size_t) col * (size_t) n_blocks_keep + (size_t) bi) * (size_t) block + (size_t) tt] = vv;
                                        }
                                    }
                                } else {
                                    if (write_base) {
                                        topk_abs_weighted(r, have_w ? &w_scale : nullptr, K_eff, topk);
                                    } else {
                                        topk_abs_weighted(w, have_w ? &w_scale : nullptr, K_eff, topk);
                                    }

                                    float ss = 1.0f;
                                    if (write_row_scale) {
                                        if (write_base) {
                                            for (int64_t rr = 0; rr < K_eff; ++rr) {
                                                const int32_t ii = topk[(size_t) rr];
                                                if (ii < 0 || ii >= (int32_t) n_in) {
                                                    continue;
                                                }

                                                const double wi = (double) w[(size_t) ii];
                                                const double di = (double) r[(size_t) ii];
                                                const double ws = have_w ? (double) w_scale[(size_t) ii] : 1.0;

                                                dot += wi * di * ws * ws;
                                                nn  += (2.0 * wi * di - di * di) * ws * ws;
                                            }
                                        } else {
                                            for (int64_t rr = 0; rr < K_eff; ++rr) {
                                                const int32_t ii = topk[(size_t) rr];
                                                if (ii < 0 || ii >= (int32_t) n_in) {
                                                    continue;
                                                }
                                                const double wi  = (double) w[(size_t) ii];
                                                const double whi = (double) w[(size_t) ii];
                                                const double ws = have_w ? (double) w_scale[(size_t) ii] : 1.0;
                                                dot += wi * whi * ws * ws;
                                                nn  += whi * whi * ws * ws;
                                            }
                                        }

                                        if (nn > 1e-30) {
                                            ss = (float) (dot / nn);
                                        }
                                        t.d_row_scale[(size_t) col] = ss;
                                    }

                                    for (int64_t rr = 0; rr < K_eff; ++rr) {
                                        const int32_t ii = topk[(size_t) rr];
                                        t.d_idx[(size_t) col * (size_t) K_eff + (size_t) rr] = ii;

                                        if (ii >= 0 && ii < (int32_t) n_in) {
                                            t.d_val[(size_t) col * (size_t) K_eff + (size_t) rr] = write_base ? r[(size_t) ii] : w[(size_t) ii];
                                        } else {
                                            t.d_val[(size_t) col * (size_t) K_eff + (size_t) rr] = 0.0f;
                                        }
                                    }
                                }
                            }
                        }
                    });
                }

                for (auto & th : workers) {
                    th.join();
                }

                if (t.re.cost.ops_dense > 0.0) {
                    if (write_base) {
                        fprintf(stderr, "  [blk.%" PRId64 ".%s] cost L=%" PRId64 " B=%" PRId64 " ops dense=%.3g base=%.3g delta=%.3g total=%.3g ratio=%.4f\n",
                                il, kind.c_str(),
                                t.re.cost.L, t.re.cost.B,
                                t.re.cost.ops_dense, t.re.cost.ops_base, t.re.cost.ops_delta, t.re.cost.ops_total, t.re.cost.ops_ratio);
                    } else {
                        fprintf(stderr, "  [blk.%" PRId64 ".%s] cost ops dense=%.3g delta=%.3g total=%.3g ratio=%.4f\n",
                                il, kind.c_str(),
                                t.re.cost.ops_dense, t.re.cost.ops_delta, t.re.cost.ops_total, t.re.cost.ops_ratio);
                    }
                }

                if (eval_cols > 0) {
                    const int64_t t0 = ggml_time_us();
                    eval_metrics em;
                    if (scheme == RESID_BLOCK) {
                        em = write_base
                                ? eval_seeddelta_base_block_residual(W, base, t.b_idx, t.b_val, block, n_blocks_keep, write_row_scale ? &t.d_row_scale : nullptr, nullptr, eval_cols, rng)
                                : eval_block_residual(W, t.b_idx, t.b_val, block, n_blocks_keep, write_row_scale ? &t.d_row_scale : nullptr, nullptr, eval_cols, rng);
                    } else {
                        em = write_base
                                ? eval_seeddelta_base_residual(W, base, t.d_idx, t.d_val, write_row_scale ? &t.d_row_scale : nullptr, nullptr, K_eff, eval_cols, rng)
                                : eval_sparse_residual(W, t.d_idx, t.d_val, write_row_scale ? &t.d_row_scale : nullptr, nullptr, K_eff, eval_cols, rng);
                    }
                    const double sec = double(ggml_time_us() - t0) / 1e6;
                    fprintf(stderr, "  [blk.%" PRId64 ".%s] eval cols=%" PRId64 " rel_l2 mean=%.4f p95=%.4f cos mean=%.4f p05=%.4f nr=%.4f (%.1fs)\n",
                            il, kind.c_str(), eval_cols, em.rel_l2_mean, em.rel_l2_p95, em.cos_mean, em.cos_p05, em.norm_ratio_mean, sec);

                    t.re.em = em;

                    if (have_w) {
                        eval_metrics em_w;
                        if (scheme == RESID_BLOCK) {
                            em_w = write_base
                                    ? eval_seeddelta_base_block_residual(W, base, t.b_idx, t.b_val, block, n_blocks_keep, write_row_scale ? &t.d_row_scale : nullptr, &w_scale, eval_cols, rng)
                                    : eval_block_residual(W, t.b_idx, t.b_val, block, n_blocks_keep, write_row_scale ? &t.d_row_scale : nullptr, &w_scale, eval_cols, rng);
                        } else {
                            em_w = write_base
                                    ? eval_seeddelta_base_residual(W, base, t.d_idx, t.d_val, write_row_scale ? &t.d_row_scale : nullptr, &w_scale, K_eff, eval_cols, rng)
                                    : eval_sparse_residual(W, t.d_idx, t.d_val, write_row_scale ? &t.d_row_scale : nullptr, &w_scale, K_eff, eval_cols, rng);
                        }
                        t.re.em_w = em_w;
                        fprintf(stderr, "  [blk.%" PRId64 ".%s] eval_w cols=%" PRId64 " rel_l2 mean=%.4f p95=%.4f cos mean=%.4f p05=%.4f nr=%.4f\n",
                                il, kind.c_str(), eval_cols, em_w.rel_l2_mean, em_w.rel_l2_p95, em_w.cos_mean, em_w.cos_p05, em_w.norm_ratio_mean);
                    }
                }

                if (eval_cols > 0 && eval_x > 0) {
                    const int64_t t0 = ggml_time_us();
                    eval_metrics emx;
                    if (scheme == RESID_BLOCK) {
                        emx = eval_seeddelta_x_block(W, write_base ? &base : nullptr, t.b_idx, t.b_val, block, n_blocks_keep, write_row_scale ? &t.d_row_scale : nullptr, nullptr, eval_cols, eval_x, rng);
                    } else {
                        emx = eval_seeddelta_x(W, write_base ? &base : nullptr, t.d_idx, t.d_val, write_row_scale ? &t.d_row_scale : nullptr, nullptr, K_eff, eval_cols, eval_x, rng);
                    }
                    const double sec = double(ggml_time_us() - t0) / 1e6;
                    fprintf(stderr, "  [blk.%" PRId64 ".%s] eval_x x=%" PRId64 " cols=%" PRId64 " rel_l2 mean=%.4f p95=%.4f cos mean=%.4f p05=%.4f nr=%.4f (%.1fs)\n",
                            il, kind.c_str(), eval_x, eval_cols, emx.rel_l2_mean, emx.rel_l2_p95, emx.cos_mean, emx.cos_p05, emx.norm_ratio_mean, sec);
                    t.re.em_x = emx;
                    t.re.has_x = true;

                    if (have_w) {
                        eval_metrics emx_w;
                        if (scheme == RESID_BLOCK) {
                            emx_w = eval_seeddelta_x_block(W, write_base ? &base : nullptr, t.b_idx, t.b_val, block, n_blocks_keep, write_row_scale ? &t.d_row_scale : nullptr, &w_scale, eval_cols, eval_x, rng);
                        } else {
                            emx_w = eval_seeddelta_x(W, write_base ? &base : nullptr, t.d_idx, t.d_val, write_row_scale ? &t.d_row_scale : nullptr, &w_scale, K_eff, eval_cols, eval_x, rng);
                        }
                        t.re.em_x_w = emx_w;
                        fprintf(stderr, "  [blk.%" PRId64 ".%s] eval_x_w x=%" PRId64 " cols=%" PRId64 " rel_l2 mean=%.4f p95=%.4f cos mean=%.4f p05=%.4f nr=%.4f\n",
                                il, kind.c_str(), eval_x, eval_cols, emx_w.rel_l2_mean, emx_w.rel_l2_p95, emx_w.cos_mean, emx_w.cos_p05, emx_w.norm_ratio_mean);
                    }
                }

                t.seconds = double(ggml_time_us() - t0_total) / 1e6;
                return t;
            };

            std::vector<int64_t> schedule;
            if (cfg.autotune.enabled) {
                schedule = (kind == "ffn_down") ? cfg.autotune.schedule_down : cfg.autotune.schedule_gate_up;
            }
            if (schedule.empty()) {
                schedule.push_back(K_budget);
            }

            // De-dup schedule while keeping order.
            std::unordered_set<int64_t> seen;
            std::vector<int64_t> sched_unique;
            for (int64_t v : schedule) {
                if (v <= 0) continue;
                if (seen.insert(v).second) {
                    sched_unique.push_back(v);
                }
            }

            const int max_iters = cfg.autotune.enabled && cfg.autotune.max_iters > 0
                    ? std::min<int>((int) cfg.autotune.max_iters, (int) sched_unique.size())
                    : (int) sched_unique.size();

            trial_data best_trial;
            bool have_best = false;
            double best_score = -1e30;

            trial_data selected_trial;
            bool have_selected = false;

            std::vector<report_entry::autotune_attempt> attempts;
            if (cfg.autotune.enabled) {
                attempts.reserve((size_t) max_iters);
            }

            for (int it = 0; it < max_iters; ++it) {
                const int64_t K_try = sched_unique[(size_t) it];
                trial_data td = run_trial(K_try);

                td.re.gating_enabled = cfg.gating_enabled;
                td.re.gating_metric_used = sd_report::metric_kind_to_string(cfg.metric);
                td.re.gating_min_mean = gating_min_mean;
                td.re.gating_min_p05  = gating_min_p05;

                const bool metric_needs_x = (cfg.metric == sd_metric_kind::cos_x || cfg.metric == sd_metric_kind::cos_x_w);
                const bool metric_needs_w = (cfg.metric == sd_metric_kind::cos_w || cfg.metric == sd_metric_kind::cos_x_w);
                bool metric_available = true;
                if (cfg.gating_enabled) {
                    if (metric_needs_x && !td.re.has_x) {
                        metric_available = false;
                    }
                    if (metric_needs_w && !have_w) {
                        metric_available = false;
                    }
                }

                const double metric_val = sd_report::pick_metric_value(td.re, cfg.metric);
                const double metric_p05 = sd_report::pick_metric_p05(td.re, cfg.metric);
                bool gating_pass = true;
                if (cfg.gating_enabled) {
                    gating_pass = metric_available && metric_val >= gating_min_mean && metric_p05 >= gating_min_p05;
                }
                const bool emit = cfg.enabled && (!cfg.gating_enabled || gating_pass);

                td.re.gating_pass = (!cfg.gating_enabled) ? true : (gating_pass && metric_available);
                td.re.gating_value = metric_val;
                td.re.gating_p05 = metric_p05;
                td.re.emit = emit;
                td.re.strip_applied = emit && cfg.strip_dense;
                if (!cfg.gating_enabled) {
                    td.re.decision_reason = sd_constants::decision_no_gating;
                } else if (!metric_available) {
                    td.re.decision_reason = sd_constants::decision_metric_unavailable;
                } else if (!gating_pass) {
                    td.re.decision_reason = sd_constants::decision_gating_fail;
                } else {
                    td.re.decision_reason = sd_constants::decision_pass_gating;
                }

                if (cfg.autotune.enabled) {
                    report_entry::autotune_attempt at;
                    at.K_budget = td.re.K_budget;
                    at.K_eff = td.re.K;
                    at.n_blocks = td.re.n_blocks;
                    at.metric_value = metric_val;
                    at.metric_p05 = metric_p05;
                    at.pass = emit;
                    at.seconds = td.seconds;
                    attempts.push_back(at);
                }

                if (metric_available && metric_val > best_score) {
                    best_score = metric_val;
                    best_trial = td;
                    have_best = true;
                }

                if (emit) {
                    selected_trial = std::move(td);
                    have_selected = true;
                    break;
                }
            }

            if (!have_best) {
                best_trial.re.layer = il;
                best_trial.re.kind = kind;
                best_trial.re.n_in = n_in;
                best_trial.re.n_out = n_out;
                best_trial.re.emit = false;
                best_trial.re.gating_enabled = cfg.gating_enabled;
                best_trial.re.gating_metric_used = sd_report::metric_kind_to_string(cfg.metric);
                best_trial.re.gating_min_mean = gating_min_mean;
                best_trial.re.gating_min_p05 = gating_min_p05;
                best_trial.re.decision_reason = "metric_unavailable";
                have_best = true;
            }

            trial_data final_td;
            if (have_selected) {
                final_td = std::move(selected_trial);
                final_td.re.autotune_enabled = cfg.autotune.enabled;
                final_td.re.autotune_selected_budget = final_td.re.K_budget;
                final_td.re.autotune_attempts = std::move(attempts);
            } else {
                final_td = std::move(best_trial);
                final_td.re.emit = false;
                final_td.re.strip_applied = false;
                final_td.re.autotune_enabled = cfg.autotune.enabled;
                final_td.re.autotune_selected_budget = 0;
                final_td.re.autotune_attempts = std::move(attempts);
                if (cfg.autotune.enabled) {
                    final_td.re.decision_reason = sd_constants::decision_autotune_failed_keep_dense;
                }
            }

            report_entry re = std::move(final_td.re);
            std::vector<int32_t> d_idx = std::move(final_td.d_idx);
            std::vector<float>   d_val = std::move(final_td.d_val);
            std::vector<int32_t> b_idx = std::move(final_td.b_idx);
            std::vector<float>   b_val = std::move(final_td.b_val);
            std::vector<float>   d_row_scale = std::move(final_td.d_row_scale);

            const int64_t n_blocks_keep = re.n_blocks;
            const int64_t K_eff = re.K;

            // FFN proxy v0 (logging-only): replace only this tensor (coo, no base).
            if (scheme == RESID_COO && !write_base && eval_x > 0) {
                const std::string wg_name = "blk." + std::to_string(il) + ".ffn_gate.weight";
                const std::string wu_name = "blk." + std::to_string(il) + ".ffn_up.weight";
                const std::string wd_name = "blk." + std::to_string(il) + ".ffn_down.weight";
                ggml_tensor * W_gate = ggml_get_tensor(ctx_data, wg_name.c_str());
                ggml_tensor * W_up   = ggml_get_tensor(ctx_data, wu_name.c_str());
                ggml_tensor * W_down = ggml_get_tensor(ctx_data, wd_name.c_str());

                ffn_proxy_metrics fpm;
                const int proxy_seed = seed + (int) il * 101 + (int) (kind == "ffn_gate" ? 1 : (kind == "ffn_up" ? 2 : 3));
                bool proxy_ok = eval_ffn_proxy_coo_replace_one(kind, W_gate, W_up, W_down, d_idx, d_val, write_base ? &base : nullptr, write_base, K_eff, eval_x, eval_cols, proxy_seed, fpm);
                if (proxy_ok) {
                    re.ffn_proxy_available = true;
                    re.ffn_proxy_scope = sd_constants::ffn_proxy_scope_replace_only_current_tensor;
                    re.ffn_proxy_base_used = write_base;
                    re.ffn_proxy_eval_x = fpm.eval_x;
                    re.ffn_proxy_eval_out = fpm.eval_out;
                    re.ffn_proxy_seed = proxy_seed;
                    re.ffn_proxy_cos_mean = fpm.cos_mean;
                    re.ffn_proxy_cos_p05 = fpm.cos_p05;
                    re.ffn_proxy_l2_mean = fpm.l2_mean;
                    re.ffn_proxy_l2_p95 = fpm.l2_p95;
                    re.ffn_proxy_log_norm_ratio_mean = fpm.log_norm_ratio_mean;
                    re.ffn_proxy_log_norm_ratio_p95 = fpm.log_norm_ratio_p95;
                } else {
                    re.ffn_proxy_available = false;
                    re.ffn_proxy_reason = (kind == "ffn_down")
                            ? sd_constants::ffn_proxy_reason_kind_not_supported
                            : sd_constants::ffn_proxy_reason_unavailable;
                }
            } else {
                re.ffn_proxy_available = false;
                re.ffn_proxy_reason = (eval_x <= 0)
                        ? sd_constants::ffn_proxy_reason_requires_eval_x
                        : sd_constants::ffn_proxy_reason_requires_coo;
            }

            // Stack-cost (v1): simple affine penalty vs targets; zero if not emitted or metric missing.
            {
                const double metric_mean = re.gating_value;
                const double metric_p05  = re.gating_p05;
                const double tau_mean    = re.target_tau_mean;
                const double tau_p05     = re.target_tau_p05;
                double cost_delta = 0.0;
                if (re.emit && std::isfinite(metric_mean) && std::isfinite(metric_p05)) {
                    const double d_mean = std::max(0.0, tau_mean - metric_mean);
                    const double d_p05  = std::max(0.0, tau_p05  - metric_p05);
                    cost_delta = d_mean + d_p05; // weights (alpha=beta=1) v1
                }
                re.stack_cost_delta = cost_delta;
                re.stack_cost_total = stack_cost_running + (re.emit ? cost_delta : 0.0);
                if (re.emit) {
                    stack_cost_running = re.stack_cost_total;
                }
            }

            if (!re.emit) {
                sd_report::finalize_report_entry(re);
                report.push_back(std::move(re));
                continue;
            }

            // Enforce global stack_cost cap if provided.
            if (stack_cost_running > stack_cost_cap) {
                re.emit = false;
                re.strip_applied = false;
                re.decision_reason = sd_constants::decision_stack_cost_cap;
                re.reject_reason = re.decision_reason;
                sd_report::finalize_report_entry(re);
                report.push_back(std::move(re));
                continue;
            }

            stack_guard.record_pass(kind);

            if (re.strip_applied) {
                strip_weights.insert(weight_name);
                any_strip = true;
            }

            sd_report::finalize_report_entry(re);
            report.push_back(std::move(re));

            // Allocate a dedicated ggml context for new tensors to avoid a giant arena.
            const size_t size_idx = scheme == RESID_BLOCK
                    ? (size_t) n_blocks_keep * (size_t) n_out * ggml_type_size(idx_type)
                    : (size_t) K_eff * (size_t) n_out * ggml_type_size(idx_type);
            const size_t size_val = scheme == RESID_BLOCK
                    ? (size_t) block * (size_t) n_blocks_keep * (size_t) n_out * ggml_type_size(val_type)
                    : (size_t) K_eff * (size_t) n_out * ggml_type_size(val_type);
            const size_t size_row_scale = write_row_scale ? (size_t) n_out * sizeof(ggml_fp16_t) : 0;
            const ggml_type perm_type = write_base ? (base.L <= 32768 ? GGML_TYPE_I16 : GGML_TYPE_I32) : GGML_TYPE_I16;
            const size_t size_base_diag = write_base ? (size_t) base.L * (size_t) base.B * sizeof(ggml_fp16_t) * 3 : 0;
            const size_t size_base_perm = write_base ? (size_t) base.L * (size_t) base.B * ggml_type_size(perm_type) : 0;
            const size_t size_base = size_base_diag + size_base_perm;
            const size_t n_tensors_new = 2 + (write_row_scale ? 1 : 0) + (write_base ? 4 : 0);
            const size_t mem_size_sd = ggml_tensor_overhead() * (n_tensors_new + 4) + size_idx + size_val + size_row_scale + size_base;

            ggml_init_params sd_params = { mem_size_sd, nullptr, false };
            ggml_context * ctx_sd = ggml_init(sd_params);
            sd_contexts.push_back(ctx_sd);

            ggml_tensor * t_idx = nullptr;
            ggml_tensor * t_val = nullptr;
            if (scheme == RESID_BLOCK) {
                t_idx = ggml_new_tensor_2d(ctx_sd, idx_type, n_blocks_keep, n_out);
                ggml_set_name(t_idx, b_idx_name.c_str());
                if (idx_type == GGML_TYPE_I16) {
                    auto * dst_i16 = (int16_t *) t_idx->data;
                    for (int64_t col = 0; col < n_out; ++col) {
                        for (int64_t bi = 0; bi < n_blocks_keep; ++bi) {
                            const int32_t blk = b_idx[(size_t) col * (size_t) n_blocks_keep + (size_t) bi];
                            dst_i16[col * n_blocks_keep + bi] = (blk < -32768 || blk > 32767) ? (int16_t) -1 : (int16_t) blk;
                        }
                    }
                } else {
                    std::memcpy(t_idx->data, b_idx.data(), b_idx.size() * sizeof(int32_t));
                }

                t_val = ggml_new_tensor_3d(ctx_sd, val_type, block, n_blocks_keep, n_out);
                ggml_set_name(t_val, b_val_name.c_str());
                if (val_type == GGML_TYPE_F16) {
                    auto * dst_f16 = (ggml_fp16_t *) t_val->data;
                    for (int64_t col = 0; col < n_out; ++col) {
                        for (int64_t bi = 0; bi < n_blocks_keep; ++bi) {
                            ggml_fp32_to_fp16_row(
                                    b_val.data() + ((size_t) col * (size_t) n_blocks_keep + (size_t) bi) * (size_t) block,
                                    dst_f16 + ((size_t) col * (size_t) n_blocks_keep + (size_t) bi) * (size_t) block,
                                    (int) block);
                        }
                    }
                } else {
                    std::memcpy(t_val->data, b_val.data(), b_val.size() * sizeof(float));
                }
            } else {
                t_idx = ggml_new_tensor_2d(ctx_sd, idx_type, K_eff, n_out);
                ggml_set_name(t_idx, d_idx_name.c_str());
                if (idx_type == GGML_TYPE_I16) {
                    auto * dst_i16 = (int16_t *) t_idx->data;
                    for (int64_t col = 0; col < n_out; ++col) {
                        for (int64_t r = 0; r < K_eff; ++r) {
                            const int32_t ii = d_idx[(size_t) col * (size_t) K_eff + (size_t) r];
                            dst_i16[col * K_eff + r] = (ii < -32768 || ii > 32767) ? (int16_t) -1 : (int16_t) ii;
                        }
                    }
                } else {
                    std::memcpy(t_idx->data, d_idx.data(), d_idx.size() * sizeof(int32_t));
                }

                t_val = ggml_new_tensor_2d(ctx_sd, val_type, K_eff, n_out);
                ggml_set_name(t_val, d_val_name.c_str());
                if (val_type == GGML_TYPE_F16) {
                    auto * dst_f16 = (ggml_fp16_t *) t_val->data;
                    for (int64_t col = 0; col < n_out; ++col) {
                        ggml_fp32_to_fp16_row(d_val.data() + (size_t) col * (size_t) K_eff, dst_f16 + (size_t) col * (size_t) K_eff, (int) K_eff);
                    }
                } else {
                    std::memcpy(t_val->data, d_val.data(), d_val.size() * sizeof(float));
                }
            }

            pending_tensor_set pts;
            pts.ctx = ctx_sd;
            pts.tensors.push_back(t_idx);
            pts.tensors.push_back(t_val);
            n_added += 2;

            if (write_row_scale) {
                ggml_tensor * t_rs = ggml_new_tensor_1d(ctx_sd, GGML_TYPE_F16, n_out);
                ggml_set_name(t_rs, d_row_scale_name.c_str());
                auto * rs_f16 = (ggml_fp16_t *) t_rs->data;
                ggml_fp32_to_fp16_row(d_row_scale.data(), rs_f16, n_out);
                pts.tensors.push_back(t_rs);
                n_added += 1;
            }

            if (write_base) {
                const std::string base_d1_name = "blk." + std::to_string(il) + "." + kind + ".base_d1";
                const std::string base_d2_name = "blk." + std::to_string(il) + "." + kind + ".base_d2";
                const std::string base_d3_name = "blk." + std::to_string(il) + "." + kind + ".base_d3";
                const std::string base_perm1_name = "blk." + std::to_string(il) + "." + kind + ".base_perm1";

                ggml_tensor * t_d1 = ggml_new_tensor_2d(ctx_sd, GGML_TYPE_F16, base.L, base.B);
                ggml_set_name(t_d1, base_d1_name.c_str());
                ggml_tensor * t_d2 = ggml_new_tensor_2d(ctx_sd, GGML_TYPE_F16, base.L, base.B);
                ggml_set_name(t_d2, base_d2_name.c_str());
                ggml_tensor * t_d3 = ggml_new_tensor_2d(ctx_sd, GGML_TYPE_F16, base.L, base.B);
                ggml_set_name(t_d3, base_d3_name.c_str());
                ggml_tensor * t_p1 = ggml_new_tensor_2d(ctx_sd, perm_type, base.L, base.B);
                ggml_set_name(t_p1, base_perm1_name.c_str());

                auto * d1_f16 = (ggml_fp16_t *) t_d1->data;
                auto * d2_f16 = (ggml_fp16_t *) t_d2->data;
                auto * d3_f16 = (ggml_fp16_t *) t_d3->data;
                for (int64_t b = 0; b < base.B; ++b) {
                    ggml_fp32_to_fp16_row(base.d1.data() + (size_t) b * (size_t) base.L, d1_f16 + (size_t) b * (size_t) base.L, base.L);
                    ggml_fp32_to_fp16_row(base.d2.data() + (size_t) b * (size_t) base.L, d2_f16 + (size_t) b * (size_t) base.L, base.L);
                    ggml_fp32_to_fp16_row(base.d3.data() + (size_t) b * (size_t) base.L, d3_f16 + (size_t) b * (size_t) base.L, base.L);
                }

                pts.tensors.push_back(t_d1);
                pts.tensors.push_back(t_d2);
                pts.tensors.push_back(t_d3);
                pts.tensors.push_back(t_p1);
                if (perm_type == GGML_TYPE_I16) {
                    auto * p1_i16 = (int16_t *) t_p1->data;
                    for (size_t i = 0; i < base.perm1.size(); ++i) {
                        const int32_t v = base.perm1[i];
                        p1_i16[i] = (v < 0 || v > 32767) ? (int16_t) 0 : (int16_t) v;
                    }
                } else {
                    std::memcpy(t_p1->data, base.perm1.data(), base.perm1.size() * sizeof(int32_t));
                }
                n_added += 4;
            }

            pending.push_back(std::move(pts));
        }
    }

    if (n_added == 0) {
        fprintf(stderr, "no new SeedΔ tensors added\n");
    }

    gguf_context * dst = gguf_init_empty();
    gguf_set_kv(dst, src);

    // Add original tensors, skipping those strippeados.
    for (int64_t ti = 0; ti < n_tensors; ++ti) {
        const char * name = gguf_get_tensor_name(src, ti);
        if (any_strip && strip_weights.count(name) > 0) {
            continue;
        }
        ggml_tensor * t = ggml_get_tensor(ctx_data, name);
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

    gguf_set_val_bool(dst, "seeddelta.enabled", n_added > 0);
    gguf_set_val_u32(dst, "seeddelta.version", 1);
    gguf_set_val_u32(dst, "seeddelta.scheme", (uint32_t) scheme);
    gguf_set_val_bool(dst, "seeddelta.row_scale", write_row_scale);
    gguf_set_val_u32(dst, "seeddelta.resid.K", (uint32_t) K_default);
    gguf_set_val_bool(dst, "seeddelta.strip_dense", any_strip);
    if (scheme == RESID_BLOCK) {
        gguf_set_val_u32(dst, "seeddelta.resid.block", (uint32_t) block);
    }
    gguf_set_val_bool(dst, "seeddelta.resid.K_variable", K_variable);
    if (K_variable) {
        gguf_set_val_u32(dst, "seeddelta.resid.K_gate", (uint32_t) K_gate_eff);
        gguf_set_val_u32(dst, "seeddelta.resid.K_up",   (uint32_t) K_up_eff);
        gguf_set_val_u32(dst, "seeddelta.resid.K_down", (uint32_t) K_down_eff);
    }
    gguf_set_val_bool(dst, "seeddelta.base.enabled", write_base);
    if (write_base) {
        gguf_set_val_u32(dst, "seeddelta.base.kind", 1);  // hadamard_acdc_stack
        gguf_set_val_u32(dst, "seeddelta.base.depth", 2); // D3*H*D2*H*D1
        gguf_set_val_u32(dst, "seeddelta.base.R", 1);
        gguf_set_val_u32(dst, "seeddelta.base.max_samples", (uint32_t) std::max<int64_t>(0, base_max_samples));
        gguf_set_val_u32(dst, "seeddelta.base.perm_trials", (uint32_t) std::max(1, base_perm_trials));
    }

    // Per-tensor audit metadata (minimal, per GGUF kv).
    for (const auto & re : report) {
        const std::string prefix = "seeddelta.blk." + std::to_string(re.layer) + "." + re.kind;
        gguf_set_val_bool(dst, (prefix + ".enabled").c_str(), re.emit);
        gguf_set_val_bool(dst, (prefix + ".gating_pass").c_str(), re.gating_pass);
        gguf_set_val_bool(dst, (prefix + ".strip_dense").c_str(), re.strip_applied);
        gguf_set_val_u32(dst, (prefix + ".K").c_str(), (uint32_t) std::max<int64_t>(0, re.K));
    }

    printf("writing %s with %" PRId64 " new tensors\n", out_fname.c_str(), n_added);
    if (!gguf_write_to_file(dst, out_fname.c_str(), false)) {
        fprintf(stderr, "seeddelta-build: failed to write %s\n", out_fname.c_str());
        return 1;
    }

    if (!report_json.empty()) {
        sd_report::report_config cfg;
        cfg.input = in_fname;
        cfg.output = out_fname;
        cfg.write_base = write_base;
        cfg.base_max_samples = base_max_samples;
        cfg.base_perm_trials = base_perm_trials;
        cfg.policy_file = policy_file;
        cfg.policy_hash = policy_hash;
        cfg.have_imatrix = have_imatrix;
        cfg.imatrix_file = imatrix_file;
        cfg.imatrix_eps = imatrix_eps;
        cfg.imatrix_power = imatrix_power;
        cfg.imatrix_datasets = imatrix_datasets;
        cfg.resid.scheme = (scheme == RESID_BLOCK) ? sd_report::seeddelta_scheme::block : sd_report::seeddelta_scheme::coo;
        cfg.resid.block = block;
        cfg.resid.K = K_default;
        cfg.resid.K_gate = K_gate_eff;
        cfg.resid.K_up = K_up_eff;
        cfg.resid.K_down = K_down_eff;
        cfg.resid.idx_type = idx_type_str;
        cfg.resid.val_type = val_type_str;
        cfg.resid.row_scale = write_row_scale;
        cfg.eval_cols = eval_cols;
        cfg.eval_x = eval_x;
        cfg.stack_pass_gate_up = stack_guard.gate_up_pass;
        cfg.stack_pass_down = stack_guard.down_pass;
        if (!sd_report::write_report_json(report_json, cfg, report)) {
            return 1;
        }
    }

    if (!policy_export_file.empty()) {
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

        auto pres = sd_policy_export_write_canonical(policy_export_file, decisions);
        if (!pres.ok) {
            fprintf(stderr, "seeddelta-build: failed to export policy: %s\n", pres.error.c_str());
            return 1;
        }
        fprintf(stderr, "seeddelta-build: wrote policy export %s\n", policy_export_file.c_str());
    }

    return 0;
}
