#include "common.h"
#include "ggml.h"
#include "gguf.h"
#include "seeddelta_policy.h"
#include "seeddelta_policy_export.h"
#include "seeddelta_policy_selftest.h"

#include "sd_cost.h"
#include "sd_eval.h"
#include "sd_ffn_proxy.h"
#include "sd_report.h"

#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <numeric>
#include <random>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <limits>
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

static void usage(const char * argv0) {
    printf("usage: %s -i in.gguf -o out.gguf [options]\n\n", argv0);
    printf("options:\n");
    printf("  --layers A-B         restrict to layer range (default: all)\n");
    printf("  --scheme coo|block   residual encoding (default: coo)\n");
    printf("  --block N            block size for --scheme block (default: 32)\n");
    printf("  --K N                top-K residual entries per output (default: 32)\n");
    printf("  --K-gate N           override top-K for ffn_gate\n");
    printf("  --K-up N             override top-K for ffn_up\n");
    printf("  --K-down N           override top-K for ffn_down\n");
    printf("  --idx-type i16|i32   index tensor type (default: i16)\n");
    printf("  --val-type f16|f32   value tensor type (default: f16)\n");
    printf("  --row-scale          write per-output d_row_scale tensor (default: off)\n");
    printf("  --no-row-scale       disable d_row_scale tensor output\n");
    printf("  --imatrix FILE       importance matrix GGUF from llama-imatrix for data-aware weighting\n");
    printf("  --imatrix-eps F      clamp min imatrix value before sqrt (default: 1e-8)\n");
    printf("  --imatrix-power F    exponent for imatrix weighting (default: 1.0)\n");
    printf("  --base               write Hadamard base tensors (base_d1/base_d2/base_d3/base_perm1) and store residual vs base\n");
    printf("  --base-max-samples N max sampled outputs per block for base fit (default: 2048, 0=all)\n");
    printf("  --base-perm-trials N random P1 trials per base block (default: 1)\n");
    printf("  --strip-dense        drop original dense weights for layers processed (reduces GGUF size; disables dense fallback)\n");
    printf("  --policy FILE        JSON policy for per-layer/tensor K/block/gating/autotune/strip overrides\n");
    printf("  --policy-strict      reject unknown keys in policy.json (default: warn and ignore)\n");
    printf("  --policy-dump-resolved print resolved config per tensor (debug)\n");
    printf("  --policy-export PATH write a canonical policy.json capturing final per-tensor decisions\n");
    printf("  --policy-self-test   run internal policy merge tests and exit\n");
    printf("  --overwrite-existing allow rebuilding tensors that already have SeedΔ (default: skip)\n");
    printf("  --stack-cost-cap F   hard cap for accumulated stack_cost (default: +inf => no cap)\n");
    printf("  -t, --threads N      worker threads (default: nproc)\n");
    printf("  --eval-cols N        evaluate reconstruction gap on N random outputs per weight (default: 0=off)\n");
    printf("  --eval-x N           evaluate functional gap on N random x vectors (requires --eval-cols, default: 0=off)\n");
    printf("  --report-json PATH   write per-weight metrics JSON report\n");
    printf("  --seed N             RNG seed (default: 1234)\n");
    exit(1);
}

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

static void read_column_f32(const ggml_tensor * W, int64_t j, std::vector<float> & out) {
    const int64_t n_in = W->ne[0];
    out.resize(n_in);
    const uint8_t * base = (const uint8_t *) W->data + j * W->nb[1];
    const auto * traits = ggml_get_type_traits(W->type);

    if (!traits->is_quantized) {
        if (W->type == GGML_TYPE_F32) {
            std::memcpy(out.data(), base, n_in * sizeof(float));
            return;
        }
        if (W->type == GGML_TYPE_F16) {
            const ggml_fp16_t * v = (const ggml_fp16_t *) base;
            for (int64_t i = 0; i < n_in; ++i) out[i] = ggml_fp16_to_fp32(v[i]);
            return;
        }
    }

    GGML_ASSERT(traits->to_float && "no to_float for tensor type");
    traits->to_float(base, out.data(), n_in);
}

static void topk_abs_weighted(
        const std::vector<float> & v,
        const std::vector<float> * w_scale,
        int64_t K,
        std::vector<int32_t> & idx_out) {
    const int64_t n = (int64_t) v.size();
    K = std::min<int64_t>(K, n);
    idx_out.resize((size_t) K);

    std::vector<int32_t> idx((size_t) n);
    std::iota(idx.begin(), idx.end(), 0);
    auto score = [&](int32_t i) -> float {
        const float s = std::fabs(v[(size_t) i]);
        return w_scale ? (s * (*w_scale)[(size_t) i]) : s;
    };
    if (K < n) {
        std::nth_element(idx.begin(), idx.begin() + K, idx.end(), [&](int32_t a, int32_t b) {
            return score(a) > score(b);
        });
        idx.resize((size_t) K);
    }
    std::sort(idx.begin(), idx.end(), [&](int32_t a, int32_t b) {
        return score(a) > score(b);
    });
    idx_out = idx;
}

// Select top blocks by residual energy (sum (v[i]*w)^2 within block).
static void topk_blocks_energy_weighted(
        const std::vector<float> & v,
        const std::vector<float> * w_scale,
        int64_t block,
        int64_t n_blocks_keep,
        std::vector<int32_t> & idx_out) {
    const int64_t n = (int64_t) v.size();
    if (n <= 0 || block <= 0 || n_blocks_keep <= 0) {
        idx_out.clear();
        return;
    }

    const int64_t n_blocks = (n + block - 1) / block;
    n_blocks_keep = std::min<int64_t>(n_blocks_keep, n_blocks);

    std::vector<double> energy((size_t) n_blocks, 0.0);
    for (int64_t b = 0; b < n_blocks; ++b) {
        const int64_t i0 = b * block;
        const int64_t i1 = std::min<int64_t>(n, i0 + block);
        double e = 0.0;
        for (int64_t i = i0; i < i1; ++i) {
            const double ws = w_scale ? (double) (*w_scale)[(size_t) i] : 1.0;
            const double x = (double) v[(size_t) i] * ws;
            e += x * x;
        }
        energy[(size_t) b] = e;
    }

    std::vector<int32_t> idx((size_t) n_blocks);
    std::iota(idx.begin(), idx.end(), 0);
    auto score = [&](int32_t b) -> double {
        return energy[(size_t) b];
    };

    if (n_blocks_keep < n_blocks) {
        std::nth_element(idx.begin(), idx.begin() + n_blocks_keep, idx.end(), [&](int32_t a, int32_t b) {
            return score(a) > score(b);
        });
        idx.resize((size_t) n_blocks_keep);
    }

    std::sort(idx.begin(), idx.end(), [&](int32_t a, int32_t b) {
        return score(a) > score(b);
    });

    idx_out = idx;
}

struct eval_metrics {
    double rel_l2_mean = 0.0;
    double rel_l2_p95  = 0.0;
    double cos_mean    = 0.0;
    double cos_p05     = 0.0;
    double norm_ratio_mean = 0.0;
};

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
                    re.ffn_proxy_scope = "replace_only_current_tensor";
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
                    re.ffn_proxy_reason = (kind == "ffn_down") ? "proxy_kind_not_supported" : "proxy_unavailable";
                }
            } else {
                re.ffn_proxy_available = false;
                re.ffn_proxy_reason = (eval_x <= 0) ? "proxy_requires_eval_x" : "proxy_requires_coo";
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
                finalize_report_entry(re);
                report.push_back(std::move(re));
                continue;
            }

            // Enforce global stack_cost cap if provided.
            if (stack_cost_running > stack_cost_cap) {
                re.emit = false;
                re.strip_applied = false;
                re.decision_reason = "stack_cost_cap";
                re.reject_reason = re.decision_reason;
                finalize_report_entry(re);
                report.push_back(std::move(re));
                continue;
            }

            stack_guard.record_pass(kind);

            if (re.strip_applied) {
                strip_weights.insert(weight_name);
                any_strip = true;
            }

            finalize_report_entry(re);
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
        const bool scheme_block = (scheme == RESID_BLOCK);
        if (!write_report_json(
                report_json,
                in_fname,
                out_fname,
                have_imatrix,
                imatrix_file,
                imatrix_datasets,
                imatrix_eps,
                imatrix_power,
                write_base,
                base_max_samples,
                base_perm_trials,
                policy_file,
                policy_hash,
                scheme_block,
                block,
                K_default,
                K_gate_eff,
                K_up_eff,
                K_down_eff,
                idx_type_str,
                val_type_str,
                write_row_scale,
                eval_cols,
                eval_x,
                stack_guard.gate_up_pass,
                stack_guard.down_pass,
                report)) {
            fprintf(stderr, "seeddelta-build: failed to write report %s\n", report_json.c_str());
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
