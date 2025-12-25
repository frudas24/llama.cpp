#include "include/sd_build.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <regex>
#include <thread>
#include <unordered_set>

#include "sd_cost.h"
#include "sd_eval.h"
#include "sd_ffn_proxy.h"
#include "sd_constants.h"
#include "sd_utils.h"

struct sd_tile_stats {
    double l2 = 0.0;
    double abs_sum = 0.0;
    double max_abs = 0.0;
    int64_t cols = 0;
};

static double sd_tensor_l2_norm(ggml_tensor * W) {
    const int64_t n = ggml_nelements(W);
    double sum = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        const float v = ggml_get_f32_1d(W, i);
        sum += (double) v * (double) v;
    }
    return sum > 0.0 ? std::sqrt(sum) : 0.0;
}

static std::vector<sd_tile_stats> sd_compute_tile_stats(
        ggml_tensor * W,
        int64_t tile_rows,
        int64_t n_in,
        int64_t n_out) {
    const int64_t n_tiles = (tile_rows > 0 && n_out > 0) ? (n_out + tile_rows - 1) / tile_rows : (n_out > 0 ? 1 : 0);
    std::vector<sd_tile_stats> stats((size_t) n_tiles);
    if (n_tiles <= 0) {
        return stats;
    }
    std::vector<float> col;
    col.reserve((size_t) n_in);
    for (int64_t col_idx = 0; col_idx < n_out; ++col_idx) {
        read_column_f32(W, col_idx, col);
        double l2 = 0.0;
        double abs_sum = 0.0;
        double max_abs = 0.0;
        for (float v : col) {
            const double dv = (double) v;
            l2 += dv * dv;
            const double av = std::fabs(dv);
            abs_sum += av;
            if (av > max_abs) {
                max_abs = av;
            }
        }
        const int64_t tile = tile_rows > 0 ? col_idx / tile_rows : 0;
        auto & ts = stats[(size_t) tile];
        ts.l2 += l2;
        ts.abs_sum += abs_sum;
        ts.max_abs = std::max(ts.max_abs, max_abs);
        ts.cols += 1;
    }
    return stats;
}

static std::vector<int64_t> sd_assign_k_per_tile(
        const std::vector<sd_tile_stats> & stats,
        const std::vector<int64_t> & k_levels,
        const sd_args & args,
        int64_t n_in,
        double & gap_vs_uniform,
        int64_t & tiles_sampled,
        int64_t & selector_rank) {
    const size_t n_tiles = stats.size();
    gap_vs_uniform = 0.0;
    tiles_sampled = 0;
    selector_rank = 0;

    if (n_tiles == 0 || k_levels.empty()) {
        return {};
    }

    std::vector<int64_t> k_per_tile((size_t) n_tiles, k_levels.back());

    if (stats.empty()) {
        for (size_t ti = 0; ti < n_tiles; ++ti) {
            k_per_tile[ti] = k_levels[ti % k_levels.size()];
        }
        selector_rank = 1;
        return k_per_tile;
    }

    const std::string selector = args.k_selector.empty() ? "cycle" : args.k_selector;
    if (selector == "uniform") {
        std::fill(k_per_tile.begin(), k_per_tile.end(), k_levels.back());
        selector_rank = 1;
    } else if (selector == "cycle") {
        for (size_t ti = 0; ti < n_tiles; ++ti) {
            k_per_tile[ti] = k_levels[ti % k_levels.size()];
        }
        selector_rank = 1;
    } else { // ttcross / unknown -> treat as ttcross heuristic
        const int rank = std::max(1, std::min(args.k_selector_rank, 2));
        selector_rank = rank;
        tiles_sampled = std::max<int64_t>(1, std::min<int64_t>(args.k_selector_samples, (int64_t) n_tiles));

        // Build feature matrix: [log1p(l2), mean_abs, max_abs]
        std::vector<std::array<double, 3>> feats(n_tiles);
        std::array<double, 3> mean = {0.0, 0.0, 0.0};
        for (size_t ti = 0; ti < n_tiles; ++ti) {
            const auto & ts = stats[ti];
            const double denom = (double) ts.cols * (double) std::max<int64_t>(1, n_in);
            const double mean_abs = denom > 0.0 ? ts.abs_sum / denom : 0.0;
            feats[ti] = { std::log1p(ts.l2), mean_abs, ts.max_abs };
            for (int d = 0; d < 3; ++d) mean[d] += feats[ti][d];
        }
        for (int d = 0; d < 3; ++d) mean[d] /= (double) n_tiles;
        std::array<double, 3> stdev = {0.0, 0.0, 0.0};
        for (size_t ti = 0; ti < n_tiles; ++ti) {
            for (int d = 0; d < 3; ++d) {
                const double diff = feats[ti][d] - mean[d];
                stdev[d] += diff * diff;
            }
        }
        for (int d = 0; d < 3; ++d) {
            stdev[d] = std::sqrt(stdev[d] / std::max<size_t>(1, n_tiles));
            if (stdev[d] < 1e-9) stdev[d] = 1.0;
        }

        std::vector<std::array<double, 3>> zfeats(n_tiles);
        std::vector<double> norms(n_tiles, 0.0);
        for (size_t ti = 0; ti < n_tiles; ++ti) {
            double norm = 0.0;
            for (int d = 0; d < 3; ++d) {
                const double z = (feats[ti][d] - mean[d]) / stdev[d];
                zfeats[ti][d] = z;
                norm += z * z;
            }
            norms[ti] = std::sqrt(norm);
        }

        const double eps = 1e-9;
        size_t j1 = 0;
        for (size_t ti = 1; ti < n_tiles; ++ti) {
            if (norms[ti] > norms[j1]) j1 = ti;
        }
        size_t j2 = n_tiles;
        if (rank > 1 && n_tiles > 1) {
            double best = -1.0;
            for (size_t ti = 0; ti < n_tiles; ++ti) {
                if (ti == j1) continue;
                double dot = 0.0;
                for (int d = 0; d < 3; ++d) dot += zfeats[ti][d] * zfeats[j1][d];
                const double denom = (norms[ti] * norms[j1]) + eps;
                const double cos2 = denom > 0.0 ? (dot / denom) * (dot / denom) : 0.0;
                const double score = norms[ti] * (1.0 - std::min(0.999, cos2));
                if (score > best) {
                    best = score;
                    j2 = ti;
                }
            }
            if (best <= 0.0) {
                j2 = n_tiles; // fallback to rank-1 if no diverse second fiber
                selector_rank = 1;
            }
        } else {
            selector_rank = 1;
        }

        std::vector<std::pair<double, size_t>> order;
        order.reserve(n_tiles);
        for (size_t ti = 0; ti < n_tiles; ++ti) {
            double sim1 = 0.0;
            double sim2 = 0.0;
            if (norms[ti] > eps && norms[j1] > eps) {
                double dot = 0.0;
                for (int d = 0; d < 3; ++d) dot += zfeats[ti][d] * zfeats[j1][d];
                sim1 = dot / ((norms[ti] * norms[j1]) + eps);
            }
            if (selector_rank > 1 && j2 < n_tiles && norms[ti] > eps && norms[j2] > eps) {
                double dot = 0.0;
                for (int d = 0; d < 3; ++d) dot += zfeats[ti][d] * zfeats[j2][d];
                sim2 = dot / ((norms[ti] * norms[j2]) + eps);
            }
            double score = sim1;
            if (selector_rank > 1 && j2 < n_tiles) {
                score = (sim1 >= sim2) ? sim1 : -sim2;
            }
            order.emplace_back(score, ti);
        }
        std::sort(order.begin(), order.end(), [](const auto & a, const auto & b) {
            return a.first > b.first;
        });

        // Assign higher K to highest score tiles.
        for (size_t idx = 0; idx < order.size(); ++idx) {
            const size_t tile = order[idx].second;
            size_t level_idx = (size_t) ((double) idx * (double) k_levels.size() / (double) n_tiles);
            if (level_idx >= k_levels.size()) level_idx = k_levels.size() - 1;
            const size_t rev_idx = (k_levels.size() - 1) - level_idx;
            k_per_tile[tile] = k_levels[rev_idx];
        }
    }

    double mean_uniform = 0.0;
    for (size_t ti = 0; ti < n_tiles; ++ti) {
        mean_uniform += (double) k_levels[ti % k_levels.size()];
    }
    mean_uniform /= (double) n_tiles;
    double mean_sel = 0.0;
    for (int64_t v : k_per_tile) mean_sel += (double) v;
    mean_sel /= (double) n_tiles;
    gap_vs_uniform = mean_sel - mean_uniform;

    return k_per_tile;
}

sd_build_result sd_build_layers(
        const sd_args & args,
        gguf_context * src,
        ggml_context * ctx_data,
        const std::vector<int64_t> & layers,
        sd_resid_scheme scheme,
        ggml_type idx_type,
        ggml_type val_type,
        int64_t block,
        int64_t K_gate_eff,
        int64_t K_up_eff,
        int64_t K_down_eff,
        const sd_policy * policy_ptr,
        bool policy_dump_resolved,
        bool overwrite_existing,
        bool have_imatrix,
        const std::unordered_map<std::string, std::vector<float>> & imatrix_data,
        float imatrix_eps,
        float imatrix_power,
        double stack_cost_cap) {
    sd_build_result result;

    const std::vector<std::string> kinds = { "ffn_gate", "ffn_up", "ffn_down" };

    std::mt19937 rng(args.seed);

    int64_t n_added = 0;
    std::vector<report_entry> report;
    std::vector<pending_tensor_set> pending;
    std::unordered_set<std::string> strip_weights;
    bool any_strip = false;
    double stack_cost_running = 0.0;

    const bool write_row_scale = args.write_row_scale;
    const bool write_base = args.write_base;
    const int64_t base_max_samples = args.base_max_samples;
    const int base_perm_trials = args.base_perm_trials;
    const int n_threads = args.n_threads;
    int64_t eval_cols = args.eval_cols;
    const int64_t eval_x = args.eval_x;
    const bool strip_dense = args.strip_dense;

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
        result.ok = false;
        result.report = std::move(report);
        return result;
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
                result.ok = false;
                result.report = std::move(report);
                return result;
            }
            if (cfg.require_imatrix && !have_imatrix) {
                fprintf(stderr, "seeddelta-build: gating metric requires imatrix for %s\n", weight_name.c_str());
                result.ok = false;
                result.report = std::move(report);
                return result;
            }

            if (!have_weight) {
                if (policy_ptr || !args.report_json.empty()) {
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

            const double w_l2_norm = (args.delta_norm_clamp_down > 0.0 && kind == "ffn_down")
                    ? sd_tensor_l2_norm(W)
                    : 0.0;

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
            const int64_t block_here = (scheme == sd_resid_scheme::block) ? cfg.block : block;
            const int64_t block_used = block_here; // shadow global block for per-tensor overrides

            std::vector<float> w_scale;
            const bool have_w = have_imatrix && sd_make_imatrix_sqrt_scale(imatrix_data, weight_name, n_in, imatrix_eps, imatrix_power, w_scale);
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

                if (scheme == sd_resid_scheme::block) {
                    n_blocks_keep = (K_eff + block_used - 1) / block_used;
                    K_eff = n_blocks_keep * block_used;
                } else {
                    n_blocks_keep = (K_eff + block_used - 1) / block_used;
                }

                t.re.layer = il;
                t.re.kind = kind;
                t.re.block = block_used;
                t.re.K_budget = K_budget_trial;
                t.re.K = K_eff;
                t.re.n_blocks = n_blocks_keep;
                t.re.n_in = n_in;
                t.re.n_out = n_out;
                t.re.strip_applied = cfg.strip_dense;
                t.re.metric_used = sd_report::metric_kind_to_string(cfg.metric);
                t.re.target_tau_mean = gating_min_mean;
                t.re.target_tau_p05 = gating_min_p05;
                t.re.has_w = have_w;
                t.re.cost = estimate_cost(write_base ? &base : nullptr, n_in, n_out, K_eff, write_row_scale);

                if (scheme == sd_resid_scheme::block) {
                    GGML_ASSERT(n_blocks_keep > 0);
                    t.b_idx.assign((size_t) n_blocks_keep * (size_t) n_out, -1);
                    t.b_val.assign((size_t) block_used * (size_t) n_blocks_keep * (size_t) n_out, 0.0f);
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

                                if (scheme == sd_resid_scheme::block) {
                                    const std::vector<float> & src_block = write_base ? r : w;
                                    topk_blocks_energy_weighted(src_block, have_w ? &w_scale : nullptr, block_used, n_blocks_keep, top_blocks);

                                    float ss = 1.0f;
                                    if (write_row_scale) {
                                        if (write_base) {
                                            for (int64_t bi = 0; bi < n_blocks_keep; ++bi) {
                                                const int32_t blk = top_blocks[(size_t) bi];
                                                if (blk < 0) {
                                                    continue;
                                                }
                                                const int64_t in0 = (int64_t) blk * block_used;
                                                const int64_t in1 = std::min<int64_t>(n_in, in0 + block_used);
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
                                                const int64_t in0 = (int64_t) blk * block_used;
                                                const int64_t in1 = std::min<int64_t>(n_in, in0 + block_used);
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

                                        const int64_t in0 = (blk >= 0) ? (int64_t) blk * block_used : 0;
                                        for (int64_t tt = 0; tt < block_used; ++tt) {
                                            const int64_t ii = in0 + tt;
                                            float vv = 0.0f;
                                            if (blk >= 0 && ii >= 0 && ii < n_in) {
                                                vv = write_base ? r[(size_t) ii] : w[(size_t) ii];
                                            }
                                            t.b_val[((size_t) col * (size_t) n_blocks_keep + (size_t) bi) * (size_t) block_used + (size_t) tt] = vv;
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

                if (args.delta_norm_clamp_down > 0.0 && kind == "ffn_down" && !t.d_val.empty() && w_l2_norm > 0.0) {
                    double delta_norm = 0.0;
                    for (float v : t.d_val) {
                        delta_norm += (double) v * (double) v;
                    }
                    delta_norm = delta_norm > 0.0 ? std::sqrt(delta_norm) : 0.0;
                    const double limit = args.delta_norm_clamp_down * w_l2_norm;
                    if (limit > 0.0 && delta_norm > limit) {
                        const double scale = limit / delta_norm;
                        for (float & v : t.d_val) {
                            v = (float) (v * scale);
                        }
                        fprintf(stderr, "  [blk.%" PRId64 ".%s] delta clamp down: norm=%.3g w_norm=%.3g scale=%.4f\n",
                                il, kind.c_str(), delta_norm, w_l2_norm, scale);
                    }
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
                    if (scheme == sd_resid_scheme::block) {
                        em = write_base
                                ? eval_seeddelta_base_block_residual(W, base, t.b_idx, t.b_val, block_used, n_blocks_keep, write_row_scale ? &t.d_row_scale : nullptr, nullptr, eval_cols, rng)
                                : eval_block_residual(W, t.b_idx, t.b_val, block_used, n_blocks_keep, write_row_scale ? &t.d_row_scale : nullptr, nullptr, eval_cols, rng);
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
                        if (scheme == sd_resid_scheme::block) {
                            em_w = write_base
                                    ? eval_seeddelta_base_block_residual(W, base, t.b_idx, t.b_val, block_used, n_blocks_keep, write_row_scale ? &t.d_row_scale : nullptr, &w_scale, eval_cols, rng)
                                    : eval_block_residual(W, t.b_idx, t.b_val, block_used, n_blocks_keep, write_row_scale ? &t.d_row_scale : nullptr, &w_scale, eval_cols, rng);
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
                    if (scheme == sd_resid_scheme::block) {
                        emx = eval_seeddelta_x_block(W, write_base ? &base : nullptr, t.b_idx, t.b_val, block_used, n_blocks_keep, write_row_scale ? &t.d_row_scale : nullptr, nullptr, eval_cols, eval_x, rng);
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
                        if (scheme == sd_resid_scheme::block) {
                            emx_w = eval_seeddelta_x_block(W, write_base ? &base : nullptr, t.b_idx, t.b_val, block_used, n_blocks_keep, write_row_scale ? &t.d_row_scale : nullptr, &w_scale, eval_cols, eval_x, rng);
                        } else {
                            emx_w = eval_seeddelta_x(W, write_base ? &base : nullptr, t.d_idx, t.d_val, write_row_scale ? &t.d_row_scale : nullptr, &w_scale, K_eff, eval_cols, eval_x, rng);
                        }
                        t.re.em_x_w = emx_w;
                        fprintf(stderr, "  [blk.%" PRId64 ".%s] eval_x_w x=%" PRId64 " cols=%" PRId64 " rel_l2 mean=%.4f p95=%.4f cos mean=%.4f p05=%.4f nr=%.4f\n",
                                il, kind.c_str(), eval_x, eval_cols, emx_w.rel_l2_mean, emx_w.rel_l2_p95, emx_w.cos_mean, emx_w.cos_p05, emx_w.norm_ratio_mean);
                    }
                }

                t.re.ffn_score = sd_report::ffn_score_from_entry(t.re);
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

            auto select_gating_metric = [&](const report_entry & re) -> sd_metric_kind {
                const bool prefer_ffn_block = (kind == "ffn_gate" || kind == "ffn_up");
                if (prefer_ffn_block && re.has_x) {
                    return sd_metric_kind::ffn_score;
                }
                return cfg.metric;
            };

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
                td.re.gating_min_mean = gating_min_mean;
                td.re.gating_min_p05  = gating_min_p05;

                const sd_metric_kind gating_metric = select_gating_metric(td.re);
                td.re.gating_metric_used = sd_report::metric_kind_to_string(gating_metric);
                td.re.metric_used = td.re.gating_metric_used;

                const bool metric_needs_x = (gating_metric == sd_metric_kind::cos_x || gating_metric == sd_metric_kind::cos_x_w || gating_metric == sd_metric_kind::ffn_score);
                const bool metric_needs_w = (gating_metric == sd_metric_kind::cos_w || gating_metric == sd_metric_kind::cos_x_w);
                bool metric_available = true;
                if (cfg.gating_enabled) {
                    if (metric_needs_x && !td.re.has_x) {
                        metric_available = false;
                    }
                    if (metric_needs_w && !have_w) {
                        metric_available = false;
                    }
                }

                const double metric_val = sd_report::pick_metric_value(td.re, gating_metric);
                const double metric_p05 = sd_report::pick_metric_p05(td.re, gating_metric);
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
                best_trial.re.metric_used = best_trial.re.gating_metric_used;
                best_trial.re.gating_min_mean = gating_min_mean;
                best_trial.re.gating_min_p05 = gating_min_p05;
                best_trial.re.target_tau_mean = gating_min_mean;
                best_trial.re.target_tau_p05 = gating_min_p05;
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
            // Populate tile metadata: if --tile-rows is set, emit one tile per contiguous chunk; otherwise single tile.
            {
                const int64_t align = std::max<int64_t>(1, args.tile_rows_align);
                const int64_t base_rows = (args.tile_rows > 0) ? std::min<int64_t>(args.tile_rows, re.n_out) : re.n_out;
                const int64_t tile_rows_aligned = (base_rows + align - 1) / align * align;
                const int64_t tile_rows = std::min<int64_t>(tile_rows_aligned, re.n_out > 0 ? re.n_out : 0);
                const int64_t n_tiles = (tile_rows > 0 && re.n_out > 0) ? (re.n_out + tile_rows - 1) / tile_rows : (re.n_out > 0 ? 1 : 0);

                re.tile_rows = tile_rows;
                re.tile_rows_align = align;
                // choose K levels: if user provided, use unique positive sorted, else fallback to K_eff
                std::vector<int64_t> k_levels = args.K_levels;
                k_levels.erase(std::remove_if(k_levels.begin(), k_levels.end(), [](int64_t v){ return v <= 0; }), k_levels.end());
                std::sort(k_levels.begin(), k_levels.end());
                k_levels.erase(std::unique(k_levels.begin(), k_levels.end()), k_levels.end());
                if (k_levels.empty()) {
                    k_levels.push_back(K_eff);
                }
                re.k_levels = k_levels;

                double kselector_gap = 0.0;
                int64_t kselector_tiles = 0;
                int64_t kselector_rank = 0;
                std::vector<int64_t> k_per_tile;
                const std::string selector = args.k_selector.empty() ? "cycle" : args.k_selector;
                if (selector == "uniform") {
                    kselector_rank = 1;
                    kselector_tiles = n_tiles;
                    k_per_tile.assign((size_t) n_tiles, k_levels.empty() ? K_eff : k_levels.back());
                } else if (selector == "cycle") {
                    kselector_rank = 1;
                    kselector_tiles = n_tiles;
                    k_per_tile.reserve((size_t) n_tiles);
                    for (int64_t ti = 0; ti < n_tiles; ++ti) {
                        const int64_t k_here = k_levels[(size_t) ti % k_levels.size()];
                        k_per_tile.push_back(k_here);
                    }
                } else { // ttcross heuristic
                    std::vector<sd_tile_stats> stats = sd_compute_tile_stats(W, tile_rows, n_in, n_out);
                    k_per_tile = sd_assign_k_per_tile(stats, k_levels, args, n_in, kselector_gap, kselector_tiles, kselector_rank);
                    if (k_per_tile.empty()) {
                        k_per_tile.reserve((size_t) n_tiles);
                        for (int64_t ti = 0; ti < n_tiles; ++ti) {
                            const int64_t k_here = k_levels[(size_t) ti % k_levels.size()];
                            k_per_tile.push_back(k_here);
                        }
                        kselector_rank = 1;
                    }
                }

                re.k_per_tile = std::move(k_per_tile);
                re.k_total_per_tensor = 0;
                for (int64_t v : re.k_per_tile) re.k_total_per_tensor += v;
                std::unordered_set<int64_t> uniq(re.k_per_tile.begin(), re.k_per_tile.end());
                re.unique_k_count = (int64_t) uniq.size();
                const bool rounded = (re.n_out > 0) && (re.n_out % tile_rows != 0);
                re.tiles_rounded_count = rounded ? 1 : 0;
                re.tiles_rounded_pct = n_tiles > 0 ? (double) re.tiles_rounded_count * 100.0 / (double) n_tiles : 0.0;
                re.kselector_mode = args.k_selector;
                re.kselector_rank = kselector_rank;
                re.kselector_tiles_sampled = kselector_tiles;
                re.kselector_gap_vs_uniform = kselector_gap;
            }

            if (scheme == sd_resid_scheme::coo && !write_base && eval_x > 0) {
                const std::string wg_name = "blk." + std::to_string(il) + ".ffn_gate.weight";
                const std::string wu_name = "blk." + std::to_string(il) + ".ffn_up.weight";
                const std::string wd_name = "blk." + std::to_string(il) + ".ffn_down.weight";
                ggml_tensor * W_gate = ggml_get_tensor(ctx_data, wg_name.c_str());
                ggml_tensor * W_up   = ggml_get_tensor(ctx_data, wu_name.c_str());
                ggml_tensor * W_down = ggml_get_tensor(ctx_data, wd_name.c_str());

                ffn_proxy_metrics fpm;
                const int proxy_seed = args.seed + (int) il * 101 + (int) (kind == "ffn_gate" ? 1 : (kind == "ffn_up" ? 2 : 3));
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

            {
                const double metric_mean = re.gating_value;
                const double metric_p05  = re.gating_p05;
                const double tau_mean    = re.target_tau_mean;
                const double tau_p05     = re.target_tau_p05;
                double cost_delta = 0.0;
                if (re.emit && std::isfinite(metric_mean) && std::isfinite(metric_p05)) {
                    const double d_mean = std::max(0.0, tau_mean - metric_mean);
                    const double d_p05  = std::max(0.0, tau_p05  - metric_p05);
                    cost_delta = d_mean + d_p05;
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

            const size_t size_idx = scheme == sd_resid_scheme::block
                    ? (size_t) n_blocks_keep * (size_t) n_out * ggml_type_size(idx_type)
                    : (size_t) K_eff * (size_t) n_out * ggml_type_size(idx_type);
            const size_t size_val = scheme == sd_resid_scheme::block
                    ? (size_t) block_used * (size_t) n_blocks_keep * (size_t) n_out * ggml_type_size(val_type)
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
            result.sd_contexts.push_back(ctx_sd);

            ggml_tensor * t_idx = nullptr;
            ggml_tensor * t_val = nullptr;
            if (scheme == sd_resid_scheme::block) {
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

                t_val = ggml_new_tensor_3d(ctx_sd, val_type, block_used, n_blocks_keep, n_out);
                ggml_set_name(t_val, b_val_name.c_str());
                if (val_type == GGML_TYPE_F16) {
                    auto * dst_f16 = (ggml_fp16_t *) t_val->data;
                    for (int64_t col = 0; col < n_out; ++col) {
                        for (int64_t bi = 0; bi < n_blocks_keep; ++bi) {
                            ggml_fp32_to_fp16_row(
                                    b_val.data() + ((size_t) col * (size_t) n_blocks_keep + (size_t) bi) * (size_t) block_used,
                                    dst_f16 + ((size_t) col * (size_t) n_blocks_keep + (size_t) bi) * (size_t) block_used,
                                    (int) block_used);
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

    result.n_added = n_added;
    result.report = std::move(report);
    result.pending = std::move(pending);
    result.strip_weights = std::move(strip_weights);
    result.any_strip = any_strip;
    result.stack_guard = stack_guard;
    return result;
}
