#include "include/sc_process.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <regex>
#include <stdexcept>

#include "include/sc_encode.h"
#include "include/sc_eval.h"
#include "include/sc_imatrix.h"
#include "include/sc_utils.h"

namespace {

struct sc_weight_names {
    std::string weight;
    std::string dict;
    std::string codes;
    std::string vals;
    std::string row_scale;
};

sc_weight_names make_weight_names(int64_t layer, const std::string & kind) {
    const std::string prefix = "blk." + std::to_string(layer) + "." + kind;
    return {
        prefix + ".weight",
        prefix + ".dict",
        prefix + ".codes",
        prefix + ".vals",
        prefix + ".row_scale",
    };
}

void flush_checkpoint(const sc_args & args, const sc_process_result & res, gguf_context * dst) {
    gguf_set_val_bool(dst, "statecells.enabled", (res.n_sc_existing + res.n_added) > 0);
    gguf_set_val_u32(dst, "statecells.dict.M", (uint32_t) args.dict_M);
    gguf_set_val_u32(dst, "statecells.dict.k", (uint32_t) args.dict_k);
    gguf_set_val_f32(dst, "statecells.dict.eta", args.dict_eta);
    gguf_set_val_u32(dst, "statecells.dict.iters", (uint32_t) args.dict_iters);
    printf("checkpoint: writing %s (%" PRId64 " new tensors so far)\n", args.out_fname.c_str(), res.n_added);
    gguf_write_to_file(dst, args.out_fname.c_str(), false);
}

bool process_weight(
        const sc_args & args,
        gguf_context * src,
        ggml_context * ctx_data,
        gguf_context * dst,
        const std::unordered_map<std::string, std::vector<float>> & imatrix_data,
        std::mt19937 & rng,
        int64_t il,
        const std::string & kind,
        ggml_type dict_type,
        sc_process_result & result) {
    const auto names = make_weight_names(il, kind);

    const bool have_dict  = gguf_find_tensor(src, names.dict.c_str())  != -1;
    const bool have_codes = gguf_find_tensor(src, names.codes.c_str()) != -1;
    const bool have_vals  = gguf_find_tensor(src, names.vals.c_str())  != -1;
    const bool have_row_scale = gguf_find_tensor(src, names.row_scale.c_str()) != -1;

    if ((have_dict || have_codes || have_vals || have_row_scale) && (have_dict != have_codes || (!have_dict && (have_vals || have_row_scale)))) {
        fprintf(stderr, "statecells-build: inconsistent existing tensors for %s (dict=%d codes=%d row_scale=%d)\n",
                names.weight.c_str(), (int) have_dict, (int) have_codes, (int) have_row_scale);
        result.ok = false;
        return false;
    }

    const bool need_dict_codes = !(have_dict && have_codes);
    const bool need_vals       = args.write_vals && !have_vals;
    const bool need_row_scale  = args.write_row_scale && !have_row_scale;
    if (!need_dict_codes && !need_vals && !need_row_scale) {
        return true;
    }

    ggml_tensor * W = ggml_get_tensor(ctx_data, names.weight.c_str());
    if (!W || ggml_n_dims(W) != 2) {
        return true;
    }

    const int64_t n_in  = W->ne[0];
    const int64_t n_out = W->ne[1];

    const auto select_M_for_kind = [&](const std::string & k) -> int64_t {
        if (k == "ffn_gate" && args.dict_M_gate > 0) return args.dict_M_gate;
        if (k == "ffn_up"   && args.dict_M_up   > 0) return args.dict_M_up;
        if (k == "ffn_down" && args.dict_M_down > 0) return args.dict_M_down;
        return args.dict_M;
    };
    const auto select_k_for_kind = [&](const std::string & k) -> int {
        if (k == "ffn_gate" && args.dict_k_gate > 0) return args.dict_k_gate;
        if (k == "ffn_up"   && args.dict_k_up   > 0) return args.dict_k_up;
        if (k == "ffn_down" && args.dict_k_down > 0) return args.dict_k_down;
        return args.dict_k;
    };

    const int64_t dict_M_cur = select_M_for_kind(kind);
    const int dict_k_cur = select_k_for_kind(kind);
    const std::string tag = "blk." + std::to_string(il) + "." + kind;

    std::vector<float> imatrix_w_scale_sqrt;
    std::vector<float> imatrix_w_scale2;
    const std::vector<float> * w_scale_train = nullptr;
    const std::vector<float> * w_scale_encode = nullptr;

    if (!imatrix_data.empty()) {
        if (make_imatrix_sqrt_scale(imatrix_data, names.weight, n_in, args.imatrix_eps, args.imatrix_power, imatrix_w_scale_sqrt)) {
            w_scale_train = &imatrix_w_scale_sqrt;
            imatrix_w_scale2 = imatrix_w_scale_sqrt;
            for (float & v : imatrix_w_scale2) {
                v *= v;
            }
            w_scale_encode = &imatrix_w_scale2;
        } else {
            fprintf(stderr, "statecells-build: warning: missing imatrix entry for %s; proceeding unweighted\n", names.weight.c_str());
        }
    }

    int64_t M_eff = 0;
    int k_eff = 0;
    std::vector<float> D;
    std::vector<int16_t> codes;
    std::vector<ggml_fp16_t> vals;

    if (need_dict_codes) {
        printf("statecells-build: layer %" PRId64 " %s [% " PRId64 " x %" PRId64 "] type=%s M=%" PRId64 " k=%d\n",
               il, kind.c_str(), n_in, n_out, ggml_type_name(W->type), dict_M_cur, dict_k_cur);

        train_dict_oja_kwta(W, dict_M_cur, dict_k_cur, args.dict_eta, args.dict_iters, args.dict_max_samples, rng, D, tag, w_scale_train);

        M_eff = std::min<int64_t>(dict_M_cur, n_out);
        k_eff = std::min<int>(dict_k_cur, (int) M_eff);

        if (w_scale_train) {
            unscale_dict_inplace(D, n_in, M_eff, *w_scale_train);
        }

        encode_codes_sign(W, D, M_eff, k_eff, args.n_threads, codes, need_vals ? &vals : nullptr, tag, w_scale_encode);
    } else {
        ggml_tensor * t_dict  = ggml_get_tensor(ctx_data, names.dict.c_str());
        ggml_tensor * t_codes = ggml_get_tensor(ctx_data, names.codes.c_str());
        if (!t_dict || !t_codes) {
            fprintf(stderr, "statecells-build: missing existing dict/codes tensors for %s\n", names.weight.c_str());
            result.ok = false;
            return false;
        }
        if (ggml_n_dims(t_dict) != 2 || ggml_n_dims(t_codes) != 2 || t_codes->type != GGML_TYPE_I16) {
            fprintf(stderr, "statecells-build: invalid existing dict/codes tensor types for %s\n", names.weight.c_str());
            result.ok = false;
            return false;
        }

        M_eff = t_dict->ne[1];
        k_eff = (int) t_codes->ne[0];

        if (t_dict->ne[0] != n_in || t_codes->ne[1] != n_out) {
            fprintf(stderr, "statecells-build: mismatched shapes for %s (W=[%" PRId64 "x%" PRId64 "] dict=[%" PRId64 "x%" PRId64 "] codes=[%" PRId64 "x%" PRId64 "])\n",
                    names.weight.c_str(), n_in, n_out,
                    (int64_t) t_dict->ne[0], (int64_t) t_dict->ne[1],
                    (int64_t) t_codes->ne[0], (int64_t) t_codes->ne[1]);
            result.ok = false;
            return false;
        }

        std::string add_what;
        if (need_vals) add_what += " vals";
        if (need_row_scale) add_what += " row_scale";
        printf("statecells-build: layer %" PRId64 " %s add%s only (M=%" PRId64 " k=%d)\n",
               il, kind.c_str(), add_what.c_str(), M_eff, k_eff);

        read_mat_f16_f32_to_f32(t_dict, D);
        codes.resize((size_t) k_eff * (size_t) n_out);
        std::memcpy(codes.data(), t_codes->data, codes.size() * sizeof(int16_t));

        if (need_vals) {
            compute_vals_sign(W, D, M_eff, k_eff, codes, args.n_threads, vals, tag, w_scale_encode);
        } else if (have_vals && (need_row_scale || args.eval_cols > 0)) {
            ggml_tensor * t_vals = ggml_get_tensor(ctx_data, names.vals.c_str());
            if (!t_vals || ggml_n_dims(t_vals) != 2) {
                fprintf(stderr, "statecells-build: missing existing vals tensor for %s\n", names.weight.c_str());
                result.ok = false;
                return false;
            }
            if (t_vals->type != GGML_TYPE_F16 && t_vals->type != GGML_TYPE_F32) {
                fprintf(stderr, "statecells-build: invalid existing vals tensor type for %s\n", names.weight.c_str());
                result.ok = false;
                return false;
            }
            if (t_vals->ne[0] != k_eff || t_vals->ne[1] != n_out) {
                fprintf(stderr, "statecells-build: mismatched vals shape for %s (vals=[%" PRId64 "x%" PRId64 "] expected=[%dx%" PRId64 "])\n",
                        names.weight.c_str(), (int64_t) t_vals->ne[0], (int64_t) t_vals->ne[1], k_eff, n_out);
                result.ok = false;
                return false;
            }

            read_mat_f16_f32_to_fp16(t_vals, vals);
        }
    }

    std::vector<ggml_fp16_t> row_scale;
    if (need_row_scale) {
        compute_row_scale_sign(W, D, M_eff, k_eff, codes, vals.empty() ? nullptr : &vals, args.n_threads, row_scale, tag);
    } else if (have_row_scale && args.eval_cols > 0) {
        ggml_tensor * t_rs = ggml_get_tensor(ctx_data, names.row_scale.c_str());
        if (!t_rs || ggml_n_dims(t_rs) != 1) {
            fprintf(stderr, "statecells-build: missing existing row_scale tensor for %s\n", names.weight.c_str());
            result.ok = false;
            return false;
        }
        if (t_rs->type != GGML_TYPE_F16 && t_rs->type != GGML_TYPE_F32) {
            fprintf(stderr, "statecells-build: invalid existing row_scale tensor type for %s\n", names.weight.c_str());
            result.ok = false;
            return false;
        }
        if (t_rs->ne[0] != n_out) {
            fprintf(stderr, "statecells-build: mismatched row_scale shape for %s (row_scale=[%" PRId64 "] expected=[%" PRId64 "])\n",
                    names.weight.c_str(), (int64_t) t_rs->ne[0], n_out);
            result.ok = false;
            return false;
        }

        read_vec_f16_f32_to_fp16(t_rs, row_scale);
    }

    eval_metrics em;
    if (args.eval_cols > 0) {
        const int64_t t0 = ggml_time_us();
        em = eval_reconstruction_sign(W, D, M_eff, k_eff, codes, vals.empty() ? nullptr : &vals, row_scale.empty() ? nullptr : &row_scale, args.eval_cols, rng, w_scale_train);
        const double sec = double(ggml_time_us() - t0) / 1e6;
        if (w_scale_train) {
            fprintf(stderr, "  [%s] eval cols=%" PRId64 " rel_l2 mean=%.4f p95=%.4f cos mean=%.4f p05=%.4f | rel_l2_w mean=%.4f p95=%.4f cos_w mean=%.4f p05=%.4f (%.1fs)\n",
                    tag.c_str(), args.eval_cols,
                    em.rel_l2_mean, em.rel_l2_p95, em.cos_mean, em.cos_p05,
                    em.rel_l2_mean_w, em.rel_l2_p95_w, em.cos_mean_w, em.cos_p05_w,
                    sec);
        } else {
            fprintf(stderr, "  [%s] eval cols=%" PRId64 " rel_l2 mean=%.4f p95=%.4f cos mean=%.4f p05=%.4f (%.1fs)\n",
                    tag.c_str(), args.eval_cols, em.rel_l2_mean, em.rel_l2_p95, em.cos_mean, em.cos_p05, sec);
        }
    }

    const size_t size_dict  = need_dict_codes ? (size_t) n_in * (size_t) M_eff * ggml_type_size(dict_type) : 0;
    const size_t size_codes = need_dict_codes ? (size_t) k_eff * (size_t) n_out * sizeof(int16_t) : 0;
    const size_t size_vals  = need_vals ? (size_t) k_eff * (size_t) n_out * sizeof(ggml_fp16_t) : 0;
    const size_t size_row_scale = need_row_scale ? (size_t) n_out * sizeof(ggml_fp16_t) : 0;
    const size_t mem_size_sc = ggml_tensor_overhead() * 8 + size_dict + size_codes + size_vals + size_row_scale;
    ggml_init_params sc_params = { mem_size_sc, nullptr, false };
    ggml_context * ctx_sc = ggml_init(sc_params);
    result.sc_contexts.push_back(ctx_sc);

    if (need_dict_codes) {
        ggml_tensor * t_dict = ggml_new_tensor_2d(ctx_sc, dict_type, n_in, M_eff);
        ggml_set_name(t_dict, names.dict.c_str());

        if (dict_type == GGML_TYPE_F16) {
            auto * dst_dict = (ggml_fp16_t *) t_dict->data;
            for (int64_t j = 0; j < M_eff; ++j) {
                ggml_fp32_to_fp16_row(D.data() + j * n_in, dst_dict + j * n_in, n_in);
            }
        } else {
            std::memcpy(t_dict->data, D.data(), (size_t) n_in * (size_t) M_eff * sizeof(float));
        }

        ggml_tensor * t_codes = ggml_new_tensor_2d(ctx_sc, GGML_TYPE_I16, k_eff, n_out);
        ggml_set_name(t_codes, names.codes.c_str());
        std::memcpy(t_codes->data, codes.data(), codes.size() * sizeof(int16_t));

        gguf_add_tensor(dst, t_dict);
        gguf_add_tensor(dst, t_codes);
        result.n_added += 2;
    }

    if (need_vals) {
        ggml_tensor * t_vals = ggml_new_tensor_2d(ctx_sc, GGML_TYPE_F16, k_eff, n_out);
        ggml_set_name(t_vals, names.vals.c_str());
        std::memcpy(t_vals->data, vals.data(), vals.size() * sizeof(ggml_fp16_t));
        gguf_add_tensor(dst, t_vals);
        result.n_added += 1;
    }

    if (need_row_scale) {
        ggml_tensor * t_rs = ggml_new_tensor_1d(ctx_sc, GGML_TYPE_F16, n_out);
        ggml_set_name(t_rs, names.row_scale.c_str());
        std::memcpy(t_rs->data, row_scale.data(), row_scale.size() * sizeof(ggml_fp16_t));
        gguf_add_tensor(dst, t_rs);
        result.n_added += 1;
    }

    if (!args.report_json.empty() || args.eval_cols > 0) {
        result.report_rows.push_back(sc_report_row{ il, kind, n_in, n_out, M_eff, k_eff, args.eval_cols, em, w_scale_train != nullptr });
    }

    return true;
}

} // namespace

sc_process_result sc_process_layers(
        const sc_args & args,
        gguf_context * src,
        ggml_context * ctx_data,
        gguf_context * dst,
        const std::unordered_map<std::string, std::vector<float>> & imatrix_data,
        std::mt19937 & rng) {
    sc_process_result result;

    const int64_t n_tensors = gguf_get_n_tensors(src);

    // Count existing StateCells payload tensors so --resume doesn't accidentally disable the feature.
    std::regex re_sc_payload(R"(blk\.(\d+)\.(ffn_gate|ffn_up|ffn_down)\.(dict|codes|row_scale)$)");
    for (int64_t ti = 0; ti < n_tensors; ++ti) {
        const char * name = gguf_get_tensor_name(src, ti);
        if (std::regex_match(name, re_sc_payload)) {
            ++result.n_sc_existing;
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
    auto layers = parse_layer_range(args.layers_range, n_layer);

    ggml_type dict_type = GGML_TYPE_F16;
    if (args.dict_type_str == "f32") dict_type = GGML_TYPE_F32;
    if (args.dict_type_str != "f16" && args.dict_type_str != "f32") {
        throw std::runtime_error("invalid --dict-type");
    }

    const std::vector<std::string> kinds = { "ffn_gate", "ffn_up", "ffn_down" };
    int64_t layers_since_checkpoint = 0;

    for (const int64_t il : layers) {
        bool layer_added_any = false;

        for (const auto & kind : kinds) {
            const int64_t added_before = result.n_added;
            const size_t reports_before = result.report_rows.size();
            const size_t ctx_before = result.sc_contexts.size();

            if (!process_weight(args, src, ctx_data, dst, imatrix_data, rng, il, kind, dict_type, result) || !result.ok) {
                return result;
            }

            if (result.n_added > added_before || result.report_rows.size() > reports_before || result.sc_contexts.size() > ctx_before) {
                layer_added_any = true;
            }
        }

        if (args.checkpoint_every > 0 && layer_added_any) {
            layers_since_checkpoint++;
            if (layers_since_checkpoint >= args.checkpoint_every) {
                flush_checkpoint(args, result, dst);
                layers_since_checkpoint = 0;
            }
        }
    }

    return result;
}
