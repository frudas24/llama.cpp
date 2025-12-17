#include "seeddelta_policy.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>

#include "vendor/nlohmann/json.hpp"

using json = nlohmann::ordered_json;

static sd_metric_kind sd_metric_from_string(const std::string & s) {
    if (s == "cos_x_w" || s == "cos-x-w") return sd_metric_kind::cos_x_w;
    if (s == "cos_x"   || s == "cos-x")   return sd_metric_kind::cos_x;
    if (s == "cos_w"   || s == "cos-w")   return sd_metric_kind::cos_w;
    return sd_metric_kind::cos;
}

static void sd_merge_rule(sd_rule & dst, const sd_rule & src) {
    auto merge_opt = [](auto & d, const auto & s) {
        if (s.has_value()) d = s;
    };
    merge_opt(dst.enabled, src.enabled);
    merge_opt(dst.strip_dense, src.strip_dense);
    merge_opt(dst.block, src.block);
    merge_opt(dst.K_gate, src.K_gate);
    merge_opt(dst.K_up, src.K_up);
    merge_opt(dst.K_down, src.K_down);
    merge_opt(dst.metric, src.metric);
    merge_opt(dst.autotune, src.autotune);

    auto merge_thr = [](sd_gating_thresholds & d, const sd_gating_thresholds & s) {
        if (s.min_mean.has_value()) d.min_mean = s.min_mean;
        if (s.min_p05.has_value())  d.min_p05  = s.min_p05;
    };
    merge_thr(dst.gate_thr, src.gate_thr);
    merge_thr(dst.up_thr,   src.up_thr);
    merge_thr(dst.down_thr, src.down_thr);

    for (const auto & kv : src.tensors) {
        auto & tdst = dst.tensors[kv.first];
        sd_merge_rule(tdst, kv.second);
    }
}

std::vector<int64_t> sd_parse_layer_selector(const std::string & s) {
    std::vector<int64_t> out;
    size_t pos = 0;
    auto push_range = [&](int64_t a, int64_t b) {
        if (a > b) std::swap(a, b);
        for (int64_t v = a; v <= b; ++v) out.push_back(v);
    };
    while (pos < s.size()) {
        while (pos < s.size() && std::isspace((unsigned char) s[pos])) ++pos;
        size_t start = pos;
        while (pos < s.size() && s[pos] != ',' ) ++pos;
        std::string token = s.substr(start, pos - start);
        if (!token.empty()) {
            auto dash = token.find('-');
            if (dash == std::string::npos) {
                out.push_back(std::stoll(token));
            } else {
                int64_t a = std::stoll(token.substr(0, dash));
                int64_t b = std::stoll(token.substr(dash + 1));
                push_range(a, b);
            }
        }
        if (pos < s.size() && s[pos] == ',') ++pos;
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

static std::optional<sd_autotune_cfg> sd_parse_autotune(const json & j) {
    if (!j.is_object()) {
        return std::nullopt;
    }
    sd_autotune_cfg cfg;
    if (j.contains("enabled")) cfg.enabled = j["enabled"].get<bool>();
    if (j.contains("max_iters")) cfg.max_iters = j["max_iters"].get<int>();
    auto parse_vec = [](const json & arr) {
        std::vector<int64_t> v;
        if (arr.is_array()) {
            for (const auto & el : arr) {
                if (el.is_number_integer()) v.push_back(el.get<int64_t>());
            }
        }
        return v;
    };
    if (j.contains("schedule_gate_up")) cfg.schedule_gate_up = parse_vec(j["schedule_gate_up"]);
    if (j.contains("schedule_down"))    cfg.schedule_down    = parse_vec(j["schedule_down"]);
    return cfg;
}

static void sd_parse_rule_object(const json & j, sd_rule & rule, bool strict, std::vector<std::string> & warnings) {
    auto warn_unknown = [&](const std::string & key) {
        if (!strict) warnings.push_back("unknown key in policy: " + key);
    };
    for (auto it = j.begin(); it != j.end(); ++it) {
        const std::string key = it.key();
        if (key == "enabled" || key == "enable") {
            rule.enabled = it.value().get<bool>();
        } else if (key == "strip_dense" || key == "strip") {
            rule.strip_dense = it.value().get<bool>();
        } else if (key == "block") {
            rule.block = it.value().get<int64_t>();
        } else if (key == "K") {
            if (it.value().is_object()) {
                if (it.value().contains("gate")) rule.K_gate = it.value()["gate"].get<int64_t>();
                if (it.value().contains("up"))   rule.K_up   = it.value()["up"].get<int64_t>();
                if (it.value().contains("down")) rule.K_down = it.value()["down"].get<int64_t>();
            } else if (it.value().is_number_integer()) {
                int64_t v = it.value().get<int64_t>();
                rule.K_gate = rule.K_up = rule.K_down = v;
            }
        } else if (key == "metric" || key == "gating_metric") {
            if (it.value().is_string()) {
                rule.metric = sd_metric_from_string(it.value().get<std::string>());
            }
        } else if (key == "gating") {
            const json & g = it.value();
            if (g.contains("metric") && g["metric"].is_string()) {
                rule.metric = sd_metric_from_string(g["metric"].get<std::string>());
            }
            if (g.contains("min_mean")) {
                const auto & mm = g["min_mean"];
                if (mm.is_number()) {
                    float v = mm.get<float>();
                    rule.gate_thr.min_mean = v;
                    rule.up_thr.min_mean   = v;
                    rule.down_thr.min_mean = v;
                } else if (mm.is_object()) {
                    if (mm.contains("gate") && mm["gate"].is_number()) rule.gate_thr.min_mean = mm["gate"].get<float>();
                    if (mm.contains("up")   && mm["up"].is_number())   rule.up_thr.min_mean   = mm["up"].get<float>();
                    if (mm.contains("down") && mm["down"].is_number()) rule.down_thr.min_mean = mm["down"].get<float>();
                } else {
                    warnings.push_back("gating.min_mean is not number/object");
                }
            }
            if (g.contains("min_p05")) {
                const auto & mp = g["min_p05"];
                if (mp.is_number()) {
                    float v = mp.get<float>();
                    rule.gate_thr.min_p05 = v;
                    rule.up_thr.min_p05   = v;
                    rule.down_thr.min_p05 = v;
                } else if (mp.is_object()) {
                    if (mp.contains("gate") && mp["gate"].is_number()) rule.gate_thr.min_p05 = mp["gate"].get<float>();
                    if (mp.contains("up")   && mp["up"].is_number())   rule.up_thr.min_p05   = mp["up"].get<float>();
                    if (mp.contains("down") && mp["down"].is_number()) rule.down_thr.min_p05 = mp["down"].get<float>();
                } else {
                    warnings.push_back("gating.min_p05 is not number/object");
                }
            }
        } else if (key == "autotune") {
            auto at = sd_parse_autotune(it.value());
            if (at.has_value()) {
                rule.autotune = at;
            }
        } else if (key == "tensors") {
            if (!it.value().is_object()) continue;
            for (auto jt = it.value().begin(); jt != it.value().end(); ++jt) {
                const std::string tname = jt.key();
                sd_rule tr;
                if (jt.value().is_object()) {
                    sd_parse_rule_object(jt.value(), tr, strict, warnings);
                    rule.tensors[tname] = std::move(tr);
                }
            }
        } else if (key == "layers" || key == "version") {
            // handled elsewhere
        } else {
            warn_unknown(key);
        }
    }
}

sd_policy_parse_result sd_policy_load(const std::string & path, bool strict, sd_policy & out) {
    sd_policy_parse_result res;
    out = sd_policy{};
    out.strict = strict;
    out.source_path = path;

    std::ifstream in(path);
    if (!in) {
        res.error = "failed to open policy file " + path;
        return res;
    }
    json j;
    try {
        in >> j;
    } catch (const std::exception & e) {
        res.error = std::string("failed to parse policy JSON: ") + e.what();
        return res;
    }

    if (j.contains("version") && j["version"].is_number_integer()) {
        out.version = j["version"].get<int>();
    }

    if (j.contains("defaults") && j["defaults"].is_object()) {
        sd_parse_rule_object(j["defaults"], out.defaults, strict, out.warnings);
    }

    if (j.contains("ranges") && j["ranges"].is_array()) {
        for (const auto & r : j["ranges"]) {
            if (!r.is_object() || !r.contains("layers")) {
                continue;
            }
            sd_policy::range_rule rr;
            try {
                rr.layers = sd_parse_layer_selector(r["layers"].get<std::string>());
            } catch (const std::exception & e) {
                out.warnings.push_back(std::string("invalid layers in range: ") + e.what());
                continue;
            }
            sd_parse_rule_object(r, rr.rule, strict, out.warnings);
            out.ranges.push_back(std::move(rr));
        }
    }

    if (j.contains("layers") && j["layers"].is_object()) {
        for (auto it = j["layers"].begin(); it != j["layers"].end(); ++it) {
            const std::string key = it.key();
            int64_t lid = -1;
            try {
                lid = std::stoll(key);
            } catch (...) {
                out.warnings.push_back("invalid layer key: " + key);
                continue;
            }
            if (!it.value().is_object()) {
                continue;
            }
            sd_rule r;
            sd_parse_rule_object(it.value(), r, strict, out.warnings);
            out.layers[lid] = std::move(r);
        }
    }

    res.ok = true;
    res.warnings = out.warnings;
    return res;
}

static std::optional<float> pick_kind_opt(const sd_gating_thresholds & t, const std::string & kind) {
    if (kind == "ffn_gate") return t.min_mean;
    if (kind == "ffn_up")   return t.min_mean;
    if (kind == "ffn_down") return t.min_mean;
    return t.min_mean;
}

static std::optional<float> pick_kind_p05(const sd_gating_thresholds & t, const std::string & kind) {
    if (kind == "ffn_gate") return t.min_p05;
    if (kind == "ffn_up")   return t.min_p05;
    if (kind == "ffn_down") return t.min_p05;
    return t.min_p05;
}

sd_resolved_tensor sd_policy_resolve(
        const sd_policy * policy,
        int64_t layer,
        const std::string & tensor_kind,
        const sd_resolved_tensor & baseline) {
    sd_resolved_tensor out = baseline;
    if (!policy) {
        out.gating_enabled = false;
        return out;
    }

    sd_rule acc;
    sd_merge_rule(acc, policy->defaults);

    for (const auto & rr : policy->ranges) {
        if (std::binary_search(rr.layers.begin(), rr.layers.end(), layer)) {
            sd_merge_rule(acc, rr.rule);
        }
    }

    const auto it_layer = policy->layers.find(layer);
    if (it_layer != policy->layers.end()) {
        sd_merge_rule(acc, it_layer->second);
        const auto it_tensor = it_layer->second.tensors.find(tensor_kind);
        if (it_tensor != it_layer->second.tensors.end()) {
            sd_merge_rule(acc, it_tensor->second);
        }
    }

    if (acc.enabled.has_value()) out.enabled = *acc.enabled;
    if (acc.strip_dense.has_value()) out.strip_dense = *acc.strip_dense;
    if (acc.block.has_value()) out.block = *acc.block;
    if (acc.K_gate.has_value()) out.K_gate = *acc.K_gate;
    if (acc.K_up.has_value())   out.K_up   = *acc.K_up;
    if (acc.K_down.has_value()) out.K_down = *acc.K_down;
    if (acc.metric.has_value()) out.metric = *acc.metric;
    if (acc.autotune.has_value()) out.autotune = *acc.autotune;

    auto pick_thr = [&](const sd_gating_thresholds & thr, float fallback) -> float {
        auto v = pick_kind_opt(thr, tensor_kind);
        return v.has_value() ? *v : fallback;
    };
    auto pick_thr_p05 = [&](const sd_gating_thresholds & thr, float fallback) -> float {
        auto v = pick_kind_p05(thr, tensor_kind);
        return v.has_value() ? *v : fallback;
    };

    out.min_mean = pick_thr(acc.gate_thr, out.min_mean);
    out.min_p05  = pick_thr_p05(acc.gate_thr, out.min_p05);
    // If per-kind thresholds are set in up/down, they override.
    if (tensor_kind == "ffn_up") {
        if (acc.up_thr.min_mean.has_value()) out.min_mean = *acc.up_thr.min_mean;
        if (acc.up_thr.min_p05.has_value())  out.min_p05  = *acc.up_thr.min_p05;
    } else if (tensor_kind == "ffn_down") {
        if (acc.down_thr.min_mean.has_value()) out.min_mean = *acc.down_thr.min_mean;
        if (acc.down_thr.min_p05.has_value())  out.min_p05  = *acc.down_thr.min_p05;
    }

    out.gating_enabled = true;
    out.require_eval_x = (out.metric == sd_metric_kind::cos_x || out.metric == sd_metric_kind::cos_x_w);
    out.require_imatrix = (out.metric == sd_metric_kind::cos_w || out.metric == sd_metric_kind::cos_x_w);

    return out;
}
