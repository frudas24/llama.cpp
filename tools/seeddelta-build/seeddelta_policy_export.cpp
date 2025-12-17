#include "seeddelta_policy_export.h"

#include <fstream>
#include <unordered_map>

#include <nlohmann/json.hpp>

using json = nlohmann::ordered_json;

sd_policy_export_result sd_policy_export_write_canonical(
        const std::string & out_path,
        const std::vector<sd_tensor_decision> & decisions) {
    sd_policy_export_result res;

    json j;
    j["version"] = 1;
    j["defaults"] = json::object();
    j["defaults"]["enabled"] = false;
    j["defaults"]["strip_dense"] = false;
    j["defaults"]["gating"] = json::object();
    j["defaults"]["gating"]["metric"] = "cos";
    j["defaults"]["gating"]["min_mean"] = -1.0;
    j["defaults"]["gating"]["min_p05"] = -1.0;
    j["defaults"]["autotune"] = json::object();
    j["defaults"]["autotune"]["enabled"] = false;

    json layers = json::object();

    // Emit explicit per-tensor rules for every decision, including disabled ones.
    for (const auto & d : decisions) {
        if (d.layer < 0) {
            continue;
        }
        const std::string layer_key = std::to_string(d.layer);
        json & layer_obj = layers[layer_key];
        if (!layer_obj.is_object()) {
            layer_obj = json::object();
        }
        if (!layer_obj.contains("tensors") || !layer_obj["tensors"].is_object()) {
            layer_obj["tensors"] = json::object();
        }

        json tr = json::object();
        tr["enabled"] = d.enabled;
        tr["strip_dense"] = d.strip_dense;
        if (d.block > 0) {
            tr["block"] = d.block;
        }
        if (d.enabled && d.K_budget > 0) {
            tr["K"] = d.K_budget;
        }
        layer_obj["tensors"][d.kind] = std::move(tr);
    }

    j["layers"] = std::move(layers);

    std::ofstream out(out_path);
    if (!out) {
        res.error = "failed to open policy export path: " + out_path;
        return res;
    }
    out << j.dump(2) << "\n";
    if (!out) {
        res.error = "failed to write policy export: " + out_path;
        return res;
    }

    res.ok = true;
    return res;
}

