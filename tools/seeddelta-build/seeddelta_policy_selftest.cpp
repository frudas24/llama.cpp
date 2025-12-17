#include "seeddelta_policy_selftest.h"

#include "seeddelta_policy.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#if defined(_WIN32)
// Self-test currently uses mkstemp+unlink; keep it POSIX-only for now.
#else
#include <unistd.h>
#endif

static bool sd_write_temp_file(const std::string & contents, std::string & path_out, std::string & err_out) {
#if defined(_WIN32)
    (void) contents;
    path_out.clear();
    err_out = "policy self-test is not supported on Windows yet";
    return false;
#else
    char tmpl[] = "/tmp/seeddelta_policy_testXXXXXX";
    const int fd = mkstemp(tmpl);
    if (fd < 0) {
        err_out = std::string("mkstemp failed: ") + std::strerror(errno);
        return false;
    }
    ::close(fd);

    path_out = tmpl;
    std::ofstream out(path_out);
    if (!out) {
        err_out = "failed to open temp file for writing: " + path_out;
        (void) ::unlink(path_out.c_str());
        return false;
    }
    out << contents;
    out.close();
    return true;
#endif
}

static void sd_cleanup_temp_file(const std::string & path) {
#if defined(_WIN32)
    (void) path;
#else
    if (!path.empty()) {
        (void) ::unlink(path.c_str());
    }
#endif
}

static bool sd_expect(bool cond, const char * msg) {
    if (!cond) {
        std::fprintf(stderr, "policy self-test FAILED: %s\n", msg);
        return false;
    }
    return true;
}

int sd_policy_self_test() {
    bool ok = true;

    // 1) Layer selector parsing.
    {
        const auto v = sd_parse_layer_selector("3-1,2,2,5");
        ok &= sd_expect(v.size() == 4, "sd_parse_layer_selector size");
        ok &= sd_expect(v.size() >= 4 && v[0] == 1 && v[1] == 2 && v[2] == 3 && v[3] == 5, "sd_parse_layer_selector values");
    }

    // 2) Unknown keys: lenient vs strict.
    {
        const std::string policy_text = R"JSON(
{
  "version": 1,
  "defaults": {
    "enabled": true,
    "unknown_key": 123
  }
}
)JSON";

        std::string path, err;
        ok &= sd_expect(sd_write_temp_file(policy_text, path, err), err.c_str());
        if (!path.empty()) {
            sd_policy pol;
            auto res = sd_policy_load(path, /* strict = */ false, pol);
            ok &= sd_expect(res.ok, "lenient policy parse should succeed");
            ok &= sd_expect(!res.warnings.empty(), "lenient policy parse should warn on unknown keys");

            sd_policy pol_strict;
            auto res2 = sd_policy_load(path, /* strict = */ true, pol_strict);
            ok &= sd_expect(!res2.ok, "strict policy parse should fail on unknown keys");
            sd_cleanup_temp_file(path);
        }
    }

    // 3) Merge precedence: CLI baseline -> defaults -> ranges (in order) -> layer -> tensor.
    {
        const std::string policy_text = R"JSON(
{
  "version": 1,
  "defaults": {
    "enabled": false,
    "strip_dense": false,
    "block": 64,
    "K": { "gate": 100, "up": 200, "down": 300 }
  },
  "ranges": [
    { "layers": "1-3", "enabled": true, "K": { "gate": 111 } },
    { "layers": "2-4", "strip_dense": true, "K": { "gate": 222 } }
  ],
  "layers": {
    "2": {
      "K": { "gate": 333 },
      "tensors": {
        "ffn_gate": { "K": 444 },
        "ffn_up":   { "enabled": false }
      }
    },
    "3": { "enabled": false }
  }
}
)JSON";

        std::string path, err;
        ok &= sd_expect(sd_write_temp_file(policy_text, path, err), err.c_str());
        if (!path.empty()) {
            sd_policy pol;
            auto res = sd_policy_load(path, /* strict = */ true, pol);
            ok &= sd_expect(res.ok, "policy parse failed (merge precedence test)");

            sd_resolved_tensor baseline;
            baseline.enabled = true;
            baseline.strip_dense = false;
            baseline.block = 11;
            baseline.K_gate = 1;
            baseline.K_up = 2;
            baseline.K_down = 3;
            baseline.metric = sd_metric_kind::cos_x_w;

            // layer0 ffn_gate: defaults only -> enabled=false, block=64, K_gate=100
            {
                auto r = sd_policy_resolve(&pol, 0, "ffn_gate", baseline);
                ok &= sd_expect(r.enabled == false, "layer0 enabled");
                ok &= sd_expect(r.block == 64, "layer0 block");
                ok &= sd_expect(r.K_gate == 100, "layer0 K_gate");
            }

            // layer1 ffn_gate: defaults + range1 -> enabled=true, K_gate=111, strip=false
            {
                auto r = sd_policy_resolve(&pol, 1, "ffn_gate", baseline);
                ok &= sd_expect(r.enabled == true, "layer1 enabled");
                ok &= sd_expect(r.K_gate == 111, "layer1 K_gate");
                ok &= sd_expect(r.strip_dense == false, "layer1 strip_dense");
            }

            // layer2 ffn_gate: defaults + range1 + range2 + layer2 + tensor override -> K_gate=444, strip=true
            {
                auto r = sd_policy_resolve(&pol, 2, "ffn_gate", baseline);
                ok &= sd_expect(r.enabled == true, "layer2 gate enabled");
                ok &= sd_expect(r.strip_dense == true, "layer2 gate strip_dense");
                ok &= sd_expect(r.K_gate == 444, "layer2 gate K_gate");
                ok &= sd_expect(r.block == 64, "layer2 gate block");
            }

            // layer2 ffn_up: tensor override disables
            {
                auto r = sd_policy_resolve(&pol, 2, "ffn_up", baseline);
                ok &= sd_expect(r.enabled == false, "layer2 up enabled override");
            }

            // layer3 ffn_gate: both ranges apply but layer disables -> enabled=false, K_gate from last matching range (222)
            {
                auto r = sd_policy_resolve(&pol, 3, "ffn_gate", baseline);
                ok &= sd_expect(r.enabled == false, "layer3 enabled");
                ok &= sd_expect(r.K_gate == 222, "layer3 K_gate from last range");
            }

            sd_cleanup_temp_file(path);
        }
    }

    // 4) Metric requirements (cos_x_w => require eval_x + imatrix).
    {
        const std::string policy_text = R"JSON(
{
  "version": 1,
  "defaults": {
    "enabled": true,
    "gating": { "metric": "cos_x_w", "min_mean": 0.1, "min_p05": 0.1 }
  }
}
)JSON";

        std::string path, err;
        ok &= sd_expect(sd_write_temp_file(policy_text, path, err), err.c_str());
        if (!path.empty()) {
            sd_policy pol;
            auto res = sd_policy_load(path, /* strict = */ true, pol);
            ok &= sd_expect(res.ok, "policy parse failed (metric requirements test)");

            sd_resolved_tensor baseline;
            baseline.metric = sd_metric_kind::cos; // overwritten by policy
            baseline.min_mean = 0.0f;
            baseline.min_p05 = 0.0f;

            auto r = sd_policy_resolve(&pol, 0, "ffn_gate", baseline);
            ok &= sd_expect(r.metric == sd_metric_kind::cos_x_w, "metric kind resolve");
            ok &= sd_expect(r.require_eval_x == true, "metric requires eval_x");
            ok &= sd_expect(r.require_imatrix == true, "metric requires imatrix");

            sd_cleanup_temp_file(path);
        }
    }

    if (ok) {
        std::fprintf(stderr, "policy self-test OK\n");
        return 0;
    }
    return 1;
}
