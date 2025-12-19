#include "include/sd_policy_resolve.h"

#include <fstream>
#include <sstream>

#include "seeddelta_policy.h"

namespace {
std::string fnv1a_hex(const std::string & data) {
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

std::string slurp_file(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return {};
    }
    std::ostringstream oss;
    oss << in.rdbuf();
    return oss.str();
}
} // namespace

sd_policy_state sd_policy_load_from_file(const std::string & path, bool strict) {
    sd_policy_state st;
    if (path.empty()) {
        st.has_policy = false;
        return st;
    }

    auto res = sd_policy_load(path, strict, st.policy);
    st.warnings = res.warnings;
    st.error = res.error;
    if (!res.ok) {
        return st;
    }

    st.has_policy = true;
    const std::string policy_bytes = slurp_file(path);
    if (!policy_bytes.empty()) {
        st.hash = "fnv1a64:" + fnv1a_hex(policy_bytes);
    }
    return st;
}
