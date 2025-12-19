#pragma once

#include <string>
#include <vector>

#include "seeddelta_policy.h"

struct sd_policy_state {
    bool has_policy = false;
    sd_policy policy;
    std::string hash;
    std::vector<std::string> warnings;
    std::string error;
};

// Loads a policy from JSON file (optional). If path is empty, returns has_policy=false.
// On success, fills policy, hash (fnv1a64 over file bytes) and warnings.
sd_policy_state sd_policy_load_from_file(const std::string & path, bool strict);
