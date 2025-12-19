#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "sc_cli.h"
#include "sc_report.h"

int sc_build(
        const sc_args & args,
        const sc_report_config & report_cfg,
        int imatrix_chunk_count,
        const std::unordered_map<std::string, std::vector<float>> & imatrix_data);
