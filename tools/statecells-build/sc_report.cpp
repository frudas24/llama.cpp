#include "include/sc_report.h"

#include <fstream>

namespace {

std::string json_escape(const std::string & s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        if (c == '\\' || c == '"') {
            out.push_back('\\');
        }
        out.push_back(c);
    }
    return out;
}

std::string quote(const std::string & s) {
    return "\"" + json_escape(s) + "\"";
}

} // namespace

bool sc_write_report_json(
        const std::string & path,
        const sc_report_config & cfg,
        const std::vector<sc_report_row> & rows) {
    std::ofstream ofs(path);
    if (!ofs) {
        return false;
    }

    ofs << "{\n";
    ofs << "  \"input\": "   << quote(cfg.input)  << ",\n";
    ofs << "  \"source\": "  << quote(cfg.source) << ",\n";
    ofs << "  \"output\": "  << quote(cfg.output) << ",\n";
    ofs << "  \"resume\": "  << (cfg.resume ? "true" : "false") << ",\n";
    ofs << "  \"vals\": "    << (cfg.write_vals ? "true" : "false") << ",\n";
    ofs << "  \"row_scale\": " << (cfg.write_row_scale ? "true" : "false") << ",\n";

    ofs << "  \"imatrix\": ";
    if (!cfg.imatrix.enabled) {
        ofs << "null,\n";
    } else {
        ofs << "{\n";
        ofs << "    \"file\": " << quote(cfg.imatrix.file) << ",\n";
        if (!cfg.imatrix.dataset.empty()) {
            ofs << "    \"dataset\": " << quote(cfg.imatrix.dataset) << ",\n";
        }
        ofs << "    \"chunks\": " << cfg.imatrix.chunks << ",\n";
        ofs << "    \"eps\": " << cfg.imatrix.eps << ",\n";
        ofs << "    \"power\": " << cfg.imatrix.power << "\n";
        ofs << "  },\n";
    }

    ofs << "  \"dict\": {\n";
    ofs << "    \"M\": " << cfg.dict.M << ",\n";
    ofs << "    \"k\": " << cfg.dict.k << ",\n";
    ofs << "    \"eta\": " << cfg.dict.eta << ",\n";
    ofs << "    \"iters\": " << cfg.dict.iters << ",\n";
    ofs << "    \"max_samples\": " << cfg.dict.max_samples << "\n";
    ofs << "  },\n";

    ofs << "  \"eval_cols\": " << cfg.eval_cols << ",\n";
    ofs << "  \"weights\": [\n";
    for (size_t i = 0; i < rows.size(); ++i) {
        const auto & r = rows[i];
        ofs << "    {\n";
        ofs << "      \"layer\": " << r.layer << ",\n";
        ofs << "      \"kind\": " << quote(r.kind) << ",\n";
        ofs << "      \"n_in\": " << r.n_in << ",\n";
        ofs << "      \"n_out\": " << r.n_out << ",\n";
        ofs << "      \"M\": " << r.M_eff << ",\n";
        ofs << "      \"k\": " << r.k_eff << ",\n";
        ofs << "      \"imatrix\": " << (r.imatrix ? "true" : "false") << ",\n";
        ofs << "      \"rel_l2_mean\": " << r.em.rel_l2_mean << ",\n";
        ofs << "      \"rel_l2_p95\": "  << r.em.rel_l2_p95  << ",\n";
        ofs << "      \"cos_mean\": "    << r.em.cos_mean    << ",\n";
        ofs << "      \"cos_p05\": "     << r.em.cos_p05     << ",\n";
        ofs << "      \"rel_l2_mean_w\": " << r.em.rel_l2_mean_w << ",\n";
        ofs << "      \"rel_l2_p95_w\": "  << r.em.rel_l2_p95_w  << ",\n";
        ofs << "      \"cos_mean_w\": "    << r.em.cos_mean_w    << ",\n";
        ofs << "      \"cos_p05_w\": "     << r.em.cos_p05_w     << "\n";
        ofs << "    }" << (i + 1 == rows.size() ? "\n" : ",\n");
    }
    ofs << "  ]\n";
    ofs << "}\n";

    return true;
}
