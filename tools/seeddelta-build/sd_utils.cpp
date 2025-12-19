#include "include/sd_utils.h"

#include <algorithm>
#include <cmath>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <numeric>
#include <regex>
#include <stdexcept>
#include <unordered_map>

#include "gguf.h"
#include "ggml.h"

std::vector<int64_t> sd_parse_layer_range(const std::string & s, int64_t n_layer) {
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

bool sd_string_remove_suffix(std::string & s, const std::string & suffix) {
    if (s.size() < suffix.size()) {
        return false;
    }
    if (s.compare(s.size() - suffix.size(), suffix.size(), suffix) != 0) {
        return false;
    }
    s.resize(s.size() - suffix.size());
    return true;
}

static int sd_load_legacy_imatrix(
        const std::string & imatrix_file,
        std::vector<std::string> & imatrix_datasets,
        std::unordered_map<std::string, std::vector<float>> & imatrix_data) {
    std::ifstream in(imatrix_file.c_str(), std::ios::binary);
    if (!in) {
        fprintf(stderr, "%s: failed to open %s\n", __func__, imatrix_file.c_str());
        return -1;
    }

    int32_t n_entries = 0;
    in.read((char *) &n_entries, sizeof(n_entries));
    if (in.fail() || n_entries <= 0) {
        fprintf(stderr, "%s: no data in file %s\n", __func__, imatrix_file.c_str());
        return -1;
    }

    imatrix_data.clear();
    imatrix_data.reserve((size_t) n_entries);

    for (int i = 0; i < n_entries; i++) {
        int32_t len = 0;
        in.read((char *) &len, sizeof(len));
        if (in.fail() || len <= 0) {
            fprintf(stderr, "%s: failed reading name for entry %d from %s\n", __func__, i + 1, imatrix_file.c_str());
            return -1;
        }

        std::vector<char> name_as_vec((size_t) len);
        in.read(name_as_vec.data(), len);
        if (in.fail()) {
            fprintf(stderr, "%s: failed reading name for entry %d from %s\n", __func__, i + 1, imatrix_file.c_str());
            return -1;
        }

        std::string name(name_as_vec.begin(), name_as_vec.end());

        int32_t ncall = 0;
        in.read((char *) &ncall, sizeof(ncall));
        if (in.fail() || ncall <= 0) {
            fprintf(stderr, "%s: invalid ncall %d for entry %s\n", __func__, ncall, name.c_str());
            return -1;
        }

        int32_t nval = 0;
        in.read((char *) &nval, sizeof(nval));
        if (in.fail() || nval <= 0) {
            fprintf(stderr, "%s: invalid nval %d for entry %s\n", __func__, nval, name.c_str());
            return -1;
        }

        auto & e = imatrix_data[name];
        e.resize((size_t) nval);
        in.read((char *) e.data(), (size_t) nval * sizeof(float));
        if (in.fail()) {
            fprintf(stderr, "%s: failed reading data for entry %s\n", __func__, name.c_str());
            return -1;
        }
    }

    int m_last_call = 0;
    if (in.peek() != EOF) {
        in.read((char *) &m_last_call, sizeof(m_last_call));
        int dataset_len = 0;
        in.read((char *) &dataset_len, sizeof(dataset_len));
        if (!in.fail() && dataset_len > 0) {
            std::vector<char> dataset_as_vec((size_t) dataset_len);
            in.read(dataset_as_vec.data(), dataset_len);
            if (!in.fail()) {
                imatrix_datasets.resize(1);
                imatrix_datasets[0].assign(dataset_as_vec.begin(), dataset_as_vec.end());
                fprintf(stderr, "%s: imatrix dataset='%s'\n", __func__, imatrix_datasets[0].c_str());
            }
        }
    }

    fprintf(stderr, "%s: loaded %d importance matrix entries from %s computed on %d chunks\n",
            __func__, int(imatrix_data.size()), imatrix_file.c_str(), m_last_call);

    return m_last_call;
}

int sd_load_imatrix(
        const std::string & imatrix_file,
        std::vector<std::string> & imatrix_datasets,
        std::unordered_map<std::string, std::vector<float>> & imatrix_data) {
    static const char * const LLM_KV_IMATRIX_DATASETS    = "imatrix.datasets";
    static const char * const LLM_KV_IMATRIX_CHUNK_COUNT = "imatrix.chunk_count";
    static const char * const LLM_KV_IMATRIX_CHUNK_SIZE  = "imatrix.chunk_size";

    struct ggml_context * ctx = nullptr;
    struct gguf_init_params meta_gguf_params = {
        /* .no_alloc = */ false,
        /* .ctx      = */ &ctx,
    };

    struct gguf_context * ctx_gguf = gguf_init_from_file(imatrix_file.c_str(), meta_gguf_params);
    if (!ctx_gguf) {
        fprintf(stderr, "%s: imatrix file '%s' is using old format\n", __func__, imatrix_file.c_str());
        return sd_load_legacy_imatrix(imatrix_file, imatrix_datasets, imatrix_data);
    }

    const int32_t n_entries = gguf_get_n_tensors(ctx_gguf);
    if (n_entries < 1) {
        fprintf(stderr, "%s: no data in file %s\n", __func__, imatrix_file.c_str());
        gguf_free(ctx_gguf);
        ggml_free(ctx);
        return -1;
    }

    const int dataset_idx     = gguf_find_key(ctx_gguf, LLM_KV_IMATRIX_DATASETS);
    const int chunk_count_idx = gguf_find_key(ctx_gguf, LLM_KV_IMATRIX_CHUNK_COUNT);
    const int chunk_size_idx  = gguf_find_key(ctx_gguf, LLM_KV_IMATRIX_CHUNK_SIZE);
    if (dataset_idx < 0 || chunk_count_idx < 0 || chunk_size_idx < 0) {
        fprintf(stderr, "%s: missing imatrix metadata in file %s\n", __func__, imatrix_file.c_str());
        gguf_free(ctx_gguf);
        ggml_free(ctx);
        return -1;
    }

    const uint32_t chunk_size = gguf_get_val_u32(ctx_gguf, chunk_size_idx);
    GGML_UNUSED(chunk_size);

    const std::string sums_suffix{ ".in_sum2" };
    const std::string counts_suffix{ ".counts" };

    std::map<std::string, std::pair<struct ggml_tensor *, struct ggml_tensor *>> sums_counts_for;

    for (struct ggml_tensor * cur = ggml_get_first_tensor(ctx); cur; cur = ggml_get_next_tensor(ctx, cur)) {
        std::string name = cur->name;

        if (name.empty()) {
            continue;
        }

        if (sd_string_remove_suffix(name, sums_suffix)) {
            sums_counts_for[std::move(name)].first = cur;
        } else if (sd_string_remove_suffix(name, counts_suffix)) {
            sums_counts_for[std::move(name)].second = cur;
        }
    }

    imatrix_data.clear();
    imatrix_data.reserve(sums_counts_for.size());

    for (const auto & sc : sums_counts_for) {
        const        std::string & name   = sc.first;
        const struct ggml_tensor * sums   = sc.second.first;
        const struct ggml_tensor * counts = sc.second.second;

        if (!sums || !counts) {
            fprintf(stderr, "%s: mismatched sums and counts for %s\n", __func__, name.c_str());
            gguf_free(ctx_gguf);
            ggml_free(ctx);
            return -1;
        }

        const int64_t ne0 = sums->ne[0];
        const int64_t ne1 = sums->ne[1];

        auto & e = imatrix_data[name];
        e.resize((size_t) ggml_nelements(sums));

        for (int64_t j = 0; j < ne1; ++j) {
            const float count = ((const float *) counts->data)[j];
            if (count > 0.0f) {
                for (int64_t i = 0; i < ne0; ++i) {
                    e[(size_t) j * (size_t) ne0 + (size_t) i] = ((const float *) sums->data)[(size_t) j * (size_t) ne0 + (size_t) i] / count;
                }
            } else {
                for (int64_t i = 0; i < ne0; ++i) {
                    e[(size_t) j * (size_t) ne0 + (size_t) i] = 1.0f;
                }
            }
        }
    }

    const int m_last_chunk = (int) gguf_get_val_u32(ctx_gguf, chunk_count_idx);

    const int64_t n_datasets = gguf_get_arr_n(ctx_gguf, dataset_idx);
    imatrix_datasets.clear();
    imatrix_datasets.reserve((size_t) n_datasets);
    for (int64_t i = 0; i < n_datasets; ++i) {
        imatrix_datasets.push_back(gguf_get_arr_str(ctx_gguf, dataset_idx, i));
    }
    if (!imatrix_datasets.empty()) {
        fprintf(stderr, "%s: imatrix datasets=['%s'", __func__, imatrix_datasets[0].c_str());
        for (size_t i = 1; i < imatrix_datasets.size(); ++i) {
            fprintf(stderr, ", '%s'", imatrix_datasets[i].c_str());
        }
        fprintf(stderr, "]\n");
    }

    fprintf(stderr, "%s: loaded %d importance matrix entries from %s computed on %d chunks\n",
            __func__, int(imatrix_data.size()), imatrix_file.c_str(), m_last_chunk);

    gguf_free(ctx_gguf);
    ggml_free(ctx);

    return m_last_chunk;
}

bool sd_make_imatrix_sqrt_scale(
        const std::unordered_map<std::string, std::vector<float>> & imatrix_data,
        const std::string & weight_name,
        int64_t n_in,
        float eps,
        float power,
        std::vector<float> & scale_out) {
    const auto it = imatrix_data.find(weight_name);
    if (it == imatrix_data.end()) {
        return false;
    }

    if ((int64_t) it->second.size() < n_in) {
        fprintf(stderr, "seeddelta-build: imatrix entry %s has %" PRId64 " values, expected >= %" PRId64 "\n",
                weight_name.c_str(), (int64_t) it->second.size(), n_in);
        return false;
    }

    eps = std::max(eps, 0.0f);
    power = std::max(power, 0.0f);

    scale_out.resize(n_in);
    double sum = 0.0;
    for (int64_t i = 0; i < n_in; ++i) {
        float s = std::pow(std::max(it->second[(size_t) i], eps), 0.5f * power);
        if (std::isnan(s) || std::isinf(s)) {
            s = 1.0f;
        }
        scale_out[(size_t) i] = s;
        sum += (double) s;
    }

    const double mean = sum / std::max<double>(1.0, (double) n_in);
    if (mean > 0.0) {
        const float inv_mean = (float) (1.0 / mean);
        for (float & s : scale_out) {
            s *= inv_mean;
        }
    }

    return true;
}
