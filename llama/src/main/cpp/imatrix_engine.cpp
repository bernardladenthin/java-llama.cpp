// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

// The collection follows llama.cpp's tools/imatrix/imatrix.cpp (MIT, the llama.cpp authors): the
// same tensors, the same sums and the same GGUF layout, so a file written here and one written by
// llama-imatrix are interchangeable. Left out: the activation sums (.in_sum, statistics only),
// merging existing files, the legacy format and the perplexity report.

#include "imatrix_engine.h"

#include "common.h"
#include "imatrix-loader.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "llama.h"
#include "rpc_support.hpp" // exclude_stale_devices

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace jllama::imatrix {

namespace {

// llama-imatrix ignores batches below this many tokens, and so does this
constexpr int MIN_BATCH_TOKENS = 16;

// the chunk size assumed for files of the legacy format, which does not record one
constexpr int DEFAULT_CHUNK_SIZE = 512;

struct stats {
    std::vector<float> sums;     // one block of ne0 values per matrix (expert)
    std::vector<int64_t> counts; // one per matrix
};

struct collector {
    std::mutex mutex;
    std::map<std::string, stats> entries;
    std::vector<char> data_copy;
    std::vector<char> ids_copy;
    std::string error;

    // CUDA0#blk.0.attn_k.weight#0 => blk.0.attn_k.weight
    static std::string weight_name(const char *name) {
        const char *p = std::strchr(name, '#');
        if (p == nullptr) {
            return name;
        }
        ++p;
        const char *q = std::strchr(p, '#');
        return q != nullptr ? std::string(p, q - p) : std::string(p);
    }

    bool wants(const ggml_tensor *t) const {
        if (t->op == GGML_OP_MUL_MAT_ID) {
            return true;
        }
        if (t->op != GGML_OP_MUL_MAT) {
            return false;
        }
        const ggml_tensor *input = t->src[1];
        if (input->ne[1] < MIN_BATCH_TOKENS || input->type != GGML_TYPE_F32) {
            return false;
        }
        return weight_name(t->src[0]->name).rfind("blk.", 0) == 0;
    }

    void fail(const std::string &message) {
        if (error.empty()) {
            error = message;
        }
    }

    void collect(const ggml_tensor *t) {
        std::lock_guard<std::mutex> lock(mutex);
        if (!error.empty()) {
            return;
        }
        const ggml_tensor *weight = t->src[0];
        const ggml_tensor *input = t->src[1];
        const std::string name = weight_name(weight->name);

        const char *data = static_cast<const char *>(input->data);
        if (!ggml_backend_buffer_is_host(input->buffer)) {
            data_copy.resize(ggml_nbytes(input));
            ggml_backend_tensor_get(input, data_copy.data(), 0, data_copy.size());
            data = data_copy.data();
        }
        const int64_t ne0 = input->ne[0];
        stats &e = entries[name];

        if (t->op == GGML_OP_MUL_MAT_ID) {
            // ids: [experts used, tokens]; input: [columns, experts used or 1, tokens]
            const ggml_tensor *ids = t->src[2];
            const int64_t n_expert = weight->ne[2];
            const int64_t n_used = ids->ne[0];
            const int64_t n_tokens = input->ne[2];
            if (ids->ne[1] != n_tokens || ggml_nrows(input) != input->ne[1] * n_tokens) {
                fail("unexpected shape of the expert input of " + name);
                return;
            }
            if (e.sums.empty()) {
                e.sums.assign(static_cast<size_t>(ne0 * n_expert), 0.0f);
                e.counts.assign(static_cast<size_t>(n_expert), 0);
            } else if (e.sums.size() != static_cast<size_t>(ne0 * n_expert)) {
                fail("inconsistent size of " + name);
                return;
            }
            ids_copy.resize(ggml_nbytes(ids));
            ggml_backend_tensor_get(ids, ids_copy.data(), 0, ids_copy.size());
            for (int64_t used = 0; used < n_used; ++used) {
                for (int64_t token = 0; token < n_tokens; ++token) {
                    int32_t expert = 0;
                    std::memcpy(&expert, ids_copy.data() + token * ids->nb[1] + used * ids->nb[0], sizeof(expert));
                    if (expert < 0 || expert >= n_expert) {
                        fail("an expert id out of range in " + name);
                        return;
                    }
                    const auto *x = reinterpret_cast<const float *>(data + (used % input->ne[1]) * input->nb[1] +
                                                                    token * input->nb[2]);
                    float *sum = e.sums.data() + expert * ne0;
                    for (int64_t j = 0; j < ne0; ++j) {
                        sum[j] += x[j] * x[j];
                    }
                    e.counts[static_cast<size_t>(expert)]++;
                }
            }
        } else {
            const int64_t n_mat = weight->ne[2] * weight->ne[3];
            if (e.sums.empty()) {
                e.sums.assign(static_cast<size_t>(ne0 * n_mat), 0.0f);
                e.counts.assign(1, 0);
            } else if (e.sums.size() != static_cast<size_t>(ne0 * n_mat)) {
                fail("inconsistent size of " + name);
                return;
            }
            for (int64_t i3 = 0; i3 < input->ne[3]; ++i3) {
                for (int64_t i2 = 0; i2 < input->ne[2]; ++i2) {
                    const int64_t mat = (i3 % weight->ne[3]) * weight->ne[2] + (i2 % weight->ne[2]);
                    float *sum = e.sums.data() + mat * ne0;
                    for (int64_t row = 0; row < input->ne[1]; ++row) {
                        const auto *x = reinterpret_cast<const float *>(data + row * input->nb[1] + i2 * input->nb[2] +
                                                                        i3 * input->nb[3]);
                        for (int64_t j = 0; j < ne0; ++j) {
                            sum[j] += x[j] * x[j];
                        }
                    }
                }
            }
            for (int64_t &count : e.counts) {
                count += ggml_nrows(input) / n_mat;
            }
        }
        for (const float v : e.sums) {
            if (!std::isfinite(v)) {
                fail("non-finite activations in " + name);
                return;
            }
        }
    }

    static bool callback(ggml_tensor *t, bool ask, void *user_data) {
        auto *self = static_cast<collector *>(user_data);
        if (ask) {
            return self->wants(t);
        }
        self->collect(t);
        return true;
    }
};

void save(const std::map<std::string, stats> &entries, const std::string &output_path,
          const std::vector<std::string> &dataset_names, int n_chunks, int chunk_size) {
    size_t data_size = 0;
    for (const auto &[name, e] : entries) {
        data_size += GGML_PAD(ggml_tensor_overhead() + sizeof(float) * e.sums.size(), GGML_MEM_ALIGN);
        data_size += GGML_PAD(ggml_tensor_overhead() + sizeof(float) * e.counts.size(), GGML_MEM_ALIGN);
    }
    ggml_init_params ip = {data_size + 1024, nullptr, false};
    ggml_context *ctx = ggml_init(ip);
    gguf_context *g = gguf_init_empty();
    if (ctx == nullptr || g == nullptr) {
        throw std::runtime_error("out of memory while writing the importance matrix");
    }
    std::vector<const char *> datasets;
    for (const std::string &name : dataset_names) {
        datasets.push_back(name.c_str());
    }
    gguf_set_val_str(g, "general.type", "imatrix");
    gguf_set_arr_str(g, "imatrix.datasets", datasets.data(), datasets.size());
    gguf_set_val_u32(g, "imatrix.chunk_count", static_cast<uint32_t>(n_chunks));
    gguf_set_val_u32(g, "imatrix.chunk_size", static_cast<uint32_t>(chunk_size));
    for (const auto &[name, e] : entries) {
        const auto n_mat = static_cast<int64_t>(e.counts.size());
        ggml_tensor *sums = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, static_cast<int64_t>(e.sums.size()) / n_mat, n_mat);
        ggml_tensor *counts = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, n_mat);
        ggml_set_name(sums, (name + ".in_sum2").c_str());
        ggml_set_name(counts, (name + ".counts").c_str());
        std::memcpy(sums->data, e.sums.data(), sizeof(float) * e.sums.size());
        for (int64_t i = 0; i < n_mat; ++i) {
            static_cast<float *>(counts->data)[i] = static_cast<float>(e.counts[static_cast<size_t>(i)]);
        }
        gguf_add_tensor(g, sums);
        gguf_add_tensor(g, counts);
    }
    const bool ok = gguf_write_to_file(g, output_path.c_str(), false);
    gguf_free(g);
    ggml_free(ctx);
    if (!ok) {
        throw std::runtime_error("could not write the importance matrix to '" + output_path + "'");
    }
}

// The binary format llama-imatrix wrote before GGUF and still reads: per entry the name, the number
// of chunks, and per column the mean times that number; then the chunk count and the dataset name.
// Fields are written in the byte order of the host, as upstream's writer does.
void save_legacy(const std::map<std::string, stats> &entries, const std::string &output_path,
                 const std::string &dataset, int n_chunks, int chunk_size) {
    std::ofstream out(output_path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("could not write the importance matrix to '" + output_path + "'");
    }
    auto put = [&out](int32_t v) { out.write(reinterpret_cast<const char *>(&v), sizeof(v)); };
    int32_t n_entries = 0;
    for (const auto &[name, e] : entries) {
        // an entry no chunk reached is left out, as upstream does (an expert set never routed to)
        n_entries += std::any_of(e.counts.begin(), e.counts.end(), [](int64_t c) { return c != 0; }) ? 1 : 0;
    }
    put(n_entries);
    for (const auto &[name, e] : entries) { // std::map: sorted by name, the order upstream writes
        if (std::none_of(e.counts.begin(), e.counts.end(), [](int64_t c) { return c != 0; })) {
            continue;
        }
        put(static_cast<int32_t>(name.size()));
        out.write(name.data(), static_cast<std::streamsize>(name.size()));
        const int64_t most = *std::max_element(e.counts.begin(), e.counts.end());
        const auto ncall = static_cast<int32_t>((most + chunk_size - 1) / chunk_size); // rounded up: never zero
        put(ncall);
        put(static_cast<int32_t>(e.sums.size()));
        const size_t per_matrix = e.sums.size() / e.counts.size();
        std::vector<float> values(e.sums.size());
        for (size_t i = 0; i < e.sums.size(); ++i) {
            const int64_t count = e.counts[i / per_matrix];
            // an expert without data gets the neutral weight 1, as upstream writes it
            values[i] = count == 0 ? static_cast<float>(ncall)
                                   : e.sums[i] / static_cast<float>(count) * static_cast<float>(ncall);
        }
        out.write(reinterpret_cast<const char *>(values.data()),
                  static_cast<std::streamsize>(values.size() * sizeof(float)));
    }
    put(n_chunks);
    put(static_cast<int32_t>(dataset.size()));
    out.write(dataset.data(), static_cast<std::streamsize>(dataset.size()));
    out.flush();
    if (!out) {
        throw std::runtime_error("could not write the importance matrix to '" + output_path + "'");
    }
}

} // namespace

int compute(const std::string &model_path, const std::string &text_path, const std::string &output_path,
            const options &opts) {
    if (opts.chunk_size < MIN_BATCH_TOKENS) {
        throw std::runtime_error("the chunk size of an importance matrix must be at least " +
                                 std::to_string(MIN_BATCH_TOKENS) + " tokens");
    }
    std::ifstream in(text_path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("could not read the calibration text '" + text_path + "'");
    }
    std::stringstream buffer;
    buffer << in.rdbuf();
    const std::string text = buffer.str();

    // Idempotent; never paired with llama_backend_free here, a loaded model shares the backend.
    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = opts.n_gpu_layers;
    // a stale RPC device of an earlier --rpc load must not be picked up (rpc_support.hpp)
    std::vector<ggml_backend_dev_t> devices;
    jllama::rpc::exclude_stale_devices(devices);
    if (!devices.empty()) {
        mparams.devices = devices.data();
    }
    const std::unique_ptr<llama_model, void (*)(llama_model *)> model(
        llama_model_load_from_file(model_path.c_str(), mparams), llama_model_free);
    if (!model) {
        throw std::runtime_error("could not load the model '" + model_path + "'");
    }

    collector collected;
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = static_cast<uint32_t>(opts.chunk_size);
    cparams.n_batch = static_cast<uint32_t>(opts.chunk_size);
    cparams.n_ubatch = static_cast<uint32_t>(opts.chunk_size);
    if (opts.n_threads > 0) {
        cparams.n_threads = opts.n_threads;
        cparams.n_threads_batch = opts.n_threads;
    }
    cparams.cb_eval = collector::callback;
    cparams.cb_eval_user_data = &collected;
    const std::unique_ptr<llama_context, void (*)(llama_context *)> ctx(llama_init_from_model(model.get(), cparams),
                                                                        llama_free);
    if (!ctx) {
        throw std::runtime_error("could not create a context of " + std::to_string(opts.chunk_size) + " tokens for '" +
                                 model_path + "'");
    }

    const llama_vocab *vocab = llama_model_get_vocab(model.get());
    const bool add_bos = llama_vocab_get_add_bos(vocab);
    std::vector<llama_token> tokens = common_tokenize(ctx.get(), text, add_bos);
    int n_chunks = static_cast<int>(tokens.size() / static_cast<size_t>(opts.chunk_size));
    if (n_chunks < 1) {
        throw std::runtime_error("the calibration text has " + std::to_string(tokens.size()) +
                                 " tokens, fewer than one chunk of " + std::to_string(opts.chunk_size));
    }
    if (opts.max_chunks > 0 && n_chunks > opts.max_chunks) {
        n_chunks = opts.max_chunks;
    }

    // Every token asks for its output: the last layer computes only the rows somebody wants, and with
    // the default (the last token alone) its feed-forward matrices would see one row and be skipped.
    const std::unique_ptr<llama_batch, void (*)(llama_batch *)> batch(
        new llama_batch(llama_batch_init(opts.chunk_size, 0, 1)), [](llama_batch *b) {
            llama_batch_free(*b);
            delete b;
        });
    for (int chunk = 0; chunk < n_chunks; ++chunk) {
        const llama_token *first = tokens.data() + static_cast<size_t>(chunk) * static_cast<size_t>(opts.chunk_size);
        llama_memory_clear(llama_get_memory(ctx.get()), true);
        batch->n_tokens = opts.chunk_size;
        for (int i = 0; i < opts.chunk_size; ++i) {
            // every chunk starts like a prompt, as llama-imatrix does
            batch->token[i] = i == 0 && add_bos ? llama_vocab_bos(vocab) : first[i];
            batch->pos[i] = i;
            batch->n_seq_id[i] = 1;
            batch->seq_id[i][0] = 0;
            batch->logits[i] = 1;
        }
        if (llama_decode(ctx.get(), *batch) != 0) {
            throw std::runtime_error("evaluating chunk " + std::to_string(chunk + 1) +
                                     " of the calibration text failed");
        }
        if (!collected.error.empty()) {
            throw std::runtime_error("importance matrix: " + collected.error);
        }
        if (opts.progress != nullptr &&
            !opts.progress(static_cast<float>(chunk + 1) / static_cast<float>(n_chunks), opts.progress_user_data)) {
            throw std::runtime_error("the importance matrix was cancelled after chunk " + std::to_string(chunk + 1) +
                                     " of " + std::to_string(n_chunks));
        }
    }
    if (collected.entries.empty()) {
        throw std::runtime_error("no weight matrix of '" + model_path + "' was reached by the calibration text");
    }
    const size_t slash = text_path.find_last_of("/\\");
    save(collected.entries, output_path, {slash == std::string::npos ? text_path : text_path.substr(slash + 1)},
         n_chunks, opts.chunk_size);
    return n_chunks;
}

int merge(const std::vector<std::string> &input_paths, const std::string &output_path, bool legacy_format) {
    if (input_paths.empty()) {
        throw std::runtime_error("no importance matrix to merge");
    }
    std::vector<common_imatrix> loaded(input_paths.size());
    int chunk_size = 0;
    for (size_t i = 0; i < input_paths.size(); ++i) {
        if (!common_imatrix_load(input_paths[i], loaded[i])) {
            throw std::runtime_error("could not read the importance matrix '" + input_paths[i] + "'");
        }
        if (loaded[i].entries.empty()) {
            throw std::runtime_error("the importance matrix '" + input_paths[i] + "' holds no entry");
        }
        if (!loaded[i].is_legacy && loaded[i].chunk_size > 0) {
            if (chunk_size == 0) {
                chunk_size = loaded[i].chunk_size;
            } else if (chunk_size != loaded[i].chunk_size) {
                // the sums of two chunk sizes weigh a token differently only through the context it saw;
                // llama-imatrix warns and goes on, a library call says what it was given
                throw std::runtime_error("the importance matrices were computed with different chunk sizes (" +
                                         std::to_string(chunk_size) + " and " + std::to_string(loaded[i].chunk_size) +
                                         " in '" + input_paths[i] + "')");
            }
        }
    }
    if (chunk_size == 0) {
        chunk_size = DEFAULT_CHUNK_SIZE; // only files of the legacy format, which does not record it
    }

    std::map<std::string, stats> entries;
    std::vector<std::string> datasets;
    int n_chunks = 0;
    for (size_t i = 0; i < input_paths.size(); ++i) {
        const common_imatrix &in = loaded[i];
        // The legacy format holds, per column, the mean times the number of chunks, and that number;
        // GGUF holds the sums and the number of tokens. The same conversion as llama-imatrix's --in-file.
        for (const auto &[name, entry] : in.entries) {
            if (entry.counts.empty() || entry.sums.empty()) {
                continue;
            }
            stats &e = entries[name];
            if (e.sums.empty()) {
                e.sums.assign(entry.sums.size(), 0.0f);
                e.counts.assign(entry.counts.size(), 0);
            } else if (e.sums.size() != entry.sums.size()) {
                throw std::runtime_error("'" + name + "' has another shape in '" + input_paths[i] +
                                         "' than in the matrices before it: they are not of the same model");
            }
            if (in.is_legacy) {
                for (size_t j = 0; j < entry.sums.size(); ++j) {
                    e.sums[j] += entry.sums[j] * static_cast<float>(chunk_size);
                }
                for (int64_t &count : e.counts) {
                    count += entry.counts[0] * chunk_size;
                }
                continue;
            }
            if (e.counts.size() == 1 && entry.counts.size() > 1) {
                e.counts.resize(entry.counts.size(), e.counts[0]); // a legacy file came first: one count for all
            } else if (e.counts.size() != entry.counts.size()) {
                throw std::runtime_error("'" + name + "' has another number of experts in '" + input_paths[i] +
                                         "' than in the matrices before it: they are not of the same model");
            }
            for (size_t j = 0; j < entry.sums.size(); ++j) {
                e.sums[j] += entry.sums[j];
            }
            for (size_t j = 0; j < entry.counts.size(); ++j) {
                e.counts[j] += entry.counts[j];
            }
        }
        n_chunks += in.chunk_count;
        if (in.datasets.empty()) {
            const size_t slash = input_paths[i].find_last_of("/\\");
            datasets.push_back(slash == std::string::npos ? input_paths[i] : input_paths[i].substr(slash + 1));
        } else {
            datasets.insert(datasets.end(), in.datasets.begin(), in.datasets.end());
        }
    }
    if (legacy_format) {
        save_legacy(entries, output_path, datasets.back(), n_chunks, chunk_size);
    } else {
        save(entries, output_path, datasets, n_chunks, chunk_size);
    }
    return n_chunks;
}

} // namespace jllama::imatrix
