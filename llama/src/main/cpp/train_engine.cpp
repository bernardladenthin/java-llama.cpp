// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

#include "train_engine.h"

#include "common.h"
#include "ggml-opt.h"
#include "gguf.h"
#include "llama.h"
#include "rpc_support.hpp"
#include "train_params.hpp"
#include "jllama_api.h"
#include "jni_utf8.hpp"

#include <nlohmann/json.hpp>

#include <jni.h>

#include <algorithm>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

using json = nlohmann::json;

namespace jllama_train {

namespace {

// The model's trained context (<arch>.context_length) read from its GGUF, 0 when the file cannot be
// read or carries none: an n_ctx of 0 trains with it, and build_train_params has to know it before
// the load to make the whole context one ubatch.
int gguf_context_length(const std::string &path) {
    std::error_code ec;
    if (path.empty() || !std::filesystem::is_regular_file(std::filesystem::u8path(path), ec)) {
        return 0; // the load names the file
    }
    gguf_init_params ip{};
    ip.no_alloc = true;
    ip.ctx = nullptr;
    gguf_context *gguf = gguf_init_from_file(path.c_str(), ip);
    if (gguf == nullptr) {
        return 0;
    }
    int64_t length = 0;
    const int64_t arch = gguf_find_key(gguf, "general.architecture");
    if (arch >= 0 && gguf_get_kv_type(gguf, arch) == GGUF_TYPE_STRING) {
        const std::string key = std::string(gguf_get_val_str(gguf, arch)) + ".context_length";
        const int64_t id = gguf_find_key(gguf, key.c_str());
        if (id >= 0) {
            switch (gguf_get_kv_type(gguf, id)) {
            case GGUF_TYPE_UINT32:
                length = gguf_get_val_u32(gguf, id);
                break;
            case GGUF_TYPE_INT32:
                length = gguf_get_val_i32(gguf, id);
                break;
            case GGUF_TYPE_UINT64:
                length = static_cast<int64_t>(std::min<uint64_t>(gguf_get_val_u64(gguf, id), INT32_MAX));
                break;
            case GGUF_TYPE_INT64:
                length = gguf_get_val_i64(gguf, id);
                break;
            default:
                break;
            }
        }
    }
    gguf_free(gguf);
    return length > 0 && length <= INT32_MAX ? static_cast<int>(length) : 0;
}

} // namespace

bool finetune(const finetune_config &cfg, std::string &err) {
    // llama_opt_epoch asserts 0 <= the training part <= the windows, so a split outside [0, 1) ended
    // the process (a negative or a too large part), and 1 trained on nothing; LlamaTrainer's
    // TrainingParameters refuses the same range, the C API did not.
    if (!(cfg.val_split >= 0.0f && cfg.val_split < 1.0f)) {
        err = "val_split must be in [0, 1), was " + std::to_string(cfg.val_split);
        return false;
    }
    // what TrainingParameters refuses, refused here too: build_train_params made 0 epochs one and an
    // unknown optimizer AdamW, so a C caller's typo trained something else than it asked for
    if (cfg.epochs < 1) {
        err = "epochs must be at least 1, was " + std::to_string(cfg.epochs);
        return false;
    }
    if (cfg.optimizer != 0 && cfg.optimizer != 1) {
        err = "optimizer must be 0 (AdamW) or 1 (SGD), was " + std::to_string(cfg.optimizer);
        return false;
    }
    if (cfg.n_ctx < 0) {
        err = "n_ctx must not be negative, was " + std::to_string(cfg.n_ctx);
        return false;
    }
    common_params params = build_train_params(cfg, cfg.n_ctx > 0 ? 0 : gguf_context_length(cfg.model_path));

    // The corpus is either read from a file or supplied inline.
    if (!cfg.training_file.empty()) {
        // u8path: the path is UTF-8, and a narrow string is read in the ANSI code page on Windows
        std::ifstream in(std::filesystem::u8path(cfg.training_file), std::ios::binary);
        if (!in) {
            err = "cannot open training file: " + cfg.training_file;
            return false;
        }
        params.prompt.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    } else {
        params.prompt = cfg.training_text;
    }

    // llama_model_save_to_file() returns nothing and ignores a failed write (upstream's saver drops
    // gguf_write_to_file's bool): a run into a directory that does not exist reported success and left
    // no file. The output is tried after the inputs and before the training, which takes long, and
    // checked after the save.
    if (cfg.output_path.empty()) {
        err = "output_path is required";
        return false;
    }
    const std::filesystem::path output = std::filesystem::u8path(cfg.output_path);
    {
        std::error_code ec;
        const bool existed = std::filesystem::exists(output, ec);
        std::ofstream probe(output, std::ios::binary | std::ios::app);
        if (!probe) {
            err = "cannot write the output file: " + cfg.output_path;
            return false;
        }
        probe.close();
        if (!existed) {
            std::filesystem::remove(output, ec);
        }
    }

    llama_backend_init();
    llama_numa_init(params.numa);
    // Keep RPC servers another load registered in this JVM out of this one (see rpc_support.hpp);
    // training loads no multimodal projector, so the mmproj half of the answer does not apply.
    (void)jllama::rpc::exclude_stale_devices(params.devices);

    common_init_result_ptr llama_init = common_init_from_params(params);
    llama_model *model = llama_init->model();
    llama_context *ctx = llama_init->context();
    if (model == nullptr || ctx == nullptr) {
        err = "failed to load model for training: " + cfg.model_path;
        return false;
    }

    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, true);
    // llama.cpp's dataset (common_opt_dataset_init) cuts the corpus into windows of n_ctx + 1 tokens,
    // one every `stride` tokens, and counts them as (tokens - n_ctx - 1) / stride in size_t: a corpus
    // shorter than one window wrapped to a huge count, and one shorter than a window and a stride gave
    // none -- ggml asserts on either, which ended the process (the JVM). The context is the padded one
    // llama.cpp made (a multiple of 256), not the n_ctx asked for.
    const int64_t n_ctx = llama_n_ctx(ctx);
    const int64_t stride = std::max<int64_t>(1, n_ctx / 2);
    const int64_t minimum = n_ctx + 1 + stride;
    if (static_cast<int64_t>(tokens.size()) < minimum) {
        err = "training corpus produced too few tokens: " + std::to_string(tokens.size()) + ", need at least " +
              std::to_string(minimum) + " (one window of n_ctx + 1 = " + std::to_string(n_ctx + 1) +
              " tokens and a stride of " + std::to_string(stride) + ")";
        return false;
    }

    ggml_opt_dataset_t dataset = common_opt_dataset_init(ctx, tokens, stride);
    // a split that holds every window out leaves nothing to train on, and the run wrote the model it
    // was given as a success
    const int64_t n_windows = ggml_opt_dataset_ndata(dataset);
    if (static_cast<int64_t>(n_windows * (1.0f - params.val_split)) < 1) {
        ggml_opt_dataset_free(dataset);
        err = "val_split " + std::to_string(params.val_split) + " leaves no training window of the " +
              std::to_string(n_windows) + " the corpus gives (" + std::to_string(tokens.size()) + " tokens)";
        return false;
    }

    if (const std::string refusal = training_batch_refusal(llama_n_ctx(ctx), llama_n_batch(ctx), llama_n_ubatch(ctx));
        !refusal.empty()) {
        ggml_opt_dataset_free(dataset);
        err = refusal;
        return false;
    }

    llama_opt_params lopt_params = {
        /*n_ctx_train     =*/0,
        /*param_filter    =*/llama_opt_param_filter_all,
        /*param_filter_ud =*/nullptr,
        /*get_opt_pars    =*/common_opt_lr_pars,
        /*get_opt_pars_ud =*/&params.lr,
        /*optimizer_type  =*/params.optimizer,
    };
    llama_opt_init(ctx, model, lopt_params);

    const int64_t idata_split = ggml_opt_dataset_ndata(dataset) * (1.0f - params.val_split);

    ggml_opt_result_t result_train = ggml_opt_result_init();
    ggml_opt_result_t result_eval = ggml_opt_result_init();

    // A model whose graph needs a gradient ggml cannot compute (MUL_MAT_ID of a mixture of experts, SSM_SCAN, NORM,
    // GEGLU, ...) ended the process in ggml_build_backward_expand, and one with weight scales trained with gradients
    // off by its scales; patches/0044 stops the epoch before it and names the op through llama_opt_error.
    std::string refused;
    for (params.lr.epoch = 0; params.lr.epoch < params.lr.epochs; ++params.lr.epoch) {
        llama_opt_epoch(ctx, dataset, result_train, result_eval, idata_split, ggml_opt_epoch_callback_progress_bar,
                        ggml_opt_epoch_callback_progress_bar);
        ggml_opt_result_reset(result_train);
        ggml_opt_result_reset(result_eval);
        if (const char *error = llama_opt_error(ctx)) {
            refused = error;
            break;
        }
    }

    ggml_opt_result_free(result_train);
    ggml_opt_result_free(result_eval);
    ggml_opt_dataset_free(dataset);
    if (!refused.empty()) {
        err = refused;
        return false;
    }

    // Gone before the save, so that a file afterwards is the save's and not an older one -- a regular
    // file only: an output such as /dev/null is written to, never removed, and not checked.
    std::error_code ec;
    const bool regular = !std::filesystem::exists(output, ec) || std::filesystem::is_regular_file(output, ec);
    if (regular) {
        std::filesystem::remove(output, ec);
    }
    llama_model_save_to_file(model, params.out_file.c_str());
    if (regular) {
        const auto written = std::filesystem::file_size(output, ec);
        if (ec || written == 0) {
            err = "could not write the trained model to " + cfg.output_path;
            return false;
        }
    }

    // Deliberately NOT calling llama_backend_free(): other live llama contexts in this JVM
    // (e.g. an inference LlamaModel) may still depend on the initialized backend.
    return true;
}

bool finetune_json(const std::string &config_json, std::string &err) {
    finetune_config cfg;
    try {
        const json j = json::parse(config_json);
        cfg.model_path = j.value(keys::MODEL_PATH, std::string());
        cfg.training_text = j.value(keys::TRAINING_TEXT, std::string());
        cfg.training_file = j.value(keys::TRAINING_FILE, std::string());
        cfg.output_path = j.value(keys::OUTPUT_PATH, std::string());
        cfg.epochs = j.value(keys::EPOCHS, 2);
        cfg.learning_rate = j.value(keys::LEARNING_RATE, 1e-5f);
        cfg.lr_min = j.value(keys::LR_MIN, -1.0f);
        cfg.decay_epochs = j.value(keys::DECAY_EPOCHS, -1.0f);
        cfg.weight_decay = j.value(keys::WEIGHT_DECAY, 0.0f);
        cfg.optimizer = j.value(keys::OPTIMIZER, 0);
        cfg.n_ctx = j.value(keys::N_CTX, 0);
        cfg.n_gpu_layers = j.value(keys::N_GPU_LAYERS, -1);
        cfg.val_split = j.value(keys::VAL_SPLIT, 0.05f);
        cfg.n_batch = j.value(keys::N_BATCH, 0);
        cfg.n_ubatch = j.value(keys::N_UBATCH, 0);
    } catch (const std::exception &e) {
        err = std::string("invalid training config: ") + e.what();
        return false;
    }
    try {
        return finetune(cfg, err);
    } catch (const std::exception &e) {
        err = e.what();
    } catch (...) {
        err = "unknown C++ exception during fine-tuning";
    }
    return false;
}

} // namespace jllama_train

// A shell over the C API (jllama_finetune): the configuration crosses as UTF-8, the outcome comes
// back as this method's return string -- "" for success, the message otherwise (never a Java
// exception, the contract LlamaTrainer relies on). Function-level backstop, not jni_guard_impl:
// this TU deliberately keeps its own nlohmann alias and never includes jni_helpers.hpp.
extern "C" JNIEXPORT jstring JNICALL Java_net_ladenthin_llama_LlamaTrainer_finetuneNative(JNIEnv *env, jclass,
                                                                                          jstring jconfig) try {
    // GetStringUTFChars hands out Modified UTF-8, which differs from UTF-8 only for U+0000 and the
    // supplementary planes. Jackson escapes U+0000 itself (a control character), and
    // TrainingParameters.toJson() escapes every other non-ASCII character as a backslash-u escape
    // (escapeNonAscii -- Jackson's default writes them raw), so the JSON read here is ASCII where
    // the two encodings differ. TrainingParametersTest and LlamaTrainerTest pin that.
    std::string config_json;
    if (jconfig != nullptr) {
        const char *c = env->GetStringUTFChars(jconfig, nullptr);
        if (c != nullptr) {
            config_json = c;
            env->ReleaseStringUTFChars(jconfig, c);
        }
    }
    if (jllama_finetune(config_json.c_str()) == JLLAMA_STATUS_OK) {
        return env->NewStringUTF(""); // empty == success
    }
    // The message quotes paths, so it is standard UTF-8 of any character (see jni_utf8.hpp).
    return new_string_utf8_impl(env, jllama_last_error());
} catch (...) {
    return env->NewStringUTF("unknown C++ exception crossed the JNI boundary");
}
