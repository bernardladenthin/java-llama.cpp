// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

#include "train_engine.h"

#include "common.h"
#include "cpu_params.hpp"
#include "ggml-opt.h"
#include "llama.h"

#include <nlohmann/json.hpp>

#include <jni.h>

#include <exception>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

using json = nlohmann::json;

namespace jllama_train {

bool finetune(const finetune_config &cfg, std::string &err) {
    common_params params;
    params.escape = false;
    params.model.path = cfg.model_path;
    params.out_file = cfg.output_path;
    params.n_ctx = cfg.n_ctx;
    params.n_gpu_layers = cfg.n_gpu_layers;
    params.val_split = cfg.val_split;
    if (cfg.n_batch > 0) {
        params.n_batch = cfg.n_batch;
    }
    if (cfg.n_ubatch > 0) {
        params.n_ubatch = cfg.n_ubatch;
    }

    params.optimizer = cfg.optimizer == 1 ? GGML_OPT_OPTIMIZER_TYPE_SGD : GGML_OPT_OPTIMIZER_TYPE_ADAMW;
    params.lr.lr0 = cfg.learning_rate;
    params.lr.lr_min = cfg.lr_min;
    params.lr.decay_epochs = cfg.decay_epochs;
    params.lr.wd = cfg.weight_decay;
    params.lr.epochs = static_cast<unsigned>(cfg.epochs > 0 ? cfg.epochs : 1);
    params.lr.init(); // required after setting lr fields, before the optimizer reads get_lr()

    // The corpus is either read from a file or supplied inline.
    if (!cfg.training_file.empty()) {
        std::ifstream in(cfg.training_file, std::ios::binary);
        if (!in) {
            err = "cannot open training file: " + cfg.training_file;
            return false;
        }
        params.prompt.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    } else {
        params.prompt = cfg.training_text;
    }

    // Training needs writable weights (mmap yields read-only pointers) and an f32 KV cache
    // (OUT_PROD has no f16 support) — same forced settings as upstream finetune.cpp.
    // b10107 replaced the use_mmap/use_mlock/use_direct_io booleans with a single load_mode
    // enum; LLAMA_LOAD_MODE_NONE disables mmap so the weight pointers stay writable.
    params.load_mode = LLAMA_LOAD_MODE_NONE;
    params.cache_type_k = GGML_TYPE_F32;
    params.cache_type_v = GGML_TYPE_F32;

    // A hand-built common_params never passes through common_params_parse, so its CPU fields are
    // still unresolved here; without this the process dies on a memset of NULL inside
    // ggml_threadpool_new. Shared with tts_params.hpp -- see cpu_params.hpp for the mechanism.
    jllama::resolve_cpu_params(params);

    llama_backend_init();
    llama_numa_init(params.numa);

    common_init_result_ptr llama_init = common_init_from_params(params);
    llama_model *model = llama_init->model();
    llama_context *ctx = llama_init->context();
    if (model == nullptr || ctx == nullptr) {
        err = "failed to load model for training: " + cfg.model_path;
        return false;
    }

    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, true);
    if (tokens.size() < 2) {
        err = "training corpus produced too few tokens (need at least 2)";
        return false;
    }

    ggml_opt_dataset_t dataset = common_opt_dataset_init(ctx, tokens, llama_n_ctx(ctx) / 2);

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

    for (params.lr.epoch = 0; params.lr.epoch < params.lr.epochs; ++params.lr.epoch) {
        llama_opt_epoch(ctx, dataset, result_train, result_eval, idata_split, ggml_opt_epoch_callback_progress_bar,
                        ggml_opt_epoch_callback_progress_bar);
        ggml_opt_result_reset(result_train);
        ggml_opt_result_reset(result_eval);
    }

    ggml_opt_result_free(result_train);
    ggml_opt_result_free(result_eval);
    ggml_opt_dataset_free(dataset);

    llama_model_save_to_file(model, params.out_file.c_str());

    // Deliberately NOT calling llama_backend_free(): other live llama contexts in this JVM
    // (e.g. an inference LlamaModel) may still depend on the initialized backend.
    return true;
}

} // namespace jllama_train

extern "C" JNIEXPORT jstring JNICALL Java_net_ladenthin_llama_LlamaTrainer_finetuneNative(JNIEnv *env, jclass,
                                                                                          jstring jconfig) {
    std::string config_json;
    if (jconfig != nullptr) {
        const char *c = env->GetStringUTFChars(jconfig, nullptr);
        if (c != nullptr) {
            config_json = c;
            env->ReleaseStringUTFChars(jconfig, c);
        }
    }

    jllama_train::finetune_config cfg;
    try {
        const json j = json::parse(config_json);
        cfg.model_path = j.value("model_path", std::string());
        cfg.training_text = j.value("training_text", std::string());
        cfg.training_file = j.value("training_file", std::string());
        cfg.output_path = j.value("output_path", std::string());
        cfg.epochs = j.value("epochs", 2);
        cfg.learning_rate = j.value("learning_rate", 1e-5f);
        cfg.lr_min = j.value("lr_min", -1.0f);
        cfg.decay_epochs = j.value("decay_epochs", -1.0f);
        cfg.weight_decay = j.value("weight_decay", 0.0f);
        cfg.optimizer = j.value("optimizer", 0);
        cfg.n_ctx = j.value("n_ctx", 0);
        cfg.n_gpu_layers = j.value("n_gpu_layers", -1);
        cfg.val_split = j.value("val_split", 0.05f);
        cfg.n_batch = j.value("n_batch", 0);
        cfg.n_ubatch = j.value("n_ubatch", 0);
    } catch (const std::exception &e) {
        return env->NewStringUTF((std::string("invalid training config: ") + e.what()).c_str());
    }

    std::string err;
    try {
        if (jllama_train::finetune(cfg, err)) {
            return env->NewStringUTF(""); // empty == success
        }
    } catch (const std::exception &e) {
        err = e.what();
    } catch (...) {
        err = "unknown C++ exception during fine-tuning";
    }
    return env->NewStringUTF(err.c_str());
}
