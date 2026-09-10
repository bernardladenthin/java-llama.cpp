// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

#include "train_engine.h"

#include "common.h"
#include "ggml-opt.h"
#include "llama.h"
#include "train_params.hpp"

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
    common_params params = build_train_params(cfg);

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
        cfg.model_path = j.value(jllama_train::keys::MODEL_PATH, std::string());
        cfg.training_text = j.value(jllama_train::keys::TRAINING_TEXT, std::string());
        cfg.training_file = j.value(jllama_train::keys::TRAINING_FILE, std::string());
        cfg.output_path = j.value(jllama_train::keys::OUTPUT_PATH, std::string());
        cfg.epochs = j.value(jllama_train::keys::EPOCHS, 2);
        cfg.learning_rate = j.value(jllama_train::keys::LEARNING_RATE, 1e-5f);
        cfg.lr_min = j.value(jllama_train::keys::LR_MIN, -1.0f);
        cfg.decay_epochs = j.value(jllama_train::keys::DECAY_EPOCHS, -1.0f);
        cfg.weight_decay = j.value(jllama_train::keys::WEIGHT_DECAY, 0.0f);
        cfg.optimizer = j.value(jllama_train::keys::OPTIMIZER, 0);
        cfg.n_ctx = j.value(jllama_train::keys::N_CTX, 0);
        cfg.n_gpu_layers = j.value(jllama_train::keys::N_GPU_LAYERS, -1);
        cfg.val_split = j.value(jllama_train::keys::VAL_SPLIT, 0.05f);
        cfg.n_batch = j.value(jllama_train::keys::N_BATCH, 0);
        cfg.n_ubatch = j.value(jllama_train::keys::N_UBATCH, 0);
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
