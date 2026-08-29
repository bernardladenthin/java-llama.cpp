// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT
//
// The common_params builder for the fine-tuning engine, split out of train_engine.cpp so it can be
// tested without a GGUF, a JVM or a loaded model -- the sibling of tts_params.hpp, and for the same
// reason.
//
// It is a header, not a .cpp, because jllama_test deliberately does not compile train_engine.cpp
// (that TU needs the llama runtime); a header lets the test exercise the real builder the engine
// uses, rather than a copy that could drift away from it. Before this split there was NO runnable
// guard on the trainer's jllama::resolve_cpu_params() call at all: train_engine.cpp is in the
// jllama target only, and LlamaTrainerIntegrationTest is gated on a system property no CI job sets
// and a model in no models.csv row -- so the JVM-abort documented in cpu_params.hpp could have
// regressed here, on every platform, unseen.
//
// Deliberately excluded: reading the training corpus. That does file I/O and can fail, so it stays
// in finetune(), which owns the error string.

#ifndef JLLAMA_TRAIN_PARAMS_HPP
#define JLLAMA_TRAIN_PARAMS_HPP

#include "common.h"
#include "cpu_params.hpp"
#include "ggml-opt.h"
#include "llama.h"
#include "train_engine.h"

namespace jllama_train {

// ---------------------------------------------------------------------------
// Maps one finetune_config onto the common_params the training run is driven with.
//
// The jllama::resolve_cpu_params() call is the load-bearing part -- without it this hand-built
// common_params reaches ggml_threadpool_new with n_threads == -1 and the process dies on a
// memset of NULL. See cpu_params.hpp for the full mechanism.
// ---------------------------------------------------------------------------
[[nodiscard]] inline common_params build_train_params(const finetune_config &cfg) {
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

    // Training needs writable weights (mmap yields read-only pointers) and an f32 KV cache
    // (OUT_PROD has no f16 support) — same forced settings as upstream finetune.cpp.
    // b10107 replaced the use_mmap/use_mlock/use_direct_io booleans with a single load_mode
    // enum; LLAMA_LOAD_MODE_NONE disables mmap so the weight pointers stay writable.
    params.load_mode = LLAMA_LOAD_MODE_NONE;
    params.cache_type_k = GGML_TYPE_F32;
    params.cache_type_v = GGML_TYPE_F32;

    // A hand-built common_params never passes through common_params_parse, so its CPU fields are
    // still unresolved here. Shared with tts_params.hpp -- see cpu_params.hpp for the mechanism.
    jllama::resolve_cpu_params(params);

    return params;
}

} // namespace jllama_train

#endif // JLLAMA_TRAIN_PARAMS_HPP
