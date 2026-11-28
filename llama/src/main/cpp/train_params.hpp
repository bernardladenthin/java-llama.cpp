// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT
//
// The common_params builder for the fine-tuning engine, split out of train_engine.cpp so it can be
// tested without a GGUF, a JVM or a loaded model -- the sibling of tts_params.hpp, and for the same
// reason.
//
// A header, so that test_tts_params.cpp exercises the real builder the engine uses without a model.
// When it was split out, train_engine.cpp was in the jllama target only and
// LlamaTrainerIntegrationTest ran in no CI job, so nothing would have seen the JVM-abort documented in
// cpu_params.hpp come back; both run now (train_engine.cpp is in jllama_test, the CApi finetune
// tests), and the model-free tests stay the cheapest guard.
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

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>

namespace jllama_train {

// The context llama.cpp makes of a requested one: llama_context rounds it up to a multiple of 256, and
// the training windows (common_opt_dataset_init) are that size. 0 for a context that is not known.
[[nodiscard]] inline int64_t padded_training_context(int64_t n_ctx) {
    return n_ctx <= 0 ? 0 : (n_ctx + 255) / 256 * 256;
}

// ---------------------------------------------------------------------------
// Maps one finetune_config onto the common_params the training run is driven with.
//
// The jllama::resolve_cpu_params() call is the load-bearing part -- without it this hand-built
// common_params reaches ggml_threadpool_new with n_threads == -1 and the process dies on a
// memset of NULL. See cpu_params.hpp for the full mechanism.
//
// model_n_ctx_train is the model's trained context (<arch>.context_length), 0 when it is not known;
// it stands in for an n_ctx of 0.
//
// The context is passed on as llama.cpp will make it (a multiple of 256), and a batch and ubatch the
// caller left unset are that context. Since upstream #28520 (b11304) the attention of a training
// step reads the keys and values of the step itself only when one ubatch is the whole context
// (llama_context::opt_init); otherwise it reads them back from the KV cache, through which no
// gradient flows, and the K and V projections are not trained. llama_context caps the batch at the
// context it was asked for before it rounds the context up, so an n_ctx of 64 gave a batch of 64 in a
// context of 256, and the native defaults (a ubatch of 512) did the same for every context but 256
// and 512: the trainer left both projections of every layer as they were.
// ---------------------------------------------------------------------------
[[nodiscard]] inline common_params build_train_params(const finetune_config &cfg, int model_n_ctx_train = 0) {
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
    const int64_t n_ctx = padded_training_context(cfg.n_ctx > 0 ? cfg.n_ctx : model_n_ctx_train);
    if (n_ctx > 0 && n_ctx <= std::numeric_limits<int32_t>::max()) {
        params.n_ctx = static_cast<int32_t>(n_ctx);
        if (cfg.n_batch <= 0) {
            params.n_batch = params.n_ctx;
        }
        if (cfg.n_ubatch <= 0) {
            params.n_ubatch = params.n_batch;
        }
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

// llama_context::opt_init asserts that the batch divides the context and the ubatch the batch (GGML_ASSERT, which
// ended the process). The three are the loaded context's (llama_n_ctx, llama_n_batch, llama_n_ubatch), which
// already carry llama_context's own rules -- the batch capped at the context for causal attention, the ubatch
// for a non-causal model -- and are taken the way opt_init takes them. Empty when they train, else the reason.
[[nodiscard]] inline std::string training_batch_refusal(int64_t n_ctx, int64_t n_batch, int64_t n_ubatch) {
    if (n_ctx <= 0 || n_batch <= 0 || n_ubatch <= 0) {
        return "the training context, batch and ubatch must be positive, were " + std::to_string(n_ctx) + ", " +
               std::to_string(n_batch) + " and " + std::to_string(n_ubatch);
    }
    n_batch = std::min(n_batch, n_ctx);
    n_ubatch = std::min(n_ubatch, n_batch);
    if (n_ctx % n_batch != 0) {
        return "n_batch " + std::to_string(n_batch) + " must divide the training context of " + std::to_string(n_ctx) +
               " tokens";
    }
    if (n_batch % n_ubatch != 0) {
        return "n_ubatch " + std::to_string(n_ubatch) + " must divide n_batch " + std::to_string(n_batch);
    }
    return {};
}

} // namespace jllama_train

#endif // JLLAMA_TRAIN_PARAMS_HPP
