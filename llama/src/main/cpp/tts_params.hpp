// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT
//
// The common_params builder for the TTS engine, split out of tts_engine.cpp so it can be tested
// without a GGUF, a JVM or a loaded model -- the same reason json_helpers.hpp exists.
//
// It is a header, not a .cpp, because jllama_test deliberately does not compile tts_engine.cpp
// (that TU needs the mtmd/llama runtime); a header lets the test exercise the real builder the
// engine uses, rather than a copy that could drift away from it.

#ifndef JLLAMA_TTS_PARAMS_HPP
#define JLLAMA_TTS_PARAMS_HPP

#include "common.h"
#include "cpu_params.hpp"

#include <string>

namespace jllama_tts {

/// Context size the TTS backbone is initialised with.
inline constexpr int TTS_N_CTX = 8192;

/// Fallback thread count when the caller passes a non-positive value.
inline constexpr int TTS_DEFAULT_THREADS = 4;

// ---------------------------------------------------------------------------
// Builds the common_params for the TTS backbone.
//
// The jllama::resolve_cpu_params() call is the load-bearing part -- without it this hand-built
// common_params reaches ggml_threadpool_new with n_threads == -1 and the process dies on a
// memset of NULL. See cpu_params.hpp for the full mechanism; it is shared with train_engine.cpp,
// which assembles its params by hand for the same reason.
// ---------------------------------------------------------------------------
[[nodiscard]] inline common_params build_tts_params(const std::string &model_path, int n_gpu_layers, int n_threads,
                                                    int n_batch) {
    common_params params;
    params.n_ctx = TTS_N_CTX;
    params.n_batch = n_batch;
    params.n_gpu_layers = n_gpu_layers;
    params.cpuparams.n_threads = n_threads > 0 ? n_threads : TTS_DEFAULT_THREADS;
    params.model.path = model_path;
    // Always enable embd so the backbone's hidden state can be handed to the audio-generation
    // helper between frames (mirrors upstream tools/tts/tts.cpp main()).
    params.embedding = true;

    jllama::resolve_cpu_params(params);

    return params;
}

} // namespace jllama_tts

#endif // JLLAMA_TTS_PARAMS_HPP
