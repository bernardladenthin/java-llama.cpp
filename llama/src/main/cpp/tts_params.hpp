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

#include <string>

namespace jllama_tts {

/// Context size the TTS backbone is initialised with.
inline constexpr int TTS_N_CTX = 8192;

/// Fallback thread count when the caller passes a non-positive value.
inline constexpr int TTS_DEFAULT_THREADS = 4;

// ---------------------------------------------------------------------------
// Builds the common_params for the TTS backbone.
//
// The two postprocess_cpu_params calls are the load-bearing part. common/arg.cpp is the ONLY
// caller of that function in upstream, so a common_params assembled by hand -- as this one is --
// never gets its CPU fields resolved: common_cpu_params::n_threads defaults to -1, and
// common_init_from_params does not fix it up. common_threadpools::init then finds tpp and
// tpp_batch mismatched, builds a separate batch pool, and ggml_threadpool_new computes
//   workers_size = sizeof(struct ggml_compute_state) * tpp->n_threads
// with n_threads == -1 -- a huge size_t. ggml_aligned_malloc returns NULL and the very next line
// memsets it unchecked, so the process dies on memset(NULL, 0, huge): SIGSEGV at address 0,
// reported as __bzero on macOS and as a bare libc frame on Linux. The role_model argument is what
// makes the batch pool inherit the main pool's count instead of staying at -1; passing nullptr for
// both would resolve each independently. This mirrors arg.cpp exactly.
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

    postprocess_cpu_params(params.cpuparams, nullptr);
    postprocess_cpu_params(params.cpuparams_batch, &params.cpuparams);

    return params;
}

} // namespace jllama_tts

#endif // JLLAMA_TTS_PARAMS_HPP
