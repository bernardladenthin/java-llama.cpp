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

// ---------------------------------------------------------------------------
// The refusal for a speaker reference the loaded pair cannot use, or "" when the request may go on.
// `supported` is llama.cpp's mtmd_helper_cap::audio_gen_spk_ref (b11901, #31183): true for Qwen3-TTS
// and Pocket TTS, false for a pipeline without a voice encoder (Soprano), whose set_input rejects any
// speaker reference -- before this check that surfaced as the generic "TTS input rejected" error.
// ---------------------------------------------------------------------------
[[nodiscard]] inline std::string speaker_reference_refusal(bool supported, const std::string &speaker_reference_path) {
    if (speaker_reference_path.empty() || supported) {
        return std::string();
    }
    return "this TTS model does not accept a speaker reference (no voice cloning): " + speaker_reference_path;
}

// ---------------------------------------------------------------------------
// The language handed to the pipeline (mtmd_helper_gen_audio_inp::lang), "" for none. A model whose
// "tts" chat template rendered the prompt (LFM3-Audio, b11941, #31257) takes the caller's value as
// given -- its codes are "en", "de", ... and its template supplies the default -- so an empty one stays
// empty, as upstream's /tts route and llama-tts leave it. A pipeline without such a template
// (Qwen3-TTS) keeps this library's default "english", which it has always sent.
// ---------------------------------------------------------------------------
[[nodiscard]] inline std::string tts_pipeline_lang(bool prompt_from_template, const std::string &lang) {
    if (!lang.empty() || prompt_from_template) {
        return lang;
    }
    return "english";
}

} // namespace jllama_tts

#endif // JLLAMA_TTS_PARAMS_HPP
