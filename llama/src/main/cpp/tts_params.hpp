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
// What a synthesis does about the predefined voices of a model (b12061, #31429: Pocket TTS without
// voice cloning ships its voices in the mmproj, mtmd_gen_audio_info::n_voices of them, and such a
// model only speaks with one -- it is handed over as the speaker reference, a "voice bitmap" from
// mtmd_gen_audio_get_voice()). The rule is upstream's, of llama-tts and the /tts route: a model
// without predefined voices names the voice in its prompt (the "tts" chat template) or has none; one
// with them takes the named voice, refuses a voice next to a speaker reference, keeps a speaker
// reference given alone, and falls back to its first voice when the caller gave neither (without a
// voice prompt it produces near silence).
enum class predefined_voice {
    NOT_USED, // no predefined voice is handed to the pipeline
    NAMED,    // the caller's voice, looked up by name
    FIRST,    // the model's first voice, as its default
    CONFLICT, // a voice and a speaker reference at once
};

[[nodiscard]] inline predefined_voice choose_predefined_voice(int n_voices, bool has_voice,
                                                              bool has_speaker_reference) {
    if (n_voices <= 0) {
        return predefined_voice::NOT_USED;
    }
    if (has_voice) {
        return has_speaker_reference ? predefined_voice::CONFLICT : predefined_voice::NAMED;
    }
    return has_speaker_reference ? predefined_voice::NOT_USED : predefined_voice::FIRST;
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
