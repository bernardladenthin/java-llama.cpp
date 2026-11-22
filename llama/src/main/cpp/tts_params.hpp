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
#include "mtmd.h"

#include <cmath>
#include <cstdint>
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
// The projector's context. mtmd_context_params_default() fixes its thread count at 4, and the
// projector computes the audio -- and the llama contexts it builds itself (tiny-voca's stacks,
// VoxCPM2's residual LM, YuE2's NAR layers) -- with it, so the engine's count has to be passed on:
// llama-tts did not either until b12584 (#32664), and TextToSpeech's thread count reached the
// backbone only. n_threads is the count build_tts_params resolved.
// ---------------------------------------------------------------------------
[[nodiscard]] inline mtmd_context_params build_tts_mtmd_params(int n_gpu_layers, int n_threads) {
    mtmd_context_params params = mtmd_context_params_default();
    params.use_gpu = n_gpu_layers != 0;
    params.n_threads = n_threads;
    return params;
}

// ---------------------------------------------------------------------------
// Whether a pipeline clones a voice from a reference clip, which the projector's audio encoder
// encodes (encode_speaker in tools/mtmd/mtmd-helper-gen.cpp). Read from the pipelines at b12541:
// Qwen3-TTS, Pocket TTS, Chatterbox, Fish Audio S2, ZONOS3 and VoxCPM2 encode a clip, and Drama-3 since
// b12590 (#32586: the clip's codes from the codec encoder, its transcript in the prompt; until then
// "voice cloning is not supported yet"); Kokoro and tiny-voca take only a predefined voice and refuse a
// clip ("use one of its voices"), LFM3-Audio names its voice in the prompt and ignores a clip, Soprano
// refuses any reference, YuE2 and MiniMax Music 3 do not read it. llama.cpp's
// mtmd_helper_cap::audio_gen_spk_ref still names only the first two
// (#31183, b11901; unchanged at b12593), while its own /tts route asks only whether the projector
// has an audio encoder -- so the capability refused cloning for five pipelines that clone.
// ---------------------------------------------------------------------------
[[nodiscard]] inline bool pipeline_clones_voice_from_clip(mtmd_gen_audio_type type) {
    switch (type) {
    case MTMD_GEN_AUDIO_TYPE_QWEN3TTS:
    case MTMD_GEN_AUDIO_TYPE_POCKETTTS:
    case MTMD_GEN_AUDIO_TYPE_CHATTERBOX:
    case MTMD_GEN_AUDIO_TYPE_FISH_S2:
    case MTMD_GEN_AUDIO_TYPE_ZONOS3:
    case MTMD_GEN_AUDIO_TYPE_VOXCPM2:
    case MTMD_GEN_AUDIO_TYPE_DRAMA3:
        return true;
    default:
        return false;
    }
}

// Whether the loaded pair takes a speaker reference: llama.cpp's capability or a pipeline above,
// and a projector that takes audio in -- a Pocket TTS or ZONOS3 projector converted without its
// voice encoder has none, and a reference clip then fails to decode (found by CI's Pocket TTS
// pair, which ships predefined voices instead).
[[nodiscard]] inline bool pair_takes_speaker_reference(bool capability, mtmd_gen_audio_type type,
                                                       bool projector_takes_audio) {
    return (capability || pipeline_clones_voice_from_clip(type)) && projector_takes_audio;
}

// ---------------------------------------------------------------------------
// The refusal for a speaker reference the loaded pair cannot use, or "" when the request may go on.
// `supported` is pair_takes_speaker_reference() above: false for a pipeline that does not clone (Soprano,
// whose set_input rejects any speaker reference -- before this check that surfaced as the generic "TTS
// input rejected" error) or a projector without a voice encoder.
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

// ---------------------------------------------------------------------------
// What one synthesis asks for besides its text. Every value starts at the one that leaves the choice
// to the engine or the model, so a caller that sets nothing synthesises as before these fields
// existed. top_p, repeat_penalty, speed and loudness came with llama.cpp b12431 (#32020, Zonos3):
// upstream's /tts reads all four, mtmd_helper_gen_audio_inp carries top_p, speed and loudness, and
// the backbone sampler takes top_p and the repetition penalty.
// ---------------------------------------------------------------------------
struct synthesis_options {
    std::string speaker_reference_path; ///< a clip to clone the voice from, "" for none
    std::string lang;                   ///< the model's language, "" for its default
    std::string voice;                  ///< a predefined or template voice, "" for the default
    int max_frames = 0;                 ///< cap on the generated frames, <= 0 for the model's own
    int top_k = 0;                      ///< top-k, <= 0 for the engine's defaults
    uint32_t seed = 0;                  ///< sampler and codec seed
    float temperature = -1.0f;          ///< audio frame temperature, < 0 = the model's default
    float top_p = 1.0f;                 ///< top-p of the backbone and the code predictor, 0 or 1 = off
    float repeat_penalty = 1.0f;        ///< repetition penalty of the backbone, 1 = off
    float speed = 1.0f;                 ///< speaking speed relative to the model's pace, > 0
    float loudness = 0.0f;              ///< target loudness in LUFS, < 0; 0 = the model's own
    std::string speaker_reference_text; ///< transcript of the speaker reference (Fish S2, Qwen3-TTS), "" for none
    float cfg_scale = 0.0f;             ///< guidance scale of the audio codes, > 0; 0 = the model's own (YuE2)
    int flow_steps = 0;                 ///< flow-matching solver steps, 1..1024; 0 = the model's own (YuE2)
    bool raw_prompt = false;            ///< the text as it is, the model's "tts" template skipped (b12494)
};

// The refusal for values no model can take, or "" when the request may go on. Whether the model
// supports speed and loudness at all is mtmd_helper_gen_audio_set_input's answer (it returns 2);
// the ranges are upstream's (set_input for speed and loudness, the /tts schema for top_p).
[[nodiscard]] inline std::string synthesis_options_refusal(const synthesis_options &options) {
    if (std::isnan(options.temperature)) {
        return "temperature must be a number";
    }
    if (!std::isfinite(options.top_p) || options.top_p < 0.0f || options.top_p > 1.0f) {
        return "top_p must be between 0 and 1";
    }
    if (!std::isfinite(options.repeat_penalty) || options.repeat_penalty <= 0.0f) {
        return "repeat_penalty must be > 0 (1 = off)";
    }
    if (!std::isfinite(options.speed) || options.speed <= 0.0f) {
        return "speed must be > 0 (1 = the model's own pace)";
    }
    if (!std::isfinite(options.loudness) || options.loudness > 0.0f) {
        return "loudness must be < 0 LUFS (0 = the model's own loudness)";
    }
    // b12459 (#32369): upstream's ranges, set_input's and the /tts schema's
    if (!std::isfinite(options.cfg_scale) || options.cfg_scale < 0.0f) {
        return "cfg_scale must be > 0 (0 = the model's own, 1 = no guidance)";
    }
    if (options.flow_steps < 0 || options.flow_steps > 1024) {
        return "flow_steps must be between 1 and 1024 (0 = the model's own)";
    }
    return "";
}

// Whether the request leaves speed and loudness to the model: only then may a model without
// delivery control (every one but Zonos3 at b12431) speak it.
[[nodiscard]] inline bool synthesis_uses_delivery(const synthesis_options &options) {
    return options.speed != 1.0f || options.loudness != 0.0f;
}

// Whether the request sets the guidance or the solver steps (b12459): only a model that supports
// them (YuE2) may take a value other than the model's own.
[[nodiscard]] inline bool synthesis_uses_generation_control(const synthesis_options &options) {
    return options.cfg_scale != 0.0f || options.flow_steps != 0;
}

// The repetition-penalty window: the caller's frame cap, else 512 -- upstream's /tts before b12459,
// kept for the window because a model limit of "none" is no window size.
[[nodiscard]] inline int synthesis_penalty_window(int requested) { return requested > 0 ? requested : 512; }

// The top-p handed on, 1 (off) for 0 -- upstream's "0 to use the defaults".
[[nodiscard]] inline float synthesis_top_p(const synthesis_options &options) {
    return options.top_p > 0.0f && options.top_p < 1.0f ? options.top_p : 1.0f;
}

} // namespace jllama_tts

#endif // JLLAMA_TTS_PARAMS_HPP
