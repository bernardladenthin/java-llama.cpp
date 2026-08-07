// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT
//
// Native text-to-speech engine: a thin orchestration over llama.cpp's upstream Qwen3-TTS
// audio-generation pipeline (tools/mtmd/mtmd-helper.h's mtmd_helper::gen_audio), single-stream
// (n_parallel = 1). Kept out of jllama.cpp so the JNI layer stays thin.
//
// Loads a backbone text model plus an mmproj (speaker encoder + code predictor + code2wav
// decoder, all in one GGUF) and drives mtmd_helper::gen_audio's streaming API: set_input() ->
// step_prompt() -> a step_gen() loop (we own the semantic-token sampling, same pattern upstream's
// own tools/tts/tts.cpp main() uses) -> get_output(). Upstream owns the DSP/prompt-building/code2wav
// internals entirely — there is nothing left here to extract or hand-copy from llama.cpp source,
// unlike the OuteTTS pipeline this replaced (see docs/history/llama-cpp-breaking-changes.md for the
// b10269->b10270 upstream architecture replacement this followed). The in-memory WAV writer is ours
// (tts_wav.hpp): the engine asks upstream for raw PCM and encodes it locally, keeping that already
// -tested code in the loop rather than trusting upstream's own WAV framing.

#ifndef JLLAMA_TTS_ENGINE_H
#define JLLAMA_TTS_ENGINE_H

#include <cstdint>
#include <string>
#include <vector>

namespace jllama_tts {

// Opaque handle owning the loaded backbone model/context and the mmproj mtmd context. Created by
// engine_init, freed by engine_free.
struct tts_engine;

// Load the backbone (text) model and the mmproj (speaker encoder + code predictor + code2wav)
// GGUF. Returns nullptr and sets `err` on failure, including when the mmproj does not support
// audio generation (e.g. a vision/audio-input mmproj passed by mistake).
tts_engine *engine_init(const std::string &model_path, const std::string &mmproj_path, int n_gpu_layers, int n_threads,
                        std::string &err);

// Synthesize `text` to a 24 kHz mono 16-bit WAV byte stream in `out_wav`. `speaker_reference_path`
// may be empty (uses the model's default voice); `lang` selects the codec_language special token
// (e.g. "english", "chinese" — see upstream tools/tts/README.md for the supported set) and may be
// empty (defaults to "english"). Returns false and sets `err` on failure. Thread-compatible but not
// re-entrant on the same engine instance.
bool engine_synthesize(tts_engine *engine, const std::string &text, const std::string &speaker_reference_path,
                       const std::string &lang, int n_predict, int top_k, uint32_t seed, std::vector<uint8_t> &out_wav,
                       std::string &err);

// Release the loaded model/context. Safe on nullptr.
void engine_free(tts_engine *engine);

} // namespace jllama_tts

#endif // JLLAMA_TTS_ENGINE_H
