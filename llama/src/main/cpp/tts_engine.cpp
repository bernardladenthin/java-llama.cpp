// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT
//
// See tts_engine.h for the design summary.

#include "tts_engine.h"

#include "tts_wav.hpp" // pcm_to_wav16_bytes

#include "common.h"
#include "llama.h"
#include "mtmd-helper.h"
#include "mtmd.h"
#include "sampling.h"

#include <mutex>
#include <string>
#include <vector>

namespace jllama_tts {

struct tts_engine {
    common_init_result_ptr init;
    llama_model *model = nullptr;
    llama_context *ctx = nullptr;
    mtmd::context_ptr mctx;
    int n_threads = 4;
    int n_batch = 2048;
    // Serializes engine_synthesize: it drives llama_decode (via mtmd_helper::gen_audio) on the
    // shared ctx/mctx, so two threads on one engine would race.
    std::mutex synthesize_mutex;
};

tts_engine *engine_init(const std::string &model_path, const std::string &mmproj_path, int n_gpu_layers, int n_threads,
                        std::string &err) {
    llama_backend_init();

    auto engine = new tts_engine();
    engine->n_threads = n_threads > 0 ? n_threads : 4;

    common_params params;
    params.n_ctx = 8192;
    params.n_batch = engine->n_batch;
    params.n_gpu_layers = n_gpu_layers;
    params.cpuparams.n_threads = engine->n_threads;
    params.model.path = model_path;
    // Always enable embd so the backbone's hidden state can be handed to the audio-generation
    // helper between frames (mirrors upstream tools/tts/tts.cpp main()).
    params.embedding = true;

    engine->init = common_init_from_params(params);
    engine->model = engine->init ? engine->init->model() : nullptr;
    engine->ctx = engine->init ? engine->init->context() : nullptr;
    if (engine->model == nullptr || engine->ctx == nullptr) {
        err = "failed to load TTS backbone model: " + model_path;
        engine_free(engine);
        return nullptr;
    }

    mtmd_context_params mtmd_params = mtmd_context_params_default();
    mtmd_params.use_gpu = n_gpu_layers != 0;
    engine->mctx.reset(mtmd_init_from_file(mmproj_path.c_str(), engine->model, mtmd_params));
    if (!engine->mctx) {
        err = "failed to load TTS mmproj: " + mmproj_path;
        engine_free(engine);
        return nullptr;
    }
    if (mtmd_gen_audio_get_info(engine->mctx.get()).type == MTMD_GEN_AUDIO_TYPE_NONE) {
        err = "mmproj does not support audio generation: " + mmproj_path;
        engine_free(engine);
        return nullptr;
    }

    return engine;
}

bool engine_synthesize(tts_engine *engine, const std::string &text, const std::string &speaker_reference_path,
                       const std::string &lang, int n_predict, int top_k, uint32_t seed, std::vector<uint8_t> &out_wav,
                       std::string &err) {
    if (engine == nullptr) {
        err = "engine is null";
        return false;
    }
    // Serialize against concurrent calls on the same engine.
    std::lock_guard<std::mutex> engine_lock(engine->synthesize_mutex);

    common_params_sampling sparams;
    sparams.top_k = top_k > 0 ? top_k : 4;
    sparams.seed = seed;
    sparams.samplers = {COMMON_SAMPLER_TYPE_TOP_K};
    common_sampler *smpl = common_sampler_init(engine->model, sparams);
    if (smpl == nullptr) {
        err = "failed to init sampler";
        return false;
    }

    mtmd::bitmap_ptr speaker_bitmap;
    if (!speaker_reference_path.empty()) {
        auto wrapper = mtmd_helper_bitmap_init_from_file(engine->mctx.get(), speaker_reference_path.c_str(), false);
        if (!wrapper.bitmap) {
            common_sampler_free(smpl);
            err = "failed to load speaker reference audio: " + speaker_reference_path;
            return false;
        }
        speaker_bitmap.reset(wrapper.bitmap);
    }

    mtmd_helper::gen_audio gen(engine->ctx, engine->mctx.get());
    mtmd_helper_gen_audio_inp inp{};
    inp.seq_id = 0;
    inp.prompt = text.c_str();
    inp.prompt_len = text.size();
    inp.speaker_ref = speaker_bitmap.get();
    inp.lang = lang.empty() ? "english" : lang.c_str();
    inp.top_k = top_k > 0 ? top_k : 50;
    inp.top_p = 1.0f;
    inp.seed = seed;
    // We do our own WAV framing (pcm_to_wav16_bytes) rather than trust upstream's own writer, so
    // that already-tested code stays in the loop; ask for raw PCM.
    inp.out_type = MTMD_HELPER_GEN_AUDIO_OUTTYPE_PCM;

    if (gen.set_input(&inp) != 0) {
        common_sampler_free(smpl);
        err = "failed to set TTS input (prompt/speaker-reference/lang rejected)";
        return false;
    }

    for (;;) {
        int32_t remaining = gen.step_prompt(engine->n_batch);
        if (remaining < 0) {
            common_sampler_free(smpl);
            err = "prompt processing failed";
            return false;
        }
        if (remaining == 0) {
            break;
        }
    }

    // note: some pipelines ignore this token and drive generation from the hidden state instead
    auto sample_semantic_code = [&]() -> llama_token {
        llama_token t = common_sampler_sample(smpl, engine->ctx, -1);
        common_sampler_accept(smpl, t, true);
        return t;
    };

    const int max_new = n_predict > 0 ? n_predict : 512;
    int n_frames = 0;
    llama_token sampled = sample_semantic_code();
    const float *h_state = llama_get_embeddings_ith(engine->ctx, -1);

    // End-of-speech is reported by step_gen() itself (out_stop / a null next hidden state) rather
    // than by an end-of-generation backbone token, so that pipelines without a discrete backbone
    // token (pocket-tts) terminate too. Mirrors upstream tools/tts/tts.cpp.
    bool stop = false;
    while (!stop && n_frames < max_new) {
        const float *h_next = nullptr;
        if (gen.step_gen(sampled, h_state, &h_next, &stop) != 0) {
            common_sampler_free(smpl);
            err = "audio-frame generation failed at frame " + std::to_string(n_frames);
            return false;
        }
        if (h_next == nullptr) {
            break; // stopped without generating a frame
        }
        n_frames++;
        h_state = h_next;
        sampled = sample_semantic_code();
    }
    common_sampler_free(smpl);

    int32_t sample_rate = 0;
    const char *data = nullptr;
    size_t data_len = 0;
    if (gen.get_output(&sample_rate, &data, &data_len) != 0) {
        err = "failed to read generated audio output";
        return false;
    }
    if (n_frames == 0 || data_len == 0) {
        err = "no audio was generated";
        return false;
    }

    // out_type=PCM: raw F32LE samples.
    const auto *samples = reinterpret_cast<const float *>(data);
    const size_t n_samples = data_len / sizeof(float);
    std::vector<float> audio(samples, samples + n_samples);

    out_wav = pcm_to_wav16_bytes(audio, sample_rate);
    return true;
}

void engine_free(tts_engine *engine) {
    if (engine == nullptr) {
        return;
    }
    // init owns the model + context and frees them on destruction; mctx (mtmd::context_ptr) frees
    // the mmproj context in its own destructor.
    delete engine;
}

} // namespace jllama_tts
