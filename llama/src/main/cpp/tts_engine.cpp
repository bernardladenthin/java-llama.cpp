// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT
//
// See tts_engine.h for the design summary.

#include "tts_engine.h"

#include "tts_params.hpp"  // build_tts_params
#include "tts_wav.hpp"     // pcm_to_wav16_bytes
#include "rpc_support.hpp" // exclude_stale_devices

#include "chat.h"
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
    // Whether the loaded pair clones a voice from a clip (pair_takes_speaker_reference), read once at load.
    bool speaker_reference = false;
    // The predefined voices of the projector (b12061, #31429), read once at load; empty for most models.
    std::vector<std::string> voices;
    // mtmd_gen_audio_info::n_channels (b12459): 1 for every model but YuE2, which is stereo.
    int channels = 1;
    // The model's chat templates: a "tts" one (LFM3-Audio, b11941) renders the prompt to speak.
    common_chat_templates_ptr templates;
    // Serializes engine_synthesize: it drives llama_decode (via mtmd_helper::gen_audio) on the
    // shared ctx/mctx, so two threads on one engine would race.
    std::mutex synthesize_mutex;
};

tts_engine *engine_init(const std::string &model_path, const std::string &mmproj_path, int n_gpu_layers, int n_threads,
                        std::string &err) {
    llama_backend_init();

    auto engine = new tts_engine();
    engine->n_threads = n_threads > 0 ? n_threads : 4;

    // Built by tts_params.hpp so the exact params the engine uses are unit-testable without a
    // model -- see the header for why the CPU-thread resolution there is load-bearing.
    common_params params = build_tts_params(model_path, n_gpu_layers, engine->n_threads, engine->n_batch);
    // Keep RPC servers another load registered in this JVM out of this one (see rpc_support.hpp).
    const auto mmproj_device = jllama::rpc::exclude_stale_devices(params.devices);

    engine->init = common_init_from_params(params);
    engine->model = engine->init ? engine->init->model() : nullptr;
    engine->ctx = engine->init ? engine->init->context() : nullptr;
    if (engine->model == nullptr || engine->ctx == nullptr) {
        err = "failed to load TTS backbone model: " + model_path;
        engine_free(engine);
        return nullptr;
    }

    mtmd_context_params mtmd_params = jllama_tts::build_tts_mtmd_params(n_gpu_layers, params.cpuparams.n_threads);
    if (mmproj_device && mmproj_device->empty()) {
        mtmd_params.use_gpu = false;
    } else if (mmproj_device) {
        mtmd_params.device = ggml_backend_dev_by_name(mmproj_device->c_str());
    }
    engine->mctx.reset(mtmd_init_from_file(mmproj_path.c_str(), engine->model, mtmd_params));
    if (!engine->mctx) {
        err = "failed to load TTS mmproj: " + mmproj_path;
        engine_free(engine);
        return nullptr;
    }
    const mtmd_gen_audio_info info = mtmd_gen_audio_get_info(engine->mctx.get());
    if (info.type == MTMD_GEN_AUDIO_TYPE_NONE) {
        err = "mmproj does not support audio generation: " + mmproj_path;
        engine_free(engine);
        return nullptr;
    }
    engine->channels = info.n_channels > 0 ? info.n_channels : 1;
    // The pipeline clones a voice and the projector has the encoder for the clip (tts_params.hpp).
    engine->speaker_reference = jllama_tts::pair_takes_speaker_reference(
        mtmd_helper_get_capabilities(engine->model, engine->mctx.get()).audio_gen_spk_ref, info.type,
        mtmd_support_audio(engine->mctx.get()));
    for (int32_t i = 0; const char *name = mtmd_gen_audio_voice_name(engine->mctx.get(), i); i++) {
        engine->voices.emplace_back(name);
    }
    // As llama-tts: the model's own templates, no override. A backbone without a chat template gets
    // the library's fallback, which has no "tts" variant, so its text is spoken as given.
    engine->templates = common_chat_templates_init(engine->model, "");

    return engine;
}

const std::vector<std::string> &engine_voices(const tts_engine *engine) {
    static const std::vector<std::string> none;
    return engine != nullptr ? engine->voices : none;
}

bool engine_supports_speaker_reference(const tts_engine *engine) {
    return engine != nullptr && engine->speaker_reference;
}

bool engine_synthesize(tts_engine *engine, const std::string &text, const synthesis_options &options,
                       std::vector<uint8_t> &out_wav, std::string &err, bool *request_fault) {
    const std::string &speaker_reference_path = options.speaker_reference_path;
    const std::string &lang = options.lang;
    const std::string &voice = options.voice;
    const int top_k = options.top_k;
    // Set where the request is what was refused (a voice the model does not have, a speaker
    // reference it cannot use or read): the C API answers those as an invalid request.
    const auto refuse = [&](const std::string &message) {
        err = message;
        if (request_fault != nullptr) {
            *request_fault = true;
        }
        return false;
    };
    if (request_fault != nullptr) {
        *request_fault = false;
    }
    if (engine == nullptr) {
        err = "engine is null";
        return false;
    }
    // Nothing to speak: upstream's /tts refuses an empty input (400), and the pipelines answered one --
    // or a text of spaces -- each its own way: audio of nothing, "failed to set TTS input" or "no audio
    // was generated", none of them the caller's mistake.
    if (text.find_first_not_of(" \t\r\n") == std::string::npos) {
        return refuse("the text is empty: there is nothing to speak");
    }
    // Values no model can take, before anything is loaded.
    const std::string invalid = synthesis_options_refusal(options);
    if (!invalid.empty()) {
        return refuse(invalid);
    }
    // Serialize against concurrent calls on the same engine.
    std::lock_guard<std::mutex> engine_lock(engine->synthesize_mutex);

    // Before anything is loaded: a pair without a voice encoder would only reject the reference in
    // set_input, with a message that does not say why.
    const std::string refusal = speaker_reference_refusal(engine->speaker_reference, speaker_reference_path);
    if (!refusal.empty()) {
        return refuse(refusal);
    }

    const float top_p = synthesis_top_p(options);

    // The backbone sampler: top-k always, the repetition penalty over the whole generation and top-p
    // when the request asks for them (upstream's /tts sets penalty_last_n to the frame cap the same way).
    common_params_sampling sparams;
    sparams.top_k = top_k > 0 ? top_k : 4;
    sparams.seed = options.seed;
    sparams.samplers = {};
    if (options.repeat_penalty != 1.0f) {
        sparams.penalty_repeat = options.repeat_penalty;
        sparams.penalty_last_n = synthesis_penalty_window(options.max_frames);
        sparams.samplers.push_back(COMMON_SAMPLER_TYPE_PENALTIES);
    }
    sparams.samplers.push_back(COMMON_SAMPLER_TYPE_TOP_K);
    if (top_p < 1.0f) {
        sparams.top_p = top_p;
        sparams.samplers.push_back(COMMON_SAMPLER_TYPE_TOP_P);
    }
    // Owned: a template or a pipeline call below can throw, and a raw pointer leaked the sampler then.
    common_sampler_ptr smpl(common_sampler_init(engine->model, sparams));
    if (smpl == nullptr) {
        err = "failed to init sampler";
        return false;
    }

    mtmd::bitmap_ptr speaker_bitmap;
    if (!speaker_reference_path.empty()) {
        // b10649 added a fourth `mtmd_helper_init_opt` parameter carrying the video-decode settings
        // (fps target, ffmpeg binary dir, timestamp interval). None of it applies to a speaker
        // reference clip, which is audio, so pass upstream's own defaults rather than inventing values.
        auto wrapper = mtmd_helper_bitmap_init_from_file(engine->mctx.get(), speaker_reference_path.c_str(), false,
                                                         mtmd_helper_init_opt_default());
        if (!wrapper.bitmap) {
            return refuse("failed to load speaker reference audio: " + speaker_reference_path);
        }
        speaker_bitmap.reset(wrapper.bitmap);
        // the helper reads images too: an image named as the reference reached set_input, which
        // refused it with a message that said neither why nor that the request was at fault
        if (!mtmd_bitmap_is_audio(speaker_bitmap.get())) {
            return refuse("the speaker reference is no audio: " + speaker_reference_path);
        }
    }

    // b12061 (#31429): mtmd_helper_gen_audio_inp lost its `voice`; a model with predefined voices
    // takes one as its speaker reference instead (see choose_predefined_voice), every other model
    // keeps naming the voice in the prompt its "tts" template renders below.
    std::string voice_name = voice;
    switch (
        choose_predefined_voice(static_cast<int>(engine->voices.size()), !voice.empty(), speaker_bitmap != nullptr)) {
    case predefined_voice::NOT_USED:
        break;
    case predefined_voice::CONFLICT:
        return refuse("a voice and a speaker reference cannot be used together");
    case predefined_voice::FIRST:
        voice_name = engine->voices.front();
        [[fallthrough]];
    case predefined_voice::NAMED:
        speaker_bitmap.reset(mtmd_gen_audio_get_voice(engine->mctx.get(), voice_name.c_str()));
        if (!speaker_bitmap) {
            std::string message = "unknown voice \"" + voice_name + "\", the model has:";
            for (const std::string &name : engine->voices) {
                message += " " + name;
            }
            return refuse(message);
        }
        break;
    }

    // b11796 (#30902): the helper is default-constructed and initialised explicitly; init() leaves it
    // invalid for a projector without audio generation (checked at load, but the contract is here),
    // and since b12459 also for a context too small for the model's pipeline.
    mtmd_helper::gen_audio gen;
    gen.init(engine->ctx, engine->mctx.get());
    if (!gen.valid()) {
        err = "failed to initialise audio generation: the context cannot run the audio generation of this model";
        return false;
    }
    // b12459: the caller's frame cap, else the model's own (512 for most; a model that stops by
    // itself, YuE2, names none) -- upstream's /tts and llama-tts do the same.
    const int max_new = gen.max_frames(options.max_frames);
    // upstream's defaults (b11796: a `stream` flag joined the struct, off for a one-shot synthesis);
    // every field the caller controls is set below
    // A model with a "tts" chat template (LFM3-Audio) speaks the prompt rendered from it, the others
    // the text itself -- upstream's /tts route and llama-tts do the same (b11941, #31257). Throws on a
    // template error, which the C API reports.
    // b12494: raw_prompt skips the template (the text is phonemes or a whole prompt), as /tts does
    // b12497: the template's tts_g2p() asks the projector's letter-to-sound model (tiny-voca), with
    // the buffer upstream's /tts and llama-tts give it; a model without one answers none
    mtmd_context *mctx = engine->mctx.get();
    const auto g2p = [mctx](const std::string &word) {
        char buf[512];
        return mtmd_gen_audio_g2p(mctx, word.c_str(), buf, sizeof(buf)) > 0 ? std::string(buf) : std::string();
    };
    const std::string rendered = options.raw_prompt
                                     ? std::string()
                                     : common_chat_get_tts_prompt(engine->templates.get(), text, lang, voice_name, g2p);
    const std::string &prompt = rendered.empty() ? text : rendered;
    const std::string pipeline_lang = tts_pipeline_lang(!rendered.empty(), lang);

    mtmd_helper_gen_audio_inp inp = mtmd_helper_gen_audio_inp_default();
    inp.seq_id = 0;
    inp.prompt = prompt.c_str();
    inp.prompt_len = prompt.size();
    inp.speaker_ref = speaker_bitmap.get();
    // b12453 (#32330): the reference's transcript, which Fish S2 puts into its prompt and Qwen3-TTS (b12491,
    // #32438) continues the reference speech from; the others ignore it
    inp.speaker_ref_text = options.speaker_reference_text.empty() ? nullptr : options.speaker_reference_text.c_str();
    inp.lang = pipeline_lang.empty() ? nullptr : pipeline_lang.c_str();
    inp.top_k = top_k > 0 ? top_k : 50;
    inp.top_p = top_p;
    inp.seed = options.seed;
    // b12233 (#31655): the audio frame temperature, < 0 = the model's default (what the struct starts with)
    if (options.temperature >= 0.0f) {
        inp.temp = options.temperature;
    }
    // b12431 (#32020): delivery, for a model that supports it; set_input answers 2 for another one
    inp.speed = options.speed;
    inp.loudness = options.loudness;
    // b12459 (#32369): guidance and solver steps, for a model that supports them; set_input answers 2 otherwise
    inp.cfg_scale = options.cfg_scale;
    inp.flow_steps = options.flow_steps;
    // We do our own WAV framing (pcm_to_wav16_bytes) rather than trust upstream's own writer, so
    // that already-tested code stays in the loop; ask for raw PCM.
    inp.out_type = MTMD_HELPER_GEN_AUDIO_OUTTYPE_PCM;

    // The pipeline numbers the prompt from position 0 and leaves the KV cache to the caller: upstream's
    // server clears the slot (prompt_clear) right before set_input. Without this the previous
    // synthesis is still in the cache and the second one on the same engine fails in step_prompt
    // ("inconsistent sequence positions", X = <last position> > Y = 0).
    llama_memory_seq_rm(llama_get_memory(engine->ctx), inp.seq_id, -1, -1);

    const int32_t input_status = gen.set_input(&inp);
    if (input_status == 2) {
        // the engine's ranges are checked above, so what is left is a model without delivery or generation
        // control, or a value outside the model's own range (VoxCPM2 takes flow_steps 2 to 32 only), which
        // llama.cpp names in its log only
        // both asked for: llama.cpp checks the delivery first, and either can be the cause
        const bool generation = synthesis_uses_generation_control(options);
        const bool delivery = synthesis_uses_delivery(options);
        return refuse(generation && delivery
                          ? "the model does not support speed or loudness, or cfg_scale or flow_steps, or not these "
                            "values (llama.cpp's log names which)"
                      : generation ? "the model does not support cfg_scale or flow_steps, or not these values (its own "
                                     "range is in llama.cpp's log)"
                      : delivery   ? "the model does not support speed or loudness"
                                   : "the TTS input is invalid for the model");
    }
    if (input_status != 0) {
        err = "failed to set TTS input (prompt/speaker-reference/lang rejected)";
        return false;
    }

    for (;;) {
        // b12453: n_batch must not exceed the context's, which embeddings cap at n_ubatch (512, not
        // the 2048 asked for at load); a larger one fails the prompt ("prompt processing failed")
        int32_t remaining = gen.step_prompt(static_cast<int32_t>(llama_n_batch(engine->ctx)));
        if (remaining < 0) {
            err = "prompt processing failed";
            return false;
        }
        if (remaining == 0) {
            break;
        }
    }

    // note: some pipelines ignore this token and drive generation from the hidden state instead
    auto sample_semantic_code = [&]() -> llama_token {
        llama_token t = common_sampler_sample(smpl.get(), engine->ctx, -1);
        common_sampler_accept(smpl.get(), t, true);
        return t;
    };

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

    int32_t sample_rate = 0;
    const char *data = nullptr;
    size_t data_len = 0;
    if (gen.get_output(&sample_rate, &data, &data_len) != 0) {
        err = "failed to read generated audio output";
        return false;
    }
    // n_frames counts the frames step_gen() reported; a pipeline that is not autoregressive (Kokoro and
    // tiny-voca, b12494 and b12497) writes the whole utterance from the prompt and reports none, which
    // is why only the output decides, as in upstream's llama-tts
    if (data_len == 0) {
        err = "no audio was generated";
        return false;
    }

    // out_type=PCM: raw F32LE samples, the channels interleaved (b12459: YuE2 is stereo).
    const auto *samples = reinterpret_cast<const float *>(data);
    const size_t n_samples = data_len / sizeof(float);
    std::vector<float> audio(samples, samples + n_samples);

    out_wav = pcm_to_wav16_bytes(audio, sample_rate, engine->channels);
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
