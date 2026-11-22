// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT
//
// Guards the TTS common_params builder. Model-free, JVM-free, runs in the ordinary C++ suite on
// every platform.
//
// Why this exists even though TtsIntegrationTest already covers it: that test does cover it, but
// only by loading a 1.7B GGUF plus an mmproj, and it reports the defect as a native SIGSEGV that
// kills the whole surefire fork and takes ~1300 unrelated tests down with it. Turning that into a
// one-line assertion is the entire point -- the failure it pins cost a full debugging cycle
// (hs_err retrieval, cross-platform frame comparison, disassembly) to identify.

#include "cpu_params.hpp"
#include "train_params.hpp"
#include "tts_params.hpp"

#include <gtest/gtest.h>

namespace {

// ---------------------------------------------------------------------------
// The regression guard. It fails if anyone drops the postprocess_cpu_params calls from
// build_tts_params, because the builder -- not a copy of it -- is what is exercised here.
// ---------------------------------------------------------------------------
TEST(TtsParams, ResolvesBothCpuThreadCounts) {
    const common_params params = jllama_tts::build_tts_params("model.gguf", 0, 4, 2048);

    EXPECT_EQ(params.cpuparams.n_threads, 4);
    // The one that actually crashed: left at -1, common_threadpools::init creates a second,
    // broken threadpool because tpp and tpp_batch no longer match.
    EXPECT_GT(params.cpuparams_batch.n_threads, 0);
    EXPECT_EQ(params.cpuparams_batch.n_threads, params.cpuparams.n_threads);
}

// A non-positive thread count must fall back, and the batch pool must still inherit it.
TEST(TtsParams, NonPositiveThreadCountFallsBackAndStillInherits) {
    for (const int requested : {0, -1, -8}) {
        const common_params params = jllama_tts::build_tts_params("model.gguf", 0, requested, 2048);

        EXPECT_EQ(params.cpuparams.n_threads, jllama_tts::TTS_DEFAULT_THREADS) << "requested=" << requested;
        EXPECT_EQ(params.cpuparams_batch.n_threads, params.cpuparams.n_threads) << "requested=" << requested;
    }
}

// The remaining fields the engine depends on, so a future edit to the builder cannot silently
// drop one of them.
TEST(TtsParams, CarriesTheEngineSettings) {
    const common_params params = jllama_tts::build_tts_params("some/model.gguf", 33, 6, 1024);

    EXPECT_EQ(params.model.path, "some/model.gguf");
    EXPECT_EQ(params.n_gpu_layers, 33);
    EXPECT_EQ(params.n_batch, 1024u);
    EXPECT_EQ(params.n_ctx, jllama_tts::TTS_N_CTX);
    // gen_audio hands the backbone's hidden state back between frames; without embd it is absent.
    EXPECT_TRUE(params.embedding);
}

// ---------------------------------------------------------------------------
// The speaker-reference rule engine_synthesize applies before it touches the model: a pair that does
// not clone (pair_takes_speaker_reference false) refuses a reference by name; no reference,
// or a pair that clones voices, goes on. The pair itself needs a TTS GGUF, which TtsIntegrationTest
// loads; this pins the rule and the message without one.
// ---------------------------------------------------------------------------
TEST(TtsParams, TheProjectorComputesWithTheEngineThreadCount) {
    // mtmd_context_params_default() says 4, so 1 and 7 are the counts that show whether the engine's
    // reached the projector (red when build_tts_mtmd_params leaves the default).
    ASSERT_EQ(mtmd_context_params_default().n_threads, 4);
    const common_params one = jllama_tts::build_tts_params("m.gguf", 0, 1, 64);
    const mtmd_context_params cpu = jllama_tts::build_tts_mtmd_params(0, one.cpuparams.n_threads);
    EXPECT_EQ(cpu.n_threads, 1);
    EXPECT_FALSE(cpu.use_gpu);
    const common_params seven = jllama_tts::build_tts_params("m.gguf", 99, 7, 64);
    const mtmd_context_params gpu = jllama_tts::build_tts_mtmd_params(99, seven.cpuparams.n_threads);
    EXPECT_EQ(gpu.n_threads, 7);
    EXPECT_TRUE(gpu.use_gpu);
    // without a count the engine's default reaches the projector too
    const common_params fallback = jllama_tts::build_tts_params("m.gguf", 0, 0, 64);
    EXPECT_EQ(jllama_tts::build_tts_mtmd_params(0, fallback.cpuparams.n_threads).n_threads,
              jllama_tts::TTS_DEFAULT_THREADS);
}

TEST(TtsSpeakerReference, NoReferenceIsAlwaysAccepted) {
    EXPECT_EQ(jllama_tts::speaker_reference_refusal(false, ""), "");
    EXPECT_EQ(jllama_tts::speaker_reference_refusal(true, ""), "");
}

TEST(TtsSpeakerReference, APairThatClonesVoicesAcceptsAReference) {
    EXPECT_EQ(jllama_tts::speaker_reference_refusal(true, "voice.wav"), "");
}

TEST(TtsSpeakerReference, APairWithoutAVoiceEncoderRefusesTheReferenceByName) {
    const std::string refusal = jllama_tts::speaker_reference_refusal(false, "clips/voice.wav");
    EXPECT_NE(refusal.find("does not accept a speaker reference"), std::string::npos) << refusal;
    EXPECT_NE(refusal.find("clips/voice.wav"), std::string::npos) << refusal;
}

// Which pipelines clone a voice from a clip (read from mtmd-helper-gen.cpp at b12541): the six that
// call encode_speaker on a reference, and none of the others. llama.cpp's audio_gen_spk_ref names
// only Qwen3-TTS and Pocket TTS, so the rule must not depend on it for the other four.
TEST(TtsSpeakerReference, TheSevenPipelinesThatEncodeAClipCloneAVoice) {
    // Drama-3 since b12590 (#32586)
    for (const mtmd_gen_audio_type type :
         {MTMD_GEN_AUDIO_TYPE_QWEN3TTS, MTMD_GEN_AUDIO_TYPE_POCKETTTS, MTMD_GEN_AUDIO_TYPE_CHATTERBOX,
          MTMD_GEN_AUDIO_TYPE_FISH_S2, MTMD_GEN_AUDIO_TYPE_ZONOS3, MTMD_GEN_AUDIO_TYPE_VOXCPM2,
          MTMD_GEN_AUDIO_TYPE_DRAMA3}) {
        EXPECT_TRUE(jllama_tts::pipeline_clones_voice_from_clip(type)) << type;
        EXPECT_TRUE(jllama_tts::pair_takes_speaker_reference(false, type, true)) << type;
    }
}

TEST(TtsSpeakerReference, ThePipelinesWithoutAClipEncoderDoNotClone) {
    for (const mtmd_gen_audio_type type :
         {MTMD_GEN_AUDIO_TYPE_NONE, MTMD_GEN_AUDIO_TYPE_SOPRANO, MTMD_GEN_AUDIO_TYPE_LFM3A, MTMD_GEN_AUDIO_TYPE_YUE2,
          MTMD_GEN_AUDIO_TYPE_MINIMAX_MUSIC3, MTMD_GEN_AUDIO_TYPE_KOKORO, MTMD_GEN_AUDIO_TYPE_TINY_VOCA}) {
        EXPECT_FALSE(jllama_tts::pipeline_clones_voice_from_clip(type)) << type;
        EXPECT_FALSE(jllama_tts::pair_takes_speaker_reference(false, type, true)) << type;
    }
}

TEST(TtsSpeakerReference, AProjectorWithoutAudioInTakesNoReferenceWhateverThePipeline) {
    EXPECT_FALSE(jllama_tts::pair_takes_speaker_reference(true, MTMD_GEN_AUDIO_TYPE_QWEN3TTS, false));
    EXPECT_FALSE(jllama_tts::pair_takes_speaker_reference(false, MTMD_GEN_AUDIO_TYPE_ZONOS3, false));
    // a pipeline llama.cpp's capability names (a future one) is taken as it says
    EXPECT_TRUE(jllama_tts::pair_takes_speaker_reference(true, MTMD_GEN_AUDIO_TYPE_NONE, true));
}

// The language handed to the pipeline (b11941, #31257): a prompt rendered from a "tts" chat template
// (LFM3-Audio) leaves an empty language empty, so the template's own default ("en") applies; the
// pipelines without such a template (Qwen3-TTS) keep this library's "english".
TEST(TtsPipelineLang, AnExplicitLanguageIsPassedAsGiven) {
    EXPECT_EQ(jllama_tts::tts_pipeline_lang(false, "chinese"), "chinese");
    EXPECT_EQ(jllama_tts::tts_pipeline_lang(true, "de"), "de");
}

TEST(TtsPipelineLang, WithoutATtsTemplateAnEmptyLanguageIsEnglish) {
    EXPECT_EQ(jllama_tts::tts_pipeline_lang(false, ""), "english");
}

TEST(TtsPipelineLang, WithATtsTemplateAnEmptyLanguageStaysEmpty) {
    EXPECT_EQ(jllama_tts::tts_pipeline_lang(true, ""), "");
}

// ---------------------------------------------------------------------------
// Pins the upstream trap the builder works around, in the same spirit as the CommonJsonEnumTrap
// pair in test_json_helpers.cpp. If upstream ever gives common_cpu_params a sane default, this
// test flips to red and tells us the workaround can be dropped -- which no other signal would.
// ---------------------------------------------------------------------------
TEST(CommonParamsCpuTrap, RawDefaultsAreUnresolved) {
    common_params raw; // default-init on purpose: {} would value-initialise and hide the defaults

    EXPECT_LT(raw.cpuparams.n_threads, 0) << "upstream default changed -- re-check build_tts_params";
    EXPECT_LT(raw.cpuparams_batch.n_threads, 0) << "upstream default changed -- re-check build_tts_params";
}

// postprocess_cpu_params' role_model argument is what makes the batch pool inherit rather than
// resolve independently; passing nullptr twice would leave the two pools mismatched.
TEST(CommonParamsCpuTrap, RoleModelMakesBatchInheritTheMainCount) {
    common_params params;
    params.cpuparams.n_threads = 3;

    postprocess_cpu_params(params.cpuparams, nullptr);
    postprocess_cpu_params(params.cpuparams_batch, &params.cpuparams);

    EXPECT_EQ(params.cpuparams_batch.n_threads, 3);
}

} // namespace

// ============================================================
// jllama::resolve_cpu_params — the shared guard for every hand-built common_params
//
//   tts_params.hpp and train_params.hpp both assemble a common_params by hand and both must route
//   it through this one resolver. Delete either call site's resolve_cpu_params() and the JVM dies
//   on memset(NULL, 0, huge) with no hs_err_pid log; these tests are the cheap, model-free way to
//   notice. Each builder has its own guard above/below, because testing the resolver alone does
//   NOT cover the call sites: train_engine.cpp is compiled into jllama only, and its integration
//   test is gated on net.ladenthin.llama.train.model, which no CI job sets.
// ============================================================

TEST(ResolveCpuParams, LeavesBothThreadCountsUsable) {
    common_params params;
    // The raw defaults are the crash input: -1 reaches ggml_threadpool_new as a huge size_t.
    ASSERT_LE(params.cpuparams.n_threads, 0);

    jllama::resolve_cpu_params(params);

    EXPECT_GT(params.cpuparams.n_threads, 0);
    EXPECT_GT(params.cpuparams_batch.n_threads, 0);
}

TEST(ResolveCpuParams, BatchInheritsTheMainCountRatherThanResolvingAlone) {
    // The role_model argument on the second postprocess_cpu_params call is the whole point: the
    // batch pool must inherit, or common_threadpools::init sees a mismatch and builds a second pool.
    //
    // The requested count is derived from the host rather than hardcoded: dropping role_model makes
    // the batch pool resolve to the host's own core count, so a fixture that happened to equal that
    // count (a 3-core runner, a 3-CPU container) would pass with the bug present.
    common_params probe;
    jllama::resolve_cpu_params(probe);
    const int host_default = probe.cpuparams.n_threads;
    ASSERT_GT(host_default, 0);
    const int requested = host_default + 1;

    common_params params;
    params.cpuparams.n_threads = requested;

    jllama::resolve_cpu_params(params);

    EXPECT_EQ(params.cpuparams.n_threads, requested);
    EXPECT_EQ(params.cpuparams_batch.n_threads, requested);
    EXPECT_NE(params.cpuparams_batch.n_threads, host_default) << "batch pool resolved alone instead of inheriting";
}

// ============================================================
// build_train_params — the trainer's builder
//
//   The sibling of the TtsParams block. This is the only runnable guard on the trainer's
//   resolve_cpu_params() call: train_engine.cpp is in the jllama target, not jllama_test, and
//   LlamaTrainerIntegrationTest self-skips on every platform because its model is in no
//   models.csv row.
// ============================================================

namespace {

jllama_train::finetune_config minimal_train_config() {
    jllama_train::finetune_config cfg{};
    cfg.model_path = "base.gguf";
    cfg.output_path = "tuned.gguf";
    cfg.epochs = 2;
    cfg.learning_rate = 1e-5f;
    cfg.lr_min = -1.0f;
    cfg.decay_epochs = 0.0f;
    cfg.weight_decay = 0.0f;
    cfg.optimizer = 0;
    cfg.n_ctx = 512;
    cfg.n_gpu_layers = 0;
    cfg.val_split = 0.0f;
    return cfg;
}

} // namespace

TEST(TrainParams, ResolvesBothCpuThreadCounts) {
    const common_params params = jllama_train::build_train_params(minimal_train_config());

    // The one that crashed the JVM: left at -1, ggml_threadpool_new memsets NULL.
    EXPECT_GT(params.cpuparams.n_threads, 0);
    EXPECT_GT(params.cpuparams_batch.n_threads, 0);
    EXPECT_EQ(params.cpuparams_batch.n_threads, params.cpuparams.n_threads);
}

TEST(TrainParams, ForcesTheSettingsTrainingRequires) {
    const common_params params = jllama_train::build_train_params(minimal_train_config());

    // Weights must stay writable (mmap yields read-only pointers) and the KV cache must be f32
    // (OUT_PROD has no f16 support). Losing either turns into a runtime failure deep in ggml-opt.
    EXPECT_EQ(params.load_mode, LLAMA_LOAD_MODE_NONE);
    EXPECT_EQ(params.cache_type_k, GGML_TYPE_F32);
    EXPECT_EQ(params.cache_type_v, GGML_TYPE_F32);
    EXPECT_FALSE(params.escape);
}

TEST(TrainParams, MapsTheConfigOntoTheOptimizerFields) {
    jllama_train::finetune_config cfg = minimal_train_config();
    cfg.optimizer = 1;
    cfg.epochs = 7;
    cfg.n_batch = 64;
    cfg.n_ubatch = 16;

    const common_params params = jllama_train::build_train_params(cfg);

    EXPECT_EQ(params.optimizer, GGML_OPT_OPTIMIZER_TYPE_SGD);
    EXPECT_EQ(params.lr.epochs, 7u);
    EXPECT_EQ(params.n_batch, 64);
    EXPECT_EQ(params.n_ubatch, 16);
    EXPECT_EQ(params.model.path, "base.gguf");
    EXPECT_EQ(params.out_file, "tuned.gguf");
}

TEST(TrainParams, NonPositiveBatchSizesKeepTheNativeDefaults) {
    const common_params defaults{};
    jllama_train::finetune_config cfg = minimal_train_config();
    cfg.n_batch = 0;
    cfg.n_ubatch = 0;

    const common_params params = jllama_train::build_train_params(cfg);

    EXPECT_EQ(params.n_batch, defaults.n_batch);
    EXPECT_EQ(params.n_ubatch, defaults.n_ubatch);
}

TEST(TrainParams, EpochsBelowOneAreClampedSoTheOptimizerRunsAtLeastOnce) {
    jllama_train::finetune_config cfg = minimal_train_config();
    cfg.epochs = 0;

    EXPECT_EQ(jllama_train::build_train_params(cfg).lr.epochs, 1u);
}

namespace {

// Every value differs from the matching common_params / lr_opt default (n_ctx 0, n_gpu_layers -1,
// val_split 0.05f, lr0 1e-5, lr_min -1, decay_epochs -1, wd 0), so a deleted assignment is
// observable. minimal_train_config()'s lr fields are byte-identical to the lr_opt defaults, which is
// why it cannot guard them itself.
//
// epochs (4) MUST stay above decay_epochs (2.5): lr_opt::init() overwrites decay_epochs with epochs
// unless 0 < decay_epochs < epochs, which would both break the decay_epochs assertion and make a
// deleted decay_epochs assignment unobservable.
jllama_train::finetune_config distinctive_train_config() {
    jllama_train::finetune_config cfg = minimal_train_config();
    cfg.epochs = 4;
    cfg.learning_rate = 3e-4f;
    cfg.lr_min = 7e-6f; // > 0 and < lr0 -- the only case in which lr_opt::init() does anything
    cfg.decay_epochs = 2.5f;
    cfg.weight_decay = 0.125f;
    cfg.n_ctx = 1024;
    cfg.n_gpu_layers = 5;
    cfg.val_split = 0.25f;
    return cfg;
}

} // namespace

// The seven cfg fields the other TrainParams tests never look at. Deleting any one of the matching
// assignments in build_train_params leaves every other test in this file green.
TEST(TrainParams, MapsTheContextAndScheduleFields) {
    const common_params params = jllama_train::build_train_params(distinctive_train_config());

    EXPECT_EQ(params.n_ctx, 1024);
    EXPECT_EQ(params.n_gpu_layers, 5);
    EXPECT_FLOAT_EQ(params.val_split, 0.25f);
    EXPECT_FLOAT_EQ(params.lr.lr0, 3e-4f);
    EXPECT_FLOAT_EQ(params.lr.lr_min, 7e-6f);
    EXPECT_FLOAT_EQ(params.lr.wd, 0.125f);
    // Survives init() only because epochs (4) > decay_epochs (2.5) -- see the fixture comment.
    EXPECT_FLOAT_EQ(params.lr.decay_epochs, 2.5f);
}

// The predefined voices of a model (b12061, #31429): upstream's rule of llama-tts and the /tts route,
// which engine_synthesize applies before it builds the input. A model without predefined voices
// never gets a voice bitmap, whatever the caller names (its template takes the name); one with them
// takes the named voice, falls back to its first voice when the caller gave neither a voice nor a
// speaker reference, keeps a speaker reference given alone, and refuses both at once.
TEST(TtsPredefinedVoice, AModelWithoutPredefinedVoicesNeverGetsOne) {
    using jllama_tts::choose_predefined_voice;
    using jllama_tts::predefined_voice;
    EXPECT_EQ(choose_predefined_voice(0, false, false), predefined_voice::NOT_USED);
    EXPECT_EQ(choose_predefined_voice(0, true, false), predefined_voice::NOT_USED);
    EXPECT_EQ(choose_predefined_voice(0, false, true), predefined_voice::NOT_USED);
    EXPECT_EQ(choose_predefined_voice(0, true, true), predefined_voice::NOT_USED);
    EXPECT_EQ(choose_predefined_voice(-1, true, false), predefined_voice::NOT_USED);
}

TEST(TtsPredefinedVoice, AModelWithPredefinedVoicesTakesTheNamedOneOrItsFirst) {
    using jllama_tts::choose_predefined_voice;
    using jllama_tts::predefined_voice;
    EXPECT_EQ(choose_predefined_voice(1, true, false), predefined_voice::NAMED);
    EXPECT_EQ(choose_predefined_voice(5, true, false), predefined_voice::NAMED);
    EXPECT_EQ(choose_predefined_voice(5, false, false), predefined_voice::FIRST);
}

TEST(TtsPredefinedVoice, ASpeakerReferenceIsKeptAloneAndRefusedNextToAVoice) {
    using jllama_tts::choose_predefined_voice;
    using jllama_tts::predefined_voice;
    EXPECT_EQ(choose_predefined_voice(5, false, true), predefined_voice::NOT_USED);
    EXPECT_EQ(choose_predefined_voice(5, true, true), predefined_voice::CONFLICT);
}

// The options of one synthesis (b12431, #32020): every default leaves the choice to the engine or the
// model, and a value no model can take is refused before engine_synthesize touches the model. The
// ranges are upstream's: speed > 0 and loudness < 0 LUFS (mtmd_helper_gen_audio_set_input), top_p in
// [0, 1] (the /tts schema); a repetition penalty must be > 0, as the sampler divides by it.
TEST(TtsSynthesisOptions, TheDefaultsAreAcceptedAndLeaveTheDeliveryToTheModel) {
    const jllama_tts::synthesis_options defaults;
    EXPECT_EQ(jllama_tts::synthesis_options_refusal(defaults), "");
    EXPECT_FALSE(jllama_tts::synthesis_uses_delivery(defaults));
    EXPECT_FLOAT_EQ(jllama_tts::synthesis_top_p(defaults), 1.0f);
    EXPECT_LT(defaults.temperature, 0.0f);
    EXPECT_FLOAT_EQ(defaults.speed, 1.0f);
    EXPECT_FLOAT_EQ(defaults.loudness, 0.0f);
    EXPECT_FLOAT_EQ(defaults.repeat_penalty, 1.0f);
}

TEST(TtsSynthesisOptions, ValuesInRangeAreAcceptedAndAskForDelivery) {
    jllama_tts::synthesis_options options;
    options.speed = 1.25f;
    options.loudness = -20.0f;
    options.top_p = 0.9f;
    options.repeat_penalty = 1.1f;
    options.temperature = 0.0f;
    EXPECT_EQ(jllama_tts::synthesis_options_refusal(options), "");
    EXPECT_TRUE(jllama_tts::synthesis_uses_delivery(options));
    EXPECT_FLOAT_EQ(jllama_tts::synthesis_top_p(options), 0.9f);
    options.speed = 1.0f;
    EXPECT_TRUE(jllama_tts::synthesis_uses_delivery(options)) << "a loudness alone asks for delivery";
    options.loudness = 0.0f;
    options.speed = 0.5f;
    EXPECT_TRUE(jllama_tts::synthesis_uses_delivery(options)) << "a speed alone asks for delivery";
}

TEST(TtsSynthesisOptions, TopPOfZeroOrOneIsOff) {
    jllama_tts::synthesis_options options;
    options.top_p = 0.0f;
    EXPECT_EQ(jllama_tts::synthesis_options_refusal(options), "");
    EXPECT_FLOAT_EQ(jllama_tts::synthesis_top_p(options), 1.0f);
    options.top_p = 1.0f;
    EXPECT_FLOAT_EQ(jllama_tts::synthesis_top_p(options), 1.0f);
}

TEST(TtsSynthesisOptions, ValuesNoModelCanTakeAreRefusedByName) {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    struct refused {
        void (*set)(jllama_tts::synthesis_options &, float);
        float value;
        const char *names;
    };
    const refused cases[] = {
        {[](jllama_tts::synthesis_options &o, float v) { o.speed = v; }, 0.0f, "speed"},
        {[](jllama_tts::synthesis_options &o, float v) { o.speed = v; }, -1.0f, "speed"},
        {[](jllama_tts::synthesis_options &o, float v) { o.speed = v; }, nan, "speed"},
        {[](jllama_tts::synthesis_options &o, float v) { o.speed = v; }, inf, "speed"},
        {[](jllama_tts::synthesis_options &o, float v) { o.loudness = v; }, 0.5f, "loudness"},
        {[](jllama_tts::synthesis_options &o, float v) { o.loudness = v; }, nan, "loudness"},
        {[](jllama_tts::synthesis_options &o, float v) { o.loudness = v; }, -inf, "loudness"},
        {[](jllama_tts::synthesis_options &o, float v) { o.top_p = v; }, 1.5f, "top_p"},
        {[](jllama_tts::synthesis_options &o, float v) { o.top_p = v; }, -0.1f, "top_p"},
        {[](jllama_tts::synthesis_options &o, float v) { o.top_p = v; }, nan, "top_p"},
        {[](jllama_tts::synthesis_options &o, float v) { o.repeat_penalty = v; }, 0.0f, "repeat_penalty"},
        {[](jllama_tts::synthesis_options &o, float v) { o.repeat_penalty = v; }, nan, "repeat_penalty"},
        {[](jllama_tts::synthesis_options &o, float v) { o.temperature = v; }, nan, "temperature"},
        {[](jllama_tts::synthesis_options &o, float v) { o.cfg_scale = v; }, -0.5f, "cfg_scale"},
        {[](jllama_tts::synthesis_options &o, float v) { o.cfg_scale = v; }, nan, "cfg_scale"},
        {[](jllama_tts::synthesis_options &o, float v) { o.cfg_scale = v; }, inf, "cfg_scale"},
        {[](jllama_tts::synthesis_options &o, float v) { o.flow_steps = static_cast<int>(v); }, -1.0f, "flow_steps"},
        {[](jllama_tts::synthesis_options &o, float v) { o.flow_steps = static_cast<int>(v); }, 1025.0f, "flow_steps"},
    };
    for (const refused &c : cases) {
        jllama_tts::synthesis_options options;
        c.set(options, c.value);
        const std::string refusal = jllama_tts::synthesis_options_refusal(options);
        EXPECT_NE(refusal.find(c.names), std::string::npos) << c.names << " = " << c.value << ": " << refusal;
    }
}

// b12459 (#32369): guidance and solver steps start at the model's own and, set, ask for a model that
// supports them (YuE2); the range ends are upstream's. The frame cap is the model's without the
// caller's, but the repetition-penalty window stays 512 then: a model that stops by itself names no cap.
TEST(TtsSynthesisOptions, GuidanceAndSolverStepsAreTheModelsUntilSet) {
    jllama_tts::synthesis_options options;
    EXPECT_FLOAT_EQ(options.cfg_scale, 0.0f);
    EXPECT_EQ(options.flow_steps, 0);
    EXPECT_FALSE(jllama_tts::synthesis_uses_generation_control(options));
    options.cfg_scale = 1.0f;
    EXPECT_EQ(jllama_tts::synthesis_options_refusal(options), "");
    EXPECT_TRUE(jllama_tts::synthesis_uses_generation_control(options)) << "1 is no guidance, but not the default";
    options.cfg_scale = 0.0f;
    options.flow_steps = 1;
    EXPECT_EQ(jllama_tts::synthesis_options_refusal(options), "");
    EXPECT_TRUE(jllama_tts::synthesis_uses_generation_control(options));
    options.flow_steps = 1024;
    EXPECT_EQ(jllama_tts::synthesis_options_refusal(options), "");
    EXPECT_FALSE(jllama_tts::synthesis_uses_delivery(options)) << "the two kinds are refused with their own names";

    EXPECT_EQ(jllama_tts::synthesis_penalty_window(0), 512);
    EXPECT_EQ(jllama_tts::synthesis_penalty_window(-1), 512);
    EXPECT_EQ(jllama_tts::synthesis_penalty_window(40), 40);
}
