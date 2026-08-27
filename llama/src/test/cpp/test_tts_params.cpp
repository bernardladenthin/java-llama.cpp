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
