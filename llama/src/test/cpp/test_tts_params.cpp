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
