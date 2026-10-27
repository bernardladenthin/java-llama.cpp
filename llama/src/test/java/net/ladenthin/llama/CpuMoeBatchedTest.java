// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.is;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import net.ladenthin.llama.loader.NativeLibraryPresence;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

@ClaudeGenerated(
        purpose = "Model-free guard that LlamaModel.setCpuMoeBatched / isCpuMoeBatched reach the CPU backend's "
                + "switch of llama/patches/0022 through JNI and the C API (it lives in a separately loaded CPU "
                + "module in the shipped builds), round-trip both values, and start on the batched path. What "
                + "the two paths compute is pinned by the C++ guard test_moe_repack.cpp.")
class CpuMoeBatchedTest {

    private boolean initial;

    @BeforeEach
    void rememberTheSwitch() {
        assumeTrue(NativeLibraryPresence.onClasspath(), "libjllama not on classpath — skipping the switch guard");
        initial = LlamaModel.isCpuMoeBatched();
    }

    @AfterEach
    void restoreTheSwitch() {
        if (NativeLibraryPresence.onClasspath()) {
            LlamaModel.setCpuMoeBatched(initial);
        }
    }

    @Test
    void theBatchedPathIsTheDefault() {
        assumeTrue(
                System.getenv("GGML_CPU_REPACK_MMID_BATCHED") == null,
                "GGML_CPU_REPACK_MMID_BATCHED is set for this JVM, so the default is not observable");
        assertThat(initial, is(true));
    }

    @Test
    void theSwitchRoundTripsBothValues() {
        LlamaModel.setCpuMoeBatched(false);
        assertThat(LlamaModel.isCpuMoeBatched(), is(false));
        LlamaModel.setCpuMoeBatched(true);
        assertThat(LlamaModel.isCpuMoeBatched(), is(true));
    }
}
