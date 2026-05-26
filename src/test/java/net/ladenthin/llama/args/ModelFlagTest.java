// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
// SPDX-FileCopyrightText: 2023-2025 Konstantin Herud
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.args;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.Arrays;
import java.util.Collection;

import static org.junit.jupiter.api.Assertions.*;

public class ModelFlagTest {

    public static Collection<Object[]> data() {
        return Arrays.asList(new Object[][]{
            {ModelFlag.NO_CONTEXT_SHIFT,       "--no-context-shift"},
            {ModelFlag.FLASH_ATTN,             "--flash-attn"},
            {ModelFlag.NO_PERF,                "--no-perf"},
            {ModelFlag.ESCAPE,                 "--escape"},
            {ModelFlag.NO_ESCAPE,              "--no-escape"},
            {ModelFlag.SPECIAL,                "--special"},
            {ModelFlag.NO_WARMUP,              "--no-warmup"},
            {ModelFlag.SPM_INFILL,             "--spm-infill"},
            {ModelFlag.IGNORE_EOS,             "--ignore-eos"},
            {ModelFlag.DUMP_KV_CACHE,          "--dump-kv-cache"},
            {ModelFlag.NO_KV_OFFLOAD,          "--no-kv-offload"},
            {ModelFlag.CONT_BATCHING,          "--cont-batching"},
            {ModelFlag.NO_CONT_BATCHING,       "--no-cont-batching"},
            {ModelFlag.MLOCK,                  "--mlock"},
            {ModelFlag.NO_MMAP,                "--no-mmap"},
            {ModelFlag.CHECK_TENSORS,          "--check-tensors"},
            {ModelFlag.EMBEDDING,              "--embedding"},
            {ModelFlag.RERANKING,              "--reranking"},
            {ModelFlag.LORA_INIT_WITHOUT_APPLY,"--lora-init-without-apply"},
            {ModelFlag.LOG_DISABLE,            "--log-disable"},
            {ModelFlag.VERBOSE,                "--verbose"},
            {ModelFlag.LOG_PREFIX,             "--log-prefix"},
            {ModelFlag.LOG_TIMESTAMPS,         "--log-timestamps"},
            {ModelFlag.JINJA,                  "--jinja"},
            {ModelFlag.VOCAB_ONLY,             "--vocab-only"},
            {ModelFlag.KV_UNIFIED,             "--kv-unified"},
            {ModelFlag.NO_KV_UNIFIED,          "--no-kv-unified"},
            {ModelFlag.CLEAR_IDLE,             "--cache-idle-slots"},
            {ModelFlag.NO_CLEAR_IDLE,          "--no-cache-idle-slots"},
            {ModelFlag.MMPROJ_AUTO,            "--mmproj-auto"},
            {ModelFlag.MMPROJ_OFFLOAD,         "--mmproj-offload"},
        });
    }

    @ParameterizedTest(name = "{0} -> {1}")
    @MethodSource("data")
    public void testGetCliFlag(ModelFlag flag, String expectedCliFlag) {
        assertEquals(expectedCliFlag, flag.getCliFlag());
    }

    // ------------------------------------------------------------------
    // Structural invariants
    // ------------------------------------------------------------------

    @Test
    public void testEnumCount() {
        assertEquals(31, ModelFlag.values().length);
    }

    @ParameterizedTest(name = "{0} -> {1}")
    @MethodSource("data")
    public void testCliFlagStartsWithDoubleDash(ModelFlag flag, String expectedCliFlag) {
        assertTrue(flag.getCliFlag().startsWith("--"), "Flag " + flag + " must start with --");
    }

    @ParameterizedTest(name = "{0} -> {1}")
    @MethodSource("data")
    public void testCliFlagNonEmpty(ModelFlag flag, String expectedCliFlag) {
        assertFalse(flag.getCliFlag().isEmpty(), "Flag " + flag + " has empty CLI string");
    }
}
