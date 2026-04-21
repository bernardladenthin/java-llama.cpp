package de.kherud.llama.args;

import org.junit.Test;

import static org.junit.Assert.*;

public class ModelFlagTest {

    @Test
    public void testEnumCount() {
        assertEquals(29, ModelFlag.values().length);
    }

    @Test
    public void testNoContextShift() {
        assertEquals("--no-context-shift", ModelFlag.NO_CONTEXT_SHIFT.getCliFlag());
    }

    @Test
    public void testFlashAttn() {
        assertEquals("--flash-attn", ModelFlag.FLASH_ATTN.getCliFlag());
    }

    @Test
    public void testNoPerf() {
        assertEquals("--no-perf", ModelFlag.NO_PERF.getCliFlag());
    }

    @Test
    public void testEscape() {
        assertEquals("--escape", ModelFlag.ESCAPE.getCliFlag());
    }

    @Test
    public void testNoEscape() {
        assertEquals("--no-escape", ModelFlag.NO_ESCAPE.getCliFlag());
    }

    @Test
    public void testSpecial() {
        assertEquals("--special", ModelFlag.SPECIAL.getCliFlag());
    }

    @Test
    public void testNoWarmup() {
        assertEquals("--no-warmup", ModelFlag.NO_WARMUP.getCliFlag());
    }

    @Test
    public void testSpmInfill() {
        assertEquals("--spm-infill", ModelFlag.SPM_INFILL.getCliFlag());
    }

    @Test
    public void testIgnoreEos() {
        assertEquals("--ignore-eos", ModelFlag.IGNORE_EOS.getCliFlag());
    }

    @Test
    public void testDumpKvCache() {
        assertEquals("--dump-kv-cache", ModelFlag.DUMP_KV_CACHE.getCliFlag());
    }

    @Test
    public void testNoKvOffload() {
        assertEquals("--no-kv-offload", ModelFlag.NO_KV_OFFLOAD.getCliFlag());
    }

    @Test
    public void testContBatching() {
        assertEquals("--cont-batching", ModelFlag.CONT_BATCHING.getCliFlag());
    }

    @Test
    public void testNoContBatching() {
        assertEquals("--no-cont-batching", ModelFlag.NO_CONT_BATCHING.getCliFlag());
    }

    @Test
    public void testMlock() {
        assertEquals("--mlock", ModelFlag.MLOCK.getCliFlag());
    }

    @Test
    public void testNoMmap() {
        assertEquals("--no-mmap", ModelFlag.NO_MMAP.getCliFlag());
    }

    @Test
    public void testCheckTensors() {
        assertEquals("--check-tensors", ModelFlag.CHECK_TENSORS.getCliFlag());
    }

    @Test
    public void testEmbedding() {
        assertEquals("--embedding", ModelFlag.EMBEDDING.getCliFlag());
    }

    @Test
    public void testReranking() {
        assertEquals("--reranking", ModelFlag.RERANKING.getCliFlag());
    }

    @Test
    public void testLoraInitWithoutApply() {
        assertEquals("--lora-init-without-apply", ModelFlag.LORA_INIT_WITHOUT_APPLY.getCliFlag());
    }

    @Test
    public void testLogDisable() {
        assertEquals("--log-disable", ModelFlag.LOG_DISABLE.getCliFlag());
    }

    @Test
    public void testVerbose() {
        assertEquals("--verbose", ModelFlag.VERBOSE.getCliFlag());
    }

    @Test
    public void testLogPrefix() {
        assertEquals("--log-prefix", ModelFlag.LOG_PREFIX.getCliFlag());
    }

    @Test
    public void testLogTimestamps() {
        assertEquals("--log-timestamps", ModelFlag.LOG_TIMESTAMPS.getCliFlag());
    }

    @Test
    public void testJinja() {
        assertEquals("--jinja", ModelFlag.JINJA.getCliFlag());
    }

    @Test
    public void testVocabOnly() {
        assertEquals("--vocab-only", ModelFlag.VOCAB_ONLY.getCliFlag());
    }

    @Test
    public void testKvUnified() {
        assertEquals("--kv-unified", ModelFlag.KV_UNIFIED.getCliFlag());
    }

    @Test
    public void testNoKvUnified() {
        assertEquals("--no-kv-unified", ModelFlag.NO_KV_UNIFIED.getCliFlag());
    }

    @Test
    public void testClearIdle() {
        assertEquals("--clear-idle", ModelFlag.CLEAR_IDLE.getCliFlag());
    }

    @Test
    public void testNoClearIdle() {
        assertEquals("--no-clear-idle", ModelFlag.NO_CLEAR_IDLE.getCliFlag());
    }

    @Test
    public void testAllFlagsStartWithDoubleDash() {
        for (ModelFlag flag : ModelFlag.values()) {
            assertTrue("Flag " + flag + " must start with --", flag.getCliFlag().startsWith("--"));
        }
    }

    @Test
    public void testAllFlagsNonEmpty() {
        for (ModelFlag flag : ModelFlag.values()) {
            assertFalse("Flag " + flag + " has empty CLI string", flag.getCliFlag().isEmpty());
        }
    }
}
