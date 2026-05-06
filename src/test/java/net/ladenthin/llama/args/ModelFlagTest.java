package net.ladenthin.llama.args;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.Arrays;
import java.util.Collection;

import static org.junit.Assert.*;

@RunWith(Parameterized.class)
public class ModelFlagTest {

    @Parameterized.Parameters(name = "{0} -> {1}")
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

    private final ModelFlag flag;
    private final String expectedCliFlag;

    public ModelFlagTest(ModelFlag flag, String expectedCliFlag) {
        this.flag = flag;
        this.expectedCliFlag = expectedCliFlag;
    }

    @Test
    public void testGetCliFlag() {
        assertEquals(expectedCliFlag, flag.getCliFlag());
    }

    // ------------------------------------------------------------------
    // Structural invariants
    // ------------------------------------------------------------------

    @Test
    public void testEnumCount() {
        assertEquals(31, ModelFlag.values().length);
    }

    @Test
    public void testCliFlagStartsWithDoubleDash() {
        assertTrue("Flag " + flag + " must start with --", flag.getCliFlag().startsWith("--"));
    }

    @Test
    public void testCliFlagNonEmpty() {
        assertFalse("Flag " + flag + " has empty CLI string", flag.getCliFlag().isEmpty());
    }
}
