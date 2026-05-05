package net.ladenthin.llama.args;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.Arrays;
import java.util.Collection;

import static org.junit.Assert.*;

@RunWith(Parameterized.class)
public class MiroStatTest {

    @Parameterized.Parameters(name = "{0} -> {1}")
    public static Collection<Object[]> data() {
        return Arrays.asList(new Object[][]{
            {MiroStat.DISABLED, "0"},
            {MiroStat.V1,       "1"},
            {MiroStat.V2,       "2"},
        });
    }

    private final MiroStat miroStat;
    private final String expectedArgValue;

    public MiroStatTest(MiroStat miroStat, String expectedArgValue) {
        this.miroStat = miroStat;
        this.expectedArgValue = expectedArgValue;
    }

    @Test
    public void testGetArgValue() {
        assertEquals(expectedArgValue, miroStat.getArgValue());
    }

    // ------------------------------------------------------------------
    // Structural invariants
    // ------------------------------------------------------------------

    @Test
    public void testEnumCount() {
        assertEquals(3, MiroStat.values().length);
    }

    @Test
    public void testImplementsCliArg() {
        assertTrue(miroStat instanceof CliArg);
    }

    @Test
    public void testArgValueNonEmpty() {
        assertNotNull(miroStat.getArgValue());
        assertFalse(miroStat.getArgValue().isEmpty());
    }
}
