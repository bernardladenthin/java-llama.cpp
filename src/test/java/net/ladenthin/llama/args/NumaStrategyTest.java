package net.ladenthin.llama.args;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.Arrays;
import java.util.Collection;

import static org.junit.Assert.*;

@RunWith(Parameterized.class)
public class NumaStrategyTest {

    @Parameterized.Parameters(name = "{0} -> {1}")
    public static Collection<Object[]> data() {
        return Arrays.asList(new Object[][]{
            {NumaStrategy.DISTRIBUTE, "distribute"},
            {NumaStrategy.ISOLATE,    "isolate"},
            {NumaStrategy.NUMACTL,    "numactl"},
        });
    }

    private final NumaStrategy numaStrategy;
    private final String expectedArgValue;

    public NumaStrategyTest(NumaStrategy numaStrategy, String expectedArgValue) {
        this.numaStrategy = numaStrategy;
        this.expectedArgValue = expectedArgValue;
    }

    @Test
    public void testGetArgValue() {
        assertEquals(expectedArgValue, numaStrategy.getArgValue());
    }

    // ------------------------------------------------------------------
    // Structural invariants
    // ------------------------------------------------------------------

    @Test
    public void testEnumCount() {
        assertEquals(3, NumaStrategy.values().length);
    }

    @Test
    public void testImplementsCliArg() {
        assertTrue(numaStrategy instanceof CliArg);
    }

    @Test
    public void testArgValueNonEmpty() {
        assertNotNull(numaStrategy.getArgValue());
        assertFalse(numaStrategy.getArgValue().isEmpty());
    }
}
