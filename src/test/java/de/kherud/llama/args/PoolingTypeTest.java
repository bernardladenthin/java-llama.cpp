package de.kherud.llama.args;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.Arrays;
import java.util.Collection;

import static org.junit.Assert.*;

@RunWith(Parameterized.class)
public class PoolingTypeTest {

    @Parameterized.Parameters(name = "{0} -> {1}")
    public static Collection<Object[]> data() {
        return Arrays.asList(new Object[][]{
            {PoolingType.UNSPECIFIED, "unspecified"},
            {PoolingType.NONE,        "none"},
            {PoolingType.MEAN,        "mean"},
            {PoolingType.CLS,         "cls"},
            {PoolingType.LAST,        "last"},
            {PoolingType.RANK,        "rank"},
        });
    }

    private final PoolingType poolingType;
    private final String expectedArgValue;

    public PoolingTypeTest(PoolingType poolingType, String expectedArgValue) {
        this.poolingType = poolingType;
        this.expectedArgValue = expectedArgValue;
    }

    @Test
    public void testGetArgValue() {
        assertEquals(expectedArgValue, poolingType.getArgValue());
    }

    // ------------------------------------------------------------------
    // Structural invariants
    // ------------------------------------------------------------------

    @Test
    public void testEnumCount() {
        assertEquals(6, PoolingType.values().length);
    }

    @Test
    public void testImplementsCliArg() {
        assertTrue(poolingType instanceof CliArg);
    }

    @Test
    public void testArgValueNonEmpty() {
        assertNotNull(poolingType.getArgValue());
        assertFalse(poolingType.getArgValue().isEmpty());
    }
}
