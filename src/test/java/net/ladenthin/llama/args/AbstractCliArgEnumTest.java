// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
// SPDX-FileCopyrightText: 2023-2025 Konstantin Herud
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.args;

import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

/**
 * Shared parameterized assertions for enums implementing {@link CliArg}.
 *
 * <p>Subclasses declare {@code @RunWith(Parameterized.class)} and provide a
 * {@code @Parameterized.Parameters} data method returning
 * {@code (enumConstant, expectedArgValue, expectedEnumCount)} rows.
 */
public abstract class AbstractCliArgEnumTest<E extends Enum<E> & CliArg> {

    private final E value;
    private final String expectedArgValue;
    private final int expectedEnumCount;

    protected AbstractCliArgEnumTest(E value, String expectedArgValue, int expectedEnumCount) {
        this.value = value;
        this.expectedArgValue = expectedArgValue;
        this.expectedEnumCount = expectedEnumCount;
    }

    @Test
    public void testGetArgValue() {
        assertEquals(expectedArgValue, value.getArgValue());
    }

    @Test
    public void testEnumCount() {
        assertEquals(expectedEnumCount, value.getDeclaringClass().getEnumConstants().length);
    }

    @Test
    public void testImplementsCliArg() {
        assertTrue(value instanceof CliArg);
    }

    @Test
    public void testArgValueNonEmpty() {
        assertNotNull(value.getArgValue());
        assertFalse(value.getArgValue().isEmpty());
    }
}
