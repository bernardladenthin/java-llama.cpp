// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.args;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

/**
 * Shared parameterized assertions for enums implementing {@link CliArg}.
 *
 * <p>Subclasses declare {@code @RunWith(Parameterized.class)} and provide a
 * {@code @Parameterized.Parameters} data method returning
 * {@code (enumConstant, expectedArgValue, expectedEnumCount)} rows.
 */
public abstract class AbstractCliArgEnumTest<E extends Enum<E> & CliArg> {

    @ParameterizedTest(name = "{0} -> {1}")
    @MethodSource("data")
    public void testGetArgValue(E value, String expectedArgValue, int expectedEnumCount) {
        assertEquals(expectedArgValue, value.getArgValue());
    }

    @ParameterizedTest(name = "{0} -> {1}")
    @MethodSource("data")
    public void testEnumCount(E value, String expectedArgValue, int expectedEnumCount) {
        assertEquals(expectedEnumCount, value.getDeclaringClass().getEnumConstants().length);
    }

    @ParameterizedTest(name = "{0} -> {1}")
    @MethodSource("data")
    public void testImplementsCliArg(E value, String expectedArgValue, int expectedEnumCount) {
        assertTrue(value instanceof CliArg);
    }

    @ParameterizedTest(name = "{0} -> {1}")
    @MethodSource("data")
    public void testArgValueNonEmpty(E value, String expectedArgValue, int expectedEnumCount) {
        assertNotNull(value.getArgValue());
        assertFalse(value.getArgValue().isEmpty());
    }
}
