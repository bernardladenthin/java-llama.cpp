// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
// SPDX-FileCopyrightText: 2023-2025 Konstantin Herud
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Round-trip tests for {@link StopReason}.
 *
 * <p>The parameterised suite drives one test per enum constant: it obtains the
 * constant's {@code "stop_type"} string via {@link StopReason#getStopType()} and
 * verifies that feeding it back into {@link StopReason#fromStopType(String)} returns
 * the original constant.  The data provider is {@link StopReason#values()} so the
 * suite automatically covers any future constant added to the enum.
 *
 * <p>Edge cases (null, empty string, unknown value) are tested in separate
 * {@code @Test} methods below the round-trip test.
 */
public class StopReasonTest {

    @ParameterizedTest(name = "{0}")
    @EnumSource(StopReason.class)
    public void testRoundTrip(StopReason reason) {
        assertSame(reason, StopReason.fromStopType(reason.getStopType()));
    }

    // ------------------------------------------------------------------
    // Edge cases — tested separately from the round-trip
    // ------------------------------------------------------------------

    @Test
    public void testFromStopType_nullReturnsNone() {
        assertSame(StopReason.NONE, StopReason.fromStopType(null));
    }

    @Test
    public void testFromStopType_emptyStringReturnsNone() {
        assertSame(StopReason.NONE, StopReason.fromStopType(""));
    }

    @Test
    public void testFromStopType_unknownReturnsNone() {
        assertSame(StopReason.NONE, StopReason.fromStopType("something_else"));
    }

    @Test
    public void testEnumCount() {
        assertEquals(4, StopReason.values().length);
    }
}
