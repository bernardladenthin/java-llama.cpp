package net.ladenthin.llama;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.Arrays;
import java.util.Collection;

import static org.junit.Assert.*;

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
@RunWith(Parameterized.class)
public class StopReasonTest {

    @Parameterized.Parameters(name = "{0}")
    public static Collection<StopReason> data() {
        return Arrays.asList(StopReason.values());
    }

    private final StopReason reason;

    public StopReasonTest(StopReason reason) {
        this.reason = reason;
    }

    @Test
    public void testRoundTrip() {
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
