package de.kherud.llama;

import org.junit.Test;

import static org.junit.Assert.*;

public class StopReasonTest {

    // ------------------------------------------------------------------
    // getStopType — forward direction
    // ------------------------------------------------------------------

    @Test
    public void testNoneStopTypeIsNull() {
        assertNull(StopReason.NONE.getStopType());
    }

    @Test
    public void testEosStopType() {
        assertEquals("eos", StopReason.EOS.getStopType());
    }

    @Test
    public void testStopStringStopType() {
        assertEquals("word", StopReason.STOP_STRING.getStopType());
    }

    @Test
    public void testMaxTokensStopType() {
        assertEquals("limit", StopReason.MAX_TOKENS.getStopType());
    }

    // ------------------------------------------------------------------
    // fromStopType — reverse direction
    // ------------------------------------------------------------------

    @Test
    public void testFromStopType_eos() {
        assertEquals(StopReason.EOS, StopReason.fromStopType("eos"));
    }

    @Test
    public void testFromStopType_word() {
        assertEquals(StopReason.STOP_STRING, StopReason.fromStopType("word"));
    }

    @Test
    public void testFromStopType_limit() {
        assertEquals(StopReason.MAX_TOKENS, StopReason.fromStopType("limit"));
    }

    @Test
    public void testFromStopType_emptyStringReturnsNone() {
        assertEquals(StopReason.NONE, StopReason.fromStopType(""));
    }

    @Test
    public void testFromStopType_nullReturnsNone() {
        assertEquals(StopReason.NONE, StopReason.fromStopType(null));
    }

    @Test
    public void testFromStopType_unknownReturnsNone() {
        assertEquals(StopReason.NONE, StopReason.fromStopType("something_else"));
    }

    // ------------------------------------------------------------------
    // Round-trips
    // ------------------------------------------------------------------

    @Test
    public void testRoundTrip_eos() {
        assertEquals(StopReason.EOS, StopReason.fromStopType(StopReason.EOS.getStopType()));
    }

    @Test
    public void testRoundTrip_stopString() {
        assertEquals(StopReason.STOP_STRING, StopReason.fromStopType(StopReason.STOP_STRING.getStopType()));
    }

    @Test
    public void testRoundTrip_maxTokens() {
        assertEquals(StopReason.MAX_TOKENS, StopReason.fromStopType(StopReason.MAX_TOKENS.getStopType()));
    }

    @Test
    public void testEnumCount() {
        assertEquals(4, StopReason.values().length);
    }
}
