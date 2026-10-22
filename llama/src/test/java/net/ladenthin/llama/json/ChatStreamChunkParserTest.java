// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.json;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.empty;
import static org.hamcrest.Matchers.hasSize;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.not;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;

/**
 * Unit tests for {@link ChatStreamChunkParser}.
 * No JVM native library or model file needed — JSON string literals only.
 */
public class ChatStreamChunkParserTest {

    private final ChatStreamChunkParser parser = new ChatStreamChunkParser();

    @Test
    public void feed_objectData_emitsOneChunk_andReportsNotStopped() {
        List<String> chunks = new ArrayList<>();
        boolean stop = parser.feed("{\"data\":{\"object\":\"chat.completion.chunk\"},\"stop\":false}", chunks::add);
        assertThat(stop, is(false));
        assertThat(chunks, hasSize(1));
        assertThat(chunks.get(0).contains("chat.completion.chunk"), is(true));
    }

    @Test
    public void feed_arrayData_emitsEachElementInOrder_andReportsStopped() {
        List<String> chunks = new ArrayList<>();
        boolean stop = parser.feed("{\"data\":[{\"i\":1},{\"i\":2}],\"stop\":true}", chunks::add);
        assertThat(stop, is(true));
        assertThat(chunks, hasSize(2));
        assertThat(chunks.get(0).contains("\"i\":1"), is(true));
        assertThat(chunks.get(1).contains("\"i\":2"), is(true));
    }

    @Test
    public void feed_emptyArrayData_emitsNothing() {
        List<String> chunks = new ArrayList<>();
        boolean stop = parser.feed("{\"data\":[],\"stop\":true}", chunks::add);
        assertThat(stop, is(true));
        assertThat(chunks, is(empty()));
    }

    /**
     * The native side writes exactly {@code {"data": <object or array>, "stop": <boolean>}}
     * ({@code wrap_stream_chunk}; {@code stream_next} never returns a null payload). Anything else is a
     * broken transport, not the end of an answer: it used to end the stream as if the answer were
     * complete, so a broken envelope looked like a short answer.
     */
    @Test
    public void feed_anEnvelopeTheNativeSideDoesNotWrite_isRejectedNamingIt() {
        String[] envelopes = {
            "this is not json",
            "",
            "   ",
            "[1]",
            "true",
            "{\"data\":{\"x\":1}}",
            "{\"data\":{\"x\":1},\"stop\":\"yes\"}",
            "{\"stop\":true}",
            "{\"data\":null,\"stop\":false}",
            "{\"data\":3,\"stop\":false}",
            "{\"data\":{},\"stop\":true} trailing",
        };
        for (String envelope : envelopes) {
            List<String> chunks = new ArrayList<>();
            IllegalArgumentException e =
                    assertThrows(IllegalArgumentException.class, () -> parser.feed(envelope, chunks::add), envelope);
            assertThat(envelope, e.getMessage(), containsString("'" + envelope + "'"));
            assertThat(envelope, chunks, is(empty()));
        }
    }

    @Test
    public void feed_aLongBrokenEnvelopeIsNamedByABoundedExcerpt() {
        String envelope = "x" + String.join("", java.util.Collections.nCopies(200, "y"));
        IllegalArgumentException e =
                assertThrows(IllegalArgumentException.class, () -> parser.feed(envelope, chunk -> {}));
        assertThat(e.getMessage(), containsString("'" + envelope.substring(0, 80) + "...'"));
        assertThat(e.getMessage(), not(containsString(envelope.substring(0, 81))));
    }
}
