// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.json;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.empty;
import static org.hamcrest.Matchers.hasSize;
import static org.hamcrest.Matchers.is;

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
    public void feed_missingData_emitsNothing() {
        List<String> chunks = new ArrayList<>();
        boolean stop = parser.feed("{\"stop\":true}", chunks::add);
        assertThat(stop, is(true));
        assertThat(chunks, is(empty()));
    }

    @Test
    public void feed_stopDefaultsFalse_whenAbsent() {
        List<String> chunks = new ArrayList<>();
        boolean stop = parser.feed("{\"data\":{\"x\":1}}", chunks::add);
        assertThat(stop, is(false));
        assertThat(chunks, hasSize(1));
    }

    @Test
    public void feed_malformedEnvelope_reportsStop_andEmitsNothing() {
        List<String> chunks = new ArrayList<>();
        boolean stop = parser.feed("this is not json", chunks::add);
        assertThat(stop, is(true));
        assertThat(chunks, is(empty()));
    }

    @Test
    public void feed_nullData_emitsNothing() {
        List<String> chunks = new ArrayList<>();
        boolean stop = parser.feed("{\"data\":null,\"stop\":false}", chunks::add);
        assertThat(stop, is(false));
        assertThat(chunks, is(empty()));
    }
}
