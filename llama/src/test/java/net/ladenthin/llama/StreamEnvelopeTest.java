// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.instanceOf;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.sameInstance;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.ArrayList;
import java.util.List;
import net.ladenthin.llama.exception.LlamaException;
import net.ladenthin.llama.json.ChatStreamChunkParser;
import org.junit.jupiter.api.Test;

/**
 * How {@link LlamaModel}'s streaming loops report the envelopes of {@code receiveChatCompletionChunk}:
 * no model is loaded and no native library, the envelope is a literal. {@link StreamEnvelope} is what
 * keeps this test free of the library: through {@code LlamaModel} itself the first call ran the class
 * initializer, which loads it, and a build without the natives (the SonarQube job) failed here.
 */
public class StreamEnvelopeTest {

    private final ChatStreamChunkParser parser = new ChatStreamChunkParser();

    @Test
    public void aWellFormedEnvelopeReachesTheSink() {
        List<String> chunks = new ArrayList<>();
        boolean stop = StreamEnvelope.feed(parser, "{\"data\":{\"x\":1},\"stop\":true}", chunks::add);
        assertThat(stop, is(true));
        assertThat(chunks.size(), is(1));
    }

    /**
     * A malformed envelope is the native layer's failure: a {@link LlamaException}, which a server
     * answers with 500, not the {@link IllegalArgumentException} the parser throws for its argument,
     * which a server would answer as the request's fault.
     */
    @Test
    public void aMalformedEnvelopeIsALlamaExceptionNamingIt() {
        LlamaException e =
                assertThrows(LlamaException.class, () -> StreamEnvelope.feed(parser, "not json", chunk -> {}));
        assertThat(e.getMessage(), containsString("'not json'"));
        assertThat(e.getCause(), instanceOf(ChatStreamChunkParser.MalformedEnvelopeException.class));
    }

    @Test
    public void anIllegalArgumentExceptionOfTheSinkPassesThroughUnchanged() {
        IllegalArgumentException own = new IllegalArgumentException("the sink's own");
        IllegalArgumentException e = assertThrows(
                IllegalArgumentException.class,
                () -> StreamEnvelope.feed(parser, "{\"data\":{\"x\":1},\"stop\":false}", chunk -> {
                    throw own;
                }));
        assertThat(e, sameInstance(own));
    }
}
