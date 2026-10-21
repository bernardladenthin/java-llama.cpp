// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.contains;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.sameInstance;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;

/**
 * Unit tests for {@link LlamaModelBackend#relay}, the bridge between a {@link ChunkSink} that may throw
 * {@link IOException} and the model's streaming API, which takes a {@link java.util.function.Consumer}.
 * Lambdas stand in for the model call, so no model is loaded.
 */
public class LlamaModelBackendTest {

    @Test
    public void everyChunkReachesTheSinkInOrder() throws IOException {
        List<String> written = new ArrayList<>();
        LlamaModelBackend.relay(
                chunkConsumer -> {
                    chunkConsumer.accept("a");
                    chunkConsumer.accept("b");
                },
                written::add);
        assertThat(written, contains("a", "b"));
    }

    // The server tells a client that went away (DEBUG, nothing written) from a failed completion (WARN,
    // an error event) by the type: the sink's IOException has to come back as itself, not wrapped.
    @Test
    public void anIOExceptionOfTheSinkComesBackAsTheSameInstance() {
        IOException disconnect = new IOException("Broken pipe");
        List<String> afterTheFailure = new ArrayList<>();
        IOException thrown = assertThrows(
                IOException.class,
                () -> LlamaModelBackend.relay(
                        chunkConsumer -> {
                            chunkConsumer.accept("a");
                            afterTheFailure.add("the model call went on after the sink failed");
                        },
                        chunk -> {
                            throw disconnect;
                        }));
        assertThat(thrown, is(sameInstance(disconnect)));
        assertThat("the failure unwinds the model call", afterTheFailure.isEmpty(), is(true));
    }

    @Test
    public void anUncheckedIOExceptionTheSinkDidNotCauseIsRethrownUnchanged() {
        UncheckedIOException foreign = new UncheckedIOException(new IOException("not the client"));
        UncheckedIOException thrown = assertThrows(
                UncheckedIOException.class,
                () -> LlamaModelBackend.relay(
                        chunkConsumer -> {
                            throw foreign;
                        },
                        chunk -> {}));
        assertThat(thrown, is(sameInstance(foreign)));
    }

    @Test
    public void aRuntimeExceptionOfTheModelCallPropagates() {
        IllegalStateException failure = new IllegalStateException("inference failed");
        IllegalStateException thrown = assertThrows(
                IllegalStateException.class,
                () -> LlamaModelBackend.relay(
                        chunkConsumer -> {
                            chunkConsumer.accept("a");
                            throw failure;
                        },
                        chunk -> {}));
        assertThat(thrown, is(sameInstance(failure)));
    }
}
