// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.contains;
import static org.hamcrest.Matchers.is;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

/**
 * Model-free, library-free unit tests for {@link NativeServer}'s pure-Java surface: it must
 * construct without any native work (libjllama is loaded lazily in {@link NativeServer#start()},
 * not in a static initializer), best-effort parse host/port from the forwarded arguments, report
 * itself not running before {@code start()}, and be a safe no-op {@link AutoCloseable} when never
 * started. Actually starting the native server is exercised by CI / manual runs with a real model.
 */
public class NativeServerSmokeTest {

    @Test
    public void parsesHostAndPortFromArgs() {
        NativeServer server = new NativeServer("-m", "m.gguf", "--host", "0.0.0.0", "--port", "1234");
        assertThat(server.getHost(), is("0.0.0.0"));
        assertThat(server.getPort(), is(1234));
        assertThat(server.isRunning(), is(false));
    }

    @Test
    public void commaSeparatedHostsAreSplitAndTrimmedAndTheFirstIsTheHost() {
        // since llama.cpp b11104, --host takes a comma-separated list and binds all of them
        NativeServer server = new NativeServer("-m", "m.gguf", "--host", "127.0.0.1, ::1 ,,/tmp/l.sock");
        assertThat(server.getHosts(), contains("127.0.0.1", "::1", "/tmp/l.sock"));
        assertThat(server.getHost(), is("127.0.0.1"));
    }

    @Test
    public void hostListOfOnlySeparatorsFallsBackToDefault() {
        NativeServer server = new NativeServer("-m", "m.gguf", "--host", " , ");
        assertThat(server.getHosts(), contains("127.0.0.1"));
        assertThat(server.getHost(), is("127.0.0.1"));
    }

    @Test
    public void shortPortFlagParsed() {
        NativeServer server = new NativeServer("-m", "m.gguf", "-p", "9099");
        assertThat(server.getPort(), is(9099));
    }

    @Test
    public void defaultsWhenFlagsAbsent() {
        NativeServer server = new NativeServer("-m", "m.gguf");
        assertThat(server.getHost(), is("127.0.0.1"));
        assertThat(server.getHosts(), contains("127.0.0.1"));
        assertThat(server.getPort(), is(9931));
    }

    @Test
    public void nonIntegerPortFallsBackToDefault() {
        NativeServer server = new NativeServer("-m", "m.gguf", "--port", "abc");
        assertThat(server.getPort(), is(9931));
    }

    @Test
    public void closeBeforeStartIsSafeNoOpViaTryWithResources() {
        try (NativeServer server = new NativeServer("-m", "m.gguf")) {
            assertThat(server.isRunning(), is(false));
        }
    }

    @Test
    public void nullArgsRejected() {
        assertThrows(NullPointerException.class, () -> new NativeServer((String[]) null));
    }

    @Test
    public void nullArgElementRejected() {
        assertThrows(NullPointerException.class, () -> new NativeServer("-m", null));
    }

    @Test
    public void attachMode_nullModelRejected() {
        assertThrows(
                NullPointerException.class,
                () -> new NativeServer((net.ladenthin.llama.LlamaModel) null, "--port", "8080"));
    }
}
