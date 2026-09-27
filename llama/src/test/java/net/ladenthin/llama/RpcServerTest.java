// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.empty;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.not;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.io.IOException;
import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.net.ServerSocket;
import java.net.Socket;
import java.nio.file.Files;
import java.nio.file.Path;
import net.ladenthin.llama.exception.LlamaException;
import net.ladenthin.llama.loader.OSInfo;
import net.ladenthin.llama.parameters.ModelParameters;
import net.ladenthin.llama.value.RpcEndpoint;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * {@link RpcServer}'s lifecycle and the client's failure path, against the real native library but
 * without a model: start and stop on a real port, the one-server-per-process rule, a port that is
 * taken, and a model load that names a server nobody runs. Self-skips when libjllama is not on the
 * classpath (a pure-Java checkout).
 */
@ClaudeGenerated(
        purpose = "Pin the RpcServer lifecycle over JNI (start, stop, restart on the same port, the "
                + "single-instance rule, bind failure) and that an unreachable --rpc server fails a model "
                + "load with a LlamaException naming it instead of aborting the JVM.")
public class RpcServerTest {

    @TempDir
    Path tempDir;

    @BeforeEach
    void requireNativeLibrary() {
        assumeTrue(nativeLibraryOnClasspath(), "libjllama not on classpath — skipping RpcServerTest");
    }

    @Test
    public void startsStopsAndStartsAgainOnTheSamePort() throws IOException {
        int port = freePort();
        try (RpcServer server = RpcServer.startLocal(port)) {
            assertThat(server.isRunning(), is(true));
            assertThat(server.getEndpoint(), is(RpcEndpoint.of("127.0.0.1", port)));
            assertThat(accepts(port), is(true));
            server.close();
            assertThat(server.isRunning(), is(false));
        }
        // the port was released: the same one can be bound again
        try (RpcServer again = RpcServer.startLocal(port)) {
            assertThat(again.isRunning(), is(true));
        }
    }

    @Test
    public void closeIsIdempotent() throws IOException {
        RpcServer server = RpcServer.startLocal(freePort());
        server.close();
        server.close();
        assertThat(server.isRunning(), is(false));
    }

    @Test
    public void onlyOneServerPerProcess() throws IOException {
        try (RpcServer server = RpcServer.startLocal(freePort())) {
            IllegalStateException e = assertThrows(IllegalStateException.class, () -> RpcServer.startLocal(freePort()));
            assertThat(e.getMessage(), containsString("already running"));
            assertThat(server.isRunning(), is(true));
        }
    }

    @Test
    public void aPortThatIsTakenIsALlamaExceptionAndLeavesNoServerBehind() throws IOException {
        try (ServerSocket taken = new ServerSocket(0, 1, InetAddress.getByName("127.0.0.1"))) {
            LlamaException e = assertThrows(LlamaException.class, () -> RpcServer.startLocal(taken.getLocalPort()));
            assertThat(e.getMessage(), containsString("could not listen on 127.0.0.1:" + taken.getLocalPort()));
        }
        // the failed attempt must not have left the single-instance slot occupied
        try (RpcServer server = RpcServer.startLocal(freePort())) {
            assertThat(server.isRunning(), is(true));
        }
    }

    @Test
    public void aTensorCacheDirectoryIsCreated() throws IOException {
        Path cache = tempDir.resolve("rpc").resolve("cache");
        try (RpcServer server = RpcServer.startLocal(freePort(), 1, cache)) {
            assertThat(Files.isDirectory(cache), is(true));
        }
    }

    @Test
    public void invalidArgumentsAreRejectedBeforeAnythingStarts() throws IOException {
        int port = freePort();
        assertThrows(IllegalArgumentException.class, () -> RpcServer.startLocal(port, 0, null));
        assertThrows(IllegalArgumentException.class, () -> RpcServer.startOnNetwork("localhost", port, 1, null));
        assertThrows(IllegalArgumentException.class, () -> RpcServer.startLocal(0));
        // none of those may have taken the single-instance slot
        try (RpcServer server = RpcServer.startLocal(port)) {
            assertThat(server.isRunning(), is(true));
        }
    }

    @Test
    public void servesAtLeastOneDeviceAndNeverAnRpcDevice() {
        assertThat(RpcServer.servedDevices(), is(not(empty())));
        for (String device : RpcServer.servedDevices()) {
            assertThat(device, not(containsString("RPC")));
        }
    }

    @Test
    public void aModelLoadNamingAnUnreachableServerFailsWithItsNameAndTheJvmLives() throws IOException {
        RpcEndpoint nobody = RpcEndpoint.of("127.0.0.1", freePort());
        ModelParameters params =
                new ModelParameters().setModel("does-not-matter.gguf").setRpcServers(nobody);
        LlamaException e = assertThrows(LlamaException.class, () -> new LlamaModel(params).close());
        assertThat(e.getMessage(), containsString(nobody.toString()));
        // and the JVM is still here to run the next line
        assertThat(RpcServer.servedDevices(), is(not(empty())));
    }

    /** A port nothing listens on right now (the usual bind-to-0-and-release race is acceptable here). */
    static int freePort() throws IOException {
        try (ServerSocket socket = new ServerSocket(0, 1, InetAddress.getByName("127.0.0.1"))) {
            return socket.getLocalPort();
        }
    }

    private static boolean accepts(int port) {
        try (Socket socket = new Socket()) {
            socket.connect(new InetSocketAddress("127.0.0.1", port), 2000);
            return true;
        } catch (IOException e) {
            return false;
        }
    }

    static boolean nativeLibraryOnClasspath() {
        String resource = "/net/ladenthin/llama/" + OSInfo.getNativeLibFolderPathForCurrentOS() + "/"
                + System.mapLibraryName("jllama");
        return RpcServerTest.class.getResource(resource) != null;
    }
}
