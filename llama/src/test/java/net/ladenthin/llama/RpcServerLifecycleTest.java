// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.empty;
import static org.hamcrest.Matchers.is;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import net.ladenthin.llama.exception.LlamaException;
import net.ladenthin.llama.value.RpcEndpoint;
import org.jspecify.annotations.Nullable;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * {@link RpcServer}'s lifecycle over a fake {@link RpcServer.Backend}: no native library, so this
 * runs everywhere, including the analysis builds that have no {@code libjllama}. The same paths
 * against the real server are in {@link RpcServerTest}.
 */
@ClaudeGenerated(
        purpose = "Pin RpcServer's Java-side lifecycle without native code: listening, bind failure, "
                + "a failing serve, the start timeout, close/idempotence, the single-instance slot being "
                + "released on every failure, the cache directory and the served-device list.")
public class RpcServerLifecycleTest {

    private static final RpcEndpoint ENDPOINT = RpcEndpoint.of("127.0.0.1", 50052);

    private static final List<String> NO_DEVICES = Collections.emptyList();

    @TempDir
    Path tempDir;

    /** serve() blocks until stop(), reporting listening in between -- a server that works. */
    static class WorkingBackend implements RpcServer.Backend {
        final CountDownLatch stopped = new CountDownLatch(1);
        final AtomicInteger stops = new AtomicInteger();
        volatile boolean listening;
        volatile @Nullable String cacheDir;
        volatile int threads;
        volatile @Nullable String served;
        volatile @Nullable String asked;

        @Override
        public void serve(String host, int port, int threads, @Nullable String cacheDir, String devices) {
            this.threads = threads;
            this.cacheDir = cacheDir;
            this.served = devices;
            listening = true;
            try {
                stopped.await();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            listening = false;
        }

        @Override
        public void stop() {
            stops.incrementAndGet();
            stopped.countDown();
        }

        @Override
        public boolean listening() {
            return listening;
        }

        @Override
        public String @Nullable [] devices(String devices) {
            asked = devices;
            return devices.isEmpty() ? new String[] {"CPU"} : devices.split(",");
        }
    }

    /** serve() returns at once without listening -- the port could not be bound. */
    static class UnboundBackend extends WorkingBackend {
        @Override
        public void serve(String host, int port, int threads, @Nullable String cacheDir, String devices) {}
    }

    @Test
    public void startsListeningAndStopsOnClose() throws Exception {
        WorkingBackend backend = new WorkingBackend();
        RpcServer server = RpcServer.start(ENDPOINT, 3, null, NO_DEVICES, backend, 5_000);
        assertThat(server.isRunning(), is(true));
        assertThat(server.getEndpoint(), is(ENDPOINT));
        assertThat(backend.threads, is(3));
        assertThat(backend.cacheDir, is((String) null));
        assertThat("no names means the default choice", backend.served, is(""));
        assertThat(server.getDevices(), is(Arrays.asList("CPU")));
        assertThat(server.toString(), is("RpcServer[127.0.0.1:50052, running]"));

        server.close();
        assertThat(server.isRunning(), is(false));
        assertThat(server.toString(), is("RpcServer[127.0.0.1:50052, stopped]"));
        assertThat(backend.stops.get(), is(1));
        server.close();
        assertThat("close is idempotent", backend.stops.get(), is(1));
        server.awaitTermination();
    }

    @Test
    public void onlyOneServerAtATimeAndTheSlotIsReleasedOnClose() {
        try (RpcServer first = RpcServer.start(ENDPOINT, 1, null, NO_DEVICES, new WorkingBackend(), 5_000)) {
            IllegalStateException e = assertThrows(
                    IllegalStateException.class,
                    () -> RpcServer.start(ENDPOINT, 1, null, NO_DEVICES, new WorkingBackend(), 5_000));
            assertThat(e.getMessage(), containsString("already running"));
            assertThat(e.getMessage(), containsString(ENDPOINT.toString()));
        }
        try (RpcServer second = RpcServer.start(ENDPOINT, 1, null, NO_DEVICES, new WorkingBackend(), 5_000)) {
            assertThat(second.isRunning(), is(true));
        }
    }

    @Test
    public void aSocketThatCannotBeBoundIsALlamaExceptionAndReleasesTheSlot() {
        LlamaException e = assertThrows(
                LlamaException.class,
                () -> RpcServer.start(ENDPOINT, 1, null, NO_DEVICES, new UnboundBackend(), 5_000));
        assertThat(e.getMessage(), containsString("could not listen on 127.0.0.1:50052"));
        assertThat(e.getMessage(), containsString("already in use"));
        assertSlotIsFree();
    }

    @Test
    public void aServeThatThrowsReportsItsMessage() {
        RpcServer.Backend throwing = new UnboundBackend() {
            @Override
            public void serve(String host, int port, int threads, @Nullable String cacheDir, String devices) {
                throw new LlamaException("no device to serve over RPC");
            }
        };
        LlamaException e = assertThrows(
                LlamaException.class, () -> RpcServer.start(ENDPOINT, 1, null, NO_DEVICES, throwing, 5_000));
        assertThat(e.getMessage(), containsString("no device to serve over RPC"));
        assertSlotIsFree();
    }

    @Test
    public void aServerThatNeverListensIsStoppedAfterTheTimeout() {
        // serve() blocks without ever listening; the start must give up, stop it, and not leak it
        WorkingBackend hanging = new WorkingBackend() {
            @Override
            public boolean listening() {
                return false;
            }
        };
        LlamaException e =
                assertThrows(LlamaException.class, () -> RpcServer.start(ENDPOINT, 1, null, NO_DEVICES, hanging, 200));
        assertThat(e.getMessage(), containsString("could not listen"));
        assertThat("the hanging serve was stopped", hanging.stops.get(), is(1));
        assertSlotIsFree();
    }

    @Test
    public void anInterruptedStartGivesUpAndKeepsTheInterruptFlag() {
        WorkingBackend hanging = new WorkingBackend() {
            @Override
            public boolean listening() {
                return false;
            }
        };
        Thread.currentThread().interrupt();
        try {
            assertThrows(LlamaException.class, () -> RpcServer.start(ENDPOINT, 1, null, NO_DEVICES, hanging, 60_000));
            assertThat(Thread.currentThread().isInterrupted(), is(true));
        } finally {
            Thread.interrupted();
        }
        assertThat(hanging.stops.get(), is(1));
        assertSlotIsFree();
    }

    @Test
    public void theCacheDirectoryIsCreatedAndPassedOn() throws Exception {
        Path cache = tempDir.resolve("a").resolve("b");
        WorkingBackend backend = new WorkingBackend();
        try (RpcServer server = RpcServer.start(ENDPOINT, 1, cache, NO_DEVICES, backend, 5_000)) {
            assertThat(Files.isDirectory(cache), is(true));
            assertThat(backend.cacheDir, is(cache.toString()));
        }
    }

    @Test
    public void aCacheDirectoryThatCannotBeCreatedFailsBeforeStartingAndReleasesTheSlot() throws IOException {
        Path file = Files.createFile(tempDir.resolve("not-a-directory"));
        UncheckedIOException e = assertThrows(
                UncheckedIOException.class,
                () -> RpcServer.start(ENDPOINT, 1, file.resolve("cache"), NO_DEVICES, new WorkingBackend(), 5_000));
        assertThat(e.getMessage(), containsString("cannot create the RPC tensor cache directory"));
        assertSlotIsFree();
    }

    @Test
    public void lessThanOneThreadIsRejectedBeforeAnythingStarts() {
        IllegalArgumentException e = assertThrows(
                IllegalArgumentException.class,
                () -> RpcServer.start(ENDPOINT, 0, null, NO_DEVICES, new WorkingBackend(), 5_000));
        assertThat(e.getMessage(), containsString("threads must be at least 1, was 0"));
        assertSlotIsFree();
    }

    @Test
    public void servedDevicesWrapsTheBackendsList() {
        assertThat(RpcServer.servedDevices(new WorkingBackend(), ""), is(Arrays.asList("CPU")));
        RpcServer.Backend none = new WorkingBackend() {
            @Override
            public String @Nullable [] devices(String devices) {
                return null;
            }
        };
        assertThat(RpcServer.servedDevices(none, ""), is(empty()));
        assertThrows(
                UnsupportedOperationException.class,
                () -> RpcServer.servedDevices(new WorkingBackend(), "").add("x"));
    }

    @Test
    public void awaitTerminationReturnsOnceClosed() throws Exception {
        RpcServer server = RpcServer.start(ENDPOINT, 1, null, NO_DEVICES, new WorkingBackend(), 5_000);
        Thread closer = new Thread(server::close);
        closer.start();
        server.awaitTermination();
        closer.join(TimeUnit.SECONDS.toMillis(10));
        assertThat(server.isRunning(), is(false));
    }

    @Test
    public void namedDevicesAreResolvedBeforeStartingAndServed() {
        WorkingBackend backend = new WorkingBackend();
        try (RpcServer server = RpcServer.start(ENDPOINT, 1, null, Arrays.asList(" CPU ", "CUDA0"), backend, 5_000)) {
            assertThat("names are trimmed and joined", backend.asked, is("CPU,CUDA0"));
            assertThat(backend.served, is("CPU,CUDA0"));
            assertThat(server.getDevices(), is(Arrays.asList("CPU", "CUDA0")));
        }
    }

    @Test
    public void anUnknownDeviceFailsTheStartAndReleasesTheSlot() {
        WorkingBackend unknown = new WorkingBackend() {
            @Override
            public String @Nullable [] devices(String devices) {
                throw new LlamaException("unknown device '" + devices + "' to serve over RPC");
            }
        };
        LlamaException e = assertThrows(
                LlamaException.class, () -> RpcServer.start(ENDPOINT, 1, null, Arrays.asList("NOPE"), unknown, 5_000));
        assertThat(e.getMessage(), containsString("unknown device 'NOPE'"));
        assertThat("never started", unknown.stops.get(), is(0));
        assertSlotIsFree();
    }

    @Test
    public void aDeviceNameThatWouldSplitDifferentlyIsRejected() {
        for (String bad : new String[] {"", " ", "CPU,CUDA0"}) {
            IllegalArgumentException e = assertThrows(
                    IllegalArgumentException.class,
                    () -> RpcServer.start(ENDPOINT, 1, null, Arrays.asList("CPU", bad), new WorkingBackend(), 5_000),
                    bad);
            assertThat(e.getMessage(), containsString("invalid device name '" + bad + "'"));
        }
        assertThat(RpcServer.joinDevices(NO_DEVICES), is(""));
        assertThat(RpcServer.joinDevices(Arrays.asList("A")), is("A"));
        assertSlotIsFree();
    }

    private static void assertSlotIsFree() {
        try (RpcServer probe = RpcServer.start(ENDPOINT, 1, null, NO_DEVICES, new WorkingBackend(), 5_000)) {
            assertThat(probe.isRunning(), is(true));
        }
    }
}
