// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.is;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import net.ladenthin.llama.ClaudeGenerated;
import net.ladenthin.llama.LlamaModel;
import org.junit.jupiter.api.Test;

@ClaudeGenerated(
        purpose = "Pin that NativeServer serializes start(), isRunning() and close() over its native "
                + "handle: close() must neither free the handle while isRunning() is inside the native "
                + "call nor return early while start() is still starting the server. Driven through the "
                + "Backend seam with a fake, so a made-up handle never reaches libjllama.")
public class NativeServerLifecycleTest {

    private static final long HANDLE = 42L;

    private static final long WAIT_SECONDS = 5L;

    /** A backend whose start and isRunning can be held inside the "native" call. */
    static class FakeBackend implements NativeServer.Backend {
        final CountDownLatch startEntered = new CountDownLatch(1);
        final CountDownLatch releaseStart = new CountDownLatch(1);
        final CountDownLatch isRunningEntered = new CountDownLatch(1);
        final CountDownLatch releaseIsRunning = new CountDownLatch(1);
        final AtomicInteger stops = new AtomicInteger();
        final AtomicInteger insideIsRunning = new AtomicInteger();
        volatile boolean holdStart;
        volatile boolean holdIsRunning;
        volatile boolean stoppedDuringIsRunning;

        @Override
        public long start(String[] args) {
            startEntered.countDown();
            if (holdStart) {
                await(releaseStart);
            }
            return HANDLE;
        }

        @Override
        public long startAttached(LlamaModel model, String[] args) {
            throw new UnsupportedOperationException("attach mode is not used here");
        }

        @Override
        public void stop(long handle) {
            if (insideIsRunning.get() > 0) {
                stoppedDuringIsRunning = true;
            }
            stops.incrementAndGet();
        }

        @Override
        public boolean isRunning(long handle) {
            insideIsRunning.incrementAndGet();
            try {
                isRunningEntered.countDown();
                if (holdIsRunning) {
                    await(releaseIsRunning);
                }
                return true;
            } finally {
                insideIsRunning.decrementAndGet();
            }
        }

        private static void await(CountDownLatch latch) {
            try {
                latch.await(WAIT_SECONDS, TimeUnit.SECONDS);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }
    }

    @Test
    public void close_whileIsRunningIsInsideTheNativeCall_stopsOnlyAfterItReturned() throws Exception {
        FakeBackend backend = new FakeBackend();
        backend.holdIsRunning = true;
        NativeServer server = new NativeServer(backend).start();
        try {
            Thread prober = new Thread(server::isRunning, "prober");
            prober.start();
            assertThat(backend.isRunningEntered.await(WAIT_SECONDS, TimeUnit.SECONDS), is(true));

            Thread closer = new Thread(server::close, "closer");
            closer.start();
            awaitBlockedOrDone(closer);
            backend.releaseIsRunning.countDown();
            prober.join(TimeUnit.SECONDS.toMillis(WAIT_SECONDS));
            closer.join(TimeUnit.SECONDS.toMillis(WAIT_SECONDS));

            assertThat(
                    "the handle was stopped while isRunning() still used it",
                    backend.stoppedDuringIsRunning,
                    is(false));
            assertThat(backend.stops.get(), is(1));
        } finally {
            backend.releaseIsRunning.countDown();
            server.close();
        }
        assertSlotIsFree();
    }

    @Test
    public void close_whileStartIsStillStarting_stopsTheServerOnceItStarted() throws Exception {
        FakeBackend backend = new FakeBackend();
        backend.holdStart = true;
        NativeServer server = new NativeServer(backend);
        Thread starter = new Thread(server::start, "starter");
        starter.start();
        try {
            assertThat(backend.startEntered.await(WAIT_SECONDS, TimeUnit.SECONDS), is(true));

            Thread closer = new Thread(server::close, "closer");
            closer.start();
            awaitBlockedOrDone(closer);
            backend.releaseStart.countDown();
            starter.join(TimeUnit.SECONDS.toMillis(WAIT_SECONDS));
            closer.join(TimeUnit.SECONDS.toMillis(WAIT_SECONDS));

            assertThat(
                    "close() returned while the server was still starting, so it kept running",
                    backend.stops.get(),
                    is(1));
            assertThat(server.isRunning(), is(false));
        } finally {
            backend.releaseStart.countDown();
            starter.join(TimeUnit.SECONDS.toMillis(WAIT_SECONDS));
            server.close();
        }
        assertSlotIsFree();
    }

    @Test
    public void close_calledTwice_stopsTheHandleOnce() {
        FakeBackend backend = new FakeBackend();
        NativeServer server = new NativeServer(backend).start();
        server.close();
        server.close();
        assertThat(backend.stops.get(), is(1));
        assertSlotIsFree();
    }

    /** Waits until the thread is blocked on a monitor or has finished, whichever comes first. */
    private static void awaitBlockedOrDone(Thread thread) throws InterruptedException {
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(WAIT_SECONDS);
        while (thread.isAlive() && thread.getState() != Thread.State.BLOCKED && System.nanoTime() < deadline) {
            thread.join(10);
        }
    }

    /** The process-wide single-instance slot must be free again: a fresh server starts and stops. */
    private static void assertSlotIsFree() {
        FakeBackend probe = new FakeBackend();
        NativeServer server = new NativeServer(probe).start();
        server.close();
        assertThat(probe.stops.get(), is(1));
    }
}
