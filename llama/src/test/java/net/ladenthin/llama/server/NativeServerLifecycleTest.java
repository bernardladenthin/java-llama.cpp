// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.is;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import net.ladenthin.llama.ClaudeGenerated;
import net.ladenthin.llama.LlamaModel;
import net.ladenthin.llama.exception.LlamaException;
import org.jspecify.annotations.Nullable;
import org.junit.jupiter.api.Test;

@ClaudeGenerated(
        purpose = "Pin that NativeServer serializes start(), isRunning() and close() over its native "
                + "handle: close() must neither free the handle while isRunning() is inside the native "
                + "call nor return early while start() is still starting the server. Also pin start()'s "
                + "guards: a second start of one instance, a start while another server holds the "
                + "process-wide slot, and a failing start that must free the slot again. Driven through "
                + "the Backend seam with a fake, so a made-up handle never reaches libjllama.")
public class NativeServerLifecycleTest {

    private static final long HANDLE = 42L;

    private static final long WAIT_SECONDS = 5L;

    /** A backend whose start and isRunning can be held inside the "native" call. */
    static class FakeBackend implements NativeServer.Backend {
        final CountDownLatch startEntered = new CountDownLatch(1);
        final CountDownLatch releaseStart = new CountDownLatch(1);
        final CountDownLatch isRunningEntered = new CountDownLatch(1);
        final CountDownLatch releaseIsRunning = new CountDownLatch(1);
        final AtomicInteger starts = new AtomicInteger();
        final AtomicInteger stops = new AtomicInteger();
        final AtomicInteger insideIsRunning = new AtomicInteger();
        volatile boolean holdStart;
        volatile boolean holdIsRunning;
        volatile boolean stoppedDuringIsRunning;
        /** Thrown from start() when set: a RuntimeException or an Error, like the native backend's. */
        volatile @Nullable Throwable startFailure;

        @Override
        public long start(String[] args) {
            starts.incrementAndGet();
            startEntered.countDown();
            final Throwable failure = startFailure;
            if (failure instanceof RuntimeException) {
                throw (RuntimeException) failure;
            }
            if (failure instanceof Error) {
                throw (Error) failure;
            }
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

    @Test
    public void start_whenTheBackendThrowsALlamaException_freesTheSingleInstanceSlot() {
        // The real backend throws a LlamaException for an argv llama-server refuses. The slot is
        // process-wide: left taken, every later NativeServer in the JVM would be refused.
        FakeBackend backend = new FakeBackend();
        backend.startFailure = new LlamaException("boom");
        NativeServer server = new NativeServer(backend);

        LlamaException e = assertThrows(LlamaException.class, server::start);

        assertThat(e.getMessage(), is("boom"));
        assertThat(server.isRunning(), is(false));
        assertThat(backend.stops.get(), is(0));
        assertSlotIsFree();
    }

    @Test
    public void start_whenTheBackendThrowsAnError_freesTheSingleInstanceSlot() {
        // The other arm of the catch: the real backend loads libjllama inside start(), where a missing
        // library is an UnsatisfiedLinkError, not an exception.
        FakeBackend backend = new FakeBackend();
        backend.startFailure = new UnsatisfiedLinkError("libjllama");
        NativeServer server = new NativeServer(backend);

        assertThrows(UnsatisfiedLinkError.class, server::start);

        assertThat(server.isRunning(), is(false));
        assertThat(backend.stops.get(), is(0));
        assertSlotIsFree();
    }

    @Test
    public void start_calledTwiceOnTheSameInstance_throwsIllegalStateException() {
        FakeBackend backend = new FakeBackend();
        NativeServer server = new NativeServer(backend).start();
        try {
            IllegalStateException e = assertThrows(IllegalStateException.class, server::start);

            // Without the handle check the second start still throws, with the other message: the
            // message is what tells the two guards apart.
            assertThat(e.getMessage(), containsString("already started"));
            assertThat(backend.starts.get(), is(1));
        } finally {
            server.close();
        }
        // The refused start did not replace the handle: the first one is stopped, once.
        assertThat(backend.stops.get(), is(1));
        assertSlotIsFree();
    }

    @Test
    public void start_whileAnotherServerRuns_throwsIllegalStateException() {
        FakeBackend firstBackend = new FakeBackend();
        NativeServer first = new NativeServer(firstBackend).start();
        try {
            FakeBackend secondBackend = new FakeBackend();
            NativeServer second = new NativeServer(secondBackend);

            IllegalStateException e = assertThrows(IllegalStateException.class, second::start);

            assertThat(e.getMessage(), containsString("another NativeServer"));
            assertThat(secondBackend.starts.get(), is(0));
            assertThat(second.isRunning(), is(false));
            // The refusal kept the slot taken (the fake's isRunning() is always true, so ask the slot
            // itself): a third server is refused the same way.
            NativeServer third = new NativeServer(new FakeBackend());
            IllegalStateException again = assertThrows(IllegalStateException.class, third::start);
            assertThat(again.getMessage(), containsString("another NativeServer"));
        } finally {
            first.close();
        }
        assertThat(firstBackend.stops.get(), is(1));
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
