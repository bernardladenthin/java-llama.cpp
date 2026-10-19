// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT
package net.ladenthin.llama.ffm;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.contains;
import static org.hamcrest.Matchers.is;

import java.lang.foreign.MemorySegment;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;

/**
 * The lifecycle locking of {@link FfmNativeServer}, over a fake backend: no library, no model.
 *
 * <p>The native stop frees the handle. Without the lock, {@code close()} freed it while an
 * {@code isRunning()} on another thread was still inside its native call on it (measured with this
 * seam on JDK 25: {@code close()} returned while {@code isRunning()} had not); {@code NativeServer}
 * has serialized the two all along.
 */
class FfmNativeServerTest {

    /** Upper bound for the waits below; reaching it is a failure, not a slow machine. */
    private static final long AWAIT_SECONDS = 10;

    /** How long {@code close()} is given to show that it does not wait. */
    private static final long CLOSE_GRACE_MILLIS = 500;

    @Test
    void closeWaitsForAnIsRunningCallInsideTheNativeCall() throws Exception {
        List<String> events = new CopyOnWriteArrayList<>();
        CountDownLatch insideIsRunning = new CountDownLatch(1);
        CountDownLatch releaseIsRunning = new CountDownLatch(1);
        FfmNativeServer server = new FfmNativeServer(MemorySegment.NULL, new FfmNativeServer.Backend() {
            @Override
            public boolean isRunning(MemorySegment handle) {
                events.add("isRunning entered");
                insideIsRunning.countDown();
                try {
                    releaseIsRunning.await(AWAIT_SECONDS, TimeUnit.SECONDS);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
                events.add("isRunning returned");
                return true;
            }

            @Override
            public void stop(MemorySegment handle) {
                events.add("stop (frees the handle)");
            }
        });

        Thread reader = new Thread(server::isRunning, "isRunning");
        reader.start();
        assertThat(insideIsRunning.await(AWAIT_SECONDS, TimeUnit.SECONDS), is(true));
        Thread closer = new Thread(server::close, "close");
        closer.start();
        closer.join(CLOSE_GRACE_MILLIS);
        boolean closeWaited = closer.isAlive();
        releaseIsRunning.countDown();
        reader.join(TimeUnit.SECONDS.toMillis(AWAIT_SECONDS));
        closer.join(TimeUnit.SECONDS.toMillis(AWAIT_SECONDS));

        assertThat(closeWaited, is(true));
        assertThat(events, contains("isRunning entered", "isRunning returned", "stop (frees the handle)"));
    }

    @Test
    void isRunningAfterCloseDoesNotQueryTheHandle() {
        List<String> events = new CopyOnWriteArrayList<>();
        FfmNativeServer server = new FfmNativeServer(MemorySegment.NULL, recordingBackend(events));

        server.close();

        assertThat(server.isRunning(), is(false));
        assertThat(events, contains("stop"));
    }

    @Test
    void secondCloseIsANoOp() {
        List<String> events = new CopyOnWriteArrayList<>();
        FfmNativeServer server = new FfmNativeServer(MemorySegment.NULL, recordingBackend(events));

        server.close();
        server.close();

        assertThat(events, contains("stop"));
    }

    @Test
    void isRunningAsksTheBackendWhileOpen() {
        List<String> events = new CopyOnWriteArrayList<>();
        FfmNativeServer server = new FfmNativeServer(MemorySegment.NULL, recordingBackend(events));

        assertThat(server.isRunning(), is(true));
        assertThat(events, contains("isRunning"));
    }

    private static FfmNativeServer.Backend recordingBackend(List<String> events) {
        return new FfmNativeServer.Backend() {
            @Override
            public boolean isRunning(MemorySegment handle) {
                events.add("isRunning");
                return true;
            }

            @Override
            public void stop(MemorySegment handle) {
                events.add("stop");
            }
        };
    }
}
