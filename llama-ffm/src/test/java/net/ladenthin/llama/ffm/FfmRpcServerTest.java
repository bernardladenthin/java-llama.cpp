// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT
package net.ladenthin.llama.ffm;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.nullValue;

import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Test;

/**
 * {@link FfmRpcServer} over loopback, serving the CPU: no model. The stop-before-bind case itself
 * needs a seam inside the serve call and is pinned in {@code test_c_api.cpp}; this pins the FFM half of
 * the contract around it.
 */
class FfmRpcServerTest {

    /** Upper bound for the waits below; reaching it is a failure, not a slow machine. */
    private static final long AWAIT_MILLIS = 10_000;

    /** Ports are tried from a range, so parallel jobs on one runner do not collide. */
    private static final int ATTEMPTS = 50;

    @Test
    void aStopWhileNoServeRunsDoesNotStopTheNextServeAndAStopEndsIt() throws Exception {
        FfmRpcServer.stop(); // nothing runs: a no-op that must not carry over

        boolean served = false;
        for (int attempt = 0; attempt < ATTEMPTS && !served; attempt++) {
            int port = 42_000 + (int) ((System.nanoTime() / 1000 + attempt * 97L) % 20_000);
            AtomicReference<Throwable> failure = new AtomicReference<>();
            Thread serve = new Thread(
                    () -> {
                        try {
                            FfmRpcServer.serve("127.0.0.1", port, 2, null, "CPU");
                        } catch (Throwable t) {
                            failure.set(t);
                        }
                    },
                    "ffm-rpc-serve-" + port);
            serve.setDaemon(true);
            serve.start();
            long deadline = System.nanoTime() + AWAIT_MILLIS * 1_000_000L;
            while (!FfmRpcServer.isListening() && serve.isAlive() && System.nanoTime() < deadline) {
                serve.join(10);
            }
            if (FfmRpcServer.isListening()) {
                served = true;
                FfmRpcServer.stop();
                serve.join(AWAIT_MILLIS);
                assertThat("serve returned after the stop", serve.isAlive(), is(false));
                assertThat(failure.get(), is(nullValue()));
            } else {
                serve.join(AWAIT_MILLIS); // the port was taken: serve returned at once
            }
        }
        assertThat("listened on a free port of the range", served, is(true));
        assertThat(FfmRpcServer.isListening(), is(false));
    }
}
