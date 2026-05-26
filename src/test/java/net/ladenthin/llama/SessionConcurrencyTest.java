// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import java.io.File;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

/**
 * Per-Session thread-safety follow-up to PR #188 (§2.6 of the
 * llama-stack-client-kotlin investigation). Verifies that:
 * <ul>
 *   <li>concurrent {@link Session#send(String)} calls from multiple threads
 *       produce a strictly alternating user/assistant transcript</li>
 *   <li>{@link Session#stream(String)} sets a "streaming in progress" guard
 *       that causes {@link Session#send(String)} / a second
 *       {@link Session#stream(String)} / {@link Session#save(String)} /
 *       {@link Session#restore(String)} to throw
 *       {@link IllegalStateException} until
 *       {@link Session#commitStreamedReply(String)} clears it</li>
 *   <li>{@link Session#commitStreamedReply(String)} called without a prior
 *       {@link Session#stream(String)} throws</li>
 * </ul>
 * Self-skips when the model GGUF is absent, matching the pattern in
 * {@link ChatScenarioTest}.
 */
@ClaudeGenerated(
        purpose = "Per-Session thread-safety follow-up: serialized send(), stream-in-progress guard, " +
                  "commit-without-stream guard."
)
public class SessionConcurrencyTest {

    private static final int N_PREDICT = 2;
    private static LlamaModel model;

    @BeforeClass
    public static void setup() {
        Assumptions.assumeTrue(new File(TestConstants.MODEL_PATH).exists(), "Model file not found, skipping SessionConcurrencyTest");
        int gpuLayers = Integer.getInteger(TestConstants.PROP_TEST_NGL, TestConstants.DEFAULT_TEST_NGL);
        model = new LlamaModel(
                new ModelParameters()
                        .setCtxSize(4096)
                        .setModel(TestConstants.MODEL_PATH)
                        .setGpuLayers(gpuLayers)
                        .setFit(false)
        );
    }

    @AfterClass
    public static void tearDown() {
        if (model != null) {
            model.close();
        }
    }

    /**
     * Spawn N threads, each calling {@link Session#send(String)} M times concurrently.
     * Every send must complete; the resulting transcript must have exactly
     * {@code 2 * N * M} entries (system message excluded) with strict
     * user/assistant alternation.
     */
    // 5 min budget: each send() resends the accumulating transcript, and the
    // synchronized(lock) in Session serializes the 2*2 = 4 calls. On Linux
    // x86_64 this typically completes in ~30 s; on the slowest CI runner
    // observed (macOS-15 no-metal CPU-only, ~500 ms / token eval) the 4 calls
    // sit around ~150 s. The 5 min ceiling leaves multiplicative headroom for
    // future Apple Silicon CPU-fallback regressions without flaking. Bump
    // callsPerThread or threads up only if the bug it is guarding regresses;
    // do not push it past what the slowest CI runner can finish in <half the
    // @Test budget.
    @Test(timeout = 300_000)
    public void testConcurrentSendProducesAlternatingTranscript() throws Exception {
        final int threads = 2;
        final int callsPerThread = 2;
        try (Session session = new Session(model, 0, null,
                p -> p.setNPredict(N_PREDICT).setTemperature(0.0f))) {

            ExecutorService pool = Executors.newFixedThreadPool(threads);
            CountDownLatch start = new CountDownLatch(1);
            CountDownLatch done = new CountDownLatch(threads);
            AtomicReference<Throwable> failure = new AtomicReference<>();

            for (int t = 0; t < threads; t++) {
                final int tid = t;
                pool.submit(() -> {
                    try {
                        start.await();
                        for (int i = 0; i < callsPerThread; i++) {
                            String reply = session.send("hi from t" + tid + " call " + i);
                            assertNotNull(reply);
                        }
                    } catch (Throwable e) {
                        failure.compareAndSet(null, e);
                    } finally {
                        done.countDown();
                    }
                });
            }

            start.countDown();
            assertTrue(done.await(270, TimeUnit.SECONDS), "threads did not finish within timeout");
            pool.shutdown();

            if (failure.get() != null) {
                throw new AssertionError("worker thread failed", failure.get());
            }

            List<ChatMessage> messages = session.getMessages();
            assertEquals(2 * threads * callsPerThread, messages.size(), "transcript must contain 2 entries per send()");
            for (int i = 0; i < messages.size(); i++) {
                String expectedRole = (i % 2 == 0) ? "user" : "assistant";
                assertEquals(expectedRole, messages.get(i).getRole(), "role mismatch at index " + i);
            }
        }
    }

    /**
     * After {@link Session#stream(String)}, calls to {@link Session#send(String)},
     * a second {@link Session#stream(String)}, {@link Session#save(String)}, and
     * {@link Session#restore(String)} must throw {@link IllegalStateException}.
     * {@link Session#commitStreamedReply(String)} clears the guard and the next
     * {@link Session#send(String)} succeeds.
     */
    @Test(timeout = 120_000)
    public void testStreamGuardBlocksOtherOperationsUntilCommit() throws Exception {
        try (Session session = new Session(model, 1, null,
                p -> p.setNPredict(N_PREDICT).setTemperature(0.0f))) {

            try (LlamaIterable stream = session.stream("hi")) {
                int before = session.getMessages().size();

                try {
                    session.send("racing send");
                    fail("send() during stream must throw");
                } catch (IllegalStateException expected) {
                    // ok
                }
                try {
                    session.stream("racing stream");
                    fail("second stream() must throw");
                } catch (IllegalStateException expected) {
                    // ok
                }
                try {
                    session.save("/tmp/should-not-be-written");
                    fail("save() during stream must throw");
                } catch (IllegalStateException expected) {
                    // ok
                }
                try {
                    session.restore("/tmp/should-not-be-read");
                    fail("restore() during stream must throw");
                } catch (IllegalStateException expected) {
                    // ok
                }

                assertEquals(before, session.getMessages().size(), "transcript must not be mutated by failed calls");

                StringBuilder reply = new StringBuilder();
                for (LlamaOutput out : stream) {
                    reply.append(out.text);
                }
                session.commitStreamedReply(reply.toString());

                List<ChatMessage> messages = session.getMessages();
                assertEquals("assistant", messages.get(messages.size() - 1).getRole(), "last message must be the committed assistant reply");
                assertEquals(reply.toString(),
                        messages.get(messages.size() - 1).getContent());

                String next = session.send("follow-up");
                assertNotNull(next);
            }
        }
    }

    /**
     * Calling {@link Session#commitStreamedReply(String)} without a preceding
     * {@link Session#stream(String)} must throw and must not mutate the transcript.
     */
    @Test(timeout = 30_000)
    public void testCommitStreamedReplyWithoutStreamThrows() {
        try (Session session = new Session(model, 2, null)) {
            int before = session.getMessages().size();
            try {
                session.commitStreamedReply("not allowed");
                fail("commitStreamedReply without stream must throw");
            } catch (IllegalStateException expected) {
                // ok
            }
            assertEquals(before, session.getMessages().size());
        }
    }

    /**
     * Sanity check: a single threaded send-then-send produces strict alternation.
     * Guards against the synchronization wrapper accidentally double-appending or
     * dropping turns.
     */
    @Test(timeout = 60_000)
    public void testSequentialSendsAlternateRoles() {
        try (Session session = new Session(model, 3, null,
                p -> p.setNPredict(N_PREDICT).setTemperature(0.0f))) {
            session.send("a");
            session.send("b");
            List<ChatMessage> messages = session.getMessages();
            assertEquals(4, messages.size());
            assertEquals("user", messages.get(0).getRole());
            assertEquals("assistant", messages.get(1).getRole());
            assertEquals("user", messages.get(2).getRole());
            assertEquals("assistant", messages.get(3).getRole());
            assertFalse(messages.get(1).getContent().isEmpty());
        }
    }
}
