// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.is;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.NoSuchElementException;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import net.ladenthin.llama.parameters.InferenceParameters;
import org.junit.jupiter.api.Test;

@ClaudeGenerated(
        purpose = "Pin that a cancel() from another thread ends a LlamaIterator for good, also while "
                + "next() is waiting for a result, and that a natural stop releases the task once. "
                + "Driven through the TaskSource seam, so no model is needed.")
public class LlamaIteratorCancelTest {

    private static final int TASK_ID = 3;

    private static final long WAIT_SECONDS = 5L;

    private static final String CHUNK = "{\"content\":\"x\",\"stop\":false}";

    private static final String LAST = "{\"content\":\"y\",\"stop\":true,\"stop_type\":\"eos\"}";

    /** A task source whose receive can be held until the test releases it. */
    static class FakeSource implements LlamaIterator.TaskSource {
        final CountDownLatch receiveEntered = new CountDownLatch(1);
        final CountDownLatch releaseReceive = new CountDownLatch(1);
        final AtomicInteger releases = new AtomicInteger();
        final AtomicInteger cancels = new AtomicInteger();
        volatile boolean holdReceive;
        volatile String[] results = {CHUNK};
        private final AtomicInteger received = new AtomicInteger();

        @Override
        public int requestCompletion(String json) {
            return TASK_ID;
        }

        @Override
        public int requestChatCompletion(String json) {
            return TASK_ID;
        }

        @Override
        public String receiveCompletionJson(int taskId) {
            receiveEntered.countDown();
            if (holdReceive) {
                try {
                    releaseReceive.await(WAIT_SECONDS, TimeUnit.SECONDS);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            }
            String[] localResults = results;
            return localResults[Math.min(received.getAndIncrement(), localResults.length - 1)];
        }

        @Override
        public void releaseTask(int taskId) {
            releases.incrementAndGet();
        }

        @Override
        public void cancelCompletion(int taskId) {
            cancels.incrementAndGet();
        }
    }

    @Test
    public void cancel_whileNextWaitsForAResultOnAnotherThread_iteratorStaysFinished() throws Exception {
        FakeSource source = new FakeSource();
        source.holdReceive = true;
        LlamaIterator iterator = new LlamaIterator(source, new InferenceParameters("hello"), false);
        Thread consumer = new Thread(iterator::next, "consumer");
        consumer.start();
        assertThat(source.receiveEntered.await(WAIT_SECONDS, TimeUnit.SECONDS), is(true));

        iterator.cancel();
        source.releaseReceive.countDown();
        consumer.join(TimeUnit.SECONDS.toMillis(WAIT_SECONDS));

        assertThat("a result that arrived after cancel() revived the iterator", iterator.hasNext(), is(false));
        assertThrows(NoSuchElementException.class, iterator::next);
        assertThat(source.cancels.get(), is(1));
    }

    @Test
    public void next_untilTheStopResult_releasesTheTaskOnceAndEnds() {
        FakeSource source = new FakeSource();
        source.results = new String[] {CHUNK, LAST};
        LlamaIterator iterator = new LlamaIterator(source, new InferenceParameters("hello"), true);

        assertThat(iterator.next().stop, is(false));
        assertThat(iterator.hasNext(), is(true));
        assertThat(iterator.next().stop, is(true));

        assertThat(iterator.hasNext(), is(false));
        assertThat(source.releases.get(), is(1));
        iterator.close();
        assertThat("close() after the natural stop must not cancel", source.cancels.get(), is(0));
    }

    @Test
    public void close_beforeTheStopResult_cancelsTheTaskOnce() {
        FakeSource source = new FakeSource();
        LlamaIterator iterator = new LlamaIterator(source, new InferenceParameters("hello"), false);
        iterator.next();

        iterator.close();
        iterator.close();

        assertThat(source.cancels.get(), is(1));
        assertThat(iterator.hasNext(), is(false));
    }
}
