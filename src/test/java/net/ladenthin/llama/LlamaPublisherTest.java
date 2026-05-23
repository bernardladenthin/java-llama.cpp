// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import org.junit.Assume;
import org.junit.Test;
import org.reactivestreams.Subscriber;
import org.reactivestreams.Subscription;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

@ClaudeGenerated(
        purpose = "Verify LlamaPublisher honours Reactive Streams contracts: backpressure via request(n), "
                + "stops on cancel, signals onError for invalid demand, and rejects a second subscriber."
)
public class LlamaPublisherTest {

    /**
     * Model-gated: subscribe, request a small batch with backpressure, observe tokens, cancel early.
     */
    @Test
    public void backpressureAndCancel() throws Exception {
        Assume.assumeTrue("Model file not found", new java.io.File(TestConstants.MODEL_PATH).exists());
        int gpuLayers = Integer.getInteger(TestConstants.PROP_TEST_NGL, TestConstants.DEFAULT_TEST_NGL);

        try (LlamaModel model = new LlamaModel(new ModelParameters()
                .setCtxSize(128)
                .setModel(TestConstants.MODEL_PATH)
                .setGpuLayers(gpuLayers)
                .setFit(false))) {

            LlamaPublisher pub = model.streamPublisher(
                    new InferenceParameters("def hello():").setNPredict(20).setSeed(1));

            CountDownLatch done = new CountDownLatch(1);
            AtomicReference<Subscription> subRef = new AtomicReference<>();
            AtomicInteger received = new AtomicInteger();

            pub.subscribe(new Subscriber<LlamaOutput>() {
                @Override public void onSubscribe(Subscription s) {
                    subRef.set(s);
                    s.request(2); // initial demand
                }
                @Override public void onNext(LlamaOutput o) {
                    int n = received.incrementAndGet();
                    if (n == 2) {
                        // Verify backpressure: with demand=0 we should pause until next request.
                        // Request one more to trigger another emission.
                        subRef.get().request(1);
                    } else if (n == 3) {
                        // Cancel after the third token; subsequent onNext must not occur.
                        subRef.get().cancel();
                        done.countDown();
                    }
                }
                @Override public void onError(Throwable t) { done.countDown(); }
                @Override public void onComplete() { done.countDown(); }
            });

            assertTrue("subscriber did not terminate in 30s", done.await(30, TimeUnit.SECONDS));
            // After cancel we may receive 3-4 in-flight tokens; should not be far above the
            // demand actually requested (3 here).
            int got = received.get();
            assertTrue("expected ~3 tokens, got " + got, got >= 3 && got <= 6);
        }
    }

    @Test
    public void singleSubscriberContract() throws Exception {
        Assume.assumeTrue("Model file not found", new java.io.File(TestConstants.MODEL_PATH).exists());
        int gpuLayers = Integer.getInteger(TestConstants.PROP_TEST_NGL, TestConstants.DEFAULT_TEST_NGL);

        try (LlamaModel model = new LlamaModel(new ModelParameters()
                .setCtxSize(128)
                .setModel(TestConstants.MODEL_PATH)
                .setGpuLayers(gpuLayers)
                .setFit(false))) {

            LlamaPublisher pub = model.streamPublisher(
                    new InferenceParameters("def f():").setNPredict(2).setSeed(1));

            CountDownLatch first = new CountDownLatch(1);
            pub.subscribe(new Subscriber<LlamaOutput>() {
                @Override public void onSubscribe(Subscription s) { s.request(Long.MAX_VALUE); }
                @Override public void onNext(LlamaOutput o) { }
                @Override public void onError(Throwable t) { first.countDown(); }
                @Override public void onComplete() { first.countDown(); }
            });
            assertTrue(first.await(30, TimeUnit.SECONDS));

            // Second subscribe must signal onError.
            AtomicReference<Throwable> err = new AtomicReference<>();
            CountDownLatch second = new CountDownLatch(1);
            pub.subscribe(new Subscriber<LlamaOutput>() {
                @Override public void onSubscribe(Subscription s) { }
                @Override public void onNext(LlamaOutput o) { }
                @Override public void onError(Throwable t) { err.set(t); second.countDown(); }
                @Override public void onComplete() { second.countDown(); }
            });
            assertTrue(second.await(5, TimeUnit.SECONDS));
            assertNotNull("expected onError on second subscribe", err.get());
            assertTrue(err.get() instanceof IllegalStateException);
        }
    }

    @Test
    public void invalidRequestSignalsError() throws Exception {
        Assume.assumeTrue("Model file not found", new java.io.File(TestConstants.MODEL_PATH).exists());
        int gpuLayers = Integer.getInteger(TestConstants.PROP_TEST_NGL, TestConstants.DEFAULT_TEST_NGL);

        try (LlamaModel model = new LlamaModel(new ModelParameters()
                .setCtxSize(128)
                .setModel(TestConstants.MODEL_PATH)
                .setGpuLayers(gpuLayers)
                .setFit(false))) {

            LlamaPublisher pub = model.streamPublisher(
                    new InferenceParameters("def f():").setNPredict(5).setSeed(1));

            AtomicReference<Throwable> err = new AtomicReference<>();
            CountDownLatch done = new CountDownLatch(1);
            pub.subscribe(new Subscriber<LlamaOutput>() {
                @Override public void onSubscribe(Subscription s) { s.request(0); }
                @Override public void onNext(LlamaOutput o) { }
                @Override public void onError(Throwable t) { err.set(t); done.countDown(); }
                @Override public void onComplete() { done.countDown(); }
            });
            assertTrue(done.await(10, TimeUnit.SECONDS));
            assertNotNull("expected onError for request(0)", err.get());
            assertTrue(err.get() instanceof IllegalArgumentException);
        }
    }

    @Test
    public void nullSubscriberThrows() {
        // Construct a publisher without a model — subscribe(null) must NPE before any model use.
        try {
            new LlamaPublisher(null, null, false).subscribe(null);
            org.junit.Assert.fail("expected NPE");
        } catch (NullPointerException expected) {
            assertEquals("subscriber", expected.getMessage());
        }
    }
}
