// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import org.reactivestreams.Publisher;
import org.reactivestreams.Subscriber;
import org.reactivestreams.Subscription;

import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Reactive Streams {@link Publisher} that emits {@link LlamaOutput} tokens from a
 * llama.cpp streaming completion. Bridges to Reactor / RxJava / Kotlin coroutines via
 * the standard {@code reactive-streams} interface.
 * <p>
 * Each {@link #subscribe(Subscriber)} starts a fresh inference task on a dedicated
 * background thread and honours {@code Subscription.request(n)} for backpressure:
 * the emitter thread only calls {@code iterator.next()} while there is outstanding
 * demand. When the iterator's stop token arrives the publisher calls
 * {@code onComplete}; on cancellation it closes the iterator and stops emitting.
 * </p>
 * <p>
 * Construct via {@link LlamaModel#streamPublisher(InferenceParameters)} or
 * {@link LlamaModel#streamChatPublisher(InferenceParameters)}. The publisher is
 * single-subscriber: a second {@link #subscribe(Subscriber)} call signals
 * {@code onError(IllegalStateException)}.
 * </p>
 */
public final class LlamaPublisher implements Publisher<LlamaOutput> {

    private final LlamaModel model;
    private final InferenceParameters parameters;
    private final boolean chat;
    private final AtomicBoolean subscribed = new AtomicBoolean(false);

    LlamaPublisher(LlamaModel model, InferenceParameters parameters, boolean chat) {
        this.model = model;
        this.parameters = parameters;
        this.chat = chat;
    }

    @Override
    public void subscribe(Subscriber<? super LlamaOutput> subscriber) {
        if (subscriber == null) {
            throw new NullPointerException("subscriber");
        }
        if (!subscribed.compareAndSet(false, true)) {
            EmptySubscription.signalError(subscriber,
                    new IllegalStateException("LlamaPublisher is single-subscriber; already subscribed"));
            return;
        }
        LlamaIterable iterable = chat ? model.generateChat(parameters) : model.generate(parameters);
        LlamaSubscription sub = new LlamaSubscription(iterable, subscriber);
        subscriber.onSubscribe(sub);
        sub.start();
    }

    /** Subscription that honours backpressure and pumps tokens on a dedicated thread. */
    private static final class LlamaSubscription implements Subscription {
        private final LlamaIterable iterable;
        private final Subscriber<? super LlamaOutput> subscriber;
        private final AtomicLong demand = new AtomicLong(0);
        private final AtomicBoolean cancelled = new AtomicBoolean(false);
        private final AtomicBoolean started = new AtomicBoolean(false);
        private final Object monitor = new Object();
        private Thread worker;

        LlamaSubscription(LlamaIterable iterable, Subscriber<? super LlamaOutput> subscriber) {
            this.iterable = iterable;
            this.subscriber = subscriber;
        }

        void start() {
            if (!started.compareAndSet(false, true)) return;
            worker = new Thread(this::pump, "LlamaPublisher-emitter");
            worker.setDaemon(true);
            worker.start();
        }

        @Override
        public void request(long n) {
            if (n <= 0) {
                cancel();
                subscriber.onError(new IllegalArgumentException(
                        "reactive-streams §3.9: request must be > 0, got " + n));
                return;
            }
            // Saturating add
            for (;;) {
                long cur = demand.get();
                long next = cur + n;
                if (next < 0) next = Long.MAX_VALUE;
                if (demand.compareAndSet(cur, next)) break;
            }
            synchronized (monitor) {
                monitor.notifyAll();
            }
        }

        @Override
        public void cancel() {
            if (cancelled.compareAndSet(false, true)) {
                try {
                    iterable.close();
                } catch (Throwable ignored) {
                    // best-effort
                }
                synchronized (monitor) {
                    monitor.notifyAll();
                }
            }
        }

        private void pump() {
            LlamaIterator iterator = iterable.iterator();
            try {
                while (!cancelled.get() && iterator.hasNext()) {
                    // Wait for demand.
                    while (demand.get() == 0 && !cancelled.get()) {
                        synchronized (monitor) {
                            if (demand.get() == 0 && !cancelled.get()) {
                                try {
                                    monitor.wait();
                                } catch (InterruptedException e) {
                                    Thread.currentThread().interrupt();
                                    cancel();
                                    return;
                                }
                            }
                        }
                    }
                    if (cancelled.get()) return;
                    LlamaOutput next = iterator.next();
                    demand.decrementAndGet();
                    subscriber.onNext(next);
                    if (next.stop) {
                        subscriber.onComplete();
                        return;
                    }
                }
                if (!cancelled.get()) {
                    subscriber.onComplete();
                }
            } catch (Throwable t) {
                if (!cancelled.get()) {
                    try {
                        subscriber.onError(t);
                    } catch (Throwable ignored) {
                        // subscriber threw from onError; nothing more we can do
                    }
                }
            } finally {
                try {
                    iterable.close();
                } catch (Throwable ignored) {
                    // best-effort
                }
            }
        }
    }

    /** No-op subscription used to signal onError on rejected subscriptions. */
    private static final class EmptySubscription implements Subscription {
        @Override public void request(long n) { }
        @Override public void cancel() { }

        static void signalError(Subscriber<?> subscriber, Throwable error) {
            subscriber.onSubscribe(new EmptySubscription());
            subscriber.onError(error);
        }
    }
}
