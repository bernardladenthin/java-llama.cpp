// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT
package net.ladenthin.llama;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Function;
import java.util.function.IntConsumer;
import java.util.function.IntFunction;
import java.util.function.ToIntFunction;

/**
 * How the batch methods of {@link LlamaModel} hand several requests to the native scheduler at once.
 *
 * <p>They used to run every request through {@code CompletableFuture.supplyAsync}, i.e. on the common
 * {@code ForkJoinPool}, whose parallelism is {@code availableProcessors() - 1}. So at most that many
 * requests were in flight, whatever {@code setParallel} allowed: measured at b11639, 3 of 8 slots on a
 * 4-core machine. They also shared the pool with whatever else the application ran on it.
 *
 * <p>Kept out of {@link LlamaModel}, whose static initializer loads the native library, so both
 * mechanisms are testable without it.
 */
final class BatchDispatch {

    /**
     * The most threads {@link LlamaModel#chatBatch} runs at once. More requests than the model has
     * slots only wait in the native scheduler, so a larger pool buys nothing but threads; 64 covers
     * every slot count this project has seen configured. Requests beyond it start as threads free up.
     */
    static final int MAX_CHAT_THREADS = 64;

    private BatchDispatch() {}

    /**
     * Submits every request on the calling thread, then receives the results in input order.
     *
     * <p>The native scheduler runs as many of the submitted tasks at once as it has slots. When a
     * submission fails, an {@link Error} included, the tasks already submitted are released before the
     * failure propagates, so no slot keeps working for a result nobody will read. A failed receive, an
     * {@link Error} included, does not stop the others from being received; the first failure
     * propagates after all of them, with the later failures attached to it as suppressed.
     *
     * @param requests the requests in order
     * @param submit   submits one request and returns its task id
     * @param receive  receives the result of one task, blocking until it is complete
     * @param release  releases a submitted task whose result will not be received
     * @param <R>      the request type
     * @param <T>      the result type
     * @return the results in input order
     */
    static <R, T> List<T> submitAllThenReceive(
            Collection<R> requests, ToIntFunction<R> submit, IntFunction<T> receive, IntConsumer release) {
        final List<Integer> taskIds = new ArrayList<Integer>(requests.size());
        try {
            for (R request : requests) {
                taskIds.add(submit.applyAsInt(request));
            }
        } catch (Throwable t) {
            // An Error included (an OutOfMemoryError while a prompt is rendered): it is not handled,
            // only the tasks already submitted are released before it propagates unchanged. The try
            // block throws nothing checked, so the precise rethrow needs no throws clause.
            for (int taskId : taskIds) {
                try {
                    release.accept(taskId);
                } catch (Throwable suppressed) {
                    suppress(t, suppressed);
                }
            }
            throw t;
        }
        final List<T> out = new ArrayList<T>(taskIds.size());
        Throwable failure = null;
        for (int taskId : taskIds) {
            try {
                out.add(receive.apply(taskId));
            } catch (Throwable e) {
                // Keep receiving the rest so no sibling task is left unread (a native reader leak)
                // before propagating the first failure, an Error included; the later ones are attached
                // to it as suppressed.
                if (failure == null) {
                    failure = e;
                } else {
                    suppress(failure, e);
                }
            }
        }
        if (failure != null) {
            throw asRuntimeException(failure);
        }
        return out;
    }

    /**
     * Runs every request on a thread of its own and returns the results in input order.
     *
     * <p>For calls that submit and wait in one native call (the OpenAI chat endpoint), where the
     * submission cannot be separated from the wait. One thread per request up to {@code maxThreads},
     * so that many reach the native scheduler at once and the rest start as threads free up; the
     * threads are daemons and end with the call. A failure, an {@link Error} included, propagates once
     * every request has finished: the first one in input order, with the later ones attached to it as
     * suppressed.
     *
     * @param requests   the requests in order
     * @param maxThreads the most requests run at once, at least 1
     * @param call       runs one request to completion
     * @param <R>      the request type
     * @param <T>      the result type
     * @return the results in input order
     */
    static <R, T> List<T> runEachOnItsOwnThread(Collection<R> requests, int maxThreads, Function<R, T> call) {
        if (maxThreads < 1) {
            throw new IllegalArgumentException("maxThreads must be at least 1, was " + maxThreads);
        }
        if (requests.isEmpty()) {
            return new ArrayList<T>(0);
        }
        final AtomicInteger threadNumber = new AtomicInteger();
        final ExecutorService executor =
                Executors.newFixedThreadPool(Math.min(requests.size(), maxThreads), runnable -> {
                    Thread thread = new Thread(runnable, "jllama-batch-" + threadNumber.incrementAndGet());
                    thread.setDaemon(true);
                    return thread;
                });
        try {
            final List<Future<T>> futures = new ArrayList<Future<T>>(requests.size());
            for (final R request : requests) {
                futures.add(executor.submit(() -> call.apply(request)));
            }
            final List<T> out = new ArrayList<T>(futures.size());
            Throwable failure = null;
            boolean interrupted = false;
            for (Future<T> future : futures) {
                // The native call behind a future cannot be abandoned, so an interrupt does not end
                // the wait; it is reported once every result is in.
                while (true) {
                    try {
                        out.add(future.get());
                        break;
                    } catch (InterruptedException e) {
                        interrupted = true;
                    } catch (ExecutionException e) {
                        Throwable cause = e.getCause() != null ? e.getCause() : e;
                        if (failure == null) {
                            failure = cause;
                        } else {
                            suppress(failure, cause);
                        }
                        break;
                    }
                }
            }
            if (interrupted) {
                Thread.currentThread().interrupt();
            }
            if (failure != null) {
                throw asRuntimeException(failure);
            }
            return out;
        } finally {
            executor.shutdown();
        }
    }

    /**
     * Attaches {@code later} to {@code failure} as suppressed. The same instance can fail two tasks (a
     * shared or cached exception), and {@link Throwable#addSuppressed} refuses the exception itself
     * with an {@link IllegalArgumentException}, which would escape the loop, leave the remaining tasks
     * unread and hide the real failure; that case is skipped. Identity, not {@code equals}, because
     * identity is what {@code addSuppressed} refuses.
     */
    @SuppressWarnings("ReferenceEquality")
    private static void suppress(Throwable failure, Throwable later) {
        if (later != failure) {
            failure.addSuppressed(later);
        }
    }

    /**
     * Returns an unchecked failure as is and throws an {@link Error} as is, so the caller can write
     * {@code throw asRuntimeException(failure)}; anything else (a checked cause, which a
     * {@link Function} or an {@link IntFunction} cannot throw) is wrapped.
     */
    private static RuntimeException asRuntimeException(Throwable cause) {
        if (cause instanceof RuntimeException) {
            return (RuntimeException) cause;
        }
        if (cause instanceof Error) {
            throw (Error) cause;
        }
        return new RuntimeException(cause);
    }
}
