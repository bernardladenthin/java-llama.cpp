// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT
package net.ladenthin.llama;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.contains;
import static org.hamcrest.Matchers.empty;
import static org.hamcrest.Matchers.is;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;

/**
 * The two dispatch mechanisms behind {@link LlamaModel}'s batch methods, without a model.
 *
 * <p>The batch methods used to run each request through {@code CompletableFuture.supplyAsync}, so
 * no more requests were in flight than the common {@code ForkJoinPool} has threads: 3 of 8 slots on a
 * 4-core machine (measured at b11639).
 */
public class BatchDispatchTest {

    /** Upper bound for the waits below; reaching it is a failure, not a slow machine. */
    private static final long AWAIT_SECONDS = 10;

    @Test
    public void submitAllThenReceive_submitsEveryRequestBeforeTheFirstReceive() {
        List<String> events = new ArrayList<>();

        List<String> results = BatchDispatch.submitAllThenReceive(
                Arrays.asList("a", "b", "c"),
                request -> {
                    events.add("submit " + request);
                    return events.size();
                },
                taskId -> {
                    events.add("receive " + taskId);
                    return "result " + taskId;
                },
                taskId -> events.add("release " + taskId));

        assertThat(events, contains("submit a", "submit b", "submit c", "receive 1", "receive 2", "receive 3"));
        assertThat(results, contains("result 1", "result 2", "result 3"));
    }

    @Test
    public void submitAllThenReceive_submitsOnTheCallingThread() {
        List<Thread> submitters = new ArrayList<>();

        BatchDispatch.submitAllThenReceive(
                Arrays.asList("a", "b"),
                request -> {
                    submitters.add(Thread.currentThread());
                    return submitters.size();
                },
                taskId -> "",
                taskId -> {});

        assertThat(submitters, contains(Thread.currentThread(), Thread.currentThread()));
    }

    @Test
    public void submitAllThenReceive_failedSubmission_releasesTheTasksAlreadySubmitted() {
        List<Integer> released = new ArrayList<>();
        IllegalStateException failure = new IllegalStateException("third submission refused");

        IllegalStateException thrown = assertThrows(
                IllegalStateException.class,
                () -> BatchDispatch.submitAllThenReceive(
                        Arrays.asList(1, 2, 3),
                        request -> {
                            if (request == 3) {
                                throw failure;
                            }
                            return request * 10;
                        },
                        taskId -> "",
                        released::add));

        assertThat(thrown, is(failure));
        assertThat(released, contains(10, 20));
    }

    @Test
    public void submitAllThenReceive_failedReceive_stillReceivesTheOthersThenThrowsTheFirstFailure() {
        List<Integer> received = new ArrayList<>();
        IllegalStateException first = new IllegalStateException("first");

        IllegalStateException thrown = assertThrows(
                IllegalStateException.class,
                () -> BatchDispatch.submitAllThenReceive(
                        Arrays.asList(1, 2, 3),
                        request -> request,
                        taskId -> {
                            received.add(taskId);
                            if (taskId == 1) {
                                throw first;
                            }
                            if (taskId == 2) {
                                throw new IllegalStateException("second");
                            }
                            return "";
                        },
                        taskId -> {}));

        assertThat(thrown, is(first));
        assertThat(received, contains(1, 2, 3));
    }

    @Test
    public void runEachOnItsOwnThread_runsMoreRequestsAtOnceThanTheCommonPoolHasThreads() {
        // Every request waits until all of them have started. On the common ForkJoinPool the ones
        // beyond its parallelism never start while the others block, so this times out there.
        int requests = ForkJoinPool.commonPool().getParallelism() + 2;
        CountDownLatch allStarted = new CountDownLatch(requests);
        List<Integer> input = new ArrayList<>();
        for (int i = 0; i < requests; i++) {
            input.add(i);
        }

        List<Boolean> results = BatchDispatch.runEachOnItsOwnThread(input, request -> {
            allStarted.countDown();
            try {
                return allStarted.await(AWAIT_SECONDS, TimeUnit.SECONDS);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                return false;
            }
        });

        assertThat(results, is(Collections.nCopies(requests, true)));
    }

    @Test
    public void runEachOnItsOwnThread_keepsTheInputOrder() {
        List<String> results = BatchDispatch.runEachOnItsOwnThread(Arrays.asList(3, 1, 2), request -> {
            try {
                Thread.sleep(request * 20L);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            return "r" + request;
        });

        assertThat(results, contains("r3", "r1", "r2"));
    }

    @Test
    public void runEachOnItsOwnThread_failure_isThrownAfterTheOthersFinished() {
        IllegalArgumentException failure = new IllegalArgumentException("bad request");
        List<Integer> finished = Collections.synchronizedList(new ArrayList<>());

        IllegalArgumentException thrown = assertThrows(
                IllegalArgumentException.class,
                () -> BatchDispatch.runEachOnItsOwnThread(Arrays.asList(1, 2, 3), request -> {
                    if (request == 1) {
                        throw failure;
                    }
                    finished.add(request);
                    return request;
                }));

        assertThat(thrown, is(failure));
        assertThat(finished.size(), is(2));
    }

    @Test
    public void runEachOnItsOwnThread_noRequests_returnsAnEmptyList() {
        assertThat(
                BatchDispatch.runEachOnItsOwnThread(Collections.<Integer>emptyList(), request -> request), is(empty()));
    }
}
