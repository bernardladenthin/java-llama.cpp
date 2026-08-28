// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.File;
import java.util.concurrent.TimeUnit;
import net.ladenthin.llama.parameters.InferenceParameters;
import net.ladenthin.llama.parameters.ModelParameters;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

@ClaudeGenerated(
        purpose = "Regression guard for the idle-sleep wake path: with --sleep-idle-seconds set, a "
                + "request issued after the server has gone to sleep must still be serviced. Before "
                + "the wake_and_post() fix the model became permanently unusable after the first "
                + "idle period, and nothing in the suite could observe it.")
public class IdleSleepWakeIntegrationTest {

    /** Idle window before the server sleeps. Kept short so the test does not dominate the suite. */
    private static final int SLEEP_IDLE_SECONDS = 1;

    /** Comfortably longer than {@link #SLEEP_IDLE_SECONDS} so the server is certainly asleep. */
    private static final long IDLE_WAIT_MILLIS = 3_000L;

    private static LlamaModel model;

    @BeforeAll
    public static void loadModel() {
        String modelPath = TestConstants.resolveModelProperty(
                "net.ladenthin.llama.model.path", TestConstants.REASONING_MODEL_PATH);
        Assumptions.assumeTrue(new File(modelPath).exists(), "Model missing: " + modelPath);
        int gpuLayers = Integer.getInteger(TestConstants.PROP_TEST_NGL, TestConstants.DEFAULT_TEST_NGL);
        model = new LlamaModel(new ModelParameters()
                .setModel(modelPath)
                .setCtxSize(512)
                .setGpuLayers(gpuLayers)
                .setSleepIdleSeconds(SLEEP_IDLE_SECONDS));
    }

    @AfterAll
    public static void closeModel() {
        if (model != null) {
            model.close();
        }
    }

    /**
     * A request after the idle window must still complete.
     *
     * <p>This is the one test that can observe the defect fixed by {@code wake_and_post()} in
     * {@code jllama.cpp}. Upstream's {@code server_queue::post()} only calls
     * {@code condition_tasks.notify_one()}; the sleeping wait predicate tests
     * {@code (!running || req_stop_sleeping)}, and only {@code wait_until_no_sleep()} sets
     * {@code req_stop_sleeping}. Upstream wakes the loop for callers that obtain their reader from
     * {@code create_response()}, but this binding uses the CLI-facing {@code get_response_reader()},
     * which does not &mdash; so a task posted while the server slept was queued and never serviced,
     * and the model stayed unusable for the rest of its life.
     *
     * <p>The bounded {@link Timeout} is the assertion that matters: without the fix this method does
     * not fail an assertion, it <em>hangs</em>. The timeout converts that into a test failure. The
     * generation itself is deliberately tiny (a handful of tokens) because what is under test is
     * whether the queue is serviced at all, not what the model says.
     *
     * <p>Model-gated like its neighbours: it needs a real load to have a queue to put to sleep.
     * CI downloads the full model set on every Java job, so it runs on all six platforms.
     */
    @Test
    @Timeout(value = 120, unit = TimeUnit.SECONDS)
    public void requestAfterTheIdleWindowIsStillServiced() throws InterruptedException {
        // First request while certainly awake -- establishes that the fixture works at all, so a
        // failure after the sleep cannot be blamed on the model or the parameters.
        String before = model.complete(new InferenceParameters("Say hi.").withNPredict(4));
        assertNotNull(before, "baseline completion returned null");

        // Let the server cross its idle threshold and actually go to sleep.
        Thread.sleep(IDLE_WAIT_MILLIS);

        // The regression: this used to queue a task the sleeping loop never picked up.
        String after = model.complete(new InferenceParameters("Say hi again.").withNPredict(4));
        assertNotNull(after, "completion after the idle window returned null");

        // And the model must remain usable -- the defect was permanent, not a one-request hiccup.
        String third = model.complete(new InferenceParameters("And once more.").withNPredict(4));
        assertNotNull(third, "completion after waking returned null");
        assertTrue(model.getMetrics() != null, "getMetrics() must still be serviced after waking");
    }
}
