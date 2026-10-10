// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.is;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.fail;

import java.io.File;
import java.io.IOException;
import java.net.ServerSocket;
import java.util.concurrent.TimeUnit;
import net.ladenthin.llama.ClaudeGenerated;
import net.ladenthin.llama.LlamaModel;
import net.ladenthin.llama.TestConstants;
import net.ladenthin.llama.parameters.InferenceParameters;
import net.ladenthin.llama.parameters.ModelParameters;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

/**
 * Attach mode over a model that goes to idle sleep ({@code --sleep-idle-seconds}): the server is
 * closed, the model sleeps and wakes, and a second server attaches to the same model.
 *
 * <p>The defect this pins: {@code llama_server_attach} ({@code patches/0007}) used to build a
 * {@code server_routes} of its own on the attach worker's stack. Its constructor registers a
 * sleeping-state callback on the model's task queue, which has no unregister, so after
 * {@link NativeServer#close()} the model's next idle sleep ran that callback on a destroyed object
 * (a lock on freed memory, a dangling {@code meta} pointer) -- undefined behaviour, in practice a
 * crashed JVM. Since the fix the attached server serves the model's own {@code server_routes}
 * ({@code jllama_context::routes}), which outlive every attach; the callback that runs on sleep is
 * the model's.
 *
 * <p>The same change decides where the endpoint toggles come from: the routes are the model's, so
 * {@code --metrics} / {@code --props} / {@code --slots} are {@link ModelParameters} now
 * ({@link ModelParameters#enableMetricsEndpoint()}, ...), not attach arguments. The second server
 * proves it by scraping {@code /metrics}, which upstream disables by default.
 */
@ClaudeGenerated(
        purpose = "Pin that closing an attached NativeServer leaves nothing behind in the model's "
                + "sleep callbacks: the model sleeps and wakes after the server is gone, a second "
                + "server attaches, and the endpoint toggles come from the model's parameters.")
public class NativeServerAttachSleepIntegrationTest extends OpenAiServerTestSupport {

    /** Idle window before the model sleeps; short so the test does not dominate the suite. */
    private static final int SLEEP_IDLE_SECONDS = 1;

    /** Comfortably longer than {@link #SLEEP_IDLE_SECONDS}, so the sleep transition has run. */
    private static final long IDLE_WAIT_MILLIS = 3_000L;

    private static LlamaModel model;

    @BeforeAll
    public static void setup() {
        Assumptions.assumeTrue(
                new File(TestConstants.DRAFT_MODEL_PATH).exists(),
                "Draft model not found, skipping NativeServerAttachSleepIntegrationTest");
        int gpuLayers = Integer.getInteger(TestConstants.PROP_TEST_NGL, TestConstants.DEFAULT_TEST_NGL);
        model = new LlamaModel(new ModelParameters()
                .setModel(TestConstants.DRAFT_MODEL_PATH)
                .setCtxSize(512)
                .setGpuLayers(gpuLayers)
                .setSleepIdleSeconds(SLEEP_IDLE_SECONDS)
                .enableMetricsEndpoint());
    }

    @AfterAll
    public static void tearDown() {
        if (model != null) {
            model.close();
        }
    }

    private static int findFreePort() throws IOException {
        try (ServerSocket socket = new ServerSocket(0)) {
            return socket.getLocalPort();
        }
    }

    private void awaitHealthy(int port) throws Exception {
        long deadline = System.currentTimeMillis() + 30_000L;
        IOException last = null;
        while (System.currentTimeMillis() < deadline) {
            try {
                if (get(port, "/health", "").code == 200) {
                    return;
                }
            } catch (IOException e) {
                last = e;
            }
            Thread.sleep(200L);
        }
        fail("attached server did not become healthy within 30s" + (last != null ? ": " + last : ""));
    }

    /**
     * Without the fix this method does not fail an assertion: the first idle sleep after the first
     * server closed runs the dead callback, and the fork dies. The {@link Timeout} turns a hang into
     * a failure; a crash is reported by surefire as a crashed fork.
     */
    @Test
    @Timeout(value = 180, unit = TimeUnit.SECONDS)
    public void theModelSleepsAndWakesAfterTheAttachedServerClosed_andASecondServerAttaches() throws Exception {
        int firstPort = findFreePort();
        try (NativeServer first =
                new NativeServer(model, "--host", "127.0.0.1", "--port", Integer.toString(firstPort)).start()) {
            awaitHealthy(firstPort);
            Response completion =
                    post(firstPort, "/completion", "{\"prompt\":\"Hello\",\"n_predict\":2,\"temperature\":0}", "");
            assertThat(completion.body, completion.code, is(200));
        }

        // The model is idle now and crosses its threshold: the sleep transition runs every callback
        // the queue holds -- with the defect, the one the first server left behind.
        Thread.sleep(IDLE_WAIT_MILLIS);
        String afterFirstSleep = model.complete(new InferenceParameters("Say hi.").withNPredict(2));
        assertNotNull(afterFirstSleep, "completion after the first sleep returned null");

        int secondPort = findFreePort();
        try (NativeServer second =
                new NativeServer(model, "--host", "127.0.0.1", "--port", Integer.toString(secondPort)).start()) {
            awaitHealthy(secondPort);
            Response props = get(secondPort, "/props", "");
            assertThat(props.body, props.code, is(200));
            assertThat(props.body, containsString("default_generation_settings"));
            // The routes are the model's: --metrics came from ModelParameters, not from the attach argv.
            Response metrics = get(secondPort, "/metrics", "");
            assertThat(metrics.body, metrics.code, is(200));
            assertThat(metrics.body, containsString("llamacpp:"));
        }

        // A second sleep after the second server closed: two dead callbacks before the fix.
        Thread.sleep(IDLE_WAIT_MILLIS);
        String afterSecondSleep = model.complete(new InferenceParameters("And again.").withNPredict(2));
        assertNotNull(afterSecondSleep, "completion after the second sleep returned null");
        assertNotNull(model.getMetrics(), "getMetrics() must still be serviced after waking");
    }
}
