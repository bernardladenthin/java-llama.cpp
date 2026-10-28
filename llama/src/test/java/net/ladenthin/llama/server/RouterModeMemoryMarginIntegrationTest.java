// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.is;
import static org.junit.jupiter.api.Assertions.fail;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Locale;
import net.ladenthin.llama.ClaudeGenerated;
import net.ladenthin.llama.TestConstants;
import net.ladenthin.llama.value.RouterModel;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * Router mode in the JVM with device-memory accounting ({@code --models-memory-margin}, llama.cpp
 * b11930, #21231). With a margin the router measures every model before it loads it: it spawns a
 * <em>measure</em> child with the instance's arguments plus {@code --offline}, which reports what the
 * model needs and exits. In a JVM router those arguments are the worker command of patch {@code 0008}
 * ({@link NativeServer#setWorkerCommand(String...)}), so the measure child is a JVM running
 * {@link NativeServer} too, and this test pins that it starts, measures and reports back: the model's
 * {@code memory.status} in {@code GET /models} becomes {@code "measured"}. A CPU-only host counts its
 * memory as unlimited, as {@code --fit} does, so the margin never refuses a load here; the measurement
 * runs all the same.
 *
 * <p>Linux-only, like {@link RouterModeIntegrationTest}, whose helpers it uses.</p>
 */
@ClaudeGenerated(
        purpose = "Prove that a JVM router with --models-memory-margin spawns its measure child as a "
                + "worker JVM (patch 0008) and gets a measurement back before the model loads.")
public class RouterModeMemoryMarginIntegrationTest extends OpenAiServerTestSupport {

    private static final long MODEL_READY_TIMEOUT_MILLIS = 240_000L;
    private static final ObjectMapper MAPPER = new ObjectMapper();

    @TempDir
    static Path modelsDir;

    private static NativeServer server;
    private static int port;
    private static String modelName;
    private static boolean workerCommandSet;

    @BeforeAll
    public static void setup() throws Exception {
        Assumptions.assumeTrue(
                System.getProperty("os.name", "").toLowerCase(Locale.ROOT).contains("linux"),
                "Router worker-relaunch test runs on Linux only");
        File reasoningModel = new File(TestConstants.REASONING_MODEL_PATH);
        Assumptions.assumeTrue(reasoningModel.exists(), "Reasoning model not found, skipping router test");
        String classpath = RouterModeIntegrationTest.workerClasspath();
        Assumptions.assumeTrue(
                !classpath.isEmpty() && !classpath.matches(".*\\s.*"),
                "Classpath contains whitespace; worker command cannot carry it");

        Files.createSymbolicLink(
                modelsDir.resolve(reasoningModel.getName()),
                reasoningModel.getAbsoluteFile().toPath());
        String javaBin =
                Paths.get(System.getProperty("java.home"), "bin", "java").toString();
        NativeServer.setWorkerCommand(javaBin, "-cp", classpath, "net.ladenthin.llama.server.NativeServer");
        workerCommandSet = true;

        int gpuLayers = Integer.getInteger(TestConstants.PROP_TEST_NGL, TestConstants.DEFAULT_TEST_NGL);
        port = RouterModeIntegrationTest.findFreePort();
        server = new NativeServer(
                        "--host",
                        "127.0.0.1",
                        "--port",
                        Integer.toString(port),
                        "--models-dir",
                        modelsDir.toString(),
                        "--models-max",
                        "1",
                        "--models-memory-margin",
                        "64",
                        "-c",
                        "512",
                        "-ngl",
                        Integer.toString(gpuLayers))
                .start();

        RouterClient client = new RouterClient(port);
        long deadline = System.currentTimeMillis() + 30_000L;
        while (true) {
            try {
                client.listModels();
                break;
            } catch (Exception e) {
                if (System.currentTimeMillis() > deadline) {
                    throw e;
                }
                Thread.sleep(200L);
            }
        }
        modelName = RouterModeIntegrationTest.discoverModelName(
                client, RouterModeIntegrationTest.expectedModelId(reasoningModel.getName()));
        client.loadModel(modelName);
        RouterModel loaded = client.awaitModelLoaded(modelName, MODEL_READY_TIMEOUT_MILLIS);
        assertThat(loaded.getStatus(), is(RouterModel.Status.LOADED));
    }

    @AfterAll
    public static void tearDown() {
        if (server != null) {
            server.close();
        }
        if (workerCommandSet) {
            NativeServer.setWorkerCommand();
        }
    }

    @Test
    public void theMeasureChildRanAsAWorkerJvmAndReportedTheModelsMemory() throws Exception {
        JsonNode memory = modelEntry().path("memory");
        assertThat("memory of " + memory, memory.path("status").asText(), is("measured"));
        assertThat("min of " + memory, memory.path("min").isObject(), is(true));
        assertThat("max of " + memory, memory.path("max").isObject(), is(true));
    }

    /** The entry of the model under test in {@code GET /models}. */
    private JsonNode modelEntry() throws Exception {
        Response response = get(port, "/models", "");
        assertThat(response.body, response.code, is(200));
        for (JsonNode entry : MAPPER.readTree(response.body).path("data")) {
            if (modelName.equals(entry.path("id").asText())) {
                return entry;
            }
        }
        fail("GET /models did not list '" + modelName + "': " + response.body);
        return MAPPER.createObjectNode(); // unreachable
    }
}
