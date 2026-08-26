// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.is;
import static org.junit.jupiter.api.Assertions.fail;

import java.io.File;
import java.io.IOException;
import java.net.ServerSocket;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.security.CodeSource;
import java.util.List;
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
 * Integration test for upstream <em>router mode</em> running inside the JVM via
 * {@link NativeServer}: started without a model argument, the server manages models from
 * {@code --models-dir} and serves each from a worker subprocess. In-JVM the upstream worker-spawn
 * (re-exec of the process binary) would relaunch {@code java} with llama-server arguments, so
 * {@link NativeServer#setWorkerCommand(String...)} redirects workers to a fresh JVM running the
 * classic single-model {@code NativeServer} (patch {@code 0008}).
 *
 * <p>Linux-only: worker relaunch and the symlinked models dir are exercised on one platform;
 * router mode itself is upstream functionality, this test pins the embedded wiring.</p>
 */
@ClaudeGenerated(
        purpose = "Prove in-JVM router mode end to end: model listing from --models-dir, "
                + "explicit /models/load spawning a worker JVM through setWorkerCommand, and a "
                + "chat completion proxied to that worker.")
public class RouterModeIntegrationTest extends OpenAiServerTestSupport {

    /** Generous ceiling for worker-JVM spawn + model load on a cold CI runner. */
    private static final long MODEL_READY_TIMEOUT_MILLIS = 240_000L;

    @TempDir
    static Path modelsDir;

    private static NativeServer server;
    private static int port;
    private static String modelName;

    /**
     * Set only after {@link NativeServer#setWorkerCommand(String...)} succeeded in setup.
     * tearDown must clear the override ONLY in that case: setWorkerCommand loads the native
     * library, and when the class self-skipped via an {@code @BeforeAll} assumption (no model /
     * non-Linux / lib-less analysis runner such as the Sonar job), {@code @AfterAll} still runs —
     * an unconditional call would then die with UnsatisfiedLinkError instead of skipping cleanly.
     */
    private static boolean workerCommandSet;

    @BeforeAll
    public static void setup() throws Exception {
        Assumptions.assumeTrue(
                System.getProperty("os.name", "").toLowerCase(Locale.ROOT).contains("linux"),
                "Router worker-relaunch test runs on Linux only");
        File reasoningModel = new File(TestConstants.REASONING_MODEL_PATH);
        Assumptions.assumeTrue(reasoningModel.exists(), "Reasoning model not found, skipping router test");
        String classpath = workerClasspath();
        Assumptions.assumeTrue(
                !classpath.isEmpty() && !classpath.matches(".*\\s.*"),
                "Classpath contains whitespace; worker command cannot carry it");

        // The models dir must contain ONLY the model under test (the CI models/ dir also holds
        // mmproj/vocoder files the router would list); a symlink avoids copying ~400 MB.
        Files.createSymbolicLink(
                modelsDir.resolve(reasoningModel.getName()),
                reasoningModel.getAbsoluteFile().toPath());

        // Workers must relaunch through this library, not through the JVM binary itself.
        String javaBin =
                Paths.get(System.getProperty("java.home"), "bin", "java").toString();
        NativeServer.setWorkerCommand(javaBin, "-cp", classpath, "net.ladenthin.llama.server.NativeServer");
        workerCommandSet = true;

        int gpuLayers = Integer.getInteger(TestConstants.PROP_TEST_NGL, TestConstants.DEFAULT_TEST_NGL);
        port = findFreePort();
        // The router forwards its own base arguments (context size, GPU layers) to each worker.
        server = new NativeServer(
                        "--host",
                        "127.0.0.1",
                        "--port",
                        Integer.toString(port),
                        "--models-dir",
                        modelsDir.toString(),
                        "--models-max",
                        "1",
                        "-c",
                        "512",
                        "-ngl",
                        Integer.toString(gpuLayers))
                .start();
        awaitHttp("/health", 30_000L);

        // Model discovery + load + readiness go through the typed RouterClient, so this
        // integration test exercises it end to end against a real router (the wire-shape
        // details are unit-tested model-free in RouterClientTest against a stub server).
        RouterClient client = new RouterClient(port);
        modelName = discoverModelName(client);
        client.loadModel(modelName);
        RouterModel loaded = client.awaitModelLoaded(modelName, MODEL_READY_TIMEOUT_MILLIS);
        assertThat(loaded.getStatus(), is(RouterModel.Status.LOADED));
    }

    @AfterAll
    public static void tearDown() {
        if (server != null) {
            server.close(); // router clean-up unloads (terminates) all worker instances
        }
        if (workerCommandSet) {
            NativeServer.setWorkerCommand(); // clear the process-wide override
        }
    }

    /**
     * Builds the classpath the worker JVM is launched with.
     *
     * <p>{@code java.class.path} alone is NOT enough. {@code target/classes} carries a
     * {@code module-info.class}, so Surefire auto-detects a named module and runs the main classes
     * on the <em>module path</em>; {@code java.class.path} then holds only {@code target/test-classes}
     * plus the dependency jars. A worker launched with just that dies immediately with
     * {@code ClassNotFoundException: net.ladenthin.llama.server.NativeServer}, which the router
     * surfaces only as the opaque "worker failed with exit code 1". The main-classes root is
     * therefore derived from the class itself — correct in either mode, and a directory or a jar
     * alike. It also carries the packaged native library the worker has to extract.</p>
     *
     * @return the worker classpath, or an empty string when the code source cannot be resolved
     */
    private static String workerClasspath() throws Exception {
        String inherited = System.getProperty("java.class.path", "");
        CodeSource source = NativeServer.class.getProtectionDomain().getCodeSource();
        if (source == null || source.getLocation() == null) {
            return inherited;
        }
        String mainClasses =
                Paths.get(source.getLocation().toURI()).toAbsolutePath().toString();
        if (inherited.isEmpty()) {
            return mainClasses;
        }
        if ((File.pathSeparator + inherited + File.pathSeparator)
                .contains(File.pathSeparator + mainClasses + File.pathSeparator)) {
            return inherited;
        }
        return mainClasses + File.pathSeparator + inherited;
    }

    private static int findFreePort() throws IOException {
        try (ServerSocket socket = new ServerSocket(0)) {
            return socket.getLocalPort();
        }
    }

    private static void awaitHttp(String path, long timeoutMillis) throws Exception {
        long deadline = System.currentTimeMillis() + timeoutMillis;
        IOException last = null;
        while (System.currentTimeMillis() < deadline) {
            try {
                if (new RouterModeIntegrationTest().get(port, path, "").code == 200) {
                    return;
                }
            } catch (IOException e) {
                last = e;
            }
            Thread.sleep(200L);
        }
        fail("router did not answer " + path + " within " + timeoutMillis + "ms" + (last != null ? ": " + last : ""));
    }

    /** Finds the entry for the symlinked GGUF in the typed model list and returns its identifier. */
    private static String discoverModelName(RouterClient client) throws Exception {
        List<RouterModel> models = client.listModels();
        for (RouterModel model : models) {
            if (model.getId().contains("Qwen3")) {
                return model.getId();
            }
        }
        fail("GET /models did not list the symlinked model: " + models);
        return ""; // unreachable
    }

    @Test
    public void chatCompletion_isProxiedToWorker() throws IOException {
        Response response = post(
                port,
                "/v1/chat/completions",
                "{\"model\":\"" + modelName + "\",\"messages\":[{\"role\":\"user\",\"content\":\"Say one word.\"}],"
                        + "\"max_tokens\":8}",
                "");
        assertThat(response.body, response.code, is(200));
        assertThat(response.body, containsString("\"choices\""));
    }

    @Test
    public void models_listContainsLoadedModel() throws IOException {
        Response models = get(port, "/models", "");
        assertThat(models.code, is(200));
        assertThat(models.body, containsString(modelName));
    }
}
