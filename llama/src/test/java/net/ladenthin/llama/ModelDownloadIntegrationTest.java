// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.sun.net.httpserver.HttpServer;
import java.io.File;
import java.io.IOException;
import java.io.OutputStream;
import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.nio.file.Files;
import java.nio.file.Path;
import net.ladenthin.llama.exception.LlamaException;
import net.ladenthin.llama.parameters.InferenceParameters;
import net.ladenthin.llama.parameters.ModelParameters;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * {@link ModelParameters#setModelUrl(String)} through JNI: the load downloads the model to the path of
 * {@link ModelParameters#setModel(String)} and loads it. Until 2026-11-19 it downloaded nothing -- the
 * load parses as llama.cpp's server example, which leaves the model files to {@code llama_server()},
 * and the load did not run upstream's models handler itself -- and failed on a path nobody had written.
 * A loopback HTTP server serves CI's draft model; the C API test {@code CApiDownload} does the same
 * with the generator's model.
 */
@ClaudeGenerated(
        purpose = "Pin that LlamaModel downloads a model URL to its model path before the load, and that a "
                + "URL that cannot be fetched fails the load by name.")
public class ModelDownloadIntegrationTest {

    private static HttpServer server;

    private static String base;

    @BeforeAll
    static void serveTheDraftModel() throws IOException {
        File model = new File(TestConstants.DRAFT_MODEL_PATH);
        Assumptions.assumeTrue(model.exists(), "Draft model not found, skipping ModelDownloadIntegrationTest");
        byte[] bytes = Files.readAllBytes(model.toPath());
        server = HttpServer.create(new InetSocketAddress(InetAddress.getLoopbackAddress(), 0), 0);
        server.createContext("/", exchange -> {
            try {
                if (!exchange.getRequestURI().getPath().equals("/model.gguf")) {
                    exchange.sendResponseHeaders(404, -1);
                    return;
                }
                exchange.getResponseHeaders().add("Content-Type", "application/octet-stream");
                if ("HEAD".equals(exchange.getRequestMethod())) {
                    exchange.getResponseHeaders().add("Content-Length", String.valueOf(bytes.length));
                    exchange.sendResponseHeaders(200, -1);
                    return;
                }
                exchange.sendResponseHeaders(200, bytes.length);
                try (OutputStream out = exchange.getResponseBody()) {
                    out.write(bytes);
                }
            } finally {
                exchange.close();
            }
        });
        server.start();
        base = "http://127.0.0.1:" + server.getAddress().getPort();
    }

    @AfterAll
    static void stop() {
        if (server != null) {
            server.stop(0);
        }
    }

    private static ModelParameters parameters(String url, Path target) {
        int gpuLayers = Integer.getInteger(TestConstants.PROP_TEST_NGL, TestConstants.DEFAULT_TEST_NGL);
        return new ModelParameters()
                .setModelUrl(url)
                .setModel(target.toString())
                .setCtxSize(256)
                .setGpuLayers(gpuLayers)
                .setFit(false);
    }

    @Test
    public void aModelUrlIsDownloadedToTheModelPathAndLoaded(@TempDir Path dir) throws IOException {
        Path target = dir.resolve("downloaded.gguf");
        try (LlamaModel model = new LlamaModel(parameters(base + "/model.gguf", target))) {
            String answer = model.complete(
                    new InferenceParameters("def main():").withNPredict(4).withTemperature(0.0f));
            assertFalse(answer.isEmpty(), "the downloaded model answered nothing");
        }
        assertTrue(Files.exists(target), "nothing was written to the model path");
        assertEquals(new File(TestConstants.DRAFT_MODEL_PATH).length(), Files.size(target));
    }

    @Test
    public void aUrlThatCannotBeFetchedFailsTheLoadByName(@TempDir Path dir) {
        Path target = dir.resolve("missing.gguf");
        LlamaException refused = assertThrows(
                LlamaException.class, () -> new LlamaModel(parameters(base + "/no-such.gguf", target)).close());
        assertTrue(refused.getMessage().contains("failed to fetch the model files"), refused.getMessage());
        assertFalse(Files.exists(target), "a failed download left a file at the model path");
    }
}
