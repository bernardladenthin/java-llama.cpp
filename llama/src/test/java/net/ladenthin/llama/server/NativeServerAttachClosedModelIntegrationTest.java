// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.is;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.fail;

import java.io.File;
import java.io.IOException;
import java.net.ServerSocket;
import net.ladenthin.llama.ClaudeGenerated;
import net.ladenthin.llama.LlamaModel;
import net.ladenthin.llama.TestConstants;
import net.ladenthin.llama.exception.LlamaException;
import net.ladenthin.llama.parameters.ModelParameters;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/**
 * {@link NativeServer}'s attach mode on a {@link LlamaModel} that is already closed: the start is
 * refused with the native side's message, before any socket is bound, and the process-wide
 * single-instance slot is given back, so an attach to an open model starts afterwards.
 */
@ClaudeGenerated(
        purpose = "Pin that attaching a NativeServer to a closed LlamaModel throws LlamaException naming the "
                + "closed model, binds nothing, and leaves the single-instance slot free for the next server.")
public class NativeServerAttachClosedModelIntegrationTest extends OpenAiServerTestSupport {

    @BeforeAll
    public static void assumeDraftModel() {
        Assumptions.assumeTrue(
                new File(TestConstants.DRAFT_MODEL_PATH).exists(),
                "Draft model not found, skipping NativeServerAttachClosedModelIntegrationTest");
    }

    private static LlamaModel loadDraftModel() {
        int gpuLayers = Integer.getInteger(TestConstants.PROP_TEST_NGL, TestConstants.DEFAULT_TEST_NGL);
        return new LlamaModel(new ModelParameters()
                .setModel(TestConstants.DRAFT_MODEL_PATH)
                .setCtxSize(256)
                .setGpuLayers(gpuLayers)
                .setFit(false));
    }

    private static int findFreePort() throws IOException {
        try (ServerSocket socket = new ServerSocket(0)) {
            return socket.getLocalPort();
        }
    }

    private void awaitHealthy(int port) throws InterruptedException {
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

    @Test
    public void attachingToAClosedModelIsRefusedAndTheNextAttachStarts() throws Exception {
        // arrange
        LlamaModel closed = loadDraftModel();
        closed.close();
        int port = findFreePort();
        NativeServer refused = new NativeServer(closed, "--host", "127.0.0.1", "--port", Integer.toString(port));

        // act
        LlamaException error = assertThrows(LlamaException.class, refused::start);

        // assert
        assertThat(error.getMessage(), containsString("model is not loaded (or already closed)"));
        assertThat(refused.isRunning(), is(false));
        assertThrows(IOException.class, () -> get(port, "/health", ""), "the refused start bound the port");
        refused.close(); // nothing to stop: a no-op
        try (LlamaModel open = loadDraftModel();
                NativeServer server =
                        new NativeServer(open, "--host", "127.0.0.1", "--port", Integer.toString(port)).start()) {
            awaitHealthy(port);
            assertThat(server.isRunning(), is(true));
        }
    }
}
