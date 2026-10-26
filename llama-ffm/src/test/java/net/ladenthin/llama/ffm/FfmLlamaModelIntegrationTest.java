// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT
package net.ladenthin.llama.ffm;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.greaterThan;
import static org.hamcrest.Matchers.is;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.IOException;
import java.net.HttpURLConnection;
import java.net.ServerSocket;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicInteger;
import net.ladenthin.llama.exception.InvalidRequestException;
import net.ladenthin.llama.exception.LlamaException;
import net.ladenthin.llama.parameters.ModelParameters;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * A model through the C API: loaded with the core's {@code ModelParameters}, the contract of each
 * operation asserted (the text of a small model is noise, the shape is what counts). Self-skips
 * without a native build or the model ({@code -Dnet.ladenthin.llama.ffm.model}).
 */
class FfmLlamaModelIntegrationTest {

    private static final ObjectMapper JSON = new ObjectMapper();
    private static final String CHAT_REQUEST =
            "{\"messages\": [{\"role\": \"user\", \"content\": \"Hi\"}], \"max_tokens\": 4, \"temperature\": 0}";

    private static FfmLlamaModel model;
    private static final AtomicInteger PROGRESS_CALLS = new AtomicInteger();

    @BeforeAll
    static void load() {
        Path path = FfmTestSupport.assumeModel();
        ModelParameters parameters = new ModelParameters()
                .setModel(path.toString())
                .setCtxSize(128)
                .setGpuLayers(0)
                .setThreads(2);
        model = FfmLlamaModel.load(parameters, progress -> {
            PROGRESS_CALLS.incrementAndGet();
            return true;
        });
    }

    @AfterAll
    static void free() {
        if (model != null) {
            model.close();
            model.close(); // idempotent
        }
    }

    @Test
    void theProgressCallbackWasReported() {
        assertThat(PROGRESS_CALLS.get(), greaterThan(0));
    }

    @Test
    void metaDescribesTheModel() throws IOException {
        JsonNode meta = JSON.readTree(model.meta());
        assertThat(meta.path("n_vocab").asInt(), greaterThan(0));
        assertTrue(meta.has("metadata"));
        // a model without a projector chats and neither generates nor streams audio (b11901)
        JsonNode capabilities = meta.path("capabilities");
        assertTrue(capabilities.path("chat").asBoolean(false), capabilities.toString());
        assertFalse(capabilities.path("audio_gen").asBoolean(true), capabilities.toString());
        assertFalse(capabilities.path("audio_stream").asBoolean(true), capabilities.toString());
    }

    @Test
    void chatCompletionIsAnOpenAiObject() throws IOException {
        JsonNode response = JSON.readTree(model.chatCompletions(CHAT_REQUEST));
        assertEquals("chat.completion", response.path("object").asText());
        assertTrue(response.path("choices").isArray());
        assertTrue(response.path("choices").get(0).path("message").has("content"));
    }

    @Test
    void chatCompletionStreamsEnvelopesUntilStop() throws IOException {
        int envelopes = 0;
        boolean sawStop = false;
        try (FfmTaskStream stream = model.chatCompletionsStream(CHAT_REQUEST)) {
            assertThat(stream.taskId(), is(greaterThan(-1)));
            while (stream.hasNext()) {
                JsonNode envelope = JSON.readTree(stream.next());
                assertTrue(envelope.has("data"));
                envelopes++;
                sawStop = envelope.path("stop").asBoolean();
            }
        }
        assertThat(envelopes, greaterThan(0));
        assertTrue(sawStop, "the last envelope carries stop");
    }

    @Test
    void aStreamClosedEarlyReleasesItsTaskAndTheModelStillAnswers() throws IOException {
        try (FfmTaskStream stream = model.chatCompletionsStream(CHAT_REQUEST)) {
            stream.next();
        }
        assertEquals("chat.completion", JSON.readTree(model.chatCompletions(CHAT_REQUEST)).path("object").asText());
    }

    @Test
    void rawCompletionAndItsStream() throws IOException {
        String request = "{\"prompt\": \"Hello\", \"n_predict\": 3, \"temperature\": 0}";
        assertTrue(JSON.readTree(model.completions(request)).has("content"));
        int results = 0;
        try (FfmTaskStream stream = model.completionsStream(request)) {
            while (stream.hasNext()) {
                JSON.readTree(stream.next());
                results++;
            }
        }
        assertThat(results, greaterThan(0));
    }

    @Test
    void oaiCompletionStreamEndsWithUsage() throws IOException {
        JsonNode last = null;
        try (FfmTaskStream stream =
                model.completionsOaiStream("{\"prompt\": \"Hello\", \"max_tokens\": 3, \"temperature\": 0}")) {
            while (stream.hasNext()) {
                JsonNode data = JSON.readTree(stream.next()).path("data");
                for (JsonNode chunk : data.isArray() ? data : JSON.createArrayNode().add(data)) {
                    assertEquals("text_completion", chunk.path("object").asText());
                    last = chunk;
                }
            }
        }
        assertTrue(last != null && last.path("usage").path("completion_tokens").asInt() > 0, String.valueOf(last));
    }

    @Test
    void tokenizeAndDetokenizeRoundTrip() throws IOException {
        int[] tokens = model.tokenize("Hello", false);
        assertThat(tokens.length, greaterThan(0));
        // A SentencePiece tokenizer (the CI draft model) renders the first piece's word-boundary
        // marker as a leading space, " Hello"; a byte-level tokenizer (the GPT-2 test fixture)
        // gives "Hello". The round trip is the text up to that boundary.
        String detokenized = model.detokenize(tokens);
        assertEquals("Hello", detokenized.trim());
        JsonNode json = JSON.readTree(model.tokenizeJson("Hello", false, true));
        assertEquals(tokens.length, json.path("tokens").size());
        assertArrayEquals(tokens, JSON.convertValue(json.path("tokens").findValues("id"), int[].class));
        StringBuilder pieces = new StringBuilder();
        for (JsonNode piece : json.path("tokens").findValues("piece")) {
            pieces.append(piece.asText());
        }
        assertEquals(detokenized, pieces.toString());
        assertTrue(JSON.readTree(model.detokenizeJson(tokens)).has("content"));
    }

    @Test
    void applyTemplateRendersAPrompt() {
        String prompt = model.applyTemplate("{\"messages\": [{\"role\": \"user\", \"content\": \"Hi\"}]}");
        assertTrue(prompt.contains("Hi"));
    }

    @Test
    void metricsListTheSlots() throws IOException {
        JsonNode metrics = JSON.readTree(model.metrics());
        assertTrue(metrics.path("slots").isArray());
        assertThat(metrics.path("slots").size(), greaterThan(0));
    }

    @Test
    void loraAdaptersAndConfigure() throws IOException {
        assertTrue(JSON.readTree(model.loraAdapters()).isArray());
        assertTrue(JSON.readTree(model.setLoraAdapters("[]")).isContainerNode());
        assertThrows(LlamaException.class, () -> model.setLoraAdapters("{}"));
        assertThrows(InvalidRequestException.class, () -> model.setLoraAdapters("["));
        model.configure("{\"slot_prompt_similarity\": 0.5}");
        assertThrows(InvalidRequestException.class, () -> model.configure("{not json"));
    }

    @Test
    void slotEraseAnswersAndASaveWithoutASlotDirectoryIsRefused() throws IOException {
        assertTrue(JSON.readTree(model.slotAction(SlotOp.ERASE, 0, null)).isObject());
        LlamaException e = assertThrows(LlamaException.class, () -> model.slotAction(SlotOp.SAVE, 0, "slot0.bin"));
        assertFalse(e instanceof InvalidRequestException, e.getMessage());
        assertEquals(
                "Saving or restoring a slot needs a slot directory: load the model with --slot-save-path <dir> "
                        + "(ModelParameters.setSlotSavePath)",
                e.getMessage());
    }

    @Test
    void aSlotFileIsANameInsideTheSlotDirectory(@TempDir Path slotDir) throws IOException {
        ModelParameters parameters = new ModelParameters()
                .setModel(FfmTestSupport.assumeModel().toString())
                .setCtxSize(128)
                .setGpuLayers(0)
                .setThreads(2)
                .setSlotSavePath(slotDir.toString());
        try (FfmLlamaModel withSlots = FfmLlamaModel.load(parameters, null)) {
            String saved = withSlots.slotAction(SlotOp.SAVE, 0, "slot0.bin");
            assertTrue(Files.exists(slotDir.resolve("slot0.bin")), saved);
            assertEquals("slot0.bin", JSON.readTree(saved).path("filename").asText(), saved);
            assertEquals(0, JSON.readTree(withSlots.slotAction(SlotOp.RESTORE, 0, "slot0.bin")).path("id_slot").asInt());
            // unique to the run, so a file another run left next to the temp directory cannot decide it
            String escapedName = "escaped-" + UUID.randomUUID() + ".bin";
            assertThrows(
                    InvalidRequestException.class, () -> withSlots.slotAction(SlotOp.SAVE, 0, "../" + escapedName));
            assertThrows(
                    InvalidRequestException.class,
                    () -> withSlots.slotAction(SlotOp.RESTORE, 0, slotDir.resolve("slot0.bin").toString()));
            assertFalse(Files.exists(slotDir.getParent().resolve(escapedName)));
        }
    }

    @Test
    void aModelWithoutEmbeddingSupportRefusesToEmbed() {
        assertThrows(LlamaException.class, () -> model.embed("Hello"));
        assertThrows(LlamaException.class, () -> model.rerank("query", "a document"));
    }

    @Test
    void anInvalidRequestIsTheCallersFault() {
        assertThrows(InvalidRequestException.class, () -> model.chatCompletions("{not json"));
        assertThrows(InvalidRequestException.class, () -> model.chatCompletions("{\"messages\": \"no\"}"));
        assertThrows(InvalidRequestException.class, () -> model.applyTemplate("{\"messages\": \"no\"}"));
    }

    @Test
    void theServerAttachedToTheModelAnswersAndTheModelOutlivesIt() throws Exception {
        int port;
        try (ServerSocket socket = new ServerSocket(0)) {
            port = socket.getLocalPort();
        }
        try (FfmNativeServer server = FfmNativeServer.startAttached(model, "--host", "127.0.0.1", "--port",
                String.valueOf(port))) {
            assertTrue(server.isRunning());
            assertEquals(200, awaitHealth(port));
        }
        assertEquals("chat.completion", JSON.readTree(model.chatCompletions(CHAT_REQUEST)).path("object").asText());
    }

    @Test
    void anErrorThrownByTheProgressCallbackAbortsTheLoadInsteadOfTheJvm() {
        ModelParameters parameters = new ModelParameters()
                .setModel(FfmTestSupport.modelPath().toString())
                .setCtxSize(64)
                .setGpuLayers(0)
                .setThreads(1);
        assertThrows(LlamaException.class, () -> FfmLlamaModel.load(parameters, progress -> {
            throw new AssertionError("from the callback");
        }));
    }

    @Test
    void aClosedModelRefusesCalls() {
        FfmLlamaModel other = FfmLlamaModel.load(new ModelParameters()
                .setModel(FfmTestSupport.modelPath().toString())
                .setCtxSize(64)
                .setGpuLayers(0)
                .setThreads(1));
        other.close();
        assertThrows(IllegalStateException.class, other::meta);
        assertFalse(FfmRpcServer.isListening(), "no RPC server runs in this JVM");
    }

    private static int awaitHealth(int port) throws Exception {
        URI health = URI.create("http://127.0.0.1:" + port + "/health");
        int last = -1;
        for (int attempt = 0; attempt < 200; attempt++) {
            try {
                HttpURLConnection connection = (HttpURLConnection) health.toURL().openConnection();
                connection.setConnectTimeout(500);
                connection.setReadTimeout(2000);
                last = connection.getResponseCode();
                connection.disconnect();
                if (last == 200) {
                    return last;
                }
            } catch (IOException e) {
                // not listening yet
            }
            Thread.sleep(50);
        }
        return last;
    }
}
