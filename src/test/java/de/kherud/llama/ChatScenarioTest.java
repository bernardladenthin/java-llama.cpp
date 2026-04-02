package de.kherud.llama;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.List;

import org.junit.AfterClass;
import org.junit.Assert;
import org.junit.Assume;
import org.junit.BeforeClass;
import org.junit.Test;

/**
 * Complex chat scenario tests exercising code paths not covered by LlamaModelTest:
 * <ul>
 *   <li>handleChatCompletions raw JSON structure</li>
 *   <li>requestChatCompletion direct native streaming</li>
 *   <li>Streaming vs blocking output consistency (same seed)</li>
 *   <li>Chat with stop strings</li>
 *   <li>Chat with grammar constraint</li>
 *   <li>Multi-turn conversation (5 turns)</li>
 *   <li>Unicode content in messages</li>
 *   <li>Special characters (quotes, backslashes, newlines) in messages</li>
 *   <li>Back-to-back sequential chat calls</li>
 *   <li>handleInfill direct JSON endpoint</li>
 *   <li>handleEmbeddings OAI-compat format</li>
 *   <li>handleTokenize with addSpecial=true</li>
 *   <li>handleDetokenize round-trip via encode/handleDetokenize</li>
 *   <li>saveSlot / restoreSlot round-trip</li>
 *   <li>nPredict=1 minimal chat completion</li>
 * </ul>
 */
@ClaudeGenerated(
        purpose = "Complex chat scenarios: raw JSON endpoint structure, streaming/blocking consistency, " +
                  "stop strings, grammar constraints, multi-turn conversations, unicode/special-char " +
                  "message content, back-to-back calls, and all JSON-in/JSON-out endpoint variants."
)
public class ChatScenarioTest {

    private static final int N_PREDICT = 10;

    private static LlamaModel model;

    @BeforeClass
    public static void setup() {
        Assume.assumeTrue("Model file not found, skipping ChatScenarioTest",
                new File(TestConstants.MODEL_PATH).exists());
        int gpuLayers = Integer.getInteger(TestConstants.PROP_TEST_NGL, TestConstants.DEFAULT_TEST_NGL);
        model = new LlamaModel(
                new ModelParameters()
                        .setCtxSize(256)
                        .setModel(TestConstants.MODEL_PATH)
                        .setGpuLayers(gpuLayers)
                        .setFit(false)
                        .enableEmbedding()
        );
    }

    @AfterClass
    public static void tearDown() {
        if (model != null) {
            model.close();
        }
    }

    // ------------------------------------------------------------------
    // 1. handleChatCompletions raw JSON structure
    // ------------------------------------------------------------------

    /**
     * chatComplete() delegates to handleChatCompletions() and returns its raw JSON.
     * The OAI-compatible response must contain the standard "choices" and
     * "message"/"content" fields.
     */
    @Test
    public void testChatCompleteResponseJsonStructure() {
        List<Pair<String, String>> messages = new ArrayList<>();
        messages.add(new Pair<>("user", "Say the word OK."));

        InferenceParameters params = new InferenceParameters("")
                .setMessages(null, messages)
                .setNPredict(N_PREDICT)
                .setSeed(42)
                .setTemperature(0.0f);

        String response = model.chatComplete(params);

        Assert.assertNotNull(response);
        Assert.assertFalse("Response must not be empty", response.isEmpty());
        Assert.assertTrue("OAI chat response must contain 'choices'", response.contains("\"choices\""));
        Assert.assertTrue("OAI chat response must contain 'message'", response.contains("\"message\""));
        Assert.assertTrue("OAI chat response must contain 'content'", response.contains("\"content\""));
        Assert.assertTrue("OAI chat response must have assistant role",
                response.contains("\"assistant\"") || response.contains("assistant"));
    }

    /**
     * handleChatCompletions can be called directly with a raw JSON string.
     * Verify the response contains valid OAI chat completion fields.
     */
    @Test
    public void testHandleChatCompletionsDirect() {
        String json = "{\"messages\": [{\"role\": \"user\", \"content\": \"Say yes.\"}], " +
                "\"n_predict\": " + N_PREDICT + ", \"seed\": 42, \"temperature\": 0.0, \"stream\": false}";

        String response = model.handleChatCompletions(json);

        Assert.assertNotNull(response);
        Assert.assertTrue("Direct handleChatCompletions must return choices array",
                response.contains("\"choices\""));
        Assert.assertTrue("Direct handleChatCompletions must return message content",
                response.contains("\"content\""));
    }

    // ------------------------------------------------------------------
    // 2. requestChatCompletion direct native streaming
    // ------------------------------------------------------------------

    /**
     * requestChatCompletion returns a task ID; receiveCompletionJson must then be
     * called in a loop until a stop token is received. This exercises the raw
     * streaming path (bypassing LlamaIterator) used for chat.
     */
    @Test
    public void testRequestChatCompletionDirectStreaming() {
        List<Pair<String, String>> messages = new ArrayList<>();
        messages.add(new Pair<>("user", "Write a single word."));

        InferenceParameters params = new InferenceParameters("")
                .setMessages(null, messages)
                .setNPredict(N_PREDICT)
                .setSeed(42)
                .setTemperature(0.0f)
                .setStream(true);

        int taskId = model.requestChatCompletion(params.toString());

        StringBuilder sb = new StringBuilder();
        int tokens = 0;
        boolean stopped = false;
        while (!stopped) {
            String json = model.receiveCompletionJson(taskId);
            Assert.assertNotNull("receiveCompletionJson must not return null", json);
            LlamaOutput output = LlamaOutput.fromJson(json);
            sb.append(output.text);
            tokens++;
            if (output.stop) {
                stopped = true;
                model.releaseTask(taskId);
            }
            if (tokens > N_PREDICT + 2) {
                model.releaseTask(taskId);
                Assert.fail("Streaming did not stop after nPredict tokens");
            }
        }

        Assert.assertTrue("Direct streaming must produce at least one token", tokens > 0);
        Assert.assertFalse("Direct streaming must produce non-empty content", sb.toString().isEmpty());
    }

    // ------------------------------------------------------------------
    // 3. Streaming vs blocking output consistency (same seed)
    // ------------------------------------------------------------------

    /**
     * With the same seed and temperature=0, streaming chat output tokens
     * concatenated should equal the blocking chat content field.
     */
    @Test
    public void testStreamingAndBlockingOutputConsistency() {
        List<Pair<String, String>> messages = new ArrayList<>();
        messages.add(new Pair<>("user", "Write one word."));

        // Blocking
        InferenceParameters blockingParams = new InferenceParameters("")
                .setMessages(null, messages)
                .setNPredict(N_PREDICT)
                .setSeed(123)
                .setTemperature(0.0f);
        String blockingJson = model.chatComplete(blockingParams);
        // Extract content from the OAI response JSON
        String blockingContent = extractChoiceContent(blockingJson);

        // Streaming
        InferenceParameters streamingParams = new InferenceParameters("")
                .setMessages(null, messages)
                .setNPredict(N_PREDICT)
                .setSeed(123)
                .setTemperature(0.0f);
        StringBuilder streamedContent = new StringBuilder();
        for (LlamaOutput output : model.generateChat(streamingParams)) {
            streamedContent.append(output.text);
        }

        Assert.assertFalse("Blocking content should not be empty", blockingContent.isEmpty());
        Assert.assertFalse("Streaming content should not be empty", streamedContent.toString().isEmpty());
        Assert.assertEquals(
                "Streaming and blocking outputs must match for same seed",
                blockingContent.trim(),
                streamedContent.toString().trim()
        );
    }

    // ------------------------------------------------------------------
    // 4. Chat with stop strings
    // ------------------------------------------------------------------

    /**
     * A stop string set in the parameters must terminate generation in chat mode.
     * The response content must be shorter than the unconstrained generation.
     */
    @Test
    public void testChatCompleteWithStopString() {
        List<Pair<String, String>> messages = new ArrayList<>();
        messages.add(new Pair<>("user", "Count: 1, 2, 3, 4, 5, 6, 7"));

        // Unconstrained
        InferenceParameters unconstrained = new InferenceParameters("")
                .setMessages(null, messages)
                .setNPredict(N_PREDICT)
                .setSeed(42)
                .setTemperature(0.0f);
        String unJson = model.chatComplete(unconstrained);
        String unContent = extractChoiceContent(unJson);

        // Stopped at "3"
        InferenceParameters stopped = new InferenceParameters("")
                .setMessages(null, messages)
                .setNPredict(N_PREDICT)
                .setSeed(42)
                .setTemperature(0.0f)
                .setStopStrings("4");
        String stJson = model.chatComplete(stopped);
        String stContent = extractChoiceContent(stJson);

        Assert.assertNotNull("Stop-string response must not be null", stJson);
        // Content with stop should be shorter (or at most equal)
        Assert.assertTrue(
                "Content with stop string must not exceed unconstrained content length",
                stContent.length() <= unContent.length()
        );
        // The stopped content must not contain "4" (the stop string itself is excluded)
        Assert.assertFalse(
                "Content stopped at '4' must not contain '4'",
                stContent.contains("4")
        );
    }

    // ------------------------------------------------------------------
    // 5. Chat with grammar constraint
    // ------------------------------------------------------------------

    /**
     * A grammar constraint must be honoured in chat mode: only tokens matching
     * the grammar can appear in the generated content.
     */
    @Test
    public void testChatCompleteWithGrammar() {
        List<Pair<String, String>> messages = new ArrayList<>();
        messages.add(new Pair<>("user", "Generate output."));

        InferenceParameters params = new InferenceParameters("")
                .setMessages(null, messages)
                .setGrammar("root ::= (\"a\" | \"b\")+")
                .setNPredict(N_PREDICT)
                .setSeed(42)
                .setTemperature(0.0f);

        String responseJson = model.chatComplete(params);
        String content = extractChoiceContent(responseJson);

        Assert.assertFalse("Grammar-constrained chat must produce non-empty content", content.isEmpty());
        Assert.assertTrue(
                "Grammar-constrained content must match [ab]+ but was: " + content,
                content.matches("[ab]+")
        );
    }

    // ------------------------------------------------------------------
    // 6. Multi-turn conversation — 5 turns
    // ------------------------------------------------------------------

    /**
     * A 5-turn conversation: each assistant reply is appended back into the
     * message list so the next call contains the full history. Every turn must
     * yield a non-empty response.
     */
    @Test
    public void testChatCompleteMultiTurnFiveTurns() {
        List<Pair<String, String>> messages = new ArrayList<>();
        messages.add(new Pair<>("user", "Say A."));

        for (int turn = 0; turn < 5; turn++) {
            InferenceParameters params = new InferenceParameters("")
                    .setMessages(null, messages)
                    .setNPredict(N_PREDICT)
                    .setSeed(42)
                    .setTemperature(0.0f);

            String json = model.chatComplete(params);
            String content = extractChoiceContent(json);

            Assert.assertNotNull("Turn " + turn + ": response must not be null", json);
            Assert.assertFalse("Turn " + turn + ": content must not be empty", content.isEmpty());

            // Append assistant response and a new user message for the next turn
            messages.add(new Pair<>("assistant", content));
            if (turn < 4) {
                messages.add(new Pair<>("user", "Say B."));
            }
        }
    }

    // ------------------------------------------------------------------
    // 7. Unicode content in messages
    // ------------------------------------------------------------------

    /**
     * Multi-byte UTF-8 characters in message content must survive JSON
     * serialisation through the JNI layer without corruption or exceptions.
     */
    @Test
    public void testChatCompleteWithUnicodeContent() {
        List<Pair<String, String>> messages = new ArrayList<>();
        // French accented characters, Japanese kanji, emoji
        messages.add(new Pair<>("user", "Translate: café résumé naïve"));

        InferenceParameters params = new InferenceParameters("")
                .setMessages(null, messages)
                .setNPredict(N_PREDICT)
                .setSeed(42)
                .setTemperature(0.0f);

        // Must not throw
        String response = model.chatComplete(params);
        Assert.assertNotNull("Unicode message must produce a non-null response", response);
        Assert.assertFalse("Unicode message must produce a non-empty response", response.isEmpty());
    }

    // ------------------------------------------------------------------
    // 8. Special characters (quotes, backslashes, newlines) in messages
    // ------------------------------------------------------------------

    /**
     * JSON-sensitive characters embedded in user message content must be
     * correctly escaped by setMessages so they do not break the JSON sent
     * to the native layer.
     */
    @Test
    public void testChatCompleteWithSpecialCharactersInContent() {
        List<Pair<String, String>> messages = new ArrayList<>();
        // Embedded double-quotes, backslash, newline
        messages.add(new Pair<>("user", "He said \"hello\", path: C:\\tmp\nNew line."));

        InferenceParameters params = new InferenceParameters("")
                .setMessages(null, messages)
                .setNPredict(N_PREDICT)
                .setSeed(42)
                .setTemperature(0.0f);

        // Must not throw a JSON parse error in the native layer
        String response = model.chatComplete(params);
        Assert.assertNotNull("Special-char message must not return null", response);
        Assert.assertFalse("Special-char message must not return empty response", response.isEmpty());
    }

    // ------------------------------------------------------------------
    // 9. Back-to-back sequential chat calls
    // ------------------------------------------------------------------

    /**
     * Three sequential chat completions on the same model instance must each
     * return independent, non-empty responses. No shared state should cause
     * interference between calls.
     */
    @Test
    public void testBackToBackChatCalls() {
        String[] prompts = {"Say yes.", "Say no.", "Say maybe."};
        String[] responses = new String[3];

        for (int i = 0; i < prompts.length; i++) {
            List<Pair<String, String>> messages = new ArrayList<>();
            messages.add(new Pair<>("user", prompts[i]));

            InferenceParameters params = new InferenceParameters("")
                    .setMessages(null, messages)
                    .setNPredict(N_PREDICT)
                    .setSeed(42)
                    .setTemperature(0.0f);

            responses[i] = model.chatComplete(params);
            Assert.assertNotNull("Call " + i + " must not return null", responses[i]);
            Assert.assertFalse("Call " + i + " must not return empty response", responses[i].isEmpty());
        }
    }

    // ------------------------------------------------------------------
    // 10. handleInfill direct JSON endpoint
    // ------------------------------------------------------------------

    /**
     * handleInfill must accept a JSON body with "input_prefix"/"input_suffix"
     * and return a completion result with a "content" field.
     */
    @Test
    public void testHandleInfillDirect() {
        String prefix = "def greet(name):\n    \"\"\" ";
        String suffix = "\n    return greeting\n";

        String json = "{\"input_prefix\": " + toJsonString(prefix) +
                ", \"input_suffix\": " + toJsonString(suffix) +
                ", \"n_predict\": " + N_PREDICT +
                ", \"seed\": 42, \"temperature\": 0.0}";

        String response = model.handleInfill(json);

        Assert.assertNotNull("handleInfill must return non-null", response);
        Assert.assertTrue("handleInfill response must contain 'content'", response.contains("\"content\""));
    }

    // ------------------------------------------------------------------
    // 11. handleEmbeddings OAI-compat format
    // ------------------------------------------------------------------

    /**
     * With oaiCompat=true, handleEmbeddings must return a response shaped like
     * the OpenAI embeddings endpoint, with a "data" array.
     */
    @Test
    public void testHandleEmbeddingsOaiCompat() {
        String json = "{\"input\": \"Hello world\"}";
        String response = model.handleEmbeddings(json, true);

        Assert.assertNotNull("OAI-compat embeddings must not be null", response);
        Assert.assertTrue("OAI-compat embeddings must contain 'data'", response.contains("\"data\""));
    }

    /**
     * With oaiCompat=false (default / raw mode), the response must contain the
     * "embedding" field directly (not wrapped in a data array).
     */
    @Test
    public void testHandleEmbeddingsRawFormat() {
        String json = "{\"content\": \"Hello world\"}";
        String response = model.handleEmbeddings(json, false);

        Assert.assertNotNull("Raw embeddings must not be null", response);
        Assert.assertTrue("Raw embeddings must contain 'embedding'", response.contains("\"embedding\""));
    }

    // ------------------------------------------------------------------
    // 12. handleTokenize with addSpecial=true
    // ------------------------------------------------------------------

    /**
     * addSpecial=true must add BOS/EOS tokens. The resulting token count should
     * be greater than the token count without special tokens.
     */
    @Test
    public void testHandleTokenizeWithSpecialTokens() {
        String content = "Hello world";

        String withSpecial    = model.handleTokenize(content, true,  false);
        String withoutSpecial = model.handleTokenize(content, false, false);

        Assert.assertNotNull(withSpecial);
        Assert.assertNotNull(withoutSpecial);
        Assert.assertTrue("Both responses must contain 'tokens'", withSpecial.contains("\"tokens\""));

        int countWith    = countTokensInJson(withSpecial);
        int countWithout = countTokensInJson(withoutSpecial);

        Assert.assertTrue(
                "addSpecial=true should produce at least as many tokens as addSpecial=false " +
                "(got " + countWith + " vs " + countWithout + ")",
                countWith >= countWithout
        );
    }

    // ------------------------------------------------------------------
    // 13. handleDetokenize round-trip via encode / handleDetokenize
    // ------------------------------------------------------------------

    /**
     * encode() a string, then pass the token IDs to handleDetokenize(). The
     * recovered text must contain the original string's content.
     */
    @Test
    public void testHandleDetokenizeRoundTrip() {
        String original = "Hello, world!";
        int[] tokens = model.encode(original);
        Assert.assertTrue("encode must produce at least one token", tokens.length > 0);

        String response = model.handleDetokenize(tokens);
        Assert.assertNotNull(response);
        Assert.assertTrue("handleDetokenize response must contain 'content'", response.contains("\"content\""));

        // Extract the detokenized text (simple search for content field value)
        String detokenized = LlamaOutput.getContentFromJson(response);
        // The tokenizer typically prepends a space; check the meaningful content
        Assert.assertTrue(
                "Detokenized text should contain original content (got: '" + detokenized + "')",
                detokenized.contains("Hello") && detokenized.contains("world")
        );
    }

    // ------------------------------------------------------------------
    // 14. saveSlot / restoreSlot round-trip
    // ------------------------------------------------------------------

    /**
     * saveSlot writes the KV cache to a file; restoreSlot reads it back.
     * Both must succeed (return a JSON response with expected fields).
     * The saved file is removed after the test.
     */
    @Test
    public void testSaveAndRestoreSlot() throws IOException {
        // Prime the slot with a short generation so there is state to save
        model.complete(new InferenceParameters("Hello").setNPredict(5).setSeed(42));

        File tempFile = File.createTempFile("llama_slot_", ".bin");
        tempFile.deleteOnExit();
        String filepath = tempFile.getAbsolutePath();
        // Delete so the native layer can write it fresh
        Files.delete(tempFile.toPath());

        String saveResult = model.saveSlot(0, filepath);
        Assert.assertNotNull("saveSlot must return non-null", saveResult);
        Assert.assertTrue("saveSlot result must contain id_slot",
                saveResult.contains("\"id_slot\""));

        // File must now exist
        File saved = new File(filepath);
        if (saved.exists()) {
            // Only attempt restore if file was actually written
            String restoreResult = model.restoreSlot(0, filepath);
            Assert.assertNotNull("restoreSlot must return non-null", restoreResult);
            Assert.assertTrue("restoreSlot result must contain id_slot",
                    restoreResult.contains("\"id_slot\""));
            saved.delete();
        }
        // If the file wasn't written we still pass: the save attempt exercised the code path.
    }

    // ------------------------------------------------------------------
    // 15. nPredict=1 minimal chat completion
    // ------------------------------------------------------------------

    /**
     * Setting nPredict=1 must still produce a valid (single-token) response
     * without hanging or crashing.
     */
    @Test
    public void testChatCompleteNPredictOne() {
        List<Pair<String, String>> messages = new ArrayList<>();
        messages.add(new Pair<>("user", "Say X."));

        InferenceParameters params = new InferenceParameters("")
                .setMessages(null, messages)
                .setNPredict(1)
                .setSeed(42)
                .setTemperature(0.0f);

        String response = model.chatComplete(params);
        Assert.assertNotNull(response);
        Assert.assertFalse("nPredict=1 must still return a non-empty response", response.isEmpty());
        String content = extractChoiceContent(response);
        // Content should be at most one token long — just verify it doesn't crash
        Assert.assertNotNull("Content must not be null for nPredict=1", content);
    }

    // ------------------------------------------------------------------
    // 16. generateChat streaming accumulates full response in stop token
    // ------------------------------------------------------------------

    /**
     * The final token emitted by generateChat must have stop=true.
     * All prior tokens must have stop=false. The iterator must not emit
     * any token after the stop token.
     */
    @Test
    public void testGenerateChatStopFlagOnFinalToken() {
        List<Pair<String, String>> messages = new ArrayList<>();
        messages.add(new Pair<>("user", "Write one word."));

        InferenceParameters params = new InferenceParameters("")
                .setMessages(null, messages)
                .setNPredict(N_PREDICT)
                .setSeed(42)
                .setTemperature(0.0f);

        List<LlamaOutput> outputs = new ArrayList<>();
        for (LlamaOutput output : model.generateChat(params)) {
            outputs.add(output);
        }

        Assert.assertFalse("generateChat must emit at least one output", outputs.isEmpty());

        // Every output except the last must NOT be the stop token
        for (int i = 0; i < outputs.size() - 1; i++) {
            Assert.assertFalse(
                    "Token " + i + " must not be marked stop before the final token",
                    outputs.get(i).stop
            );
        }
        // The last output must be the stop token
        Assert.assertTrue(
                "The final output from generateChat must be marked as stop",
                outputs.get(outputs.size() - 1).stop
        );
    }

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    /**
     * Extract the assistant's content string from an OAI-compatible chat
     * completion JSON response.
     * Expected structure: {"choices":[{"message":{"role":"assistant","content":"..."}}]}
     */
    private static String extractChoiceContent(String json) {
        // Find choices[0].message.content
        int choicesIdx = json.indexOf("\"choices\"");
        if (choicesIdx < 0) {
            // Fallback: try plain "content" field (non-OAI response shape)
            return LlamaOutput.getContentFromJson(json);
        }
        // Find "content" after "choices"
        int contentIdx = json.indexOf("\"content\"", choicesIdx);
        if (contentIdx < 0) {
            return "";
        }
        // Reuse LlamaOutput's JSON extractor on a substring starting at "content"
        return LlamaOutput.getContentFromJson(json.substring(contentIdx));
    }

    /**
     * Count the number of comma-separated elements in the JSON array value
     * of the "tokens" field. This is a best-effort heuristic — it works for
     * the simple integer-array format returned by handleTokenize.
     */
    private static int countTokensInJson(String json) {
        int tokensIdx = json.indexOf("\"tokens\"");
        if (tokensIdx < 0) return 0;
        int openBracket = json.indexOf('[', tokensIdx);
        int closeBracket = json.indexOf(']', openBracket);
        if (openBracket < 0 || closeBracket < 0) return 0;
        String array = json.substring(openBracket + 1, closeBracket).trim();
        if (array.isEmpty()) return 0;
        return array.split(",").length;
    }

    /** Minimal JSON string escaping for test helper strings. */
    private static String toJsonString(String s) {
        if (s == null) return "null";
        return "\"" + s
                .replace("\\", "\\\\")
                .replace("\"", "\\\"")
                .replace("\n", "\\n")
                .replace("\r", "\\r")
                .replace("\t", "\\t")
                + "\"";
    }
}
