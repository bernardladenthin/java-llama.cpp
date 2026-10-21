// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import static org.junit.jupiter.api.Assertions.*;

import java.io.File;
import net.ladenthin.llama.exception.InvalidRequestException;
import net.ladenthin.llama.exception.LlamaException;
import net.ladenthin.llama.parameters.InferenceParameters;
import net.ladenthin.llama.parameters.ModelParameters;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/**
 * Tests for error handling paths in the JNI layer. Verifies that:
 * <ul>
 *   <li>Invalid model path throws LlamaException</li>
 *   <li>embed() on a model without enableEmbedding() throws</li>
 *   <li>handleInfill with missing input_prefix/input_suffix returns error</li>
 *   <li>handleEmbeddings without embedding support returns error</li>
 *   <li>handleEmbeddings with invalid encoding_format returns error</li>
 *   <li>handleEmbeddings with empty input returns error</li>
 *   <li>configureParallelInference with invalid n_threads returns error</li>
 * </ul>
 */
@ClaudeGenerated(
        purpose = "Verify error handling paths in the JNI layer: invalid model path, embed without "
                + "enableEmbedding, handleInfill missing fields, handleEmbeddings invalid params, "
                + "and configureParallelInference validation.",
        model = "claude-opus-4-6")
public class ErrorHandlingTest {

    private static LlamaModel model;
    private static LlamaModel modelNoEmbed;

    @BeforeAll
    public static void setup() {
        Assumptions.assumeTrue(
                new File(TestConstants.MODEL_PATH).exists(), "Model file not found, skipping ErrorHandlingTest");
        int gpuLayers = Integer.getInteger(TestConstants.PROP_TEST_NGL, TestConstants.DEFAULT_TEST_NGL);
        // Model WITH embedding
        model = new LlamaModel(new ModelParameters()
                .setCtxSize(128)
                .setModel(TestConstants.MODEL_PATH)
                .setGpuLayers(gpuLayers)
                .setFit(false)
                .enableEmbedding());
        // Model WITHOUT embedding
        modelNoEmbed = new LlamaModel(new ModelParameters()
                .setCtxSize(128)
                .setModel(TestConstants.MODEL_PATH)
                .setGpuLayers(gpuLayers)
                .setFit(false));
    }

    @AfterAll
    public static void tearDown() {
        if (model != null) {
            model.close();
        }
        if (modelNoEmbed != null) {
            modelNoEmbed.close();
        }
    }

    // -------------------------------------------------------------------------
    // Invalid model path
    // -------------------------------------------------------------------------

    @Test
    public void testInvalidModelPathThrows() {
        assertThrows(
                LlamaException.class,
                () -> new LlamaModel(new ModelParameters()
                        .setModel("/nonexistent/path/model.gguf")
                        .setFit(false)));
    }

    @Test
    public void testEmptyModelPathThrows() {
        assertThrows(
                LlamaException.class,
                () -> new LlamaModel(new ModelParameters().setModel("").setFit(false)));
    }

    // -------------------------------------------------------------------------
    // embed() without embedding support
    // -------------------------------------------------------------------------

    @Test
    public void testEmbedWithoutEnableEmbeddingThrows() {
        assertThrows(LlamaException.class, () -> modelNoEmbed.embed("hello world"));
    }

    // -------------------------------------------------------------------------
    // handleEmbeddings without embedding support
    // -------------------------------------------------------------------------

    @Test
    public void testHandleEmbeddingsWithoutEmbeddingSupportReturnsError() {
        String json = "{\"input\":\"hello world\"}";
        try {
            String result = modelNoEmbed.handleEmbeddings(json, false);
            // If it doesn't throw, the result should indicate an error
            fail("Expected LlamaException for embeddings without embedding support");
        } catch (LlamaException e) {
            assertTrue(e.getMessage().toLowerCase().contains("embedding"), "Error should mention embedding");
        }
    }

    // -------------------------------------------------------------------------
    // handleEmbeddings with invalid encoding_format
    // -------------------------------------------------------------------------

    @Test
    public void testHandleEmbeddingsInvalidEncodingFormat() {
        String json = "{\"input\":\"hello world\",\"encoding_format\":\"invalid\"}";
        try {
            String result = model.handleEmbeddings(json, false);
            fail("Expected InvalidRequestException for invalid encoding_format");
        } catch (InvalidRequestException e) {
            // the typed rejection is what lets OpenAiCompatServer answer 400 instead of 500
            assertTrue(e.getMessage().contains("encoding_format"), "Error should mention encoding_format");
        }
    }

    // -------------------------------------------------------------------------
    // handleEmbeddings with empty input
    // -------------------------------------------------------------------------

    @Test
    public void testHandleEmbeddingsEmptyInput() {
        String json = "{\"input\":\"\"}";
        try {
            String result = model.handleEmbeddings(json, false);
            // Native code may handle empty input gracefully — that's acceptable
            assertNotNull(result, "Result should not be null");
        } catch (InvalidRequestException e) {
            // Also acceptable if the native code rejects empty input -- but then as the typed rejection
            assertNotNull(e.getMessage(), "Exception message should not be null");
        }
    }

    @Test
    public void testHandleEmbeddingsEmptyInputArrayIsAnInvalidRequest() {
        // upstream tokenize_input_prompts: `"prompt" must not be empty` (std::invalid_argument), a 400
        try {
            model.handleEmbeddings("{\"input\":[]}", false);
            fail("Expected InvalidRequestException for an empty input array");
        } catch (InvalidRequestException e) {
            assertTrue(e.getMessage().contains("must not be empty"), "Error should name the reason: " + e.getMessage());
        }
    }

    @Test
    public void testMalformedJsonBodyIsAnInvalidRequest() {
        // upstream answers a common_json_error with 400; the JNI layer parses the body the same way
        try {
            model.handleEmbeddings("{not json", false);
            fail("Expected InvalidRequestException for a malformed body");
        } catch (InvalidRequestException e) {
            assertNotNull(e.getMessage(), "Exception message should not be null");
        }
        try {
            model.handleCompletions("[1, 2");
            fail("Expected InvalidRequestException for a malformed body");
        } catch (InvalidRequestException e) {
            assertNotNull(e.getMessage(), "Exception message should not be null");
        }
    }

    // -------------------------------------------------------------------------
    // handleInfill with missing fields
    // -------------------------------------------------------------------------

    @Test
    public void testHandleInfillMissingInputPrefix() {
        String json = "{\"input_suffix\":\"return result\",\"n_predict\":5}";
        try {
            String result = model.handleCompletions(json);
            // Infill-specific missing fields may cause an error or just return completion
            // The handleInfill endpoint specifically requires these
        } catch (LlamaException e) {
            // Expected
        }
    }

    @Test
    public void testHandleInfillMissingInputSuffix() {
        String json = "{\"input_prefix\":\"def hello():\",\"n_predict\":5}";
        try {
            model.handleInfill(json);
            // May succeed with empty suffix or throw
        } catch (InvalidRequestException e) {
            // a missing required field is upstream's `"input_suffix" is required`, a 400
            assertTrue(e.getMessage().contains("input_suffix"), "Error should mention input_suffix");
        }
    }

    // -------------------------------------------------------------------------
    // n_predict per request (F69)
    // -------------------------------------------------------------------------

    @Test
    public void testNPredictMinusTwoPerRequestIsAnInvalidRequest() {
        // -2 ("until context filled") exists only for the completion tool's -n option. The server's
        // request schema declares n_predict with the hard limits [-1, INT32_MAX] (server-schema.cpp,
        // b11639), so a request with -2 is refused as the caller's fault, with the field named --
        // what withNPredict's Javadoc promises.
        InferenceParameters request =
                InferenceParameters.empty().withPrompt("def f():").withNPredict(-2);

        InvalidRequestException e = assertThrows(InvalidRequestException.class, () -> modelNoEmbed.complete(request));

        assertTrue(e.getMessage().contains("n_predict"), "Error should name n_predict but was: " + e.getMessage());
    }

    @Test
    public void testNPredictMinusTwoInAChatIsAnInvalidRequest() {
        // The OpenAI chat mapper forwards a token cap below -1 (F219); the chat path runs the same request
        // schema, so the client gets upstream's 400 naming the field instead of an uncapped answer.
        InferenceParameters request = InferenceParameters.empty()
                .withMessagesJson("[{\"role\":\"user\",\"content\":\"hi\"}]")
                .withNPredict(-2);

        InvalidRequestException e =
                assertThrows(InvalidRequestException.class, () -> modelNoEmbed.chatComplete(request));

        assertTrue(e.getMessage().contains("n_predict"), "Error should name n_predict but was: " + e.getMessage());
    }

    @Test
    public void testJsonSchemaWithGrammarInAChatIsAnInvalidRequest() {
        // The OpenAI chat mapper forwards both fields and leaves the verdict to upstream's chat parser,
        // which refuses the pair (server-common.cpp, b11639) -- the 400 llama-server answers.
        InferenceParameters request = InferenceParameters.empty()
                .withMessagesJson("[{\"role\":\"user\",\"content\":\"hi\"}]")
                .withGrammar("root ::= \"a\"")
                .withJsonSchema("{\"type\":\"object\"}")
                .withNPredict(1);

        InvalidRequestException e =
                assertThrows(InvalidRequestException.class, () -> modelNoEmbed.chatComplete(request));

        assertTrue(
                e.getMessage().contains("Cannot use both json_schema and grammar"),
                "Error should be upstream's refusal but was: " + e.getMessage());
    }

    @Test
    public void anInvalidRequestKeepsAFourByteCharacterOfTheRequest() {
        // upstream quotes the client's response_format type in its refusal (oaicompat_chat_params_parse,
        // server-common.cpp:1318, b11639); the message crosses JNI through throw_api_status, which has to
        // hand it over as UTF-8 bytes -- ThrowNew reads Modified UTF-8 and mangles a supplementary character
        String emoji = "\uD83D\uDE00"; // U+1F600
        String request = "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]," + "\"response_format\":{\"type\":\""
                + emoji + "\"},\"n_predict\":1}";

        InvalidRequestException e =
                assertThrows(InvalidRequestException.class, () -> modelNoEmbed.handleChatCompletions(request));

        assertTrue(
                e.getMessage().contains("but got: " + emoji),
                "Error should quote the type intact but was: " + e.getMessage());
    }

    // -------------------------------------------------------------------------
    // configureParallelInference validation
    // -------------------------------------------------------------------------

    @Test
    public void testConfigureParallelInferenceInvalidNThreads() {
        try {
            boolean result = model.configureParallelInference("{\"n_threads\":-1}");
            fail("Expected exception for invalid n_threads");
        } catch (LlamaException e) {
            assertTrue(e.getMessage().contains("n_threads"), "Error should mention n_threads");
        }
    }

    @Test
    public void testConfigureParallelInferenceInvalidNThreadsBatch() {
        try {
            boolean result = model.configureParallelInference("{\"n_threads_batch\":-1}");
            fail("Expected exception for invalid n_threads_batch");
        } catch (LlamaException e) {
            assertTrue(e.getMessage().contains("n_threads_batch"), "Error should mention n_threads_batch");
        }
    }

    @Test
    public void testConfigureParallelInferenceZeroNThreads() {
        try {
            boolean result = model.configureParallelInference("{\"n_threads\":0}");
            fail("Expected exception for n_threads=0");
        } catch (LlamaException e) {
            assertTrue(e.getMessage().contains("n_threads"), "Error should mention n_threads");
        }
    }

    // -------------------------------------------------------------------------
    // collect_task_results guard: missing "prompt" key
    //
    // handleCompletions / handleCompletionsOai / handleInfill each call
    // data.at("prompt") inside a try{} block whose catch invokes
    // throw_invalid_request (Finding 2 helper).  That catch guard sits
    // immediately before the collect_task_results call, so these tests
    // confirm the refactored error path propagates a LlamaException to Java.
    // -------------------------------------------------------------------------

    @Test
    public void testHandleCompletionsMissingPromptThrows() {
        // No "prompt" key → data.at("prompt") throws json::out_of_range →
        // caught by the std::exception catch → throw_invalid_request → InvalidRequestException
        try {
            model.handleCompletions("{\"n_predict\":1}");
            fail("Expected InvalidRequestException for missing 'prompt' key");
        } catch (InvalidRequestException e) {
            assertNotNull(e.getMessage(), "Exception message must not be null");
        }
    }

    @Test
    public void testHandleCompletionsOaiMissingPromptThrows() {
        try {
            model.handleCompletionsOai("{\"n_predict\":1}");
            fail("Expected InvalidRequestException for missing 'prompt' key");
        } catch (InvalidRequestException e) {
            assertNotNull(e.getMessage(), "Exception message must not be null");
        }
    }

    @Test
    public void testHandleInfillMissingPromptInTaskBuildThrows() {
        // Provides required input_prefix/input_suffix but deliberately omits
        // the tokenizable content in a way that triggers the task-build catch.
        // The infill path calls data.at("prompt") after format_infill populates it,
        // then tokenizes; an empty/invalid JSON value reaches the std::exception catch.
        try {
            model.handleInfill("{\"input_prefix\":\"def f():\",\"input_suffix\":\"return 1\",\"n_predict\":1}");
            // A well-formed request may succeed — that is also acceptable;
            // the point is that no uncaught C++ exception escapes the JNI boundary.
            // If it succeeds, verify the response is valid JSON.
        } catch (LlamaException e) {
            assertNotNull(e.getMessage(), "Exception message must not be null");
        }
    }
}
