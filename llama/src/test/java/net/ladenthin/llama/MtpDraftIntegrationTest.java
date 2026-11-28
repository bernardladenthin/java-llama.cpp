// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import net.ladenthin.llama.args.SpecType;
import net.ladenthin.llama.parameters.InferenceParameters;
import net.ladenthin.llama.parameters.ModelParameters;
import net.ladenthin.llama.value.CompletionResult;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/**
 * Speculative decoding with a model's own multi-token-prediction head ({@code --spec-type draft-mtp}),
 * over a model whose head was trained: llama.cpp's server CI fixture {@code small-test} (b12838,
 * #28815), a 95M Qwen3.5 -- gated delta net layers, so every rejected draft is rolled back through the
 * recurrent state. The C API tests (CApiMtp, CApiQwen35) hold the same paths over random weights, whose
 * drafts are all rejected; here the head proposes tokens the model takes, and the answer must still be
 * the greedy answer of the model alone, token for token, which is upstream's own check of the fixture.
 */
@ClaudeGenerated(
        purpose = "Pin that a model's trained MTP head drafts tokens the model accepts through JNI, and that "
                + "the greedy answer with the drafts is the answer without them, token for token.")
public class MtpDraftIntegrationTest {

    private static final String PROMPT = "<|im_start|>user\nList three colors.<|im_end|>\n<|im_start|>assistant\n";

    private static final int N_PREDICT = 48;

    private static final ObjectMapper MAPPER = new ObjectMapper();

    @BeforeAll
    static void needsTheMtpModel() {
        Assumptions.assumeTrue(
                new File(TestConstants.MTP_MODEL_PATH).exists(),
                "MTP model not found, skipping MtpDraftIntegrationTest");
    }

    /**
     * On the CPU, as upstream's own check runs: a GPU backend verifies a draft in one batch and generates
     * without one token at a time, with kernels that may round a near tie the other way, and the comparison
     * is token for token. On the CPU the two paths agree to 1e-4 in every log probability (measured), against
     * a smallest margin of 0.06 between the first two tokens over the 48.
     */
    private static ModelParameters parameters() {
        return new ModelParameters()
                .setModel(TestConstants.MTP_MODEL_PATH)
                .setCtxSize(1024)
                .setGpuLayers(0)
                .setFit(false);
    }

    private static InferenceParameters request() {
        return new InferenceParameters(PROMPT)
                .withNPredict(N_PREDICT)
                .withTemperature(0.0f)
                .withTopK(1)
                .withReturnTokens(true)
                .withCachePrompt(false);
    }

    private static List<Integer> tokens(CompletionResult result) throws IOException {
        JsonNode tokens = MAPPER.readTree(result.getRawJson()).path("tokens");
        assertTrue(tokens.isArray(), "the answer carries its tokens: " + result.getRawJson());
        List<Integer> ids = new ArrayList<>();
        for (JsonNode id : tokens) {
            ids.add(id.asInt());
        }
        return ids;
    }

    @Test
    void theTrainedHeadDraftsTokensTheModelTakesAndTheAnswerStaysTheModelsOwn() throws IOException {
        CompletionResult alone;
        try (LlamaModel model = new LlamaModel(parameters())) {
            alone = model.completeWithStats(request());
        }
        assertFalse(alone.getText().isEmpty());
        assertEquals(0, alone.getTimings().getDraftN(), "without a speculative type the head drafts nothing");

        CompletionResult drafted;
        try (LlamaModel model = new LlamaModel(parameters().setSpecTypes(SpecType.DRAFT_MTP))) {
            drafted = model.completeWithStats(request());
        }
        assertTrue(drafted.getTimings().getDraftN() > 0, "the MTP head drafts: " + drafted.getRawJson());
        assertTrue(
                drafted.getTimings().getDraftNAccepted() > 0,
                "a trained head proposes tokens the model takes: " + drafted.getRawJson());
        assertEquals(tokens(alone), tokens(drafted));
        assertEquals(alone.getText(), drafted.getText());
    }
}
