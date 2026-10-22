// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import net.ladenthin.llama.args.QuantizationType;
import net.ladenthin.llama.exception.LlamaException;
import net.ladenthin.llama.parameters.InferenceParameters;
import net.ladenthin.llama.parameters.ModelParameters;
import net.ladenthin.llama.value.ModelMeta;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * Integration tests for {@link LlamaQuantizer} against a real GGUF. Uses the small draft model
 * (AMD-Llama-135m, Q2_K) so the re-quantization round trip stays fast; the produced Q4_0 file is
 * then loaded and asked for a short completion, proving the output is a valid, loadable model.
 */
@ClaudeGenerated(
        purpose = "Exercise llama_model_quantize end to end over the JNI surface: successful "
                + "re-quantization producing a loadable GGUF, the default refusal to requantize "
                + "an already-quantized input, and the missing-input error path.")
public class QuantizerIntegrationTest {

    @TempDir
    static Path tempDir;

    private static void assumeDraftModel() {
        Assumptions.assumeTrue(
                new File(TestConstants.DRAFT_MODEL_PATH).exists(),
                "Draft model not found, skipping QuantizerIntegrationTest");
    }

    /**
     * Below this the Q4_0 output can be under the 10 MB the round trip asserts: Q4_0 stores 4.5 bits a
     * weight, so 100 M parameters are about 56 MB. CI's draft model, AMD-Llama-135m, has about 134 M.
     */
    private static final long MIN_PARAMETERS_FOR_SIZE_CHECK = 100_000_000L;

    /** The file types llama.cpp does not count as quantized ({@code llama_ftype_name}); the rest are. */
    private static final java.util.List<String> UNQUANTIZED_FTYPES = java.util.Arrays.asList("all F32", "F16", "BF16");

    /**
     * The metadata of the draft model as llama.cpp loads it: parameter count and file type, which a
     * GGUF need not declare in its key/value table (llama.cpp guesses the file type from the tensors).
     */
    private static ModelMeta draftModelMeta() {
        int gpuLayers = Integer.getInteger(TestConstants.PROP_TEST_NGL, TestConstants.DEFAULT_TEST_NGL);
        try (LlamaModel model = new LlamaModel(new ModelParameters()
                .setModel(TestConstants.DRAFT_MODEL_PATH)
                .setCtxSize(128)
                .setGpuLayers(gpuLayers)
                .setFit(false))) {
            return model.getModelMeta();
        }
    }

    private static boolean isQuantized(String ftype) {
        return !UNQUANTIZED_FTYPES.contains(ftype.replace("(guessed) ", ""));
    }

    @Test
    public void quantize_producesLoadableModel() throws Exception {
        assumeDraftModel();
        long parameters = draftModelMeta().getNParams();
        Assumptions.assumeTrue(
                parameters >= MIN_PARAMETERS_FOR_SIZE_CHECK,
                "needs CI's AMD-Llama-135m: the test asserts a Q4_0 output above 10 MB, this model has " + parameters
                        + " parameters");
        Path output = tempDir.resolve("draft-q4_0.gguf");

        LlamaQuantizer.quantize(TestConstants.DRAFT_MODEL_PATH, output.toString(), QuantizationType.Q4_0, 0, true);

        assertTrue(Files.exists(output), "quantized output must exist");
        assertTrue(Files.size(output) > 10_000_000L, "quantized 135M model should be well above 10 MB");

        int gpuLayers = Integer.getInteger(TestConstants.PROP_TEST_NGL, TestConstants.DEFAULT_TEST_NGL);
        try (LlamaModel model = new LlamaModel(new ModelParameters()
                .setModel(output.toString())
                .setCtxSize(128)
                .setGpuLayers(gpuLayers)
                .setFit(false))) {
            String completion = model.complete(
                    new InferenceParameters("def main():").withNPredict(4).withTemperature(0.0f));
            assertNotNull(completion, "quantized model must be able to complete");
        }
    }

    /** Re-quantizing an already-quantized GGUF without the explicit opt-in must fail loudly. */
    @Test
    public void quantize_requantizeWithoutOptIn_throws() {
        assumeDraftModel();
        String ftype = draftModelMeta().getFtype();
        Assumptions.assumeTrue(
                isQuantized(ftype),
                "needs CI's quantized draft model (Q2_K): the test asserts the refusal to requantize, and this "
                        + "model's file type is " + ftype);
        Path output = tempDir.resolve("draft-requant-refused.gguf");
        assertThrows(
                LlamaException.class,
                () -> LlamaQuantizer.quantize(
                        TestConstants.DRAFT_MODEL_PATH, output.toString(), QuantizationType.Q4_0));
    }

    @Test
    public void quantize_missingInput_throws() {
        assumeDraftModel(); // gates on the native lib being present in this environment
        Path output = tempDir.resolve("never-written.gguf");
        assertThrows(
                LlamaException.class,
                () -> LlamaQuantizer.quantize(
                        "models/does-not-exist.gguf", output.toString(), QuantizationType.Q8_0, 0, true));
    }
}
