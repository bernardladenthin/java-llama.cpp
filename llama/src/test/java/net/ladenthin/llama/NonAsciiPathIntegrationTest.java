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
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.stream.Stream;
import net.ladenthin.llama.args.QuantizationType;
import net.ladenthin.llama.parameters.InferenceParameters;
import net.ladenthin.llama.parameters.ModelParameters;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * The file paths of the Java API under a directory and a file name outside ASCII: the model, the slot
 * directory and slot file, and the quantizer's and importance matrix's inputs and outputs. Java hands
 * every path to the native side as UTF-8; on Windows the native side read some of them in the
 * process's active code page (patches/0024 to 0030, and the library's own streams), where a
 * character like the umlaut is mis-decoded and one like {@code 日} has no representation at all.
 * The C API tests (CApiNonAsciiPaths, CApiSlotFiles) hold the native sites; this test holds the
 * same paths through JNI. On Linux and macOS it passes trivially, on a Windows runner (code page
 * 1252) it is the measurement.
 *
 * <p>Not here: {@code --log-file}, whose file stays attached to the process-wide log after the model
 * is closed (a Windows temp directory holding it could not be removed), and the extraction directory,
 * which the loader has chosen before any test runs.</p>
 */
@ClaudeGenerated(
        purpose = "Pin the Java API's file paths under a non-ASCII directory and file name on every "
                + "platform: model load, slot save/restore into one file, re-quantization, importance "
                + "matrix computation and the legacy merge read back by the quantizer.")
public class NonAsciiPathIntegrationTest {

    private static final ObjectMapper JSON = new ObjectMapper();

    /** An umlaut (in code page 1252) and a CJK character (in no single-byte code page). */
    private static final String NON_ASCII = "ä-日";

    @TempDir
    Path tempDir;

    private static void assumeDraftModel() {
        Assumptions.assumeTrue(
                new File(TestConstants.DRAFT_MODEL_PATH).exists(),
                "Draft model not found, skipping NonAsciiPathIntegrationTest");
    }

    private static ModelParameters parameters(Path model) {
        int gpuLayers = Integer.getInteger(TestConstants.PROP_TEST_NGL, TestConstants.DEFAULT_TEST_NGL);
        return new ModelParameters()
                .setModel(model.toString())
                .setCtxSize(128)
                .setGpuLayers(gpuLayers)
                .setFit(false);
    }

    /** A directory outside ASCII holding a copy of the draft model under a name outside ASCII. */
    private Path modelUnderNonAsciiPath() throws IOException {
        Path dir = Files.createDirectories(tempDir.resolve("d-" + NON_ASCII));
        Path model = dir.resolve("m-" + NON_ASCII + ".gguf");
        Files.copy(new File(TestConstants.DRAFT_MODEL_PATH).toPath(), model);
        return model;
    }

    private static List<String> namesIn(Path dir) throws IOException {
        List<String> names = new ArrayList<>();
        try (Stream<Path> entries = Files.list(dir)) {
            entries.forEach(p -> names.add(p.getFileName().toString()));
        }
        return names;
    }

    private static void assertAnswers(Path model) {
        try (LlamaModel loaded = new LlamaModel(parameters(model))) {
            String completion = loaded.complete(
                    new InferenceParameters("def main():").withNPredict(4).withTemperature(0.0f));
            assertFalse(completion.isEmpty(), "the model under " + model + " gave no completion");
        }
    }

    @Test
    public void aModelIsLoadedFromANonAsciiPath() throws IOException {
        assumeDraftModel();
        assertAnswers(modelUnderNonAsciiPath());
    }

    /**
     * Both halves of a slot path: a slot directory outside ASCII with an ASCII file name, and a file
     * name outside ASCII. The slot file has to be one file of the size the save reports, and the
     * restore has to read all of it.
     */
    @Test
    public void aSlotIsSavedAndRestoredUnderANonAsciiPath() throws IOException {
        assumeDraftModel();
        Path model = modelUnderNonAsciiPath();
        String[][] places = {{"slots-" + NON_ASCII, "slot.bin"}, {"slots-ascii", "s-" + NON_ASCII + ".bin"}};
        for (String[] place : places) {
            Path dir = Files.createDirectories(tempDir.resolve(place[0]));
            String name = place[1];
            try (LlamaModel loaded = new LlamaModel(parameters(model).setSlotSavePath(dir.toString()))) {
                loaded.complete(
                        new InferenceParameters("def main():").withNPredict(8).withTemperature(0.0f));
                JsonNode saved = JSON.readTree(loaded.saveSlot(0, name));
                assertEquals(Collections.singletonList(name), namesIn(dir), "the slot file must be one file");
                long written = saved.path("n_written").asLong();
                assertEquals(written, Files.size(dir.resolve(name)), saved.toString());

                JsonNode restored = JSON.readTree(loaded.restoreSlot(0, name));
                assertEquals(written, restored.path("n_read").asLong(), restored.toString());
                assertEquals(
                        saved.path("n_saved").asInt(),
                        restored.path("n_restored").asInt(),
                        restored.toString());
            }
        }
    }

    /**
     * The quantizer and the importance matrix read and write under the non-ASCII path: a
     * re-quantization, an importance matrix from a calibration text there, its legacy merge, and a
     * quantization that reads the legacy file. Each output is a file under its own name, and the
     * quantized model loads and answers.
     */
    @Test
    public void theQuantizerAndTheImportanceMatrixWorkUnderANonAsciiPath() throws IOException {
        assumeDraftModel();
        Path model = modelUnderNonAsciiPath();
        Path dir = model.getParent();

        Path q4 = dir.resolve("q4-" + NON_ASCII + ".gguf");
        LlamaQuantizer.quantize(model.toString(), q4.toString(), QuantizationType.Q4_0, 0, true);
        assertTrue(Files.size(q4) > 0, "no quantized model under its own name, found " + namesIn(dir));
        assertAnswers(q4);
        Files.delete(q4);

        Path text = dir.resolve("calibration-" + NON_ASCII + ".txt");
        StringBuilder words = new StringBuilder();
        for (int i = 0; i < 200; i++) {
            words.append("def add(a, b):\n    return a + b\n\nprint(add(")
                    .append(i)
                    .append(", 1))\n\n");
        }
        Files.write(text, words.toString().getBytes(StandardCharsets.UTF_8));
        Path imatrix = dir.resolve("imatrix-" + NON_ASCII + ".gguf");
        int chunks = LlamaQuantizer.computeImportanceMatrix(
                model.toString(), text.toString(), imatrix.toString(), 64, 2, 0, 0);
        assertEquals(2, chunks);
        assertTrue(Files.size(imatrix) > 0, "no importance matrix under its own name, found " + namesIn(dir));

        Path legacy = dir.resolve("imatrix-" + NON_ASCII + ".dat");
        assertEquals(
                chunks,
                LlamaQuantizer.mergeImportanceMatrices(
                        Collections.singletonList(imatrix.toString()), legacy.toString(), true));
        assertTrue(Files.size(legacy) > 0, "no legacy importance matrix under its own name, found " + namesIn(dir));

        Path q4WithImatrix = dir.resolve("q4-imatrix-" + NON_ASCII + ".gguf");
        LlamaQuantizer.quantize(
                model.toString(), q4WithImatrix.toString(), QuantizationType.Q4_0, 0, true, legacy.toString());
        assertTrue(Files.size(q4WithImatrix) > 0, "no quantized model under its own name, found " + namesIn(dir));
    }
}
