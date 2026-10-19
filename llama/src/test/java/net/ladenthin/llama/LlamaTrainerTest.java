// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.is;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.nio.file.InvalidPathException;
import java.nio.file.Path;
import net.ladenthin.llama.exception.LlamaException;
import net.ladenthin.llama.loader.NativeLibraryPresence;
import net.ladenthin.llama.parameters.TrainingParameters;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * Model-free checks of the trainer's JNI shell. Needs {@code libjllama} and skips without it.
 */
class LlamaTrainerTest {

    /**
     * The trainer's error message comes back to Java as standard UTF-8. The shell returned it with
     * {@code NewStringUTF}, which reads Modified UTF-8: on HotSpot 21 the 4-byte sequence of U+1F600
     * in the quoted path came out as the four characters U+00F0 U+009F U+0098 U+0080, and the text
     * after it was cut short ({@code .gguf} became {@code .g}); Android's CheckJNI aborts the process
     * on the same input (F181). Skips where the file-name encoding cannot represent the path.
     */
    @Test
    void aFourByteCharacterInAPathComesBackIntactInTheErrorMessage(@TempDir Path tmp) {
        assumeTrue(NativeLibraryPresence.onClasspath(), "libjllama not on the classpath");
        Path model;
        try {
            model = tmp.resolve("missing-\uD83D\uDE00.gguf"); // U+1F600
        } catch (InvalidPathException e) {
            assumeTrue(false, "the file-name encoding cannot represent U+1F600: " + e.getMessage());
            return;
        }
        Path missing = model;

        LlamaException e = assertThrows(
                LlamaException.class,
                () -> LlamaTrainer.finetune(TrainingParameters.builder()
                        .modelPath(missing)
                        .trainingText("x")
                        .outputPath(tmp.resolve("out.gguf"))
                        .build()));

        assertThat(e.getMessage(), is("failed to load model for training: " + missing));
    }

    /**
     * The configuration crosses with {@code GetStringUTFChars}, which hands out Modified UTF-8, and
     * that is safe only because Jackson writes a supplementary-plane character as an escaped
     * surrogate pair (two {@code \\u} escapes), so the JSON is ASCII where the two encodings differ.
     * Should Jackson ever write the character raw, the native parser would receive two 3-byte
     * surrogates, reject them as ill-formed UTF-8 and fail with {@code invalid training config}
     * before looking at the model -- this test is what would say so.
     */
    @Test
    void aFourByteCharacterInTheTrainingTextReachesTheTrainer(@TempDir Path tmp) {
        assumeTrue(NativeLibraryPresence.onClasspath(), "libjllama not on the classpath");
        Path model = tmp.resolve("missing.gguf");

        LlamaException e = assertThrows(
                LlamaException.class,
                () -> LlamaTrainer.finetune(TrainingParameters.builder()
                        .modelPath(model)
                        .trainingText("Hallo \uD83D\uDE00 Welt") // U+1F600
                        .outputPath(tmp.resolve("out.gguf"))
                        .build()));

        assertThat(e.getMessage(), is("failed to load model for training: " + model));
    }
}
