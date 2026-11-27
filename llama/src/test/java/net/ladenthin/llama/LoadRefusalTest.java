// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.not;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import net.ladenthin.llama.exception.LlamaException;
import net.ladenthin.llama.loader.NativeLibraryPresence;
import net.ladenthin.llama.parameters.ModelParameters;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * Parameters llama.cpp refuses before it reads the model file. A batch of no tokens and a chat
 * template that does not parse used to end the process -- the JVM -- instead of failing the load,
 * so the test running on after the refusal is half of what they assert; an option's refused value
 * failed the load before too, but without its reason. Every refusal is a plain {@link LlamaException} (the C
 * API's {@code JLLAMA_STATUS_ERROR}), not the subclass for an invalid request. No GGUF is needed:
 * the refusals come before the model path is opened.
 */
@ClaudeGenerated(
        purpose = "Model-free guard that a --batch-size of 0 and a chat template llama.cpp cannot parse "
                + "fail the load with a LlamaException naming the reason instead of ending the JVM, and that "
                + "a file the loader cannot read names the loader's reason.")
class LoadRefusalTest {

    private static final String NO_MODEL = "/nonexistent/jllama-load-refusal.gguf";

    @BeforeEach
    void needsTheNativeLibrary() {
        assumeTrue(NativeLibraryPresence.onClasspath(), "libjllama not on classpath — skipping load refusals");
    }

    @Test
    void aBatchOfNoTokensIsRefusedByName() {
        // a context whose batch holds no token aborts the process at its first decode
        LlamaException refused = assertThrows(
                LlamaException.class,
                () -> new LlamaModel(
                        new ModelParameters().setModel(NO_MODEL).setBatchSize(0).setFit(false)));
        assertThat(refused.getMessage(), containsString("--batch-size must be at least 1"));
        assertEquals(LlamaException.class, refused.getClass(), "the status of a refused parse");
    }

    @Test
    void aChatTemplateThatDoesNotParseIsRefusedWithItsReason() {
        // the template is verified after the arguments are read; upstream called exit(1) there
        LlamaException refused = assertThrows(
                LlamaException.class,
                () -> new LlamaModel(new ModelParameters()
                        .setModel(NO_MODEL)
                        .setChatTemplate("{% if %}")
                        .setFit(false)));
        assertThat(refused.getMessage(), containsString("Failed to parse model parameters"));
        assertThat(refused.getMessage(), containsString("the supplied chat template is not supported"));
        assertThat(refused.getMessage(), not(containsString("usage:")));
        assertEquals(LlamaException.class, refused.getClass());
    }

    @Test
    void aValueAnOptionCannotTakeIsNamedWithTheOption() {
        LlamaException refused = assertThrows(
                LlamaException.class,
                () -> new LlamaModel(new ModelParameters()
                        .setModel(NO_MODEL)
                        .setOverrideKv("no-type-and-no-value")
                        .setFit(false)));
        assertThat(
                refused.getMessage(),
                containsString("Failed to parse model parameters: error while handling argument \"--override-kv\""));
        assertThat(refused.getMessage(), not(containsString("usage:")));
    }

    @Test
    void aFileThatIsNoModelFailsWithTheLoadersReason(@TempDir Path dir) throws IOException {
        // llama.cpp's loader only logs why a file did not load; the message carries that reason now
        Path notAModel = dir.resolve("not-a-model.gguf");
        Files.write(notAModel, "this is no GGUF file".getBytes(StandardCharsets.UTF_8));
        LlamaException refused = assertThrows(
                LlamaException.class,
                () -> new LlamaModel(
                        new ModelParameters().setModel(notAModel.toString()).setFit(false)));
        assertThat(
                refused.getMessage(),
                containsString("could not load model from given file path: llama_model_loader: failed to load model"));
        assertThat(refused.getMessage(), containsString("not-a-model.gguf"));
        assertEquals(LlamaException.class, refused.getClass());
    }
}
