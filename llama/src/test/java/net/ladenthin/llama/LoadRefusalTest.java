// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.not;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import net.ladenthin.llama.exception.LlamaException;
import net.ladenthin.llama.loader.NativeLibraryPresence;
import net.ladenthin.llama.parameters.ModelParameters;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

/**
 * Parameters llama.cpp refuses before it reads the model file. Each of them used to end the
 * process -- the JVM -- instead of failing the load; the test running on after the refusal is half
 * of what it asserts. No GGUF is needed: the refusals come before the model path is opened.
 */
@ClaudeGenerated(
        purpose = "Model-free guard that a refused --batch-size and a chat template llama.cpp cannot parse "
                + "fail the load with a LlamaException naming the reason instead of ending the JVM.")
class LoadRefusalTest {

    private static final String NO_MODEL = "/nonexistent/jllama-load-refusal.gguf";

    @BeforeEach
    void needsTheNativeLibrary() {
        assumeTrue(NativeLibraryPresence.onClasspath(), "libjllama not on classpath — skipping load refusals");
    }

    @Test
    void aBatchSizeBelowTwoIsRefusedByName() {
        // llama.cpp's load evaluates two tokens in one batch and aborted the process on a batch of 1
        LlamaException refused = assertThrows(
                LlamaException.class,
                () -> new LlamaModel(
                        new ModelParameters().setModel(NO_MODEL).setBatchSize(1).setFit(false)));
        assertThat(refused.getMessage(), containsString("--batch-size must be at least 2"));
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
}
