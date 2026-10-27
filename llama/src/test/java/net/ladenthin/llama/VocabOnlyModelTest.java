// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.greaterThan;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.startsWith;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.io.File;
import java.util.Collections;
import net.ladenthin.llama.exception.LlamaException;
import net.ladenthin.llama.parameters.InferenceParameters;
import net.ladenthin.llama.parameters.ModelParameters;
import net.ladenthin.llama.value.Pair;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.function.Executable;

/**
 * A model loaded with {@link ModelParameters#setVocabOnly()} answers the model meta, {@code encode}
 * and {@code decode}, and refuses every operation that needs the inference server with a
 * {@link LlamaException} naming vocab-only mode (F266). Before, such an operation reached the
 * never-loaded server context: {@code applyTemplate} and {@code complete} ended the JVM with a
 * {@code SIGSEGV} in {@code llama_vocab::token_bos()}, {@code getMetrics} waited forever.
 */
@ClaudeGenerated(
        purpose = "Pin that a vocab-only model refuses the server operations with a LlamaException "
                + "instead of crashing or hanging the JVM, and still tokenizes.")
public class VocabOnlyModelTest {

    private static final String REFUSAL = "is not available in vocab-only mode";

    private static LlamaModel model;

    @BeforeAll
    public static void setup() {
        Assumptions.assumeTrue(new File(TestConstants.DRAFT_MODEL_PATH).exists(), "Draft model file not found");
        model = new LlamaModel(
                new ModelParameters().setModel(TestConstants.DRAFT_MODEL_PATH).setVocabOnly());
    }

    @AfterAll
    public static void tearDown() {
        if (model != null) {
            model.close();
        }
    }

    private static InferenceParameters chat() {
        return new InferenceParameters("")
                .withMessages(null, Collections.singletonList(new Pair<>("user", "Hi")))
                .withNPredict(2);
    }

    private static void assertRefused(Executable call) {
        LlamaException e = assertThrows(LlamaException.class, call);
        assertThat(e.getMessage(), containsString(REFUSAL));
        assertThat(e.getMessage(), containsString("--vocab-only"));
    }

    @Test
    public void theTokenizerOperationsAnswer() {
        int[] tokens = model.encode("Hello");
        assertThat(tokens.length, is(greaterThan(0)));
        assertThat(model.decode(tokens), containsString("Hello"));
        assertThat(model.getModelMeta().getNVocab(), is(greaterThan(0)));
    }

    @Test
    public void theMetaClaimsNoCapability() {
        // Without a server context the model can only tokenize: it refuses chat, so it must not
        // report the chat capability either.
        assertThat(model.getModelMeta().supportsChat(), is(false));
        assertThat(model.getModelMeta().supportsAudioGeneration(), is(false));
        assertThat(model.getModelMeta().supportsAudioStream(), is(false));
    }

    @Test
    public void applyTemplateIsRefused() {
        assertRefused(() -> model.applyTemplate(chat()));
    }

    @Test
    public void completeIsRefused() {
        assertRefused(() -> model.complete(new InferenceParameters("Hi").withNPredict(2)));
    }

    @Test
    public void chatCompleteIsRefused() {
        assertRefused(() -> model.chatComplete(chat()));
    }

    @Test
    public void getMetricsIsRefused() {
        assertRefused(() -> model.getMetrics());
    }

    @Test
    public void eraseSlotIsRefused() {
        assertRefused(() -> model.eraseSlot(0));
    }

    @Test
    public void setLoraAdapterIsRefused() {
        assertRefused(() -> model.setLoraAdapter(0, 1.0f));
    }

    @Test
    public void embedIsRefused() {
        assertRefused(() -> model.embed("Hi"));
    }

    @Test
    public void theRefusalNamesTheOperation() {
        LlamaException e = assertThrows(LlamaException.class, () -> model.applyTemplate(chat()));
        assertThat(e.getMessage(), startsWith("jllama_apply_template " + REFUSAL));
    }
}
