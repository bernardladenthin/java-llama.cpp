// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.File;
import net.ladenthin.llama.exception.InvalidRequestException;
import net.ladenthin.llama.parameters.ModelParameters;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/**
 * {@link LlamaModel#decode(int...)} of a token id the vocabulary does not have is the caller's fault: an
 * {@link InvalidRequestException} naming the id. Until 2026-11-19 it was a plain {@code LlamaException}
 * with the C++ library's {@code vector::_M_range_check} message, which named neither the id nor the
 * vocabulary. The model and a vocab-only model answer the next call. The C API test is
 * {@code CApi.ATokenIdOutsideTheVocabularyIsTheCallersFaultAndTheModelGoesOn}.
 */
@ClaudeGenerated(
        purpose = "Pin that decoding a token id outside the vocabulary is an InvalidRequestException naming it, "
                + "for a full and a vocab-only model, and that the model goes on.")
public class DecodeOutOfVocabularyIntegrationTest {

    @BeforeAll
    static void needsTheDraftModel() {
        Assumptions.assumeTrue(
                new File(TestConstants.DRAFT_MODEL_PATH).exists(),
                "Draft model not found, skipping DecodeOutOfVocabularyIntegrationTest");
    }

    private static void expectRefusedAndGoOn(LlamaModel model) {
        for (int id : new int[] {10_000_000, -1}) {
            InvalidRequestException e = assertThrows(InvalidRequestException.class, () -> model.decode(id));
            assertTrue(e.getMessage().contains(String.valueOf(id)), e.getMessage());
            assertTrue(e.getMessage().contains("outside the vocabulary"), e.getMessage());
        }
        assertEquals("Hello", model.decode(model.encode("Hello")).trim());
    }

    @Test
    public void aTokenIdOutsideTheVocabularyIsRefusedByAFullModel() {
        try (LlamaModel model = new LlamaModel(new ModelParameters()
                .setModel(TestConstants.DRAFT_MODEL_PATH)
                .setCtxSize(128)
                .setGpuLayers(0)
                .setFit(false))) {
            expectRefusedAndGoOn(model);
        }
    }

    @Test
    public void aTokenIdOutsideTheVocabularyIsRefusedByAVocabOnlyModel() {
        try (LlamaModel model = new LlamaModel(
                new ModelParameters().setModel(TestConstants.DRAFT_MODEL_PATH).setVocabOnly())) {
            expectRefusedAndGoOn(model);
        }
    }
}
