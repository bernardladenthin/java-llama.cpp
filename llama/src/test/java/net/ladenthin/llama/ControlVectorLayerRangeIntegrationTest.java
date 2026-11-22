// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import static org.junit.jupiter.api.Assertions.assertFalse;

import java.io.File;
import net.ladenthin.llama.parameters.InferenceParameters;
import net.ladenthin.llama.parameters.ModelParameters;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/**
 * {@link ModelParameters#setControlVectorLayerRange(int, int)} reaches llama.cpp's parser as the two
 * arguments it reads ({@code --control-vector-layer-range START END}). Until 2026-11-19 the setter
 * wrote {@code "START,END"} as one, which upstream read as START and then took the next argument as
 * END -- the next option's name -- so every load with the setter failed. A range without a control
 * vector is accepted and does nothing, which is what lets this test use CI's draft model; the C API
 * test {@code CApiControlVector} applies a vector the generator writes.
 */
@ClaudeGenerated(
        purpose = "Pin that a model loads through JNI with a control-vector layer range, whose two values the "
                + "builder once joined into one argument.")
public class ControlVectorLayerRangeIntegrationTest {

    @BeforeAll
    static void needsTheDraftModel() {
        Assumptions.assumeTrue(
                new File(TestConstants.DRAFT_MODEL_PATH).exists(),
                "Draft model not found, skipping ControlVectorLayerRangeIntegrationTest");
    }

    @Test
    public void aModelLoadsWithAControlVectorLayerRangeFollowedByOtherOptions() {
        int gpuLayers = Integer.getInteger(TestConstants.PROP_TEST_NGL, TestConstants.DEFAULT_TEST_NGL);
        ModelParameters parameters = new ModelParameters()
                .setModel(TestConstants.DRAFT_MODEL_PATH)
                .setControlVectorLayerRange(1, 2)
                .setCtxSize(256)
                .setGpuLayers(gpuLayers)
                .setFit(false);
        try (LlamaModel model = new LlamaModel(parameters)) {
            String answer = model.complete(
                    new InferenceParameters("def main():").withNPredict(4).withTemperature(0.0f));
            assertFalse(answer.isEmpty(), "the model answered nothing");
        }
    }
}
