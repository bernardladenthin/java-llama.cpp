// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import org.junit.Assume;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

@ClaudeGenerated(
        purpose = "Verify LoadProgressCallback receives non-decreasing progress values in [0,1] "
                + "during a real model load, and that returning false from the callback aborts the load."
)
public class LoadProgressCallbackTest {

    @Test
    public void receivesProgressUpdates() {
        Assume.assumeTrue("Model file not found", new java.io.File(TestConstants.MODEL_PATH).exists());

        List<Float> updates = new ArrayList<Float>();
        int gpuLayers = Integer.getInteger(TestConstants.PROP_TEST_NGL, TestConstants.DEFAULT_TEST_NGL);

        try (LlamaModel m = new LlamaModel(
                new ModelParameters()
                        .setCtxSize(128)
                        .setModel(TestConstants.MODEL_PATH)
                        .setGpuLayers(gpuLayers)
                        .setFit(false),
                progress -> {
                    updates.add(progress);
                    return true;
                })) {
            // model load completed
        }

        assertFalse("expected at least one progress update", updates.isEmpty());
        for (Float p : updates) {
            assertTrue("progress out of range: " + p, p >= 0.0f && p <= 1.0f);
        }
        // Last update should reach (or be very close to) 1.0
        assertTrue("last progress should reach completion, got " + updates.get(updates.size() - 1),
                updates.get(updates.size() - 1) >= 0.9f);
        // Non-decreasing
        for (int i = 1; i < updates.size(); i++) {
            assertTrue("progress decreased at index " + i + ": " + updates.get(i - 1) + " -> " + updates.get(i),
                    updates.get(i) >= updates.get(i - 1));
        }
        // Sanity: progress actually advanced
        assertNotEquals("progress never advanced", updates.get(0), updates.get(updates.size() - 1));
    }

    @Test
    public void returningFalseAbortsLoad() {
        Assume.assumeTrue("Model file not found", new java.io.File(TestConstants.MODEL_PATH).exists());

        int gpuLayers = Integer.getInteger(TestConstants.PROP_TEST_NGL, TestConstants.DEFAULT_TEST_NGL);
        try {
            new LlamaModel(
                    new ModelParameters()
                            .setCtxSize(128)
                            .setModel(TestConstants.MODEL_PATH)
                            .setGpuLayers(gpuLayers)
                            .setFit(false),
                    progress -> false).close();
            fail("expected LlamaException when callback aborts load");
        } catch (LlamaException expected) {
            // pass
        }
    }

    @Test
    public void nullCallbackBehavesAsDefault() {
        Assume.assumeTrue("Model file not found", new java.io.File(TestConstants.MODEL_PATH).exists());
        int gpuLayers = Integer.getInteger(TestConstants.PROP_TEST_NGL, TestConstants.DEFAULT_TEST_NGL);
        try (LlamaModel m = new LlamaModel(
                new ModelParameters()
                        .setCtxSize(128)
                        .setModel(TestConstants.MODEL_PATH)
                        .setGpuLayers(gpuLayers)
                        .setFit(false),
                null)) {
            // no callback wired; just verifies the null-overload routes to plain loadModel
        }
    }
}
