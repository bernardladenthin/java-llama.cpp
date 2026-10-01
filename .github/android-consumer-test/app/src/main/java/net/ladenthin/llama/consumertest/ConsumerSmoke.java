// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

package net.ladenthin.llama.consumertest;

import net.ladenthin.llama.LlamaModel;
import net.ladenthin.llama.parameters.InferenceParameters;
import net.ladenthin.llama.parameters.ModelParameters;

/**
 * Compile-time consumer check: references the binding's API surface exactly the way an app
 * would, so the AGP build fails if the AAR's classes.jar is broken or its POM dependencies
 * (Jackson etc.) do not resolve. No code here runs in CI — the fixture is never installed on a
 * device; on-device inference is covered by the LLM Service app (android-llmservice, separate project).
 */
public final class ConsumerSmoke {

    private ConsumerSmoke() {}

    /**
     * Opens a model from an absolute on-device path.
     *
     * @param modelPath absolute path to a GGUF file on device storage
     * @return the loaded model (caller closes)
     */
    public static LlamaModel open(String modelPath) {
        return new LlamaModel(new ModelParameters().setModel(modelPath).setCtxSize(512));
    }

    /**
     * Runs a short completion.
     *
     * @param model the loaded model
     * @param prompt the prompt text
     * @return the generated text
     */
    public static String prompt(LlamaModel model, String prompt) {
        return model.complete(new InferenceParameters(prompt).withNPredict(8));
    }
}
