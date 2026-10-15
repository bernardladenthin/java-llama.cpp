// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT
package net.ladenthin.llama.ffm;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import org.junit.jupiter.api.Assumptions;

/** What the tests of this module share: the natives check and the model path resolution. */
final class FfmTestSupport {

    /** The model the integration test loads; CI's draft model by default (.github/models.csv). */
    static final String MODEL_PROPERTY = "net.ladenthin.llama.ffm.model";

    static final String DEFAULT_MODEL = "models/AMD-Llama-135m-code.Q2_K.gguf";

    private FfmTestSupport() {}

    /**
     * Skips the test when libjllama is not loadable (a checkout without a native build), the same
     * self-skip as the core's native-backed tests.
     */
    static void assumeNativesLoadable() {
        try {
            Jllama.apiVersion();
        } catch (UnsatisfiedLinkError | ExceptionInInitializerError | NoClassDefFoundError e) {
            Assumptions.abort("libjllama is not loadable here: " + e);
        }
    }

    /**
     * The configured model path, module-relative first and then relative to the reactor root (CI
     * restores the models to {@code <workspace>/models/} while surefire runs in {@code llama-ffm/}),
     * the resolution the core's {@code TestConstants} does.
     */
    static Path modelPath() {
        String configured = System.getProperty(MODEL_PROPERTY, DEFAULT_MODEL);
        Path candidate = Paths.get(configured);
        if (candidate.isAbsolute() || Files.exists(candidate)) {
            return candidate;
        }
        Path fromParent = Paths.get("..").resolve(candidate).toAbsolutePath().normalize();
        return Files.exists(fromParent) ? fromParent : candidate;
    }

    /** Skips the test when the model is absent. */
    static Path assumeModel() {
        assumeNativesLoadable();
        Path model = modelPath();
        Assumptions.assumeTrue(Files.exists(model), "model not found: " + model + " (-D" + MODEL_PROPERTY + ")");
        return model;
    }
}
