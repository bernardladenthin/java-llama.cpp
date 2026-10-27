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

    /** The TTS backbone the TTS integration test loads; CI's Qwen3-TTS backbone by default. */
    static final String TTS_MODEL_PROPERTY = "net.ladenthin.llama.ffm.tts.model";

    static final String DEFAULT_TTS_MODEL = "models/Qwen3-TTS-12Hz-1.7B-Base-Q4_K_M.gguf";

    /** The TTS mmproj the TTS integration test loads; CI's Qwen3-TTS mmproj by default. */
    static final String TTS_MMPROJ_PROPERTY = "net.ladenthin.llama.ffm.tts.mmproj";

    static final String DEFAULT_TTS_MMPROJ = "models/mmproj-Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf";

    /** The speaker reference clip committed with the core's tests (2.6 s, 48 kHz stereo WAV). */
    static final String SPEAKER_REFERENCE = "llama/src/test/resources/audios/sample.wav";

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
        return resolve(System.getProperty(MODEL_PROPERTY, DEFAULT_MODEL));
    }

    /**
     * {@code configured} as an existing path: absolute or module-relative first, then relative to
     * the reactor root; unchanged when neither exists.
     *
     * @param configured the path as configured
     * @return the path to use
     */
    static Path resolve(String configured) {
        Path candidate = Paths.get(configured);
        if (candidate.isAbsolute() || Files.exists(candidate)) {
            return candidate;
        }
        Path fromParent = Paths.get("..").resolve(candidate).toAbsolutePath().normalize();
        return Files.exists(fromParent) ? fromParent : candidate;
    }

    /**
     * Skips the test when libjllama is not loadable or the file the property names (or its default)
     * is absent.
     *
     * @param property the system property naming the file
     * @param defaultPath the path used when the property is unset
     * @return the resolved, existing path
     */
    static Path assumeFile(String property, String defaultPath) {
        assumeNativesLoadable();
        Path file = resolve(System.getProperty(property, defaultPath));
        Assumptions.assumeTrue(Files.exists(file), "file not found: " + file + " (-D" + property + ")");
        return file;
    }

    /** Skips the test when the model is absent. */
    static Path assumeModel() {
        assumeNativesLoadable();
        Path model = modelPath();
        Assumptions.assumeTrue(Files.exists(model), "model not found: " + model + " (-D" + MODEL_PROPERTY + ")");
        return model;
    }
}
