// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
// SPDX-FileCopyrightText: 2023-2025 Konstantin Herud
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import java.io.File;
import java.nio.file.Path;
import java.nio.file.Paths;
import net.ladenthin.llama.loader.LlamaSystemProperties;

public class TestConstants {

    /**
     * Resolves a test-fixture path that may be stated relative to either the reactor root or the
     * {@code llama/} module directory.
     *
     * <p>Surefire's working directory defaults to the module basedir ({@code <repo>/llama}), while
     * the shared GGUF cache CI restores &mdash; and the download commands in {@code CLAUDE.md}
     * &mdash; put models in {@code <repo>/models}. A bare {@code models/…} therefore resolves to a
     * path that does not exist, every model-gated test aborts in its {@code @BeforeAll}
     * {@code Assumptions.assumeTrue(file.exists())}, and the job still reports success &mdash; so the
     * whole model-backed suite silently self-skipped on every platform. This resolver accepts both
     * layouts instead of forcing one, which also keeps a developer who put models under
     * {@code llama/models/} working.</p>
     *
     * <p>An absolute path is returned unchanged. A relative path is returned unchanged when it
     * exists relative to the working directory; otherwise the parent directory is tried and, on a
     * hit, an absolute path is returned. A path that exists in neither place is returned unchanged,
     * so the caller's "model missing" skip message still names what it looked for.</p>
     *
     * @param path the configured path, may be {@code null} or empty
     * @return the resolved path, or {@code path} itself when it is null, empty, absolute, or unresolvable
     */
    public static String resolveModelPath(String path) {
        if (path == null || path.isEmpty()) {
            return path;
        }
        Path candidate = Paths.get(path);
        if (candidate.isAbsolute() || new File(path).exists()) {
            return path;
        }
        Path fromParent = Paths.get("..").resolve(candidate);
        if (fromParent.toFile().exists()) {
            return fromParent.toAbsolutePath().normalize().toString();
        }
        return path;
    }

    /**
     * Reads a system property holding a fixture path and resolves it via
     * {@link #resolveModelPath(String)}. CI passes these as {@code models/<name>}, which is subject
     * to exactly the working-directory mismatch described there.
     *
     * @param key the system-property name
     * @param defaultValue value to use when the property is unset, may be {@code null}
     * @return the resolved path, or {@code null} when neither the property nor a default is set
     */
    public static String resolveModelProperty(String key, String defaultValue) {
        return resolveModelPath(System.getProperty(key, defaultValue));
    }

    /**
     * Reads a system property holding a fixture path, with no default.
     *
     * @param key the system-property name
     * @return the resolved path, or {@code null} when the property is unset
     */
    public static String resolveModelProperty(String key) {
        return resolveModelProperty(key, null);
    }

    /** System property to override GPU layers used in tests. */
    public static final String PROP_TEST_NGL = LlamaSystemProperties.PREFIX + ".test.ngl";

    public static final int DEFAULT_TEST_NGL = 43;

    /** Path to the main text generation model used in tests. */
    public static final String MODEL_PATH = resolveModelPath("models/codellama-7b.Q2_K.gguf");

    /** Path to the draft model used for speculative decoding tests. */
    public static final String DRAFT_MODEL_PATH = resolveModelPath("models/AMD-Llama-135m-code.Q2_K.gguf");

    /** Path to the Qwen3 thinking model used for reasoning budget tests. */
    public static final String REASONING_MODEL_PATH = resolveModelPath("models/Qwen3-0.6B-Q4_K_M.gguf");

    /** Path to the reranking model used in tests (loaded with {@code enableReranking()}). */
    public static final String RERANKING_MODEL_PATH = resolveModelPath("models/jina-reranker-v1-tiny-en-Q4_0.gguf");

    /** System property overriding the GGUF used by the real tool-calling integration tests. */
    public static final String PROP_TOOL_MODEL_PATH = LlamaSystemProperties.PREFIX + ".tool.model";

    /** Qwen2.5 tool-capable model used by upstream llama.cpp's blocking and streaming tests. */
    public static final String DEFAULT_TOOL_MODEL_PATH = resolveModelPath("models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf");

    /**
     * System property holding a path to a Nomic embedding model
     * ({@code nomic-embed-text-v1.5.f16.gguf} or a compatible BERT-family encoder).
     * Used by {@link LlamaEmbeddingsTest#testNomicEmbedLoads} to confirm upstream
     * issue #98 (BERT-encoder result_output assertion) stays resolved.
     * When the property is unset the test self-skips.
     */
    public static final String PROP_NOMIC_MODEL_PATH = LlamaSystemProperties.PREFIX + ".nomic.path";

    /** Expected embedding dimension of nomic-embed-text-v1.5 (hidden size = 768). */
    public static final int NOMIC_EMBED_DIM = 768;

    /**
     * System property holding a path to a vision-capable model GGUF. Consumed by
     * {@code MultimodalIntegrationTest}. The CI default is the
     * SmolVLM-500M Q8_0 GGUF; the test self-skips when the property is unset or
     * the file is missing.
     */
    public static final String PROP_VISION_MODEL_PATH = LlamaSystemProperties.PREFIX + ".vision.model";

    /** System property holding a path to the matching mmproj GGUF for the vision model. */
    public static final String PROP_VISION_MMPROJ_PATH = LlamaSystemProperties.PREFIX + ".vision.mmproj";

    /**
     * System property holding a path to an image used as the visual prompt in
     * {@code MultimodalIntegrationTest}. When unset the test falls back to
     * {@link #DEFAULT_VISION_IMAGE_PATH}, which points at a small image
     * committed under {@code src/test/resources/images/}. Any png/jpeg/webp/gif
     * works; the matching extension drives MIME detection in
     * {@code ContentPart.imageFile(Path)}.
     */
    public static final String PROP_VISION_IMAGE_PATH = LlamaSystemProperties.PREFIX + ".vision.image";

    /**
     * Path used by {@code MultimodalIntegrationTest} when
     * {@link #PROP_VISION_IMAGE_PATH} is unset. Points at the committed test
     * resource so the test needs no network access for the visual prompt.
     */
    public static final String DEFAULT_VISION_IMAGE_PATH = resolveModelPath("src/test/resources/images/test-image.jpg");

    /**
     * System property holding a path to an audio-input model GGUF (e.g. Ultravox / Qwen2.5-Omni).
     * Consumed by {@code AudioInputIntegrationTest} (llama.cpp discussion #13759). The test self-skips
     * when this, the mmproj, or the audio clip is unset/missing.
     */
    public static final String PROP_AUDIO_MODEL_PATH = LlamaSystemProperties.PREFIX + ".audio.model";

    /** System property holding a path to the matching audio mmproj (encoder) GGUF. */
    public static final String PROP_AUDIO_MMPROJ_PATH = LlamaSystemProperties.PREFIX + ".audio.mmproj";

    /**
     * System property holding a path to a {@code .wav} or {@code .mp3} clip used as the audio prompt in
     * {@code AudioInputIntegrationTest}. When unset the test falls back to
     * {@link #DEFAULT_AUDIO_INPUT_PATH}, which points at a small clip committed under
     * {@code src/test/resources/audios/}. The matching extension drives format detection in
     * {@code ContentPart.audioFile(Path)}.
     */
    public static final String PROP_AUDIO_PATH = LlamaSystemProperties.PREFIX + ".audio.input";

    /**
     * Path used by {@code AudioInputIntegrationTest} when {@link #PROP_AUDIO_PATH} is unset. Points at
     * the committed test resource so only the (large) audio model + mmproj have to be staged
     * out-of-band.
     */
    public static final String DEFAULT_AUDIO_INPUT_PATH = resolveModelPath("src/test/resources/audios/sample.wav");

    /**
     * System property holding a path to the Qwen3-TTS backbone GGUF used by
     * {@code TtsIntegrationTest}. The test self-skips when this or the mmproj is unset/missing.
     */
    public static final String PROP_TTS_MODEL = LlamaSystemProperties.PREFIX + ".tts.model";

    /**
     * System property holding a path to the Qwen3-TTS mmproj GGUF (speaker encoder + code
     * predictor + code2wav decoder).
     */
    public static final String PROP_TTS_MMPROJ = LlamaSystemProperties.PREFIX + ".tts.mmproj";
}
