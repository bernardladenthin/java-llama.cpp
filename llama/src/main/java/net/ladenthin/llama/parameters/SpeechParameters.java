// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.parameters;

import lombok.Builder;
import lombok.Getter;
import lombok.ToString;
import org.jspecify.annotations.Nullable;

/**
 * Immutable options of one {@link net.ladenthin.llama.TextToSpeech} synthesis: everything llama.cpp's
 * own speech route ({@code POST /tts}) reads besides the text.
 *
 * <p>Build with {@code builder()}; every value defaults to the one the shorter {@code synthesize}
 * overloads pass, which leaves the choice to the engine or the model. {@code topP},
 * {@code repeatPenalty}, {@code speed} and {@code loudness} came with llama.cpp b12431: the first two
 * shape the backbone sampler (top-p also the acoustic code predictor), the last two the delivery of a
 * model that supports it (Zonos3). Another model refuses a speed other than {@code 1} or a loudness
 * other than {@code 0}, with an {@link net.ladenthin.llama.exception.InvalidRequestException}, as it
 * refuses values out of range. {@code speakerReferenceText} (b12453) is the transcript of the speaker
 * reference, which Fish Audio S2 puts into its prompt and Qwen3-TTS (b12491) clones the voice from
 * more closely (its "ICL" mode: a clip of 30 s or less and an mmproj with the reference encoder, else
 * the speaker embedding alone); the other models ignore it. {@code cfgScale} and
 * {@code flowSteps} (b12459) are the guidance and the solver steps of a flow-matching model (YuE2); 0
 * leaves both to the model, and another model refuses any other value. A {@code maxFrames} of 0 or less
 * leaves the frame cap to the model (512 for most; a model that stops by itself names none). The C
 * counterpart is {@code jllama_tts_options}.
 */
@Builder(toBuilder = true)
@Getter
@ToString
public final class SpeechParameters {

    /** The frame cap of {@link net.ladenthin.llama.TextToSpeech#synthesize(String)}. */
    public static final int DEFAULT_MAX_FRAMES = 512;

    /** The top-k of {@link net.ladenthin.llama.TextToSpeech#synthesize(String)}. */
    public static final int DEFAULT_TOP_K = 4;

    /** The audio frame temperature that leaves the choice to the model. */
    public static final float MODEL_DEFAULT_TEMPERATURE = -1.0f;

    /** The top-p that switches top-p sampling off. */
    public static final float TOP_P_OFF = 1.0f;

    /** The repetition penalty that switches the penalty off. */
    public static final float REPEAT_PENALTY_OFF = 1.0f;

    /** The speaking speed of the model's own pace. */
    public static final float MODEL_DEFAULT_SPEED = 1.0f;

    /** The loudness that leaves it to the model. */
    public static final float MODEL_DEFAULT_LOUDNESS = 0.0f;

    // Path to a reference clip whose voice is cloned, or null for the model's own voice.
    private final @Nullable String speakerReferenceAudioPath;

    // The model's language ("english", "en", ...), or null for its default.
    private final @Nullable String language;

    // A predefined or template voice, or null for the model's default.
    private final @Nullable String voice;

    // Cap on the generated audio frames (longer = longer audio); 0 or less leaves it to the model.
    @Builder.Default
    private final int maxFrames = DEFAULT_MAX_FRAMES;

    // Top-k cutoff of the semantic (backbone) token stream.
    @Builder.Default
    private final int topK = DEFAULT_TOP_K;

    // Sampler and codec seed.
    private final int seed;

    // Audio frame temperature: 0 is greedy, a negative value the model's default.
    @Builder.Default
    private final float temperature = MODEL_DEFAULT_TEMPERATURE;

    // Top-p of the backbone sampler and the code predictor, in [0, 1]; 0 or 1 is off.
    @Builder.Default
    private final float topP = TOP_P_OFF;

    // Repetition penalty of the backbone over the whole generation, > 0; 1 is off.
    @Builder.Default
    private final float repeatPenalty = REPEAT_PENALTY_OFF;

    // Speaking speed relative to the model's pace, > 0.
    @Builder.Default
    private final float speed = MODEL_DEFAULT_SPEED;

    // Target loudness in LUFS, < 0; 0 leaves it to the model.
    @Builder.Default
    private final float loudness = MODEL_DEFAULT_LOUDNESS;

    // The transcript of the speaker reference, or null. Fish Audio S2 (llama.cpp b12453) and Qwen3-TTS
    // (b12491) follow the reference's voice more closely with it; the others ignore it.
    private final @Nullable String speakerReferenceText;

    // Classifier-free guidance scale of the audio codes, > 0 (1 = no guidance); 0 = the model's own.
    // Only a model that supports it (YuE2, llama.cpp b12459); another refuses any other value.
    private final float cfgScale;

    // Steps of the flow-matching ODE solver, 1 to 1024 (fewer is faster, less accurate); 0 = the
    // model's own. Only a model that supports it (YuE2, llama.cpp b12459); another refuses any other value.
    private final int flowSteps;

    /**
     * The options of a synthesis that sets none.
     *
     * @return the defaults
     */
    public static SpeechParameters defaults() {
        return builder().build();
    }
}
