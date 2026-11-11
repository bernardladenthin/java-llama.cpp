// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import org.jspecify.annotations.Nullable;

/**
 * What a speech request may set besides its text, voice and language: the fields of llama.cpp's
 * {@code POST /tts} that a {@link SpeechBackend} can apply. Every one is optional; {@code null}
 * leaves the engine's own default. Immutable.
 */
public final class SpeechOptions {

    private static final SpeechOptions DEFAULTS = new SpeechOptions(null, null, null, null, null);

    private final byte @Nullable [] speakerReference;
    private final @Nullable Integer maxFrames;
    private final @Nullable Integer topK;
    private final @Nullable Integer seed;
    private final @Nullable Float temperature;

    /**
     * Create the options of one request.
     *
     * @param speakerReference an audio file (WAV, MP3, ...) whose voice is cloned, or {@code null}
     * @param maxFrames the cap on the generated audio frames, or {@code null}
     * @param topK the top-k of the sampler, or {@code null}
     * @param seed the seed of the sampler, or {@code null}
     * @param temperature the temperature the audio frames are sampled at, or {@code null}
     */
    public SpeechOptions(
            byte @Nullable [] speakerReference,
            @Nullable Integer maxFrames,
            @Nullable Integer topK,
            @Nullable Integer seed,
            @Nullable Float temperature) {
        this.speakerReference = speakerReference == null ? null : speakerReference.clone();
        this.maxFrames = maxFrames;
        this.topK = topK;
        this.seed = seed;
        this.temperature = temperature;
    }

    /**
     * The options of a request that sets none.
     *
     * @return options with every value {@code null}
     */
    public static SpeechOptions defaults() {
        return DEFAULTS;
    }

    /**
     * The audio file whose voice is cloned.
     *
     * @return a copy of the file's bytes, or {@code null} for the model's own voice
     */
    public byte @Nullable [] getSpeakerReference() {
        return speakerReference == null ? null : speakerReference.clone();
    }

    /**
     * The cap on the generated audio frames.
     *
     * @return the cap, or {@code null} for the engine's default
     */
    public @Nullable Integer getMaxFrames() {
        return maxFrames;
    }

    /**
     * The top-k of the sampler.
     *
     * @return the top-k, or {@code null} for the engine's default
     */
    public @Nullable Integer getTopK() {
        return topK;
    }

    /**
     * The seed of the sampler.
     *
     * @return the seed, or {@code null} for the engine's default
     */
    public @Nullable Integer getSeed() {
        return seed;
    }

    /**
     * The temperature the audio frames are sampled at.
     *
     * @return the temperature, or {@code null} for the model's default
     */
    public @Nullable Float getTemperature() {
        return temperature;
    }

    @Override
    public String toString() {
        return "SpeechOptions{speakerReference="
                + (speakerReference == null ? "none" : speakerReference.length + " bytes")
                + ", maxFrames=" + maxFrames
                + ", topK=" + topK
                + ", seed=" + seed
                + ", temperature=" + temperature
                + '}';
    }
}
