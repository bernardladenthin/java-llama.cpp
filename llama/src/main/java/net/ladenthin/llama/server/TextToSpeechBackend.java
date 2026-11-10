// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import java.util.List;
import net.ladenthin.llama.TextToSpeech;
import org.jspecify.annotations.Nullable;

/** The {@link SpeechBackend} over a loaded {@link TextToSpeech}, with its default sampling settings. */
final class TextToSpeechBackend implements SpeechBackend {

    /** The frame cap, top-k and seed of {@link TextToSpeech#synthesize(String)}. */
    private static final int MAX_FRAMES = 512;

    private static final int TOP_K = 4;
    private static final int SEED = 0;

    private final TextToSpeech speech;

    TextToSpeechBackend(TextToSpeech speech) {
        this.speech = speech;
    }

    @Override
    public synchronized byte[] synthesize(String text, @Nullable String voice, @Nullable String language) {
        return speech.synthesize(text, null, language, voice, MAX_FRAMES, TOP_K, SEED);
    }

    @Override
    public List<String> voices() {
        return speech.getVoices();
    }

    @Override
    public String toString() {
        return "TextToSpeechBackend{maxFrames=" + MAX_FRAMES + ", topK=" + TOP_K + ", seed=" + SEED + '}';
    }
}
