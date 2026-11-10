// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
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
    public byte[] synthesize(String text, @Nullable String voice, @Nullable String language) throws IOException {
        return synthesize(text, voice, language, SpeechOptions.defaults());
    }

    @Override
    public synchronized byte[] synthesize(
            String text, @Nullable String voice, @Nullable String language, SpeechOptions options) throws IOException {
        Integer maxFrames = options.getMaxFrames();
        Integer topK = options.getTopK();
        Integer seed = options.getSeed();
        Float temperature = options.getTemperature();
        byte[] reference = options.getSpeakerReference();
        // TextToSpeech reads a speaker reference from a file (llama.cpp decodes it by its content)
        Path referenceFile = null;
        try {
            if (reference != null) {
                referenceFile = Files.createTempFile("jllama-speaker-ref", ".bin");
                Files.write(referenceFile, reference);
            }
            return speech.synthesize(
                    text,
                    referenceFile == null ? null : referenceFile.toString(),
                    language,
                    voice,
                    maxFrames == null || maxFrames <= 0 ? MAX_FRAMES : maxFrames,
                    topK == null || topK <= 0 ? TOP_K : topK,
                    seed == null ? SEED : seed,
                    temperature == null ? TextToSpeech.MODEL_DEFAULT_TEMPERATURE : temperature);
        } finally {
            if (referenceFile != null) {
                File file = referenceFile.toFile();
                if (!file.delete()) {
                    file.deleteOnExit();
                }
            }
        }
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
