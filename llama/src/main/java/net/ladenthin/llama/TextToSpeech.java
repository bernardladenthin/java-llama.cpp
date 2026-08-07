// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import net.ladenthin.llama.loader.LlamaLoader;
import org.jspecify.annotations.Nullable;

/**
 * Text-to-speech synthesis over llama.cpp's upstream Qwen3-TTS audio-generation pipeline. Loads a
 * backbone text model plus an mmproj GGUF (speaker encoder + code predictor + code2wav decoder,
 * all bundled in one file) and turns text into a 24&nbsp;kHz mono 16-bit WAV byte stream.
 *
 * <p>This is a separate native type from {@link LlamaModel} because TTS uses its own model pair
 * and does not go through the chat/completion server path. Native memory is not GC-managed: use
 * try-with-resources or call {@link #close()} explicitly.
 *
 * <pre>{@code
 * try (TextToSpeech tts = new TextToSpeech(
 *         "models/qwen3-tts-backbone.gguf", "models/qwen3-tts-mmproj.gguf")) {
 *     byte[] wav = tts.synthesize("Hello from llama dot c p p.");
 *     Files.write(Paths.get("out.wav"), wav);
 * }
 * }</pre>
 */
public final class TextToSpeech implements AutoCloseable {

    static {
        LlamaLoader.initialize();
    }

    private long handle;

    /**
     * Load the TTS pipeline CPU-only.
     *
     * @param modelPath path to the backbone (text) GGUF
     * @param mmprojPath path to the mmproj GGUF (speaker encoder + code predictor + code2wav)
     */
    public TextToSpeech(String modelPath, String mmprojPath) {
        this(modelPath, mmprojPath, 0, 0);
    }

    /**
     * Load the TTS pipeline.
     *
     * @param modelPath path to the backbone (text) GGUF
     * @param mmprojPath path to the mmproj GGUF (speaker encoder + code predictor + code2wav)
     * @param gpuLayers number of layers to offload to the GPU (0 = CPU only)
     * @param threads CPU threads for the backbone (0 = a small default)
     */
    public TextToSpeech(String modelPath, String mmprojPath, int gpuLayers, int threads) {
        this.handle = loadNative(modelPath, mmprojPath, gpuLayers, threads);
    }

    /**
     * Synthesize speech with the model's default voice and default sampling (top-k 4, seed 0, up
     * to 512 audio frames).
     *
     * @param text the text to speak
     * @return a 24&nbsp;kHz mono 16-bit WAV byte stream
     */
    public byte[] synthesize(String text) {
        return synthesize(text, 512, 4, 0);
    }

    /**
     * Synthesize speech with the model's default voice and explicit sampling parameters.
     *
     * @param text the text to speak
     * @param maxFrames cap on generated audio frames (longer = longer audio)
     * @param topK top-k sampling cutoff for the semantic (backbone) token stream
     * @param seed sampler seed
     * @return a 24&nbsp;kHz mono 16-bit WAV byte stream
     */
    public byte[] synthesize(String text, int maxFrames, int topK, int seed) {
        return synthesize(text, null, null, maxFrames, topK, seed);
    }

    /**
     * Synthesize speech with an optional cloned voice and language.
     *
     * @param text the text to speak
     * @param speakerReferenceAudioPath path to a reference audio clip (wav/mp3) whose voice is
     *     cloned, or {@code null}/empty for the model's default voice
     * @param language ISO 639-1-ish language name understood by the model (e.g. {@code "english"},
     *     {@code "chinese"} — see the model's own documentation for the supported set), or
     *     {@code null}/empty for the model's default
     * @param maxFrames cap on generated audio frames (longer = longer audio)
     * @param topK top-k sampling cutoff for the semantic (backbone) token stream
     * @param seed sampler seed
     * @return a 24&nbsp;kHz mono 16-bit WAV byte stream
     */
    public byte[] synthesize(
            String text,
            @Nullable String speakerReferenceAudioPath,
            @Nullable String language,
            int maxFrames,
            int topK,
            int seed) {
        if (handle == 0L) {
            throw new IllegalStateException("TextToSpeech is closed");
        }
        return synthesizeNative(handle, text, speakerReferenceAudioPath, language, maxFrames, topK, seed);
    }

    @Override
    public synchronized void close() {
        if (handle != 0L) {
            deleteNative(handle);
            handle = 0L;
        }
    }

    private static native long loadNative(String modelPath, String mmprojPath, int gpuLayers, int threads);

    private static native byte[] synthesizeNative(
            long handle,
            String text,
            @Nullable String speakerReferenceAudioPath,
            @Nullable String language,
            int maxFrames,
            int topK,
            int seed);

    private static native void deleteNative(long handle);
}
