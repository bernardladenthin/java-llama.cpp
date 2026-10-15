// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT
package net.ladenthin.llama.ffm;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static net.ladenthin.llama.ffm.JllamaNative.cString;
import static net.ladenthin.llama.ffm.JllamaNative.invoke;
import static net.ladenthin.llama.ffm.JllamaNative.longSlot;
import static net.ladenthin.llama.ffm.JllamaNative.pointerSlot;
import static net.ladenthin.llama.ffm.JllamaNative.takeBytes;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;
import net.ladenthin.llama.exception.LlamaException;

/**
 * Qwen3-TTS speech synthesis ({@code TextToSpeech}): a backbone GGUF plus an mmproj GGUF, text in, a
 * 24 kHz mono 16-bit WAV out. One synthesis at a time per instance.
 */
public final class FfmTextToSpeech implements AutoCloseable {

    private final MemorySegment handle;
    private final AtomicBoolean closed = new AtomicBoolean();

    private FfmTextToSpeech(MemorySegment handle) {
        this.handle = handle;
    }

    /**
     * Loads the backbone and the mmproj.
     *
     * @param modelPath the backbone GGUF
     * @param mmprojPath the mmproj GGUF (speaker encoder, code predictor, decoder)
     * @param gpuLayers the layers to offload, {@code -1} for automatic
     * @param threads the CPU threads, {@code 0} or less for the default
     * @return the loaded pair
     * @throws LlamaException when a model does not load
     */
    public static FfmTextToSpeech load(String modelPath, String mmprojPath, int gpuLayers, int threads) {
        Objects.requireNonNull(modelPath, "modelPath");
        Objects.requireNonNull(mmprojPath, "mmprojPath");
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment out = pointerSlot(arena);
            JllamaNative.call(JllamaNative.TTS_LOAD, cString(arena, modelPath), cString(arena, mmprojPath), gpuLayers,
                    threads, out);
            return new FfmTextToSpeech(out.get(ADDRESS, 0));
        }
    }

    /**
     * Synthesises text.
     *
     * @param text the text to speak
     * @param speakerReferencePath a WAV to clone the voice from, or {@code null}
     * @param language the language ({@code "english"}, {@code "chinese"}, ...), or {@code null}
     * @param maxFrames the bound on the generated frames
     * @param topK the sampling top-k
     * @param seed the sampling seed
     * @return the WAV bytes
     * @throws LlamaException when the synthesis fails
     */
    public byte[] synthesize(String text, String speakerReferencePath, String language, int maxFrames, int topK,
            int seed) {
        Objects.requireNonNull(text, "text");
        if (closed.get()) {
            throw new IllegalStateException("the model is closed");
        }
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment wav = pointerSlot(arena);
            MemorySegment length = longSlot(arena);
            JllamaNative.call(JllamaNative.TTS_SYNTHESIZE, handle, cString(arena, text),
                    cString(arena, speakerReferencePath), cString(arena, language), maxFrames, topK, seed, wav,
                    length);
            return takeBytes(wav, length);
        }
    }

    /** Frees the model pair; a second call is a no-op. */
    @Override
    public void close() {
        if (closed.compareAndSet(false, true)) {
            invoke(JllamaNative.TTS_FREE, handle);
        }
    }
}
