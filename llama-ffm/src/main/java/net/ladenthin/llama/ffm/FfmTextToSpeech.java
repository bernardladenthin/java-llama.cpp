// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT
package net.ladenthin.llama.ffm;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_BOOLEAN;
import static net.ladenthin.llama.ffm.JllamaNative.booleanSlot;
import static net.ladenthin.llama.ffm.JllamaNative.cString;
import static net.ladenthin.llama.ffm.JllamaNative.invoke;
import static net.ladenthin.llama.ffm.JllamaNative.longSlot;
import static net.ladenthin.llama.ffm.JllamaNative.pointerSlot;
import static net.ladenthin.llama.ffm.JllamaNative.takeBytes;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Objects;
import java.util.concurrent.locks.ReentrantReadWriteLock;
import net.ladenthin.llama.exception.LlamaException;

/**
 * Qwen3-TTS speech synthesis ({@code TextToSpeech}): a backbone GGUF plus an mmproj GGUF, text in, a
 * 24 kHz mono 16-bit WAV out. One synthesis at a time per instance.
 */
public final class FfmTextToSpeech implements AutoCloseable {

    /** The native calls on a loaded pair's handle; a seam so the locking is testable without it. */
    interface Backend {

        /**
         * Synthesises {@code text} on {@code handle}.
         *
         * @param handle the loaded pair
         * @param text the text to speak
         * @param speakerReferencePath a WAV to clone the voice from, or {@code null}
         * @param language the language, or {@code null}
         * @param maxFrames the bound on the generated frames
         * @param topK the sampling top-k
         * @param seed the sampling seed
         * @return the WAV bytes
         */
        byte[] synthesize(MemorySegment handle, String text, String speakerReferencePath, String language,
                int maxFrames, int topK, int seed);

        /**
         * Whether the pair on {@code handle} clones a voice from a speaker reference.
         *
         * @param handle the loaded pair
         * @return {@code true} if a speaker reference is accepted
         */
        boolean supportsSpeakerReference(MemorySegment handle);

        /**
         * Frees {@code handle}, the engine with it.
         *
         * @param handle the loaded pair
         */
        void free(MemorySegment handle);
    }

    /** The real calls into the library. */
    private static final Backend NATIVE = new Backend() {
        @Override
        public byte[] synthesize(MemorySegment handle, String text, String speakerReferencePath, String language,
                int maxFrames, int topK, int seed) {
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment wav = pointerSlot(arena);
                MemorySegment length = longSlot(arena);
                JllamaNative.call(JllamaNative.TTS_SYNTHESIZE, handle, cString(arena, text),
                        cString(arena, speakerReferencePath), cString(arena, language), maxFrames, topK, seed, wav,
                        length);
                return takeBytes(wav, length);
            }
        }

        @Override
        public boolean supportsSpeakerReference(MemorySegment handle) {
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment supported = booleanSlot(arena);
                JllamaNative.call(JllamaNative.TTS_SUPPORTS_SPEAKER_REFERENCE, handle, supported);
                return supported.get(JAVA_BOOLEAN, 0);
            }
        }

        @Override
        public void free(MemorySegment handle) {
            invoke(JllamaNative.TTS_FREE, handle);
        }
    };

    private final MemorySegment handle;
    private final Backend backend;

    /**
     * Serializes {@link #close()} against {@link #synthesize}, as in the JNI {@code TextToSpeech}
     * (F16): the native free deletes the engine, and nothing on the native side counts the threads
     * still inside a call on it, so a free during a synthesis left that synthesis on freed memory.
     * Syntheses share the read lock (the engine serializes them itself); close takes the write lock
     * and so waits for every synthesis in progress.
     */
    private final ReentrantReadWriteLock lifecycleLock = new ReentrantReadWriteLock();

    /** Set by {@link #close()} under the write lock. */
    private boolean closed;

    private FfmTextToSpeech(MemorySegment handle) {
        this(handle, NATIVE);
    }

    /**
     * A pair over {@code handle} that calls {@code backend}; for tests.
     *
     * @param handle the loaded pair
     * @param backend the native calls
     */
    FfmTextToSpeech(MemorySegment handle, Backend backend) {
        this.handle = handle;
        this.backend = backend;
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
     * Whether this pair clones a voice from a speaker reference, i.e. whether {@link #synthesize}
     * accepts a {@code speakerReferencePath} ({@code mtmd_helper_cap::audio_gen_spk_ref}: Qwen3-TTS
     * and Pocket TTS do, Soprano does not).
     *
     * @return {@code true} if a speaker reference is accepted
     * @throws IllegalStateException when the pair is closed
     */
    public boolean supportsSpeakerReference() {
        lifecycleLock.readLock().lock();
        try {
            if (closed) {
                throw new IllegalStateException("the model is closed");
            }
            return backend.supportsSpeakerReference(handle);
        } finally {
            lifecycleLock.readLock().unlock();
        }
    }

    /**
     * Synthesises text.
     *
     * @param text the text to speak
     * @param speakerReferencePath a WAV to clone the voice from, or {@code null}; refused with a
     *     {@link LlamaException} when {@link #supportsSpeakerReference()} is {@code false}
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
        lifecycleLock.readLock().lock();
        try {
            if (closed) {
                throw new IllegalStateException("the model is closed");
            }
            return backend.synthesize(handle, text, speakerReferencePath, language, maxFrames, topK, seed);
        } finally {
            lifecycleLock.readLock().unlock();
        }
    }

    /** Frees the model pair once every synthesis in progress has returned; a second call is a no-op. */
    @Override
    public void close() {
        lifecycleLock.writeLock().lock();
        try {
            if (!closed) {
                closed = true;
                backend.free(handle);
            }
        } finally {
            lifecycleLock.writeLock().unlock();
        }
    }
}
