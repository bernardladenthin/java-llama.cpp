// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import java.util.concurrent.locks.ReentrantReadWriteLock;
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

    /**
     * The native calls behind {@link TextToSpeech}. The JNI methods stay in this class (their symbol
     * names are bound to it); this seam exists so the lifecycle can be tested with a fake that never
     * hands a made-up handle to libjllama.
     */
    interface Backend {
        /**
         * Loads the pipeline.
         *
         * @param modelPath  path to the backbone GGUF
         * @param mmprojPath path to the mmproj GGUF
         * @param gpuLayers  layers to offload to the GPU
         * @param threads    CPU threads for the backbone
         * @return the native handle
         */
        long load(String modelPath, String mmprojPath, int gpuLayers, int threads);

        /**
         * Synthesizes speech.
         *
         * @param handle                    a handle returned by {@link #load}
         * @param text                      the text to speak
         * @param speakerReferenceAudioPath reference clip, or {@code null}
         * @param language                  language name, or {@code null}
         * @param voice                     voice preset, or {@code null}
         * @param maxFrames                 cap on generated audio frames
         * @param topK                      top-k cutoff
         * @param seed                      sampler seed
         * @return the WAV bytes
         */
        byte[] synthesize(
                long handle,
                String text,
                @Nullable String speakerReferenceAudioPath,
                @Nullable String language,
                @Nullable String voice,
                int maxFrames,
                int topK,
                int seed);

        /**
         * Whether the loaded pair clones a voice from a speaker reference.
         *
         * @param handle a handle returned by {@link #load}
         * @return {@code true} if a speaker reference is accepted
         */
        boolean supportsSpeakerReference(long handle);

        /**
         * Frees the pipeline.
         *
         * @param handle a handle returned by {@link #load}
         */
        void delete(long handle);
    }

    /** The real backend: loads libjllama on the first load, then calls the JNI methods of this class. */
    private static final Backend NATIVE_BACKEND = new Backend() {
        @Override
        public long load(String modelPath, String mmprojPath, int gpuLayers, int threads) {
            LlamaLoader.initialize();
            return loadNative(modelPath, mmprojPath, gpuLayers, threads);
        }

        @Override
        public byte[] synthesize(
                long handle,
                String text,
                @Nullable String speakerReferenceAudioPath,
                @Nullable String language,
                @Nullable String voice,
                int maxFrames,
                int topK,
                int seed) {
            return synthesizeNative(handle, text, speakerReferenceAudioPath, language, voice, maxFrames, topK, seed);
        }

        @Override
        public boolean supportsSpeakerReference(long handle) {
            return supportsSpeakerReferenceNative(handle);
        }

        @Override
        public void delete(long handle) {
            deleteNative(handle);
        }
    };

    private final Backend backend;

    /**
     * Guards {@link #handle}: {@code synthesize} holds the read lock for the whole native call and
     * {@code close} the write lock, so the engine is never freed while a synthesis still runs in it
     * (its own mutex is part of the freed memory). Concurrent syntheses stay possible; the engine
     * orders them itself.
     */
    private final ReentrantReadWriteLock lifecycleLock = new ReentrantReadWriteLock();

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
        this(NATIVE_BACKEND, modelPath, mmprojPath, gpuLayers, threads);
    }

    /**
     * Load the TTS pipeline through the given backend (tests use a fake one).
     *
     * @param backend    the native entry points
     * @param modelPath  path to the backbone (text) GGUF
     * @param mmprojPath path to the mmproj GGUF
     * @param gpuLayers  number of layers to offload to the GPU
     * @param threads    CPU threads for the backbone
     */
    TextToSpeech(Backend backend, String modelPath, String mmprojPath, int gpuLayers, int threads) {
        this.backend = backend;
        this.handle = backend.load(modelPath, mmprojPath, gpuLayers, threads);
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
     * Whether this model pair clones a voice from a speaker reference clip, i.e. whether
     * {@link #synthesize(String, String, String, int, int, int)} accepts a
     * {@code speakerReferenceAudioPath}. llama.cpp decides it per pipeline
     * ({@code mtmd_helper_cap::audio_gen_spk_ref}, b11901): Qwen3-TTS and Pocket TTS do, a pipeline
     * without a voice encoder such as Soprano does not.
     *
     * @return {@code true} if a speaker reference is accepted
     * @throws IllegalStateException if this instance is closed
     */
    public boolean supportsSpeakerReference() {
        lifecycleLock.readLock().lock();
        try {
            requireOpen();
            return backend.supportsSpeakerReference(handle);
        } finally {
            lifecycleLock.readLock().unlock();
        }
    }

    /**
     * Synthesize speech with an optional cloned voice and language.
     *
     * @param text the text to speak
     * @param speakerReferenceAudioPath path to a reference audio clip (wav/mp3) whose voice is
     *     cloned, or {@code null}/empty for the model's default voice; a model pair that cannot clone
     *     a voice ({@link #supportsSpeakerReference()} is {@code false}) refuses it with a
     *     {@link net.ladenthin.llama.exception.LlamaException}
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
        return synthesize(text, speakerReferenceAudioPath, language, null, maxFrames, topK, seed);
    }

    /**
     * Synthesize speech with an optional cloned voice, language and voice preset.
     *
     * <p>A model whose GGUF carries a {@code "tts"} chat template (LFM3-Audio) speaks the prompt
     * rendered from that template with {@code text}, {@code language} and {@code voice}, as llama.cpp's
     * own {@code /tts} route and {@code llama-tts} do (b11941); the other pipelines (Qwen3-TTS, Pocket
     * TTS, Soprano) speak {@code text} itself and ignore an unknown voice.</p>
     *
     * @param text the text to speak
     * @param speakerReferenceAudioPath path to a reference audio clip whose voice is cloned, or
     *     {@code null}/empty for the model's default voice; refused by a model pair that cannot clone
     *     a voice ({@link #supportsSpeakerReference()} is {@code false}, e.g. LFM3-Audio)
     * @param language the model's language: Qwen3-TTS takes names ({@code "english"}, {@code
     *     "chinese"}, ...; {@code null}/empty is {@code "english"}), LFM3-Audio codes ({@code "en"},
     *     {@code "de"}, ...; {@code null}/empty is its template's default, {@code "en"})
     * @param voice a model-dependent voice preset, e.g. {@code "female"} (the default) or {@code
     *     "male"} for LFM3-Audio, or {@code null}/empty for the model's default
     * @param maxFrames cap on generated audio frames (longer = longer audio)
     * @param topK top-k sampling cutoff for the semantic (backbone) token stream; LFM3-Audio samples
     *     its frames with the settings of its model card instead
     * @param seed sampler seed
     * @return a mono 16-bit WAV byte stream at the pipeline's sample rate
     */
    public byte[] synthesize(
            String text,
            @Nullable String speakerReferenceAudioPath,
            @Nullable String language,
            @Nullable String voice,
            int maxFrames,
            int topK,
            int seed) {
        lifecycleLock.readLock().lock();
        try {
            requireOpen();
            return backend.synthesize(handle, text, speakerReferenceAudioPath, language, voice, maxFrames, topK, seed);
        } finally {
            lifecycleLock.readLock().unlock();
        }
    }

    /** Throws when {@link #close()} has freed the engine; called with the read lock held. */
    private void requireOpen() {
        if (handle == 0L) {
            throw new IllegalStateException("TextToSpeech is closed");
        }
    }

    @Override
    public void close() {
        lifecycleLock.writeLock().lock();
        try {
            if (handle != 0L) {
                backend.delete(handle);
                handle = 0L;
            }
        } finally {
            lifecycleLock.writeLock().unlock();
        }
    }

    private static native long loadNative(String modelPath, String mmprojPath, int gpuLayers, int threads);

    private static native byte[] synthesizeNative(
            long handle,
            String text,
            @Nullable String speakerReferenceAudioPath,
            @Nullable String language,
            @Nullable String voice,
            int maxFrames,
            int topK,
            int seed);

    private static native boolean supportsSpeakerReferenceNative(long handle);

    private static native void deleteNative(long handle);
}
