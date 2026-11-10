// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import java.io.IOException;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.attribute.PosixFilePermissions;
import java.util.List;
import net.ladenthin.llama.TextToSpeech;
import org.jspecify.annotations.Nullable;

/** The {@link SpeechBackend} over a loaded {@link TextToSpeech}, with its default sampling settings. */
final class TextToSpeechBackend implements SpeechBackend {

    /** The frame cap, top-k and seed of {@link TextToSpeech#synthesize(String)}. */
    private static final int MAX_FRAMES = 512;

    private static final int TOP_K = 4;
    private static final int SEED = 0;

    private static final String REFERENCE_PREFIX = "jllama-speaker-ref";
    private static final String REFERENCE_SUFFIX = ".bin";

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
                referenceFile = writeReferenceFile(reference);
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
                deleteReferenceFile(referenceFile);
            }
        }
    }

    /**
     * Writes a speaker reference into a temporary file only its owner can read: the clip is the
     * caller's voice, and the temporary directory is shared with every other user of the machine.
     *
     * @param reference the bytes of the clip
     * @return the file, to be removed with {@link #deleteReferenceFile(Path)}
     * @throws IOException when the file cannot be created or written
     */
    static Path writeReferenceFile(byte[] reference) throws IOException {
        Path file;
        if (FileSystems.getDefault().supportedFileAttributeViews().contains("posix")) {
            file = Files.createTempFile(
                    REFERENCE_PREFIX,
                    REFERENCE_SUFFIX,
                    PosixFilePermissions.asFileAttribute(PosixFilePermissions.fromString("rw-------")));
        } else {
            // Windows: the temporary directory is the user's own profile, and the file inherits its ACL
            file = Files.createTempFile(REFERENCE_PREFIX, REFERENCE_SUFFIX);
        }
        try {
            Files.write(file, reference);
        } catch (IOException | RuntimeException e) {
            deleteReferenceFile(file);
            throw e;
        }
        return file;
    }

    /**
     * Removes a file of {@link #writeReferenceFile(byte[])}; one that cannot be removed now (still
     * open on Windows) is removed when the JVM exits.
     *
     * @param file the file
     */
    static void deleteReferenceFile(Path file) {
        try {
            Files.deleteIfExists(file);
        } catch (IOException e) {
            file.toFile().deleteOnExit();
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
