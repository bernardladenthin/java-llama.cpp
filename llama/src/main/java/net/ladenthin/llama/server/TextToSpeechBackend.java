// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import java.io.IOException;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.attribute.AclEntry;
import java.nio.file.attribute.AclEntryPermission;
import java.nio.file.attribute.AclEntryType;
import java.nio.file.attribute.FileAttribute;
import java.nio.file.attribute.PosixFilePermissions;
import java.nio.file.attribute.UserPrincipal;
import java.util.Collections;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;
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

    /** What this backend needs of a {@link TextToSpeech}; the seam its tests fake. */
    interface Engine {

        /**
         * {@link TextToSpeech#synthesize(String, String, String, String, int, int, int, float)}.
         *
         * @param text the text to speak
         * @param speakerReferenceAudioPath a clip of the voice to clone, or {@code null}
         * @param language the language, or {@code null}
         * @param voice a predefined voice, or {@code null}
         * @param maxFrames the frame cap
         * @param topK the sampler's top-k
         * @param seed the sampler's seed
         * @param temperature the audio frame temperature
         * @return the WAV bytes
         */
        byte[] synthesize(
                String text,
                @Nullable String speakerReferenceAudioPath,
                @Nullable String language,
                @Nullable String voice,
                int maxFrames,
                int topK,
                int seed,
                float temperature);

        /**
         * {@link TextToSpeech#getVoices()}.
         *
         * @return the predefined voices
         */
        List<String> voices();
    }

    private final Engine speech;

    TextToSpeechBackend(final TextToSpeech speech) {
        this(new Engine() {
            @Override
            public byte[] synthesize(
                    String text,
                    @Nullable String speakerReferenceAudioPath,
                    @Nullable String language,
                    @Nullable String voice,
                    int maxFrames,
                    int topK,
                    int seed,
                    float temperature) {
                return speech.synthesize(
                        text, speakerReferenceAudioPath, language, voice, maxFrames, topK, seed, temperature);
            }

            @Override
            public List<String> voices() {
                return speech.getVoices();
            }
        });
    }

    TextToSpeechBackend(Engine speech) {
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
        Path file = Files.createTempFile(
                REFERENCE_PREFIX,
                REFERENCE_SUFFIX,
                ownerOnlyAttributes(
                        FileSystems.getDefault().supportedFileAttributeViews(), System.getProperty("user.name")));
        try {
            Files.write(file, reference);
        } catch (IOException | RuntimeException e) {
            deleteReferenceFile(file);
            throw e;
        }
        return file;
    }

    /**
     * The attributes a file is created with so that only its owner can open it: POSIX permissions
     * where the file system has them, else an access control list naming the owner alone (Windows),
     * else none (a file system with neither has no way to say it).
     *
     * @param views the attribute views the file system supports
     * @param userName the name of the user the file is for
     * @return the attributes for {@link Files#createTempFile(String, String, FileAttribute...)}
     * @throws IOException when the file system has access control lists and the user name resolves
     *     to no principal: the file would then get what its directory hands down
     */
    static FileAttribute<?>[] ownerOnlyAttributes(Set<String> views, String userName) throws IOException {
        if (views.contains("posix")) {
            return new FileAttribute<?>[] {
                PosixFilePermissions.asFileAttribute(PosixFilePermissions.fromString("rw-------"))
            };
        }
        if (views.contains("acl")) {
            try {
                UserPrincipal owner =
                        FileSystems.getDefault().getUserPrincipalLookupService().lookupPrincipalByName(userName);
                final List<AclEntry> acl = Collections.singletonList(AclEntry.newBuilder()
                        .setType(AclEntryType.ALLOW)
                        .setPrincipal(owner)
                        .setPermissions(EnumSet.allOf(AclEntryPermission.class))
                        .build());
                return new FileAttribute<?>[] {
                    new FileAttribute<List<AclEntry>>() {
                        @Override
                        public String name() {
                            return "acl:acl";
                        }

                        @Override
                        public List<AclEntry> value() {
                            return acl;
                        }
                    }
                };
            } catch (IOException | RuntimeException e) {
                // no principal for the user name: without one the file would get whatever its
                // directory hands down, which in a shared temporary directory may be everybody
                throw new IOException("cannot restrict the speaker reference file to its owner", e);
            }
        }
        return new FileAttribute<?>[0];
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
        return speech.voices();
    }

    @Override
    public String toString() {
        return "TextToSpeechBackend{maxFrames=" + MAX_FRAMES + ", topK=" + TOP_K + ", seed=" + SEED + '}';
    }
}
