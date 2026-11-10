// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.equalTo;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.nullValue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.attribute.AclEntry;
import java.nio.file.attribute.AclEntryType;
import java.nio.file.attribute.FileAttribute;
import java.nio.file.attribute.PosixFilePermissions;
import java.util.Collections;
import java.util.List;
import net.ladenthin.llama.TextToSpeech;
import org.jspecify.annotations.Nullable;
import org.junit.jupiter.api.Test;

/** The temporary file a speaker reference crosses to the native engine in; no native library. */
class TextToSpeechBackendTest {

    private static final byte[] CLIP = "RIFF-not-really".getBytes(StandardCharsets.US_ASCII);

    @Test
    void theReferenceFileHoldsTheClipAndIsGoneAfterTheDelete() throws IOException {
        Path file = TextToSpeechBackend.writeReferenceFile(CLIP);
        try {
            assertThat(Files.readAllBytes(file), is(equalTo(CLIP)));
        } finally {
            TextToSpeechBackend.deleteReferenceFile(file);
        }
        assertThat(Files.exists(file), is(false));
    }

    @Test
    void theReferenceFileIsReadableByItsOwnerOnly() throws IOException {
        assumeTrue(FileSystems.getDefault().supportedFileAttributeViews().contains("posix"));
        Path file = TextToSpeechBackend.writeReferenceFile(CLIP);
        try {
            assertThat(PosixFilePermissions.toString(Files.getPosixFilePermissions(file)), is(equalTo("rw-------")));
        } finally {
            TextToSpeechBackend.deleteReferenceFile(file);
        }
    }

    @Test
    void deletingAFileThatIsAlreadyGoneIsNotAnError() throws IOException {
        Path file = TextToSpeechBackend.writeReferenceFile(CLIP);
        TextToSpeechBackend.deleteReferenceFile(file);
        TextToSpeechBackend.deleteReferenceFile(file);
        assertThat(Files.exists(file), is(false));
    }

    @Test
    void aFileSystemWithAccessControlListsGetsOneEntryForTheOwner() {
        FileAttribute<?>[] attributes = TextToSpeechBackend.ownerOnlyAttributes(Collections.singleton("acl"));
        assertThat(attributes.length, is(equalTo(1)));
        assertThat(attributes[0].name(), is(equalTo("acl:acl")));
        List<?> entries = (List<?>) attributes[0].value();
        assertThat(entries.size(), is(equalTo(1)));
        AclEntry entry = (AclEntry) entries.get(0);
        assertThat(entry.type(), is(equalTo(AclEntryType.ALLOW)));
        assertThat(entry.principal().getName(), is(equalTo(System.getProperty("user.name"))));
    }

    @Test
    void aFileSystemWithPosixPermissionsGetsThem() {
        FileAttribute<?>[] attributes = TextToSpeechBackend.ownerOnlyAttributes(Collections.singleton("posix"));
        assertThat(attributes.length, is(equalTo(1)));
        assertThat(attributes[0].name(), is(equalTo("posix:permissions")));
    }

    @Test
    void aFileSystemWithNeitherGetsNoAttribute() {
        assertThat(TextToSpeechBackend.ownerOnlyAttributes(Collections.singleton("basic")).length, is(equalTo(0)));
    }

    /** Records what the backend hands the engine, and reads the reference file while it exists. */
    private static final class RecordingEngine implements TextToSpeechBackend.Engine {
        @Nullable
        String referencePath;

        byte @Nullable [] referenceBytes;

        @Nullable
        String voice;

        @Nullable
        String language;

        int maxFrames;
        int topK;
        int seed;
        float temperature;

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
            this.referencePath = speakerReferenceAudioPath;
            this.language = language;
            this.voice = voice;
            this.maxFrames = maxFrames;
            this.topK = topK;
            this.seed = seed;
            this.temperature = temperature;
            if (speakerReferenceAudioPath != null) {
                try {
                    this.referenceBytes = Files.readAllBytes(Paths.get(speakerReferenceAudioPath));
                } catch (IOException e) {
                    throw new UncheckedIOException(e);
                }
            }
            return text.getBytes(StandardCharsets.US_ASCII);
        }

        @Override
        public List<String> voices() {
            return Collections.singletonList("alba");
        }
    }

    @Test
    void aRequestWithoutOptionsUsesTheDefaultsAndNoReferenceFile() throws IOException {
        RecordingEngine engine = new RecordingEngine();
        TextToSpeechBackend backend = new TextToSpeechBackend(engine);
        assertThat(backend.synthesize("hi", "alba", "en"), is(equalTo("hi".getBytes(StandardCharsets.US_ASCII))));
        assertThat(engine.referencePath, is(nullValue()));
        assertThat(engine.voice, is(equalTo("alba")));
        assertThat(engine.language, is(equalTo("en")));
        assertThat(engine.maxFrames, is(equalTo(512)));
        assertThat(engine.topK, is(equalTo(4)));
        assertThat(engine.seed, is(equalTo(0)));
        assertThat(engine.temperature, is(equalTo(TextToSpeech.MODEL_DEFAULT_TEMPERATURE)));
        assertThat(backend.voices(), is(equalTo(Collections.singletonList("alba"))));
        assertThat(backend.toString(), containsString("maxFrames=512"));
    }

    @Test
    void theOptionsOfARequestReachTheEngineAndTheReferenceFileIsGoneAfterwards() throws IOException {
        RecordingEngine engine = new RecordingEngine();
        TextToSpeechBackend backend = new TextToSpeechBackend(engine);
        backend.synthesize("hi", null, null, new SpeechOptions(CLIP, 7, 9, 11, 0.5f));
        assertThat(engine.referenceBytes, is(equalTo(CLIP)));
        assertThat(Files.exists(Paths.get(engine.referencePath)), is(false));
        assertThat(engine.maxFrames, is(equalTo(7)));
        assertThat(engine.topK, is(equalTo(9)));
        assertThat(engine.seed, is(equalTo(11)));
        assertThat(engine.temperature, is(equalTo(0.5f)));
    }

    @Test
    void aFrameCapOrTopKOfZeroFallsBackToTheDefault() throws IOException {
        RecordingEngine engine = new RecordingEngine();
        new TextToSpeechBackend(engine).synthesize("hi", null, null, new SpeechOptions(null, 0, 0, null, null));
        assertThat(engine.maxFrames, is(equalTo(512)));
        assertThat(engine.topK, is(equalTo(4)));
    }
}
