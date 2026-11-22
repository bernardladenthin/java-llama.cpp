// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.endsWith;
import static org.hamcrest.Matchers.equalTo;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.nullValue;
import static org.junit.jupiter.api.Assertions.assertThrows;
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
import net.ladenthin.llama.parameters.SpeechParameters;
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
    void aFileSystemWithAccessControlListsGetsOneEntryForTheOwner() throws IOException {
        FileAttribute<?>[] attributes =
                TextToSpeechBackend.ownerOnlyAttributes(Collections.singleton("acl"), System.getProperty("user.name"));
        assertThat(attributes.length, is(equalTo(1)));
        assertThat(attributes[0].name(), is(equalTo("acl:acl")));
        List<?> entries = (List<?>) attributes[0].value();
        assertThat(entries.size(), is(equalTo(1)));
        AclEntry entry = (AclEntry) entries.get(0);
        assertThat(entry.type(), is(equalTo(AclEntryType.ALLOW)));
        // Windows names a principal "<machine or domain>\\<user>"
        assertThat(entry.principal().getName(), endsWith(System.getProperty("user.name")));
    }

    @Test
    void aUserNameThatResolvesToNoPrincipalFailsInsteadOfWritingAnOpenFile() {
        // the file would get whatever its directory hands down, which may be everybody
        assertThrows(
                IOException.class,
                () -> TextToSpeechBackend.ownerOnlyAttributes(
                        Collections.singleton("acl"), "no-such-user-jllama-\u00e4-" + System.nanoTime()));
    }

    @Test
    void aFileSystemWithPosixPermissionsGetsThem() throws IOException {
        FileAttribute<?>[] attributes = TextToSpeechBackend.ownerOnlyAttributes(Collections.singleton("posix"), "x");
        assertThat(attributes.length, is(equalTo(1)));
        assertThat(attributes[0].name(), is(equalTo("posix:permissions")));
    }

    @Test
    void aFileSystemWithNeitherGetsNoAttribute() throws IOException {
        assertThat(TextToSpeechBackend.ownerOnlyAttributes(Collections.singleton("basic"), "x").length, is(equalTo(0)));
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
        float topP;
        float repeatPenalty;
        float speed;
        float loudness;
        float cfgScale;
        int flowSteps;
        boolean rawPrompt;

        @Nullable
        String referenceText;

        @Override
        public byte[] synthesize(String text, SpeechParameters parameters) {
            String speakerReferenceAudioPath = parameters.getSpeakerReferenceAudioPath();
            this.referencePath = speakerReferenceAudioPath;
            this.language = parameters.getLanguage();
            this.voice = parameters.getVoice();
            this.maxFrames = parameters.getMaxFrames();
            this.topK = parameters.getTopK();
            this.seed = parameters.getSeed();
            this.temperature = parameters.getTemperature();
            this.topP = parameters.getTopP();
            this.repeatPenalty = parameters.getRepeatPenalty();
            this.speed = parameters.getSpeed();
            this.loudness = parameters.getLoudness();
            this.referenceText = parameters.getSpeakerReferenceText();
            this.cfgScale = parameters.getCfgScale();
            this.flowSteps = parameters.getFlowSteps();
            this.rawPrompt = parameters.isRawPrompt();
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
        assertThat(engine.maxFrames, is(equalTo(0)));
        assertThat(engine.topK, is(equalTo(4)));
        assertThat(engine.seed, is(equalTo(0)));
        assertThat(engine.temperature, is(equalTo(TextToSpeech.MODEL_DEFAULT_TEMPERATURE)));
        assertThat(engine.topP, is(equalTo(SpeechParameters.TOP_P_OFF)));
        assertThat(engine.repeatPenalty, is(equalTo(SpeechParameters.REPEAT_PENALTY_OFF)));
        assertThat(engine.speed, is(equalTo(SpeechParameters.MODEL_DEFAULT_SPEED)));
        assertThat(engine.loudness, is(equalTo(SpeechParameters.MODEL_DEFAULT_LOUDNESS)));
        assertThat(engine.referenceText, is(nullValue()));
        assertThat(engine.cfgScale, is(equalTo(0.0f)));
        assertThat(engine.flowSteps, is(equalTo(0)));
        assertThat(engine.rawPrompt, is(false));
        assertThat(backend.voices(), is(equalTo(Collections.singletonList("alba"))));
        assertThat(backend.toString(), containsString("maxFrames=0"));
    }

    @Test
    void aRawPromptReachesTheEngine() throws IOException {
        RecordingEngine engine = new RecordingEngine();
        new TextToSpeechBackend(engine)
                .synthesize("hi", null, null, SpeechOptions.defaults().withRawPrompt(true));
        assertThat(engine.rawPrompt, is(true));
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
    void theSamplingAndDeliveryOfARequestReachTheEngine() throws IOException {
        RecordingEngine engine = new RecordingEngine();
        new TextToSpeechBackend(engine)
                .synthesize("hi", null, null, new SpeechOptions(null, null, null, null, null, 0.9f, 1.1f, 1.5f, -20f));
        assertThat(engine.topP, is(equalTo(0.9f)));
        assertThat(engine.repeatPenalty, is(equalTo(1.1f)));
        assertThat(engine.speed, is(equalTo(1.5f)));
        assertThat(engine.loudness, is(equalTo(-20f)));
    }

    @Test
    void theGuidanceAndSolverStepsOfARequestReachTheEngine() throws IOException {
        RecordingEngine engine = new RecordingEngine();
        new TextToSpeechBackend(engine)
                .synthesize("hi", null, null, SpeechOptions.defaults().withGenerationControl(1.5f, 16));
        assertThat(engine.cfgScale, is(equalTo(1.5f)));
        assertThat(engine.flowSteps, is(equalTo(16)));
    }

    @Test
    void theTranscriptOfTheReferenceReachesTheEngine() throws IOException {
        RecordingEngine engine = new RecordingEngine();
        SpeechOptions options =
                new SpeechOptions(CLIP, null, null, null, null).withSpeakerReferenceText("What the clip says.");
        new TextToSpeechBackend(engine).synthesize("hi", null, null, options);
        assertThat(engine.referenceText, is(equalTo("What the clip says.")));
        assertThat(engine.referenceBytes, is(equalTo(CLIP)));
        // the copy keeps every other option, the original stays without a transcript
        assertThat(options.getSpeakerReference(), is(equalTo(CLIP)));
        assertThat(new SpeechOptions(CLIP, null, null, null, null).getSpeakerReferenceText(), is(nullValue()));
    }

    @Test
    void aFrameCapOfZeroLeavesItToTheModelAndATopKOfZeroFallsBackToTheDefault() throws IOException {
        RecordingEngine engine = new RecordingEngine();
        new TextToSpeechBackend(engine).synthesize("hi", null, null, new SpeechOptions(null, 0, 0, null, null));
        assertThat(engine.maxFrames, is(equalTo(0)));
        assertThat(engine.topK, is(equalTo(4)));
    }
}
