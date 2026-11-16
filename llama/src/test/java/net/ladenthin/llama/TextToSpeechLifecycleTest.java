// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.sameInstance;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.Arrays;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import net.ladenthin.llama.parameters.SpeechParameters;
import org.jspecify.annotations.Nullable;
import org.junit.jupiter.api.Test;

@ClaudeGenerated(
        purpose = "Pin that TextToSpeech.close() never frees the native engine while synthesize() is "
                + "still inside it: the engine's own mutex is part of the freed memory, so a close() "
                + "during a synthesis was a use-after-free; and that supportsSpeakerReference() asks the "
                + "open engine and never a closed one. Driven through the Backend seam with a "
                + "fake, so no made-up handle reaches libjllama.")
public class TextToSpeechLifecycleTest {

    private static final long HANDLE = 7L;

    private static final long WAIT_SECONDS = 5L;

    private static final byte[] WAV = {1, 2, 3};

    /** A backend whose synthesize can be held inside the "native" call. */
    static class FakeBackend implements TextToSpeech.Backend {
        final CountDownLatch synthesizeEntered = new CountDownLatch(1);
        final CountDownLatch releaseSynthesize = new CountDownLatch(1);
        final AtomicInteger insideSynthesize = new AtomicInteger();
        final AtomicInteger deletes = new AtomicInteger();
        final AtomicInteger speakerReferenceQueries = new AtomicInteger();
        final AtomicInteger voiceQueries = new AtomicInteger();
        volatile String[] voices = new String[0];
        volatile boolean speakerReference;
        volatile boolean holdSynthesize;
        volatile boolean deletedDuringSynthesize;
        volatile @Nullable String lastLanguage;
        volatile @Nullable String lastVoice;
        volatile float lastTemperature = Float.NaN;
        volatile @Nullable SpeechParameters lastParameters;

        @Override
        public long load(String modelPath, String mmprojPath, int gpuLayers, int threads) {
            return HANDLE;
        }

        @Override
        public byte[] synthesize(long handle, String text, SpeechParameters parameters) {
            lastParameters = parameters;
            lastTemperature = parameters.getTemperature();
            lastLanguage = parameters.getLanguage();
            lastVoice = parameters.getVoice();
            insideSynthesize.incrementAndGet();
            try {
                synthesizeEntered.countDown();
                if (holdSynthesize) {
                    try {
                        releaseSynthesize.await(WAIT_SECONDS, TimeUnit.SECONDS);
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                    }
                }
                return WAV.clone();
            } finally {
                insideSynthesize.decrementAndGet();
            }
        }

        @Override
        public boolean supportsSpeakerReference(long handle) {
            speakerReferenceQueries.incrementAndGet();
            return speakerReference;
        }

        @Override
        public String[] voices(long handle) {
            voiceQueries.incrementAndGet();
            return voices;
        }

        @Override
        public void delete(long handle) {
            if (insideSynthesize.get() > 0) {
                deletedDuringSynthesize = true;
            }
            deletes.incrementAndGet();
        }
    }

    @Test
    public void close_whileSynthesizeIsInsideTheEngine_freesItOnlyAfterSynthesizeReturned() throws Exception {
        FakeBackend backend = new FakeBackend();
        backend.holdSynthesize = true;
        TextToSpeech tts = new TextToSpeech(backend, "model.gguf", "mmproj.gguf", 0, 0);
        Thread speaker = new Thread(() -> tts.synthesize("hello"), "speaker");
        speaker.start();
        try {
            assertThat(backend.synthesizeEntered.await(WAIT_SECONDS, TimeUnit.SECONDS), is(true));

            Thread closer = new Thread(tts::close, "closer");
            closer.start();
            awaitWaitingOrDone(closer);
            backend.releaseSynthesize.countDown();
            speaker.join(TimeUnit.SECONDS.toMillis(WAIT_SECONDS));
            closer.join(TimeUnit.SECONDS.toMillis(WAIT_SECONDS));

            assertThat(
                    "the engine was freed while synthesize() still ran in it",
                    backend.deletedDuringSynthesize,
                    is(false));
            assertThat(backend.deletes.get(), is(1));
        } finally {
            backend.releaseSynthesize.countDown();
            speaker.join(TimeUnit.SECONDS.toMillis(WAIT_SECONDS));
            tts.close();
        }
    }

    @Test
    public void synthesize_afterClose_throwsIllegalState() {
        FakeBackend backend = new FakeBackend();
        TextToSpeech tts = new TextToSpeech(backend, "model.gguf", "mmproj.gguf", 0, 0);
        tts.close();
        assertThrows(IllegalStateException.class, () -> tts.synthesize("hello"));
    }

    @Test
    public void close_calledTwice_freesTheEngineOnce() {
        FakeBackend backend = new FakeBackend();
        TextToSpeech tts = new TextToSpeech(backend, "model.gguf", "mmproj.gguf", 0, 0);
        tts.close();
        tts.close();
        assertThat(backend.deletes.get(), is(1));
    }

    @Test
    public void synthesize_whileOpen_returnsTheBackendsWav() {
        FakeBackend backend = new FakeBackend();
        try (TextToSpeech tts = new TextToSpeech(backend, "model.gguf", "mmproj.gguf", 0, 0)) {
            assertThat(tts.synthesize("hello"), is(WAV));
        }
    }

    @Test
    public void synthesize_withVoice_handsLanguageAndVoiceToTheBackend() {
        FakeBackend backend = new FakeBackend();
        try (TextToSpeech tts = new TextToSpeech(backend, "model.gguf", "mmproj.gguf", 0, 0)) {
            assertThat(tts.synthesize("hello", null, "de", "male", 64, 4, 0), is(WAV));
            assertThat(backend.lastLanguage, is("de"));
            assertThat(backend.lastVoice, is("male"));
        }
    }

    @Test
    public void synthesize_withTemperature_handsItToTheBackend() {
        FakeBackend backend = new FakeBackend();
        try (TextToSpeech tts = new TextToSpeech(backend, "model.gguf", "mmproj.gguf", 0, 0)) {
            assertThat(tts.synthesize("hello", null, null, "alba", 64, 4, 0, 0.7f), is(WAV));
            assertThat(backend.lastTemperature, is(0.7f));
            assertThat(backend.lastVoice, is("alba"));
        }
    }

    @Test
    public void synthesize_withoutTemperature_leavesItToTheModel() {
        FakeBackend backend = new FakeBackend();
        try (TextToSpeech tts = new TextToSpeech(backend, "model.gguf", "mmproj.gguf", 0, 0)) {
            tts.synthesize("hello");
            assertThat(backend.lastTemperature, is(TextToSpeech.MODEL_DEFAULT_TEMPERATURE));
            assertThat(TextToSpeech.MODEL_DEFAULT_TEMPERATURE < 0.0f, is(true));
        }
    }

    @Test
    public void synthesize_withParameters_handsEveryOptionToTheBackend() {
        FakeBackend backend = new FakeBackend();
        SpeechParameters parameters = SpeechParameters.builder()
                .speakerReferenceAudioPath("ref.wav")
                .language("en")
                .voice("female")
                .maxFrames(64)
                .topK(7)
                .seed(42)
                .temperature(0.5f)
                .topP(0.9f)
                .repeatPenalty(1.1f)
                .speed(1.25f)
                .loudness(-20.0f)
                .build();
        try (TextToSpeech tts = new TextToSpeech(backend, "model.gguf", "mmproj.gguf", 0, 0)) {
            assertThat(tts.synthesize("hello", parameters), is(WAV));
            assertThat(backend.lastParameters, is(sameInstance(parameters)));
        }
    }

    @Test
    public void synthesize_shortOverloads_passTheDefaultSamplingAndDelivery() {
        FakeBackend backend = new FakeBackend();
        try (TextToSpeech tts = new TextToSpeech(backend, "model.gguf", "mmproj.gguf", 0, 0)) {
            tts.synthesize("hello");
            SpeechParameters p = backend.lastParameters;
            assertThat(p.getMaxFrames(), is(512));
            assertThat(p.getTopK(), is(4));
            assertThat(p.getSeed(), is(0));
            assertThat(p.getTopP(), is(SpeechParameters.TOP_P_OFF));
            assertThat(p.getRepeatPenalty(), is(SpeechParameters.REPEAT_PENALTY_OFF));
            assertThat(p.getSpeed(), is(1.0f));
            assertThat(p.getLoudness(), is(0.0f));

            tts.synthesize("hello", "ref.wav", "de", "male", 64, 9, 3, 0.25f);
            p = backend.lastParameters;
            assertThat(p.getSpeakerReferenceAudioPath(), is("ref.wav"));
            assertThat(p.getMaxFrames(), is(64));
            assertThat(p.getTopK(), is(9));
            assertThat(p.getSeed(), is(3));
            assertThat(p.getTemperature(), is(0.25f));
            assertThat(p.getSpeed(), is(SpeechParameters.MODEL_DEFAULT_SPEED));
            assertThat(p.getLoudness(), is(SpeechParameters.MODEL_DEFAULT_LOUDNESS));
        }
    }

    @Test
    public void synthesize_withParameters_afterClose_throwsWithoutCallingTheBackend() {
        FakeBackend backend = new FakeBackend();
        TextToSpeech tts = new TextToSpeech(backend, "model.gguf", "mmproj.gguf", 0, 0);
        tts.close();
        assertThrows(IllegalStateException.class, () -> tts.synthesize("hello", SpeechParameters.defaults()));
        assertThat(backend.lastParameters, is((SpeechParameters) null));
    }

    @Test
    public void synthesize_withoutVoice_handsNoVoiceToTheBackend() {
        FakeBackend backend = new FakeBackend();
        try (TextToSpeech tts = new TextToSpeech(backend, "model.gguf", "mmproj.gguf", 0, 0)) {
            tts.synthesize("hello", null, "english", 64, 4, 0);
            assertThat(backend.lastLanguage, is("english"));
            assertThat(backend.lastVoice, is((String) null));
        }
    }

    @Test
    public void supportsSpeakerReference_whileOpen_returnsTheBackendsAnswer() {
        FakeBackend backend = new FakeBackend();
        try (TextToSpeech tts = new TextToSpeech(backend, "model.gguf", "mmproj.gguf", 0, 0)) {
            assertThat(tts.supportsSpeakerReference(), is(false));
            backend.speakerReference = true;
            assertThat(tts.supportsSpeakerReference(), is(true));
        }
    }

    @Test
    public void supportsSpeakerReference_afterClose_throwsWithoutAskingTheFreedEngine() {
        FakeBackend backend = new FakeBackend();
        TextToSpeech tts = new TextToSpeech(backend, "model.gguf", "mmproj.gguf", 0, 0);
        tts.close();
        assertThrows(IllegalStateException.class, tts::supportsSpeakerReference);
        assertThat(backend.speakerReferenceQueries.get(), is(0));
    }

    @Test
    public void getVoices_whileOpen_returnsTheBackendsNamesInOrder() {
        FakeBackend backend = new FakeBackend();
        try (TextToSpeech tts = new TextToSpeech(backend, "model.gguf", "mmproj.gguf", 0, 0)) {
            assertThat(tts.getVoices().isEmpty(), is(true));
            backend.voices = new String[] {"alba", "marius"};
            assertThat(tts.getVoices(), is(Arrays.asList("alba", "marius")));
        }
    }

    @Test
    public void getVoices_returnsAListTheCallerCannotChangeTheModelsVoicesThrough() {
        FakeBackend backend = new FakeBackend();
        backend.voices = new String[] {"alba"};
        try (TextToSpeech tts = new TextToSpeech(backend, "model.gguf", "mmproj.gguf", 0, 0)) {
            List<String> voices = tts.getVoices();
            assertThrows(UnsupportedOperationException.class, () -> voices.add("x"));
            assertThrows(UnsupportedOperationException.class, () -> voices.set(0, "x"));
            assertThat(backend.voices[0], is("alba"));
        }
    }

    @Test
    public void getVoices_afterClose_throwsWithoutAskingTheFreedEngine() {
        FakeBackend backend = new FakeBackend();
        TextToSpeech tts = new TextToSpeech(backend, "model.gguf", "mmproj.gguf", 0, 0);
        tts.close();
        assertThrows(IllegalStateException.class, tts::getVoices);
        assertThat(backend.voiceQueries.get(), is(0));
    }

    /** Waits until the thread waits for a lock or has finished, whichever comes first. */
    private static void awaitWaitingOrDone(Thread thread) throws InterruptedException {
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(WAIT_SECONDS);
        while (thread.isAlive()
                && thread.getState() != Thread.State.BLOCKED
                && thread.getState() != Thread.State.WAITING
                && System.nanoTime() < deadline) {
            thread.join(10);
        }
    }
}
