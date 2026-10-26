// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.is;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
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
        volatile boolean speakerReference;
        volatile boolean holdSynthesize;
        volatile boolean deletedDuringSynthesize;

        @Override
        public long load(String modelPath, String mmprojPath, int gpuLayers, int threads) {
            return HANDLE;
        }

        @Override
        public byte[] synthesize(
                long handle,
                String text,
                @Nullable String speakerReferenceAudioPath,
                @Nullable String language,
                int maxFrames,
                int topK,
                int seed) {
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
