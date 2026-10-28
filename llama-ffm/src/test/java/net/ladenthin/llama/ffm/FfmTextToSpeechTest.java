// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT
package net.ladenthin.llama.ffm;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.contains;
import static org.hamcrest.Matchers.is;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.lang.foreign.MemorySegment;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;

/**
 * The lifecycle locking of {@link FfmTextToSpeech}, over a fake backend: no library, no model.
 *
 * <p>The native free deletes the engine (and the mutex a synthesis holds), and nothing counts users on
 * the native side. Without a lock, {@code close()} freed it while a {@code synthesize()} on another
 * thread was still inside its native call on it — the use-after-free F16 fixed for the JNI
 * {@code TextToSpeech}.
 */
class FfmTextToSpeechTest {

    /** Upper bound for the waits below; reaching it is a failure, not a slow machine. */
    private static final long AWAIT_SECONDS = 10;

    /** How long {@code close()} is given to show that it does not wait. */
    private static final long CLOSE_GRACE_MILLIS = 500;

    private static final byte[] WAV = {1, 2, 3};

    /** A backend that records the calls and holds a synthesis until it is released. */
    private static final class GatedBackend implements FfmTextToSpeech.Backend {

        final List<String> events = new CopyOnWriteArrayList<>();
        final CountDownLatch insideSynthesize = new CountDownLatch(1);
        final CountDownLatch releaseSynthesize = new CountDownLatch(1);
        volatile boolean speakerReference;

        @Override
        public byte[] synthesize(MemorySegment handle, String text, String speakerReferencePath, String language,
                String voice, int maxFrames, int topK, int seed) {
            events.add("synthesize entered");
            insideSynthesize.countDown();
            try {
                releaseSynthesize.await(AWAIT_SECONDS, TimeUnit.SECONDS);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            events.add("synthesize returned");
            return WAV;
        }

        @Override
        public boolean supportsSpeakerReference(MemorySegment handle) {
            events.add("supportsSpeakerReference");
            return speakerReference;
        }

        @Override
        public void free(MemorySegment handle) {
            events.add("free (deletes the engine)");
        }
    }

    @Test
    void closeWaitsForASynthesisInsideTheNativeCall() throws Exception {
        GatedBackend backend = new GatedBackend();
        FfmTextToSpeech tts = new FfmTextToSpeech(MemorySegment.NULL, backend);

        Thread speaker = new Thread(() -> tts.synthesize("hi", null, null, 10, 1, 1), "synthesize");
        speaker.start();
        assertThat(backend.insideSynthesize.await(AWAIT_SECONDS, TimeUnit.SECONDS), is(true));
        Thread closer = new Thread(tts::close, "close");
        closer.start();
        closer.join(CLOSE_GRACE_MILLIS);
        boolean closeWaited = closer.isAlive();
        backend.releaseSynthesize.countDown();
        speaker.join(TimeUnit.SECONDS.toMillis(AWAIT_SECONDS));
        closer.join(TimeUnit.SECONDS.toMillis(AWAIT_SECONDS));

        assertThat(closeWaited, is(true));
        assertThat(
                backend.events,
                contains("synthesize entered", "synthesize returned", "free (deletes the engine)"));
    }

    @Test
    void synthesizeAfterCloseIsRefusedWithoutTouchingTheHandle() {
        GatedBackend backend = new GatedBackend();
        backend.releaseSynthesize.countDown();
        FfmTextToSpeech tts = new FfmTextToSpeech(MemorySegment.NULL, backend);

        tts.close();

        assertThrows(IllegalStateException.class, () -> tts.synthesize("hi", null, null, 10, 1, 1));
        assertThat(backend.events, contains("free (deletes the engine)"));
    }

    @Test
    void closeTwiceFreesOnce() {
        GatedBackend backend = new GatedBackend();
        FfmTextToSpeech tts = new FfmTextToSpeech(MemorySegment.NULL, backend);

        tts.close();
        tts.close();

        assertThat(backend.events, contains("free (deletes the engine)"));
    }

    @Test
    void synthesizeReturnsTheBackendsWav() {
        GatedBackend backend = new GatedBackend();
        backend.releaseSynthesize.countDown();
        FfmTextToSpeech tts = new FfmTextToSpeech(MemorySegment.NULL, backend);

        assertThat(tts.synthesize("hi", null, null, 10, 1, 1), is(WAV));
    }

    @Test
    void supportsSpeakerReferenceReturnsTheBackendsAnswer() {
        GatedBackend backend = new GatedBackend();
        FfmTextToSpeech tts = new FfmTextToSpeech(MemorySegment.NULL, backend);

        assertThat(tts.supportsSpeakerReference(), is(false));
        backend.speakerReference = true;
        assertThat(tts.supportsSpeakerReference(), is(true));
    }

    @Test
    void supportsSpeakerReferenceAfterCloseIsRefusedWithoutTouchingTheHandle() {
        GatedBackend backend = new GatedBackend();
        FfmTextToSpeech tts = new FfmTextToSpeech(MemorySegment.NULL, backend);

        tts.close();

        assertThrows(IllegalStateException.class, tts::supportsSpeakerReference);
        assertThat(backend.events, contains("free (deletes the engine)"));
    }

    @Test
    void synthesizeWithoutTextIsRefused() {
        FfmTextToSpeech tts = new FfmTextToSpeech(MemorySegment.NULL, new GatedBackend());

        assertThrows(NullPointerException.class, () -> tts.synthesize(null, null, null, 10, 1, 1));
    }
}
