// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT
package net.ladenthin.llama.ffm;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_BOOLEAN;
import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.util.concurrent.atomic.AtomicInteger;
import net.ladenthin.llama.callback.VideoDecoder;
import net.ladenthin.llama.callback.VideoStream;
import org.junit.jupiter.api.Test;

/**
 * The bridge between {@code jllama_video_decoder} and the core's {@link VideoDecoder}, driven the way
 * llama.cpp drives it, without the native library: the upcall targets are called directly with
 * memory a test arena allocates, and the struct's layout is checked against the C declaration.
 */
class FfmVideoDecoderTest {

    /** A stream of {@code frames} frames of 4x2 pixels at 5 per second, each painted in its index. */
    private static class Stream implements VideoStream {
        final AtomicInteger closed = new AtomicInteger();
        final int frames;
        int next;
        boolean resamples;
        float rateAsked = -1;
        boolean throwOnStart;

        Stream(int frames) {
            this.frames = frames;
        }

        @Override
        public int getWidth() {
            return 4;
        }

        @Override
        public int getHeight() {
            return 2;
        }

        @Override
        public float getFramesPerSecond() {
            return 5.0f;
        }

        @Override
        public int getFrameCount() {
            return frames;
        }

        @Override
        public float getDurationSeconds() {
            return frames / 5.0f;
        }

        @Override
        public boolean start(int width, int height, float framesPerSecond) throws IOException {
            if (throwOnStart) {
                throw new IOException("cannot start");
            }
            rateAsked = framesPerSecond;
            return resamples;
        }

        @Override
        public boolean readFrame(ByteBuffer rgb, int width, int height) {
            if (next >= frames) {
                return false;
            }
            if (rgb != null) {
                assertTrue(rgb.isDirect());
                assertEquals(width * height * 3, rgb.remaining());
                while (rgb.hasRemaining()) {
                    rgb.put((byte) (10 + next));
                }
            }
            next++;
            return true;
        }

        @Override
        public void close() {
            closed.incrementAndGet();
        }
    }

    private static MemorySegment bytes(Arena arena, byte... data) {
        MemorySegment segment = arena.allocate(Math.max(1, data.length));
        MemorySegment.copy(data, 0, segment, JAVA_BYTE, 0, data.length);
        return segment;
    }

    private static long offset(String field) {
        return FfmVideoDecoder.VIDEO_INFO.byteOffset(MemoryLayout.PathElement.groupElement(field));
    }

    @Test
    void theStructHasTheLayoutOfTheCDeclarationAndOffersNoSoundtrack() {
        assertEquals(20, FfmVideoDecoder.VIDEO_INFO.byteSize(), "jllama_video_info: 5 four-byte fields");
        assertEquals(7 * ADDRESS.byteSize(), FfmVideoDecoder.DECODER.byteSize(), "six functions and user_data");
        try (FfmVideoDecoder video = new FfmVideoDecoder(data -> null)) {
            MemorySegment struct = video.struct();
            assertEquals(struct, video.struct(), "built once");
            for (String function : new String[] {"open", "start", "read_frame", "close", "release"}) {
                long at = FfmVideoDecoder.DECODER.byteOffset(MemoryLayout.PathElement.groupElement(function));
                assertNotEquals(MemorySegment.NULL, struct.get(ADDRESS, at), function);
            }
            long audio = FfmVideoDecoder.DECODER.byteOffset(MemoryLayout.PathElement.groupElement("read_audio"));
            assertEquals(MemorySegment.NULL, struct.get(ADDRESS, audio));
        }
    }

    @Test
    void aStreamIsOpenedFromTheRequestsBytesReadAndClosed() {
        Stream stream = new Stream(3);
        byte[][] seen = new byte[1][];
        try (Arena arena = Arena.ofConfined();
                FfmVideoDecoder video = new FfmVideoDecoder(data -> {
                    seen[0] = data;
                    return stream;
                })) {
            MemorySegment info = arena.allocate(FfmVideoDecoder.VIDEO_INFO);
            MemorySegment state = video.onOpen(MemorySegment.NULL, MemorySegment.NULL, bytes(arena, (byte) 1, (byte) 2,
                    (byte) 3), 3, info);
            assertNotEquals(MemorySegment.NULL, state);
            assertArrayEquals(new byte[] {1, 2, 3}, seen[0], "the bytes of the request, copied");
            assertEquals(4, info.get(JAVA_INT, offset("width")));
            assertEquals(2, info.get(JAVA_INT, offset("height")));
            assertEquals(5.0f, info.get(JAVA_FLOAT, offset("fps")));
            assertEquals(3, info.get(JAVA_INT, offset("n_frames")));
            assertEquals(0.6f, info.get(JAVA_FLOAT, offset("duration")), 1e-6);
            assertEquals(1, video.openStreams());

            MemorySegment picked = arena.allocate(JAVA_BOOLEAN);
            stream.resamples = true;
            assertTrue(video.onStart(state, 4, 2, 2.0f, picked));
            assertEquals(2.0f, stream.rateAsked);
            assertTrue(picked.get(JAVA_BOOLEAN, 0), "the stream resamples itself");

            MemorySegment rgb = arena.allocate(4 * 2 * 3);
            assertTrue(video.onReadFrame(state, rgb, 4, 2));
            assertEquals(10, rgb.get(JAVA_BYTE, 0));
            assertEquals(10, rgb.get(JAVA_BYTE, 23));
            assertTrue(video.onReadFrame(state, MemorySegment.NULL, 4, 2), "a null buffer skips a frame");
            assertTrue(video.onReadFrame(state, rgb, 4, 2));
            assertEquals(12, rgb.get(JAVA_BYTE, 5));
            assertFalse(video.onReadFrame(state, rgb, 4, 2), "the end of the stream");

            video.onClose(state);
            assertEquals(1, stream.closed.get());
            assertEquals(0, video.openStreams());
            assertFalse(video.onReadFrame(state, rgb, 4, 2), "a closed state is unknown");
            assertFalse(video.onStart(state, 4, 2, 2.0f, picked));
        }
    }

    @Test
    void whatTheDecoderCannotOpenIsTheNullStateAndNothingStaysOpen() throws Exception {
        Stream zeroRate = new Stream(1) {
            @Override
            public float getFramesPerSecond() {
                return 0.0f;
            }
        };
        VideoDecoder[] decoders = {
            data -> null,
            data -> {
                throw new IOException("not now");
            },
            data -> {
                throw new IllegalStateException("a bug in the decoder");
            },
            data -> zeroRate
        };
        try (Arena arena = Arena.ofConfined()) {
            for (VideoDecoder decoder : decoders) {
                try (FfmVideoDecoder video = new FfmVideoDecoder(decoder)) {
                    MemorySegment state = video.onOpen(MemorySegment.NULL, MemorySegment.NULL, bytes(arena, (byte) 7),
                            1, arena.allocate(FfmVideoDecoder.VIDEO_INFO));
                    assertEquals(MemorySegment.NULL, state);
                    assertEquals(0, video.openStreams());
                }
            }
        }
        assertEquals(1, zeroRate.closed.get(), "a stream it refuses is closed");
    }

    @Test
    void aStreamThatFailsToStartAnswersFalseAndCloseClosesWhatIsStillOpen() {
        Stream stream = new Stream(2);
        stream.throwOnStart = true;
        FfmVideoDecoder video = new FfmVideoDecoder(data -> stream);
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment state = video.onOpen(MemorySegment.NULL, MemorySegment.NULL, bytes(arena), 0,
                    arena.allocate(FfmVideoDecoder.VIDEO_INFO));
            assertNotEquals(MemorySegment.NULL, state, "an empty file is the decoder's to judge");
            assertFalse(video.onStart(state, 4, 2, 0.0f, arena.allocate(JAVA_BOOLEAN)));
        }
        assertFalse(video.isReleased());
        video.onRelease(MemorySegment.NULL);
        assertTrue(video.isReleased());
        video.close();
        assertEquals(1, stream.closed.get(), "the model's free closes a stream llama.cpp left open");
    }
}
