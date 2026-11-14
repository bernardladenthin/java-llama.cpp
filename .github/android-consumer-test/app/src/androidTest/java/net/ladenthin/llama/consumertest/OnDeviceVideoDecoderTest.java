// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

package net.ladenthin.llama.consumertest;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.platform.app.InstrumentationRegistry;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import net.ladenthin.llama.android.MediaRetrieverVideoDecoder;
import net.ladenthin.llama.callback.VideoStream;
import org.junit.Test;
import org.junit.runner.RunWith;

/**
 * On-emulator test of the AAR's video decoder: {@link MediaRetrieverVideoDecoder} asks the
 * platform's {@code MediaMetadataRetriever} for the frames of the core's test clip (two seconds
 * at four frames per second, 128x96, H.264 in MP4; an asset of this test).
 *
 * <p>It drives the decoder the way llama.cpp does -- the description, {@code start}, then
 * {@code readFrame} with a direct buffer or with {@code null} to skip -- and checks what comes
 * back. It does not run a model: the emulator job has no vision model, so that the frames reach a
 * model is what the JVM test {@code VideoDecoderIntegrationTest} shows with a stand-in decoder.
 */
@RunWith(AndroidJUnit4.class)
public class OnDeviceVideoDecoderTest {

    private static final int WIDTH = 128;
    private static final int HEIGHT = 96;
    private static final int FRAMES = 8;

    private static byte[] clip() throws IOException {
        try (InputStream in = InstrumentationRegistry.getInstrumentation()
                        .getContext()
                        .getAssets()
                        .open("sample.mp4");
                ByteArrayOutputStream out = new ByteArrayOutputStream()) {
            byte[] buffer = new byte[8192];
            for (int n = in.read(buffer); n >= 0; n = in.read(buffer)) {
                out.write(buffer, 0, n);
            }
            return out.toByteArray();
        }
    }

    @Test
    public void thePlatformDescribesTheClip() throws Exception {
        VideoStream stream = new MediaRetrieverVideoDecoder().open(clip());
        assertNotNull("the platform found no video in the clip", stream);
        try {
            assertEquals(WIDTH, stream.getWidth());
            assertEquals(HEIGHT, stream.getHeight());
            assertEquals(FRAMES, stream.getFrameCount());
            assertEquals(2.0f, stream.getDurationSeconds(), 0.3f);
            assertEquals(4.0f, stream.getFramesPerSecond(), 0.6f);
        } finally {
            stream.close();
        }
    }

    @Test
    public void everyFrameIsDeliveredAsRgbAndTheStreamEndsAfterTheLast() throws Exception {
        VideoStream stream = new MediaRetrieverVideoDecoder().open(clip());
        assertNotNull(stream);
        try {
            // the decoder leaves the sampling to the model
            assertFalse(stream.start(WIDTH, HEIGHT, 4.0f));
            ByteBuffer rgb = ByteBuffer.allocateDirect(WIDTH * HEIGHT * 3);
            byte[] first = null;
            byte[] last = null;
            for (int i = 0; i < FRAMES; i++) {
                rgb.clear();
                assertTrue("frame " + i, stream.readFrame(rgb, WIDTH, HEIGHT));
                assertEquals("frame " + i + " did not fill the buffer", 0, rgb.remaining());
                byte[] frame = new byte[rgb.capacity()];
                rgb.flip();
                rgb.get(frame);
                // ffmpeg's test picture is colour bars: a decoded frame is neither black nor one colour
                int min = 255;
                int max = 0;
                for (byte b : frame) {
                    min = Math.min(min, b & 0xFF);
                    max = Math.max(max, b & 0xFF);
                }
                assertTrue("frame " + i + " is flat (" + min + ".." + max + ")", max - min > 100);
                if (i == 0) {
                    first = frame;
                }
                last = frame;
            }
            // the picture carries a running counter, so the first and the last frame differ
            assertFalse("the first and the last frame are the same picture", Arrays.equals(first, last));
            rgb.clear();
            assertFalse("a ninth frame", stream.readFrame(rgb, WIDTH, HEIGHT));
        } finally {
            stream.close();
        }
    }

    @Test
    public void aFrameIsScaledToTheSizeAskedForAndSkippedFramesCount() throws Exception {
        VideoStream stream = new MediaRetrieverVideoDecoder().open(clip());
        assertNotNull(stream);
        try {
            stream.start(64, 48, 2.0f);
            ByteBuffer rgb = ByteBuffer.allocateDirect(64 * 48 * 3);
            int read = 0;
            // every second frame, the way the model samples a 4 fps clip at 2 fps
            for (int i = 0; i < FRAMES; i++) {
                if (i % 2 == 0) {
                    rgb.clear();
                    assertTrue(stream.readFrame(rgb, 64, 48));
                    assertEquals(0, rgb.remaining());
                    read++;
                } else {
                    assertTrue(stream.readFrame(null, 64, 48));
                }
            }
            assertEquals(FRAMES / 2, read);
            assertFalse(stream.readFrame(null, 64, 48));
        } finally {
            stream.close();
        }
    }

    @Test
    public void bytesThatAreNoVideoOpenNothing() {
        assertNull(new MediaRetrieverVideoDecoder().open("not a video".getBytes(StandardCharsets.UTF_8)));
        assertNull(new MediaRetrieverVideoDecoder().open(new byte[0]));
    }
}
