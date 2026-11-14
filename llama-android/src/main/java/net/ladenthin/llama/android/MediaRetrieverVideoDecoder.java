// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.android;

import android.graphics.Bitmap;
import android.media.MediaDataSource;
import android.media.MediaMetadataRetriever;
import java.io.IOException;
import java.nio.ByteBuffer;
import net.ladenthin.llama.callback.VideoDecoder;
import net.ladenthin.llama.callback.VideoStream;

/**
 * Video input on Android: the frames of a video come from the platform's own
 * {@link MediaMetadataRetriever}, which decodes with the codecs of the device.
 *
 * <p>llama.cpp reads a video by running the {@code ffmpeg} program, which an app cannot do. A model
 * loaded with this decoder takes video all the same:
 *
 * <pre>{@code
 * LlamaModel model = new LlamaModel(parameters.setMmproj(projector), null, new MediaRetrieverVideoDecoder());
 * model.chatCompleteText(new InferenceParameters("").withMessages(Collections.singletonList(
 *         ChatMessage.userMultimodal(ContentPart.text("What happens here?"),
 *                 ContentPart.videoBytes(bytes, "video/mp4")))));
 * }</pre>
 *
 * <p>Nothing is decoded here: this class asks the platform for frame number <em>i</em>
 * ({@link MediaMetadataRetriever#getFrameAtIndex(int, MediaMetadataRetriever.BitmapParams)}, API 28,
 * the floor of this artifact) and copies its pixels. What the device plays, it reads: the containers
 * and codecs are the platform's, and the rotation of a recording is applied by it.
 *
 * <p><strong>Cost.</strong> The model samples a few frames per second
 * ({@link net.ladenthin.llama.parameters.ModelParameters#setVideoFps(float)}, 4 by default), and
 * every sampled frame is one seek-and-decode by the platform -- tens of milliseconds each on a
 * phone -- and as many prompt tokens as an image. The whole file is held in memory while it is read.
 *
 * <p>Stateless and safe to share between models.
 */
public final class MediaRetrieverVideoDecoder implements VideoDecoder {

    private static final int BYTES_PER_PIXEL = 3;
    private static final int MILLIS_PER_SECOND = 1000;
    private static final int QUARTER_TURN = 90;
    private static final int THREE_QUARTER_TURN = 270;

    /** Creates the decoder. */
    public MediaRetrieverVideoDecoder() {}

    /**
     * Opens a video with the platform's retriever.
     *
     * @param data the bytes of the video file
     * @return the stream, or {@code null} when the platform finds no video in these bytes
     */
    @Override
    public VideoStream open(byte[] data) {
        MediaMetadataRetriever retriever = new MediaMetadataRetriever();
        try {
            retriever.setDataSource(new ByteArraySource(data));
            if (!"yes".equals(retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_HAS_VIDEO))) {
                release(retriever);
                return null;
            }
            int width = number(retriever, MediaMetadataRetriever.METADATA_KEY_VIDEO_WIDTH);
            int height = number(retriever, MediaMetadataRetriever.METADATA_KEY_VIDEO_HEIGHT);
            int rotation = number(retriever, MediaMetadataRetriever.METADATA_KEY_VIDEO_ROTATION);
            int frames = number(retriever, MediaMetadataRetriever.METADATA_KEY_VIDEO_FRAME_COUNT);
            int durationMillis = number(retriever, MediaMetadataRetriever.METADATA_KEY_DURATION);
            if (width <= 0 || height <= 0 || frames <= 0 || durationMillis <= 0) {
                release(retriever);
                return null;
            }
            // the size as displayed: a recording held upright is stored lying on its side
            boolean sideways = rotation == QUARTER_TURN || rotation == THREE_QUARTER_TURN;
            return new RetrieverStream(
                    retriever,
                    sideways ? height : width,
                    sideways ? width : height,
                    frames,
                    durationMillis / (float) MILLIS_PER_SECOND);
        } catch (RuntimeException e) {
            // setDataSource refuses what is no media file with an IllegalArgumentException or a
            // plain RuntimeException, depending on the release
            release(retriever);
            return null;
        }
    }

    private static int number(MediaMetadataRetriever retriever, int key) {
        String text = retriever.extractMetadata(key);
        if (text == null) {
            return 0;
        }
        try {
            return Integer.parseInt(text.trim());
        } catch (NumberFormatException e) {
            return 0;
        }
    }

    private static void release(MediaMetadataRetriever retriever) {
        try {
            retriever.release();
        } catch (IOException | RuntimeException e) {
            // nothing left to release
        }
    }

    /** One opened video: frame {@code i} on demand, scaled and copied as RGB. */
    private static final class RetrieverStream implements VideoStream {

        private final MediaMetadataRetriever retriever;
        private final MediaMetadataRetriever.BitmapParams params;
        private final int width;
        private final int height;
        private final int frames;
        private final float durationSeconds;
        private int next;
        private int[] pixels = new int[0];

        RetrieverStream(MediaMetadataRetriever retriever, int width, int height, int frames, float durationSeconds) {
            this.retriever = retriever;
            this.width = width;
            this.height = height;
            this.frames = frames;
            this.durationSeconds = durationSeconds;
            this.params = new MediaMetadataRetriever.BitmapParams();
            // the default is RGB_565, which would cost the low bits of every colour
            this.params.setPreferredConfig(Bitmap.Config.ARGB_8888);
        }

        @Override
        public int getWidth() {
            return width;
        }

        @Override
        public int getHeight() {
            return height;
        }

        @Override
        public float getFramesPerSecond() {
            return frames / durationSeconds;
        }

        @Override
        public int getFrameCount() {
            return frames;
        }

        @Override
        public float getDurationSeconds() {
            return durationSeconds;
        }

        @Override
        public boolean readFrame(ByteBuffer rgb, int frameWidth, int frameHeight) throws IOException {
            if (next >= frames) {
                return false;
            }
            int index = next++;
            if (rgb == null) {
                return true; // skipped: nothing was decoded for it
            }
            Bitmap frame = retriever.getFrameAtIndex(index, params);
            if (frame == null) {
                throw new IOException("the platform returned no frame " + index + " of " + frames);
            }
            Bitmap scaled = frame;
            if (frame.getWidth() != frameWidth || frame.getHeight() != frameHeight) {
                scaled = Bitmap.createScaledBitmap(frame, frameWidth, frameHeight, true);
            }
            int count = frameWidth * frameHeight;
            if (pixels.length != count) {
                pixels = new int[count];
            }
            scaled.getPixels(pixels, 0, frameWidth, 0, 0, frameWidth, frameHeight);
            if (scaled != frame) {
                scaled.recycle();
            }
            frame.recycle();
            if (rgb.remaining() < count * BYTES_PER_PIXEL) {
                throw new IOException("the frame buffer holds " + rgb.remaining() + " bytes, the frame needs "
                        + count * BYTES_PER_PIXEL);
            }
            for (int i = 0; i < count; i++) {
                int argb = pixels[i];
                rgb.put((byte) (argb >> 16));
                rgb.put((byte) (argb >> 8));
                rgb.put((byte) argb);
            }
            return true;
        }

        @Override
        public void close() {
            release(retriever);
        }
    }

    /** The bytes of a request as a data source the retriever can seek in. */
    private static final class ByteArraySource extends MediaDataSource {

        private final byte[] data;

        ByteArraySource(byte[] data) {
            this.data = data;
        }

        @Override
        public int readAt(long position, byte[] buffer, int offset, int size) {
            if (position >= data.length) {
                return -1;
            }
            int count = (int) Math.min(size, data.length - position);
            System.arraycopy(data, (int) position, buffer, offset, count);
            return count;
        }

        @Override
        public long getSize() {
            return data.length;
        }

        @Override
        public void close() {
            // the array belongs to the caller
        }
    }
}
