// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.callback;

import java.io.IOException;
import java.nio.ByteBuffer;
import org.jspecify.annotations.Nullable;

/**
 * One opened video of a {@link VideoDecoder}: what it is, and its frames in presentation order.
 *
 * <p>The model reads it in this order: the description ({@link #getWidth()} and the other getters),
 * {@link #start(int, int, float)} once, {@link #readFrame(ByteBuffer, int, int)} until it has the
 * frames it wants or the stream ends, {@link #close()}. All of it happens on one thread, which is
 * not the caller's (see {@link VideoDecoder}).
 */
public interface VideoStream extends AutoCloseable {

    /**
     * The width of the video as displayed: rotation and pixel aspect ratio applied.
     *
     * @return the width in pixels, greater than zero
     */
    int getWidth();

    /**
     * The height of the video as displayed.
     *
     * @return the height in pixels, greater than zero
     */
    int getHeight();

    /**
     * The frame rate of the video.
     *
     * @return frames per second, greater than zero
     */
    float getFramesPerSecond();

    /**
     * The number of frames of the video.
     *
     * @return the frame count, or {@code -1} when it is unknown
     */
    default int getFrameCount() {
        return -1;
    }

    /**
     * The length of the video.
     *
     * @return the duration in seconds, or {@code -1} when it is unknown
     */
    default float getDurationSeconds() {
        return -1.0f;
    }

    /**
     * Called once before the first frame, with the size every frame will be asked in and the frame
     * rate the model samples at ({@code 0}: every frame).
     *
     * <p>A stream that delivers frames at {@code framesPerSecond} itself returns {@code true}, and
     * {@link #readFrame(ByteBuffer, int, int)} is then asked for exactly the frames the model takes.
     * Otherwise (the default) the model picks frames by their index and skips the others, each
     * skipped frame being one {@code readFrame} call with a {@code null} buffer.
     *
     * @param width the width frames will be asked in
     * @param height the height frames will be asked in
     * @param framesPerSecond the rate the model samples at, or {@code 0} for every frame
     * @return {@code true} if this stream resamples to {@code framesPerSecond} itself
     * @throws IOException when the stream cannot deliver frames; the request is refused
     */
    default boolean start(int width, int height, float framesPerSecond) throws IOException {
        return false;
    }

    /**
     * Delivers or skips the next frame.
     *
     * @param rgb where to write the frame: {@code width * height * 3} bytes, RGB, rows top to
     *     bottom, no padding, scaled to {@code width} x {@code height}. A direct buffer over the
     *     model's own memory, valid during this call only; {@code null} to skip the frame
     * @param width the width of the frame to write
     * @param height the height of the frame to write
     * @return {@code false} at the end of the video
     * @throws IOException when the frame cannot be read; treated as the end of the video
     */
    boolean readFrame(@Nullable ByteBuffer rgb, int width, int height) throws IOException;

    /** Releases what the stream holds. Called once, also after a failure. */
    @Override
    void close();
}
