// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.callback;

import java.io.IOException;
import org.jspecify.annotations.Nullable;

/**
 * Hands the frames of a video to the model, in place of the {@code ffmpeg} program.
 *
 * <p>llama.cpp decodes the video of a request by running {@code ffmpeg} and {@code ffprobe}, which
 * an Android app cannot do and a server may not have. A model loaded with a {@code VideoDecoder}
 * ({@link net.ladenthin.llama.LlamaModel#LlamaModel(net.ladenthin.llama.parameters.ModelParameters,
 * LoadProgressCallback, VideoDecoder)}) asks it instead: for every video a request carries,
 * {@link #open(byte[])} is called with the bytes of the file, and the returned {@link VideoStream}
 * is read frame by frame and closed.
 *
 * <p>This library implements no decoder. An implementation takes the frames from a media library
 * that is there anyway: on Android the platform's own ({@code MediaRetrieverVideoDecoder} of the
 * {@code llama-android} artifact), elsewhere whatever the application ships.
 *
 * <p><strong>Threads.</strong> The calls do not arrive on the thread that made the request:
 * llama.cpp reads the media of a request on a thread of its own, which is attached to the JVM as a
 * daemon thread named {@code jllama-video}. An implementation must not rely on thread-locals or on
 * the context class loader of the caller. One stream is used by one thread; {@code open} may be
 * called while another stream is still open. The decoder is used until the model is closed.
 */
@FunctionalInterface
public interface VideoDecoder {

    /**
     * Opens a video.
     *
     * @param data the bytes of the video file, as the request carried them
     * @return the stream to read the frames from, or {@code null} when these bytes are no video
     *     this decoder reads (the request is then refused as invalid)
     * @throws IOException when the video cannot be opened; treated like {@code null}
     */
    @Nullable
    VideoStream open(byte[] data) throws IOException;
}
