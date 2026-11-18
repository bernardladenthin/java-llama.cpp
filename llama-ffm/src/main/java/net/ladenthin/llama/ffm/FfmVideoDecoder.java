// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT
package net.ladenthin.llama.ffm;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_BOOLEAN;
import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.StructLayout;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.nio.ByteBuffer;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import net.ladenthin.llama.callback.VideoDecoder;
import net.ladenthin.llama.callback.VideoStream;

/**
 * The {@code jllama_video_decoder} of {@link FfmLlamaModel#load(String[],
 * net.ladenthin.llama.callback.LoadProgressCallback, VideoDecoder)}: the core's {@link VideoDecoder} behind
 * upcall stubs, what {@code jllama.cpp}'s {@code jni_video_*} functions are for {@code LlamaModel}.
 *
 * <p>llama.cpp calls the functions on a thread of its own (the server reads a request's media there),
 * so the stubs live in a shared arena, and the open streams are kept in a table: the state pointer
 * llama.cpp hands back is a key into it, never a Java object. Nothing may cross an upcall boundary --
 * an exception there ends the JVM -- so every bridge method catches everything and answers the
 * failure value instead, which refuses the request. {@code read_audio} is not offered: the core's
 * interface has no soundtrack. The arena outlives the model: {@link #close()} runs after
 * {@code jllama_model_free} (or a failed load), which is when llama.cpp calls {@code release}.
 */
final class FfmVideoDecoder implements AutoCloseable {

    /** {@code jllama_video_info}: what {@code open} reports about the stream. */
    static final StructLayout VIDEO_INFO = MemoryLayout.structLayout(
            JAVA_INT.withName("width"),
            JAVA_INT.withName("height"),
            JAVA_FLOAT.withName("fps"),
            JAVA_INT.withName("n_frames"),
            JAVA_FLOAT.withName("duration"));

    /** {@code jllama_video_decoder}: six function pointers and the user data. */
    static final StructLayout DECODER = MemoryLayout.structLayout(
            ADDRESS.withName("open"),
            ADDRESS.withName("start"),
            ADDRESS.withName("read_frame"),
            ADDRESS.withName("read_audio"),
            ADDRESS.withName("close"),
            ADDRESS.withName("release"),
            ADDRESS.withName("user_data"));

    /** {@code void *(*open)(void *user_data, const char *path, const unsigned char *data, size_t, info *)}. */
    static final FunctionDescriptor OPEN_FN =
            FunctionDescriptor.of(ADDRESS, ADDRESS, ADDRESS, ADDRESS, JAVA_LONG, ADDRESS);
    /** {@code bool (*start)(void *state, unsigned w, unsigned h, float fps, bool *fps_picked)}. */
    static final FunctionDescriptor START_FN =
            FunctionDescriptor.of(JAVA_BOOLEAN, ADDRESS, JAVA_INT, JAVA_INT, JAVA_FLOAT, ADDRESS);
    /** {@code bool (*read_frame)(void *state, unsigned char *rgb, unsigned w, unsigned h)}. */
    static final FunctionDescriptor READ_FRAME_FN =
            FunctionDescriptor.of(JAVA_BOOLEAN, ADDRESS, ADDRESS, JAVA_INT, JAVA_INT);
    /** {@code void (*close)(void *state)} and {@code void (*release)(void *user_data)}. */
    static final FunctionDescriptor POINTER_FN = FunctionDescriptor.ofVoid(ADDRESS);

    private final VideoDecoder decoder;
    private final Map<Long, VideoStream> streams = new ConcurrentHashMap<>();
    private final AtomicLong nextState = new AtomicLong(1);
    private final AtomicBoolean released = new AtomicBoolean();
    private final Arena arena = Arena.ofShared();
    private MemorySegment struct = MemorySegment.NULL;

    FfmVideoDecoder(VideoDecoder decoder) {
        this.decoder = Objects.requireNonNull(decoder, "decoder");
    }

    /** The {@code jllama_video_decoder} struct with its upcall stubs, built once. */
    synchronized MemorySegment struct() {
        if (struct.equals(MemorySegment.NULL)) {
            Linker linker = Linker.nativeLinker();
            MemorySegment segment = arena.allocate(DECODER);
            segment.set(ADDRESS, DECODER.byteOffset(path("open")), linker.upcallStub(target("onOpen",
                    MethodType.methodType(MemorySegment.class, MemorySegment.class, MemorySegment.class,
                            MemorySegment.class, long.class, MemorySegment.class)), OPEN_FN, arena));
            segment.set(ADDRESS, DECODER.byteOffset(path("start")), linker.upcallStub(target("onStart",
                    MethodType.methodType(boolean.class, MemorySegment.class, int.class, int.class, float.class,
                            MemorySegment.class)), START_FN, arena));
            segment.set(ADDRESS, DECODER.byteOffset(path("read_frame")), linker.upcallStub(target("onReadFrame",
                    MethodType.methodType(boolean.class, MemorySegment.class, MemorySegment.class, int.class,
                            int.class)), READ_FRAME_FN, arena));
            segment.set(ADDRESS, DECODER.byteOffset(path("read_audio")), MemorySegment.NULL);
            segment.set(ADDRESS, DECODER.byteOffset(path("close")), linker.upcallStub(target("onClose",
                    MethodType.methodType(void.class, MemorySegment.class)), POINTER_FN, arena));
            segment.set(ADDRESS, DECODER.byteOffset(path("release")), linker.upcallStub(target("onRelease",
                    MethodType.methodType(void.class, MemorySegment.class)), POINTER_FN, arena));
            segment.set(ADDRESS, DECODER.byteOffset(path("user_data")), MemorySegment.NULL);
            struct = segment;
        }
        return struct;
    }

    private static MemoryLayout.PathElement path(String name) {
        return MemoryLayout.PathElement.groupElement(name);
    }

    private MethodHandle target(String name, MethodType type) {
        try {
            return MethodHandles.lookup().findVirtual(FfmVideoDecoder.class, name, type).bindTo(this);
        } catch (NoSuchMethodException | IllegalAccessException e) {
            throw new IllegalStateException(e);
        }
    }

    /** The upcall target of {@code open}; also called directly by the tests. */
    MemorySegment onOpen(MemorySegment userData, MemorySegment path, MemorySegment data, long size,
            MemorySegment info) {
        VideoStream stream = null;
        try {
            byte[] bytes = size == 0 ? new byte[0] : data.reinterpret(size).toArray(JAVA_BYTE);
            stream = decoder.open(bytes);
            if (stream == null) {
                return MemorySegment.NULL;
            }
            int width = stream.getWidth();
            int height = stream.getHeight();
            float fps = stream.getFramesPerSecond();
            if (width <= 0 || height <= 0 || !(fps > 0.0f)) {
                stream.close();
                return MemorySegment.NULL;
            }
            MemorySegment out = info.reinterpret(VIDEO_INFO.byteSize());
            out.set(JAVA_INT, VIDEO_INFO.byteOffset(path("width")), width);
            out.set(JAVA_INT, VIDEO_INFO.byteOffset(path("height")), height);
            out.set(JAVA_FLOAT, VIDEO_INFO.byteOffset(path("fps")), fps);
            out.set(JAVA_INT, VIDEO_INFO.byteOffset(path("n_frames")), stream.getFrameCount());
            out.set(JAVA_FLOAT, VIDEO_INFO.byteOffset(path("duration")), stream.getDurationSeconds());
            long key = nextState.getAndIncrement();
            streams.put(key, stream);
            return MemorySegment.ofAddress(key);
        } catch (Throwable e) {
            closeQuietly(stream);
            return MemorySegment.NULL;
        }
    }

    /** The upcall target of {@code start}. */
    boolean onStart(MemorySegment state, int width, int height, float fps, MemorySegment fpsPicked) {
        try {
            VideoStream stream = streams.get(state.address());
            if (stream == null) {
                return false;
            }
            boolean picked = stream.start(width, height, fps);
            fpsPicked.reinterpret(JAVA_BOOLEAN.byteSize()).set(JAVA_BOOLEAN, 0, picked);
            return true;
        } catch (Throwable e) {
            return false;
        }
    }

    /** The upcall target of {@code read_frame}: the frame memory as a direct buffer, or null to skip. */
    boolean onReadFrame(MemorySegment state, MemorySegment rgb, int width, int height) {
        try {
            VideoStream stream = streams.get(state.address());
            if (stream == null) {
                return false;
            }
            ByteBuffer buffer = rgb.equals(MemorySegment.NULL)
                    ? null
                    : rgb.reinterpret(3L * Integer.toUnsignedLong(width) * Integer.toUnsignedLong(height))
                            .asByteBuffer();
            return stream.readFrame(buffer, width, height);
        } catch (Throwable e) {
            return false;
        }
    }

    /** The upcall target of {@code close}. */
    void onClose(MemorySegment state) {
        closeQuietly(streams.remove(state.address()));
    }

    /** The upcall target of {@code release}: llama.cpp no longer uses the decoder. */
    void onRelease(MemorySegment userData) {
        released.set(true);
    }

    /** Whether llama.cpp has released the decoder. */
    boolean isReleased() {
        return released.get();
    }

    /** How many streams are open. */
    int openStreams() {
        return streams.size();
    }

    private static void closeQuietly(VideoStream stream) {
        if (stream == null) {
            return;
        }
        try {
            stream.close();
        } catch (Throwable e) {
            // a decoder's close must not take the library down
        }
    }

    /** Closes what is still open and frees the stubs; only after llama.cpp released the decoder. */
    @Override
    public void close() {
        for (Long key : streams.keySet()) {
            closeQuietly(streams.remove(key));
        }
        arena.close();
    }
}
