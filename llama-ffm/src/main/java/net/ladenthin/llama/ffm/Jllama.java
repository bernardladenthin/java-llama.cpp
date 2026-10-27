// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT
package net.ladenthin.llama.ffm;

import static java.lang.foreign.ValueLayout.JAVA_BOOLEAN;
import static net.ladenthin.llama.ffm.JllamaNative.call;
import static net.ladenthin.llama.ffm.JllamaNative.cString;
import static net.ladenthin.llama.ffm.JllamaNative.invoke;
import static net.ladenthin.llama.ffm.JllamaNative.pointerSlot;
import static net.ladenthin.llama.ffm.JllamaNative.readString;
import static net.ladenthin.llama.ffm.JllamaNative.takeString;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.util.Objects;
import java.util.function.BiConsumer;
import net.ladenthin.llama.exception.LlamaException;
import net.ladenthin.llama.value.LogLevel;

/**
 * The model-free calls of the C API: version and build information, the process-wide log callback,
 * the JSON-schema-to-grammar conversion, quantisation and fine-tuning.
 */
public final class Jllama {

    /** The upcall stub of the current log callback, kept alive until the next swap. */
    private static Arena logArena;

    private Jllama() {}

    /**
     * The {@code net.ladenthin:llama} version the loaded library was built as ({@code x.y.z} or
     * {@code x.y.z-SNAPSHOT}); the C API is versioned with the library.
     *
     * @return the version string of {@code jllama_api_version()}
     */
    public static String apiVersion() {
        return readString((MemorySegment) invoke(JllamaNative.API_VERSION));
    }

    /**
     * The llama.cpp build inside the loaded library, {@code b<number>-<commit>}.
     *
     * @return the build string of {@code jllama_build_info()}
     */
    public static String buildInfo() {
        return readString((MemorySegment) invoke(JllamaNative.BUILD_INFO));
    }

    /**
     * Routes every log line of the library (llama.cpp, ggml, the server) to {@code callback} instead of
     * the console; {@code null} restores the console after flushing the lines queued for the previous
     * callback, synchronously. Process-wide, and the same sink as {@code LlamaModel.setLogger} of the
     * core: the two replace each other, so a later {@code setLogger} silently detaches this callback. The
     * callback runs on the logger's own worker thread, never on the thread that logged, and gets the bare
     * formatted message (no prefix, no timestamp; a {@code CONT} line arrives as {@link LogLevel#INFO}).
     * Anything it throws, an {@code Error} included, is swallowed: an exception crossing an FFM upcall
     * would end the JVM.
     *
     * @param callback the sink, or {@code null} for the console
     */
    public static synchronized void setLogCallback(BiConsumer<LogLevel, String> callback) {
        Arena previous = logArena;
        if (callback == null) {
            call(JllamaNative.SET_LOG_CALLBACK, MemorySegment.NULL, MemorySegment.NULL);
            logArena = null;
        } else {
            Arena arena = Arena.ofShared();
            try {
                MethodHandle target = MethodHandles.lookup()
                        .findStatic(Jllama.class, "onLog", MethodType.methodType(void.class, BiConsumer.class,
                                int.class, MemorySegment.class, MemorySegment.class))
                        .bindTo(callback);
                MemorySegment stub = JllamaNative.LINKER.upcallStub(target, JllamaNative.LOG_FN, arena);
                call(JllamaNative.SET_LOG_CALLBACK, stub, MemorySegment.NULL);
            } catch (NoSuchMethodException | IllegalAccessException e) {
                arena.close();
                throw new IllegalStateException(e);
            } catch (RuntimeException e) {
                arena.close();
                throw e;
            }
            logArena = arena;
        }
        // The swap drained the previous sink before returning, so its stub is no longer reachable.
        if (previous != null) {
            previous.close();
        }
    }

    /**
     * Chooses how the CPU backend runs the expert matrix multiplications of a mixture-of-experts model
     * over repacked weights: batched per expert with the work handed out dynamically ({@code true}, the
     * default) or as upstream llama.cpp does, one multiplication per token over a fixed split
     * ({@code false}). Process-wide and the same switch as {@code LlamaModel.setCpuMoeBatched} of the
     * core, whose documentation has the details; {@code GGML_CPU_REPACK_MMID_BATCHED=0} in the
     * environment makes {@code false} the initial value.
     *
     * @param batched {@code true} for the batched path, {@code false} for upstream's
     * @throws LlamaException if the loaded library was built without the switch
     */
    public static void setCpuMoeBatched(boolean batched) {
        call(JllamaNative.SET_CPU_MOE_BATCHED, batched);
    }

    /**
     * Whether the CPU backend runs the expert matrix multiplications batched; see
     * {@link #setCpuMoeBatched(boolean)}.
     *
     * @return {@code true} for the batched path (the default), {@code false} for upstream's
     * @throws LlamaException if the loaded library was built without the switch
     */
    public static boolean isCpuMoeBatched() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment out = arena.allocate(JAVA_BOOLEAN);
            call(JllamaNative.GET_CPU_MOE_BATCHED, out);
            return out.get(JAVA_BOOLEAN, 0);
        }
    }

    @SuppressWarnings("unused") // the upcall target of setLogCallback
    private static void onLog(BiConsumer<LogLevel, String> callback, int level, MemorySegment text,
            MemorySegment userData) {
        try {
            callback.accept(logLevel(level), readString(text));
        } catch (Throwable ignored) {
            // Nothing may cross the upcall boundary: a RuntimeException or an Error alike would end the JVM.
        }
    }

    /** The core's {@link LogLevel} for a {@code jllama_log_level} value. */
    static LogLevel logLevel(int level) {
        return switch (level) {
            case 1 -> LogLevel.DEBUG;
            case 3 -> LogLevel.WARN;
            case 4 -> LogLevel.ERROR;
            default -> LogLevel.INFO;
        };
    }

    /**
     * Converts a JSON schema to a GBNF grammar, as llama.cpp's own converter does.
     *
     * @param schemaJson the JSON schema
     * @return the grammar text
     * @throws LlamaException when the schema cannot be converted
     */
    public static String jsonSchemaToGrammar(String schemaJson) {
        Objects.requireNonNull(schemaJson, "schemaJson");
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment out = pointerSlot(arena);
            call(JllamaNative.JSON_SCHEMA_TO_GRAMMAR, cString(arena, schemaJson), out);
            return takeString(out);
        }
    }

    /**
     * Quantises a GGUF file, as {@code LlamaQuantizer} does.
     *
     * @param inputPath the model to read
     * @param outputPath the model to write
     * @param ftype the target {@code llama_ftype} value (e.g. 7 for Q8_0)
     * @param nthread the threads to use, {@code 0} for the default
     * @param allowRequantize whether an already quantised tensor may be quantised again
     * @throws LlamaException when the quantisation fails
     */
    public static void quantize(String inputPath, String outputPath, int ftype, int nthread, boolean allowRequantize) {
        Objects.requireNonNull(inputPath, "inputPath");
        Objects.requireNonNull(outputPath, "outputPath");
        try (Arena arena = Arena.ofConfined()) {
            call(JllamaNative.QUANTIZE, cString(arena, inputPath), cString(arena, outputPath), ftype, nthread,
                    allowRequantize);
        }
    }

    /**
     * Runs a fine-tuning job from the trainer's JSON configuration (the keys {@code LlamaTrainer}
     * writes, see the core's {@code TrainingField}).
     *
     * @param configJson the configuration object
     * @throws LlamaException when the run fails
     */
    public static void finetune(String configJson) {
        Objects.requireNonNull(configJson, "configJson");
        try (Arena arena = Arena.ofConfined()) {
            call(JllamaNative.FINETUNE, cString(arena, configJson));
        }
    }
}
