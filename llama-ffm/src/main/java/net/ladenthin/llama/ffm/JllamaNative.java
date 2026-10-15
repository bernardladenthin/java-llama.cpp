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
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;
import java.nio.charset.StandardCharsets;
import java.util.List;
import net.ladenthin.llama.LlamaModel;
import net.ladenthin.llama.exception.InvalidRequestException;
import net.ladenthin.llama.exception.LlamaException;

/**
 * The {@code jllama_*} functions of {@code jllama_api.h} as method handles, one per declaration, plus
 * the few helpers every call needs: the argv and string marshalling, the owned-string and array
 * read-outs, and the status check that turns a failed call into the core's exceptions.
 *
 * <p>The library is loaded by the core's {@code LlamaLoader} (the same natives jars, {@code lib.path}
 * and build check as {@code LlamaModel}), reached through {@link LlamaModel}'s class initialization --
 * the loader's package is deliberately not exported by the core's module, {@code LlamaModel} is -- and
 * the symbols come from {@link SymbolLookup#loaderLookup()}, which sees what {@code System.load} put
 * into this class loader. {@code size_t} is {@code long}: every platform this project ships natives
 * for is 64-bit.
 */
final class JllamaNative {

    /** {@code jllama_status}: done. */
    static final int STATUS_OK = 0;
    /** {@code jllama_status}: a server-side failure. */
    static final int STATUS_ERROR = 1;
    /** {@code jllama_status}: the request is the caller's fault (what llama.cpp answers 400). */
    static final int STATUS_INVALID_REQUEST = 2;
    /** {@code jllama_status}: a null handle or out-parameter. */
    static final int STATUS_INVALID_ARGUMENT = 3;
    /** {@code jllama_status}: a wait cut short because the model is being freed. */
    static final int STATUS_INTERRUPTED = 4;

    /** {@code jllama_load_progress_fn}: {@code bool (*)(float progress, void *user_data)}. */
    static final FunctionDescriptor LOAD_PROGRESS_FN = FunctionDescriptor.of(JAVA_BOOLEAN, JAVA_FLOAT, ADDRESS);
    /** {@code jllama_log_fn}: {@code void (*)(int level, const char *text, void *user_data)}. */
    static final FunctionDescriptor LOG_FN = FunctionDescriptor.ofVoid(JAVA_INT, ADDRESS, ADDRESS);

    static final Linker LINKER = Linker.nativeLinker();
    private static final SymbolLookup LOOKUP;

    static {
        try {
            // LlamaModel's static initializer runs LlamaLoader.initialize(): the one loading path.
            Class.forName(LlamaModel.class.getName(), true, LlamaModel.class.getClassLoader());
        } catch (ClassNotFoundException e) {
            throw new IllegalStateException(e);
        }
        LOOKUP = SymbolLookup.loaderLookup();
    }

    // -- the model-free calls ---------------------------------------------------------------------
    static final MethodHandle API_VERSION = downcall("jllama_api_version", FunctionDescriptor.of(ADDRESS));
    static final MethodHandle BUILD_INFO = downcall("jllama_build_info", FunctionDescriptor.of(ADDRESS));
    static final MethodHandle LAST_ERROR = downcall("jllama_last_error", FunctionDescriptor.of(ADDRESS));
    static final MethodHandle STRING_FREE = downcall("jllama_string_free", FunctionDescriptor.ofVoid(ADDRESS));
    static final MethodHandle FLOATS_FREE = downcall("jllama_floats_free", FunctionDescriptor.ofVoid(ADDRESS));
    static final MethodHandle TOKENS_FREE = downcall("jllama_tokens_free", FunctionDescriptor.ofVoid(ADDRESS));
    static final MethodHandle STRINGS_FREE =
            downcall("jllama_strings_free", FunctionDescriptor.ofVoid(ADDRESS, JAVA_LONG));
    static final MethodHandle BYTES_FREE = downcall("jllama_bytes_free", FunctionDescriptor.ofVoid(ADDRESS));
    static final MethodHandle SET_LOG_CALLBACK =
            downcall("jllama_set_log_callback", FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS));
    static final MethodHandle JSON_SCHEMA_TO_GRAMMAR =
            downcall("jllama_json_schema_to_grammar", FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS));
    static final MethodHandle QUANTIZE = downcall(
            "jllama_quantize", FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, JAVA_INT, JAVA_INT, JAVA_BOOLEAN));
    static final MethodHandle FINETUNE = downcall("jllama_finetune", FunctionDescriptor.of(JAVA_INT, ADDRESS));

    // -- the model ----------------------------------------------------------------------------------
    static final MethodHandle MODEL_LOAD = downcall(
            "jllama_model_load", FunctionDescriptor.of(JAVA_INT, JAVA_INT, ADDRESS, ADDRESS, ADDRESS, ADDRESS));
    static final MethodHandle MODEL_FREE = downcall("jllama_model_free", FunctionDescriptor.ofVoid(ADDRESS));
    static final MethodHandle MODEL_META = downcall("jllama_model_meta", json(ADDRESS, ADDRESS));
    static final MethodHandle CHAT_COMPLETIONS = downcall("jllama_chat_completions", json(ADDRESS, ADDRESS, ADDRESS));
    static final MethodHandle CHAT_COMPLETIONS_STREAM_REQUEST =
            downcall("jllama_chat_completions_stream_request", json(ADDRESS, ADDRESS, ADDRESS));
    static final MethodHandle CHAT_COMPLETIONS_STREAM_RECEIVE =
            downcall("jllama_chat_completions_stream_receive", json(ADDRESS, JAVA_INT, ADDRESS, ADDRESS));
    static final MethodHandle CHAT_COMPLETIONS_REQUEST =
            downcall("jllama_chat_completions_request", json(ADDRESS, ADDRESS, ADDRESS));
    static final MethodHandle COMPLETIONS = downcall("jllama_completions", json(ADDRESS, ADDRESS, ADDRESS));
    static final MethodHandle COMPLETIONS_OAI = downcall("jllama_completions_oai", json(ADDRESS, ADDRESS, ADDRESS));
    static final MethodHandle INFILL = downcall("jllama_infill", json(ADDRESS, ADDRESS, ADDRESS));
    static final MethodHandle COMPLETIONS_STREAM_REQUEST =
            downcall("jllama_completions_stream_request", json(ADDRESS, ADDRESS, ADDRESS));
    static final MethodHandle COMPLETIONS_STREAM_RECEIVE =
            downcall("jllama_completions_stream_receive", json(ADDRESS, JAVA_INT, ADDRESS, ADDRESS));
    static final MethodHandle TASK_RELEASE = downcall("jllama_task_release", json(ADDRESS, JAVA_INT));
    static final MethodHandle APPLY_TEMPLATE = downcall("jllama_apply_template", json(ADDRESS, ADDRESS, ADDRESS));
    static final MethodHandle EMBED = downcall("jllama_embed", json(ADDRESS, ADDRESS, ADDRESS, ADDRESS));
    static final MethodHandle EMBEDDINGS = downcall("jllama_embeddings", json(ADDRESS, ADDRESS, JAVA_BOOLEAN, ADDRESS));
    static final MethodHandle RERANK = downcall("jllama_rerank", json(ADDRESS, ADDRESS, JAVA_INT, ADDRESS, ADDRESS));
    static final MethodHandle TOKENIZE = downcall("jllama_tokenize", json(ADDRESS, ADDRESS, JAVA_BOOLEAN, ADDRESS, ADDRESS));
    static final MethodHandle TOKENIZE_JSON =
            downcall("jllama_tokenize_json", json(ADDRESS, ADDRESS, JAVA_BOOLEAN, JAVA_BOOLEAN, ADDRESS));
    static final MethodHandle DETOKENIZE = downcall("jllama_detokenize", json(ADDRESS, ADDRESS, JAVA_LONG, ADDRESS, ADDRESS));
    static final MethodHandle DETOKENIZE_JSON = downcall("jllama_detokenize_json", json(ADDRESS, ADDRESS, JAVA_LONG, ADDRESS));
    static final MethodHandle METRICS = downcall("jllama_metrics", json(ADDRESS, ADDRESS));
    static final MethodHandle SLOT_ACTION = downcall("jllama_slot_action", json(ADDRESS, JAVA_INT, JAVA_INT, ADDRESS, ADDRESS));
    static final MethodHandle LORA_ADAPTERS = downcall("jllama_lora_adapters", json(ADDRESS, ADDRESS));
    static final MethodHandle SET_LORA_ADAPTERS = downcall("jllama_set_lora_adapters", json(ADDRESS, ADDRESS, ADDRESS));
    static final MethodHandle CONFIGURE = downcall("jllama_configure", json(ADDRESS, ADDRESS));
    static final MethodHandle SYSTEMONE = downcall("jllama_systemone", json(ADDRESS, ADDRESS, ADDRESS));

    // -- the embedded server (absent from the Android library, which has no FFM API anyway) --------
    static final MethodHandle SERVER_START = downcall("jllama_server_start", json(JAVA_INT, ADDRESS, ADDRESS));
    static final MethodHandle SERVER_START_ATTACHED =
            downcall("jllama_server_start_attached", json(ADDRESS, JAVA_INT, ADDRESS, ADDRESS));
    static final MethodHandle SERVER_STOP = downcall("jllama_server_stop", json(ADDRESS));
    static final MethodHandle SERVER_IS_RUNNING =
            downcall("jllama_server_is_running", FunctionDescriptor.of(JAVA_BOOLEAN, ADDRESS));
    static final MethodHandle SERVER_SET_WORKER_COMMAND = downcall("jllama_server_set_worker_command", json(ADDRESS));

    // -- the RPC server -------------------------------------------------------------------------------
    static final MethodHandle RPC_SERVER_SERVE =
            downcall("jllama_rpc_server_serve", json(ADDRESS, JAVA_INT, JAVA_INT, ADDRESS, ADDRESS));
    static final MethodHandle RPC_SERVER_STOP = downcall("jllama_rpc_server_stop", FunctionDescriptor.of(JAVA_INT));
    static final MethodHandle RPC_SERVER_LISTENING =
            downcall("jllama_rpc_server_listening", FunctionDescriptor.of(JAVA_BOOLEAN));
    static final MethodHandle RPC_SERVER_DEVICES = downcall("jllama_rpc_server_devices", json(ADDRESS, ADDRESS, ADDRESS));

    // -- text to speech -------------------------------------------------------------------------------
    static final MethodHandle TTS_LOAD = downcall("jllama_tts_load", json(ADDRESS, ADDRESS, JAVA_INT, JAVA_INT, ADDRESS));
    static final MethodHandle TTS_SYNTHESIZE = downcall(
            "jllama_tts_synthesize",
            json(ADDRESS, ADDRESS, ADDRESS, ADDRESS, JAVA_INT, JAVA_INT, JAVA_INT, ADDRESS, ADDRESS));
    static final MethodHandle TTS_FREE = downcall("jllama_tts_free", FunctionDescriptor.ofVoid(ADDRESS));

    private JllamaNative() {}

    /** A function returning {@code jllama_status} (a C enum, an {@code int}). */
    private static FunctionDescriptor json(java.lang.foreign.MemoryLayout... arguments) {
        return FunctionDescriptor.of(JAVA_INT, arguments);
    }

    private static MethodHandle downcall(String name, FunctionDescriptor descriptor) {
        MemorySegment symbol = LOOKUP.find(name)
                .orElseThrow(() -> new UnsatisfiedLinkError("the loaded libjllama exports no " + name
                        + ": this module needs the natives of the same version (" + Jllama.class.getPackage() + ")"));
        return LINKER.downcallHandle(symbol, descriptor);
    }

    // -- helpers ----------------------------------------------------------------------------------------

    /** Invokes a handle whose checked exception type is {@code Throwable} only on paper. */
    static Object invoke(MethodHandle handle, Object... arguments) {
        try {
            return handle.invokeWithArguments(arguments);
        } catch (RuntimeException | Error e) {
            throw e;
        } catch (Throwable t) {
            throw new LlamaException("native call failed: " + t, t);
        }
    }

    /** Invokes a {@code jllama_status} function and maps its status to the core's exceptions. */
    static void call(MethodHandle handle, Object... arguments) {
        check((int) invoke(handle, arguments));
    }

    /**
     * Turns a status into the exception the core throws for the same outcome: 400-class failures are
     * {@link InvalidRequestException}, a null argument {@link IllegalArgumentException}, everything else
     * {@link LlamaException}, each with {@code jllama_last_error()} as the message.
     */
    static void check(int status) {
        if (status == STATUS_OK) {
            return;
        }
        String message = lastError();
        switch (status) {
            case STATUS_INVALID_REQUEST -> throw new InvalidRequestException(message);
            case STATUS_INVALID_ARGUMENT -> throw new IllegalArgumentException(message);
            default -> throw new LlamaException(message);
        }
    }

    /** {@code jllama_last_error()} of this thread. */
    static String lastError() {
        return readString((MemorySegment) invoke(LAST_ERROR));
    }

    /** A NUL-terminated UTF-8 string the library owns (a static or thread-local one); {@code null} for NULL. */
    static String readString(MemorySegment pointer) {
        if (pointer == null || pointer.equals(MemorySegment.NULL)) {
            return null;
        }
        return pointer.reinterpret(Long.MAX_VALUE).getString(0, StandardCharsets.UTF_8);
    }

    /** A {@code char **} out-parameter: the string the library wrote, released with {@code jllama_string_free}. */
    static String takeString(MemorySegment out) {
        MemorySegment pointer = out.get(ADDRESS, 0);
        if (pointer.equals(MemorySegment.NULL)) {
            return null;
        }
        try {
            return readString(pointer);
        } finally {
            invoke(STRING_FREE, pointer);
        }
    }

    /** A {@code char **} plus {@code size_t *} out-pair of raw bytes (text with an explicit length). */
    static String takeBytesAsString(MemorySegment out, MemorySegment length) {
        MemorySegment pointer = out.get(ADDRESS, 0);
        if (pointer.equals(MemorySegment.NULL)) {
            return null;
        }
        try {
            long count = length.get(JAVA_LONG, 0);
            return new String(pointer.reinterpret(count).toArray(JAVA_BYTE), StandardCharsets.UTF_8);
        } finally {
            invoke(STRING_FREE, pointer);
        }
    }

    /** A {@code float **} plus {@code size_t *} out-pair, released with {@code jllama_floats_free}. */
    static float[] takeFloats(MemorySegment out, MemorySegment length) {
        MemorySegment pointer = out.get(ADDRESS, 0);
        if (pointer.equals(MemorySegment.NULL)) {
            return new float[0];
        }
        try {
            long count = length.get(JAVA_LONG, 0);
            return pointer.reinterpret(count * JAVA_FLOAT.byteSize()).toArray(JAVA_FLOAT);
        } finally {
            invoke(FLOATS_FREE, pointer);
        }
    }

    /** An {@code int **} plus {@code size_t *} out-pair, released with {@code jllama_tokens_free}. */
    static int[] takeTokens(MemorySegment out, MemorySegment length) {
        MemorySegment pointer = out.get(ADDRESS, 0);
        if (pointer.equals(MemorySegment.NULL)) {
            return new int[0];
        }
        try {
            long count = length.get(JAVA_LONG, 0);
            return pointer.reinterpret(count * JAVA_INT.byteSize()).toArray(JAVA_INT);
        } finally {
            invoke(TOKENS_FREE, pointer);
        }
    }

    /** An {@code unsigned char **} plus {@code size_t *} out-pair, released with {@code jllama_bytes_free}. */
    static byte[] takeBytes(MemorySegment out, MemorySegment length) {
        MemorySegment pointer = out.get(ADDRESS, 0);
        if (pointer.equals(MemorySegment.NULL)) {
            return new byte[0];
        }
        try {
            long count = length.get(JAVA_LONG, 0);
            return pointer.reinterpret(count).toArray(JAVA_BYTE);
        } finally {
            invoke(BYTES_FREE, pointer);
        }
    }

    /** A {@code char ***} plus {@code size_t *} out-pair, released with {@code jllama_strings_free}. */
    static List<String> takeStrings(MemorySegment out, MemorySegment length) {
        MemorySegment pointer = out.get(ADDRESS, 0);
        long count = length.get(JAVA_LONG, 0);
        if (pointer.equals(MemorySegment.NULL) || count == 0) {
            return List.of();
        }
        try {
            MemorySegment array = pointer.reinterpret(count * ADDRESS.byteSize());
            String[] strings = new String[(int) count];
            for (int i = 0; i < strings.length; i++) {
                strings[i] = readString(array.getAtIndex(ADDRESS, i));
            }
            return List.of(strings);
        } finally {
            invoke(STRINGS_FREE, pointer, count);
        }
    }

    /** A C string in {@code arena}, or NULL for a {@code null} value. */
    static MemorySegment cString(Arena arena, String value) {
        return value == null ? MemorySegment.NULL : arena.allocateFrom(value, StandardCharsets.UTF_8);
    }

    /** A {@code const char *const *} of {@code strings} in {@code arena} (a null element is ""). */
    static MemorySegment cStrings(Arena arena, String[] strings) {
        MemorySegment array = arena.allocate(ADDRESS, Math.max(strings.length, 1));
        for (int i = 0; i < strings.length; i++) {
            array.setAtIndex(ADDRESS, i, cString(arena, strings[i] == null ? "" : strings[i]));
        }
        return array;
    }

    /** An out-slot for a pointer. */
    static MemorySegment pointerSlot(Arena arena) {
        return arena.allocate(ADDRESS);
    }

    /** An out-slot for a {@code size_t} or {@code int}. */
    static MemorySegment longSlot(Arena arena) {
        return arena.allocate(JAVA_LONG);
    }
}
