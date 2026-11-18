// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT
package net.ladenthin.llama.ffm;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_BOOLEAN;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static net.ladenthin.llama.ffm.JllamaNative.cString;
import static net.ladenthin.llama.ffm.JllamaNative.cStrings;
import static net.ladenthin.llama.ffm.JllamaNative.invoke;
import static net.ladenthin.llama.ffm.JllamaNative.longSlot;
import static net.ladenthin.llama.ffm.JllamaNative.pointerSlot;
import static net.ladenthin.llama.ffm.JllamaNative.takeBytesAsString;
import static net.ladenthin.llama.ffm.JllamaNative.takeFloats;
import static net.ladenthin.llama.ffm.JllamaNative.takeString;
import static net.ladenthin.llama.ffm.JllamaNative.takeTokens;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.locks.ReentrantReadWriteLock;
import net.ladenthin.llama.callback.LoadProgressCallback;
import net.ladenthin.llama.callback.VideoDecoder;
import net.ladenthin.llama.exception.InvalidRequestException;
import net.ladenthin.llama.exception.LlamaException;
import net.ladenthin.llama.parameters.ModelParameters;

/**
 * A model loaded through the C API: the operations of {@code LlamaModel}, one C function each, with
 * JSON text in and out. A handle may be used from several threads; {@link #close()} interrupts the calls
 * in flight (a blocking completion throws {@link LlamaException} within about a second) and waits for
 * them, and a call that starts after it throws {@link IllegalStateException} -- a call never reaches a
 * freed handle, as with {@code LlamaModel}.
 *
 * <p>Requests are the bodies llama.cpp's own HTTP server reads ({@code InferenceParameters.toJson()},
 * or an OpenAI chat body written by hand) and responses the objects it writes. A request llama.cpp
 * rejects throws {@link InvalidRequestException}, any other failure {@link LlamaException}.
 */
public final class FfmLlamaModel implements AutoCloseable {

    private final MemorySegment handle;
    private final AtomicBoolean closed = new AtomicBoolean();
    /**
     * Every native call on the handle holds the read lock, {@link #close()} the write lock: a close
     * waits for the calls in flight, and a call checks {@link #closed} under the lock, so it never
     * reaches the handle after {@code jllama_model_free}. Without it a call that passed the check
     * while another thread closed the model used the freed handle.
     */
    private final ReentrantReadWriteLock lifecycle = new ReentrantReadWriteLock();
    /** The video decoder's stubs, freed after the model; null for a model loaded without one. */
    private final FfmVideoDecoder video;

    private FfmLlamaModel(MemorySegment handle, FfmVideoDecoder video) {
        this.handle = handle;
        this.video = video;
    }

    /**
     * Loads a model from llama-server's command line, as {@code new LlamaModel(parameters)} does.
     *
     * @param parameters the model parameters ({@code -m}, {@code -c}, {@code -ngl}, ...)
     * @return the loaded model
     * @throws LlamaException when the model does not load
     */
    public static FfmLlamaModel load(ModelParameters parameters) {
        return load(parameters, null);
    }

    /**
     * Loads a model, reporting progress; the callback may abort the load by returning {@code false}.
     *
     * @param parameters the model parameters
     * @param progress the progress callback, or {@code null}
     * @return the loaded model
     * @throws LlamaException when the model does not load, or the callback aborted it
     */
    public static FfmLlamaModel load(ModelParameters parameters, LoadProgressCallback progress) {
        Objects.requireNonNull(parameters, "parameters");
        return load(parameters.toArray(), progress);
    }

    /**
     * Loads a model from a raw argv: {@code argv[0]} is a program name (any), the rest the flags
     * llama-server accepts.
     *
     * @param argv the argv, program name first
     * @param progress the progress callback, or {@code null}
     * @return the loaded model
     * @throws LlamaException when the model does not load, or the callback aborted it
     */
    public static FfmLlamaModel load(String[] argv, LoadProgressCallback progress) {
        Objects.requireNonNull(argv, "argv");
        try (Arena arena = Arena.ofConfined()) {
            // The callback runs on the loading thread, i.e. this one: a confined arena suffices.
            MemorySegment stub = progressStub(arena, progress);
            MemorySegment out = pointerSlot(arena);
            JllamaNative.call(JllamaNative.MODEL_LOAD, argv.length, cStrings(arena, argv), stub, MemorySegment.NULL,
                    out);
            return new FfmLlamaModel(out.get(ADDRESS, 0), null);
        }
    }

    /**
     * Loads a model whose requests read their videos through {@code decoder}, as
     * {@code new LlamaModel(parameters, progress, decoder)} does: a video in a request is handed to the
     * decoder instead of the {@code ffmpeg} program. The decoder is called on a thread of the library,
     * not on the caller's.
     *
     * @param parameters the model parameters (a vision {@code --mmproj} for the video to be seen)
     * @param progress the progress callback, or {@code null}
     * @param decoder the video decoder
     * @return the loaded model
     * @throws LlamaException when the model does not load, or the callback aborted it
     */
    public static FfmLlamaModel load(ModelParameters parameters, LoadProgressCallback progress, VideoDecoder decoder) {
        Objects.requireNonNull(parameters, "parameters");
        return load(parameters.toArray(), progress, decoder);
    }

    /**
     * {@link #load(ModelParameters, LoadProgressCallback, VideoDecoder)} from a raw argv.
     *
     * @param argv the argv, program name first
     * @param progress the progress callback, or {@code null}
     * @param decoder the video decoder
     * @return the loaded model
     * @throws LlamaException when the model does not load, or the callback aborted it
     */
    public static FfmLlamaModel load(String[] argv, LoadProgressCallback progress, VideoDecoder decoder) {
        Objects.requireNonNull(argv, "argv");
        FfmVideoDecoder video = new FfmVideoDecoder(Objects.requireNonNull(decoder, "decoder"));
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment stub = progressStub(arena, progress);
            MemorySegment out = pointerSlot(arena);
            JllamaNative.call(JllamaNative.MODEL_LOAD_VIDEO, argv.length, cStrings(arena, argv), stub,
                    MemorySegment.NULL, video.struct(), out);
            return new FfmLlamaModel(out.get(ADDRESS, 0), video);
        } catch (RuntimeException | Error e) {
            // a failed load has released the decoder already (jllama_model_load_video calls release)
            video.close();
            throw e;
        }
    }

    /** The upcall stub of a progress callback in {@code arena}, or the null pointer for none. */
    static MemorySegment progressStub(Arena arena, LoadProgressCallback progress) {
        if (progress == null) {
            return MemorySegment.NULL;
        }
        try {
            MethodHandle target = MethodHandles.lookup()
                    .findStatic(FfmLlamaModel.class, "onProgress", MethodType.methodType(boolean.class,
                            LoadProgressCallback.class, float.class, MemorySegment.class))
                    .bindTo(progress);
            return JllamaNative.LINKER.upcallStub(target, JllamaNative.LOAD_PROGRESS_FN, arena);
        } catch (NoSuchMethodException | IllegalAccessException e) {
            throw new IllegalStateException(e);
        }
    }

    @SuppressWarnings("unused") // the upcall target of progressStub
    private static boolean onProgress(LoadProgressCallback callback, float progress, MemorySegment userData) {
        try {
            return callback.onProgress(progress);
        } catch (Throwable e) {
            // Nothing may cross the upcall boundary -- a RuntimeException or an Error alike would end the
            // JVM ("Unrecoverable uncaught exception") -- so the load is aborted instead.
            return false;
        }
    }

    /**
     * Run by {@link #handle()} after its check, before the native call: lets a test hold a call in
     * exactly the window a racing {@link #close()} used to free the handle in. Null outside tests.
     */
    static volatile Runnable afterCheckForTesting;

    /** The raw handle for the attach mode of {@link FfmNativeServer}, and for every call here. */
    MemorySegment handle() {
        if (closed.get()) {
            throw new IllegalStateException("the model is closed");
        }
        Runnable hook = afterCheckForTesting;
        if (hook != null) {
            hook.run();
        }
        return handle;
    }

    /**
     * Frees the model: cancels its streaming tasks, waits for the calls in flight, stops its inference
     * loop and releases the weights. A second call is a no-op.
     */
    @Override
    public void close() {
        if (!closed.compareAndSet(false, true)) {
            return;
        }
        // A new call is refused from here on (handle() checks the flag under the read lock). The calls
        // in flight hold the read lock for their whole native duration, so they are interrupted first
        // -- a blocking completion returns within about a second instead of finishing its generation --
        // and the write lock then waits only for them to unwind. Only this close frees the handle, so
        // the interrupt needs no lock of its own.
        invoke(JllamaNative.MODEL_INTERRUPT, handle);
        lifecycle.writeLock().lock();
        try {
            invoke(JllamaNative.MODEL_FREE, handle);
            if (video != null) {
                video.close();
            }
        } finally {
            lifecycle.writeLock().unlock();
        }
    }

    /**
     * Runs {@code body} on the live handle under the read lock, for a native call made outside this
     * class ({@link FfmNativeServer#startAttached}): a concurrent {@link #close()} waits for it, and
     * after the close it throws {@link IllegalStateException} instead of reaching the freed handle.
     */
    <T> T withHandle(java.util.function.Function<MemorySegment, T> body) {
        try (InUse _ = new InUse()) {
            return body.apply(handle());
        }
    }

    /** The read lock of one native call on the handle, released by try-with-resources. */
    private final class InUse implements AutoCloseable {
        InUse() {
            lifecycle.readLock().lock();
        }

        @Override
        public void close() {
            lifecycle.readLock().unlock();
        }
    }

    /**
     * The model's metadata object (vocab, context, modalities, chat template, the GGUF key/value map).
     *
     * @return the JSON object
     */
    public String meta() {
        return stringOut(JllamaNative.MODEL_META);
    }

    /**
     * One OpenAI chat completion, blocking.
     *
     * @param requestJson the OpenAI chat body ({@code {"messages": [...], ...}})
     * @return the {@code chat.completion} object
     * @throws InvalidRequestException if the server refuses the body, including {@code "n"} greater than 1
     *     (one completion per request; send separate requests)
     */
    public String chatCompletions(String requestJson) {
        return stringOut(JllamaNative.CHAT_COMPLETIONS, requireNonNull(requestJson, "requestJson"));
    }

    /**
     * A streamed OpenAI chat completion: each {@link FfmTaskStream#next()} is one
     * {@code {"data": <chunk>, "stop": <bool>}} envelope, as {@code LlamaModel.receiveChatCompletionChunk}.
     *
     * @param requestJson the OpenAI chat body
     * @return the stream; close it to release the task early
     * @throws InvalidRequestException if the server refuses the body, including {@code "n"} greater than 1
     *     (one completion per request; send separate requests)
     */
    public FfmTaskStream chatCompletionsStream(String requestJson) {
        int task = taskOut(JllamaNative.CHAT_COMPLETIONS_STREAM_REQUEST, requireNonNull(requestJson, "requestJson"));
        return new FfmTaskStream(this, task, JllamaNative.CHAT_COMPLETIONS_STREAM_RECEIVE);
    }

    /**
     * A streamed chat completion whose results are llama.cpp's raw result objects (as
     * {@code LlamaModel.requestChatCompletion}).
     *
     * @param requestJson the OpenAI chat body
     * @return the stream of raw result objects
     * @throws InvalidRequestException if the server refuses the body, including {@code "n"} greater than 1
     *     (one completion per request; send separate requests)
     */
    public FfmTaskStream chatCompletionsRaw(String requestJson) {
        int task = taskOut(JllamaNative.CHAT_COMPLETIONS_REQUEST, requireNonNull(requestJson, "requestJson"));
        return new FfmTaskStream(this, task, JllamaNative.COMPLETIONS_STREAM_RECEIVE);
    }

    /**
     * A raw {@code /completion} request, blocking.
     *
     * @param requestJson the request object ({@code {"prompt": ..., ...}})
     * @return the result object
     * @throws InvalidRequestException if the server refuses the body, including several prompts in one
     *     request or {@code "n"} greater than 1 (one prompt and one completion per request; send separate
     *     requests)
     */
    public String completions(String requestJson) {
        return stringOut(JllamaNative.COMPLETIONS, requireNonNull(requestJson, "requestJson"));
    }

    /**
     * An OpenAI {@code /v1/completions} request, blocking.
     *
     * @param requestJson the request object
     * @return the {@code text_completion} object
     * @throws InvalidRequestException if the server refuses the body, including several prompts in one
     *     request or {@code "n"} greater than 1 (one prompt and one completion per request; send separate
     *     requests)
     */
    public String completionsOai(String requestJson) {
        return stringOut(JllamaNative.COMPLETIONS_OAI, requireNonNull(requestJson, "requestJson"));
    }

    /**
     * A streamed OpenAI {@code /v1/completions} request: each {@link FfmTaskStream#next()} is one
     * {@code {"data": <text_completion chunk>, "stop": <bool>}} envelope, the last chunk with
     * {@code usage} (as {@code LlamaModel.streamCompletionsOai}).
     *
     * @param requestJson the request object, parsed as {@link #completionsOai(String)} parses it
     * @return the stream; close it to release the task early
     * @throws InvalidRequestException if the server refuses the body, including several prompts in one
     *     request or {@code "n"} greater than 1 (one prompt and one completion per request; send separate
     *     requests)
     */
    public FfmTaskStream completionsOaiStream(String requestJson) {
        int task = taskOut(JllamaNative.COMPLETIONS_OAI_STREAM_REQUEST, requireNonNull(requestJson, "requestJson"));
        return new FfmTaskStream(this, task, JllamaNative.CHAT_COMPLETIONS_STREAM_RECEIVE);
    }

    /**
     * A fill-in-the-middle request, blocking (the model needs FIM tokens).
     *
     * @param requestJson the request object ({@code input_prefix}, {@code input_suffix}, ...)
     * @return the result object
     * @throws InvalidRequestException if the server refuses the body, including {@code "n"} greater than 1
     *     (one completion per request)
     */
    public String infill(String requestJson) {
        return stringOut(JllamaNative.INFILL, requireNonNull(requestJson, "requestJson"));
    }

    /**
     * A streamed raw completion: each {@link FfmTaskStream#next()} is one of llama.cpp's result objects.
     *
     * @param requestJson the request object
     * @return the stream
     * @throws InvalidRequestException if the server refuses the body, including several prompts in one
     *     request or {@code "n"} greater than 1 (one prompt and one completion per request; send separate
     *     requests)
     */
    public FfmTaskStream completionsStream(String requestJson) {
        int task = taskOut(JllamaNative.COMPLETIONS_STREAM_REQUEST, requireNonNull(requestJson, "requestJson"));
        return new FfmTaskStream(this, task, JllamaNative.COMPLETIONS_STREAM_RECEIVE);
    }

    /**
     * The chat template applied to an OpenAI chat body.
     *
     * @param requestJson {@code {"messages": [...]}}
     * @return the prompt text
     */
    public String applyTemplate(String requestJson) {
        return stringOut(JllamaNative.APPLY_TEMPLATE, requireNonNull(requestJson, "requestJson"));
    }

    /**
     * The embedding of one prompt (a model loaded with {@code --embedding}).
     *
     * @param prompt the text
     * @return the vector
     */
    public float[] embed(String prompt) {
        requireNonNull(prompt, "prompt");
        try (InUse _ = new InUse();
                Arena arena = Arena.ofConfined()) {
            MemorySegment values = pointerSlot(arena);
            MemorySegment count = longSlot(arena);
            JllamaNative.call(JllamaNative.EMBED, handle(), cString(arena, prompt), values, count);
            return takeFloats(values, count);
        }
    }

    /**
     * An embeddings request, blocking.
     *
     * @param requestJson {@code {"input": ..., ...}}
     * @param oaiCompat whether to answer in the OpenAI shape ({@code /v1/embeddings}) or llama.cpp's own
     * @return the response object
     */
    public String embeddings(String requestJson, boolean oaiCompat) {
        return stringOut(JllamaNative.EMBEDDINGS, requireNonNull(requestJson, "requestJson"), oaiCompat);
    }

    /**
     * Reranks documents against a query (a model loaded with {@code --reranking}).
     *
     * @param query the query
     * @param documents the documents
     * @return the results array, as {@code LlamaModel.rerank}
     */
    public String rerank(String query, String... documents) {
        requireNonNull(query, "query");
        requireNonNull(documents, "documents");
        try (InUse _ = new InUse();
                Arena arena = Arena.ofConfined()) {
            MemorySegment out = pointerSlot(arena);
            JllamaNative.call(JllamaNative.RERANK, handle(), cString(arena, query), documents.length,
                    cStrings(arena, documents), out);
            return takeString(out);
        }
    }

    /**
     * Tokenises text.
     *
     * @param text the text
     * @param addSpecial whether to add the special tokens (BOS)
     * @return the token ids
     */
    public int[] tokenize(String text, boolean addSpecial) {
        requireNonNull(text, "text");
        try (InUse _ = new InUse();
                Arena arena = Arena.ofConfined()) {
            MemorySegment tokens = pointerSlot(arena);
            MemorySegment count = longSlot(arena);
            JllamaNative.call(JllamaNative.TOKENIZE, handle(), cString(arena, text), addSpecial, tokens, count);
            return takeTokens(tokens, count);
        }
    }

    /**
     * Tokenises text into the {@code /tokenize} response object.
     *
     * @param text the text
     * @param addSpecial whether to add the special tokens
     * @param withPieces whether to include each token's text
     * @return the response object
     */
    public String tokenizeJson(String text, boolean addSpecial, boolean withPieces) {
        return stringOut(JllamaNative.TOKENIZE_JSON, requireNonNull(text, "text"), addSpecial, withPieces);
    }

    /**
     * Turns tokens back into text.
     *
     * @param tokens the token ids
     * @return the text
     */
    public String detokenize(int[] tokens) {
        requireNonNull(tokens, "tokens");
        try (InUse _ = new InUse();
                Arena arena = Arena.ofConfined()) {
            MemorySegment text = pointerSlot(arena);
            MemorySegment length = longSlot(arena);
            JllamaNative.call(JllamaNative.DETOKENIZE, handle(), arena.allocateFrom(JAVA_INT, tokens),
                    (long) tokens.length, text, length);
            return takeBytesAsString(text, length);
        }
    }

    /**
     * Turns tokens into the {@code /detokenize} response object.
     *
     * @param tokens the token ids
     * @return the response object
     */
    public String detokenizeJson(int[] tokens) {
        requireNonNull(tokens, "tokens");
        try (InUse _ = new InUse();
                Arena arena = Arena.ofConfined()) {
            MemorySegment out = pointerSlot(arena);
            JllamaNative.call(JllamaNative.DETOKENIZE_JSON, handle(), arena.allocateFrom(JAVA_INT, tokens),
                    (long) tokens.length, out);
            return takeString(out);
        }
    }

    /**
     * The server's metrics and slots, the object {@code LlamaModel.getMetrics} returns.
     *
     * @return the JSON object
     */
    public String metrics() {
        return stringOut(JllamaNative.METRICS);
    }

    /**
     * A slot save, restore or erase. As {@code LlamaModel.saveSlot}, the file of a save or restore is a
     * file name inside the directory the model was loaded with ({@code --slot-save-path}), not a path.
     *
     * @param op the operation
     * @param slotId the slot
     * @param filename the name of the file inside the slot directory for {@link SlotOp#SAVE} and
     *     {@link SlotOp#RESTORE}; ignored for erase
     * @return the server's result object
     * @throws InvalidRequestException if {@code filename} has a directory part, contains {@code ..} or
     *     is empty
     * @throws LlamaException if the model was loaded without {@code --slot-save-path}, or the action fails
     */
    public String slotAction(SlotOp op, int slotId, String filename) {
        requireNonNull(op, "op");
        return stringOutRaw(JllamaNative.SLOT_ACTION, op.code(), slotId, filename == null ? "" : filename);
    }

    /**
     * The LoRA adapters of the model.
     *
     * @return the {@code GET /lora-adapters} array
     */
    public String loraAdapters() {
        return stringOut(JllamaNative.LORA_ADAPTERS);
    }

    /**
     * Sets the LoRA adapter scales.
     *
     * @param adaptersJson a JSON array of {@code {"id": n, "scale": f}}
     * @return the resulting adapter list
     */
    public String setLoraAdapters(String adaptersJson) {
        return stringOut(JllamaNative.SET_LORA_ADAPTERS, requireNonNull(adaptersJson, "adaptersJson"));
    }

    /**
     * Runtime tuning from a JSON object with any of {@code n_threads}, {@code n_threads_batch} and
     * {@code slot_prompt_similarity}.
     *
     * @param configJson the object
     */
    public void configure(String configJson) {
        requireNonNull(configJson, "configJson");
        try (InUse _ = new InUse();
                Arena arena = Arena.ofConfined()) {
            JllamaNative.call(JllamaNative.CONFIGURE, handle(), cString(arena, configJson));
        }
    }

    /**
     * A {@code /v1/systemone} decision request (a decision model).
     *
     * @param requestJson the request object
     * @return the response object
     */
    public String systemOne(String requestJson) {
        return stringOut(JllamaNative.SYSTEMONE, requireNonNull(requestJson, "requestJson"));
    }

    // -- the task stream's half --------------------------------------------------------------------------

    /** One envelope of a streaming task: the JSON text, with {@code stop} set when it was the last. */
    String receive(MethodHandle receive, int taskId, boolean[] stop) {
        try (InUse _ = new InUse();
                Arena arena = Arena.ofConfined()) {
            MemorySegment out = pointerSlot(arena);
            MemorySegment stopSlot = arena.allocate(JAVA_BOOLEAN);
            JllamaNative.call(receive, handle(), taskId, out, stopSlot);
            stop[0] = stopSlot.get(JAVA_BOOLEAN, 0);
            return takeString(out);
        }
    }

    /** Cancels or releases a streaming task; an unknown id is ignored. */
    void releaseTask(int taskId) {
        try (InUse _ = new InUse()) {
            if (!closed.get()) {
                JllamaNative.call(JllamaNative.TASK_RELEASE, handle, taskId);
            }
        }
    }

    // -- helpers ----------------------------------------------------------------------------------------------

    private static <T> T requireNonNull(T value, String name) {
        return Objects.requireNonNull(value, name);
    }

    /** A call {@code (model, <string args...>, char **out)} whose strings are marshalled in a confined arena. */
    private String stringOut(MethodHandle handle, Object... arguments) {
        try (InUse _ = new InUse();
                Arena arena = Arena.ofConfined()) {
            Object[] all = new Object[arguments.length + 2];
            all[0] = handle();
            for (int i = 0; i < arguments.length; i++) {
                all[i + 1] = arguments[i] instanceof String s ? cString(arena, s) : arguments[i];
            }
            MemorySegment out = pointerSlot(arena);
            all[all.length - 1] = out;
            JllamaNative.call(handle, all);
            return takeString(out);
        }
    }

    /** As {@link #stringOut} for the slot action, whose arguments are two ints and a string. */
    private String stringOutRaw(MethodHandle handle, int a, int b, String s) {
        try (InUse _ = new InUse();
                Arena arena = Arena.ofConfined()) {
            MemorySegment out = pointerSlot(arena);
            JllamaNative.call(handle, handle(), a, b, cString(arena, s), out);
            return takeString(out);
        }
    }

    /** A call {@code (model, const char *request, int *task_id)}: the task id. */
    private int taskOut(MethodHandle handle, String requestJson) {
        try (InUse _ = new InUse();
                Arena arena = Arena.ofConfined()) {
            MemorySegment task = arena.allocate(JAVA_INT);
            JllamaNative.call(handle, handle(), cString(arena, requestJson), task);
            return task.get(JAVA_INT, 0);
        }
    }
}
