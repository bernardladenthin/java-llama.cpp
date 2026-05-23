// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
// SPDX-FileCopyrightText: 2023-2025 Konstantin Herud
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import net.ladenthin.llama.args.LogFormat;
import net.ladenthin.llama.json.ChatResponseParser;
import net.ladenthin.llama.json.CompletionResponseParser;
import net.ladenthin.llama.json.RerankResponseParser;
import java.lang.annotation.Native;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.function.BiConsumer;

/**
 * This class is a wrapper around the llama.cpp functionality.
 * Upon being created, it natively allocates memory for the model context.
 * Thus, this class is an {@link AutoCloseable}, in order to de-allocate the memory when it is no longer being needed.
 * <p>
 * The main functionality of this class is:
 * <ul>
 *     <li>Streaming answers (and probabilities) via {@link #generate(InferenceParameters)}</li>
 *     <li>Creating whole responses to prompts via {@link #complete(InferenceParameters)}</li>
 *     <li>Creating embeddings via {@link #embed(String)} (make sure to configure {@link ModelParameters#enableEmbedding()}</li>
 *     <li>Accessing the tokenizer via {@link #encode(String)} and {@link #decode(int[])}</li>
 * </ul>
 */
public class LlamaModel implements AutoCloseable {

	static {
		LlamaLoader.initialize();
	}

	@Native
	private long ctx;

	private final CompletionResponseParser completionParser = new CompletionResponseParser();
	private final ChatResponseParser chatParser = new ChatResponseParser();
	private final RerankResponseParser rerankParser = new RerankResponseParser();

	/**
	 * Load with the given {@link ModelParameters}. Make sure to either set
	 * <ul>
	 *     <li>{@link ModelParameters#setModel(String)}</li>
	 *     <li>{@link ModelParameters#setModelUrl(String)}</li>
	 *     <li>{@link ModelParameters#setHfRepo(String)}, {@link ModelParameters#setHfFile(String)}</li>
	 * </ul>
	 *
	 * @param parameters the set of options
	 * @throws LlamaException if no model could be loaded from the given file path
	 */
	public LlamaModel(ModelParameters parameters) {
		loadModel(parameters.toArray());
	}

	/**
	 * Generate and return a whole answer with custom parameters. Note, that the prompt isn't preprocessed in any
	 * way, nothing like "User: ", "###Instruction", etc. is added.
	 *
	 * @param parameters the inference configuration
	 * @return an LLM response
	 */
	public String complete(InferenceParameters parameters) {
		parameters.setStream(false);
		int taskId = requestCompletion(parameters.toString());
		String json = receiveCompletionJson(taskId);
		return completionParser.parse(json).text;
	}

	/**
	 * Cancellable variant of {@link #complete(InferenceParameters)}. Runs in streaming mode
	 * internally so the inference loop can observe a {@link CancellationToken#cancel()} call
	 * from another thread and return early with whatever text was accumulated so far.
	 * <p>
	 * The token is rebound to this call (any prior {@code cancel} state is cleared on entry).
	 * On return &mdash; whether by natural stop or cancellation &mdash; the token is unbound.
	 * </p>
	 *
	 * @param parameters the inference configuration (its {@code stream} flag will be set to true)
	 * @param token cancellation handle; {@link CancellationToken#cancel()} aborts the loop
	 * @return the text generated up to the point of stop or cancellation
	 */
	/**
	 * Asynchronous variant of {@link #complete(InferenceParameters)}. Runs the inference on
	 * the common {@link java.util.concurrent.ForkJoinPool} so it does not block the calling
	 * thread. The native worker thread inside the JNI context still serializes the actual
	 * model work &mdash; this wrapper only moves the blocking Java call off the caller.
	 *
	 * @param parameters the inference configuration
	 * @return a future completed with the generated text
	 */
	public CompletableFuture<String> completeAsync(InferenceParameters parameters) {
		return CompletableFuture.supplyAsync(() -> complete(parameters));
	}

	/**
	 * Cancellable async variant. The returned future is wired to the supplied
	 * {@link CancellationToken}: calling {@code future.cancel(true)} also invokes
	 * {@link CancellationToken#cancel()} so the inference loop returns early.
	 *
	 * @param parameters the inference configuration
	 * @param token cancellation handle bound to the underlying inference loop
	 * @return a future completed with whatever text was generated up to the point of stop or cancellation
	 */
	public CompletableFuture<String> completeAsync(InferenceParameters parameters, CancellationToken token) {
		CompletableFuture<String> future = CompletableFuture.supplyAsync(() -> complete(parameters, token));
		future.whenComplete((result, ex) -> {
			if (ex instanceof java.util.concurrent.CancellationException) {
				token.cancel();
			}
		});
		return future;
	}

	/**
	 * Asynchronous variant of {@link #chatComplete(InferenceParameters)}.
	 *
	 * @param parameters the inference parameters including messages
	 * @return a future completed with the raw OAI-format JSON response
	 */
	public CompletableFuture<String> chatCompleteAsync(InferenceParameters parameters) {
		return CompletableFuture.supplyAsync(() -> chatComplete(parameters));
	}

	/**
	 * Asynchronous variant of {@link #chatCompleteText(InferenceParameters)}.
	 *
	 * @param parameters the inference parameters including messages
	 * @return a future completed with the assistant's reply text
	 */
	public CompletableFuture<String> chatCompleteTextAsync(InferenceParameters parameters) {
		return CompletableFuture.supplyAsync(() -> chatCompleteText(parameters));
	}

	/**
	 * Cancellable variant of {@link #complete(InferenceParameters)}. Runs in streaming mode
	 * internally so the inference loop can observe a {@link CancellationToken#cancel()} call
	 * from another thread between token boundaries and return early with whatever text was
	 * accumulated so far.
	 *
	 * @param parameters the inference configuration (its {@code stream} flag is set to {@code true})
	 * @param token cancellation handle observed at each token boundary
	 * @return the text generated up to the point of stop or cancellation
	 */
	public String complete(InferenceParameters parameters, CancellationToken token) {
		token.reset();
		parameters.setStream(true);
		int taskId = requestCompletion(parameters.toString());
		StringBuilder sb = new StringBuilder();
		try {
			while (true) {
				if (token.isCancelled()) {
					// Best-effort native release. Safe to call here because we are not
					// concurrently inside receiveCompletionJson — the cooperative cancel
					// flag stopped the loop at a token boundary.
					cancelCompletion(taskId);
					break;
				}
				String json = receiveCompletionJson(taskId);
				LlamaOutput out = completionParser.parse(json);
				sb.append(out.text);
				if (out.stop) {
					break;
				}
			}
		} finally {
			token.reset();
		}
		return sb.toString();
	}

	/**
	 * Generate and stream outputs with custom inference parameters. Note, that the prompt isn't preprocessed in any
	 * way, nothing like "User: ", "###Instruction", etc. is added.
	 *
	 * <p>The returned {@link LlamaIterable} implements {@link AutoCloseable}. Wrap it in a
	 * try-with-resources block to guarantee the native task slot is released even when the
	 * consumer exits the loop early:
	 * <pre>{@code
	 * try (LlamaIterable it = model.generate(params)) {
	 *     for (LlamaOutput out : it) {
	 *         if (shouldStop(out)) break;   // close() cancels the native task automatically
	 *     }
	 * }
	 * }</pre>
	 *
	 * @param parameters the inference configuration
	 * @return iterable LLM outputs
	 */
	public LlamaIterable generate(InferenceParameters parameters) {
		return new LlamaIterable(new LlamaIterator(this, parameters));
	}
	
	
    
	/**
	 * Get the embedding of a string. Note, that the prompt isn't preprocessed in any way, nothing like
	 * "User: ", "###Instruction", etc. is added.
	 *
	 * @param prompt the string to embed
	 * @return an embedding float array
	 * @throws IllegalStateException if embedding mode was not activated (see {@link ModelParameters#enableEmbedding()})
	 */
	public  native float[] embed(String prompt);
		

	/**
	 * Tokenize a prompt given the native tokenizer
	 *
	 * @param prompt the prompt to tokenize
	 * @return an array of integers each representing a token id
	 */
	public native int[] encode(String prompt);

	/**
	 * Convert an array of token ids to its string representation
	 *
	 * @param tokens an array of tokens
	 * @return the token ids decoded to a string
	 */
	public String decode(int[] tokens) {
		byte[] bytes = decodeBytes(tokens);
		return new String(bytes, StandardCharsets.UTF_8);
	}

	/**
	 * Sets a callback for native llama.cpp log messages.
	 * Per default, log messages are written in JSON to stdout. Note, that in text mode the callback will be also
	 * invoked with log messages of the GGML backend, while JSON mode can only access request log messages.
	 * In JSON mode, GGML messages will still be written to stdout.
	 * To only change the log format but keep logging to stdout, the given callback can be <code>null</code>.
	 * To disable logging, pass an empty callback, i.e., <code>(level, msg) {@literal ->} {}</code>.
	 *
	 * @param format the log format to use
	 * @param callback a method to call for log messages
	 */
	public static native void setLogger(LogFormat format, BiConsumer<LogLevel, String> callback);

	@Override
	public void close() {
		delete();
	}

	// don't overload native methods since the C++ function names get nasty
	native int requestCompletion(String params) throws LlamaException;

	native String receiveCompletionJson(int taskId) throws LlamaException;

	native void cancelCompletion(int taskId);

	native byte[] decodeBytes(int[] tokens);

	private native void loadModel(String... parameters) throws LlamaException;

	private native void delete();
	
	native void releaseTask(int taskId);

	private static native byte[] jsonSchemaToGrammarBytes(String schema);
	
	/**
	 * Converts a JSON schema to a grammar string usable by {@link ModelParameters#setGrammar(String)}.
	 *
	 * @param schema the JSON schema as a string
	 * @return the converted grammar string
	 */
	public static String jsonSchemaToGrammar(String schema) {
		return new String(jsonSchemaToGrammarBytes(schema), StandardCharsets.UTF_8);
	}
	
	/**
	 * Rerank the given documents against the query.
	 *
	 * @param reRank whether to sort results by score in descending order
	 * @param query the query string
	 * @param documents the documents to rank
	 * @return a list of document/score pairs, sorted if {@code reRank} is {@code true}
	 */
	public List<Pair<String, Float>> rerank(boolean reRank, String query, String... documents) {
		String json = handleRerank(query, documents);
		List<Pair<String, Float>> rankedDocuments = rerankParser.parse(json);
		if (reRank) {
			rankedDocuments.sort((a, b) -> Float.compare(b.getValue(), a.getValue()));
		}
		return rankedDocuments;
	}

	/**
	 * Rerank the given documents against the query, returning a {@link LlamaOutput} with scored documents
	 * in the probabilities map.
	 *
	 * @param query the query string
	 * @param documents the documents to rank
	 * @return a LlamaOutput with document/score pairs in the probabilities map
	 */
	public LlamaOutput rerank(String query, String... documents) {
		String json = handleRerank(query, documents);
		List<Pair<String, Float>> results = rerankParser.parse(json);
		Map<String, Float> probabilities = new HashMap<>();
		for (Pair<String, Float> pair : results) {
			probabilities.put(pair.getKey(), pair.getValue());
		}
		return new LlamaOutput(query, probabilities, true, StopReason.EOS);
	}

	native String handleRerank(String query, String... documents) throws LlamaException;
	
	/**
	 * Applies the chat template to the given inference parameters and returns the formatted string.
	 *
	 * @param parameters the inference parameters containing message configuration
	 * @return the formatted chat template string
	 */
	public String applyTemplate(InferenceParameters parameters) {
		return applyTemplate(parameters.toString());
	}
	/**
	 * @param parametersJson JSON-serialized inference parameters
	 * @return the formatted chat template string
	 */
	public native String applyTemplate(String parametersJson);

	/**
	 * Run an OpenAI-compatible chat completion. The parameters must contain a "messages" array
	 * in the standard OpenAI chat format (objects with "role" and "content" fields). The model's
	 * chat template is automatically applied.
	 * <p>
	 * Example usage:
	 * <pre>{@code
	 * List<Pair<String, String>> messages = new ArrayList<>();
	 * messages.add(new Pair<>("user", "What is the capital of France?"));
	 *
	 * InferenceParameters params = new InferenceParameters("")
	 *     .setMessages("You are a helpful assistant.", messages)
	 *     .setNPredict(128)
	 *     .setTemperature(0.7f);
	 *
	 * String response = model.chatComplete(params);
	 * }</pre>
	 *
	 * @param parameters the inference parameters including messages
	 * @return the model's response as a JSON string containing the completion result
	 * @throws LlamaException if the model was loaded in embedding mode or if inference fails
	 */
	public String chatComplete(InferenceParameters parameters) {
		parameters.setStream(false);
		return handleChatCompletions(parameters.toString());
	}

	/**
	 * Run an OpenAI-compatible chat completion and return only the assistant's text content.
	 * This is the plain-string equivalent of {@link #chatComplete(InferenceParameters)}, which
	 * returns the raw OAI JSON. Use this when you want the generated text directly, the same
	 * way {@link #complete(InferenceParameters)} works for raw completions.
	 *
	 * @param parameters the inference parameters including messages
	 * @return the assistant's reply text (extracted from {@code choices[0].message.content})
	 * @throws LlamaException if the model was loaded in embedding mode or if inference fails
	 */
	public String chatCompleteText(InferenceParameters parameters) {
		return chatParser.extractChoiceContent(chatComplete(parameters));
	}

	/**
	 * Stream an OpenAI-compatible chat completion token by token. The parameters must contain a
	 * "messages" array in the standard OpenAI chat format. The model's chat template is automatically applied.
	 * <p>
	 * Example usage:
	 * <pre>{@code
	 * List<Pair<String, String>> messages = new ArrayList<>();
	 * messages.add(new Pair<>("user", "Tell me a story."));
	 *
	 * InferenceParameters params = new InferenceParameters("")
	 *     .setMessages("You are a storyteller.", messages)
	 *     .setNPredict(128);
	 *
	 * for (LlamaOutput output : model.generateChat(params)) {
	 *     System.out.print(output.text);
	 * }
	 * }</pre>
	 *
	 * @param parameters the inference parameters including messages
	 * @return iterable LLM outputs with the chat template applied
	 * @throws LlamaException if inference fails
	 */
	public LlamaIterable generateChat(InferenceParameters parameters) {
		return new LlamaIterable(new LlamaIterator(this, parameters, true));
	}

	/**
	 * Run a blocking completion and return the full result as a JSON string.
	 * This is the JSON-in/JSON-out equivalent of {@link #complete(InferenceParameters)}.
	 *
	 * @param paramsJson JSON string with at least a "prompt" field
	 * @return JSON response from the server
	 */
	public native String handleCompletions(String paramsJson) throws LlamaException;

	/**
	 * Run an OpenAI-compatible completion (mirrors /v1/completions endpoint).
	 * Returns the result in OAI format with choices array.
	 *
	 * @param paramsJson JSON string with OAI-compatible completion parameters
	 * @return JSON response in OAI format
	 */
	public native String handleCompletionsOai(String paramsJson) throws LlamaException;

	/**
	 * Run a text infill completion with explicit prefix/suffix.
	 * The request JSON must contain "input_prefix" and "input_suffix" fields.
	 *
	 * @param paramsJson JSON string with infill parameters
	 * @return JSON response from the server
	 */
	public native String handleInfill(String paramsJson) throws LlamaException;

	/**
	 * Generate embeddings for the given input. The request JSON should contain
	 * an "input" (OAI-compat) or "content" field.
	 *
	 * @param paramsJson JSON string with embedding request
	 * @param oaiCompat whether to format the response in OAI-compatible format
	 * @return JSON response with embedding vectors
	 */
	public native String handleEmbeddings(String paramsJson, boolean oaiCompat) throws LlamaException;

	/**
	 * Tokenize text content, optionally including token piece information.
	 *
	 * @param content the text to tokenize
	 * @param addSpecial whether to add special tokens (BOS/EOS)
	 * @param withPieces whether to include token piece strings in the response
	 * @return JSON response with token data
	 */
	public native String handleTokenize(String content, boolean addSpecial, boolean withPieces) throws LlamaException;

	/**
	 * Detokenize an array of token IDs back to text.
	 *
	 * @param tokens array of token IDs
	 * @return JSON response with the decoded text
	 */
	public native String handleDetokenize(int[] tokens) throws LlamaException;

	// ------------------------------------------------------------------
	// Server management
	// ------------------------------------------------------------------

	/**
	 * Get server metrics and slot information as a JSON string.
	 *
	 * @return JSON with slot data, idle/processing counts, and performance metrics
	 */
	public String getMetrics() {
		return handleSlotAction(0, 0, null);
	}

	private static final com.fasterxml.jackson.databind.ObjectMapper OBJECT_MAPPER =
			new com.fasterxml.jackson.databind.ObjectMapper();

	/**
	 * Run {@link #complete(InferenceParameters)} constrained to the supplied JSON Schema
	 * and deserialize the result into an instance of {@code type}. The schema is applied
	 * via {@link InferenceParameters#setJsonSchema(String)} for the duration of this call;
	 * the supplied {@code parameters} object is mutated.
	 * <p>
	 * Callers are responsible for producing a JSON Schema that matches the target type;
	 * this project intentionally does not pull in a schema-from-POJO generator. Use the
	 * single-argument overload {@link #completeAsJson(Class, InferenceParameters)} when
	 * the schema has already been set on {@code parameters}.
	 *
	 * @param type       the target POJO class for Jackson deserialization
	 * @param schema     JSON Schema string applied via {@code setJsonSchema}
	 * @param parameters inference parameters (will be mutated to include the schema)
	 * @param <T>        target type
	 * @return parsed POJO of type {@code T}
	 * @throws LlamaException when the response is not valid JSON for the target type
	 */
	public <T> T completeAsJson(Class<T> type, String schema, InferenceParameters parameters) throws LlamaException {
		parameters.setJsonSchema(schema);
		return completeAsJson(type, parameters);
	}

	/**
	 * Run {@link #complete(InferenceParameters)} and deserialize the result as JSON into
	 * {@code type}. The {@code parameters} object should already have a JSON Schema set
	 * via {@link InferenceParameters#setJsonSchema(String)} or a grammar via
	 * {@link InferenceParameters#setGrammar(String)} — otherwise the model output is
	 * unlikely to parse.
	 *
	 * @param type       the target POJO class for Jackson deserialization
	 * @param parameters inference parameters (schema/grammar already set by the caller)
	 * @param <T>        target type
	 * @return parsed POJO of type {@code T}
	 * @throws LlamaException when the response is not valid JSON for the target type
	 */
	public <T> T completeAsJson(Class<T> type, InferenceParameters parameters) throws LlamaException {
		String raw = complete(parameters);
		try {
			return OBJECT_MAPPER.readValue(raw, type);
		} catch (java.io.IOException e) {
			throw new LlamaException("Failed to parse completion as " + type.getSimpleName() + ": " + e.getMessage());
		}
	}

	/**
	 * Typed accessor for {@link #getMetrics()}. Parses the raw JSON into a
	 * {@link ServerMetrics} view that exposes cumulative {@link Usage} and
	 * {@link Timings}, slot counts, and a passthrough to the underlying JSON.
	 *
	 * @return parsed {@link ServerMetrics}
	 * @throws LlamaException if the native call fails or the response cannot be parsed
	 */
	public ServerMetrics getMetricsTyped() throws LlamaException {
		try {
			return new ServerMetrics(OBJECT_MAPPER.readTree(getMetrics()));
		} catch (java.io.IOException e) {
			throw new LlamaException("Failed to parse server metrics JSON: " + e.getMessage());
		}
	}

	/**
	 * Returns model metadata with typed accessors for vocab, context, embedding,
	 * parameter count, size, and modality support flags (vision, audio).
	 * <p>
	 * The returned {@link ModelMeta} wraps the raw JSON from the native layer.
	 * Call {@link ModelMeta#toString()} to re-serialize to compact JSON for use
	 * in {@code assertEquals}.
	 * </p>
	 *
	 * @return {@link ModelMeta} parsed from the native {@code model_meta()} response
	 * @throws LlamaException if the native call fails or the response cannot be parsed
	 */
	public ModelMeta getModelMeta() throws LlamaException {
		try {
			return new ModelMeta(OBJECT_MAPPER.readTree(getModelMetaJson()));
		} catch (java.io.IOException e) {
			throw new LlamaException("Failed to parse model meta JSON: " + e.getMessage());
		}
	}

	native String getModelMetaJson() throws LlamaException;

	/**
	 * Erase the KV cache for a specific slot.
	 *
	 * @param slotId the slot ID to erase
	 * @return JSON with erase result
	 */
	public String eraseSlot(int slotId) {
		return handleSlotAction(3, slotId, null);
	}

	/**
	 * Save a slot's KV cache state to a file.
	 *
	 * @param slotId the slot ID to save
	 * @param filepath the file path to save to
	 * @return JSON with save result
	 */
	public String saveSlot(int slotId, String filepath) {
		return handleSlotAction(1, slotId, filepath);
	}

	/**
	 * Restore a slot's KV cache state from a file.
	 *
	 * @param slotId the slot ID to restore
	 * @param filepath the file path to restore from
	 * @return JSON with restore result
	 */
	public String restoreSlot(int slotId, String filepath) {
		return handleSlotAction(2, slotId, filepath);
	}

	/**
	 * Configure runtime inference parameters.
	 * Accepts a JSON string with optional keys:
	 * <ul>
	 *   <li>"slot_prompt_similarity" (float, 0.0-1.0)</li>
	 *   <li>"n_threads" (int, &gt; 0)</li>
	 *   <li>"n_threads_batch" (int, &gt; 0)</li>
	 * </ul>
	 *
	 * @param configJson JSON configuration string
	 * @return true if configuration was applied successfully
	 */
	public native boolean configureParallelInference(String configJson) throws LlamaException;

	native String handleSlotAction(int action, int slotId, String filename) throws LlamaException;

	native String handleChatCompletions(String params) throws LlamaException;

	native int requestChatCompletion(String params) throws LlamaException;
}
