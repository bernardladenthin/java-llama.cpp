package de.kherud.llama;

import de.kherud.llama.args.LogFormat;
import java.lang.annotation.Native;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
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
		return LlamaOutput.getContentFromJson(json);
	}

	/**
	 * Generate and stream outputs with custom inference parameters. Note, that the prompt isn't preprocessed in any
	 * way, nothing like "User: ", "###Instruction", etc. is added.
	 *
	 * @param parameters the inference configuration
	 * @return iterable LLM outputs
	 */
	public LlamaIterable generate(InferenceParameters parameters) {
		return () -> new LlamaIterator(this, parameters);
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
		List<Pair<String, Float>> rankedDocuments = LlamaOutput.parseRerankResults(json);
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
		List<Pair<String, Float>> results = LlamaOutput.parseRerankResults(json);
		Map<String, Float> probabilities = new HashMap<>();
		for (Pair<String, Float> pair : results) {
			probabilities.put(pair.getKey(), pair.getValue());
		}
		return new LlamaOutput(query, probabilities, true);
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
		return () -> new LlamaIterator(this, parameters, true);
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

	native String handleChatCompletions(String params) throws LlamaException;

	native int requestChatCompletion(String params) throws LlamaException;
}
