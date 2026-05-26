// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
// SPDX-FileCopyrightText: 2023-2025 Konstantin Herud
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import java.io.*;
import java.util.*;
import java.util.regex.Pattern;

import net.ladenthin.llama.args.LogFormat;
import org.junit.jupiter.api.AfterAll;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

public class LlamaModelTest {

	private static final String prefix = "def remove_non_ascii(s: str) -> str:\n    \"\"\" ";
	private static final String suffix = "\n    return result\n";
	private static final int nPredict = 10;

	/**
	 * Minimum expected tokens when testing cancellation.
	 * The test cancels generation after reaching maxExpectedTokensOnCancel.
	 * Due to significant performance variations across different platforms and accelerators,
	 * the actual token count may vary greatly:
	 * - macOS with Metal (slower): ~2 tokens
	 * - Linux with CUDA (faster): ~4-5 tokens
	 * This range accounts for such variations across different hardware, OS, and versions.
	 */
	private static final int minExpectedTokensOnCancel = 2;

	/**
	 * Maximum expected tokens when testing cancellation.
	 * The test will trigger cancellation when reaching this count to ensure
	 * the cancellation mechanism is properly exercised.
	 * @see #minExpectedTokensOnCancel
	 */
	private static final int maxExpectedTokensOnCancel = 5;

	private static LlamaModel model;

	@BeforeClass
	public static void setup() {
		Assumptions.assumeTrue(new java.io.File("models/codellama-7b.Q2_K.gguf").exists(), "Model file not found, skipping LlamaModelTest");
//		LlamaModel.setLogger(LogFormat.TEXT, (level, msg) -> System.out.println(level + ": " + msg));
		int gpuLayers = Integer.getInteger(TestConstants.PROP_TEST_NGL, TestConstants.DEFAULT_TEST_NGL);
		model = new LlamaModel(
				new ModelParameters()
						.setCtxSize(128)
						.setModel(TestConstants.MODEL_PATH)
						//.setModelUrl("https://huggingface.co/TheBloke/CodeLlama-7B-GGUF/resolve/main/codellama-7b.Q2_K.gguf")
						.setGpuLayers(gpuLayers)
						.setFit(false)
						.enableEmbedding().enableLogTimestamps().enableLogPrefix()
		);
	}

	@AfterClass
	public static void tearDown() {
		if (model != null) {
			model.close();
		}
	}

	@Test
	public void testGenerateAnswer() {
		Map<Integer, Float> logitBias = new HashMap<>();
		logitBias.put(2, 2.0f);
		InferenceParameters params = new InferenceParameters(prefix)
				.setTemperature(0.95f)
				.setStopStrings("\"\"\"")
				.setNPredict(nPredict)
				.setTokenIdBias(logitBias);

		int generated = 0;
		for (LlamaOutput ignored : model.generate(params)) {
			generated++;
		}
		// todo: currently, after generating nPredict tokens, there is an additional empty output
		assertTrue(generated > 0 && generated <= nPredict + 1);
	}

	@Test
	public void testGenerateInfill() {
		Map<Integer, Float> logitBias = new HashMap<>();
		logitBias.put(2, 2.0f);
		InferenceParameters params = new InferenceParameters("")
				.setInputPrefix(prefix)
				.setInputSuffix(suffix )
				.setTemperature(0.95f)
				.setStopStrings("\"\"\"")
				.setNPredict(nPredict)
				.setTokenIdBias(logitBias)
				.setSeed(42);

		int generated = 0;
		for (LlamaOutput ignored : model.generate(params)) {
			generated++;
		}
		assertTrue(generated > 0 && generated <= nPredict + 1);
	}

	@Test
	public void testGenerateGrammar() {
		InferenceParameters params = new InferenceParameters("")
				.setGrammar("root ::= (\"a\" | \"b\")+")
				.setNPredict(nPredict);
		StringBuilder sb = new StringBuilder();
		for (LlamaOutput output : model.generate(params)) {
			sb.append(output);
		}
		String output = sb.toString();

		assertTrue(output.matches("[ab]+"));
		int generated = model.encode(output).length;
		assertTrue(generated > 0 && generated <= nPredict + 1);
	}

	@Test
	public void testCompleteAnswer() {
		Map<Integer, Float> logitBias = new HashMap<>();
		logitBias.put(2, 2.0f);
		InferenceParameters params = new InferenceParameters(prefix)
				.setTemperature(0.95f)
				.setStopStrings("\"\"\"")
				.setNPredict(nPredict)
				.setTokenIdBias(logitBias)
				.setSeed(42);

		String output = model.complete(params);
		assertFalse(output.isEmpty());
	}

	@Test
	public void testCompleteInfillCustom() {
		Map<Integer, Float> logitBias = new HashMap<>();
		logitBias.put(2, 2.0f);
		InferenceParameters params = new InferenceParameters("")
				.setInputPrefix(prefix)
				.setInputSuffix(suffix)
				.setTemperature(0.95f)
				.setStopStrings("\"\"\"")
				.setNPredict(nPredict)
				.setTokenIdBias(logitBias)
				.setSeed(42);

		String output = model.complete(params);
		assertFalse(output.isEmpty());
	}

	@Test
	public void testCompleteGrammar() {
		InferenceParameters params = new InferenceParameters("")
				.setGrammar("root ::= (\"a\" | \"b\")+")
				.setNPredict(nPredict);
		String output = model.complete(params);
		assertTrue(output.matches("[ab]+"), output + " doesn't match [ab]+");
		int generated = model.encode(output).length;
		assertTrue(generated > 0 && generated <= nPredict + 1, "generated count is: " + generated);
		
	}

	@Test
	public void testCancelGenerating() {
		InferenceParameters params = new InferenceParameters(prefix).setNPredict(nPredict);

		int generated = 0;
		LlamaIterator iterator = model.generate(params).iterator();
		while (iterator.hasNext()) {
			iterator.next();
			generated++;
			if (generated == maxExpectedTokensOnCancel) {
				iterator.cancel();
			}
		}
		String errorMessage = String.format(
			"Expected between %d and %d tokens, but got %d. " +
			"This can happen due to timing variations in the llama.cpp inference engine.",
			minExpectedTokensOnCancel, maxExpectedTokensOnCancel, generated
		);
		assertTrue(generated >= minExpectedTokensOnCancel && generated <= maxExpectedTokensOnCancel, errorMessage);
	}

	/**
	 * LlamaIterable implements AutoCloseable. Breaking out of a for-each loop early inside a
	 * try-with-resources block must not throw and must not leave the task slot hanging — the
	 * iterator's close() cancels the native task automatically.
	 */
	@Test
	public void testGenerateAutoCloseOnEarlyBreak() throws Exception {
		InferenceParameters params = new InferenceParameters(prefix).setNPredict(nPredict);

		int collected = 0;
		try (LlamaIterable iterable = model.generate(params)) {
			for (LlamaOutput ignored : iterable) {
				collected++;
				if (collected >= 1) {
					break; // exit before stop token
				}
			}
		} // close() must cancel without throwing

		assertTrue(collected >= 1, "Should have collected at least one token before break");

		// The model must still be usable after an early-exit close
		String result = model.complete(new InferenceParameters(prefix).setNPredict(5));
		assertNotNull(result, "Model must be functional after autoclosed iterator");
	}

	/**
	 * Regression: {@link LlamaIterator#close()} must be idempotent. Calling it
	 * after natural completion (the iterator already drained to its stop token)
	 * and calling it twice on an already-cancelled iterator must not throw and
	 * must not affect subsequent inference.
	 */
	@Test
	public void testIteratorCloseIdempotent() {
		InferenceParameters params = new InferenceParameters(prefix).setNPredict(3);

		// Case A: drain to natural stop, then close()
		LlamaIterable a = model.generate(params);
		for (LlamaOutput ignored : a) {
			// drain
		}
		a.close();
		a.close(); // second close still a no-op

		// Case B: cancel mid-stream, then close()
		LlamaIterator b = model.generate(params).iterator();
		if (b.hasNext()) b.next();
		b.cancel();
		b.close();
		b.close();

		// Model must still be usable
		assertNotNull(model.complete(new InferenceParameters(prefix).setNPredict(3)));
	}

	/**
	 * Regression: {@link LlamaModel#complete(InferenceParameters, CancellationToken)}
	 * must return when {@link CancellationToken#cancel()} is invoked from another
	 * thread, returning whatever text was generated up to that point without
	 * throwing. Cancellation is cooperative — the loop checks the flag at token
	 * boundaries — so the budget here is "much less than full n_predict completion
	 * would take", not instantaneous.
	 */
	@Test
	public void testCompleteWithCancellationToken() throws Exception {
		InferenceParameters params = new InferenceParameters(prefix).setNPredict(512);
		CancellationToken token = new CancellationToken();

		Thread canceller = new Thread(() -> {
			try {
				Thread.sleep(200);
			} catch (InterruptedException ignored) {
			}
			token.cancel();
		});

		long start = System.currentTimeMillis();
		canceller.start();
		String partial = model.complete(params, token);
		long elapsed = System.currentTimeMillis() - start;
		canceller.join();

		// 512 tokens on CPU would take many tens of seconds; cancellation should bring
		// this well under that. Tolerate ~10s for the in-flight token to finish.
		assertTrue(elapsed < 30000, "complete should return within 30s of cancel, took " + elapsed + "ms");
		assertNotNull(partial);
		// Token is reset on return so it can be reused.
		assertFalse(token.isCancelled(), "token should be reset after call returns");

		// Model is still usable
		assertNotNull(model.complete(new InferenceParameters(prefix).setNPredict(3)));
	}

	/**
	 * Regression: {@link LlamaModel#completeAsync(InferenceParameters)} must
	 * complete with the same text {@link LlamaModel#complete(InferenceParameters)}
	 * would have produced, on a background thread.
	 */
	@Test
	public void testCompleteAsync() throws Exception {
		InferenceParameters params = new InferenceParameters(prefix).setNPredict(8).setSeed(42);
		String sync = model.complete(new InferenceParameters(prefix).setNPredict(8).setSeed(42));
		String async = model.completeAsync(params).get(30, java.util.concurrent.TimeUnit.SECONDS);
		assertEquals(sync, async);
	}

	/**
	 * Regression: cancelling the future from {@link LlamaModel#completeAsync(InferenceParameters, CancellationToken)}
	 * must not leak the underlying inference loop or destabilise the model. The
	 * worker thread keeps running until the next token boundary, then returns;
	 * future.cancel(true) only flips the future's state, the whenComplete handler
	 * flips the token, and the cooperative loop unwinds shortly after.
	 */
	@Test
	public void testCompleteAsyncCancelPropagates() throws Exception {
		InferenceParameters params = new InferenceParameters(prefix).setNPredict(512);
		CancellationToken token = new CancellationToken();
		java.util.concurrent.CompletableFuture<String> future = model.completeAsync(params, token);

		Thread.sleep(200);
		future.cancel(true);
		assertTrue(future.isCancelled(), "future should report cancelled");

		// Give the cooperative cancel time to unwind the worker thread before the
		// next call. Polling the model state directly is racy; sleeping a generous
		// interval (one token + cancel propagation) is sufficient on CPU.
		Thread.sleep(5000);

		// Model is still usable
		assertNotNull(model.complete(new InferenceParameters(prefix).setNPredict(3)));
	}

	/**
	 * Regression: {@link Session} must accumulate user/assistant turns across
	 * multiple {@link Session#send(String)} calls and expose them via
	 * {@link Session#getMessages()}. Save/restore round-trip is exercised
	 * separately in slot save/restore tests.
	 */
	@Test
	public void testSessionMultiTurn() {
		try (Session session = new Session(model, 0, "You are a terse assistant.",
				params -> params.setNPredict(8).setSeed(1))) {
			String r1 = session.send("Say hi.");
			assertNotNull(r1);
			String r2 = session.send("Say bye.");
			assertNotNull(r2);

			java.util.List<ChatMessage> msgs = session.getMessages();
			// system + user + assistant + user + assistant
			assertEquals(5, msgs.size());
			assertEquals("system", msgs.get(0).getRole());
			assertEquals("user", msgs.get(1).getRole());
			assertEquals("Say hi.", msgs.get(1).getContent());
			assertEquals("assistant", msgs.get(2).getRole());
			assertEquals("user", msgs.get(3).getRole());
			assertEquals("Say bye.", msgs.get(3).getContent());
			assertEquals("assistant", msgs.get(4).getRole());
		}
	}

	/**
	 * Regression: {@link LlamaModel#chat(ChatRequest)} returns a typed
	 * {@link ChatResponse} with usage / timings populated and at least one
	 * choice carrying assistant content.
	 */
	@Test
	public void testTypedChat() {
		ChatRequest req = new ChatRequest()
				.addMessage("user", "Say hi in one word.")
				.setInferenceCustomizer(p -> p.setNPredict(8).setSeed(1));
		ChatResponse r = model.chat(req);
		assertNotNull(r);
		assertFalse(r.getChoices().isEmpty());
		assertNotNull(r.getFirstMessage());
		assertTrue(r.getUsage().getTotalTokens() > 0);
	}

	/**
	 * Regression: {@link LlamaModel#chatWithTools(ChatRequest, java.util.Map)}
	 * runs at least one round and returns a final {@link ChatResponse} even when
	 * no tools are triggered. CodeLlama-7B is not a tool-trained model, so this
	 * primarily exercises the loop contract; tool wiring is unit-tested in
	 * ChatResponseTest.
	 */
	@Test
	public void testChatWithToolsLoopShortCircuits() {
		ToolDefinition echo = new ToolDefinition("echo", "Echo a string",
				"{\"type\":\"object\",\"properties\":{\"s\":{\"type\":\"string\"}},\"required\":[\"s\"]}");
		ChatRequest req = new ChatRequest()
				.addMessage("user", "Hello.")
				.addTool(echo)
				.setMaxToolRounds(2)
				.setInferenceCustomizer(p -> p.setNPredict(8).setSeed(1));
		java.util.Map<String, ToolHandler> handlers = new java.util.HashMap<>();
		handlers.put("echo", args -> args);
		ChatResponse r = model.chatWithTools(req, handlers);
		assertNotNull(r);
		assertFalse(r.getChoices().isEmpty());
	}

	/**
	 * Regression: {@link LlamaModel#completeBatch(java.util.List)} returns results in
	 * the same order as the input list, with one non-null text per request. The shared
	 * test model is single-slot, so this primarily exercises the parallel dispatch and
	 * order-preservation contract, not actual parallel throughput.
	 */
	@Test
	public void testCompleteBatch() {
		java.util.List<InferenceParameters> requests = java.util.Arrays.asList(
				new InferenceParameters(prefix).setNPredict(3).setSeed(1),
				new InferenceParameters(prefix).setNPredict(3).setSeed(2),
				new InferenceParameters(prefix).setNPredict(3).setSeed(3));
		java.util.List<String> results = model.completeBatch(requests);
		assertEquals(3, results.size());
		for (String r : results) {
			assertNotNull(r);
		}
	}

	@Test
	public void testCompleteBatchWithStats() {
		java.util.List<InferenceParameters> requests = java.util.Arrays.asList(
				new InferenceParameters(prefix).setNPredict(3).setSeed(1),
				new InferenceParameters(prefix).setNPredict(3).setSeed(2));
		java.util.List<CompletionResult> results = model.completeBatchWithStats(requests);
		assertEquals(2, results.size());
		for (CompletionResult r : results) {
			assertNotNull(r);
			assertTrue(r.getUsage().getTotalTokens() > 0, "expected non-zero total tokens, got " + r.getUsage().getTotalTokens());
		}
	}

	@Test
	public void testChatBatch() {
		java.util.List<ChatRequest> requests = java.util.Arrays.asList(
				new ChatRequest().addMessage("user", "Say hi.").setInferenceCustomizer(p -> p.setNPredict(4).setSeed(1)),
				new ChatRequest().addMessage("user", "Say bye.").setInferenceCustomizer(p -> p.setNPredict(4).setSeed(2)));
		java.util.List<ChatResponse> results = model.chatBatch(requests);
		assertEquals(2, results.size());
		for (ChatResponse r : results) {
			assertFalse(r.getChoices().isEmpty());
		}
	}

	@Test
	public void testEmbedding() {
		float[] embedding = model.embed(prefix);
		assertEquals(4096, embedding.length);
	}
	
	
	@Disabled
	/**
	 * To run this test download the model from here https://huggingface.co/mradermacher/jina-reranker-v1-tiny-en-GGUF/tree/main
	 * remove .enableEmbedding() from model setup and add .enableReRanking() and then enable the test.
	 */
	public void testReRanking() {
		
		String query = "Machine learning is";
		String [] TEST_DOCUMENTS = new String[] {
				                  "A machine is a physical system that uses power to apply forces and control movement to perform an action. The term is commonly applied to artificial devices, such as those employing engines or motors, but also to natural biological macromolecules, such as molecular machines.",
				                  "Learning is the process of acquiring new understanding, knowledge, behaviors, skills, values, attitudes, and preferences. The ability to learn is possessed by humans, non-human animals, and some machines; there is also evidence for some kind of learning in certain plants.",
				                  "Machine learning is a field of study in artificial intelligence concerned with the development and study of statistical algorithms that can learn from data and generalize to unseen data, and thus perform tasks without explicit instructions.",
				                  "Paris, capitale de la France, est une grande ville européenne et un centre mondial de l'art, de la mode, de la gastronomie et de la culture. Son paysage urbain du XIXe siècle est traversé par de larges boulevards et la Seine."
		};
		LlamaOutput llamaOutput = model.rerank(query, TEST_DOCUMENTS[0], TEST_DOCUMENTS[1], TEST_DOCUMENTS[2], TEST_DOCUMENTS[3] );
		
		System.out.println(llamaOutput);
	}

	@Test
	public void testTokenization() {
		String prompt = "Hello, world!";
		int[] encoded = model.encode(prompt);
		String decoded = model.decode(encoded);
		// the llama tokenizer adds a space before the prompt
		assertEquals(" " +prompt, decoded);
	}

	@Test
	public void testVocabOnly() {
		try (LlamaModel vocabModel = new LlamaModel(
				new ModelParameters()
						.setModel(TestConstants.MODEL_PATH)
						.setVocabOnly()
		)) {
			String prompt = "Hello, world!";
			int[] encoded = vocabModel.encode(prompt);
			assertTrue(encoded.length > 0, "Should produce at least one token");
			String decoded = vocabModel.decode(encoded);
			assertEquals(" " + prompt, decoded);
		}
	}

	@Test
	public void testVocabOnlyMatchesFullModel() {
		try (LlamaModel vocabModel = new LlamaModel(
				new ModelParameters()
						.setModel(TestConstants.MODEL_PATH)
						.setVocabOnly()
		)) {
			String prompt = "def remove_non_ascii(s: str) -> str:";
			int[] vocabTokens = vocabModel.encode(prompt);
			int[] fullTokens = model.encode(prompt);
			assertArrayEquals(fullTokens, vocabTokens, "Vocab-only tokenization should match full model");
		}
	}

	@Test
	public void testVocabOnlyDecodeEmptyArray() {
		try (LlamaModel vocabModel = new LlamaModel(
				new ModelParameters()
						.setModel(TestConstants.MODEL_PATH)
						.setVocabOnly()
		)) {
			String decoded = vocabModel.decode(new int[0]);
			assertEquals("", decoded, "Decoding empty token array should give empty string");
		}
	}

	@Test
	public void testVocabOnlyUnicodeRoundTrip() {
		try (LlamaModel vocabModel = new LlamaModel(
				new ModelParameters()
						.setModel(TestConstants.MODEL_PATH)
						.setVocabOnly()
		)) {
			// Multi-byte characters: accents, CJK ideograph, emoji
			String prompt = "naïve café résumé";
			int[] tokens = vocabModel.encode(prompt);
			assertTrue(tokens.length > 0, "Unicode string should tokenise to at least one token");
			String decoded = vocabModel.decode(tokens);
			// Leading space is normal (llama tokenizer behaviour); compare content
			assertTrue(decoded.contains("na") && decoded.contains("caf") && decoded.contains("sum"), "Decoded text should contain original characters");
		}
	}

	@Test
	public void testVocabOnlyTwoInstancesCoexist() {
		// Two independent vocab-only models open simultaneously must not interfere.
		try (LlamaModel a = new LlamaModel(
				new ModelParameters().setModel(TestConstants.MODEL_PATH).setVocabOnly());
			 LlamaModel b = new LlamaModel(
				new ModelParameters().setModel(TestConstants.MODEL_PATH).setVocabOnly())
		) {
			String prompt = "hello";
			int[] tokensA = a.encode(prompt);
			int[] tokensB = b.encode(prompt);
			assertArrayEquals(tokensA, tokensB, "Two concurrent vocab-only instances must produce equal tokens");
		}
	}

	@Test
	public void testVocabOnlyCoexistsWithFullModel() {
		// A vocab-only instance must work correctly while the class-level full model is loaded.
		try (LlamaModel vocabModel = new LlamaModel(
				new ModelParameters().setModel(TestConstants.MODEL_PATH).setVocabOnly()
		)) {
			String prompt = "int main()";
			int[] vocabTokens = vocabModel.encode(prompt);
			int[] fullTokens  = model.encode(prompt);
			assertArrayEquals(fullTokens, vocabTokens, "Vocab-only instance must match full-model tokenization");
		}
	}

	@Disabled
	public void testLogText() {
		List<LogMessage> messages = new ArrayList<>();
		LlamaModel.setLogger(LogFormat.TEXT, (level, msg) -> messages.add(new LogMessage(level, msg)));

		InferenceParameters params = new InferenceParameters(prefix)
				.setNPredict(nPredict)
				.setSeed(42);
		model.complete(params);

		assertFalse(messages.isEmpty());

		Pattern jsonPattern = Pattern.compile("^\\s*[\\[{].*[}\\]]\\s*$");
		for (LogMessage message : messages) {
			assertNotNull(message.level);
			assertFalse(jsonPattern.matcher(message.text).matches());
		}
	}

	@Disabled
	public void testLogJSON() {
		List<LogMessage> messages = new ArrayList<>();
		LlamaModel.setLogger(LogFormat.JSON, (level, msg) -> messages.add(new LogMessage(level, msg)));

		InferenceParameters params = new InferenceParameters(prefix)
				.setNPredict(nPredict)
				.setSeed(42);
		model.complete(params);

		assertFalse(messages.isEmpty());

		Pattern jsonPattern = Pattern.compile("^\\s*[\\[{].*[}\\]]\\s*$");
		for (LogMessage message : messages) {
			assertNotNull(message.level);
			assertTrue(jsonPattern.matcher(message.text).matches());
		}
	}

	@Disabled
	@Test
	public void testLogStdout() {
		// Unfortunately, `printf` can't be easily re-directed to Java. This test only works manually, thus.
		InferenceParameters params = new InferenceParameters(prefix)
				.setNPredict(nPredict)
				.setSeed(42);

		System.out.println("########## Log Text ##########");
		LlamaModel.setLogger(LogFormat.TEXT, null);
		model.complete(params);

		System.out.println("########## Log JSON ##########");
		LlamaModel.setLogger(LogFormat.JSON, null);
		model.complete(params);

		System.out.println("########## Log None ##########");
		LlamaModel.setLogger(LogFormat.TEXT, (level, msg) -> {});
		model.complete(params);

		System.out.println("##############################");
	}

	private String completeAndReadStdOut() {
		PrintStream stdOut = System.out;
		ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
		@SuppressWarnings("ImplicitDefaultCharsetUsage") PrintStream printStream = new PrintStream(outputStream);
		System.setOut(printStream);

		try {
			InferenceParameters params = new InferenceParameters(prefix)
					.setNPredict(nPredict)
					.setSeed(42);
			model.complete(params);
		} finally {
			System.out.flush();
			System.setOut(stdOut);
			printStream.close();
		}

		return outputStream.toString();
	}

	private List<String> splitLines(String text) {
		List<String> lines = new ArrayList<>();

		Scanner scanner = new Scanner(text);
		while (scanner.hasNextLine()) {
			String line = scanner.nextLine();
			lines.add(line);
		}
		scanner.close();

		return lines;
	}

	private static final class LogMessage {
		private final LogLevel level;
		private final String text;

		private LogMessage(LogLevel level, String text) {
			this.level = level;
			this.text = text;
		}
	}
	
	@Test
	public void testJsonSchemaToGrammar() {
		String schema = "{\n" +
                "    \"properties\": {\n" +
                "        \"a\": {\"type\": \"string\"},\n" +
                "        \"b\": {\"type\": \"string\"},\n" +
                "        \"c\": {\"type\": \"string\"}\n" +
                "    },\n" +
                "    \"additionalProperties\": false\n" +
                "}";
		
		String expectedGrammar = "a-kv ::= \"\\\"a\\\"\" space \":\" space string\n" +
                "a-rest ::= ( \",\" space b-kv )? b-rest\n" +
                "b-kv ::= \"\\\"b\\\"\" space \":\" space string\n" +
                "b-rest ::= ( \",\" space c-kv )?\n" +
                "c-kv ::= \"\\\"c\\\"\" space \":\" space string\n" +
                "char ::= [^\"\\\\\\x7F\\x00-\\x1F] | [\\\\] ([\"\\\\bfnrt] | \"u\" [0-9a-fA-F]{4})\n" +
                "root ::= \"{\" space  (a-kv a-rest | b-kv b-rest | c-kv )? \"}\" space\n" +
                "space ::= | \" \" | \"\\n\"{1,2} [ \\t]{0,20}\n" +
                "string ::= \"\\\"\" char* \"\\\"\" space\n";
		
		String actualGrammar = LlamaModel.jsonSchemaToGrammar(schema);
		assertEquals(expectedGrammar, actualGrammar);
	}
	
	@Test
	public void testTemplate() {

		List<Pair<String, String>> userMessages = new ArrayList<>();
        userMessages.add(new Pair<>("user", "What is the best book?"));
        userMessages.add(new Pair<>("assistant", "It depends on your interests. Do you like fiction or non-fiction?"));

		InferenceParameters params = new InferenceParameters("A book recommendation system.")
				.setMessages("Book", userMessages)
				.setTemperature(0.95f)
				.setStopStrings("\"\"\"")
				.setNPredict(nPredict)
				.setSeed(42);
		assertEquals(model.applyTemplate(params), "<|im_start|>system\nBook<|im_end|>\n<|im_start|>user\nWhat is the best book?<|im_end|>\n<|im_start|>assistant\nIt depends on your interests. Do you like fiction or non-fiction?");
	}

	// ------------------------------------------------------------------
	// chatComplete / handleChatCompletions
	// ------------------------------------------------------------------

	@Test
	public void testChatComplete() {
		List<Pair<String, String>> messages = new ArrayList<>();
		messages.add(new Pair<>("user", "Write a single word."));

		InferenceParameters params = new InferenceParameters("")
				.setMessages(null, messages)
				.setNPredict(nPredict)
				.setSeed(42)
				.setTemperature(0.0f);

		String response = model.chatComplete(params);
		assertNotNull(response, "Chat completion should return a non-null response");
		assertFalse(response.isEmpty(), "Chat completion should return a non-empty response");
	}

	@Test
	public void testChatCompleteWithSystemMessage() {
		List<Pair<String, String>> messages = new ArrayList<>();
		messages.add(new Pair<>("user", "Say hello."));

		InferenceParameters params = new InferenceParameters("")
				.setMessages("You are a helpful assistant.", messages)
				.setNPredict(nPredict)
				.setSeed(42)
				.setTemperature(0.0f);

		String response = model.chatComplete(params);
		assertNotNull(response);
		assertFalse(response.isEmpty());
	}

	@Test
	public void testGenerateChat() {
		List<Pair<String, String>> messages = new ArrayList<>();
		messages.add(new Pair<>("user", "Write a single word."));

		InferenceParameters params = new InferenceParameters("")
				.setMessages(null, messages)
				.setNPredict(nPredict)
				.setSeed(42)
				.setTemperature(0.0f);

		int generated = 0;
		StringBuilder sb = new StringBuilder();
		for (LlamaOutput output : model.generateChat(params)) {
			sb.append(output.text);
			generated++;
		}
		assertTrue(generated > 0, "Expected at least one token from streaming chat");
		assertTrue(generated <= nPredict + 1, "Expected at most nPredict+1 tokens");
		assertFalse(sb.toString().isEmpty(), "Streamed content should not be empty");
	}

	@Test
	public void testGenerateChatCancel() {
		List<Pair<String, String>> messages = new ArrayList<>();
		messages.add(new Pair<>("user", "Count from 1 to 100."));

		InferenceParameters params = new InferenceParameters("")
				.setMessages(null, messages)
				.setNPredict(nPredict);

		int generated = 0;
		LlamaIterator iterator = model.generateChat(params).iterator();
		while (iterator.hasNext()) {
			iterator.next();
			generated++;
			if (generated == maxExpectedTokensOnCancel) {
				iterator.cancel();
			}
		}
		assertTrue(generated >= minExpectedTokensOnCancel, "Expected at least " + minExpectedTokensOnCancel + " tokens, got " + generated);
		assertTrue(generated <= maxExpectedTokensOnCancel, "Expected at most " + maxExpectedTokensOnCancel + " tokens, got " + generated);
	}

	@Test
	public void testChatCompleteMultiTurn() {
		List<Pair<String, String>> messages = new ArrayList<>();
		messages.add(new Pair<>("user", "What is 2+2?"));
		messages.add(new Pair<>("assistant", "4"));
		messages.add(new Pair<>("user", "And 3+3?"));

		InferenceParameters params = new InferenceParameters("")
				.setMessages(null, messages)
				.setNPredict(nPredict)
				.setSeed(42)
				.setTemperature(0.0f);

		String response = model.chatComplete(params);
		assertNotNull(response);
		assertFalse(response.isEmpty());
	}

	@Test
	public void testChatCompleteWithTemplateKwargs() {
		List<Pair<String, String>> messages = new ArrayList<>();
		messages.add(new Pair<>("user", "Hello"));

		Map<String, String> kwargs = new HashMap<>();
		kwargs.put("custom_var", "\"test_value\"");

		InferenceParameters params = new InferenceParameters("")
				.setMessages(null, messages)
				.setChatTemplateKwargs(kwargs)
				.setNPredict(nPredict)
				.setSeed(42)
				.setTemperature(0.0f);

		// Template kwargs should pass through without error even if
		// the template doesn't use them — they're simply ignored.
		String response = model.chatComplete(params);
		assertNotNull(response);
		assertFalse(response.isEmpty());
	}

	@Test
	public void testApplyTemplateWithKwargs() {
		List<Pair<String, String>> messages = new ArrayList<>();
		messages.add(new Pair<>("user", "Hello"));

		Map<String, String> kwargs = new HashMap<>();
		kwargs.put("custom_var", "\"test_value\"");

		InferenceParameters params = new InferenceParameters("")
				.setMessages(null, messages)
				.setChatTemplateKwargs(kwargs);

		// Should not throw — kwargs are passed through to the template
		String result = model.applyTemplate(params);
		assertNotNull(result);
		assertTrue(result.contains("Hello"));
	}

	// ------------------------------------------------------------------
	// applyTemplate / oaicompat_chat_params_parse (changed in b8576)
	// ------------------------------------------------------------------

	/**
	 * oaicompat_chat_params_parse with a single user message and no system message.
	 * The existing testTemplate() only tests system + user + assistant.
	 * This exercises the minimal messages path and verifies that the
	 * generation prompt (assistant prefix) is appended when the last
	 * message is from the user.
	 */
	@Test
	public void testApplyTemplateUserOnly() {
		List<Pair<String, String>> messages = new ArrayList<>();
		messages.add(new Pair<>("user", "Tell me a joke"));

		InferenceParameters params = new InferenceParameters("")
				.setMessages(null, messages);

		String result = model.applyTemplate(params);

		assertNotNull(result);
		assertTrue(result.contains("<|im_start|>user"), "Expected user role marker");
		assertTrue(result.contains("Tell me a joke"), "Expected message content");
		assertFalse(result.contains("<|im_start|>system"), "Should not have system block when none given");
		// add_generation_prompt defaults to true → assistant continuation is appended
		assertTrue(result.contains("<|im_start|>assistant"), "Expected assistant continuation prompt");
	}

	/**
	 * oaicompat_chat_params_parse with multiple turns: system + user → assistant → user.
	 * Verifies that all messages appear in correct order and the assistant turn
	 * in the middle is correctly delimited.
	 */
	@Test
	public void testApplyTemplateMultipleTurns() {
		List<Pair<String, String>> messages = new ArrayList<>();
		messages.add(new Pair<>("user",      "What is 2+2?"));
		messages.add(new Pair<>("assistant", "4"));
		messages.add(new Pair<>("user",      "And 3+3?"));

		InferenceParameters params = new InferenceParameters("")
				.setMessages("Math tutor", messages);

		String result = model.applyTemplate(params);

		assertTrue(result.contains("What is 2+2?"));
		assertTrue(result.contains("And 3+3?"));
		// The intermediate assistant reply must also be present
		assertTrue(result.contains("4"), "Intermediate assistant turn missing");
		// Last message is user → generation prompt adds assistant prefix
		assertTrue(result.contains("<|im_start|>assistant"));
	}

	/**
	 * Empty system message must be treated the same as no system message
	 * (setMessages skips the system block when the string is empty).
	 */
	@Test
	public void testApplyTemplateEmptySystemSkipped() {
		List<Pair<String, String>> messages = new ArrayList<>();
		messages.add(new Pair<>("user", "Hello"));

		// empty string → setMessages skips the system block
		InferenceParameters params = new InferenceParameters("")
				.setMessages("", messages);

		String result = model.applyTemplate(params);

		assertFalse(result.contains("<|im_start|>system"), "Empty system message must not produce a system block");
		assertTrue(result.contains("Hello"));
	}

	/**
	 * When the conversation ends with an assistant turn, oaicompat_chat_params_parse
	 * must NOT append another generation prompt — it should instead allow the
	 * caller to continue the partially generated assistant response.
	 */
	@Test
	public void testApplyTemplateLastMessageAssistantNoContinuationPrompt() {
		List<Pair<String, String>> messages = new ArrayList<>();
		messages.add(new Pair<>("user",      "Capital of France?"));
		messages.add(new Pair<>("assistant", "The capital of France is"));

		InferenceParameters params = new InferenceParameters("")
				.setMessages(null, messages);

		String result = model.applyTemplate(params);

		assertTrue(result.contains("The capital of France is"));
		// There must not be a second <|im_start|>assistant after the partial reply
		int firstAssistant = result.indexOf("<|im_start|>assistant");
		int secondAssistant = result.indexOf("<|im_start|>assistant", firstAssistant + 1);
		assertEquals(-1, secondAssistant, "Should have exactly one assistant block");
	}

	// ------------------------------------------------------------------
	// server_tokens::detokenize / validate — exercised via generate/complete
	// ------------------------------------------------------------------

	/**
	 * Multi-byte UTF-8 in the prompt exercises server_tokens construction
	 * from tokenized_prompts and subsequently server_tokens::validate(ctx)
	 * and detokenize() for the generated output.
	 */
	@Test
	public void testCompleteNonAsciiPrompt() {
		// café, naïve, résumé contain multi-byte UTF-8 sequences
		InferenceParameters params = new InferenceParameters("Translate to English: café")
				.setNPredict(nPredict)
				.setSeed(42);

		String output = model.complete(params);

		// If server_tokens / detokenize is broken, this throws or returns garbage
		assertNotNull(output);
	}

	/**
	 * Verifies that the JNI string conversion ({@code parse_jstring}) correctly
	 * handles multi-byte UTF-8 input through the full model's encode/decode path.
	 *
	 * <p>The test covers three UTF-8 byte widths:
	 * <ul>
	 *   <li>2-byte sequences: Latin characters with diacritics (ü, ö, é)</li>
	 *   <li>3-byte sequences: CJK ideographs (日, 本, 語)</li>
	 *   <li>Mixing both in a single prompt</li>
	 * </ul>
	 *
	 * <p>A successful encode→decode round-trip proves that the native layer
	 * receives the bytes intact and that no truncation or mojibake occurs.
	 */
	@Test
	public void testTokenizationUnicode() {
		// 2-byte UTF-8: Latin extended (U+00FC, U+00F6, U+00E9)
		String latin = "über, größe, résumé";
		int[] latinTokens = model.encode(latin);
		assertTrue(latinTokens.length > 0, "Latin extended string should produce at least one token");
		String latinDecoded = model.decode(latinTokens);
		assertTrue(latinDecoded.contains("ber") && latinDecoded.contains("r") && latinDecoded.contains("sum"), "Decoded Latin-extended text should preserve multi-byte chars");

		// 3-byte UTF-8: CJK (U+65E5, U+672C, U+8A9E)
		String cjk = "日本語";
		int[] cjkTokens = model.encode(cjk);
		assertTrue(cjkTokens.length > 0, "CJK string should produce at least one token");
		// Decode must not throw and must return a non-empty string
		String cjkDecoded = model.decode(cjkTokens);
		assertNotNull(cjkDecoded, "CJK decode result must not be null");

		// Mixed 2-byte and 3-byte in one prompt – exercises the full JNI path with a
		// realistic combined payload to catch any length-calculation off-by-one errors.
		String mixed = "résumé 日本語 über";
		int[] mixedTokens = model.encode(mixed);
		assertTrue(mixedTokens.length > 0, "Mixed Unicode string should produce at least one token");
		String mixedDecoded = model.decode(mixedTokens);
		assertNotNull(mixedDecoded, "Mixed Unicode decode result must not be null");
		assertFalse(mixedDecoded.isEmpty(), "Mixed Unicode decode result must not be empty");
	}

	/**
	 * Returns true if the file at {@code path} exists and begins with the 4-byte GGUF magic
	 * (0x47 0x47 0x55 0x46 = "GGUF"), distinguishing a properly downloaded model from a
	 * truncated file or an HTML error page saved by {@code curl} without {@code --fail}.
	 */
	private static boolean isValidGGUF(String path) {
		File f = new File(path);
		if (!f.exists() || f.length() < 4) return false;
		try (FileInputStream fis = new FileInputStream(f)) {
			byte[] magic = new byte[4];
			if (fis.read(magic) < 4) return false;
			return magic[0] == 0x47 && magic[1] == 0x47 && magic[2] == 0x55 && magic[3] == 0x46;
		} catch (IOException e) {
			return false;
		}
	}

	// ------------------------------------------------------------------
	// Phase 5: JSON-in/JSON-out endpoints
	// ------------------------------------------------------------------

	@Test
	public void testHandleCompletions() {
		String json = "{\"prompt\": \"Hello\", \"n_predict\": " + nPredict + ", \"seed\": 42, \"temperature\": 0.0}";
		String response = model.handleCompletions(json);
		assertNotNull(response);
		assertTrue(response.contains("\"content\""), "Response should contain content field");
	}

	@Test
	public void testHandleCompletionsOai() {
		String json = "{\"prompt\": \"Hello\", \"max_tokens\": " + nPredict + ", \"seed\": 42, \"temperature\": 0.0}";
		String response = model.handleCompletionsOai(json);
		assertNotNull(response);
		assertTrue(response.contains("\"choices\""), "OAI response should contain choices");
	}

	@Test
	public void testHandleEmbeddings() {
		String json = "{\"content\": \"Hello world\"}";
		String response = model.handleEmbeddings(json, false);
		assertNotNull(response);
		assertTrue(response.contains("\"embedding\""), "Embedding response should contain embedding data");
	}

	@Test
	public void testHandleTokenize() {
		String response = model.handleTokenize("Hello world", false, false);
		assertNotNull(response);
		assertTrue(response.contains("\"tokens\""), "Tokenize response should contain tokens");
	}

	@Test
	public void testHandleTokenizeWithPieces() {
		String response = model.handleTokenize("Hello world", false, true);
		assertNotNull(response);
		assertTrue(response.contains("\"piece\""), "Response should contain token pieces");
	}

	@Test
	public void testHandleDetokenize() {
		int[] tokens = model.encode("Hello");
		String response = model.handleDetokenize(tokens);
		assertNotNull(response);
		assertTrue(response.contains("\"content\""), "Detokenize response should contain content");
		assertTrue(response.contains("Hello"), "Detokenize should contain original text");
	}

	// ------------------------------------------------------------------
	// Thread cleanup / model lifecycle
	// ------------------------------------------------------------------

	@Test
	public void testCreateAndImmediatelyClose() {
		// Verifies that close() joins the background thread without hanging or crashing.
		int gpuLayers = Integer.getInteger(TestConstants.PROP_TEST_NGL, TestConstants.DEFAULT_TEST_NGL);
		try (LlamaModel m = new LlamaModel(
				new ModelParameters()
						.setModel(TestConstants.MODEL_PATH)
						.setCtxSize(32)
						.setGpuLayers(gpuLayers)
						.setFit(false))) {
			// Immediately closed by try-with-resources
		}
		// If we get here without SIGABRT, the thread was joined cleanly
	}

	@Test
	public void testCloseAfterGeneration() {
		// Verifies that close() works correctly after active generation.
		int gpuLayers = Integer.getInteger(TestConstants.PROP_TEST_NGL, TestConstants.DEFAULT_TEST_NGL);
		try (LlamaModel m = new LlamaModel(
				new ModelParameters()
						.setModel(TestConstants.MODEL_PATH)
						.setCtxSize(64)
						.setGpuLayers(gpuLayers)
						.setFit(false))) {
			String output = m.complete(new InferenceParameters("Hello")
					.setNPredict(5)
					.setSeed(42));
			assertNotNull(output);
		}
		// Background thread should be fully joined before we reach here
	}

	// ------------------------------------------------------------------
	// Phase 6: Server management
	// ------------------------------------------------------------------

	@Test
	public void testGetMetrics() {
		String metrics = model.getMetrics();
		assertNotNull(metrics);
		assertTrue(metrics.contains("\"slots\""), "Metrics should contain slots data");
		assertTrue(metrics.contains("\"idle\""), "Metrics should contain idle count");
	}

	@Test
	public void testEraseSlot() {
		String result = model.eraseSlot(0);
		assertNotNull(result);
		assertTrue(result.contains("\"id_slot\""), "Erase result should contain id_slot");
		assertTrue(result.contains("\"n_erased\""), "Erase result should contain n_erased");
	}

	@Test
	public void testConfigureParallelInference() {
		boolean result = model.configureParallelInference("{\"slot_prompt_similarity\": 0.5}");
		assertTrue(result, "Configuration should succeed");
	}

	@Test
	public void testConfigureParallelInferenceInvalidSimilarity() {
	assertThrows(LlamaException.class, () -> model.configureParallelInference("{\"slot_prompt_similarity\": 2.0}"));
	}

	@Test
	public void testSpeculativeDecoding() {
		Assumptions.assumeTrue(isValidGGUF(TestConstants.DRAFT_MODEL_PATH), "Draft model not available or not a valid GGUF; skipping speculative decoding test");
		int gpuLayers = Integer.getInteger(TestConstants.PROP_TEST_NGL, TestConstants.DEFAULT_TEST_NGL);
		try (LlamaModel specModel = new LlamaModel(
				new ModelParameters()
						.setModel(TestConstants.MODEL_PATH)
						.setModelDraft(TestConstants.DRAFT_MODEL_PATH)
						.setCtxSize(128)
						.setDraftMax(8)
						.setDraftMin(1)
						.setGpuLayers(gpuLayers)
						.setGpuLayersDraft(gpuLayers)
		)) {
			InferenceParameters params = new InferenceParameters(prefix)
					.setNPredict(nPredict)
					.setSeed(42);

			// test streaming generation with speculative decoding
			int generated = 0;
			for (LlamaOutput ignored : specModel.generate(params)) {
				generated++;
			}
			assertTrue(generated > 0 && generated <= nPredict + 1, "Expected tokens from speculative generate, got " + generated);

			// test complete with speculative decoding
			String response = specModel.complete(params);
			assertNotNull(response);
			assertFalse(response.isEmpty(), "Expected non-empty response from speculative complete");
		}
	}

	@Test
	public void testGetModelMeta() throws LlamaException {
		ModelMeta meta = model.getModelMeta();

		// Typed getters — exact values depend on the loaded model; fill in after first run
		assertTrue(meta.getNVocab() > 0, "n_vocab must be positive");
		assertTrue(meta.getNCtxTrain() > 0, "n_ctx_train must be positive");
		assertTrue(meta.getNEmbd() > 0, "n_embd must be positive");
		assertTrue(meta.getNParams() > 0, "n_params must be positive");
		assertTrue(meta.getSize() > 0, "size must be positive");

		// CodeLlama (text-only model) must not report multimodal support
		assertFalse(meta.supportsVision(), "text-only model must not report vision support");
		assertFalse(meta.supportsAudio(), "text-only model must not report audio support");

		// Dynamic access via the underlying JsonNode
		assertTrue(meta.asJson().has("modalities"), "modalities field must be present");
		assertTrue(meta.asJson().has("vocab_type"), "vocab_type field must be present");

		// Architecture and name from GGUF general.* metadata
		String architecture = meta.getArchitecture();
		assertNotNull(architecture, "getArchitecture() must not return null");
		assertFalse(architecture.isEmpty(), "CodeLlama GGUF must have general.architecture set");

		// general.name may or may not be present in the GGUF; just verify the getter does not throw
		String modelName = meta.getModelName();
		assertNotNull(modelName, "getModelName() must not return null");

		// Round-trip: toString() must produce valid compact JSON containing all top-level keys
		String json = meta.toString();
		assertNotNull(json);
		assertTrue(json.contains("\"vocab_type\""));
		assertTrue(json.contains("\"n_vocab\""));
		assertTrue(json.contains("\"n_ctx_train\""));
		assertTrue(json.contains("\"n_embd\""));
		assertTrue(json.contains("\"n_params\""));
		assertTrue(json.contains("\"size\""));
		assertTrue(json.contains("\"modalities\""));
		assertTrue(json.contains("\"vision\""));
		assertTrue(json.contains("\"audio\""));
		assertTrue(json.contains("\"architecture\""));
		assertTrue(json.contains("\"name\""));
	}

	/**
	 * Upstream issue <a href="https://github.com/kherud/llama.cpp/issues/95">#95</a>:
	 * reporter argued the iterator could continue emitting tokens after {@code stop=true}.
	 * Current {@link LlamaIterator#next()} reads the JSON output, sets
	 * {@code hasNext = !output.stop}, releases the task on stop, and returns the current
	 * output. The next {@code hasNext()} call then returns false.
	 *
	 * <p>This regression test drives the iterator with a deliberately repetitive prompt
	 * (a sampler-tuning corner case) and asserts iteration terminates deterministically
	 * within {@code nPredict + 1} steps. {@code nPredict + 1} accounts for the one trailing
	 * empty output noted in {@link #testGenerateAnswer()}.
	 */
	@Test
	public void testIteratorTerminatesOnRepetitivePrompt() {
		final int iterNPredict = 30;
		InferenceParameters infer = new InferenceParameters("Repeat AAA forever: AAA AAA")
				.setNPredict(iterNPredict)
				.setTemperature(0.0f);

		int count = 0;
		try (LlamaIterable iterable = model.generate(infer)) {
			for (LlamaOutput ignored : iterable) {
				count++;
				assertTrue(count <= iterNPredict + 1, "iterator overran nPredict=" + iterNPredict + " (count=" + count + ")");
			}
		}
		assertTrue(count >= 1, "iterator must produce at least one token");
	}
}
