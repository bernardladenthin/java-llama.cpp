package de.kherud.llama;

import java.io.*;
import java.util.*;
import java.util.regex.Pattern;

import de.kherud.llama.args.LogFormat;
import org.junit.AfterClass;
import org.junit.Assert;
import org.junit.Assume;
import org.junit.BeforeClass;
import org.junit.Ignore;
import org.junit.Test;

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
		Assume.assumeTrue("Model file not found, skipping LlamaModelTest", new java.io.File("models/codellama-7b.Q2_K.gguf").exists());
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
		Assert.assertTrue(generated > 0 && generated <= nPredict + 1);
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
		Assert.assertTrue(generated > 0 && generated <= nPredict + 1);
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

		Assert.assertTrue(output.matches("[ab]+"));
		int generated = model.encode(output).length;
		Assert.assertTrue(generated > 0 && generated <= nPredict + 1);
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
		Assert.assertFalse(output.isEmpty());
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
		Assert.assertFalse(output.isEmpty());
	}

	@Test
	public void testCompleteGrammar() {
		InferenceParameters params = new InferenceParameters("")
				.setGrammar("root ::= (\"a\" | \"b\")+")
				.setNPredict(nPredict);
		String output = model.complete(params);
		Assert.assertTrue(output + " doesn't match [ab]+", output.matches("[ab]+"));
		int generated = model.encode(output).length;
		Assert.assertTrue("generated count is: " + generated,  generated > 0 && generated <= nPredict + 1);
		
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
		Assert.assertTrue(errorMessage,
			generated >= minExpectedTokensOnCancel && generated <= maxExpectedTokensOnCancel
		);
	}

	@Test
	public void testEmbedding() {
		float[] embedding = model.embed(prefix);
		Assert.assertEquals(4096, embedding.length);
	}
	
	
	@Ignore
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
		Assert.assertEquals(" " +prompt, decoded);
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
			Assert.assertTrue("Should produce at least one token", encoded.length > 0);
			String decoded = vocabModel.decode(encoded);
			Assert.assertEquals(" " + prompt, decoded);
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
			Assert.assertArrayEquals("Vocab-only tokenization should match full model", fullTokens, vocabTokens);
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
			Assert.assertEquals("Decoding empty token array should give empty string", "", decoded);
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
			Assert.assertTrue("Unicode string should tokenise to at least one token", tokens.length > 0);
			String decoded = vocabModel.decode(tokens);
			// Leading space is normal (llama tokenizer behaviour); compare content
			Assert.assertTrue("Decoded text should contain original characters",
					decoded.contains("na") && decoded.contains("caf") && decoded.contains("sum"));
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
			Assert.assertArrayEquals("Two concurrent vocab-only instances must produce equal tokens",
					tokensA, tokensB);
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
			Assert.assertArrayEquals("Vocab-only instance must match full-model tokenization",
					fullTokens, vocabTokens);
		}
	}

	@Ignore
	public void testLogText() {
		List<LogMessage> messages = new ArrayList<>();
		LlamaModel.setLogger(LogFormat.TEXT, (level, msg) -> messages.add(new LogMessage(level, msg)));

		InferenceParameters params = new InferenceParameters(prefix)
				.setNPredict(nPredict)
				.setSeed(42);
		model.complete(params);

		Assert.assertFalse(messages.isEmpty());

		Pattern jsonPattern = Pattern.compile("^\\s*[\\[{].*[}\\]]\\s*$");
		for (LogMessage message : messages) {
			Assert.assertNotNull(message.level);
			Assert.assertFalse(jsonPattern.matcher(message.text).matches());
		}
	}

	@Ignore
	public void testLogJSON() {
		List<LogMessage> messages = new ArrayList<>();
		LlamaModel.setLogger(LogFormat.JSON, (level, msg) -> messages.add(new LogMessage(level, msg)));

		InferenceParameters params = new InferenceParameters(prefix)
				.setNPredict(nPredict)
				.setSeed(42);
		model.complete(params);

		Assert.assertFalse(messages.isEmpty());

		Pattern jsonPattern = Pattern.compile("^\\s*[\\[{].*[}\\]]\\s*$");
		for (LogMessage message : messages) {
			Assert.assertNotNull(message.level);
			Assert.assertTrue(jsonPattern.matcher(message.text).matches());
		}
	}

	@Ignore
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
		Assert.assertEquals(expectedGrammar, actualGrammar);
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
		Assert.assertEquals(model.applyTemplate(params), "<|im_start|>system\nBook<|im_end|>\n<|im_start|>user\nWhat is the best book?<|im_end|>\n<|im_start|>assistant\nIt depends on your interests. Do you like fiction or non-fiction?");
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
		Assert.assertNotNull("Chat completion should return a non-null response", response);
		Assert.assertFalse("Chat completion should return a non-empty response", response.isEmpty());
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
		Assert.assertNotNull(response);
		Assert.assertFalse(response.isEmpty());
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
		Assert.assertTrue("Expected at least one token from streaming chat", generated > 0);
		Assert.assertTrue("Expected at most nPredict+1 tokens", generated <= nPredict + 1);
		Assert.assertFalse("Streamed content should not be empty", sb.toString().isEmpty());
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
		Assert.assertTrue("Expected at least " + minExpectedTokensOnCancel + " tokens, got " + generated,
				generated >= minExpectedTokensOnCancel);
		Assert.assertTrue("Expected at most " + maxExpectedTokensOnCancel + " tokens, got " + generated,
				generated <= maxExpectedTokensOnCancel);
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
		Assert.assertNotNull(response);
		Assert.assertFalse(response.isEmpty());
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

		Assert.assertNotNull(result);
		Assert.assertTrue("Expected user role marker", result.contains("<|im_start|>user"));
		Assert.assertTrue("Expected message content", result.contains("Tell me a joke"));
		Assert.assertFalse("Should not have system block when none given",
				result.contains("<|im_start|>system"));
		// add_generation_prompt defaults to true → assistant continuation is appended
		Assert.assertTrue("Expected assistant continuation prompt",
				result.contains("<|im_start|>assistant"));
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

		Assert.assertTrue(result.contains("What is 2+2?"));
		Assert.assertTrue(result.contains("And 3+3?"));
		// The intermediate assistant reply must also be present
		Assert.assertTrue("Intermediate assistant turn missing", result.contains("4"));
		// Last message is user → generation prompt adds assistant prefix
		Assert.assertTrue(result.contains("<|im_start|>assistant"));
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

		Assert.assertFalse("Empty system message must not produce a system block",
				result.contains("<|im_start|>system"));
		Assert.assertTrue(result.contains("Hello"));
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

		Assert.assertTrue(result.contains("The capital of France is"));
		// There must not be a second <|im_start|>assistant after the partial reply
		int firstAssistant = result.indexOf("<|im_start|>assistant");
		int secondAssistant = result.indexOf("<|im_start|>assistant", firstAssistant + 1);
		Assert.assertEquals("Should have exactly one assistant block", -1, secondAssistant);
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
		Assert.assertNotNull(output);
	}

	/**
	 * Tokenise and immediately decode a string containing multi-byte UTF-8.
	 * Exercises the tokenisation round-trip used to build server_tokens.
	 */
	@Test
	public void testTokenizationUnicode() {
		String prompt = "naïve résumé café";
		int[] tokens = model.encode(prompt);
		Assert.assertTrue("Unicode string should tokenise to at least one token",
				tokens.length > 0);

		String decoded = model.decode(tokens);
		// Leading space may be added by the tokenizer; check content is preserved
		Assert.assertTrue("Decoded text should contain original characters",
				decoded.contains("na") && decoded.contains("sum"));
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
		Assert.assertNotNull(response);
		Assert.assertTrue("Response should contain content field", response.contains("\"content\""));
	}

	@Test
	public void testHandleCompletionsOai() {
		String json = "{\"prompt\": \"Hello\", \"max_tokens\": " + nPredict + ", \"seed\": 42, \"temperature\": 0.0}";
		String response = model.handleCompletionsOai(json);
		Assert.assertNotNull(response);
		Assert.assertTrue("OAI response should contain choices", response.contains("\"choices\""));
	}

	@Test
	public void testHandleEmbeddings() {
		String json = "{\"content\": \"Hello world\"}";
		String response = model.handleEmbeddings(json, false);
		Assert.assertNotNull(response);
		Assert.assertTrue("Embedding response should contain embedding data", response.contains("\"embedding\""));
	}

	@Test
	public void testHandleTokenize() {
		String response = model.handleTokenize("Hello world", false, false);
		Assert.assertNotNull(response);
		Assert.assertTrue("Tokenize response should contain tokens", response.contains("\"tokens\""));
	}

	@Test
	public void testHandleTokenizeWithPieces() {
		String response = model.handleTokenize("Hello world", false, true);
		Assert.assertNotNull(response);
		Assert.assertTrue("Response should contain token pieces", response.contains("\"piece\""));
	}

	@Test
	public void testHandleDetokenize() {
		int[] tokens = model.encode("Hello");
		String response = model.handleDetokenize(tokens);
		Assert.assertNotNull(response);
		Assert.assertTrue("Detokenize response should contain content", response.contains("\"content\""));
		Assert.assertTrue("Detokenize should contain original text", response.contains("Hello"));
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
			Assert.assertNotNull(output);
		}
		// Background thread should be fully joined before we reach here
	}

	// ------------------------------------------------------------------
	// Phase 6: Server management
	// ------------------------------------------------------------------

	@Test
	public void testGetMetrics() {
		String metrics = model.getMetrics();
		Assert.assertNotNull(metrics);
		Assert.assertTrue("Metrics should contain slots data", metrics.contains("\"slots\""));
		Assert.assertTrue("Metrics should contain idle count", metrics.contains("\"idle\""));
	}

	@Test
	public void testEraseSlot() {
		String result = model.eraseSlot(0);
		Assert.assertNotNull(result);
		Assert.assertTrue("Erase result should contain id_slot", result.contains("\"id_slot\""));
		Assert.assertTrue("Erase result should contain n_erased", result.contains("\"n_erased\""));
	}

	@Test
	public void testConfigureParallelInference() {
		boolean result = model.configureParallelInference("{\"slot_prompt_similarity\": 0.5}");
		Assert.assertTrue("Configuration should succeed", result);
	}

	@Test(expected = LlamaException.class)
	public void testConfigureParallelInferenceInvalidSimilarity() {
		model.configureParallelInference("{\"slot_prompt_similarity\": 2.0}");
	}

	@Test
	public void testSpeculativeDecoding() {
		Assume.assumeTrue(
				"Draft model not available or not a valid GGUF; skipping speculative decoding test",
				isValidGGUF(TestConstants.DRAFT_MODEL_PATH));
		int gpuLayers = Integer.getInteger(TestConstants.PROP_TEST_NGL, TestConstants.DEFAULT_TEST_NGL);
		try (LlamaModel specModel = new LlamaModel(
				new ModelParameters()
						.setModel(TestConstants.MODEL_PATH)
						.setModelDraft(TestConstants.DRAFT_MODEL_PATH)
						.setCtxSize(128)
						.setCtxSizeDraft(128)
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
			Assert.assertTrue("Expected tokens from speculative generate, got " + generated,
					generated > 0 && generated <= nPredict + 1);

			// test complete with speculative decoding
			String response = specModel.complete(params);
			Assert.assertNotNull(response);
			Assert.assertFalse("Expected non-empty response from speculative complete", response.isEmpty());
		}
	}
}
