package net.ladenthin.llama;

class TestConstants {

	/** System property to override GPU layers used in tests. */
	static final String PROP_TEST_NGL = LlamaSystemProperties.PREFIX + ".test.ngl";

	static final int DEFAULT_TEST_NGL = 43;

	/** Path to the main text generation model used in tests. */
	static final String MODEL_PATH = "models/codellama-7b.Q2_K.gguf";

	/** Path to the draft model used for speculative decoding tests. */
	static final String DRAFT_MODEL_PATH = "models/AMD-Llama-135m-code.Q2_K.gguf";

}
