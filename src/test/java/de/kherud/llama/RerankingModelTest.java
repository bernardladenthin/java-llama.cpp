package de.kherud.llama;

import java.util.List;
import java.util.Map;

import org.junit.AfterClass;
import org.junit.Assert;
import org.junit.BeforeClass;
import org.junit.Test;

public class RerankingModelTest {

	private static LlamaModel model;
	
	String query = "Machine learning is";
	String[] TEST_DOCUMENTS = new String[] {
			"A machine is a physical system that uses power to apply forces and control movement to perform an action. The term is commonly applied to artificial devices, such as those employing engines or motors, but also to natural biological macromolecules, such as molecular machines.",
			"Learning is the process of acquiring new understanding, knowledge, behaviors, skills, values, attitudes, and preferences. The ability to learn is possessed by humans, non-human animals, and some machines; there is also evidence for some kind of learning in certain plants.",
			"Machine learning is a field of study in artificial intelligence concerned with the development and study of statistical algorithms that can learn from data and generalize to unseen data, and thus perform tasks without explicit instructions.",
			"Paris, capitale de la France, est une grande ville européenne et un centre mondial de l'art, de la mode, de la gastronomie et de la culture. Son paysage urbain du XIXe siècle est traversé par de larges boulevards et la Seine." };

	@BeforeClass
	public static void setup() {
		int gpuLayers = Integer.getInteger(TestConstants.PROP_TEST_NGL, TestConstants.DEFAULT_TEST_NGL);
		model = new LlamaModel(
				new ModelParameters().setCtxSize(128).setModel("models/jina-reranker-v1-tiny-en-Q4_0.gguf")
						.setGpuLayers(gpuLayers).enableReranking().enableLogTimestamps().enableLogPrefix());
	}

	@AfterClass
	public static void tearDown() {
		if (model != null) {
			model.close();
		}
	}

	@Test
	public void testReRanking() {

		
		LlamaOutput llamaOutput = model.rerank(query, TEST_DOCUMENTS[0], TEST_DOCUMENTS[1], TEST_DOCUMENTS[2],
				TEST_DOCUMENTS[3]);

		Map<String, Float> rankedDocumentsMap = llamaOutput.probabilities;
		Assert.assertTrue(rankedDocumentsMap.size()==TEST_DOCUMENTS.length);
		
		 // Finding the most and least relevant documents
        String mostRelevantDoc = null;
        String leastRelevantDoc = null;
        float maxScore = Float.MIN_VALUE;
        float minScore = Float.MAX_VALUE;

        for (Map.Entry<String, Float> entry : rankedDocumentsMap.entrySet()) {
            if (entry.getValue() > maxScore) {
                maxScore = entry.getValue();
                mostRelevantDoc = entry.getKey();
            }
            if (entry.getValue() < minScore) {
                minScore = entry.getValue();
                leastRelevantDoc = entry.getKey();
            }
        }

        // Assertions
        Assert.assertTrue(maxScore > minScore);
        Assert.assertEquals("Machine learning is a field of study in artificial intelligence concerned with the development and study of statistical algorithms that can learn from data and generalize to unseen data, and thus perform tasks without explicit instructions.", mostRelevantDoc);
        Assert.assertEquals("Paris, capitale de la France, est une grande ville européenne et un centre mondial de l'art, de la mode, de la gastronomie et de la culture. Son paysage urbain du XIXe siècle est traversé par de larges boulevards et la Seine.", leastRelevantDoc);

		
	}
	
	@Test
	public void testSortedReRanking() {
		List<Pair<String, Float>> rankedDocuments = model.rerank(true, query, TEST_DOCUMENTS);
		Assert.assertEquals(rankedDocuments.size(), TEST_DOCUMENTS.length);

		// Check the ranking order: each score should be >= the next one
	    for (int i = 0; i < rankedDocuments.size() - 1; i++) {
	        float currentScore = rankedDocuments.get(i).getValue();
	        float nextScore = rankedDocuments.get(i + 1).getValue();
	        Assert.assertTrue("Ranking order incorrect at index " + i, currentScore >= nextScore);
	    }
	}

	// ------------------------------------------------------------------
	// format_rerank(vocab, query, doc) — changed in b8576:
	//   EOS token falls back to SEP when EOS is LLAMA_TOKEN_NULL.
	// These tests exercise the full rerank path end-to-end and verify
	// that the token sequence built by format_rerank produces meaningful
	// scores (which would be wrong / NaN / zero if BOS/EOS/SEP tokens
	// were incorrect).
	// ------------------------------------------------------------------

	/**
	 * Rerank a single document.
	 * Exercises the minimal format_rerank path (one BOS+query+EOS+SEP+doc+EOS
	 * sequence) and verifies a non-zero score is returned.
	 */
	@Test
	public void testRerankSingleDocument() {
		// The ML document is the most relevant one for the query
		LlamaOutput output = model.rerank(query, TEST_DOCUMENTS[2]);

		Assert.assertNotNull(output);
		Assert.assertEquals("Expected exactly one score", 1, output.probabilities.size());

		float score = output.probabilities.values().iterator().next();
		Assert.assertTrue("Score should be positive for a relevant document: " + score, score > 0.0f);
	}

	/**
	 * Verify that all reranking scores are in [0, 1].
	 * The Jina reranker uses RANK pooling with sigmoid activation, so every
	 * score must be a valid probability.  A broken format_rerank (wrong
	 * EOS/SEP tokens) would produce garbage logits and likely scores outside
	 * this range or NaN values.
	 */
	@Test
	public void testRerankScoreRange() {
		LlamaOutput output = model.rerank(query, TEST_DOCUMENTS);

		Assert.assertEquals(TEST_DOCUMENTS.length, output.probabilities.size());

		for (Map.Entry<String, Float> entry : output.probabilities.entrySet()) {
			float score = entry.getValue();
			Assert.assertFalse("Score must not be NaN: " + entry.getKey(), Float.isNaN(score));
			Assert.assertFalse("Score must not be Inf: " + entry.getKey(), Float.isInfinite(score));
			Assert.assertTrue("Score must be >= 0: " + score, score >= 0.0f);
			Assert.assertTrue("Score must be <= 1: " + score, score <= 1.0f);
		}
	}

	/**
	 * Calling rerank twice with the same input must return identical scores.
	 * Verifies determinism of the format_rerank token sequence and the
	 * inference pipeline (server_tokens construction → validate → slot eval).
	 */
	@Test
	public void testRerankConsistency() {
		String doc = TEST_DOCUMENTS[2]; // ML document

		LlamaOutput first  = model.rerank(query, doc);
		LlamaOutput second = model.rerank(query, doc);

		float score1 = first.probabilities.values().iterator().next();
		float score2 = second.probabilities.values().iterator().next();

		Assert.assertEquals("Reranking must be deterministic", score1, score2, 1e-4f);
	}

	/**
	 * The irrelevant (French) document must score lower than the directly
	 * relevant ML document when ranked individually against the same query.
	 * This validates that format_rerank produces a token sequence that
	 * encodes semantic content rather than returning a constant score.
	 */
	@Test
	public void testRerankRelevantVsIrrelevant() {
		LlamaOutput mlOutput     = model.rerank(query, TEST_DOCUMENTS[2]); // ML doc
		LlamaOutput frenchOutput = model.rerank(query, TEST_DOCUMENTS[3]); // French doc

		float mlScore     = mlOutput.probabilities.values().iterator().next();
		float frenchScore = frenchOutput.probabilities.values().iterator().next();

		Assert.assertTrue(
				"ML document should score higher than the French document. " +
				"ml=" + mlScore + " french=" + frenchScore,
				mlScore > frenchScore);
	}
}
