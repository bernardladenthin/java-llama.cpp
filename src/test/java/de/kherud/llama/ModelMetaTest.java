package de.kherud.llama;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * Unit tests for {@link ModelMeta} typed getters.
 * Constructs {@code ModelMeta} directly from JSON strings — no native library or model file required.
 */
@ClaudeGenerated(
        purpose = "Verify that ModelMeta typed getters map correctly from the underlying JsonNode, " +
                  "including the new architecture and name fields from GGUF general.* metadata."
)
public class ModelMetaTest {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    private ModelMeta parse(String json) throws Exception {
        return new ModelMeta(MAPPER.readTree(json));
    }

    @Test
    public void testNumericGetters() throws Exception {
        ModelMeta meta = parse("{\"vocab_type\":1,\"n_vocab\":32016,\"n_ctx_train\":16384," +
                "\"n_embd\":4096,\"n_params\":6738546688,\"size\":2825274880," +
                "\"modalities\":{\"vision\":false,\"audio\":false}," +
                "\"architecture\":\"llama\",\"name\":\"CodeLlama-7B\"}");

        assertEquals(1, meta.getVocabType());
        assertEquals(32016, meta.getNVocab());
        assertEquals(16384, meta.getNCtxTrain());
        assertEquals(4096, meta.getNEmbd());
        assertEquals(6738546688L, meta.getNParams());
        assertEquals(2825274880L, meta.getSize());
    }

    @Test
    public void testModalityGetters() throws Exception {
        ModelMeta textOnly = parse("{\"vocab_type\":1,\"n_vocab\":100,\"n_ctx_train\":4096," +
                "\"n_embd\":512,\"n_params\":1000000,\"size\":500000," +
                "\"modalities\":{\"vision\":false,\"audio\":false}," +
                "\"architecture\":\"llama\",\"name\":\"\"}");
        assertFalse(textOnly.supportsVision());
        assertFalse(textOnly.supportsAudio());

        ModelMeta multimodal = parse("{\"vocab_type\":1,\"n_vocab\":100,\"n_ctx_train\":4096," +
                "\"n_embd\":512,\"n_params\":1000000,\"size\":500000," +
                "\"modalities\":{\"vision\":true,\"audio\":true}," +
                "\"architecture\":\"gemma3\",\"name\":\"Gemma-3\"}");
        assertTrue(multimodal.supportsVision());
        assertTrue(multimodal.supportsAudio());
    }

    @Test
    public void testGetArchitecture() throws Exception {
        ModelMeta meta = parse("{\"vocab_type\":1,\"n_vocab\":32016,\"n_ctx_train\":16384," +
                "\"n_embd\":4096,\"n_params\":6738546688,\"size\":2825274880," +
                "\"modalities\":{\"vision\":false,\"audio\":false}," +
                "\"architecture\":\"llama\",\"name\":\"CodeLlama-7B\"}");

        assertEquals("llama", meta.getArchitecture());
    }

    @Test
    public void testGetModelName() throws Exception {
        ModelMeta meta = parse("{\"vocab_type\":1,\"n_vocab\":32016,\"n_ctx_train\":16384," +
                "\"n_embd\":4096,\"n_params\":6738546688,\"size\":2825274880," +
                "\"modalities\":{\"vision\":false,\"audio\":false}," +
                "\"architecture\":\"mistral\",\"name\":\"Mistral-7B-v0.1\"}");

        assertEquals("Mistral-7B-v0.1", meta.getModelName());
    }

    @Test
    public void testGetArchitectureEmptyWhenAbsent() throws Exception {
        ModelMeta meta = parse("{\"vocab_type\":1,\"n_vocab\":100,\"n_ctx_train\":4096," +
                "\"n_embd\":512,\"n_params\":1000000,\"size\":500000," +
                "\"modalities\":{\"vision\":false,\"audio\":false}}");

        assertEquals("", meta.getArchitecture());
    }

    @Test
    public void testGetModelNameEmptyWhenAbsent() throws Exception {
        ModelMeta meta = parse("{\"vocab_type\":1,\"n_vocab\":100,\"n_ctx_train\":4096," +
                "\"n_embd\":512,\"n_params\":1000000,\"size\":500000," +
                "\"modalities\":{\"vision\":false,\"audio\":false}}");

        assertEquals("", meta.getModelName());
    }

    @Test
    public void testGetArchitectureVariousModels() throws Exception {
        for (String arch : new String[]{"llama", "gemma3", "mistral", "falcon", "phi3"}) {
            ModelMeta meta = parse("{\"vocab_type\":1,\"n_vocab\":100,\"n_ctx_train\":4096," +
                    "\"n_embd\":512,\"n_params\":1000000,\"size\":500000," +
                    "\"modalities\":{\"vision\":false,\"audio\":false}," +
                    "\"architecture\":\"" + arch + "\",\"name\":\"\"}");
            assertEquals(arch, meta.getArchitecture());
        }
    }

    @Test
    public void testToStringContainsNewFields() throws Exception {
        ModelMeta meta = parse("{\"vocab_type\":1,\"n_vocab\":32016,\"n_ctx_train\":16384," +
                "\"n_embd\":4096,\"n_params\":6738546688,\"size\":2825274880," +
                "\"modalities\":{\"vision\":false,\"audio\":false}," +
                "\"architecture\":\"llama\",\"name\":\"CodeLlama-7B\"}");

        String json = meta.toString();
        assertTrue(json.contains("\"architecture\""));
        assertTrue(json.contains("\"name\""));
        assertTrue(json.contains("\"llama\""));
        assertTrue(json.contains("\"CodeLlama-7B\""));
    }
}
