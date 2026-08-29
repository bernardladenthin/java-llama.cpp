// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
// SPDX-FileCopyrightText: 2023-2025 Konstantin Herud
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.value;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.is;

import com.fasterxml.jackson.databind.ObjectMapper;
import net.ladenthin.llama.ClaudeGenerated;
import org.junit.jupiter.api.Test;

/**
 * Unit tests for {@link ModelMeta} typed getters.
 * Constructs {@code ModelMeta} directly from JSON strings — no native library or model file required.
 */
@ClaudeGenerated(
        purpose = "Verify that ModelMeta typed getters map correctly from the underlying JsonNode, "
                + "including the new architecture and name fields from GGUF general.* metadata.")
public class ModelMetaTest {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    private ModelMeta parse(String json) throws Exception {
        return new ModelMeta(MAPPER.readTree(json));
    }

    @Test
    public void testNumericGetters() throws Exception {
        ModelMeta meta = parse("{\"vocab_type\":1,\"n_vocab\":32016,\"n_ctx_train\":16384,"
                + "\"n_embd\":4096,\"n_params\":6738546688,\"size\":2825274880,"
                + "\"modalities\":{\"vision\":false,\"audio\":false},"
                + "\"architecture\":\"llama\",\"name\":\"CodeLlama-7B\"}");

        assertThat(meta.getVocabType(), is(1));
        assertThat(meta.getNVocab(), is(32016));
        assertThat(meta.getNCtxTrain(), is(16384));
        assertThat(meta.getNEmbd(), is(4096));
        assertThat(meta.getNParams(), is(6738546688L));
        assertThat(meta.getSize(), is(2825274880L));
    }

    @Test
    public void testModalityGetters() throws Exception {
        ModelMeta textOnly = parse("{\"vocab_type\":1,\"n_vocab\":100,\"n_ctx_train\":4096,"
                + "\"n_embd\":512,\"n_params\":1000000,\"size\":500000,"
                + "\"modalities\":{\"vision\":false,\"audio\":false,\"video\":false},"
                + "\"architecture\":\"llama\",\"name\":\"\"}");
        assertThat(textOnly.supportsVision(), is(false));
        assertThat(textOnly.supportsAudio(), is(false));
        assertThat(textOnly.supportsVideo(), is(false));

        ModelMeta multimodal = parse("{\"vocab_type\":1,\"n_vocab\":100,\"n_ctx_train\":4096,"
                + "\"n_embd\":512,\"n_params\":1000000,\"size\":500000,"
                + "\"modalities\":{\"vision\":true,\"audio\":true,\"video\":true},"
                + "\"architecture\":\"gemma3\",\"name\":\"Gemma-3\"}");
        assertThat(multimodal.supportsVision(), is(true));
        assertThat(multimodal.supportsAudio(), is(true));
        assertThat(multimodal.supportsVideo(), is(true));
    }

    @Test
    public void eachModalityGetterReadsItsOwnKey() throws Exception {
        // The other modality fixtures set audio and video to the same value throughout, so a getter
        // wired to the wrong sibling key still satisfies them. These three isolate one key each.
        assertThat(parse(modalities("true", "false", "false")).supportsVision(), is(true));
        assertThat(parse(modalities("true", "false", "false")).supportsAudio(), is(false));
        assertThat(parse(modalities("true", "false", "false")).supportsVideo(), is(false));

        assertThat(parse(modalities("false", "true", "false")).supportsVision(), is(false));
        assertThat(parse(modalities("false", "true", "false")).supportsAudio(), is(true));
        assertThat(parse(modalities("false", "true", "false")).supportsVideo(), is(false));

        assertThat(parse(modalities("false", "false", "true")).supportsVision(), is(false));
        assertThat(parse(modalities("false", "false", "true")).supportsAudio(), is(false));
        assertThat(parse(modalities("false", "false", "true")).supportsVideo(), is(true));
    }

    private static String modalities(String vision, String audio, String video) {
        return "{\"vocab_type\":1,\"n_vocab\":100,\"n_ctx_train\":4096,"
                + "\"n_embd\":512,\"n_params\":1000000,\"size\":500000,"
                + "\"modalities\":{\"vision\":" + vision + ",\"audio\":" + audio + ",\"video\":" + video + "},"
                + "\"architecture\":\"llama\",\"name\":\"\"}";
    }

    @Test
    public void modalityGettersDefaultToFalseOnOlderMetadata() throws Exception {
        // Metadata captured from a build that predates the `video` key: every accessor must answer
        // false rather than throw, so feature detection degrades instead of breaking.
        ModelMeta legacy = parse("{\"vocab_type\":1,\"n_vocab\":100,\"n_ctx_train\":4096,"
                + "\"n_embd\":512,\"n_params\":1000000,\"size\":500000,"
                + "\"modalities\":{\"vision\":true,\"audio\":false},"
                + "\"architecture\":\"llama\",\"name\":\"\"}");
        assertThat(legacy.supportsVision(), is(true));
        assertThat(legacy.supportsAudio(), is(false));
        assertThat(legacy.supportsVideo(), is(false));
    }

    @Test
    public void testGetArchitecture() throws Exception {
        ModelMeta meta = parse("{\"vocab_type\":1,\"n_vocab\":32016,\"n_ctx_train\":16384,"
                + "\"n_embd\":4096,\"n_params\":6738546688,\"size\":2825274880,"
                + "\"modalities\":{\"vision\":false,\"audio\":false},"
                + "\"architecture\":\"llama\",\"name\":\"CodeLlama-7B\"}");

        assertThat(meta.getArchitecture(), is("llama"));
    }

    @Test
    public void testGetModelName() throws Exception {
        ModelMeta meta = parse("{\"vocab_type\":1,\"n_vocab\":32016,\"n_ctx_train\":16384,"
                + "\"n_embd\":4096,\"n_params\":6738546688,\"size\":2825274880,"
                + "\"modalities\":{\"vision\":false,\"audio\":false},"
                + "\"architecture\":\"mistral\",\"name\":\"Mistral-7B-v0.1\"}");

        assertThat(meta.getModelName(), is("Mistral-7B-v0.1"));
    }

    @Test
    public void testGetFtype() throws Exception {
        ModelMeta meta = parse("{\"vocab_type\":1,\"n_vocab\":32016,\"n_ctx_train\":16384,"
                + "\"n_embd\":4096,\"n_params\":6738546688,\"size\":2825274880,"
                + "\"modalities\":{\"vision\":false,\"audio\":false},"
                + "\"architecture\":\"mistral\",\"name\":\"Mistral-7B-v0.1\",\"ftype\":\"Q4_K - Medium\"}");

        assertThat(meta.getFtype(), is("Q4_K - Medium"));
    }

    @Test
    public void testGetFtypeEmptyWhenAbsent() throws Exception {
        ModelMeta meta = parse("{\"vocab_type\":1,\"n_vocab\":100,\"n_ctx_train\":4096,"
                + "\"n_embd\":512,\"n_params\":1000000,\"size\":500000,"
                + "\"modalities\":{\"vision\":false,\"audio\":false}}");

        assertThat(meta.getFtype(), is(""));
    }

    @Test
    public void testGetArchitectureEmptyWhenAbsent() throws Exception {
        ModelMeta meta = parse("{\"vocab_type\":1,\"n_vocab\":100,\"n_ctx_train\":4096,"
                + "\"n_embd\":512,\"n_params\":1000000,\"size\":500000,"
                + "\"modalities\":{\"vision\":false,\"audio\":false}}");

        assertThat(meta.getArchitecture(), is(""));
    }

    @Test
    public void testGetModelNameEmptyWhenAbsent() throws Exception {
        ModelMeta meta = parse("{\"vocab_type\":1,\"n_vocab\":100,\"n_ctx_train\":4096,"
                + "\"n_embd\":512,\"n_params\":1000000,\"size\":500000,"
                + "\"modalities\":{\"vision\":false,\"audio\":false}}");

        assertThat(meta.getModelName(), is(""));
    }

    @Test
    public void testGetArchitectureVariousModels() throws Exception {
        for (String arch : new String[] {"llama", "gemma3", "mistral", "falcon", "phi3"}) {
            ModelMeta meta = parse("{\"vocab_type\":1,\"n_vocab\":100,\"n_ctx_train\":4096,"
                    + "\"n_embd\":512,\"n_params\":1000000,\"size\":500000,"
                    + "\"modalities\":{\"vision\":false,\"audio\":false},"
                    + "\"architecture\":\""
                    + arch + "\",\"name\":\"\"}");
            assertThat(meta.getArchitecture(), is(arch));
        }
    }

    @Test
    public void testAsJsonReturnsBackingNode() throws Exception {
        ModelMeta meta = parse("{\"vocab_type\":1,\"n_vocab\":32016,\"n_ctx_train\":16384,"
                + "\"n_embd\":4096,\"n_params\":6738546688,\"size\":2825274880,"
                + "\"modalities\":{\"vision\":false,\"audio\":false},"
                + "\"architecture\":\"llama\",\"name\":\"CodeLlama-7B\"}");
        // Dereferencing the returned node kills the "return null" mutant on asJson().
        assertThat(meta.asJson().get("architecture").asText(), is("llama"));
    }

    @Test
    public void testToStringContainsNewFields() throws Exception {
        ModelMeta meta = parse("{\"vocab_type\":1,\"n_vocab\":32016,\"n_ctx_train\":16384,"
                + "\"n_embd\":4096,\"n_params\":6738546688,\"size\":2825274880,"
                + "\"modalities\":{\"vision\":false,\"audio\":false},"
                + "\"architecture\":\"llama\",\"name\":\"CodeLlama-7B\"}");

        String json = meta.toString();
        assertThat(json, containsString("\"architecture\""));
        assertThat(json, containsString("\"name\""));
        assertThat(json, containsString("\"llama\""));
        assertThat(json, containsString("\"CodeLlama-7B\""));
    }

    @Test
    public void testChatTemplateSpecialTokensAndMetadata() throws Exception {
        ModelMeta meta = parse("{\"n_vocab\":32000,"
                + "\"chat_template\":\"{% for m in messages %}{{ m.content }}{% endfor %}\","
                + "\"special_tokens\":{\"bos\":1,\"eos\":2,\"eot\":32000,\"sep\":-1,\"nl\":13,\"pad\":-1},"
                + "\"metadata\":{\"general.architecture\":\"llama\",\"general.quantization_version\":\"2\"}}");

        assertThat(meta.getChatTemplate(), containsString("for m in messages"));
        assertThat(meta.getBosTokenId(), is(1));
        assertThat(meta.getEosTokenId(), is(2));
        assertThat(meta.getEotTokenId(), is(32000));
        assertThat(meta.getMetadata("general.architecture"), is("llama"));
        assertThat(meta.getMetadata("general.quantization_version"), is("2"));
    }

    @Test
    public void testNewGettersDefaultWhenAbsent() throws Exception {
        ModelMeta meta = parse("{\"n_vocab\":100}");

        assertThat(meta.getChatTemplate(), is(""));
        assertThat(meta.getBosTokenId(), is(-1));
        assertThat(meta.getEosTokenId(), is(-1));
        assertThat(meta.getEotTokenId(), is(-1));
        assertThat(meta.getMetadata("general.architecture"), is(""));
    }
}
