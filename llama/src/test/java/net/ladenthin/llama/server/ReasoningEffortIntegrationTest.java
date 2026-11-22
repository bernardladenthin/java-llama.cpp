// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.hasItem;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.not;
import static org.hamcrest.Matchers.notNullValue;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.File;
import java.util.Collections;
import net.ladenthin.llama.ClaudeGenerated;
import net.ladenthin.llama.LlamaModel;
import net.ladenthin.llama.TestConstants;
import net.ladenthin.llama.exception.InvalidRequestException;
import net.ladenthin.llama.parameters.InferenceParameters;
import net.ladenthin.llama.parameters.ModelParameters;
import net.ladenthin.llama.value.Pair;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

/**
 * {@code reasoning_effort} reaches llama.cpp's chat parser as the field itself, from
 * {@link InferenceParameters#withReasoningEffort(String)} and from {@code OpenAiCompatServer}'s chat
 * mapping alike, so the parser's rules apply: {@code "none"} reaches a template that renders it as a level
 * of its own (b12624), and a level the template raises on is refused naming the levels it supports
 * (b12431). Until 2026-11-22 the server's mapping translated the field into {@code chat_template_kwargs},
 * the rule before both, and {@code LlamaModel} had no way to set it. CI's draft model with a template
 * that prints the level and raises on {@code max}.
 */
@ClaudeGenerated(
        purpose = "Pin that reasoning_effort reaches llama.cpp's chat parser as the field itself, from the "
                + "builder and from OpenAiCompatServer's mapping: none renders, max is refused by name.")
public class ReasoningEffortIntegrationTest {

    private static final ObjectMapper JSON = new ObjectMapper();

    /** Prints the level (or "unset"), raises on {@code max}. */
    static final String TEMPLATE = "{%- if reasoning_effort is defined and reasoning_effort == 'max' -%}"
            + "{{ raise_exception('max is not supported') }}{%- endif -%}"
            + "{%- for message in messages -%}<{{ message.role }}>{{ message.content }}{%- endfor -%}"
            + "[{{ reasoning_effort if reasoning_effort is defined else 'unset' }}]"
            + "{%- if add_generation_prompt -%}<assistant>{%- endif -%}";

    private static InferenceParameters hi() {
        return new InferenceParameters("").withMessages(null, Collections.singletonList(new Pair<>("user", "Hi")));
    }

    @Test
    public void reasoningEffortReachesTheChatParserFromTheBuilderAndTheServerMapping() throws Exception {
        Assumptions.assumeTrue(
                new File(TestConstants.DRAFT_MODEL_PATH).exists(),
                "Draft model not found, skipping ReasoningEffortIntegrationTest");
        ModelParameters parameters = new ModelParameters()
                .setModel(TestConstants.DRAFT_MODEL_PATH)
                .setCtxSize(256)
                .setGpuLayers(Integer.getInteger(TestConstants.PROP_TEST_NGL, TestConstants.DEFAULT_TEST_NGL))
                .setFit(false)
                .enableJinja()
                .setChatTemplate(TEMPLATE);
        try (LlamaModel model = new LlamaModel(parameters)) {
            assertThat(model.getModelMeta().getReasoningEffortLevels(), hasItem("none"));
            assertThat(model.getModelMeta().getReasoningEffortLevels(), not(hasItem("max")));

            assertThat(model.applyTemplate(hi()), containsString("[unset]"));
            assertThat(model.applyTemplate(hi().withReasoningEffort("none")), containsString("[none]"));
            assertThat(model.applyTemplate(hi().withReasoningEffort("high")), containsString("[high]"));

            // the server's mapping of an OpenAI chat request takes the same way
            OpenAiRequestMapper mapper = new OpenAiRequestMapper();
            InferenceParameters mapped = mapper.toInferenceParameters(JSON.readTree(
                    "{\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"reasoning_effort\":\"none\"}"));
            assertThat(model.applyTemplate(mapped), containsString("[none]"));

            InferenceParameters max = mapper.toInferenceParameters(
                    JSON.readTree("{\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"reasoning_effort\":\"max\","
                            + "\"max_tokens\":2}"));
            InvalidRequestException refused =
                    assertThrows(InvalidRequestException.class, () -> model.chatComplete(max));
            assertThat(refused.getMessage(), containsString("supported levels"));
            assertThat(refused.getMessage(), containsString("high"));
            // and the model answers the next request, with a level it renders
            assertThat(model.chatComplete(hi().withReasoningEffort("high").withNPredict(2)), is(notNullValue()));
        }
    }
}
