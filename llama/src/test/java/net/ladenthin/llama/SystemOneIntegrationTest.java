// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.closeTo;
import static org.hamcrest.Matchers.contains;
import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.greaterThan;
import static org.hamcrest.Matchers.is;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.File;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import net.ladenthin.llama.exception.LlamaException;
import net.ladenthin.llama.loader.NativeLibraryPresence;
import net.ladenthin.llama.parameters.ModelParameters;
import org.junit.jupiter.api.Test;

/**
 * {@link LlamaModel#handleSystemOne(String)}, llama.cpp's {@code /v1/systemone} decision API (upstream
 * b11361). The JNI method forwards to upstream's own route handler, so what this pins is the bridge:
 * that the handler is reachable from a loaded model, that its answer comes back verbatim, and that an
 * error body turns into a {@link LlamaException} carrying upstream's message.
 *
 * <p>The rejection case runs with the cached draft model on every CI job. The answering cases need a
 * decision model, which is not in the CI set; they self-skip unless
 * {@link TestConstants#PROP_DECISION_MODEL_PATH} names one.</p>
 */
@ClaudeGenerated(
        purpose = "Pin the JNI bridge to llama.cpp's /v1/systemone handler: a non-decision model is "
                + "rejected with upstream's message, and a decision model answers every question type.")
public class SystemOneIntegrationTest {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    /** The request of upstream's own {@code test_systemone.py}: one question of each type. */
    private static final String REQUEST = "{"
            + "\"state\":\"I was charged twice for my order last week and nobody has replied.\","
            + "\"questions\":{"
            + "\"route\":{\"type\":\"choice\",\"instructions\":\"Which team should handle this?\","
            + "\"criteria\":{\"billing\":\"payments and refunds\",\"shipping\":null,\"technical\":null}},"
            + "\"urgency\":{\"type\":\"score\",\"instructions\":\"How urgent is this?\","
            + "\"criteria\":[\"can wait\",\"this week\",\"today\",\"right now\"]},"
            + "\"angry\":{\"type\":\"noul\",\"instructions\":\"Is the customer angry?\"}"
            + "}}";

    @Test
    public void aModelThatIsNotADecisionModelIsRejected() {
        assumeTrue(NativeLibraryPresence.onClasspath(), "libjllama not on classpath");
        assumeTrue(new File(TestConstants.DRAFT_MODEL_PATH).exists(), "draft model not found");

        try (LlamaModel model = new LlamaModel(new ModelParameters()
                .setModel(TestConstants.DRAFT_MODEL_PATH)
                .setCtxSize(256)
                .setGpuLayers(0)
                .setFit(false))) {
            LlamaException e = assertThrows(LlamaException.class, () -> model.handleSystemOne(REQUEST));
            assertThat(e.getMessage(), containsString("not a decision model"));
        }
    }

    @Test
    public void aDecisionModelAnswersEveryQuestionType() throws Exception {
        try (LlamaModel model = loadDecisionModel()) {
            JsonNode response = MAPPER.readTree(model.handleSystemOne(REQUEST));

            JsonNode answers = response.get("answers");
            assertThat(fieldNames(answers), contains("route", "urgency", "angry"));

            JsonNode route = answers.get("route");
            assertThat(route.get("type").asText(), is("choice"));
            assertThat(sum(route.get("probabilities")), closeTo(1.0, 1e-3));

            JsonNode urgency = answers.get("urgency");
            assertThat(urgency.get("type").asText(), is("score"));
            assertThat(sum(urgency.get("probabilities")), closeTo(1.0, 1e-3));
            assertThat(urgency.get("score").asDouble(), closeTo(1.5, 1.5));

            JsonNode angry = answers.get("angry");
            assertThat(angry.get("type").asText(), is("noul"));
            assertThat(angry.get("noul").asDouble(), closeTo(0.5, 0.5));

            assertThat(response.get("usage").get("input_tokens").asInt(), greaterThan(0));
            assertThat(response.get("usage").get("output_tokens").asInt(), is(0));
        }
    }

    @Test
    public void aDecisionModelRejectsAnUnknownQuestionType() throws Exception {
        try (LlamaModel model = loadDecisionModel()) {
            String request = "{\"state\":\"x\",\"questions\":{\"q\":{\"type\":\"bogus\",\"instructions\":\"?\"}}}";
            assertThrows(LlamaException.class, () -> model.handleSystemOne(request));
        }
    }

    private static LlamaModel loadDecisionModel() {
        assumeTrue(NativeLibraryPresence.onClasspath(), "libjllama not on classpath");
        String path = TestConstants.resolveModelProperty(TestConstants.PROP_DECISION_MODEL_PATH);
        assumeTrue(
                path != null && !path.isEmpty(),
                "decision model not set (-D" + TestConstants.PROP_DECISION_MODEL_PATH + "=...)");
        assumeTrue(new File(path).exists(), "decision model file missing: " + path);
        return new LlamaModel(new ModelParameters()
                .setModel(path)
                .setCtxSize(1024)
                .setGpuLayers(Integer.getInteger(TestConstants.PROP_TEST_NGL, TestConstants.DEFAULT_TEST_NGL))
                .setFit(false));
    }

    private static List<String> fieldNames(JsonNode node) {
        List<String> names = new ArrayList<>();
        for (Iterator<String> it = node.fieldNames(); it.hasNext(); ) {
            names.add(it.next());
        }
        return names;
    }

    private static double sum(JsonNode probabilities) {
        double total = 0;
        for (JsonNode value : probabilities) {
            total += value.asDouble();
        }
        return total;
    }
}
