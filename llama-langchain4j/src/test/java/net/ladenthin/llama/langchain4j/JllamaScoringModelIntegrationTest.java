// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.langchain4j;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.notNullValue;

import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.output.Response;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import net.ladenthin.llama.LlamaModel;
import net.ladenthin.llama.parameters.ModelParameters;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

/**
 * End-to-end smoke test for {@link JllamaScoringModel} (re-ranker) over a real reranking model.
 * Self-skips unless {@code -Dnet.ladenthin.llama.langchain4j.rerank.model=/abs/path/to/reranker.gguf}
 * points at a GGUF loadable in reranking mode (and the native library is present), mirroring the
 * core project's model-gated tests. CI reuses the already-cached jina reranker model, so no extra
 * download is introduced.
 */
class JllamaScoringModelIntegrationTest {

    private static Path modelPath() {
        Path resolved = TestModelPaths.fromProperty("net.ladenthin.llama.langchain4j.rerank.model");
        Assumptions.assumeTrue(resolved != null, "rerank model path property not set");
        Assumptions.assumeTrue(Files.exists(resolved), "rerank model file not present: " + resolved);
        return resolved;
    }

    @Test
    void scoresEverySegmentInInputOrder() {
        Path model = modelPath();
        try (LlamaModel llama =
                new LlamaModel(new ModelParameters().setModel(model.toString()).enableReranking())) {
            JllamaScoringModel scoringModel = new JllamaScoringModel(llama);

            List<TextSegment> segments =
                    Arrays.asList(
                            TextSegment.from("A cat sat on the mat."),
                            TextSegment.from("The stock market fell today."));
            Response<List<Double>> response = scoringModel.scoreAll(segments, "domestic pets");

            assertThat(response, is(notNullValue()));
            assertThat(response.content().size(), is(2));
        }
    }
}
