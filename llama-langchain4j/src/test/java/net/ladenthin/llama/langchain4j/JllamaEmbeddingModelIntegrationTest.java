// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.langchain4j;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.greaterThan;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.notNullValue;

import dev.langchain4j.data.embedding.Embedding;
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
 * End-to-end smoke test for {@link JllamaEmbeddingModel} over a real embedding model. Self-skips
 * unless {@code -Dnet.ladenthin.llama.langchain4j.embedding.model=/abs/path/to/embedding.gguf}
 * points at a GGUF loadable in embedding mode (and the native library is present), mirroring the
 * core project's model-gated tests. CI reuses the already-cached nomic embedding model, so no extra
 * download is introduced.
 */
class JllamaEmbeddingModelIntegrationTest {

    private static Path modelPath() {
        Path resolved = TestModelPaths.fromProperty("net.ladenthin.llama.langchain4j.embedding.model");
        Assumptions.assumeTrue(resolved != null, "embedding model path property not set");
        Assumptions.assumeTrue(Files.exists(resolved), "embedding model file not present: " + resolved);
        return resolved;
    }

    @Test
    void embedsAllSegmentsInInputOrder() {
        Path model = modelPath();
        try (LlamaModel llama =
                new LlamaModel(new ModelParameters().setModel(model.toString()).enableEmbedding())) {
            JllamaEmbeddingModel embeddingModel = new JllamaEmbeddingModel(llama);

            List<TextSegment> segments =
                    Arrays.asList(TextSegment.from("hello world"), TextSegment.from("goodbye world"));
            Response<List<Embedding>> response = embeddingModel.embedAll(segments);

            assertThat(response, is(notNullValue()));
            assertThat(response.content().size(), is(2));
            assertThat(response.content().get(0).vector().length, is(greaterThan(0)));
        }
    }
}
