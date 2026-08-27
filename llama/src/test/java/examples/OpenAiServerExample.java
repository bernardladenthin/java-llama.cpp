// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
// SPDX-FileCopyrightText: 2023-2025 Konstantin Herud
//
// SPDX-License-Identifier: MIT

package examples;

import java.io.IOException;
import net.ladenthin.llama.LlamaModel;
import net.ladenthin.llama.TestConstants;
import net.ladenthin.llama.parameters.ModelParameters;
import net.ladenthin.llama.server.OpenAiCompatServer;
import net.ladenthin.llama.server.OpenAiServerConfig;
import org.junit.jupiter.api.Disabled;

// Runnable demo (no @Test): starts a local OpenAI-compatible HTTP endpoint over a GGUF model so an
// editor such as VS Code Copilot (Custom Endpoint) can drive it. Point the model path at a local
// GGUF via -Dnet.ladenthin.llama.server.model=... ; @Disabled keeps it out of `mvn test`.
@Disabled
public class OpenAiServerExample {

    public static void main(String... args) throws IOException, InterruptedException {
        String modelPath =
                TestConstants.resolveModelProperty("net.ladenthin.llama.server.model", TestConstants.MODEL_PATH);
        int port = Integer.getInteger("net.ladenthin.llama.server.port", 8080);

        // Two parallel slots let the editor's chat and its background title/summary requests run
        // concurrently instead of serializing behind one another.
        ModelParameters modelParams =
                new ModelParameters().setModel(modelPath).setCtxSize(8192).setParallel(2);

        OpenAiServerConfig config = OpenAiServerConfig.builder()
                .port(port)
                .modelId("local-model")
                .maxInputTokens(6144)
                .maxOutputTokens(2048)
                .build();

        try (LlamaModel model = new LlamaModel(modelParams);
                OpenAiCompatServer server = new OpenAiCompatServer(model, config).start()) {
            String url = "http://127.0.0.1:" + server.getPort() + OpenAiCompatServer.PATH_CHAT_COMPLETIONS;
            System.out.println("OpenAI-compatible endpoint ready: " + url);
            System.out.println("In VS Code: Chat: Manage Language Models -> Add Models -> Custom Endpoint ->");
            System.out.println("  API type 'Chat Completions', then set the model 'url' to: " + url);
            System.out.println("Press Ctrl+C to stop.");
            Thread.currentThread().join();
        }
    }
}
