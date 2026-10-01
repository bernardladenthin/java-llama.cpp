// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package examples;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import net.ladenthin.llama.LlamaModel;
import net.ladenthin.llama.Session;
import net.ladenthin.llama.TestConstants;
import net.ladenthin.llama.parameters.ModelParameters;
import net.ladenthin.llama.value.LlamaOutput;

/**
 * An interactive multi-turn chat on the console. A {@link Session} keeps the conversation, so every turn
 * passes only the new message, and the model's chat template is applied for it; each reply is streamed as
 * it is generated. An empty line or the end of the input (Ctrl+D, on Windows Ctrl+Z) ends the chat.
 *
 * <p>Pass a GGUF file of an instruction-tuned model as the first argument; without one
 * {@code models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf} of the test set is used (see {@code .github/models.csv}).
 */
public final class ChatExample {

    private static final String SYSTEM_MESSAGE = "You are a concise, helpful assistant.";

    private ChatExample() {}

    public static void main(String... args) throws IOException {
        String modelPath = args.length > 0 ? args[0] : TestConstants.DEFAULT_TOOL_MODEL_PATH;
        BufferedReader console = new BufferedReader(new InputStreamReader(System.in, StandardCharsets.UTF_8));
        run(new ModelParameters().setModel(modelPath).setCtxSize(4096), console, System.out);
    }

    /** The example itself; the model parameters and the console come in from outside, so a test can run it. */
    static void run(ModelParameters modelParameters, BufferedReader console, PrintStream out) throws IOException {
        try (LlamaModel model = new LlamaModel(modelParameters);
                // Slot 0 holds this conversation; the customizer applies to every turn.
                Session session = new Session(
                        model, 0, SYSTEM_MESSAGE, p -> p.withNPredict(512).withTemperature(0.7f))) {
            while (true) {
                out.print("\nYou: ");
                String message = console.readLine();
                if (message == null || message.trim().isEmpty()) {
                    return;
                }
                out.print("Assistant: ");
                StringBuilder reply = new StringBuilder();
                try {
                    for (LlamaOutput output : session.stream(message)) {
                        out.print(output.text);
                        reply.append(output.text);
                    }
                } catch (RuntimeException e) {
                    // Without a committed reply the session would refuse every further turn.
                    session.cancelStream();
                    throw e;
                }
                session.commitStreamedReply(reply.toString());
                out.println();
            }
        }
    }
}
