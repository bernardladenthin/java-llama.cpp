// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package examples;

import java.io.PrintStream;
import net.ladenthin.llama.LlamaModel;
import net.ladenthin.llama.TestConstants;
import net.ladenthin.llama.parameters.InferenceParameters;
import net.ladenthin.llama.parameters.ModelParameters;
import net.ladenthin.llama.value.CompletionResult;
import net.ladenthin.llama.value.LlamaOutput;
import net.ladenthin.llama.value.StopReason;

/**
 * Plain text completion, the shortest way through the API: one blocking call that returns the text
 * together with what it cost, and the same kind of request streamed piece by piece as it is generated.
 *
 * <p>Pass a GGUF file as the first argument; without one the small code model of the test set is used
 * ({@code models/AMD-Llama-135m-code.Q2_K.gguf}, listed in {@code .github/models.csv}).
 */
public final class MainExample {

    private MainExample() {}

    public static void main(String... args) {
        String modelPath = args.length > 0 ? args[0] : TestConstants.DRAFT_MODEL_PATH;
        run(new ModelParameters().setModel(modelPath).setCtxSize(2048), System.out);
    }

    /** The example itself; the model parameters and the console come in from outside, so a test can run it. */
    static void run(ModelParameters modelParameters, PrintStream out) {
        try (LlamaModel model = new LlamaModel(modelParameters)) {
            // Blocking: the whole answer at once, with token counts and speed.
            String prompt = "// Returns the n-th Fibonacci number.\nstatic long fibonacci(int n) {\n";
            CompletionResult result = model.completeWithStats(InferenceParameters.of(prompt)
                    .withNPredict(96)
                    .withTemperature(0.2f)
                    .withStopStrings("\n}\n"));
            out.println(prompt + result.getText());
            out.printf(
                    "-- %d prompt + %d generated tokens, %.1f tokens/s, stopped: %s%n%n",
                    result.getUsage().getPromptTokens(),
                    result.getUsage().getCompletionTokens(),
                    result.getTimings().getPredictedPerSecond(),
                    result.getStopReason());

            // Streaming: every piece is printed the moment it is generated.
            String streamedPrompt =
                    "// Returns true if the given year is a leap year.\nstatic boolean isLeapYear(int year) {\n";
            out.print(streamedPrompt);
            StopReason stopReason = StopReason.NONE;
            for (LlamaOutput output : model.generate(InferenceParameters.of(streamedPrompt)
                    .withNPredict(96)
                    .withTemperature(0.2f)
                    .withStopStrings("\n}\n"))) {
                out.print(output.text);
                stopReason = output.stopReason;
            }
            out.printf("%n-- stopped: %s%n", stopReason);
        }
    }
}
