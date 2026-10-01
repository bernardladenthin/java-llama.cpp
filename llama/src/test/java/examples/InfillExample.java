// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package examples;

import java.io.PrintStream;
import net.ladenthin.llama.LlamaModel;
import net.ladenthin.llama.TestConstants;
import net.ladenthin.llama.parameters.InferenceParameters;
import net.ladenthin.llama.parameters.ModelParameters;

/**
 * Fill-in-the-middle: given the code before and after a gap, a model trained for it writes what belongs in
 * between -- what an editor's code completion does at the cursor. Setting an input prefix and suffix is what
 * makes a completion request an infill request; there is no separate call.
 *
 * <p>Pass a GGUF file of an infill-capable model as the first argument; without one
 * {@code models/codellama-7b.Q2_K.gguf} of the test set is used (see {@code .github/models.csv}).
 */
public final class InfillExample {

    private InfillExample() {}

    public static void main(String... args) {
        String modelPath = args.length > 0 ? args[0] : TestConstants.MODEL_PATH;
        run(new ModelParameters().setModel(modelPath).setCtxSize(2048), System.out);
    }

    /** The example itself; the model parameters and the console come in from outside, so a test can run it. */
    static void run(ModelParameters modelParameters, PrintStream out) {
        String before = "/** Returns whether the text reads the same backwards, ignoring case. */\n"
                + "static boolean isPalindrome(String text) {\n";
        String after = "\n}\n";

        try (LlamaModel model = new LlamaModel(modelParameters)) {
            String middle = model.complete(InferenceParameters.of("")
                    .withInputPrefix(before)
                    .withInputSuffix(after)
                    .withNPredict(128)
                    .withTemperature(0.1f));
            out.print(before + middle + after);
        }
    }
}
