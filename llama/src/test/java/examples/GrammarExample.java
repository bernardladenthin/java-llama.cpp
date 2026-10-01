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
 * Output the model cannot get wrong in form. A GBNF grammar restricts every generated token to what the
 * grammar allows -- here one of two words. A JSON schema does the same for structured data: it is turned
 * into such a grammar internally, and {@link LlamaModel#completeAsJson(Class, String, InferenceParameters)}
 * binds the reply straight to a Java object.
 *
 * <p>Pass a GGUF file as the first argument; without one {@code models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf} of
 * the test set is used (see {@code .github/models.csv}).
 */
public final class GrammarExample {

    /** Accepts exactly {@code yes} or {@code no}. */
    private static final String YES_OR_NO = "root ::= \"yes\" | \"no\"";

    /** The shape of {@link City}: every field required, nothing else allowed. */
    private static final String CITY_SCHEMA = "{\"type\":\"object\",\"properties\":{"
            + "\"name\":{\"type\":\"string\"},"
            + "\"country\":{\"type\":\"string\"},"
            + "\"population\":{\"type\":\"integer\"}},"
            + "\"required\":[\"name\",\"country\",\"population\"],"
            + "\"additionalProperties\":false}";

    /** What the model fills in; public fields, so Jackson can bind the JSON to them. */
    public static final class City {
        /** The city's name. */
        public String name = "";
        /** The country it lies in. */
        public String country = "";
        /** Its population, as far as the model knows it. */
        public long population;
    }

    private GrammarExample() {}

    public static void main(String... args) {
        String modelPath = args.length > 0 ? args[0] : TestConstants.DEFAULT_TOOL_MODEL_PATH;
        run(new ModelParameters().setModel(modelPath).setCtxSize(2048), System.out);
    }

    /** The example itself; the model parameters and the console come in from outside, so a test can run it. */
    static void run(ModelParameters modelParameters, PrintStream out) {
        try (LlamaModel model = new LlamaModel(modelParameters)) {
            String answer = model.complete(
                    InferenceParameters.of("Question: Is Paris the capital of France?\nAnswer (yes or no): ")
                            .withGrammar(YES_OR_NO));
            out.println("Is Paris the capital of France? " + answer);

            City city = model.completeAsJson(
                    City.class,
                    CITY_SCHEMA,
                    InferenceParameters.of("The largest city of Japan, as JSON with name, country and population: ")
                            .withNPredict(128)
                            .withTemperature(0.0f));
            out.printf("%s, %s: about %,d inhabitants%n", city.name, city.country, city.population);
        }
    }
}
