// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package examples;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.endsWith;
import static org.hamcrest.Matchers.matchesPattern;
import static org.hamcrest.Matchers.startsWith;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.PrintStream;
import java.io.StringReader;
import java.nio.charset.StandardCharsets;
import net.ladenthin.llama.ClaudeGenerated;
import net.ladenthin.llama.TestConstants;
import net.ladenthin.llama.parameters.ModelParameters;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

/**
 * Runs every example against the model it names by default, so an example that no longer compiles against
 * the API -- or no longer works with it -- fails here instead of in a reader's hands. Each test self-skips
 * while its GGUF is missing; CI downloads all of them (.github/models.csv).
 */
@ClaudeGenerated(purpose = "Keep the runnable examples working against the current API.")
public class ExamplesTest {

    private static ModelParameters model(String path) {
        Assumptions.assumeTrue(new File(path).exists(), "Model file not found: " + path);
        int gpuLayers = Integer.getInteger(TestConstants.PROP_TEST_NGL, TestConstants.DEFAULT_TEST_NGL);
        return new ModelParameters().setModel(path).setCtxSize(2048).setGpuLayers(gpuLayers);
    }

    private static PrintStream capture(ByteArrayOutputStream bytes) {
        return new PrintStream(bytes, true, StandardCharsets.UTF_8);
    }

    @Test
    public void mainExampleCompletesAndStreams() {
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        MainExample.run(model(TestConstants.DRAFT_MODEL_PATH), capture(bytes));
        String out = bytes.toString(StandardCharsets.UTF_8);
        assertThat(out, startsWith("// Returns the n-th Fibonacci number."));
        assertThat(out, containsString("generated tokens"));
        assertThat(out, containsString("static boolean isLeapYear(int year) {"));
        assertThat(out, containsString("-- stopped: "));
    }

    @Test
    public void grammarExampleAnswersWithinTheGrammarAndBindsTheJson() {
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        GrammarExample.run(model(TestConstants.DEFAULT_TOOL_MODEL_PATH), capture(bytes));
        String out = bytes.toString(StandardCharsets.UTF_8);
        assertThat(out, matchesPattern("(?s)Is Paris the capital of France\\? (yes|no)\\R.*"));
        // the second line exists only when the reply parsed into a City
        assertThat(out, matchesPattern("(?s).*\\n.+, .+: about [0-9,.\\s\\u00a0]+ inhabitants\\R"));
    }

    @Test
    public void infillExampleKeepsPrefixAndSuffix() {
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        InfillExample.run(model(TestConstants.MODEL_PATH), capture(bytes));
        String out = bytes.toString(StandardCharsets.UTF_8);
        assertThat(out, startsWith("/** Returns whether the text reads the same backwards"));
        assertThat(out, endsWith("\n}\n"));
    }

    @Test
    public void chatExampleRepliesAndEndsOnAnEmptyLine() throws IOException {
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        BufferedReader console = new BufferedReader(new StringReader("Say hello.\nAnd now goodbye.\n\n"));
        ChatExample.run(model(TestConstants.DEFAULT_TOOL_MODEL_PATH).setCtxSize(4096), console, capture(bytes));
        String out = bytes.toString(StandardCharsets.UTF_8);
        // two turns, each answered -- the second one only works if the first reply was committed
        assertThat(out, matchesPattern("(?s)\\nYou: Assistant: \\S.*\\nYou: Assistant: \\S.*\\nYou: "));
    }
}
