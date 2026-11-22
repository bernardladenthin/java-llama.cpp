// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.TimeUnit;
import net.ladenthin.llama.loader.LlamaSystemProperties;
import net.ladenthin.llama.parameters.InferenceParameters;
import net.ladenthin.llama.parameters.ModelParameters;
import net.ladenthin.llama.value.Pair;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

/**
 * A JVM started in a German environment ({@code LC_ALL=de_DE.UTF-8}) takes its C numeric locale from it,
 * and llama.cpp reads and writes numbers through that locale: the argv's (until 2026-11-12, "--temp 0.7"
 * read as 0) and a chat template's (until 2026-11-19, {@code {{ 0.5 }}} rendered as "0,"). The library
 * runs llama.cpp under the "C" numeric locale of its own threads. This test starts such a JVM -- the case
 * the C API tests ({@code CApiLocale}) can only imitate with {@code setlocale} -- loads CI's draft model
 * with a chat template full of numbers and a temperature, and reads both back.
 */
@ClaudeGenerated(
        purpose = "Pin, in a JVM started under a decimal-comma locale, that a chat template's numbers and the "
                + "argv's are read and written with a decimal point.")
public class DecimalCommaLocaleIntegrationTest {

    static final String TEMPLATE = "[{{ 0.5 }}|{{ '0.25' | float }}|{{ 1.5 * 2 }}]{{ messages[0].content }}";

    private static boolean germanLocaleInstalled() throws IOException, InterruptedException {
        if (!System.getProperty("os.name", "").toLowerCase(Locale.ROOT).contains("linux")) {
            return false;
        }
        Process locales =
                new ProcessBuilder("locale", "-a").redirectErrorStream(true).start();
        String listed = read(locales.getInputStream());
        locales.waitFor(30, TimeUnit.SECONDS);
        String lower = listed.toLowerCase(Locale.ROOT);
        return lower.contains("de_de.utf8") || lower.contains("de_de.utf-8");
    }

    @Test
    public void aJvmUnderADecimalCommaLocaleReadsAndWritesTheNumbersWithAPoint() throws Exception {
        Assumptions.assumeTrue(
                new File(TestConstants.DRAFT_MODEL_PATH).exists(),
                "Draft model not found, skipping DecimalCommaLocaleIntegrationTest");
        boolean installed = germanLocaleInstalled();
        if (!installed && System.getenv("JLLAMA_REQUIRE_DECIMAL_COMMA_LOCALE") != null) {
            throw new AssertionError("JLLAMA_REQUIRE_DECIMAL_COMMA_LOCALE is set, and de_DE.UTF-8 is not installed");
        }
        Assumptions.assumeTrue(installed, "de_DE.UTF-8 is not installed (locale-gen de_DE.UTF-8)");

        List<String> command = new ArrayList<>();
        command.add(new File(System.getProperty("java.home"), "bin/java").getPath());
        command.add("-cp");
        command.add(System.getProperty("java.class.path"));
        // the library's own properties (a lib.path, the test's GPU layers) reach the child as they are
        java.util.Properties properties = System.getProperties();
        for (String name : properties.stringPropertyNames()) {
            if (name.startsWith(LlamaSystemProperties.PREFIX + ".")) {
                command.add("-D" + name + "=" + properties.getProperty(name));
            }
        }
        command.add(Child.class.getName());
        command.add(new File(TestConstants.DRAFT_MODEL_PATH).getAbsolutePath());
        ProcessBuilder builder = new ProcessBuilder(command).redirectErrorStream(true);
        builder.environment().put("LC_ALL", "de_DE.UTF-8");
        builder.environment().put("LANG", "de_DE.UTF-8");
        Process child = builder.start();
        String output = read(child.getInputStream());
        assertTrue(child.waitFor(5, TimeUnit.MINUTES), "the child JVM did not finish:\n" + output);
        assertEquals(0, child.exitValue(), "the child JVM failed:\n" + output);
        assertTrue(output.contains("rendered=[0.5|0.25|3.0]Hi"), output);
        assertTrue(output.contains("temperature=0.25"), output);
    }

    /** What the child JVM runs: load the model with the template, print what it renders and reads. */
    public static final class Child {
        private Child() {}

        public static void main(String[] args) {
            ModelParameters parameters = new ModelParameters()
                    .setModel(args[0])
                    .setCtxSize(256)
                    .setGpuLayers(0)
                    .setFit(false)
                    .enableJinja()
                    .setChatTemplate(TEMPLATE)
                    .setTemp(0.25f);
            try (LlamaModel model = new LlamaModel(parameters)) {
                String rendered = model.applyTemplate(new InferenceParameters("")
                        .withMessages(null, Collections.singletonList(new Pair<>("user", "Hi"))));
                System.out.println("rendered=" + rendered);
                String completion = model.handleCompletions("{\"prompt\":\"Hi\",\"n_predict\":1}");
                int at = completion.indexOf("\"temperature\":");
                System.out.println("temperature=" + completion.substring(at + 14, at + 18));
            }
        }
    }

    private static String read(InputStream in) throws IOException {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        byte[] buffer = new byte[8192];
        for (int n; (n = in.read(buffer)) > 0; ) {
            out.write(buffer, 0, n);
        }
        return new String(out.toByteArray(), StandardCharsets.UTF_8);
    }
}
