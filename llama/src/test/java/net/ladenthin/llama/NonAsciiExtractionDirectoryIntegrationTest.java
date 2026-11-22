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
import java.net.URL;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import net.ladenthin.llama.loader.LlamaSystemProperties;
import net.ladenthin.llama.loader.OSInfo;
import net.ladenthin.llama.parameters.InferenceParameters;
import net.ladenthin.llama.parameters.ModelParameters;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * The natives extracted into a directory whose name is outside ASCII. The loader extracts the library
 * directory of the classpath's natives into {@code net.ladenthin.llama.tmpdir}; on Windows the library
 * then finds its own directory with {@code GetModuleFileNameW} and hands it to ggml as UTF-8 to load the
 * CPU (and GPU) modules from it -- handed over in the ANSI code page, a path with an umlaut loaded no
 * module and logged no error, and the model had no backend. The load happens once per JVM, before any
 * other test of a fork could choose the directory, so the case runs in a child JVM: started with this
 * fork's classpath, without {@code lib.path}, it extracts and loads from the directory it is given and
 * answers a completion with CI's draft model.
 *
 * <p>The name holds {@code ä}, which is in code page 1252, the Windows runner's: that is the case the
 * native side ({@code GetModuleFileNameW}, then UTF-8 to ggml) has to handle. A character outside the
 * code page, such as {@code 日}, is deliberately not tested: the JVM hands {@link System#load} its path
 * in {@code sun.jnu.encoding}, so on code page 1252 a directory {@code tmp-日} reaches the OS as
 * {@code tmp-?} and cannot be loaded from, although Java extracts the files into it -- a measured limit
 * of the JVM, not of this library (TODO.md), which the loader's error names together with
 * {@code net.ladenthin.llama.tmpdir}. On Linux and macOS (UTF-8) the test passes trivially.
 *
 * <p>The child reads the directory from a UTF-8 file rather than from its command line, which the
 * Windows launcher reads in the ANSI code page. When this fork runs with {@code lib.path}, that
 * directory is put on the child's classpath in the natives jar layout, so the child extracts exactly
 * those natives; otherwise the child extracts what the classpath carries ({@code llama/src/main/natives},
 * which surefire adds, and which CI's Java test jobs fill with the {@code natives-<classifier>} artifact).
 */
@ClaudeGenerated(
        purpose = "Pin, in a child JVM, that the classpath natives extracted into a directory named outside "
                + "ASCII (an umlaut, inside Windows code page 1252) load their CPU modules and answer a completion.")
public class NonAsciiExtractionDirectoryIntegrationTest {

    /** The resource directory of this platform's natives, as the loader looks it up. */
    private static final String NATIVE_RESOURCE_PATH =
            "net/ladenthin/llama/" + OSInfo.getNativeLibFolderPathForCurrentOS();

    private static final String LIBRARY = System.mapLibraryName("jllama");

    private static final String LIB_PATH_PROPERTY = LlamaSystemProperties.PREFIX + ".lib.path";

    private static final String TMPDIR_PROPERTY = LlamaSystemProperties.PREFIX + ".tmpdir";

    /** The extraction directory's name: an umlaut, which code page 1252 holds (see the class comment). */
    private static final String EXTRACTION_NAME = "natives-\u00e4";

    @TempDir
    Path tempDir;

    @Test
    public void anExtractionDirectoryWithAnUmlautLoadsTheModulesAndAnswers() throws Exception {
        ChildResult result = runChild(EXTRACTION_NAME);
        Path extraction = tempDir.resolve(EXTRACTION_NAME);

        assertEquals(0, result.exitCode, "the child JVM failed:\n" + result.output);
        assertTrue(result.output.contains("answered="), result.output);
        assertExtractedInto(extraction);
    }

    /** One extraction directory of the build below {@code extraction}, holding the library. */
    private static void assertExtractedInto(Path extraction) throws IOException {
        List<Path> dirs;
        try (Stream<Path> entries = Files.list(extraction)) {
            dirs = entries.filter(p -> p.getFileName().toString().startsWith("jllama-backend-"))
                    .collect(Collectors.toList());
        }
        assertEquals(1, dirs.size(), "extraction directories below " + extraction + ": " + dirs);
        assertTrue(Files.isRegularFile(dirs.get(0).resolve(LIBRARY)), "no " + LIBRARY + " in " + dirs.get(0));
    }

    /** What the child JVM printed and how it exited. */
    private static final class ChildResult {
        final int exitCode;
        final String output;

        ChildResult(int exitCode, String output) {
            this.exitCode = exitCode;
            this.output = output;
        }
    }

    private ChildResult runChild(String extractionName) throws Exception {
        Assumptions.assumeTrue(
                new File(TestConstants.DRAFT_MODEL_PATH).exists(),
                "Draft model not found, skipping NonAsciiExtractionDirectoryIntegrationTest");
        // a JVM whose platform encoding cannot name the directory cannot create it either (a POSIX
        // locale, LANG unset: ASCII only) -- that is the JVM, not the loader, so there is nothing to test
        String encoding = System.getProperty("sun.jnu.encoding");
        Assumptions.assumeTrue(
                encoding == null || Charset.forName(encoding).newEncoder().canEncode(extractionName),
                "the platform encoding " + encoding + " cannot name " + extractionName
                        + " (run with a UTF-8 locale, e.g. LC_ALL=C.UTF-8)");
        Path extraction = tempDir.resolve(extractionName);
        String classpath = childClasspath();

        Path directoryFile = tempDir.resolve("extraction-" + System.nanoTime() + ".txt");
        Files.write(directoryFile, extraction.toAbsolutePath().toString().getBytes(StandardCharsets.UTF_8));

        List<String> command = new ArrayList<>();
        command.add(new File(System.getProperty("java.home"), "bin/java").getPath());
        command.add("-cp");
        command.add(classpath);
        // the library's own properties reach the child, but not the two this test decides
        Properties properties = System.getProperties();
        for (String name : properties.stringPropertyNames()) {
            if (name.startsWith(LlamaSystemProperties.PREFIX + ".")
                    && !name.equals(LIB_PATH_PROPERTY)
                    && !name.equals(TMPDIR_PROPERTY)) {
                command.add("-D" + name + "=" + properties.getProperty(name));
            }
        }
        command.add(Child.class.getName());
        command.add(new File(TestConstants.DRAFT_MODEL_PATH).getAbsolutePath());
        command.add(directoryFile.toAbsolutePath().toString());
        Process child = new ProcessBuilder(command).redirectErrorStream(true).start();
        String output = read(child.getInputStream());
        assertTrue(child.waitFor(5, TimeUnit.MINUTES), "the child JVM did not finish:\n" + output);
        return new ChildResult(child.exitValue(), output);
    }

    /**
     * This fork's classpath for the child, with the natives the fork was told to use first: the
     * {@code lib.path} directory in the natives jar layout when it is set, else the classpath's own
     * natives, and a skip when there are none.
     */
    private String childClasspath() throws IOException {
        String classpath = System.getProperty("java.class.path");
        String libPath = System.getProperty(LIB_PATH_PROPERTY);
        if (libPath != null && !libPath.isEmpty()) {
            Path source = Paths.get(libPath);
            Assumptions.assumeTrue(
                    Files.isRegularFile(source.resolve(LIBRARY)), "no " + LIBRARY + " in lib.path " + source);
            Path root = tempDir.resolve("natives-classpath");
            Path target = root.resolve(NATIVE_RESOURCE_PATH).resolve(backendOf(source));
            Files.createDirectories(target);
            try (Stream<Path> files = Files.list(source)) {
                for (Path file : files.filter(Files::isRegularFile).collect(Collectors.toList())) {
                    Files.copy(file, target.resolve(file.getFileName().toString()));
                }
            }
            return root.toAbsolutePath() + File.pathSeparator + classpath;
        }
        ClassLoader loader = NonAsciiExtractionDirectoryIntegrationTest.class.getClassLoader();
        URL metal = loader.getResource(NATIVE_RESOURCE_PATH + "/metal/" + LIBRARY);
        URL cpu = loader.getResource(NATIVE_RESOURCE_PATH + "/cpu/" + LIBRARY);
        Assumptions.assumeTrue(
                metal != null || cpu != null,
                "no natives of " + NATIVE_RESOURCE_PATH + " on the classpath: build them into llama/src/main/natives"
                        + " (cmake), or pass -D" + LIB_PATH_PROPERTY);
        return classpath;
    }

    /** The {@code backend=} of a natives directory's {@code jllama-build.txt}, {@code cpu} without one. */
    private static String backendOf(Path nativesDirectory) throws IOException {
        Path build = nativesDirectory.resolve("jllama-build.txt");
        if (Files.isRegularFile(build)) {
            for (String line : Files.readAllLines(build, StandardCharsets.UTF_8)) {
                if (line.startsWith("backend=")) {
                    return line.substring("backend=".length()).trim();
                }
            }
        }
        return "cpu";
    }

    /**
     * What the child JVM runs: take the extraction directory from the file named by the second
     * argument, then load the model from the first and print the length of a short answer.
     */
    public static final class Child {
        private Child() {}

        public static void main(String[] args) throws IOException {
            String extraction = new String(Files.readAllBytes(Paths.get(args[1])), StandardCharsets.UTF_8);
            System.setProperty(TMPDIR_PROPERTY, extraction);
            ModelParameters parameters = new ModelParameters()
                    .setModel(args[0])
                    .setCtxSize(128)
                    .setGpuLayers(Integer.getInteger(TestConstants.PROP_TEST_NGL, 0))
                    .setFit(false);
            try (LlamaModel model = new LlamaModel(parameters)) {
                String answer = model.complete(new InferenceParameters("def main():").withNPredict(4));
                System.out.println("answered=" + answer.length());
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
