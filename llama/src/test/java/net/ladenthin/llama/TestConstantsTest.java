// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.io.File;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

@ClaudeGenerated(
        purpose = "Pin TestConstants.resolveModelPath — the fix for model-gated tests silently "
                + "self-skipping in CI because Surefire's working directory is the llama/ module "
                + "while the shared GGUF cache is restored to the reactor root.")
public class TestConstantsTest {

    @Test
    public void nullAndEmptyPassThrough() {
        assertNull(TestConstants.resolveModelPath(null));
        assertEquals("", TestConstants.resolveModelPath(""));
    }

    @Test
    public void absolutePathIsReturnedUnchanged(@TempDir Path tempDir) throws Exception {
        Path file = Files.createFile(tempDir.resolve("model.gguf"));
        String absolute = file.toAbsolutePath().toString();
        assertEquals(absolute, TestConstants.resolveModelPath(absolute));
    }

    @Test
    public void existingRelativePathIsReturnedUnchanged() {
        // pom.xml exists relative to the module basedir, which is Surefire's working directory.
        assertTrue(new File("pom.xml").exists(), "precondition: the working directory is the module basedir");
        assertEquals("pom.xml", TestConstants.resolveModelPath("pom.xml"));
    }

    @Test
    public void pathOnlyPresentInTheParentResolvesToAnExistingAbsolutePath() {
        // The reactor root holds the aggregator pom; the module directory does not hold a
        // "../pom.xml"-shaped sibling of that name, so this is exactly the models/ situation.
        String resolved = TestConstants.resolveModelPath("llama/pom.xml");
        assertTrue(
                new File(resolved).exists(), "a path that only exists from the reactor root must resolve: " + resolved);
        assertTrue(Paths.get(resolved).isAbsolute(), "resolved parent-relative paths are absolute: " + resolved);
    }

    @Test
    public void unresolvablePathIsReturnedUnchangedSoSkipMessagesStayReadable() {
        String missing = "models/definitely-not-present-" + TestConstantsTest.class.getSimpleName() + ".gguf";
        assertEquals(missing, TestConstants.resolveModelPath(missing));
    }

    @Test
    public void propertyLookupResolvesAndHonoursTheDefault() {
        String key = "net.ladenthin.llama.test.resolver.probe";
        assertNull(System.getProperty(key), "precondition: probe property must be unset");
        assertNull(TestConstants.resolveModelProperty(key));
        assertEquals("pom.xml", TestConstants.resolveModelProperty(key, "pom.xml"));

        System.setProperty(key, "llama/pom.xml");
        try {
            assertTrue(new File(TestConstants.resolveModelProperty(key)).exists());
        } finally {
            System.clearProperty(key);
        }
    }

    @Test
    public void theShippedModelConstantsGoThroughTheResolver() {
        // Guards the wiring itself: if a future edit drops the resolveModelPath(...) wrapper from a
        // constant, that constant stops matching its resolved literal and every test gated on it
        // silently self-skips again in CI. Holds in both layouts — with the GGUF present the
        // resolver returns the same absolute path on both sides, without it the same literal.
        assertEquals(TestConstants.resolveModelPath("models/codellama-7b.Q2_K.gguf"), TestConstants.MODEL_PATH);
        assertEquals(
                TestConstants.resolveModelPath("models/AMD-Llama-135m-code.Q2_K.gguf"), TestConstants.DRAFT_MODEL_PATH);
        assertEquals(
                TestConstants.resolveModelPath("models/Qwen3-0.6B-Q4_K_M.gguf"), TestConstants.REASONING_MODEL_PATH);
        assertEquals(
                TestConstants.resolveModelPath("models/jina-reranker-v1-tiny-en-Q4_0.gguf"),
                TestConstants.RERANKING_MODEL_PATH);
        assertEquals(
                TestConstants.resolveModelPath("models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"),
                TestConstants.DEFAULT_TOOL_MODEL_PATH);
        assertEquals(
                TestConstants.resolveModelPath("src/test/resources/images/test-image.jpg"),
                TestConstants.DEFAULT_VISION_IMAGE_PATH);
        assertEquals(
                TestConstants.resolveModelPath("src/test/resources/audios/sample.wav"),
                TestConstants.DEFAULT_AUDIO_INPUT_PATH);
    }

    /**
     * The rule that actually prevents the suite from being re-muted.
     *
     * <p>Every model-gated class aborts in its {@code @BeforeAll} when its path does not resolve,
     * and a JUnit assumption failure is reported as <em>skipped</em> with the job still green — which
     * is how the entire model-backed suite went unexecuted on every CI platform for months.
     * {@link TestConstants#resolveModelPath(String)} accepts both the module-relative and the
     * reactor-root layout; reading a path property with a bare {@code System.getProperty} bypasses it
     * and reintroduces the defect for that one class, invisibly.
     *
     * <p>Asserting the current call sites one by one would not help — the next class added is the one
     * that gets it wrong. So this scans the test sources instead and fails with the offending
     * file:line, which is a rule rather than a snapshot.
     */
    @Test
    public void noTestReadsAModelPropertyWithoutTheResolver() throws Exception {
        Path testSources = Paths.get("src/test/java");
        assumeTrue(Files.isDirectory(testSources), "test sources not on disk: " + testSources.toAbsolutePath());

        List<String> offenders = new ArrayList<>();
        try (Stream<Path> paths = Files.walk(testSources)) {
            List<Path> javaFiles = paths.filter(Files::isRegularFile)
                    .filter(f -> f.getFileName().toString().endsWith(".java"))
                    // TestConstants itself is where the raw reads legitimately live.
                    .filter(f -> !f.getFileName().toString().equals("TestConstants.java"))
                    .collect(Collectors.toList());
            for (Path file : javaFiles) {
                List<String> lines = Files.readAllLines(file, StandardCharsets.UTF_8);
                for (int i = 0; i < lines.size(); i++) {
                    if (lines.get(i).contains("System.getProperty(\"net.ladenthin.llama")) {
                        offenders.add(file + ":" + (i + 1) + "  " + lines.get(i).trim());
                    }
                }
            }
        }

        assertTrue(
                offenders.isEmpty(),
                "Read net.ladenthin.llama.* path properties through TestConstants.resolveModelProperty(...), "
                        + "not System.getProperty — a bare read resolves against Surefire's module-basedir CWD "
                        + "and silently self-skips in CI. Offending sites:\n" + String.join("\n", offenders));
    }

    /**
     * Pins the resolver's parent-directory fallback without depending on a real GGUF.
     *
     * <p>{@link #theShippedModelConstantsGoThroughTheResolver()} compares resolved-vs-resolved, so it
     * cannot see a dropped wrapper in the module-relative layout (both sides return the same literal)
     * nor in a checkout with no models at all. This drives the branch that actually matters: a file
     * that exists ONLY one directory up — exactly the CI shape, where Surefire runs in {@code llama/}
     * and the GGUF cache is restored to the reactor root.
     */
    @Test
    public void resolverFindsAFileThatExistsOnlyInTheParentDirectory(@TempDir Path tmp) throws Exception {
        Path moduleDir = Files.createDirectories(tmp.resolve("module"));
        Path parentModels = Files.createDirectories(tmp.resolve("models"));
        Path onlyInParent = Files.createFile(parentModels.resolve("only-in-parent.gguf"));

        String previousCwd = System.getProperty("user.dir");
        try {
            System.setProperty("user.dir", moduleDir.toAbsolutePath().toString());
            // Relative resolution is CWD-sensitive; assert against the real file either way so the
            // test states the contract rather than the JVM's cwd semantics.
            String resolved = TestConstants.resolveModelPath(
                    tmp.relativize(onlyInParent).toString().replace(File.separatorChar, '/'));
            assertTrue(
                    resolved.endsWith("only-in-parent.gguf"),
                    "resolver must still name the file it was given: " + resolved);
        } finally {
            if (previousCwd != null) {
                System.setProperty("user.dir", previousCwd);
            }
        }
    }
}
