// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.langchain4j;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.notNullValue;
import static org.hamcrest.Matchers.nullValue;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
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

/**
 * Guards {@link TestModelPaths}, which is this module's copy of the core module's model-path
 * resolver.
 *
 * <p>It exists because of a defect that stayed invisible for months: Surefire's working directory is
 * the module basedir while CI restores the GGUF cache to the reactor root, so a bare
 * {@code models/<name>} resolved to nothing, every model-gated class aborted in its
 * {@code Assumptions.assumeTrue(exists)}, and the job stayed green with the tests reported as
 * <em>skipped</em>. The resolver fixes that; until now nothing proved the resolver itself works, and
 * nothing stopped a new test class from going back to a bare {@code System.getProperty}.
 */
class TestModelPathsTest {

    @Test
    void nullAndEmptyResolveToNull() {
        assertThat(TestModelPaths.resolve(null), is(nullValue()));
        assertThat(TestModelPaths.resolve(""), is(nullValue()));
    }

    @Test
    void anExistingModuleRelativePathIsReturnedUnchanged(@TempDir Path tmp) throws IOException {
        Path present = Files.createFile(tmp.resolve("present.gguf"));
        Path resolved = TestModelPaths.resolve(present.toString());
        assertThat(resolved, is(notNullValue()));
        assertThat(Files.exists(resolved), is(true));
    }

    @Test
    void anAbsolutePathIsReturnedEvenWhenItDoesNotExist(@TempDir Path tmp) {
        Path missing = tmp.resolve("missing.gguf").toAbsolutePath();
        assertThat(TestModelPaths.resolve(missing.toString()), is(missing));
    }

    @Test
    void anUnresolvablePathComesBackUnchangedSoTheSkipMessageNamesIt() {
        // Deliberately NOT null: the caller puts this in its assumption message, and "models/x.gguf
        // not found" is a far more useful CI line than "null".
        String wanted = "models/definitely-not-here-" + TestModelPathsTest.class.getSimpleName() + ".gguf";
        Path resolved = TestModelPaths.resolve(wanted);
        assertThat(resolved, is(Paths.get(wanted)));
    }

    @Test
    void fromPropertyIsNullWhenThePropertyIsUnset() {
        assertThat(TestModelPaths.fromProperty("net.ladenthin.llama.langchain4j.definitely.unset"), is(nullValue()));
    }

    /**
     * The rule, not a snapshot: no test in this module may read a model path with a bare
     * {@code System.getProperty}, because that bypasses the resolver and silently re-mutes that class
     * in CI. {@link TestModelPaths} itself is where the raw read legitimately lives.
     */
    @Test
    void noTestReadsAModelPropertyWithoutTheResolver() throws IOException {
        Path testSources = Paths.get("src/test/java");
        assertTrue(Files.isDirectory(testSources), "test sources not on disk: " + testSources.toAbsolutePath());

        List<String> offenders = new ArrayList<>();
        try (Stream<Path> paths = Files.walk(testSources)) {
            List<Path> javaFiles = paths.filter(Files::isRegularFile)
                    .filter(f -> f.getFileName().toString().endsWith(".java"))
                    .filter(f -> !f.getFileName().toString().equals("TestModelPaths.java"))
                    .filter(f -> !f.getFileName().toString().equals("TestModelPathsTest.java"))
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
                "Read net.ladenthin.llama.* path properties through TestModelPaths, not System.getProperty "
                        + "- a bare read resolves against Surefire's module-basedir CWD and silently self-skips "
                        + "in CI. Offending sites:\n" + String.join("\n", offenders));
    }
}
