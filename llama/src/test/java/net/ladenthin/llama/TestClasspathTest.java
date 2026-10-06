// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.hasItem;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.notNullValue;

import com.fasterxml.jackson.annotation.JsonAutoDetect;
import com.fasterxml.jackson.annotation.PropertyAccessor;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.File;
import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

/**
 * Pins how the build lays out the test classpath, so a pom change that breaks it fails here,
 * model-free and on every run, instead of in a model-gated test that only CI reaches.
 *
 * <p>Two of these failed on {@code main} after the natives moved out of {@code target/classes}:
 * Surefire's default module mode put Jackson on the module path (ExamplesTest could not bind a
 * test type, the router worker JVM found no Jackson on {@code java.class.path}), and PIT, which
 * builds its own classpath, had no native library. The pom fixes are
 * {@code useModulePath=false} on Surefire and {@code additionalClasspathElements} on PIT; this
 * class is in PIT's {@code targetTests} so it runs under both.</p>
 */
@ClaudeGenerated(
        purpose = "Guard the Surefire and PIT test-classpath settings whose loss broke "
                + "ExamplesTest, RouterModeIntegrationTest and the PIT gate on main.")
public class TestClasspathTest {

    /** The resource root of the native libraries, as {@code LlamaLoader} looks them up. */
    private static final String NATIVE_RESOURCE_ROOT = "net/ladenthin/llama";

    /** Bound by Jackson below; private, so binding it needs a reflective access check. */
    private static final class Probe {
        private String name;

        private Probe() {}
    }

    @Test
    public void testsRunOnTheClasspathNotInANamedModule() {
        assertThat(TestClasspathTest.class.getModule().isNamed(), is(false));
    }

    @Test
    public void jacksonBindsAPrivateTestType() throws IOException {
        ObjectMapper mapper = new ObjectMapper();
        mapper.setVisibility(PropertyAccessor.FIELD, JsonAutoDetect.Visibility.ANY);

        Probe probe = mapper.readValue("{\"name\":\"x\"}", Probe.class);

        assertThat(probe.name, is("x"));
    }

    @Test
    public void jacksonIsOnTheJavaClassPathAWorkerJvmIsStartedWith() throws URISyntaxException {
        String jackson = location(ObjectMapper.class).toString();

        assertThat(classPathEntries(), hasItem(jackson));
    }

    @Test
    public void everyLocallyBuiltNativeLibraryIsAResourceOnTheClasspath() throws Exception {
        Path nativesRoot = moduleBaseDir().resolve("src/main/natives");
        Path libraries = nativesRoot.resolve(NATIVE_RESOURCE_ROOT);
        List<Path> built = nativeLibraries(libraries);
        Assumptions.assumeFalse(built.isEmpty(), "no native build under " + libraries);

        ClassLoader loader = TestClasspathTest.class.getClassLoader();
        for (Path library : built) {
            String resource = nativesRoot.relativize(library).toString().replace(File.separatorChar, '/');
            assertThat(resource, loader.getResource(resource), notNullValue());
        }
    }

    private static List<Path> nativeLibraries(Path dir) throws IOException {
        if (!Files.isDirectory(dir)) {
            return Collections.emptyList();
        }
        try (Stream<Path> files = Files.walk(dir)) {
            return files.filter(Files::isRegularFile)
                    .filter(p -> p.getFileName().toString().contains("jllama"))
                    .collect(Collectors.toList());
        }
    }

    /**
     * The {@code llama} module directory, derived from where this class was loaded
     * ({@code llama/target/test-classes}), so it holds whatever the working directory is:
     * Surefire runs in the module, PIT and CI's {@code mvn -f llama/pom.xml} may not.
     */
    private static Path moduleBaseDir() throws URISyntaxException {
        return location(TestClasspathTest.class).getParent().getParent();
    }

    private static Path location(Class<?> type) throws URISyntaxException {
        return Paths.get(
                        type.getProtectionDomain().getCodeSource().getLocation().toURI())
                .toAbsolutePath();
    }

    private static List<String> classPathEntries() {
        return Arrays.stream(System.getProperty("java.class.path", "").split(File.pathSeparator))
                .map(entry -> Paths.get(entry).toAbsolutePath().toString())
                .collect(Collectors.toList());
    }
}
