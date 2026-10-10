// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.loader;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import net.ladenthin.llama.ClaudeGenerated;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

@ClaudeGenerated(
        purpose = "Drive LlamaLoader.initialize() end-to-end against the committed fixture trees "
                + "(src/test/resources/net/ladenthin/llama/Linux/backendtest*/<backend>/) by redirecting the "
                + "arch component via the osinfo.architecture override. backendtest/ holds an unloadable fake "
                + "library in cpu/ (with a fixture CPU module in its jllama-files.txt) and fake GPU modules in "
                + "cuda13/ (plus an unloadable extra), rocm/ (built from another llama.cpp tag) and vulkan/, "
                + "exercising the build check, the module filter, the extraction of every selected module into "
                + "the library's directory, an extra that fails to load, the clean failure of the library load "
                + "and the no-library case; backendtest-ok/ has a trivial real x86-64 ELF dummy as the library "
                + "and as cuda13's extra, so the success path executes too, with and without modules. "
                + "Linux-only (the trees are committed under the Linux OS folder); self-skips elsewhere.")
public class BackendLoadTest {

    /** Arch-folder override selecting the committed fixture tree {@code Linux/backendtest/}. */
    private static final String FIXTURE_ARCH = "backendtest";

    private static final String ARCH_PROP = LlamaSystemProperties.PREFIX + ".osinfo.architecture";
    private static final String TMPDIR_PROP = LlamaSystemProperties.PREFIX + ".tmpdir";
    private static final String BACKEND_PROP = LlamaSystemProperties.PREFIX + ".backend";

    private String previousArch;
    private String previousTmpDir;
    private String previousBackend;

    @TempDir
    Path tempDir;

    @BeforeEach
    public void redirectLoaderToFixtures() {
        previousArch = System.getProperty(ARCH_PROP);
        previousTmpDir = System.getProperty(TMPDIR_PROP);
        previousBackend = System.getProperty(BACKEND_PROP);
        System.setProperty(ARCH_PROP, FIXTURE_ARCH);
        System.setProperty(TMPDIR_PROP, tempDir.toString());
        System.clearProperty(BACKEND_PROP);
    }

    @AfterEach
    public void restoreProperties() {
        restore(ARCH_PROP, previousArch);
        restore(TMPDIR_PROP, previousTmpDir);
        restore(BACKEND_PROP, previousBackend);
    }

    private static void restore(String key, String value) {
        if (value == null) {
            System.clearProperty(key);
        } else {
            System.setProperty(key, value);
        }
    }

    /** The extraction directories below the temp dir (there is one per library-and-modules combination). */
    private List<Path> extractionDirs() throws IOException {
        try (Stream<Path> entries = Files.list(tempDir)) {
            return entries.filter(p -> p.getFileName().toString().startsWith(LlamaLoader.BACKEND_TEMP_DIR_PREFIX))
                    .sorted()
                    .collect(Collectors.toList());
        }
    }

    /** The one extraction directory this start used. */
    private Path theExtractionDir() throws IOException {
        List<Path> dirs = extractionDirs();
        assertEquals(1, dirs.size(), "expected exactly one extraction directory: " + dirs);
        return dirs.get(0);
    }

    private static void assumeLinuxFixtureTree() {
        // The fixture tree is committed under the Linux OS folder; the OS path component
        // cannot be overridden, so these tests are meaningful only on a Linux JVM (which is
        // where the CI coverage run executes).
        assumeTrue("Linux".equals(OSInfo.getOSName()), "backend fixtures are committed for Linux only");
    }

    private static void assumeX86_64() {
        // The loadable dummy libraries in backendtest-ok/ are real ELF shared objects
        // compiled for x86-64 Linux (see their fixture README); loading them anywhere else
        // fails for the wrong reason.
        String osArch = System.getProperty("os.arch", "");
        assumeTrue(
                "amd64".equals(osArch) || "x86_64".equals(osArch),
                "loadable dummy backend libraries are built for x86-64 only");
    }

    private static Set<String> names(Path dir) throws IOException {
        try (Stream<Path> entries = Files.list(dir)) {
            return entries.map(p -> p.getFileName().toString()).collect(Collectors.toSet());
        }
    }

    @Test
    public void aModuleOfAnotherBuildIsRefusedBeforeAnythingIsExtracted() throws IOException {
        assumeLinuxFixtureTree();
        // rocm's jllama-build.txt names another llama.cpp tag than cpu's; with every module selected
        // the start must refuse it, and the refusal comes before the first file is written.
        UnsatisfiedLinkError error = assertThrows(UnsatisfiedLinkError.class, LlamaLoader::initialize);
        String message = error.getMessage();
        assertTrue(message.contains("'rocm' was built from llama.cpp bOTHER"), message);
        assertTrue(message.contains("'cpu' from llama.cpp bTEST"), message);
        assertTrue(extractionDirs().isEmpty(), "nothing may be extracted: " + extractionDirs());
    }

    @Test
    public void theSelectedModulesLandNextToTheLibraryWhichThenFailsToLoadCleanly() throws IOException {
        assumeLinuxFixtureTree();
        System.setProperty(BACKEND_PROP, "vulkan");
        UnsatisfiedLinkError error = assertThrows(UnsatisfiedLinkError.class, LlamaLoader::initialize);
        String message = error.getMessage();
        assertTrue(message.contains("backend 'cpu' could not be loaded"), message);
        assertTrue(message.contains("GPU modules: vulkan"), message);
        // One directory, keyed by the library and the vulkan module: the library's own module file and
        // the GPU module were put in place (never loaded by the JVM -- the fixtures are not libraries),
        // then the fake library failed its load. cuda13 and rocm were not selected: nothing of theirs.
        Path dir = theExtractionDir();
        String base = "net/ladenthin/llama/Linux/" + FIXTURE_ARCH;
        assertEquals(
                LlamaLoader.extractionDirectoryName(
                        "cpu",
                        LlamaLoader.resource(base + "/cpu/libjllama.so"),
                        java.util.Collections.singletonList(LlamaLoader.resource(base + "/vulkan/jllama-files.txt"))),
                dir.getFileName().toString());
        Set<String> files = names(dir);
        assertTrue(files.contains("libggml-cpu-fixture.so"), files.toString());
        assertTrue(files.contains("libggml-vulkan.so"), files.toString());
        assertTrue(files.contains("libjllama.so"), files.toString());
        assertFalse(files.contains("libggml-cuda.so"), files.toString());
        assertFalse(files.contains("libggml-hip.so"), files.toString());
    }

    @Test
    public void anExtraOfAModuleThatFailsToLoadFailsTheStartBeforeTheLibrary() throws IOException {
        assumeLinuxFixtureTree();
        System.setProperty(BACKEND_PROP, "cuda13,vulkan");
        UnsatisfiedLinkError error = assertThrows(UnsatisfiedLinkError.class, LlamaLoader::initialize);
        assertTrue(error.getMessage().contains("GPU modules: cuda13, vulkan"), error.getMessage());
        // cuda13's jllama-extras.txt names libextra.so, a fake: extracted, its load fails, and the start
        // stops there -- the plain files and the library are never written.
        Set<String> files = names(theExtractionDir());
        assertTrue(files.contains("libextra.so"), files.toString());
        assertFalse(files.contains("libggml-vulkan.so"), files.toString());
        assertFalse(files.contains("libjllama.so"), files.toString());
    }

    @Test
    public void cpuAloneSelectsNoModuleAndUsesTheLibrarysOwnDirectory() throws IOException {
        assumeLinuxFixtureTree();
        System.setProperty(BACKEND_PROP, "cpu");
        UnsatisfiedLinkError error = assertThrows(UnsatisfiedLinkError.class, LlamaLoader::initialize);
        assertTrue(error.getMessage().contains("GPU modules: none"), error.getMessage());
        Path dir = theExtractionDir();
        assertEquals(
                LlamaLoader.extractionDirectoryName(
                        "cpu", LlamaLoader.resource("net/ladenthin/llama/Linux/" + FIXTURE_ARCH + "/cpu/libjllama.so")),
                dir.getFileName().toString());
        Set<String> files = names(dir);
        assertEquals(new java.util.HashSet<>(java.util.Arrays.asList("libggml-cpu-fixture.so", "libjllama.so")), files);
    }

    @Test
    public void aNamedModuleThatIsNotOnTheClasspathFailsLoud() throws IOException {
        assumeLinuxFixtureTree();
        System.setProperty(BACKEND_PROP, "opencl");
        UnsatisfiedLinkError error = assertThrows(UnsatisfiedLinkError.class, LlamaLoader::initialize);
        assertTrue(error.getMessage().contains("names 'opencl', but no natives jar"), error.getMessage());
        assertTrue(extractionDirs().isEmpty(), extractionDirs().toString());
    }

    @Test
    public void withoutAnyLibraryTheErrorNamesTheMissingNativesJar() {
        assumeLinuxFixtureTree();
        System.setProperty(ARCH_PROP, "backendtest-none");
        UnsatisfiedLinkError error = assertThrows(UnsatisfiedLinkError.class, LlamaLoader::initialize);
        String message = error.getMessage();
        assertTrue(message.contains("os.arch=backendtest-none"), message);
        assertTrue(message.contains("cpu-<os>-<arch>"), message);
        assertTrue(message.contains("A GPU natives jar alone is not enough"), message);
    }

    @Test
    public void loadsTheLibraryWithEveryModuleNextToIt() throws IOException {
        assumeLinuxFixtureTree();
        assumeX86_64();
        System.setProperty(ARCH_PROP, "backendtest-ok");
        // Pre-create stale extraction artifacts so cleanup()'s recursive directory branch
        // executes on this initialize() call (deletion is not asserted: cleanup is skipped
        // when an earlier test class in the same JVM already loaded a native library).
        Path stale =
                tempDir.resolve(LlamaLoader.BACKEND_TEMP_DIR_PREFIX + "stale").resolve("nested");
        Files.createDirectories(stale);
        Files.write(stale.resolve("libjllama.so"), new byte[] {1});
        // Must succeed: cuda13's real extra library loads, both fake modules are only put in place,
        // and the real dummy library loads (it has no JNI_OnLoad, so nothing looks at the modules).
        LlamaLoader.initialize();
        List<Path> dirs = extractionDirs().stream()
                .filter(p -> !p.getFileName().toString().equals(LlamaLoader.BACKEND_TEMP_DIR_PREFIX + "stale"))
                .collect(Collectors.toList());
        assertEquals(1, dirs.size(), dirs.toString());
        Set<String> files = names(dirs.get(0));
        for (String expected : new String[] {"libjllama.so", "libextra.so", "libggml-cuda.so", "libggml-vulkan.so"}) {
            assertTrue(files.contains(expected), expected + " missing in " + files);
        }
    }

    @Test
    public void loadsTheLibraryAloneWhenCpuIsForced() throws IOException {
        assumeLinuxFixtureTree();
        assumeX86_64();
        System.setProperty(ARCH_PROP, "backendtest-ok");
        System.setProperty(BACKEND_PROP, "cpu");
        LlamaLoader.initialize();
        Path dir = theExtractionDir();
        assertEquals(
                LlamaLoader.extractionDirectoryName(
                        "cpu", LlamaLoader.resource("net/ladenthin/llama/Linux/backendtest-ok/cpu/libjllama.so")),
                dir.getFileName().toString());
        assertEquals(java.util.Collections.singleton("libjllama.so"), names(dir));
    }
}
