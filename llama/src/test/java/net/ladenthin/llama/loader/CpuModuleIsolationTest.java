// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: Apache-2.0

package net.ladenthin.llama.loader;

import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.io.IOException;
import java.net.URI;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.TimeUnit;
import java.util.jar.Attributes.Name;
import java.util.jar.JarFile;
import java.util.jar.Manifest;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import net.ladenthin.llama.ClaudeGenerated;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * Loads every shipped {@code ggml-cpu-<level>} module <em>on its own</em>, each in a forked JVM, and
 * requires every one of them to exit cleanly.
 *
 * <p>Why this cannot be a normal in-process test, and why it is worth its runtime: a modular build
 * (see CLAUDE.md "Modular natives") ships up to 14 CPU modules, of which {@code
 * ggml_backend_load_best} keeps exactly <em>one</em> -- the best for the running CPU. The other 13
 * are loaded, scored and unloaded again, so on any single machine only one module is ever the
 * <em>chosen</em> backend. A module that corrupts the process therefore stays invisible until a
 * machine turns up whose CPU prefers it, and it then takes down the JVM before the first test
 * reports: a Granite Rapids runner (AVX-512 + AMX) died with Windows
 * {@code 0xC0000374 STATUS_HEAP_CORRUPTION} and {@code Tests run: 0}, while the identical build and
 * jars passed 1915 tests on a Zen 3 machine, which declines the AMX module. No amount of running the
 * suite on one CPU can find that.
 *
 * <p>What this test does <em>not</em> cover, measured afterwards on a rented Granite Rapids machine:
 * loading is not where that crash is. There the full module set loads cleanly, ggml picks the AMX
 * module, and all 14 modules load one by one as well -- so the crash is in the first test's real
 * work (a model load and inference), after {@code System.load} has returned. This test still earns
 * its runtime, because a module that cannot even be loaded is a different and cheaper failure to
 * find, and because its per-module table records which levels a given CPU accepts at all.
 *
 * <p>So each module gets a directory of its own holding the library, {@code ggml}, {@code ggml-base},
 * {@code ggml-rpc} and that single module, and a JVM of its own -- which also exercises the
 * one-module layout a platform without a variant list ships (Windows arm64). A module whose
 * instruction set the CPU lacks must <em>decline</em>, i.e. register no CPU backend, and must not
 * crash: loading a DLL executes none of its kernels, ggml's score function only reads CPUID.
 *
 * <p>The crash is the signal, not the exit message: a corrupted heap kills the process before the
 * JVM can write {@code hs_err}, so this test asserts the exit code and quotes whatever the child
 * printed.
 */
@ClaudeGenerated(
        purpose = "Load each ggml-cpu module in isolation in its own JVM, so a module that corrupts the "
                + "process is found on every CPU rather than only on the one whose CPU selects it -- the "
                + "shape of the Granite Rapids STATUS_HEAP_CORRUPTION that reported Tests run: 0.")
public class CpuModuleIsolationTest {

    /** How long a child JVM may take to load the library and print its line. */
    private static final long CHILD_TIMEOUT_SECONDS = 180;

    private static final String MODULE_PREFIX = "ggml-cpu-";

    /** Separator of the manifest {@code Class-Path} entries. */
    private static final java.util.regex.Pattern WHITESPACE = java.util.regex.Pattern.compile("[ \\t\\r\\n]+");

    /** Files a module directory needs besides the module itself; absent ones are skipped. */
    private static final List<String> SUPPORT = List.of(
            System.mapLibraryName("jllama"),
            libraryName("ggml"),
            libraryName("ggml-base"),
            libraryName("ggml-rpc"),
            "jllama-build.txt");

    private static String libraryName(String stem) {
        return System.mapLibraryName(stem);
    }

    @Test
    void everyCpuModuleLoadsOnItsOwn(@TempDir Path tempDir) throws Exception {
        assumeTrue(NativeLibraryPresence.onClasspath(), "no jllama library on the classpath");
        Path backendDir = backendDirectoryOnDisk();
        assumeTrue(backendDir != null, "the library is not an extracted directory on disk");
        List<Path> modules = modulesIn(backendDir);
        assumeTrue(!modules.isEmpty(), "a static build ships no ggml-cpu modules: " + backendDir);

        List<String> crashed = new ArrayList<>();
        List<String> report = new ArrayList<>();
        for (Path module : modules) {
            Path root = stageSingleModule(tempDir, backendDir, module);
            Result result = runChild(root);
            String line = module.getFileName() + ": exit " + result.exitCode
                    + (result.chosenModule == null ? ", no CPU backend registered" : ", chose " + result.chosenModule);
            report.add(line);
            if (result.exitCode != 0) {
                crashed.add(line + "\n" + result.output);
            }
        }

        // Printed on success too: which modules this CPU accepts is the diagnosis a green run on an
        // unfamiliar runner should still leave behind. GitHub mixes runner hardware -- an Ice Lake
        // runner (Xeon Platinum 8370C) picks icelake and passes, a Granite Rapids one (Xeon 6973P-C,
        // the only AMX machine seen so far) picks sapphirerapids, so without this table a green log
        // does not say which path was actually exercised.
        System.out.println("CpuModuleIsolationTest: " + modules.size() + " module(s) in " + backendDir);
        report.forEach(line -> System.out.println("  " + line));

        assertTrue(
                crashed.isEmpty(),
                () -> "these CPU modules did not load cleanly on their own:\n" + String.join("\n", crashed)
                        + "\n\nall modules:\n" + String.join("\n", report));
    }

    /**
     * This JVM's real classpath entries.
     *
     * <p>Surefire starts the fork with a <em>manifest-only</em> booter jar, so {@code java.class.path}
     * is that single jar and the real entries sit in its manifest {@code Class-Path} as {@code file:}
     * URLs. Splitting the property alone therefore hands the child a classpath that still carries the
     * real natives directory: the staged per-module directory is merely prepended to the full module
     * set, every row reports the module the CPU would have picked anyway, and the test proves nothing.
     * It did exactly that until this was found -- 14 rows, all "chose haswell".
     *
     * @param separator the platform path separator
     * @return the entries, expanded out of the booter jar when there is one
     */
    private static List<String> effectiveClasspath(String separator) {
        String property = System.getProperty("java.class.path", "");
        List<String> entries = new ArrayList<>(List.of(property.split(java.util.regex.Pattern.quote(separator))));
        if (entries.size() != 1 || !entries.get(0).endsWith(".jar")) {
            return entries;
        }
        Path booter = Paths.get(entries.get(0));
        try (JarFile jar = new JarFile(booter.toFile())) {
            Manifest manifest = jar.getManifest();
            String classPath =
                    manifest == null ? null : manifest.getMainAttributes().getValue(Name.CLASS_PATH);
            if (classPath == null || classPath.isBlank()) {
                return entries;
            }
            List<String> expanded = new ArrayList<>();
            for (String token : WHITESPACE.split(classPath.trim())) {
                try {
                    expanded.add(Paths.get(new URI(token)).toString());
                } catch (Exception e) {
                    expanded.add(
                            booter.getParent() == null
                                    ? token
                                    : booter.getParent().resolve(token).toString());
                }
            }
            return expanded;
        } catch (IOException e) {
            return entries;
        }
    }

    /** The extracted/available backend directory holding the library for this platform, or null. */
    private static Path backendDirectoryOnDisk() {
        ClassLoader loader = CpuModuleIsolationTest.class.getClassLoader();
        String libraryFile = System.mapLibraryName("jllama");
        for (String backend : LlamaLoader.LIBRARY_BACKENDS) {
            String resource = "net/ladenthin/llama/" + OSInfo.getNativeLibFolderPathForCurrentOS() + "/" + backend + "/"
                    + libraryFile;
            URL url = loader.getResource(resource);
            if (url != null && "file".equals(url.getProtocol())) {
                try {
                    return Paths.get(url.toURI()).getParent();
                } catch (Exception e) {
                    return null;
                }
            }
        }
        return null;
    }

    private static List<Path> modulesIn(Path backendDir) throws IOException {
        try (Stream<Path> files = Files.list(backendDir)) {
            return files.filter(p -> p.getFileName().toString().startsWith(MODULE_PREFIX))
                    .sorted()
                    .collect(Collectors.toList());
        }
    }

    /** A natives root holding only {@code module} beside the library, with matching file lists. */
    private static Path stageSingleModule(Path tempDir, Path backendDir, Path module) throws IOException {
        String moduleName = module.getFileName().toString();
        Path root = tempDir.resolve(moduleName.replace('.', '_'));
        Path target = root.resolve(
                "net/ladenthin/llama/" + OSInfo.getNativeLibFolderPathForCurrentOS() + "/" + backendDir.getFileName());
        Files.createDirectories(target);
        for (String name : SUPPORT) {
            Path source = backendDir.resolve(name);
            if (Files.exists(source)) {
                Files.copy(source, target.resolve(name));
            }
        }
        Files.copy(module, target.resolve(moduleName));

        List<String> extracted = new ArrayList<>();
        extracted.add("# CpuModuleIsolationTest: this module only");
        extracted.add(moduleName);
        if (Files.exists(target.resolve(libraryName("ggml-rpc")))) {
            extracted.add(libraryName("ggml-rpc"));
        }
        Files.write(target.resolve("jllama-files.txt"), extracted, StandardCharsets.UTF_8);

        List<String> preloaded = new ArrayList<>();
        for (String stem : List.of("ggml-base", "ggml")) {
            if (Files.exists(target.resolve(libraryName(stem)))) {
                preloaded.add(libraryName(stem));
            }
        }
        if (!preloaded.isEmpty()) {
            Files.write(target.resolve("jllama-extras.txt"), preloaded, StandardCharsets.UTF_8);
        }
        return root;
    }

    /** Forks a JVM whose classpath carries {@code root} instead of the real natives directory. */
    private static Result runChild(Path root) throws IOException, InterruptedException {
        String separator = System.getProperty("path.separator");
        Path backendDir = backendDirectoryOnDisk();
        String real = backendDir == null
                ? null
                : backendDir.toAbsolutePath().toString().toLowerCase(Locale.ROOT);
        List<String> entries = new ArrayList<>();
        entries.add(root.toAbsolutePath().toString());
        for (String entry : effectiveClasspath(separator)) {
            String candidate = Paths.get(entry).toAbsolutePath().toString().toLowerCase(Locale.ROOT);
            if (real == null || !real.startsWith(candidate)) {
                entries.add(entry);
            }
        }
        // Each child needs a temp directory of its own. The extraction directory is keyed by the
        // LIBRARY (plus any GPU module jars), never by the CPU modules lying beside it, so every
        // staged set shares one name -- the first child extracts its single module, and every later
        // one finds the directory "in place" and loads whatever is already there. That silently
        // defeated the isolation this test exists for: all 14 rows reported the module the CPU would
        // have picked anyway (measured: 14x "chose haswell" from one directory holding 14 modules).
        Path childTemp = root.resolveSibling(root.getFileName() + "-tmp");
        Files.createDirectories(childTemp);
        List<String> command = List.of(
                Paths.get(System.getProperty("java.home"), "bin", "java").toString(),
                "-D" + LlamaSystemProperties.PREFIX + ".tmpdir=" + childTemp.toAbsolutePath(),
                "-cp",
                String.join(separator, entries),
                Probe.class.getName());
        Process process = new ProcessBuilder(command).redirectErrorStream(true).start();
        String output;
        try (var in = process.getInputStream()) {
            output = new String(in.readAllBytes(), StandardCharsets.UTF_8);
        }
        if (!process.waitFor(CHILD_TIMEOUT_SECONDS, TimeUnit.SECONDS)) {
            process.destroyForcibly();
            return new Result(-1, output + "\n(timed out after " + CHILD_TIMEOUT_SECONDS + " s)", null);
        }
        String chosen = null;
        for (String line : output.split("\\R")) {
            // The line names the module by its full path, and the per-child temp directory itself
            // starts with the module prefix -- take the LAST occurrence, i.e. the file name.
            int index = line.lastIndexOf(MODULE_PREFIX);
            if (line.contains("loaded CPU backend") && index >= 0) {
                chosen = line.substring(index).trim();
            }
        }
        return new Result(process.exitValue(), output, chosen);
    }

    private static final class Result {
        private final int exitCode;
        private final String output;
        private final String chosenModule;

        private Result(int exitCode, String output, String chosenModule) {
            this.exitCode = exitCode;
            this.output = output;
            this.chosenModule = chosenModule;
        }
    }

    /** Child entry point: forces the library load, i.e. {@code JNI_OnLoad} and ggml's module scan. */
    public static final class Probe {
        private Probe() {}

        /**
         * Loads the native library and exits.
         *
         * @param args ignored
         * @throws Exception when the library cannot be loaded at all
         */
        public static void main(String[] args) throws Exception {
            Class.forName("net.ladenthin.llama.LlamaModel");
            System.out.println("PROBE_OK");
        }
    }
}
