// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.loader;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.io.File;
import java.io.InputStream;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import net.ladenthin.llama.ClaudeGenerated;
import net.ladenthin.llama.LlamaModel;
import net.ladenthin.llama.LlamaQuantizer;
import net.ladenthin.llama.args.QuantizationType;
import net.ladenthin.llama.exception.LlamaException;
import net.ladenthin.llama.value.LlamaCppVersion;
import org.jspecify.annotations.Nullable;
import org.junit.jupiter.api.Test;

/**
 * Model-free smoke test that the bundled native library actually loads and its
 * {@code JNI_OnLoad} resolves every Java class it looks up by name.
 *
 * <p>Forcing {@code LlamaModel.<clinit>} runs
 * {@code LlamaLoader.initialize() -> System.load() -> JNI_OnLoad}, which calls
 * {@code FindClass(...)} for the JNI-referenced classes ({@code LlamaException},
 * {@code LogLevel}, {@code LogFormat}, ...). No GGUF model is required, so this
 * catches the two failure modes that the model-gated tests cannot exercise when
 * models are absent (e.g. in a restricted-network sandbox):
 *
 * <ul>
 *   <li>a wrong native-resource path in {@link LlamaLoader} (lib not found), and</li>
 *   <li>a stale {@code FindClass} FQN in {@code jllama.cpp} after a Java package
 *       move (lib loads but {@code JNI_OnLoad} throws
 *       {@code NoClassDefFoundError}).</li>
 * </ul>
 *
 * <p>Both bugs shipped once on this branch precisely because they only surface
 * when the library is loaded — see the regression history in {@code CLAUDE.md}.
 *
 * <p>The test self-skips when {@code libjllama} is not on the classpath (a
 * pure-Java checkout with no native build), so a plain {@code mvn test} stays
 * green without a CMake build; CI's {@code test-java-*} jobs and any local build
 * have the library and run it for real. The presence check
 * ({@link NativeLibraryPresence}) spells out the canonical resource layout rather
 * than calling {@link LlamaLoader#getNativeResourcePath()}, so a regression in that
 * method cannot silently skip this guard.
 */
@ClaudeGenerated(
        purpose = "Model-free native-load smoke: force LlamaModel.<clinit> so System.load + JNI_OnLoad "
                + "run and resolve every FindClass'd Java class. Guards against native-resource-path and "
                + "stale-JNI-FQN regressions that only appear when the library is actually loaded; skips "
                + "cleanly when libjllama is not on the classpath.")
class NativeLibraryLoadSmokeTest {

    @Test
    void loadingNativeLibraryRunsJniOnLoadWithoutError() {
        assumeTrue(NativeLibraryPresence.onClasspath(), "libjllama not on classpath — skipping native-load smoke");
        assertDoesNotThrow(
                () -> Class.forName("net.ladenthin.llama.LlamaModel"),
                "LlamaModel.<clinit> must load the native library and JNI_OnLoad must resolve "
                        + "every FindClass'd Java class");
    }

    /**
     * The native build-info getter must resolve (proving the {@code nativeLlamaCppBuildInfo} JNI
     * symbol has C linkage and is reachable) and its value must agree with the compile-time
     * {@link LlamaCppVersion#LLAMA_CPP_VERSION} pin — catching the exact drift the "Upgrading
     * llama.cpp Version" checklist warns about (a {@code GIT_TAG} bump that forgets the constant).
     * {@code llama_build_info()} returns {@code "b<number>-<commit>"}, so it must start with the
     * pinned tag followed by {@code '-'}.
     *
     * <p><strong>A stale local build fails this too, and is not a drift.</strong>
     * {@link LlamaCppVersion#LLAMA_CPP_VERSION} is a {@code static final String} constant, so javac
     * inlines its value into every referencing class &mdash; including this one. After a version
     * bump, an incremental {@code mvn test} recompiles the changed constant but not this test class
     * (its own source did not change), so the assertion compares the freshly linked native tag
     * against the <em>previous</em> literal baked into the test bytecode. The give-away is that the
     * build-info in the message is the tag you just bumped <em>to</em>. Fix with
     * {@code mvn -pl llama clean test}; CI always builds from a clean checkout and cannot hit it.
     */
    @Test
    void nativeBuildInfoMatchesPinnedVersionConstant() {
        assumeTrue(
                NativeLibraryPresence.onClasspath(), "libjllama not on classpath — skipping native build-info check");
        String buildInfo = LlamaModel.getLlamaCppBuildInfo();
        assertNotNull(buildInfo, "getLlamaCppBuildInfo() must return the linked llama.cpp build identifier");
        assertTrue(
                buildInfo.startsWith(LlamaCppVersion.LLAMA_CPP_VERSION + "-"),
                "Linked build-info \"" + buildInfo + "\" must start with the pinned tag \""
                        + LlamaCppVersion.LLAMA_CPP_VERSION + "-\". Two different causes produce this, "
                        + "and only the first is a real defect: (1) GIT_TAG in llama/CMakeLists.txt and "
                        + "LlamaCppVersion.LLAMA_CPP_VERSION have drifted apart; or (2) this is a stale "
                        + "local build. LLAMA_CPP_VERSION is a compile-time constant, so javac inlines "
                        + "its value into THIS test class -- an incremental build after a version bump "
                        + "leaves the old literal here while the native library is already the new tag. "
                        + "If the tag shown above is the one you just bumped to, it is cause (2): run "
                        + "`mvn -pl llama clean test`. CI always builds clean and never hits it.");
    }

    /**
     * The provenance the build writes next to the library reaches Java: after a load from a natives
     * directory of this build, {@link LlamaModel#getJllamaBuildInfo()} names the pinned llama.cpp tag,
     * the carried patches and the rest of the keys {@code jllama-build.txt} carries. The tag is checked
     * against the same constant as the linked build-info above, so a directory of another build on the
     * classpath is caught here as well.
     */
    @Test
    void jllamaBuildInfoNamesThePinnedTagAndTheCarriedPatches() throws ClassNotFoundException {
        assumeTrue(NativeLibraryPresence.onClasspath(), "libjllama not on classpath — skipping build-info check");
        Class.forName("net.ladenthin.llama.LlamaModel");
        Map<String, String> build = LlamaModel.getJllamaBuildInfo();
        assertEquals(LlamaCppVersion.LLAMA_CPP_VERSION, build.get("llama.cpp"), "llama.cpp tag of " + build);
        assertTrue(
                String.valueOf(build.get("llama.cpp.commit")).matches("[0-9a-f]{40}"), "llama.cpp.commit of " + build);
        assertTrue(String.valueOf(build.get("patches")).matches("\\d{4}(,\\d{4})*"), "patches of " + build);
        assertTrue(String.valueOf(build.get("patches.sha256")).matches("[0-9a-f]{64}"), "patches.sha256 of " + build);
        assertTrue(
                String.valueOf(build.get("jllama.version")).matches("\\d+\\.\\d+\\.\\d+(-SNAPSHOT)?"),
                "jllama.version of " + build);
        for (String key : new String[] {"jllama.commit", "classifier", "compiler", "toolkit", "signed", "kind"}) {
            assertNotNull(build.get(key), key + " of " + build);
        }
    }

    /**
     * The natives directory contract, for real and on every platform the Java tests run on: a fresh
     * JVM with the classes on its classpath but <em>no</em> natives (the {@code src/main/natives}
     * tree is left out) loads the library from the very directory this build wrote, named with
     * {@code net.ladenthin.llama.lib.path}. The child's loader line must say so and name the directory;
     * its build info must carry the pinned tag. This is the test that runs the directory branch on
     * Windows, where {@code jllama.dll} imports {@code ggml.dll}/{@code ggml-base.dll} and a bare
     * {@code System.load} used to fail -- nothing else in CI launches that branch there.
     */
    @Test
    void libPathLoadsTheNativesDirectoryInAFreshJvm() throws Exception {
        assumeTrue(NativeLibraryPresence.onClasspath(), "libjllama not on classpath — skipping lib.path check");
        URL library = libraryUrl();
        assumeTrue(
                library != null && "file".equals(library.getProtocol()),
                "the library is not a plain file on the classpath (packaged jar): the lib.path child needs a directory");
        Path dir = Paths.get(library.toURI()).getParent();
        String classpath = classpathWithoutNatives(System.getProperty("java.class.path"), dir);
        String java = Paths.get(System.getProperty("java.home"), "bin", "java").toString();
        ProcessBuilder builder = new ProcessBuilder(
                java,
                "-cp",
                classpath,
                "-D" + LlamaSystemProperties.PREFIX + ".lib.path=" + dir,
                LibPathChild.class.getName());
        builder.redirectErrorStream(true);
        Process child = builder.start();
        String output;
        try (InputStream in = child.getInputStream()) {
            output = new String(readAll(in), StandardCharsets.UTF_8);
        }
        assertTrue(child.waitFor(5, TimeUnit.MINUTES), "the lib.path child JVM did not finish:\n" + output);
        assertEquals(0, child.exitValue(), "the lib.path child JVM failed:\n" + output);
        assertTrue(output.contains("native backend '") && output.contains("' loaded from " + dir), output);
        assertTrue(output.contains("named by " + LlamaSystemProperties.PREFIX + ".lib.path"), output);
        assertTrue(output.contains("child build llama.cpp=" + LlamaCppVersion.LLAMA_CPP_VERSION), output);
    }

    /** What the lib.path child runs: load the library through {@link LlamaModel}, print the build facts. */
    public static final class LibPathChild {
        private LibPathChild() {}

        public static void main(String[] args) throws ClassNotFoundException {
            Class.forName("net.ladenthin.llama.LlamaModel");
            Map<String, String> build = LlamaModel.getJllamaBuildInfo();
            System.out.println(
                    "child build llama.cpp=" + build.get("llama.cpp") + " linked=" + LlamaModel.getLlamaCppBuildInfo());
        }
    }

    private static @Nullable URL libraryUrl() {
        ClassLoader loader = NativeLibraryLoadSmokeTest.class.getClassLoader();
        for (String backend : LlamaLoader.LIBRARY_BACKENDS) {
            URL url = loader.getResource("net/ladenthin/llama/" + OSInfo.getNativeLibFolderPathForCurrentOS() + "/"
                    + backend + "/" + System.mapLibraryName("jllama"));
            if (url != null) {
                return url;
            }
        }
        return null;
    }

    /** The classpath without the entry that holds the natives tree (so only lib.path can find them). */
    static String classpathWithoutNatives(String classpath, Path nativesDir) {
        List<String> kept = new ArrayList<>();
        for (String entry : classpath.split(java.util.regex.Pattern.quote(File.pathSeparator))) {
            if (entry.isEmpty()) {
                continue;
            }
            Path candidate = Paths.get(entry).toAbsolutePath().normalize();
            if (nativesDir.toAbsolutePath().normalize().startsWith(candidate)) {
                continue;
            }
            kept.add(entry);
        }
        return String.join(File.pathSeparator, kept);
    }

    private static byte[] readAll(InputStream in) throws java.io.IOException {
        java.io.ByteArrayOutputStream out = new java.io.ByteArrayOutputStream();
        byte[] buffer = new byte[8192];
        int n;
        while ((n = in.read(buffer)) != -1) {
            out.write(buffer, 0, n);
        }
        return out.toByteArray();
    }

    /**
     * {@link LlamaQuantizer#quantize} must reach its native implementation. The declarations that
     * give the {@code LlamaModel} entry points C linkage come from the javac-generated
     * {@code jllama.h}, which covers <em>only</em> {@code LlamaModel} — so every JNI function for
     * any other class has to say {@code extern "C"} itself. {@code quantizeNative} did not, and was
     * therefore exported under its C++-mangled name, making this public API throw
     * {@link UnsatisfiedLinkError} on every platform in every published jar.
     *
     * <p>Nothing caught it: the only coverage was {@code QuantizerIntegrationTest}, which gates on a
     * GGUF being present and so skipped in CI for as long as the model paths resolved to the wrong
     * directory. Asserting it here — model-free, next to the build-info linkage check — means a
     * future entry point that forgets {@code extern "C"} fails a test that always runs wherever the
     * library exists.
     *
     * <p>A missing input file is the cheapest call that still crosses JNI: reaching the native side
     * at all produces {@link LlamaException}, whereas a linkage regression fails earlier and
     * differently with {@link UnsatisfiedLinkError} (which is an {@link Error}, so it propagates out
     * of an {@code assertThrows(LlamaException.class, ...)} rather than being reported as a
     * wrong-exception mismatch).
     */
    @Test
    void quantizerNativeEntryPointResolves() {
        assumeTrue(
                NativeLibraryPresence.onClasspath(), "libjllama not on classpath — skipping quantizer linkage check");
        assertThrows(
                LlamaException.class,
                () -> LlamaQuantizer.quantize(
                        "does-not-exist-quantizer-linkage-probe.gguf",
                        "never-written.gguf",
                        QuantizationType.Q8_0,
                        0,
                        true),
                "LlamaQuantizer.quantize must reach the native implementation and fail with "
                        + "LlamaException; an UnsatisfiedLinkError here means quantizeNative lost its "
                        + "extern \"C\" linkage and is exported under a C++-mangled name");
    }

    /**
     * {@link LlamaModel#jsonSchemaToGrammar(String)} needs no model, so it belongs in the model-free
     * suite.
     *
     * <p>Its only assertions used to live in {@code LlamaModelTest}, whose {@code @BeforeAll} gates
     * the entire class on a GGUF being present &mdash; so on a host without models the class runs
     * <em>zero</em> tests and the native call is never made. That mattered concretely during this
     * bump: the b10585 {@code common_json} switch changed this method's native implementation from
     * {@code nlohmann::ordered_json::parse} to {@code json::parse}, a hard compile error that a
     * green build hid nothing of &mdash; but any future runtime regression on the same path would
     * have had no runnable guard wherever models are absent.
     *
     * <p>Both directions are asserted, because they fail differently: a valid schema must cross JNI
     * and come back with a non-empty grammar, and a malformed one must surface as a catchable
     * {@link LlamaException} rather than letting a C++ {@code json::parse} exception escape across
     * the JNI boundary and abort the JVM (the PR #258 exception-boundary fix).
     */
    @Test
    void jsonSchemaToGrammarWorksWithoutAModel() {
        assumeTrue(NativeLibraryPresence.onClasspath(), "libjllama not on classpath — skipping grammar check");

        String grammar =
                LlamaModel.jsonSchemaToGrammar("{\"type\":\"object\",\"properties\":{\"a\":{\"type\":\"string\"}}}");
        assertNotNull(grammar, "jsonSchemaToGrammar must return a grammar for a valid schema");
        assertTrue(grammar.contains("root ::="), "a GBNF grammar must define a root rule; got: " + grammar);

        assertThrows(
                LlamaException.class,
                () -> LlamaModel.jsonSchemaToGrammar("{ this is not valid json"),
                "a malformed schema must surface as LlamaException, not abort the JVM");
    }

    /**
     * An error message that quotes the caller's text keeps a four-byte character. Upstream's schema
     * converter names the {@code $ref} it cannot resolve, and the message reaches Java through the
     * same {@code throw_api_status} as every failed {@code LlamaModel} call; {@code ThrowNew} would
     * read its standard UTF-8 as Modified UTF-8 and mangle the character.
     */
    @Test
    void anErrorMessageKeepsAFourByteCharacterOfTheRequest() {
        assumeTrue(NativeLibraryPresence.onClasspath(), "libjllama not on classpath — skipping grammar check");
        String emoji = "\uD83D\uDE00"; // U+1F600

        LlamaException e = assertThrows(
                LlamaException.class, () -> LlamaModel.jsonSchemaToGrammar("{\"$ref\":\"#/$defs/" + emoji + "\"}"));

        String message = String.valueOf(e.getMessage());
        assertTrue(message.contains("#/$defs/" + emoji), "the $ref must come back intact; got: " + message);
    }
}
