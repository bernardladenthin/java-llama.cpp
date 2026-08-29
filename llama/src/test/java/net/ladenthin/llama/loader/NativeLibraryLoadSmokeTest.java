// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.loader;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import net.ladenthin.llama.ClaudeGenerated;
import net.ladenthin.llama.LlamaModel;
import net.ladenthin.llama.LlamaQuantizer;
import net.ladenthin.llama.args.QuantizationType;
import net.ladenthin.llama.exception.LlamaException;
import net.ladenthin.llama.value.LlamaCppVersion;
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
 * have the library and run it for real. The presence check uses the canonical
 * resource layout directly (not {@link LlamaLoader#getNativeResourcePath()}) so
 * a regression in that method cannot silently skip this guard.
 */
@ClaudeGenerated(
        purpose = "Model-free native-load smoke: force LlamaModel.<clinit> so System.load + JNI_OnLoad "
                + "run and resolve every FindClass'd Java class. Guards against native-resource-path and "
                + "stale-JNI-FQN regressions that only appear when the library is actually loaded; skips "
                + "cleanly when libjllama is not on the classpath.")
class NativeLibraryLoadSmokeTest {

    private static boolean nativeLibraryOnClasspath() {
        String resource = "/net/ladenthin/llama/" + OSInfo.getNativeLibFolderPathForCurrentOS() + "/"
                + System.mapLibraryName("jllama");
        return NativeLibraryLoadSmokeTest.class.getResource(resource) != null;
    }

    @Test
    void loadingNativeLibraryRunsJniOnLoadWithoutError() {
        assumeTrue(nativeLibraryOnClasspath(), "libjllama not on classpath — skipping native-load smoke");
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
        assumeTrue(nativeLibraryOnClasspath(), "libjllama not on classpath — skipping native build-info check");
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
        assumeTrue(nativeLibraryOnClasspath(), "libjllama not on classpath — skipping quantizer linkage check");
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
        assumeTrue(nativeLibraryOnClasspath(), "libjllama not on classpath — skipping grammar check");

        String grammar =
                LlamaModel.jsonSchemaToGrammar("{\"type\":\"object\",\"properties\":{\"a\":{\"type\":\"string\"}}}");
        assertNotNull(grammar, "jsonSchemaToGrammar must return a grammar for a valid schema");
        assertTrue(grammar.contains("root ::="), "a GBNF grammar must define a root rule; got: " + grammar);

        assertThrows(
                LlamaException.class,
                () -> LlamaModel.jsonSchemaToGrammar("{ this is not valid json"),
                "a malformed schema must surface as LlamaException, not abort the JVM");
    }
}
