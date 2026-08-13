// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

import net.ladenthin.llama.LlamaModel;
import net.ladenthin.llama.value.LlamaCppVersion;

/**
 * Model-free load smoke for the PACKAGED native library, run by {@code .github/smoke-native-macos.sh}
 * via the JDK single-file source launcher ({@code java -cp <fatjar> NativeLoadSmoke.java}) — no
 * Maven, no test framework, no GGUF.
 *
 * <p>Touching {@link LlamaModel} runs its static initializer, which is
 * {@code LlamaLoader.initialize() -> System.load() -> JNI_OnLoad}: the library is extracted from the
 * jar, mapped by the OS loader, and every class its {@code FindClass} calls reference is resolved.
 * On macOS that mapping is also where a dylib whose ad-hoc signature no longer covers its own
 * {@code __TEXT} pages is rejected — the failure that shipped in 5.0.6. Then
 * {@link LlamaModel#getLlamaCppBuildInfo()} crosses the JNI boundary for real and is checked against
 * the compile-time pin, so a jar carrying a stale or foreign library fails here rather than in a
 * user's process.</p>
 *
 * <p>This is the packaged-artifact counterpart of {@code NativeLibraryLoadSmokeTest}, which asserts
 * the same two things against {@code target/classes} during {@code mvn test} — that one cannot see
 * what the assembled jar contains.</p>
 */
public final class NativeLoadSmoke {

    private NativeLoadSmoke() {}

    public static void main(String[] args) {
        final String pinned = LlamaCppVersion.LLAMA_CPP_VERSION;

        // Forces LlamaLoader.initialize() -> System.load() -> JNI_OnLoad.
        final String build = LlamaModel.getLlamaCppBuildInfo();

        if (build == null || build.isEmpty()) {
            throw new IllegalStateException("getLlamaCppBuildInfo() returned no build identifier");
        }
        if (!build.startsWith(pinned)) {
            throw new IllegalStateException("packaged native library reports llama.cpp build '" + build
                    + "' but this jar pins '" + pinned + "' — the jar carries the wrong native library");
        }
        System.out.println("native load smoke OK: pinned=" + pinned + " linked=" + build);
    }
}
