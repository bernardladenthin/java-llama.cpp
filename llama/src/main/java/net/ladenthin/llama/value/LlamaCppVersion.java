// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
// SPDX-FileCopyrightText: 2023-2025 Konstantin Herud
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.value;

/**
 * The pinned upstream <a href="https://github.com/ggml-org/llama.cpp">llama.cpp</a> build tag this
 * library was compiled against, exposed as a compile-time constant so callers can render a badge or
 * emit a startup log line without loading the native library.
 *
 * <p>{@link #LLAMA_CPP_VERSION} is a pure-Java string ({@code "b10092"}) that mirrors the
 * {@code GIT_TAG} in {@code llama/CMakeLists.txt}. It is available even when {@code libjllama} is
 * absent (pure-Java checkout, before {@code System.load}), which is what makes it suitable for a
 * lightweight version badge in Android or other UIs.</p>
 *
 * <p>For the <em>authoritative</em> value that is baked into the native binary — the build number
 * plus the resolved upstream commit, e.g. {@code "b10092-<commit>"} — call
 * {@link net.ladenthin.llama.LlamaModel#getLlamaCppBuildInfo()} instead; that reads llama.cpp's own
 * {@code build-info} through JNI and therefore cannot drift from the compiled library (but requires
 * the native library to be loaded).</p>
 */
public final class LlamaCppVersion {

    /**
     * The pinned llama.cpp release tag this library was built against, e.g. {@code "b10092"}.
     *
     * <p>Kept in lockstep with {@code GIT_TAG} in {@code llama/CMakeLists.txt} — see the
     * "Upgrading/Downgrading llama.cpp Version" checklist in {@code CLAUDE.md}. This is the
     * compile-time pin; use {@link net.ladenthin.llama.LlamaModel#getLlamaCppBuildInfo()} for the
     * value actually linked into the native binary.</p>
     */
    public static final String LLAMA_CPP_VERSION = "b10092";

    // Constants holder — not instantiable.
    private LlamaCppVersion() {}
}
