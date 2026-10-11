/*
 * SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0
 */
package net.ladenthin.llama;

import java.io.PrintStream;
import net.ladenthin.llama.loader.DiagnosticReport;
import net.ladenthin.llama.loader.LlamaLoader;
import net.ladenthin.llama.value.LlamaCppVersion;

/**
 * Prints what this library sees of its own installation, so that "it does not load" becomes one
 * paste instead of a conversation.
 *
 * <p>Every question a failed load actually raises is answerable from inside the process: which
 * platform the JVM reports, which natives jars are on the classpath and which llama.cpp build each
 * one carries, which library backend and GPU modules the loader selects, and -- when the load fails
 * -- the operating system's own reason for refusing the library. This class prints exactly that.
 *
 * <pre>{@code
 * java -cp 'libs/*' net.ladenthin.llama.Diagnose
 * }</pre>
 *
 * <p>The report is split across two classes for a reason worth knowing before moving anything:
 * every platform and classpath fact comes from a package-private member of
 * {@code LlamaLoader}, and Java access is per package rather than per module -- so that half lives
 * in {@code net.ladenthin.llama.loader.DiagnosticReport}. The main class has to be here instead,
 * because the loader's package is deliberately not exported and a consumer on the module path could
 * not run a main class in it. The architecture rule allows this direction and not the other, which
 * is also why the pinned-version constant and the load itself are read here rather than there.
 *
 * <p><b>No probe throws.</b> A diagnostic that fails while reporting a failure is worse than none, so
 * each one reports its own problem as a line of the output instead.
 *
 * <p>The one exception is deliberate and is at the very end: when the library did not load,
 * {@link #main(String[])} throws after printing the whole report, so the process exits non-zero and a
 * script can tell. It is not {@code System.exit} -- the project's architecture rule forbids that in
 * every class here and says to throw instead, which happens to be the better shape anyway: a caller
 * that invokes {@code main} programmatically keeps its JVM, and {@link #report(PrintStream)} returns
 * the outcome as a value for anyone who wants neither.
 */
public final class Diagnose {

    /** The section separator, narrow enough to survive a paste into an issue comment. */
    private static final String RULE = "--------------------------------------------------------";

    private Diagnose() {
        // Not instantiable: a main class, not an object.
    }

    /**
     * Prints the report to {@code System.out}, and throws afterwards when the native library did not
     * load so that the process exits non-zero.
     *
     * @param args ignored, so that this is an ordinary main class
     * @throws UnsatisfiedLinkError when the native library did not load -- after the report, never
     *     instead of it
     */
    public static void main(String[] args) {
        boolean loaded = report(System.out);
        System.out.flush();
        if (!loaded) {
            throw new UnsatisfiedLinkError("the native library did not load -- see the report above");
        }
    }

    /**
     * Writes the whole report, including an attempt to load the native library. This is the seam a
     * test uses, and the entry point for an embedding caller that wants the report without the
     * throw {@link #main(String[])} ends in.
     *
     * @param out where to write
     * @return {@code true} when the native library loaded
     */
    public static boolean report(PrintStream out) {
        out.println("jllama diagnose");
        out.println(RULE);
        out.println("llama.cpp pinned:   " + LlamaCppVersion.LLAMA_CPP_VERSION);
        DiagnosticReport.environment(out);
        out.println(RULE);
        DiagnosticReport.natives(out);
        out.println(RULE);
        return load(out);
    }

    /**
     * Loads the library and reports what came back -- or, on failure, the reason the operating system
     * gave, which is the single most useful line of the report.
     *
     * @param out where to write
     * @return {@code true} when the library loaded
     */
    private static boolean load(PrintStream out) {
        try {
            LlamaLoader.initialize();
        } catch (UnsatisfiedLinkError | RuntimeException e) {
            out.println("load:               FAILED");
            out.println("  " + e);
            String reason = DiagnosticReport.lastLoadFailure();
            if (reason != null) {
                out.println("  the operating system's reason: " + reason);
            }
            return false;
        }
        out.println("load:               ok");
        try {
            String linked = LlamaModel.getLlamaCppBuildInfo();
            out.println("linked llama.cpp:   " + linked);
            if (!linked.startsWith(LlamaCppVersion.LLAMA_CPP_VERSION)) {
                out.println("  WARNING: the loaded library is not the pinned "
                        + LlamaCppVersion.LLAMA_CPP_VERSION
                        + " -- a stale extraction directory, or a natives jar of another release on"
                        + " the classpath");
            }
        } catch (UnsatisfiedLinkError | RuntimeException e) {
            out.println("linked llama.cpp:   could not be read: " + e);
        }
        out.println("The loader printed the directory it used, and ggml its backend registry and");
        out.println("devices, on stderr above -- include those lines when reporting a problem.");
        return true;
    }
}
