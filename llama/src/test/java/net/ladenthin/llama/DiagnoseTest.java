/*
 * SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0
 */
package net.ladenthin.llama;

import static org.junit.jupiter.api.Assertions.*;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.io.UnsupportedEncodingException;
import java.nio.charset.StandardCharsets;
import net.ladenthin.llama.value.LlamaCppVersion;
import org.junit.jupiter.api.Test;

/**
 * The shipped diagnostic entry point.
 *
 * <p>These tests need no native library and no model, which is the point: the report exists for the
 * case where the library does <em>not</em> load, so it has to be exercised in exactly that state.
 * The suite therefore runs both ways -- with natives on the classpath (CI, or after a local cmake
 * build) and without (a plain checkout) -- and every assertion here holds for both.
 *
 * <p>{@code Diagnose.main} is never called here: on a checkout without natives it ends by throwing,
 * which is the whole point of it (the process has to exit non-zero), but says nothing a test of the
 * report wants. {@link Diagnose#report(PrintStream)} is the seam that exists for that.
 *
 * <p>The pure helpers live in {@code loader.DiagnosticReport} and are covered by its own test.
 */
@ClaudeGenerated(
        purpose = "Verify the shipped diagnostic entry point without a native library: report() never "
                + "throws and always states exactly one of the two load outcomes (the property "
                + "the whole class exists for -- a diagnostic that fails while reporting a "
                + "failure is worse than none); it names the pinned llama.cpp version, the "
                + "JVM's os.name/os.arch and the resolved platform directory; and it either "
                + "lists the natives directories of this platform with the llama.cpp tag of "
                + "each jllama-build.txt or states that there is none, never staying silent.")
public class DiagnoseTest {

    /**
     * Runs the full report into a buffer.
     *
     * @return the report text
     */
    private static String report() {
        ByteArrayOutputStream buffer = new ByteArrayOutputStream();
        try (PrintStream out = new PrintStream(buffer, true, StandardCharsets.UTF_8.name())) {
            Diagnose.report(out);
        } catch (UnsupportedEncodingException e) {
            throw new AssertionError("every JVM is required to support UTF-8", e);
        }
        return new String(buffer.toByteArray(), StandardCharsets.UTF_8);
    }

    /**
     * The property this class exists for: the report has to survive the state it describes. A
     * diagnostic that throws while reporting a failure is worse than no diagnostic at all, so this
     * is the assertion that matters most -- the others only check that the content is useful.
     */
    @Test
    public void theReportNeverThrowsWhetherTheNativeLibraryLoadsOrNot() {
        String text = report();
        assertFalse(text.isEmpty(), "the report must not be empty");
        // Exactly one of the two outcomes, and never neither: the load section always says what
        // happened, so a reader is not left guessing whether it even got that far.
        boolean ok = text.contains("load:               ok");
        boolean failed = text.contains("load:               FAILED");
        assertTrue(ok || failed, "the report must state the load outcome, got:\n" + text);
        assertNotEquals(ok, failed, "the load outcome must be exactly one of ok and FAILED");
    }

    /** The platform half is what whoever reports a problem is asked for first. */
    @Test
    public void theReportNamesThePinnedVersionTheJvmAndTheResolvedPlatform() {
        String text = report();
        assertTrue(text.contains(LlamaCppVersion.LLAMA_CPP_VERSION), "the pinned version must appear");
        assertTrue(text.contains(System.getProperty("os.name")), "os.name must appear verbatim");
        assertTrue(text.contains(System.getProperty("os.arch")), "os.arch must appear verbatim");
        assertTrue(text.contains("resolved platform:"), "the derived platform directory must appear");
    }

    /**
     * Either there are natives for this platform and each is named with the llama.cpp tag of its
     * {@code jllama-build.txt} -- which is how a mismatched set becomes visible -- or the report says
     * there are none. Both are verdicts; saying nothing would not be.
     */
    @Test
    public void theReportEitherListsTheNativesWithTheirBuildTagOrSaysThereAreNone() {
        String text = report();
        boolean none = text.contains("(none -- no natives jar for this platform is on the classpath)");
        boolean listed = text.contains("llama.cpp=");
        assertTrue(none || listed, "the natives section must reach a verdict, got:\n" + text);
        assertNotEquals(none, listed, "it cannot both list a natives directory and report none");
    }
}
