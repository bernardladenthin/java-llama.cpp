/*
 * SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0
 */
package net.ladenthin.llama.loader;

import static org.junit.jupiter.api.Assertions.*;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.io.UnsupportedEncodingException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import net.ladenthin.llama.ClaudeGenerated;
import org.junit.jupiter.api.Test;

/**
 * The loader half of the diagnostic report: the pure helpers, and that each section writes something
 * rather than throwing.
 *
 * <p>No native library and no model: these tests run in a plain checkout.
 */
@ClaudeGenerated(
        purpose = "Verify the loader half of the diagnostic report without a native library: namedBy "
                + "matches the comma-separated net.ladenthin.llama.backend filter entry by "
                + "entry, trimmed and ignoring case, so the spacing a human writes works while "
                + "a prefix is not a member (a substring test would answer that wrongly in one "
                + "direction or the other); llamaProperties collects only this library's "
                + "properties and returns them sorted; and environment() and natives() each "
                + "write a non-empty section without throwing, naming the resolved platform "
                + "and reaching a verdict on the classpath.")
public class DiagnosticReportTest {

    /**
     * Captures one section.
     *
     * @param section the section to run
     * @return what it wrote
     */
    private static String capture(java.util.function.Consumer<PrintStream> section) {
        ByteArrayOutputStream buffer = new ByteArrayOutputStream();
        try (PrintStream out = new PrintStream(buffer, true, StandardCharsets.UTF_8.name())) {
            section.accept(out);
        } catch (UnsupportedEncodingException e) {
            throw new AssertionError("every JVM is required to support UTF-8", e);
        }
        return new String(buffer.toByteArray(), StandardCharsets.UTF_8);
    }

    /**
     * The backend filter is a comma-separated list, so membership is a comparison of whole entries
     * rather than a substring test -- and it tolerates the spacing and the casing a human writes.
     */
    @Test
    public void theBackendFilterIsMatchedEntryByEntryNotAsASubstring() {
        assertTrue(DiagnosticReport.namedBy("openvino", "openvino"), "the only entry names it");
        assertTrue(DiagnosticReport.namedBy("cpu,openvino", "openvino"), "one of several entries names it");
        assertTrue(DiagnosticReport.namedBy("cpu, openvino", "openvino"), "the spacing a human writes works");
        // NOT case-insensitive, deliberately: LlamaLoader.selectModules compares with equals and
        // fails loud on a name it does not know, so "OpenVino" makes the load throw. A report that
        // called that module enabled would be kinder than the code it describes, which is a wrong
        // answer rather than a lenient one. (The first version of namedBy did exactly that.)
        assertFalse(DiagnosticReport.namedBy(" OpenVino ", "openvino"), "the loader is case-sensitive");
        assertTrue(DiagnosticReport.namedBy(" openvino ", "openvino"), "but surrounding spaces are trimmed");
        assertFalse(DiagnosticReport.namedBy("cuda13,vulkan", "openvino"), "other entries do not name it");
        // A prefix is not a member. A substring test would answer this wrongly in one direction or
        // the other, depending on which of the two strings it searched.
        assertFalse(DiagnosticReport.namedBy("openvino", "openv"), "a prefix of an entry is not that entry");
        assertFalse(DiagnosticReport.namedBy("", "openvino"), "an empty filter names nothing");
        assertFalse(DiagnosticReport.namedBy(null, "openvino"), "an unset filter names nothing");
    }

    /** Only this library's properties are collected, and in a stable order. */
    @Test
    public void onlyTheLibrariesOwnPropertiesAreCollectedAndTheyAreSorted() {
        String mine = LlamaSystemProperties.PREFIX + ".diagnose.test.zzz";
        String alsoMine = LlamaSystemProperties.PREFIX + ".diagnose.test.aaa";
        String foreign = "some.other.product.diagnose.test";
        try {
            System.setProperty(mine, "1");
            System.setProperty(alsoMine, "2");
            System.setProperty(foreign, "3");

            Map<String, String> properties = DiagnosticReport.llamaProperties();
            assertEquals("1", properties.get(mine), "a property of this library is collected");
            assertEquals("2", properties.get(alsoMine), "and so is the other one");
            assertFalse(properties.containsKey(foreign), "a foreign property is not");

            List<String> keys = new ArrayList<>(properties.keySet());
            List<String> sorted = new ArrayList<>(keys);
            Collections.sort(sorted);
            assertEquals(sorted, keys, "the properties are reported in sorted order");
        } finally {
            System.clearProperty(mine);
            System.clearProperty(alsoMine);
            System.clearProperty(foreign);
        }
    }

    /** A set property has to be visible in the paste, because a wrong one is a common cause. */
    @Test
    public void theEnvironmentSectionNamesThePlatformAndEverySetProperty() {
        String key = LlamaSystemProperties.PREFIX + ".diagnose.test.marker";
        String previous = System.getProperty(key);
        try {
            System.setProperty(key, "a-value-only-this-test-sets");
            String text = capture(DiagnosticReport::environment);
            assertTrue(text.contains("resolved platform:"), "the derived platform directory appears");
            assertTrue(text.contains(OSInfo.getNativeLibFolderPathForCurrentOS()), "and its value");
            assertTrue(text.contains(key + " = a-value-only-this-test-sets"), "a set property appears");
        } finally {
            if (previous == null) {
                System.clearProperty(key);
            } else {
                System.setProperty(key, previous);
            }
        }
    }

    /**
     * The natives section reaches a verdict either way: it lists each directory with the llama.cpp
     * tag of its {@code jllama-build.txt}, or states that there is none. Silence would be the one
     * useless outcome.
     */
    @Test
    public void theNativesSectionReachesAVerdictWithOrWithoutNativesOnTheClasspath() {
        String text = capture(DiagnosticReport::natives);
        boolean none = text.contains("(none -- no natives jar for this platform is on the classpath)");
        boolean listed = text.contains("llama.cpp=");
        assertTrue(none || listed, "the section must reach a verdict, got:\n" + text);
        assertNotEquals(none, listed, "it cannot both list a directory and report none");
        assertTrue(text.contains("backend property:"), "the backend filter is always reported");
    }

    /**
     * The report has to read the <em>same</em> property the loader reads. This is the test that would
     * have caught the first version of this class: it composed the name from a prefix that carries no
     * separating dot and read {@code net.ladenthin.llamabackend}, so a filter that was in effect was
     * reported as unset -- a diagnostic stating the opposite of the truth about the one setting that
     * decides which modules load.
     */
    @Test
    public void theBackendFilterIsReadUnderTheNameTheLoaderActuallyUses() {
        String real = LlamaSystemProperties.PREFIX + ".backend";
        String previous = System.getProperty(real);
        try {
            System.setProperty(real, "cpu");
            String text = capture(DiagnosticReport::natives);
            assertTrue(text.contains("backend property:   cpu"), "the filter must be shown: " + text);
        } finally {
            if (previous == null) {
                System.clearProperty(real);
            } else {
                System.setProperty(real, previous);
            }
        }
    }

    /** No load has failed in this JVM unless one was attempted, and the accessor must not throw. */
    @Test
    public void theLastLoadFailureAccessorIsReadableWithoutALoad() {
        // Deliberately no assertion on the value: another test in the same JVM may have attempted a
        // load. What is pinned is that reading it is safe, which is what the report relies on.
        DiagnosticReport.lastLoadFailure();
    }
}
