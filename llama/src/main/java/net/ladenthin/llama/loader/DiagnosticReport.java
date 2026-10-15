/*
 * SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0
 */
package net.ladenthin.llama.loader;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.jspecify.annotations.Nullable;

/**
 * The half of the diagnostic report that only the loader can answer: which platform this JVM
 * resolves to, which natives directories are on the classpath and which llama.cpp build each of them
 * carries, and -- after a failed load -- the reason the operating system gave.
 *
 * <p>It lives here, next to {@link LlamaLoader}, because every one of those facts comes from a
 * package-private member of it. Java access is per package, not per module, so a class in
 * {@code net.ladenthin.llama} cannot read them; and this package is deliberately not exported, so a
 * public method here is still invisible to a consumer on the module path.
 *
 * <p>The counterpart is {@code net.ladenthin.llama.Diagnose}, which owns the parts this class must
 * not touch: the pinned-version constant lives in {@code value} and the load itself needs
 * {@code LlamaModel}, and the architecture rule allows neither direction out of this package.
 *
 * <p><b>Nothing here throws.</b> Each probe reports its own failure as a line of the report, because
 * a diagnostic that dies while describing a failure is worse than none.
 */
public final class DiagnosticReport {

    /**
     * The prefix of every system property this library reads, with the separating dot that
     * {@link LlamaSystemProperties#PREFIX} does not carry -- its own accessors supply it per suffix
     * ({@code ".backend"}). Getting that wrong is not cosmetic: the first version of this class
     * composed {@code PREFIX + "backend"} and read {@code net.ladenthin.llamabackend}, a name nothing
     * ever sets, so the report would have claimed no filter was in effect while one was.
     */
    private static final String PROPERTY_PREFIX = LlamaSystemProperties.PREFIX + ".";

    /**
     * Read through the same accessors the loader uses, so the report cannot disagree with it about
     * which property name carries which setting.
     */
    private static final LlamaSystemProperties PROPERTIES = new LlamaSystemProperties();

    /** The longest backend directory name, for aligning the columns of a pasted report. */
    private static final int NAME_WIDTH = "openvino".length();

    private DiagnosticReport() {
        // Not instantiable: a report builder, not an object.
    }

    /**
     * The JVM, the platform as {@link OSInfo} derives it, and the {@code net.ladenthin.llama.*}
     * properties that are set. Needs no native library.
     *
     * @param out where to write
     */
    public static void environment(PrintStream out) {
        out.println("java:               " + property("java.version") + "  (" + property("java.vendor") + ")");
        out.println("os.name / os.arch:  " + property("os.name") + " / " + property("os.arch"));
        out.println("resolved platform:  " + call(OSInfo::getNativeLibFolderPathForCurrentOS));
        out.println("  OSInfo os/arch:   " + call(OSInfo::getOSName) + " / " + call(OSInfo::getArchName));
        out.println("  android / musl:   "
                + call(() -> String.valueOf(OSInfo.isAndroid()))
                + " / "
                + call(() -> String.valueOf(OSInfo.isMusl())));
        Map<String, String> properties = llamaProperties();
        if (properties.isEmpty()) {
            out.println("properties:         none of " + PROPERTY_PREFIX + "* is set");
        } else {
            out.println("properties:");
            for (Map.Entry<String, String> entry : properties.entrySet()) {
                out.println("  " + entry.getKey() + " = " + entry.getValue());
            }
        }
    }

    /**
     * What the classpath offers: the library backend directory of this platform and every GPU module
     * directory, each with the jar it comes from and the llama.cpp tag it was built from -- which is
     * what a mismatched set looks like, and the one thing a consumer cannot otherwise see.
     *
     * @param out where to write
     */
    public static void natives(PrintStream out) {
        String base = LlamaLoader.NATIVE_RESOURCE_BASE + "/" + call(OSInfo::getNativeLibFolderPathForCurrentOS);
        out.println("natives on the classpath, below " + base);
        int found = 0;
        for (String backend : LlamaLoader.LIBRARY_BACKENDS) {
            if (describe(out, base, backend, "library")) {
                found++;
            }
        }
        List<String> modules = new ArrayList<>();
        for (String backend : LlamaLoader.MODULE_BACKENDS) {
            if (describe(out, base, backend, "module ")) {
                found++;
                modules.add(backend);
            }
        }
        if (found == 0) {
            out.println("  (none -- no natives jar for this platform is on the classpath)");
        }
        String filter = PROPERTIES.getBackend();
        out.println("backend property:   " + (filter == null ? "unset (every module is used)" : filter));
        for (String backend : modules) {
            if (LlamaLoader.OPT_IN_BACKENDS.contains(backend) && !namedBy(filter, backend)) {
                out.println("  note: module '"
                        + backend
                        + "' is opt-in and is NOT used -- name it in "
                        + PROPERTY_PREFIX
                        + "backend to enable it");
            }
        }
    }

    /**
     * The reason a failed {@code System.load} gave, which is the single most useful line of the
     * report: the library is usually present and one of its dependencies is not.
     *
     * @return the reason, or {@code null} when no load has failed
     */
    public static @Nullable String lastLoadFailure() {
        return LlamaLoader.lastLoadFailure;
    }

    /**
     * Whether a comma-separated backend filter names a backend, matched <b>exactly the way
     * {@code LlamaLoader.selectModules} matches it</b>: split on commas, each entry trimmed, then
     * compared with {@code equals}.
     *
     * <p>That mirroring is the point, and the first version of this method got it wrong in a way
     * worth keeping on record: it compared case-insensitively, which is more lenient than the loader.
     * The loader fails loud on a name it does not know, so {@code OpenVino} makes the load throw --
     * while a lenient report would have called that module enabled. A diagnostic may not be kinder
     * than the code it describes; being a line more cautious costs nothing, being a line more
     * generous is a wrong answer.
     *
     * @param filter  the property value, or {@code null} when unset
     * @param backend the backend name
     * @return {@code true} when the filter names it, by the loader's own comparison
     */
    public static boolean namedBy(@Nullable String filter, String backend) {
        if (filter == null) {
            return false;
        }
        // The same split the loader uses, with the same suppression and for the same reason: the
        // "trailing empties dropped" quirk is benign because an empty entry names no backend.
        @SuppressWarnings("StringSplitter")
        final String[] entries = filter.split(",");
        for (String entry : entries) {
            if (entry.trim().equals(backend)) {
                return true;
            }
        }
        return false;
    }

    /**
     * Every {@code net.ladenthin.llama.*} system property that is set, sorted by name.
     *
     * @return the properties, possibly empty
     */
    public static Map<String, String> llamaProperties() {
        List<String> names = new ArrayList<>();
        try {
            for (String name : System.getProperties().stringPropertyNames()) {
                if (name.startsWith(PROPERTY_PREFIX)) {
                    names.add(name);
                }
            }
        } catch (SecurityException e) {
            Map<String, String> refused = new LinkedHashMap<>(1);
            refused.put("(properties unreadable)", e.toString());
            return refused;
        }
        Collections.sort(names);
        Map<String, String> found = new LinkedHashMap<>(names.size());
        for (String name : names) {
            found.put(name, String.valueOf(System.getProperty(name)));
        }
        return found;
    }

    /**
     * One natives directory, if it is on the classpath at all.
     *
     * @param out     where to write
     * @param base    the platform's resource prefix
     * @param backend the backend directory name
     * @param kind    {@code library} or {@code module}, for the output
     * @return {@code true} when the directory is present
     */
    private static boolean describe(PrintStream out, String base, String backend, String kind) {
        String name = base + "/" + backend + "/" + LlamaLoader.BACKEND_BUILD_FILE;
        URL url = resourceOrNull(name);
        if (url == null) {
            return false;
        }
        out.println("  " + kind + " " + pad(backend) + " " + buildOf(name) + "  " + url);
        return true;
    }

    /**
     * What a {@code jllama-build.txt} says, read through the loader's own parser so the two cannot
     * disagree on the format: the llama.cpp tag first (every build names it), then the provenance
     * keys a build since the delivery roadmap writes -- the upstream commit, this project's version
     * and commit, the carried patches, compiler, toolkit and signature state -- each only when present,
     * so a directory of an older build prints its tag alone.
     *
     * @param name the build file's resource name
     * @return {@code llama.cpp=<tag> ...}, or a short description of why it could not be read
     */
    static String buildOf(String name) {
        try (InputStream in = LlamaLoader.resourceAsStream(name)) {
            if (in == null) {
                return "llama.cpp=(vanished)";
            }
            BufferedReader reader = new BufferedReader(new InputStreamReader(in, StandardCharsets.UTF_8));
            return summarize(LlamaLoader.parseBuild(reader));
        } catch (IOException | RuntimeException e) {
            return "llama.cpp=(unreadable: " + e + ")";
        }
    }

    /**
     * One line of the facts a parsed build file carries, in a fixed order, commits shortened to
     * their first eight characters the way {@code git log --oneline} prints them.
     *
     * @param build the parsed build file
     * @return the summary, starting with {@code llama.cpp=}
     */
    static String summarize(Map<String, String> build) {
        String tag = build.get(LlamaLoader.BUILD_KEY_LLAMA_CPP);
        StringBuilder line = new StringBuilder("llama.cpp=").append(tag != null ? tag : "(not named)");
        appendIfPresent(line, build, LlamaLoader.BUILD_KEY_LLAMA_CPP_COMMIT, true);
        appendIfPresent(line, build, LlamaLoader.BUILD_KEY_JLLAMA_VERSION, false);
        appendIfPresent(line, build, LlamaLoader.BUILD_KEY_JLLAMA_COMMIT, true);
        appendIfPresent(line, build, LlamaLoader.BUILD_KEY_PATCHES, false);
        appendIfPresent(line, build, LlamaLoader.BUILD_KEY_COMPILER, false);
        appendIfPresent(line, build, LlamaLoader.BUILD_KEY_TOOLKIT, false);
        appendIfPresent(line, build, LlamaLoader.BUILD_KEY_SIGNED, false);
        return line.toString();
    }

    private static void appendIfPresent(StringBuilder line, Map<String, String> build, String key, boolean commit) {
        String value = build.get(key);
        if (value == null) {
            return;
        }
        line.append(' ').append(key).append('=');
        Matcher full = commit ? FULL_COMMIT.matcher(value) : null;
        if (full != null && full.matches()) {
            String suffix = full.group(2);
            line.append(full.group(1)).append(suffix != null ? suffix : "");
        } else {
            line.append(value);
        }
    }

    /** A full 40-hex commit, optionally {@code -dirty}: group 1 is its short form, group 2 the suffix. */
    private static final Pattern FULL_COMMIT = Pattern.compile("([0-9a-f]{8})[0-9a-f]{32}(-dirty)?");

    /**
     * Looks a resource up the way the loader does, never throwing.
     *
     * @param name the resource name
     * @return the URL, or {@code null} when absent or unreadable
     */
    private static @Nullable URL resourceOrNull(String name) {
        try {
            return LlamaLoader.resource(name);
        } catch (RuntimeException | LinkageError e) {
            return null;
        }
    }

    /**
     * Pads a backend name so the columns line up for whoever reads a pasted report.
     *
     * @param backend the name
     * @return the name, padded to the width of the longest backend name
     */
    private static String pad(String backend) {
        StringBuilder padded = new StringBuilder(backend);
        while (padded.length() < NAME_WIDTH) {
            padded.append(' ');
        }
        return padded.toString();
    }

    /**
     * A system property, or a placeholder -- never an exception.
     *
     * @param name the property name
     * @return the value, or {@code (unset)}
     */
    private static String property(String name) {
        String value = System.getProperty(name);
        return value != null ? value : "(unset)";
    }

    /**
     * Calls a probe that may throw, so one unhappy line cannot take the report down.
     *
     * @param probe the probe
     * @return its value, or a short description of the failure
     */
    private static String call(Probe probe) {
        try {
            String value = probe.get();
            return value != null ? value : "(null)";
        } catch (RuntimeException | LinkageError e) {
            return "(failed: " + e + ")";
        }
    }

    /** A probe that returns one value of the report and is allowed to fail. */
    @FunctionalInterface
    interface Probe {

        /**
         * Reads one value.
         *
         * @return the value, or {@code null}
         */
        @Nullable
        String get();
    }
}
