// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import java.util.ArrayList;
import java.util.List;

/**
 * The classes jar's {@code Main-Class}: dispatches to one of the two server modes based on a single
 * selector flag (started from Maven Central with
 * {@code jbang --deps net.ladenthin:llama:<v>:<natives classifier> net.ladenthin:llama:<v> …}, or with
 * {@code java -cp} over the classes jar, the natives jars and the dependencies). With {@value #OPENAI_COMPAT_FLAG} present it runs {@link OpenAiCompatServer} (the
 * Java-transport, OpenAI-compatible JSON API); without it, {@link NativeServer} (the full native
 * llama.cpp server with embedded WebUI, the default).
 *
 * <p>The dispatch uses a single primitive, {@link #withoutFlag(String[], String)}: it strips the
 * selector from the arguments (the flag is not a llama.cpp flag, and {@code llama_server} rejects
 * unknown flags), and the mode is chosen purely by whether that shortened the list — present iff the
 * result is smaller. Every other argument is forwarded verbatim.</p>
 *
 * <p><strong>Flag sets differ.</strong> {@link NativeServer} forwards <em>every</em> llama-server
 * flag to {@code llama_server}, whereas {@link OpenAiCompatServer}'s CLI ({@link OpenAiServerCli})
 * accepts a curated subset and rejects unknown flags — so native-only flags (e.g. {@code --ui},
 * {@code -fa}) cannot be combined with {@value #OPENAI_COMPAT_FLAG}.</p>
 *
 * <p>Both underlying mains remain directly runnable by class name via {@code java -cp}; this
 * launcher is purely a convenience so a single {@code Main-Class} covers both.</p>
 */
public final class ServerLauncher {

    /**
     * Selector flag: when present, run {@link OpenAiCompatServer} instead of the default
     * {@link NativeServer}.
     *
     * <p>Namespaced with the {@code jllama} prefix (this project's native-library name) so it can
     * never collide with a current or future llama.cpp / llama-server flag — upstream owns the
     * {@code --*} space, this launcher owns {@code --jllama-*}. The launcher strips it before
     * forwarding, so it never reaches {@code llama_server} (which rejects unknown flags).</p>
     */
    public static final String OPENAI_COMPAT_FLAG = "--jllama-openai-compat";

    private ServerLauncher() {}

    /**
     * Dispatches to {@link OpenAiCompatServer#main(String[])} when {@value #OPENAI_COMPAT_FLAG} is
     * present (with that marker removed), otherwise to {@link NativeServer#main(String[])} with all
     * arguments forwarded unchanged. Selection is derived from whether stripping the flag shortened
     * the argument list.
     *
     * @param args the process arguments
     * @throws Exception if the selected server's {@code main} throws (it blocks until shutdown)
     */
    public static void main(String[] args) throws Exception {
        final String[] forwarded = withoutFlag(args, OPENAI_COMPAT_FLAG);
        if (forwarded.length != args.length) {
            OpenAiCompatServer.main(forwarded);
        } else {
            NativeServer.main(args);
        }
    }

    /**
     * Returns a copy of {@code args} with every occurrence of {@code flag} removed, preserving the
     * order of the remaining arguments. The result is shorter than {@code args} exactly when
     * {@code flag} was present — which is how {@link #main(String[])} selects the server mode.
     *
     * @param args the arguments
     * @param flag the flag token to strip
     * @return a new array without {@code flag}
     */
    static String[] withoutFlag(String[] args, String flag) {
        final List<String> filtered = new ArrayList<>(args.length);
        for (final String arg : args) {
            if (!flag.equals(arg)) {
                filtered.add(arg);
            }
        }
        return filtered.toArray(new String[0]);
    }
}
