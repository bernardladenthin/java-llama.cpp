// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.atmosphere;

import java.util.Locale;
import java.util.function.Function;
import org.jspecify.annotations.Nullable;

/**
 * The handful of ANSI styles this agent uses, and the one decision of whether to emit them at all.
 *
 * <p>The decision is made once at startup, not per line: when colour is off every helper returns its
 * argument unchanged, so no other class branches on it. The order follows the conventions the
 * terminal ecosystem actually honours:
 *
 * <ol>
 *   <li>{@code CLICOLOR_FORCE=1} forces colour on, even when the output is piped (for {@code less -R}
 *       and CI logs);
 *   <li>{@code NO_COLOR} set to any non-empty value turns it off — that is the whole
 *       <a href="https://no-color.org/">NO_COLOR</a> specification;
 *   <li>{@code TERM=dumb} or {@code CLICOLOR=0} turns it off;
 *   <li>otherwise colour is on only when the output really is a terminal.
 * </ol>
 *
 * <p>The terminal test is {@code Console.isTerminal()} where it exists (JDK 22+) and
 * {@code System.console() != null} below that. The distinction matters: on JDK 22 to 24
 * {@code System.console()} also returns a console for redirected output, so the older test alone
 * would colour a file. Reflection keeps the code compiling and running on JDK 21.
 *
 * <p>Windows: Windows Terminal and the VS Code terminal process escape sequences without any setup.
 * The classic {@code conhost.exe} does not unless {@code HKCU\Console\VirtualTerminalLevel} is 1 —
 * enabling it from the process needs native code, which this agent deliberately does not use, so
 * there a user may see the raw sequences and can set {@code NO_COLOR=1}.
 */
public final class Ansi {

    private static final String RESET = "\u001b[0m";

    /** Never emits escape sequences. */
    public static final Ansi PLAIN = new Ansi(false);

    private final boolean enabled;

    private Ansi(boolean enabled) {
        this.enabled = enabled;
    }

    /**
     * Decide from the environment whether to use colour.
     *
     * @return a colouring or a plain instance
     */
    public static Ansi detect() {
        return detect(System::getenv, Ansi::consoleIsTerminal);
    }

    /**
     * The decision itself, with the environment injected so it can be tested.
     *
     * @param env reads an environment variable
     * @param isTerminal whether standard output is a terminal
     * @return a colouring or a plain instance
     */
    static Ansi detect(Function<String, @Nullable String> env, java.util.function.BooleanSupplier isTerminal) {
        if ("1".equals(trimmed(env.apply("CLICOLOR_FORCE")))) {
            return new Ansi(true);
        }
        String noColor = env.apply("NO_COLOR");
        if (noColor != null && !noColor.isEmpty()) {
            return PLAIN;
        }
        if ("dumb".equals(lower(env.apply("TERM"))) || "0".equals(trimmed(env.apply("CLICOLOR")))) {
            return PLAIN;
        }
        return isTerminal.getAsBoolean() ? new Ansi(true) : PLAIN;
    }

    private static boolean consoleIsTerminal() {
        java.io.Console console = System.console();
        if (console == null) {
            return false;
        }
        try {
            // JDK 22+: the only reliable "is a terminal" test; JDK 21 has no such method.
            return (boolean) java.io.Console.class.getMethod("isTerminal").invoke(console);
        } catch (ReflectiveOperationException | RuntimeException e) {
            return true;
        }
    }

    private static @Nullable String trimmed(@Nullable String value) {
        return value == null ? null : value.trim();
    }

    private static @Nullable String lower(@Nullable String value) {
        return value == null ? null : value.trim().toLowerCase(Locale.ROOT);
    }

    /**
     * Whether escape sequences are emitted.
     *
     * @return {@code true} when styling is on
     */
    public boolean isEnabled() {
        return enabled;
    }

    private String style(String code, String text) {
        return enabled ? "\u001b[" + code + "m" + text + RESET : text;
    }

    /**
     * Bold text.
     *
     * @param text the text
     * @return the styled text
     */
    public String bold(String text) {
        return style("1", text);
    }

    /**
     * Dimmed text, for secondary output such as tool results.
     *
     * @param text the text
     * @return the styled text
     */
    public String dim(String text) {
        return style("2", text);
    }

    /**
     * Cyan text, for code and tool names.
     *
     * @param text the text
     * @return the styled text
     */
    public String cyan(String text) {
        return style("36", text);
    }

    /**
     * Green text, for the marker of a running tool.
     *
     * @param text the text
     * @return the styled text
     */
    public String green(String text) {
        return style("32", text);
    }

    /**
     * Yellow text, for questions that need an answer.
     *
     * @param text the text
     * @return the styled text
     */
    public String yellow(String text) {
        return style("33", text);
    }

    /**
     * Red text, for errors and denials.
     *
     * @param text the text
     * @return the styled text
     */
    public String red(String text) {
        return style("31", text);
    }
}
