// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.atmosphere;

import java.util.Locale;

/**
 * Whether a tool call that changes something asks before it runs.
 *
 * <p>The default is {@link #MANUAL}: the model may read freely, but every write and every shell
 * command is confirmed on the console. {@link #AUTO} is what the {@code --auto} flag and the
 * {@code [a]} answer of a single approval prompt switch to — it stays on for the rest of the session
 * until {@code /mode manual} switches back.
 */
public enum ApprovalMode {

    /** Ask before every gated tool call. */
    MANUAL("⏸"),

    /** Run every tool call without asking. */
    AUTO("⏵⏵");

    private final String symbol;

    ApprovalMode(String symbol) {
        this.symbol = symbol;
    }

    /**
     * The lower-case name used on the console and in {@code /mode}.
     *
     * @return {@code "manual"} or {@code "auto"}
     */
    public String label() {
        return name().toLowerCase(Locale.ROOT);
    }

    /**
     * The glyph shown in front of the name on the status line.
     *
     * <p>Two transport symbols, the way the established terminal agents mark the same distinction:
     * {@code ⏸} for a session that stops at every gated call, {@code ⏵⏵} for one that runs through. The
     * name stays next to it — the glyph makes the mode findable at a glance, it does not replace the
     * word.
     *
     * @return {@code "⏸"} or {@code "⏵⏵"}
     */
    public String symbol() {
        return symbol;
    }

    /**
     * Symbol and name together, as the status line and {@code /mode} print them.
     *
     * @return e.g. {@code "⏸ manual"}
     */
    public String badge() {
        return symbol + " " + label();
    }

    /**
     * Parse a mode name as typed by the user.
     *
     * @param text the name, in any case, optionally surrounded by whitespace
     * @return the mode
     * @throws IllegalArgumentException when {@code text} names no mode
     */
    public static ApprovalMode parse(String text) {
        String normalized = text == null ? "" : text.trim().toLowerCase(Locale.ROOT);
        for (ApprovalMode mode : values()) {
            if (mode.label().equals(normalized)) {
                return mode;
            }
        }
        throw new IllegalArgumentException("Unknown mode: " + text + " (expected manual or auto)");
    }
}
