// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.atmosphere;

import java.util.Locale;

/**
 * The one-line status pinned below the answer: workspace, approval mode, context usage, tool count
 * and model id — the four things whose answer changes what the next request does.
 *
 * <p>Context usage is the <b>input</b> side of the last completed turn — the prompt the server had to
 * process, which is what fills the context window; the generated tokens of that turn are already part
 * of the next request's input. Claude Code's status line computes its percentage the same way.
 * The count comes from the server when it reports usage. llama.cpp only sends the trailing usage
 * chunk when the client asks for it ({@code stream_options.include_usage}) and Atmosphere's client
 * does not, so in practice the number is an estimate from the text length (four characters per
 * token, the usual rule of thumb) and is then marked with a {@code ~}. It is meant to answer "am I
 * close to the limit, should I /compact", not to be exact.
 *
 * <p>The size is known with {@code --model} (it is the
 * {@code --ctx-size} the agent loaded the model with) and with {@code --base-url} it comes from the
 * server's {@code /props}; when that lookup fails the line shows the token count alone instead of
 * inventing a denominator.
 */
public final class StatusLine {

    /** Above this many characters the workspace path is shortened to its last two segments. */
    private static final int MAX_PATH_CHARS = 40;

    /** The context size is unknown (no {@code /props}, and no in-process model). */
    public static final int UNKNOWN_CONTEXT = 0;

    private StatusLine() {}

    /**
     * Render the status line.
     *
     * @param workspace the directory the tools work in
     * @param mode the approval mode
     * @param inputTokens the input tokens of the last turn, or {@code 0} before the first one
     * @param estimated whether that number is an estimate rather than the server's own count
     * @param contextSize the context window in tokens, or {@link #UNKNOWN_CONTEXT}
     * @param tools how many tools are offered to the model
     * @param modelId the model id sent in every request
     * @return one line, without a trailing newline
     */
    public static String render(
            java.nio.file.Path workspace,
            ApprovalMode mode,
            long inputTokens,
            boolean estimated,
            int contextSize,
            int tools,
            String modelId) {
        return "[" + shorten(workspace) + " · " + mode.label() + " · " + context(inputTokens, estimated, contextSize)
                + " · " + tools + " tools · " + modelId + "]";
    }

    /**
     * The workspace path, shortened from the left when it would take over the line.
     *
     * <p>The tools work relative to this directory and the shell starts in it, so it belongs on the
     * line that is always visible — but a deep path would push everything else off the screen, so only
     * the last two segments survive, marked with a leading ellipsis.
     *
     * @param workspace the workspace directory
     * @return the path, or its tail
     */
    static String shorten(java.nio.file.Path workspace) {
        String full = workspace.toString();
        if (full.length() <= MAX_PATH_CHARS || workspace.getNameCount() < 2) {
            return full;
        }
        return "…" + workspace.getFileSystem().getSeparator()
                + workspace.subpath(workspace.getNameCount() - 2, workspace.getNameCount());
    }

    /**
     * The context part of the line on its own.
     *
     * @param inputTokens the input tokens of the last turn
     * @param estimated whether that number is an estimate (rendered with a leading {@code ~})
     * @param contextSize the context window in tokens, or {@link #UNKNOWN_CONTEXT}
     * @return e.g. {@code "ctx 1.2k/16k"}, or {@code "ctx 1.2k"} when the size is unknown
     */
    static String context(long inputTokens, boolean estimated, int contextSize) {
        // Two numbers in k, no percentage: everyone reads 12k/16k at a glance, and a percentage of a
        // number that is itself an estimate suggests a precision this does not have.
        String used = (estimated ? "~" : "") + abbreviate(inputTokens);
        return contextSize <= UNKNOWN_CONTEXT ? "ctx " + used : "ctx " + used + "/" + abbreviate(contextSize);
    }

    /**
     * A short token count: {@code 812}, {@code 1.2k}, {@code 16k}.
     *
     * @param tokens the count
     * @return the abbreviated form
     */
    static String abbreviate(long tokens) {
        if (tokens < 1000) {
            return Long.toString(tokens);
        }
        double thousands = tokens / 1000.0;
        // one decimal below 10k (1.2k), none above (16k) -- the decimal carries no information there
        return thousands < 10
                ? String.format(Locale.ROOT, "%.1fk", thousands)
                : String.format(Locale.ROOT, "%.0fk", thousands);
    }
}
