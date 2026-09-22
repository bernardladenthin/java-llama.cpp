// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.atmosphere;

import java.io.PrintStream;

/**
 * Renders the streamed answer as it arrives: just enough Markdown to make it readable.
 *
 * <p><b>Append-only, one line at a time.</b> Tokens are buffered until a line is complete, then that
 * line is written and never touched again. Redrawing the answer on every token — what the Ink and
 * Bubble Tea based clients do — is what produces their overdraw and truncation bugs, and it needs
 * cursor control that breaks as soon as the output is piped into a file. The price here is that a
 * line appears only once it ends.
 *
 * <p>Handled: fenced code blocks (one bit of state), ATX headings, bullet markers, and inline
 * {@code **bold**} / {@code `code`}. Italics are deliberately not handled — a lone {@code *} is more
 * often a glob or a multiplication than emphasis, and getting that wrong garbles ordinary text.
 * Everything else is passed through unchanged, which is what a terminal wants anyway.
 *
 * <p>Only the console sees this; {@link ConsoleSession} keeps the raw text for the history, so
 * nothing that goes back to the model is affected.
 */
public final class MarkdownConsole {

    private final PrintStream out;
    private final Ansi ansi;
    private final StringBuilder pending = new StringBuilder();
    private boolean inFence;

    /**
     * Create a renderer.
     *
     * @param out where the rendered text goes
     * @param ansi the styles (a plain instance writes the text unchanged)
     */
    public MarkdownConsole(PrintStream out, Ansi ansi) {
        this.out = out;
        this.ansi = ansi;
    }

    /**
     * Take the next streamed chunk; every complete line in it is rendered and written.
     *
     * @param chunk the chunk as it arrived, possibly a fragment of a line
     */
    public void append(String chunk) {
        pending.append(chunk);
        int newline;
        while ((newline = pending.indexOf("\n")) >= 0) {
            String line = pending.substring(0, newline);
            pending.delete(0, newline + 1);
            out.println(render(line.endsWith("\r") ? line.substring(0, line.length() - 1) : line));
        }
        out.flush();
    }

    /** Write what is left of an unfinished line, e.g. an answer that does not end with a newline. */
    public void flush() {
        if (pending.length() > 0) {
            out.println(render(pending.toString()));
            pending.setLength(0);
        }
        inFence = false;
        out.flush();
    }

    /**
     * Render one complete line.
     *
     * @param line the line without its terminator
     * @return the line with escape sequences, or unchanged when styling is off
     */
    String render(String line) {
        String content = line.stripLeading();
        String indent = line.substring(0, line.length() - content.length());
        if (content.startsWith("```") || content.startsWith("~~~")) {
            inFence = !inFence;
            return ansi.dim(line);
        }
        if (inFence) {
            return ansi.cyan(line);
        }
        int heading = 0;
        while (heading < content.length() && content.charAt(heading) == '#') {
            heading++;
        }
        if (heading > 0 && heading <= 6 && content.startsWith("# ", heading - 1)) {
            return indent + ansi.bold(inline(content.substring(heading + 1).strip()));
        }
        if (content.length() > 2 && "-*+".indexOf(content.charAt(0)) >= 0 && content.charAt(1) == ' ') {
            return indent + ansi.cyan("•") + " " + inline(content.substring(2));
        }
        return indent + inline(content);
    }

    /**
     * Style {@code **bold**} and {@code `code`} inside one line.
     *
     * @param text the line content
     * @return the styled content
     */
    String inline(String text) {
        StringBuilder result = new StringBuilder(text.length());
        int index = 0;
        while (index < text.length()) {
            int code = text.indexOf('`', index);
            int bold = text.indexOf("**", index);
            boolean codeFirst = code >= 0 && (bold < 0 || code < bold);
            int start = codeFirst ? code : bold;
            if (start < 0) {
                break;
            }
            String marker = codeFirst ? "`" : "**";
            int end = text.indexOf(marker, start + marker.length());
            if (end < 0) {
                break; // an unclosed marker: leave the rest as typed
            }
            String span = text.substring(start + marker.length(), end);
            result.append(text, index, start).append(codeFirst ? ansi.cyan(span) : ansi.bold(span));
            index = end + marker.length();
        }
        return result.append(text.substring(index)).toString();
    }
}
