// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.atmosphere;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.PrintStream;
import java.util.Locale;
import org.jspecify.annotations.Nullable;

/**
 * An {@link AgentTerminal} over a plain stream pair: what a one-shot run, a piped session and the
 * tests get.
 *
 * <p>No cursor control at all — the status line is printed as an ordinary line before the prompt and
 * scrolls away like everything else, and an answer needs Enter. That is the point: this
 * implementation stays correct when the output is a file.
 */
public final class PlainTerminal implements AgentTerminal {

    private final PrintStream out;
    private final @Nullable BufferedReader in;
    private final Ansi ansi;

    /**
     * Create a plain console.
     *
     * @param out where output goes
     * @param in where input is read, or {@code null} when there is none (one-shot runs)
     * @param ansi the styles
     */
    public PlainTerminal(PrintStream out, @Nullable BufferedReader in, Ansi ansi) {
        this.out = out;
        this.in = in;
        this.ansi = ansi;
    }

    @Override
    public void line(String text) {
        out.println(text);
        out.flush();
    }

    @Override
    public @Nullable String readLine(String prompt) {
        out.print(prompt);
        out.flush();
        return read();
    }

    @Override
    public @Nullable String readKey(String prompt) {
        out.print(prompt);
        out.flush();
        String answer = read();
        return answer == null ? null : answer.trim().toLowerCase(Locale.ROOT);
    }

    @Override
    public void status(String text) {
        // Nothing can be pinned on a plain stream, and the caller already prints the status line above
        // the prompt. Dropping it here is what keeps a piped session free of half-drawn spinner lines.
    }

    @Override
    public Ansi ansi() {
        return ansi;
    }

    @Override
    public void close() {
        out.flush();
    }

    private @Nullable String read() {
        try {
            return in == null ? null : in.readLine();
        } catch (IOException e) {
            return null;
        }
    }
}
