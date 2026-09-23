// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.atmosphere;

import java.io.IOException;
import java.util.List;
import java.util.Locale;
import org.jline.reader.EndOfFileException;
import org.jline.reader.LineReader;
import org.jline.reader.LineReaderBuilder;
import org.jline.reader.UserInterruptException;
import org.jline.reader.impl.completer.StringsCompleter;
import org.jline.terminal.Attributes;
import org.jline.terminal.Terminal;
import org.jline.terminal.TerminalBuilder;
import org.jline.utils.AttributedString;
import org.jline.utils.AttributedStyle;
import org.jline.utils.Status;
import org.jspecify.annotations.Nullable;

/**
 * An {@link AgentTerminal} on a real terminal, via JLine: line editing and history at the prompt, tab
 * completion of the commands, a status line pinned to the bottom of the window, and single-key
 * answers.
 *
 * <p>Streamed output goes through {@link LineReader#printAbove(String)}, which scrolls it in above
 * the prompt while the bottom block stays where it is — the one thing a plain {@code println} cannot
 * do. Nothing is ever redrawn above that block, so the scrollback stays exactly as it was written.
 *
 * <p>{@link #open} returns {@code null} instead of throwing when there is no usable terminal (piped
 * input, a "dumb" terminal, a missing native provider); the caller then uses {@link PlainTerminal}.
 * Ctrl-C at the prompt clears the line and returns an empty one — it does not end the session; Ctrl-D
 * ends input like end-of-file.
 */
public final class JLineTerminal implements AgentTerminal {

    private final Terminal terminal;
    private final LineReader reader;
    private final Status status;
    private final Ansi ansi;
    private volatile boolean reading;

    private JLineTerminal(Terminal terminal, LineReader reader, Status status, Ansi ansi) {
        this.terminal = terminal;
        this.reader = reader;
        this.status = status;
        this.ansi = ansi;
    }

    /**
     * Open the system terminal.
     *
     * @param completions the words tab completes, e.g. the command names
     * @return the terminal, or {@code null} when this is not an interactive terminal
     */
    public static @Nullable JLineTerminal open(List<String> completions) {
        try {
            Terminal terminal = TerminalBuilder.builder().system(true).build();
            if (terminal.getType().startsWith(Terminal.TYPE_DUMB)) {
                terminal.close();
                return null;
            }
            LineReader reader = LineReaderBuilder.builder()
                    .terminal(terminal)
                    .completer(new StringsCompleter(completions))
                    .build();
            return new JLineTerminal(terminal, reader, Status.getStatus(terminal), Ansi.detect());
        } catch (IOException | RuntimeException e) {
            // No terminal, no native provider, a restricted environment: the plain console still works.
            return null;
        }
    }

    @Override
    public void line(String text) {
        if (text.indexOf('\n') >= 0 || text.indexOf('\r') >= 0) {
            // One call must be one screen line: the status block is sized in lines, so a multi-line
            // string handed over as "a line" desynchronises the reserved region. Callers fold their
            // text themselves; this is the backstop for the ones that forget.
            text.lines().forEach(this::line);
            return;
        }
        if (reading) {
            // Only while the line reader owns the screen: printAbove scrolls the text in above the
            // prompt and redraws that prompt afterwards. Calling it when nobody is reading redraws a
            // prompt that is not there, which is where the repeated "you>" lines came from.
            reader.printAbove(text);
        } else {
            terminal.writer().println(text);
            terminal.writer().flush();
        }
    }

    @Override
    public @Nullable String readLine(String prompt) {
        reading = true;
        try {
            return reader.readLine(prompt);
        } catch (UserInterruptException e) {
            return ""; // Ctrl-C: drop the line, ask again
        } catch (EndOfFileException e) {
            return null; // Ctrl-D
        } finally {
            reading = false;
        }
    }

    @Override
    public @Nullable String readKey(String prompt) {
        terminal.writer().print(prompt);
        terminal.writer().flush();
        Attributes saved = terminal.enterRawMode();
        try {
            int key = terminal.reader().read();
            if (key < 0 || key == 4) { // end of input, Ctrl-D
                return null;
            }
            if (key == 3) { // Ctrl-C: treat as "no", the safe answer
                terminal.writer().print("^C" + System.lineSeparator());
                terminal.writer().flush();
                return "n";
            }
            String answer = String.valueOf((char) key).toLowerCase(Locale.ROOT);
            terminal.writer().print(answer + System.lineSeparator());
            terminal.writer().flush();
            return answer;
        } catch (IOException e) {
            return null;
        } finally {
            terminal.setAttributes(saved);
        }
    }

    @Override
    public void status(List<String> lines) {
        if (lines.isEmpty()) {
            status.update(List.of());
            return;
        }
        // A rule above the block separates it from the scrollback, the way the established terminal
        // agents frame their input.
        // One row must never wrap: a wrapped row occupies two screen lines, the reserved region is
        // sized in lines, and everything below it is then drawn in the wrong place -- which is how a
        // long summary tore the block apart.
        int width = Math.max(10, terminal.getSize().getColumns() - 1);
        List<AttributedString> block = new java.util.ArrayList<>();
        block.add(new AttributedString("─".repeat(width), AttributedStyle.DEFAULT.foreground(AttributedStyle.BRIGHT)));
        for (String line : lines) {
            block.add(
                    new AttributedString(fit(line, width), AttributedStyle.DEFAULT.foreground(AttributedStyle.BRIGHT)));
        }
        status.update(block);
    }

    /**
     * Cut a status row to the window width.
     *
     * @param text the row
     * @param width how many characters fit
     * @return the row, ending in {@code …} when it had to be cut
     */
    static String fit(String text, int width) {
        return text.length() <= width ? text : text.substring(0, Math.max(1, width - 1)) + "…";
    }

    @Override
    public boolean pinsStatus() {
        return true;
    }

    @Override
    public Ansi ansi() {
        return ansi;
    }

    @Override
    public void close() {
        try {
            status.update(List.of());
            status.close();
            terminal.close();
        } catch (IOException | RuntimeException e) {
            // closing a terminal that is already gone must not fail the session
        }
    }
}
