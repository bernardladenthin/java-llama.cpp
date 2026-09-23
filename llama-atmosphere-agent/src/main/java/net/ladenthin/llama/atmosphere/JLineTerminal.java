// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.atmosphere;

import java.io.IOException;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import org.jline.keymap.KeyMap;
import org.jline.reader.Binding;
import org.jline.reader.EndOfFileException;
import org.jline.reader.LineReader;
import org.jline.reader.LineReaderBuilder;
import org.jline.reader.Reference;
import org.jline.reader.UserInterruptException;
import org.jline.reader.impl.completer.StringsCompleter;
import org.jline.terminal.Terminal;
import org.jline.terminal.TerminalBuilder;
import org.jline.utils.AttributedString;
import org.jline.utils.AttributedStyle;
import org.jline.utils.InfoCmp;
import org.jline.utils.Status;
import org.jspecify.annotations.Nullable;

/**
 * An {@link AgentTerminal} on a real terminal, via JLine: line editing and history at the prompt, tab
 * completion of the commands, a status line pinned to the bottom of the window, and a prompt that is
 * there at all times — including while the agent is working.
 *
 * <p>That last part is why a thread of its own owns the keyboard (see {@code startReading}): it sits
 * in {@code readLine} for the whole session and puts what is typed on a queue, and every read in this
 * class is served from that queue. Streamed output goes through {@link LineReader#printAbove(String)},
 * which scrolls it in above the prompt while the bottom block stays where it is — the one thing a
 * plain {@code println} cannot do. Nothing is ever redrawn above that block, so the scrollback stays
 * exactly as it was written.
 *
 * <p>{@link #open} returns {@code null} instead of throwing when there is no usable terminal (piped
 * input, a "dumb" terminal, a missing native provider); the caller then uses {@link PlainTerminal}.
 * Ctrl-C at the prompt clears the line and returns an empty one — it does not end the session; Ctrl-D
 * ends input like end-of-file.
 */
public final class JLineTerminal implements AgentTerminal {

    /**
     * What is queued in place of a line when input ends.
     *
     * <p>A queue of lines cannot carry "no more lines" as a value, and the reader thread is the only
     * one that learns it. The sentinel is put back on every take, so end of input stays end of input
     * for every later caller instead of turning back into "nothing typed yet".
     */
    private static final String END_OF_INPUT = "\u0000end-of-input";

    private final Terminal terminal;
    private final LineReader reader;
    private final Status status;
    private final Ansi ansi;
    private final BlockingQueue<String> typed = new LinkedBlockingQueue<>();
    private volatile boolean reading;
    private volatile boolean closed;
    private @Nullable Thread input;

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
                    // The input sits in a framed box at the bottom. Without this the box would be
                    // left behind in the scrollback on every Enter, so a few empty lines would print
                    // a wall of rules; the line the user typed is echoed above it instead.
                    .option(LineReader.Option.ERASE_LINE_ON_FINISH, true)
                    // "!" is a shell history expansion in the reader's default configuration, which
                    // silently rewrites a request like: git commit -m "fixed!"
                    .option(LineReader.Option.DISABLE_EVENT_EXPANSION, true)
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

    /**
     * Start the one thread that owns the keyboard, if it is not running yet.
     *
     * <p><b>Why a thread of its own.</b> The prompt is supposed to be there at all times — while the
     * agent is working, not only between turns — and only a thread that sits in {@code readLine} can
     * offer that. Everything else then writes through {@link LineReader#printAbove}, which scrolls
     * text in above the prompt and leaves it where it is.
     *
     * <p><b>Why exactly one.</b> A terminal has one keyboard, and two threads reading it take turns at
     * random. So every read in this class — a request, an approval answer — is served from the one
     * queue this thread fills, and nothing else ever reads the terminal.
     *
     */
    private synchronized void startReading() {
        if (input != null) {
            return;
        }
        input = new Thread(
                () -> {
                    while (!closed) {
                        reading = true;
                        try {
                            // Built fresh each time: the rule has to match the window, which can be
                            // resized between two requests.
                            String line = reader.readLine(rule() + System.lineSeparator() + "> ");
                            if (!line.isBlank()) {
                                // The box is erased on Enter, so the conversation would lose what was
                                // asked. Echoing it above keeps the transcript readable.
                                line(ansi.bold("› " + line.strip()));
                            }
                            typed.put(line);
                        } catch (UserInterruptException e) {
                            // Ctrl-C: drop what was typed and ask again, as before.
                        } catch (EndOfFileException e) {
                            typed.offer(END_OF_INPUT); // Ctrl-D
                            return;
                        } catch (InterruptedException e) {
                            Thread.currentThread().interrupt();
                            return;
                        } catch (RuntimeException e) {
                            typed.offer(END_OF_INPUT); // the terminal is gone; stop reading it
                            return;
                        } finally {
                            reading = false;
                        }
                    }
                },
                "agent-input");
        input.setDaemon(true);
        input.start();
    }

    /**
     * The rule that frames the input box, as wide as the window.
     *
     * @return a line of {@code ─}
     */
    private String rule() {
        return "─".repeat(Math.max(10, terminal.getSize().getColumns() - 1));
    }

    @Override
    public @Nullable String readLine(String prompt) {
        // The prompt is the box this terminal draws, so the caller's is ignored: a "you> " in front of
        // an input line that already sits in a frame is noise, and the frame cannot be handed in as a
        // string because it is rebuilt on every window size.
        startReading();
        try {
            return take(typed.take());
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            return null;
        }
    }

    /**
     * Hand out a queued line, keeping the end-of-input marker in the queue.
     *
     * @param line what came off the queue
     * @return the line, or {@code null} when input has ended
     */
    private @Nullable String take(String line) {
        if (END_OF_INPUT.equals(line)) {
            typed.offer(END_OF_INPUT);
            return null;
        }
        return line;
    }

    @Override
    public boolean hasPendingInput() {
        return !typed.isEmpty();
    }

    @Override
    public @Nullable String readKey(String prompt) {
        // The prompt belongs to the input thread and cannot be changed while it is waiting, so the
        // question is printed as an ordinary line above it and answered in the same input line as
        // everything else. That costs an Enter, and buys the one thing worth more: a prompt that is
        // there while the agent works. A single-key read here would need a second reader on the same
        // terminal, and the two would take turns at random.
        line(prompt);
        startReading();
        try {
            String answer = take(typed.take());
            return answer == null ? null : answer.trim().toLowerCase(Locale.ROOT);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            return null;
        }
    }

    @Override
    public void status(List<String> lines) {
        if (lines.isEmpty()) {
            status.update(List.of());
            return;
        }
        // The rule on top of this block is the bottom edge of the input box: the reader draws the top
        // edge as the first line of its prompt, so the two together frame the input the way the
        // established terminal agents do.
        // One row must never wrap: a wrapped row occupies two screen lines, the reserved region is
        // sized in lines, and everything below it is then drawn in the wrong place -- which is how a
        // long summary tore the block apart.
        int width = Math.max(10, terminal.getSize().getColumns() - 1);
        List<AttributedString> block = new java.util.ArrayList<>();
        block.add(new AttributedString(rule(), AttributedStyle.DEFAULT.foreground(AttributedStyle.BRIGHT)));
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

    /**
     * What a terminal sends for shift+tab: {@code ESC [ Z}, "backtab" (CSI Z).
     *
     * <p>It is bound literally as well as through terminfo, because JLine's Windows terminfo
     * (<code>windows-vtp.caps</code>) declares no <code>key_btab</code> at all — so the capability
     * lookup yields nothing there while the terminal itself, in virtual-terminal input mode, does
     * send the sequence.
     */
    private static final String BACKTAB = "\033[Z";

    /** The name the cycle action is registered under; a widget is addressed by name, not by object. */
    private static final String CYCLE_MODE_WIDGET = "jllama-cycle-approval-mode";

    @Override
    public boolean onCycleMode(Runnable action) {
        KeyMap<Binding> keys = reader.getKeyMaps().get(LineReader.MAIN);
        if (keys == null) {
            return false;
        }
        reader.getWidgets().put(CYCLE_MODE_WIDGET, () -> {
            action.run();
            return true;
        });
        Reference widget = new Reference(CYCLE_MODE_WIDGET);
        String fromTerminfo = KeyMap.key(terminal, InfoCmp.Capability.key_btab);
        if (fromTerminfo != null) {
            keys.bind(widget, fromTerminfo);
        }
        keys.bind(widget, BACKTAB);
        return true;
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
        closed = true;
        Thread reading = input;
        if (reading != null) {
            reading.interrupt();
        }
        try {
            status.update(List.of());
            status.close();
            terminal.close();
        } catch (IOException | RuntimeException e) {
            // closing a terminal that is already gone must not fail the session
        }
    }
}
