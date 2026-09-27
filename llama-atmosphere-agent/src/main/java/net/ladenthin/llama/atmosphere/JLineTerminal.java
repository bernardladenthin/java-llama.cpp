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
import org.jline.terminal.Size;
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
    /**
     * Held for the length of every write this class makes.
     *
     * <p>Two threads write here as a matter of course: the turn runs on a thread of Atmosphere's and
     * prints its tool lines and streamed text, while the console thread refreshes the pinned block
     * four times a second. Neither JLine's {@code printAbove} nor {@code Status.update} knows about
     * the other, so without this their escape sequences interleave and a fragment lands on screen as
     * text — a stray {@code 1H}, the tail of a cursor-position sequence, drawn into the middle of the
     * rule.
     */
    private final Object writing = new Object();

    /**
     * The block as it was last handed over, so it can be put back after the screen is wiped.
     *
     * <p>{@code Status} draws only what has changed, and a wipe does not change its content — it just
     * removes it from the screen. Without keeping a copy there is nothing to redraw it from, and the
     * bottom of the window stays empty until the next refresh happens to differ.
     */
    private volatile List<AttributedString> block = List.of();

    /**
     * The block as the caller asked for it, before being cut to the window.
     *
     * <p>Kept so it can be drawn from scratch after the screen is wiped. Handing the rendered rows
     * back is not enough: {@code Status} draws the difference between them and what it believes is on
     * screen, and after a wipe that belief is wrong in a way it cannot detect — measured, it emitted a
     * single character where a whole block was missing.
     */
    private volatile List<String> requested = List.of();

    /** How often the window size is polled; one drag emits an event roughly every 125 ms. */
    private static final long WATCH_INTERVAL_MILLIS = 120;

    private volatile boolean closed;
    private @Nullable Thread input;
    private @Nullable Thread sizes;

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
            return over(terminal, completions);
        } catch (IOException | RuntimeException e) {
            // No terminal, no native provider, a restricted environment: the plain console still works.
            return null;
        }
    }

    /**
     * Wrap a terminal that has already been built.
     *
     * <p>The seam the tests use: a terminal over a pair of streams renders exactly like a real one —
     * same escape sequences, same line reader — so what the screen would look like can be asserted on
     * the emitted bytes, without a TTY. That is the only way to catch a drawing bug like a prompt whose
     * height does not match what the reader erases when the line is submitted.
     *
     * @param terminal the terminal to drive
     * @param completions the words tab completes
     * @return the wrapper
     */
    static JLineTerminal over(Terminal terminal, List<String> completions) {
        LineReader reader = LineReaderBuilder.builder()
                .terminal(terminal)
                .completer(new StringsCompleter(completions))
                // The input line is erased when submitted and echoed above instead, so it does
                // not pile up in the scrollback. It erases exactly ONE line, which is why the
                // prompt has to stay one line -- see startReading.
                .option(LineReader.Option.ERASE_LINE_ON_FINISH, true)
                // "!" is a shell history expansion in the reader's default configuration, which
                // silently rewrites a request like: git commit -m "fixed!"
                .option(LineReader.Option.DISABLE_EVENT_EXPANSION, true)
                .build();
        // No WINCH handler here on purpose. The line reader installs its own for as long as it is
        // reading -- which is the whole session -- and it already resizes the pinned region itself
        // (LineReaderImpl.handleSignal calls Status.resize). Adding one of ours only put a second
        // writer on the terminal, on the signal thread, at the exact moment the reader was redrawing:
        // the row of "> > > > >" after a resize got worse, not better, when it was tried. The block is
        // rebuilt from a poll instead -- see startWatchingSize.
        JLineTerminal console = new JLineTerminal(terminal, reader, Status.getStatus(terminal), Ansi.detect());
        console.startWatchingSize();
        return console;
    }

    /**
     * Rebuild the pinned block whenever the window changes size.
     *
     * <p>What JLine holds are the rows it was handed, so a rule built for a 113-column window stays
     * 113 columns wide: on a resize the row is padded with spaces or cut with an ellipsis, never
     * re-made. The next row then continues on the same screen line and the three rows run together
     * with growing gaps — which is what was reported, repeatedly. Only the caller knows that a rule is
     * meant to span the window, so only the caller can fix it.
     *
     * <p>Measured on a real Windows console rather than reasoned: a probe that left the block alone on
     * a resize reproduced the report, and the same probe rebuilding all three rows at the new width on
     * every size event rendered cleanly. Four theories had been measured and discarded in the harness
     * before that (buffer-vs-window width, reflow by joining the rows, a wide-to-narrow-to-wide drag,
     * an accumulating cursor drift) — and the same probe recorded window and buffer at identical
     * widths throughout, so it is not the one the terminal's own API could explain.
     *
     * <p><b>Not covered by a test, and a written one was deleted rather than kept.</b> The
     * stream-backed harness has no screen model, so all it can observe after a size change is that
     * JLine re-emits the rule at its OLD width without this poll (59 columns, in the prompt's DEC
     * line-drawing form) and emits nothing with it. Neither says the rule was redrawn at the new
     * width, so every assertion built on them was satisfiable by the broken behaviour -- the first
     * version passed with the poll disabled. Green either way is worse than none, the same call
     * already made for two write-lock tests in this class's history. What this rests on is the probe
     * above, which ran on the console where the defect appears.
     *
     * <p>A poll, not a signal: the reader owns WINCH for the whole session. This writes through the
     * same lock as every other write, which is what the turn loop already does four times a second.
     * The probe recorded one size event per ~125 ms for a single drag, so the interval follows a drag
     * without redrawing between two of its events.
     */
    private synchronized void startWatchingSize() {
        if (sizes != null) {
            return;
        }
        sizes = new Thread(
                () -> {
                    Size last = terminal.getSize();
                    while (!closed) {
                        try {
                            Thread.sleep(WATCH_INTERVAL_MILLIS);
                        } catch (InterruptedException e) {
                            Thread.currentThread().interrupt();
                            return;
                        }
                        Size now;
                        try {
                            now = terminal.getSize();
                        } catch (RuntimeException e) {
                            // A closed terminal throws here, and a poll must not turn a normal exit into
                            // an error -- but ONLY a closed terminal ends this loop. Returning on any
                            // failure was a defect with a very confusing symptom: one transient throw
                            // killed the thread, the block then kept whatever width it had been built
                            // for, and a window shrunk afterwards showed a rule WIDER than itself,
                            // wrapping onto a second screen line and pushing everything below it one row
                            // out of place. Reported as "beim Groesse veraendern geht es immer noch
                            // kaputt" on a screen whose block was otherwise correct.
                            if (closed) {
                                return;
                            }
                            continue;
                        }
                        if (now.getColumns() == last.getColumns() && now.getRows() == last.getRows()) {
                            continue;
                        }
                        last = now;
                        if (!refreshBlockForCurrentSize() && closed) {
                            return;
                        }
                    }
                },
                "agent-size-watch");
        sizes.setDaemon(true);
        sizes.start();
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
        synchronized (writing) {
            if (input != null) {
                // Once the reader thread exists it owns the screen, and nothing may write around it --
                // not even in the instant between two reads. A direct write there cuts into the escape
                // sequence the next read is emitting and half of it lands in the scrollback as text:
                // a stray "[?1h" above the prompt was exactly that.
                reader.printAbove(text);
            } else {
                terminal.writer().println(text);
                terminal.writer().flush();
            }
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
        scrollToBottom();
        input = new Thread(
                () -> {
                    while (!closed) {
                        try {
                            // One line, and it has to stay one line: the reader erases a single line
                            // when the input is submitted, so a two-line prompt -- the rule above the
                            // input, which is what was tried first -- leaves that rule behind on
                            // every Enter, a column of them after a few.
                            String line = reader.readLine("> ");
                            if (!line.isBlank()) {
                                // The input line is erased on Enter, so the conversation would lose
                                // what was asked. Echoing it above keeps the transcript readable.
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
                        }
                    }
                },
                "agent-input");
        input.setDaemon(true);
        input.start();
    }

    @Override
    public void clearScreen() {
        String capability = terminal.getStringCapability(InfoCmp.Capability.clear_screen);
        if (capability == null) {
            return; // a terminal that cannot clear: better nothing than a guessed escape sequence
        }
        // The capability is terminfo source, not the sequence itself: it reads "\E[H\E[2J", with the
        // escape spelled out. Writing it as it comes prints that text on the screen, which is what a
        // test caught. Curses expands it the way terminal.puts would, but into a string this class can
        // hand to the reader instead of writing behind its back.
        StringBuilder expanded = new StringBuilder();
        org.jline.utils.Curses.tputs(expanded, capability);
        String clear = expanded.toString();
        synchronized (writing) {
            if (input == null) {
                terminal.writer().print(clear);
                terminal.writer().flush();
                return;
            }
            // Through the reader, like every other write once it exists: printAbove leaves the prompt
            // redrawn and the reader's idea of the cursor intact, which writing the escape sequence
            // around it would not.
            //
            // The wipe is followed by ADDRESSING the cursor to the last usable row, never by blank
            // rows, and the difference between those two is a reported defect in each direction.
            // Blank rows were the first attempt, to push the input back down the way startup does:
            // erasing the display clears the *visible* area and leaves those lines in the terminal's
            // scrollback, so scrolling afterwards pulls them straight back into view -- the screen
            // came back with text above the prompt. Wipe and stop was the second, and left the input
            // at the top left, which was reported just as often. A wiped screen needs no scrolling to
            // reach its last row: cursor_address goes there directly, nothing is written, so nothing
            // enters the scrollback and the reader draws its prompt where the cursor now is. Erasing
            // the scrollback as well (ESC[3J) would also allow the blank rows, and stays refused:
            // this console promises that what was written stays reachable with the scrollbar.
            // The wipe and NOTHING else. Addressing the cursor to the last usable row was tried here and
            // has been taken back out: printAbove owns the cursor -- it moves up, writes, and redraws the
            // prompt below -- so a cursor_address smuggled into its argument leaves its bookkeeping wrong,
            // and the screen came back with a single character stranded above the prompt. Reported twice,
            // the second time as "nach cls weiterhin eingabe ueber dem eingabe > zeichen". Both ways of
            // putting the input back at the bottom are now spent: blank rows scroll the wiped scrollback
            // into view, and moving the cursor corrupts printAbove. So the prompt sits where a wipe leaves
            // it, which is what the shell's own `clear` and Ctrl-L do, and output moves it down again.
            reader.printAbove(clear);
            // reset() makes it forget what it believes is on screen; without that the update below is
            // a no-op, because the content it would draw is the content it thinks is already there.
            List<String> lines = requested;
            // Three steps, and all three were needed to make the block come back after a wipe:
            // forget the drawing state, hand over an empty block so nothing is believed to be on
            // screen, then render it again. With only the first two, Status drew the difference it
            // computed against a belief the wipe had invalidated -- measured as a single character
            // where a whole block was missing.
            status.reset();
            block = List.of();
            status.update(List.of());
            if (!lines.isEmpty()) {
                updateStatus(lines);
            }
        }
    }

    /**
     * Rebuild the pinned block for the size the window has now.
     *
     * <p>The one thing the size poll does, and a method rather than three lines inside the thread so a
     * test can drive it at a known moment. The tests that read an interpreted screen call this directly
     * after changing the size: waiting for the poll made them pass alone and fail in a full run, and a
     * flaky test is worse than none. What that leaves uncovered is the thread itself — a loop that
     * compares two sizes and calls this — and that is the trade, stated rather than implied.
     *
     * <p>It rebuilds the rows and does <b>nothing else</b>. Telling the pinned region the new geometry
     * here ({@code Status.resize}) is what an earlier unit test appeared to require, because no reader
     * runs in one — and it destroyed the real console: {@code 36;1H} and a stray {@code 1} printed as
     * text inside the rule, a {@code [} in front of the state row, the block drawn twice. That call
     * writes to the terminal directly, so it lands in the middle of what the reader is drawing for the
     * same size change and an {@code ESC} byte is lost. The reader has already done that resize
     * ({@code LineReaderImpl.handleSignal}) by the time a poll notices the change.
     *
     * @return {@code false} when the redraw threw, which a caller in a loop should treat as "skip this
     *     size" rather than as a reason to stop: a redraw colliding with the reader's own throws, and
     *     giving up on the first one froze the block at whatever width it had reached — measured on an
     *     interpreted screen as a rule 62 columns wide in a 100-column window
     */
    boolean refreshBlockForCurrentSize() {
        List<String> lines = requested;
        if (lines.isEmpty()) {
            return true;
        }
        try {
            synchronized (writing) {
                // Re-establish the reserved region, then rebuild the rows. The region is what keeps the
                // block's rows off the prompt's row, and a probe on the reporter's console measured the
                // console's own cursor drifting UP by one to three rows after dragging -- never more than
                // the block's height -- while the content still looked plausible. Re-asserting the region
                // is what JLine's own handleSignal does on a size change.
                //
                // This was here before, removed, and is back for a reason: it writes to the terminal
                // directly, and without Status being synchronized that landed inside what the reader was
                // drawing for the same size change, which put "36;1H" on screen as text. That race is the
                // fourth fix carried against JLine (Status's public methods are synchronized there), and
                // refreshingTheBlockWhileTheReaderRedrawsNeverThrows is what holds it -- red 2/2 without
                // that fix, green 3/3 with it. So this line depends on that fix and must not be kept
                // without it.
                status.resize(terminal.getSize());
                updateStatus(lines);
            }
            return true;
        } catch (RuntimeException e) {
            return false;
        }
    }

    /**
     * How many rows to fill so the cursor ends up on the last usable one.
     *
     * @return the count, never negative
     */
    private int blankRows() {
        return Math.max(0, terminal.getSize().getRows() - 1);
    }

    /**
     * The sequence that puts the cursor on the last row the prompt may use.
     *
     * <p>Counted from the bottom rather than from the top: the pinned block occupies the rows below
     * the prompt, so the prompt's own row is the last one above it. An empty block reserves nothing
     * and the prompt may use the very last row.
     *
     * @return the expanded {@code cursor_address} sequence, or an empty string on a terminal that
     *     cannot address the cursor (then the caller simply leaves it where the wipe put it)
     */
    private String cursorToLastUsableRow() {
        String capability = terminal.getStringCapability(InfoCmp.Capability.cursor_address);
        if (capability == null) {
            return "";
        }
        int row = Math.max(0, terminal.getSize().getRows() - 1 - block.size());
        StringBuilder expanded = new StringBuilder();
        org.jline.utils.Curses.tputs(expanded, capability, row, 0);
        return expanded.toString();
    }

    /**
     * Push the cursor to the last usable row, once, before the first prompt is drawn.
     *
     * <p>The line reader draws its prompt wherever the cursor happens to be, which is directly after
     * the last thing printed; only the status block is pinned to the bottom of the window. On a
     * half-empty screen that leaves the input floating in the middle with the block far below it, and
     * it only looks like one piece once enough output has scrolled the cursor down by itself — which
     * is why it looked right after a few turns and wrong at the start.
     *
     * <p>Scrolling the screen once at startup makes that the state from the first prompt on: from
     * then on every line printed scrolls, so the cursor stays on the last row for the rest of the
     * session. The cost is a screen of blank lines above the session, which is what any program that
     * wants its input at the bottom without taking over the whole screen has to pay.
     */
    private void scrollToBottom() {
        synchronized (writing) {
            for (int row = 0; row < blankRows(); row++) {
                terminal.writer().println();
            }
            terminal.writer().flush();
        }
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
        // A blank line is not a request, so it must not count: the REPL skips it, and counting it
        // meant that holding Enter cancelled one turn per keystroke and produced nothing. It stays in
        // the queue, because an empty answer to an approval question means yes.
        return typed.stream().anyMatch(line -> !line.isBlank());
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
        synchronized (writing) {
            updateStatus(lines);
        }
    }

    private void updateStatus(List<String> lines) {
        requested = List.copyOf(lines);
        if (lines.isEmpty()) {
            if (block.isEmpty()) {
                return;
            }
            block = List.of();
            status.update(List.of());
            return;
        }
        // The rule on top of this block is the one that separates the conversation from the input.
        // It is the ONLY one drawn: a rule above the input line cannot be pinned (JLine's status
        // region is below the prompt, never above it) and drawing it as output leaves one behind in
        // the scrollback per turn, which is what "the line keeps travelling along" was.
        // One row must never wrap: a wrapped row occupies two screen lines, the reserved region is
        // sized in lines, and everything below it is then drawn in the wrong place -- which is how a
        // long summary tore the block apart.
        int width = Math.max(10, terminal.getSize().getColumns() - 1);
        List<AttributedString> rows = new java.util.ArrayList<>();
        rows.add(new AttributedString(rule(), AttributedStyle.DEFAULT.foreground(AttributedStyle.BRIGHT)));
        for (String line : lines) {
            rows.add(
                    new AttributedString(fit(line, width), AttributedStyle.DEFAULT.foreground(AttributedStyle.BRIGHT)));
        }
        block = List.copyOf(rows);
        status.update(rows);
    }

    /**
     * Cut a status row to the window width.
     *
     * @param text the row
     * @param width how many characters fit
     * @return the row, ending in {@code …} when it had to be cut
     */
    static String fit(String text, int width) {
        // Counted in screen columns, not characters. An icon or an emoji occupies two columns and one
        // character, so cutting by character length lets a row come out wider than the window, wrap
        // onto a second screen line, and push everything below the reserved region out of place --
        // the same tearing a long summary caused, arriving through a different door.
        AttributedString measured = new AttributedString(text);
        if (measured.columnLength() <= width) {
            return text;
        }
        return measured.columnSubSequence(0, Math.max(1, width - 1)).toString() + "…";
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
        Thread watching = sizes;
        if (watching != null) {
            watching.interrupt();
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
