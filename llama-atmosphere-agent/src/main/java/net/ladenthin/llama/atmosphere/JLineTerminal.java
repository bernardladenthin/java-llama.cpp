// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.atmosphere;

import java.io.IOException;
import java.lang.reflect.Method;
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
     * {@code Status.repaint()}, or {@code null} on a JLine that does not have it.
     *
     * <p><b>Why this is looked up rather than called.</b> The other four fixes this project carries against
     * JLine change how the library <em>behaves</em>, so the code compiles against the released version and
     * merely shows the symptom there. This one is a new method — the library offers no way to ask the pinned
     * region for a repaint, which is what a screen the console has changed behind its back needs (see
     * {@link #repaintBlockFromScratch()}). Calling it directly would make the released library fail to
     * compile, and this project must stay buildable with whatever JLine a copy of it finds: the pom's
     * {@code jline.version} is the released one on purpose.
     *
     * <p>So it is looked up once. Present: the block is repainted. Absent: nothing is forced, and the
     * artefact stays until something prints — exactly the trade the other four fixes make.
     */
    private static final @Nullable Method REPAINT = lookUpRepaint();

    /**
     * Find {@code Status.repaint()} if this JLine has it.
     *
     * @return the method, or {@code null}
     */
    private static @Nullable Method lookUpRepaint() {
        try {
            return Status.class.getMethod("repaint");
        } catch (NoSuchMethodException | RuntimeException e) {
            return null;
        }
    }

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

    /** How long the window size must hold still before the block is drawn one last time. */
    private static final long SETTLE_MILLIS = 400;

    /** When to draw the settled block, or {@code 0} when nothing is pending. */
    private long settleAt;

    /** The size the pinned region was just told about, so the rows are built for the same one. */
    private volatile @Nullable Size pendingSize;

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
        console.bindClearScreen();
        console.startWatchingSize();
        return console;
    }

    /**
     * Make Ctrl-L do what {@code /cls} does.
     *
     * <p>Ctrl-L is the second way into this command, and the two have to end with the prompt on the same
     * row. JLine's keymap dispatches the key by <em>name</em> to the widget registered under
     * {@link LineReader#CLEAR_SCREEN}, and its own widget is {@code clear_screen} plus a line redraw — so
     * it reproduced the identical defect, measured on an interpreted screen in {@code ScreenUseCasesTest}.
     * Replacing the map entry re-points the key without touching the keymap, so a binding a user already
     * knows keeps working and now does the same thing the command does.
     */
    private void bindClearScreen() {
        reader.getWidgets().put(LineReader.CLEAR_SCREEN, () -> {
            scrollAWindowAway();
            return true;
        });
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
                        if (now.getColumns() != last.getColumns() || now.getRows() != last.getRows()) {
                            last = now;
                            settleAt = System.currentTimeMillis() + SETTLE_MILLIS;
                            if (!refreshBlockForCurrentSize() && closed) {
                                return;
                            }
                            continue;
                        }
                        // The size stopped changing: draw once more, a moment later, and this one is a
                        // repaint rather than a diff. A console being enlarged reports the new width before
                        // its screen has applied it, so the LAST event of a drag is processed against a size
                        // the screen does not have yet -- the rule then covers two screen rows and
                        // everything above it moves up. Nothing re-renders afterwards, because the size no
                        // longer changes, so the wrong render is what stays. Reported after enlarging, with
                        // the block otherwise in place.
                        //
                        // From scratch, because a diff cannot see it: the console also REFLOWS its screen
                        // buffer when the window is widened, joining rows it had marked as wrapped -- every
                        // row of the region is padded to the last column, so all of them are -- and JLine is
                        // not told. Its model still matches what it wrote, so an ordinary redraw computes an
                        // empty diff and emits nothing, which is why the artefact survived every redraw and
                        // why holding Enter was what repaired it. See repaintBlockFromScratch.
                        if (settleAt != 0 && System.currentTimeMillis() >= settleAt) {
                            settleAt = 0;
                            if (!repaintBlockFromScratch() && closed) {
                                return;
                            }
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
        for (String piece : fold(text, Math.max(20, terminal.getSize().getColumns() - 1))) {
            write(piece);
        }
    }

    /**
     * Write one line that is known to fit, through whichever path is safe right now.
     *
     * @param text the line
     */
    private void write(String text) {
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
     * Break a line into pieces that each fit the window, so the CONSOLE never wraps it.
     *
     * <p><b>This is what makes the whole console reflow-proof, and the reasoning is the point.</b> A line the
     * console wrapped is <em>one</em> logical line spanning two screen rows, and Windows joins such lines
     * again when the window is widened. The text above then occupies fewer rows and <b>everything below moves
     * up</b> — including the block rows last rendered, which end up above the pinned region where nothing ever
     * writes again. One leftover per drag step, which is the reported staircase of rules climbing "von unten
     * rechts nach oben links"; narrowing does it in reverse and walks the input upwards. No program can
     * observe a reflow or prevent one. What it can do is deny it a target: a line that was never soft-wrapped
     * has nothing to join.
     *
     * <p>Counted in <b>screen columns</b>, not characters, and that distinction has cost this class a defect
     * before: an icon is one character and two columns, so folding by length lets a piece come out wider than
     * the window after all. {@link AttributedString#fromAnsi} parses the colours a caller already put in, so
     * the pieces keep their styling and the escape sequences do not count towards the width.
     *
     * <p>The price is stated rather than hidden: text keeps the line breaks it was printed with, so widening
     * the window does not re-flow the conversation. That is the same trade this console already makes by
     * rendering append-only — and the alternative is what the reports were about.
     *
     * @param text the line, possibly carrying ANSI styling
     * @param width how many columns a piece may use
     * @return the pieces, in order; a single-element list when the line already fits
     */
    static List<String> fold(String text, int width) {
        AttributedString measured = AttributedString.fromAnsi(text);
        if (measured.columnLength() <= width) {
            return List.of(text);
        }
        List<String> pieces = new java.util.ArrayList<>();
        int totalColumns = measured.columnLength();
        int at = 0;
        while (at < totalColumns) {
            // The window walks in COLUMNS, because that is the only unit the terminal cares about, and the
            // piece says how many it actually took: a double-width character straddling the boundary is left
            // for the next piece, so a piece can be one column short of the width.
            AttributedString piece = measured.columnSubSequence(at, Math.min(totalColumns, at + width));
            int consumed = piece.columnLength();
            if (consumed <= 0) {
                // A single character wider than the whole window. Cannot happen with the width the caller
                // passes, and a guard rather than a loop that never ends if it ever does.
                piece = measured.columnSubSequence(at, at + 2);
                consumed = Math.max(1, piece.columnLength());
            }
            pieces.add(piece.toAnsi());
            at += consumed;
        }
        return pieces;
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
    public String terminalType() {
        return terminal.getType();
    }

    @Override
    public void clearScreen() {
        synchronized (writing) {
            scrollAWindowAway();
        }
    }

    /**
     * Scroll a window's worth of blank lines in above the prompt, which is how this console clears.
     *
     * <p><b>SCROLLED, not erased — and the difference is the whole history of this method.</b> What this
     * console needs from a wipe is two things at once: a blank screen and the input back on the row the
     * pinned block leaves for it. Erasing gives the first and takes the second, because
     * {@code clear_screen} puts the cursor home and the reader draws its prompt where the cursor is —
     * reported as "nach /cls ist der cursor auch ganz oben und nicht unten". Everything tried on top of
     * an erase to put it back was reported as a new defect: blank rows after the erase scrolled the
     * erased lines back into view, and a {@code cursor_address} smuggled into {@code printAbove}'s
     * argument corrupted its bookkeeping and stranded a character above the prompt. The screen tests
     * then showed the erase alone is worse than it looks: the reader redraws its prompt as a diff
     * against what it believes is on screen, the erase invalidates that belief, and the measured result
     * was no prompt on screen at all.
     *
     * <p>Scrolling has none of those problems because it is nothing but output. A window's worth of
     * blank lines pushes everything above the window, so the screen is blank and what was written stays
     * reachable with the scrollbar — which erasing the scrollback ({@code ESC[3J}) would have broken
     * anyway. Nothing is erased, so nothing can be pulled back into view. The reader's bookkeeping stays
     * right, because printing above the prompt is exactly what {@code printAbove} is for. The block is
     * never touched at all: it is pinned, so it needs neither a reset nor a rebuild. And the cursor ends
     * on its row by construction, since printing is what pushes it there — which is also why pressing
     * Enter a few times repairs a screen that has lost rows.
     *
     * <p><b>It takes no lock of its own, and that is deliberate.</b> {@link #clearScreen()} holds
     * {@code writing} around it, but the Ctrl-L widget cannot: a widget runs on the reader's thread with
     * the reader's own lock held, and {@link #line(String)} takes {@code writing} first and the reader's
     * lock second — so acquiring {@code writing} there inverts the order and hangs the session. What
     * serialises the widget path instead is the reader's lock itself, which every {@code printAbove}
     * needs, so no other output can interleave; a block refresh still can, which is the exposure JLine's
     * own Ctrl-L widget has today as well.
     */
    private void scrollAWindowAway() {
        int lines = Math.max(1, terminal.getSize().getRows());
        for (int line = 0; line < lines; line++) {
            if (input == null) {
                // Before the reader exists there is no prompt to print above and no bookkeeping to
                // keep: writing straight to the terminal is both allowed and the only option.
                terminal.writer().println();
            } else {
                reader.printAbove("");
            }
        }
        terminal.writer().flush();
    }

    /**
     * Rebuild the pinned block and make the region forget what it believes is on screen first.
     *
     * <p><b>The difference from {@link #refreshBlockForCurrentSize()} is one call, and it is the difference
     * between a redraw that writes something and one that writes nothing.</b> JLine's pinned region is a
     * diff: it compares the rows it is handed with the rows it last wrote and emits only what changed. That
     * is right as long as nobody else touches the screen — and the console touches it. Windows reflows its
     * screen buffer when the window is widened, joining rows it had marked as wrapped, which is every row
     * the region writes, because each one is padded to the last column. JLine is never told, so its model
     * still matches what it wrote, every later update computes an empty diff, and the joined rows stay on
     * screen for the rest of the session. That is the reported screen — a rule with the activity row cut
     * short beside it on one line — and it is why pressing Enter a dozen times repaired it while every
     * redraw did not.
     *
     * <p><b>{@code Status.repaint()} is the fifth fix carried against JLine, and this method is why.</b>
     * There was no way to ask for a repaint from outside. {@code Status.redraw()} is {@code update(lines)}
     * under another name and diffs like it — it has to, being called from
     * {@code LineReaderImpl.redisplay()} on every keystroke. {@code Status.reset()} clears the model but
     * also forgets the scroll region, so the next update believes it must grow the region and scrolls to
     * make room: the stale rows were pushed <em>up</em> rather than cleared and the block stood on screen
     * <b>twice</b>, four rows apart — measured on the interpreted screen. Handing over an empty block does
     * the same thing for the same reason, and handing over blank rows of the same height works but makes
     * the block vanish for an instant, which two other screen cases caught as a flicker. The added
     * {@code repaint()} clears the model and nothing else, so the following write covers every reserved
     * row, the scroll region stays put, and nothing blinks.
     *
     * <p>{@code Status.resize(Size)} already does all of this — {@code display.reset()}, the scroll region,
     * and clearing a band of old remnants — but only <b>if the grid size actually changed</b>. After a drag
     * has settled it has not, so that whole body is skipped and the following update is an empty diff.
     * Which is the precise reason the reflowed screen was never repaired.
     *
     * <p>Only the <b>settle</b> redraw does this, not every size event: a drag reports a size every ~125 ms
     * and a full repaint on each of them is bytes spent against a screen that is about to change again. The
     * reflow happens when the console applies the final size, so the redraw that matters is the one after
     * the size stops changing.
     *
     * <p>Note what this cannot reach: the <b>prompt</b> has a display of its own, with the same diff and no
     * {@code reset()} a caller can call. That asymmetry is exactly why the block comes back and the prompt
     * row can stay blank, which {@code thePromptItselfStaysVisibleAfterEnlarging} records.
     *
     * @return {@code false} when the redraw threw, which a caller in a loop should treat as "skip this
     *     size" rather than as a reason to stop
     */
    boolean repaintBlockFromScratch() {
        if (requested.isEmpty()) {
            return true;
        }
        try {
            synchronized (writing) {
                if (REPAINT == null) {
                    // The released library has no repaint(), so there is nothing to force here and the
                    // reflowed rows stay until something prints -- which is the same shape every one of the
                    // other four fixes has on an unpatched library: the symptom comes back, nothing breaks.
                    // /cls and Ctrl-L repair it there, because they only print.
                    return true;
                }
                REPAINT.invoke(status);
            }
        } catch (ReflectiveOperationException | RuntimeException e) {
            return false;
        }
        return refreshBlockForCurrentSize();
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
                Size size = terminal.getSize();
                pendingSize = size;
                status.resize(size);
                updateStatus(lines);
                // NOT followed by a redisplay, and the attempt is recorded because it looks obvious.
                // Re-establishing the region CLEARS rows, and on a shrink it clears a band above the region
                // too -- the prompt's row. Asking the reader to redisplay() there writes nothing: its own
                // display still believes the prompt is on screen, so the diff is empty. Invalidating that
                // belief needs Display.reset(), which is not reachable from outside the reader, and
                // printAbove -- the one call documented as safe from another thread -- would print a line and
                // scroll. So the row stays blank, which thePromptItselfStaysVisibleAfterEnlarging records.
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
    /**
     * The size the block is built from, so every part of it agrees.
     *
     * <p>Set by {@link #refreshBlockForCurrentSize()} right before it tells the pinned region the new
     * geometry, and consumed once: any other caller reads the terminal as before. The point is that the
     * region and the rows it holds are never built from two different reads of the window.
     *
     * @return the size to build the block for
     */
    private Size sizeForBlock() {
        Size pending = pendingSize;
        pendingSize = null;
        return pending != null ? pending : terminal.getSize();
    }

    private String rule(int columns) {
        // TWO columns short, not one, and the second one is a measured defence rather than taste. A row as
        // wide as the window risks wrapping, and a wrapped row costs a second screen line while the region
        // is reserved in lines -- the row the wrap eats is the prompt's, which is the reported pair "nur die
        // Eingabe wandert hoch" and "wenn ich groesser ziehe kommen viel mehr Striche", and the probe's
        // cursor drifting up by one to three rows. A console being dragged reports a width it has not
        // finished applying, so the size the rule is built from can be ahead of the screen by a column;
        // aWindowThatReportsMoreColumnsThanItHasMustNotCostThePromptItsRow forces exactly that and is red
        // with one column of slack. It cannot defend against an arbitrarily large overshoot -- nothing
        // built from a reported width can -- but one column is the lag that actually occurs.
        return "─".repeat(Math.max(10, columns - 2));
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
        // ONE read of the size for the whole block, and for the region it is pinned in. There used to be
        // three -- one for the reserved region, one for this width, one inside rule() -- and during a drag
        // they can each see a different window: a rule built from a size the console has not applied yet is
        // wider than the window, wraps onto a second screen line, and the region then needs four lines
        // where three are reserved. Which is exactly the pair of reports "nur die Eingabe wandert hoch" and
        // "wenn ich groesser ziehe kommen viel mehr Striche": the wrap costs the prompt its row, and the
        // rule looks far too long. Whatever the size is, the rows and the region are now built from the
        // same one.
        int columns = sizeForBlock().getColumns();
        int width = Math.max(10, columns - 1);
        List<AttributedString> rows = new java.util.ArrayList<>();
        rows.add(new AttributedString(rule(columns), AttributedStyle.DEFAULT.foreground(AttributedStyle.BRIGHT)));
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
