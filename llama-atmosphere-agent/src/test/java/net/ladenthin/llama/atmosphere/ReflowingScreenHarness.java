// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.atmosphere;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * A screen that <b>reflows</b> when its width changes, the way a Windows console does.
 *
 * <p><b>Why this class had to exist.</b> Every automated reproduction of the resize reports failed for one
 * reason, and it took several rounds to name it: {@link ScreenTerminalHarness}'s screen — JLine's own
 * {@code ScreenTerminal} — adjusts its buffer on a resize but <b>never reflows</b>. A real console does: a row
 * that was written past the right margin is remembered as a <em>continuation</em> of the one above it, and when
 * the window is widened the two are joined again. Everything below then moves up. That single behaviour is
 * what turns a correct pinned block into the reported screens — leftover bars above the region, rows running
 * together, the input floating in the middle — and a model without it reports every one of those cases green.
 *
 * <p><b>What it does, stated so it can be checked rather than believed.</b> On a width change it reads the
 * screen it has, rebuilds the logical lines, re-wraps them to the new width and writes the result back. Two
 * rules, and both are deliberately simple:
 *
 * <ol>
 *   <li><b>A row continues into the next one when its last cell is not blank.</b> That is the terminal's own
 *       rule in practice: the flag is set when output passes the right margin, which can only happen if the
 *       last cell was written. It is also exactly why the seventh JLine fix — never padding a status row to the
 *       full width — is expected to matter, and this harness is what can show that.
 *   <li><b>The re-wrapped content is bottom-aligned</b>, because a console keeps the cursor's line in view: if
 *       joining lines frees rows, the content moves up and blank rows appear at the bottom.
 * </ol>
 *
 * <p>The result is written with {@link #writeBehindTheApplicationsBack}, which is the honest channel for it:
 * a console's reflow changes the screen without telling the program, and so does this.
 *
 * <p><b>The limit, stated rather than discovered later.</b> The cursor is not reflowed with the content — it
 * stays where the screen model has it. So this harness is evidence about what is ON the screen after a resize,
 * not about where the cursor ends up; the cursor cases stay with {@link ScreenTerminalHarness}. And the
 * continuation rule is an inference from how terminals set the flag, which is why {@code ReflowProbe} exists:
 * it prints wide and short lines on the reporter's real console so the rule can be confirmed against it.
 */
class ReflowingScreenHarness extends ScreenTerminalHarness {

    /**
     * Which edge of the window keeps its content when a reflow frees or needs rows.
     *
     * <p><b>Deliberately a choice and not a decision.</b> Joining wrapped lines makes the content occupy fewer
     * rows, and where the slack appears decides everything that follows: with {@link #BOTTOM} the last row keeps
     * what it had and the freed rows appear at the top (a console that keeps the cursor's line in view, pulling
     * scrollback down), while with {@link #TOP} the first row keeps what it had and everything below moves up —
     * which takes a pinned bar with it and leaves the rows it used to occupy behind.
     *
     * <p><b>Measured, on the reporter's Windows console, and the default follows the measurement.</b>
     * {@code ReflowProbe} filled a 32-row window, printed three lines twelve columns longer than the window (so
     * each occupied two rows) and asked the console for the cursor's row after each drag — a number, because a
     * pasted screen cannot answer this: Windows Terminal copies the whole scrollback and rejoins wrapped runs,
     * so wrapping is invisible in a paste. The readings:
     *
     * <pre>
     * widened   86 -> 111 columns:  cursor on row 28 of 32   (last row would be 31)
     * narrowed 111 ->  72 columns:  cursor on row 31 of 32
     * quickly back to 82 columns:   cursor on row 29 of 32
     * </pre>
     *
     * So widening moved the content <b>up by exactly the three rows</b> that joining the three wrapped lines
     * freed, while narrowing kept the cursor on the last row (the content grows downwards and the top falls into
     * the scrollback). {@link #TOP} is therefore the default: the top keeps its content and everything below
     * moves up — which is what carries a pinned bar away from its rows and leaves a copy behind, one per size
     * event.
     */
    enum Anchor {
        /** The bottom row keeps its content; freed rows appear at the top. */
        BOTTOM,
        /** The top row keeps its content; freed rows appear at the bottom and content moves up. */
        TOP
    }

    private final Anchor anchor;

    /**
     * Which screen rows this harness created by wrapping, i.e. which ones continue into the row below.
     *
     * <p><b>Inferring this was a real defect and it is worth keeping the reason.</b> The first version decided
     * "this row continues" by looking at its last cell: not blank meant wrapped. A console does not guess — it
     * sets a flag when output passes the right margin — and the inference fails exactly where a break lands on a
     * space, which is most of the time for prose. Measured: an answer re-wrapped at 50 columns broke after
     * "... Could you please ", the 50th character was a space, the two rows were then not recognised as one
     * logical line, and widening did not join them. The harness reported the case green while the console did
     * not.
     *
     * <p>So the rows this harness wraps itself carry a real flag, and only rows it has not touched fall back to
     * the inference — which is sound there, because a row whose last cell the application filled is exactly the
     * one a terminal flags.
     */
    private boolean[] continues;

    ReflowingScreenHarness(String type, int columns, int rows) throws IOException {
        this(type, columns, rows, Anchor.TOP);
    }

    ReflowingScreenHarness(String type, int columns, int rows, Anchor anchor) throws IOException {
        super(type, columns, rows);
        this.anchor = anchor;
    }

    @Override
    void resize(int columns, int rows) {
        int oldColumns = getSize().getColumns();
        String[] before = rows();
        super.resize(columns, rows);
        if (columns != oldColumns) {
            reflow(before, columns, rows);
        }
    }

    @Override
    Thread resizeAsynchronously(int columns, int rows) {
        int oldColumns = getSize().getColumns();
        String[] before = rows();
        Thread pump = super.resizeAsynchronously(columns, rows);
        if (columns != oldColumns) {
            reflow(before, columns, rows);
        }
        return pump;
    }

    /**
     * Join the rows into logical lines, re-wrap them to the new width, and put them back on the screen.
     *
     * @param before the rows as they were, at the old width
     * @param columns the new width
     * @param rows the new height
     */
    private void reflow(String[] before, int columns, int rows) {
        List<String> logical = joinContinuations(before);
        List<String> rewrapped = new ArrayList<>();
        for (String line : logical) {
            String rest = line;
            do {
                int take = Math.min(columns, rest.length());
                rewrapped.add(rest.substring(0, take));
                rest = rest.substring(take);
            } while (!rest.isEmpty());
        }
        // Which edge keeps its content is this harness's one open question -- see Anchor.
        while (rewrapped.size() > rows) {
            rewrapped.remove(anchor == Anchor.BOTTOM ? 0 : rewrapped.size() - 1);
        }
        int firstRow = anchor == Anchor.BOTTOM ? rows - rewrapped.size() : 0;
        continues = new boolean[rows];
        for (int row = 0; row < rows; row++) {
            int at = row - firstRow;
            // A piece continues into the next row when it filled the width AND something follows it, which is
            // precisely the flag a terminal sets.
            continues[row] =
                    at >= 0 && at + 1 < rewrapped.size() && rewrapped.get(at).length() == columns;
        }
        // Saved and restored around it, because a reflow rearranges CONTENT: leaving the cursor somewhere
        // else would be a second thing to explain in every case built on this.
        StringBuilder painted = new StringBuilder("\u001b7");
        for (int row = 0; row < rows; row++) {
            painted.append("\u001b[").append(row + 1).append(";1H").append("\u001b[K");
            int at = row - firstRow;
            if (at >= 0 && at < rewrapped.size()) {
                painted.append(trimTrailing(rewrapped.get(at)));
            }
        }
        painted.append("\u001b8");
        try {
            writeBehindTheApplicationsBack(painted.toString());
        } catch (IOException e) {
            throw new IllegalStateException("the screen is gone", e);
        }
    }

    /**
     * Rebuild logical lines: a row whose last cell is not blank continues into the row below it.
     *
     * @param screen the rows as they are
     * @return the logical lines, trailing blanks of a non-continued row removed
     */
    private List<String> joinContinuations(String[] screen) {
        List<String> logical = new ArrayList<>();
        StringBuilder current = new StringBuilder();
        boolean continuing = false;
        for (int index = 0; index < screen.length; index++) {
            String row = screen[index];
            boolean wrapped = continues != null && index < continues.length
                    ? continues[index]
                    : !row.isEmpty() && row.charAt(row.length() - 1) != ' ';
            if (continuing) {
                current.append(row);
            } else {
                current.setLength(0);
                current.append(row);
            }
            if (wrapped) {
                continuing = true;
                continue;
            }
            continuing = false;
            logical.add(trimTrailing(current.toString()));
        }
        if (continuing) {
            logical.add(trimTrailing(current.toString()));
        }
        // A screen that is entirely blank has one logical line per row; keeping them all would push real
        // content off the top, so the blank ones at the END are dropped and re-created as blank rows.
        while (!logical.isEmpty() && logical.get(logical.size() - 1).isEmpty()) {
            logical.remove(logical.size() - 1);
        }
        return logical;
    }

    private String trimTrailing(String text) {
        int end = text.length();
        while (end > 0 && text.charAt(end - 1) == ' ') {
            end--;
        }
        return text.substring(0, end);
    }
}
