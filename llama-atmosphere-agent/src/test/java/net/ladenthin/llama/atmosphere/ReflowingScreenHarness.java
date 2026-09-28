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
     * which takes a pinned bar with it and leaves the rows it used to occupy behind. That second shape is what
     * the reports look like, but "looks like" is not evidence, and guessing here would make every case built on
     * this harness a proof of my own assumption. {@code ReflowProbe} prints the same pattern on the reporter's
     * console so the answer comes from there.
     */
    enum Anchor {
        /** The bottom row keeps its content; freed rows appear at the top. */
        BOTTOM,
        /** The top row keeps its content; freed rows appear at the bottom and content moves up. */
        TOP
    }

    private final Anchor anchor;

    ReflowingScreenHarness(String type, int columns, int rows) throws IOException {
        this(type, columns, rows, Anchor.BOTTOM);
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
        for (String row : screen) {
            boolean wrapped = !row.isEmpty() && row.charAt(row.length() - 1) != ' ';
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
