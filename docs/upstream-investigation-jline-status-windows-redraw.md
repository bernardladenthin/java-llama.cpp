<!--
SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>

SPDX-License-Identifier: MIT
-->

# JLine: resizing with a pinned `Status` redraws the edit line once per size event

**Status:** root cause found, fixed and verified against JLine 4.4.6 with a one-line change; not yet
filed upstream.
**Affects:** JLine 4.4.5 and 4.4.6, every terminal type. On types without `cursor_up` — `windows-vtp`,
i.e. Windows Terminal and PowerShell — the copies land side by side on one logical line, which is the
shape it was reported as.
**Where it shows in this repo:** `llama-atmosphere-agent`, whose console pins a three-row status block.

## Symptom

Start a session, type a few characters **without** pressing Enter, then drag the window wider:

```
> Hallo> Hallo> Hallo> Hallo> Hallo> Hallo> Hallo> Hallo> Hallo> Hallo> Hallo> Hallo
```

The prompt and the input buffer appear a dozen times side by side.

**It is one copy per size event.** A drag reports a new size per step, so the count follows the drag,
not the keystrokes. An earlier version of this document claimed the opposite ("the count follows the
number of keystrokes") on the strength of five characters producing twelve to seventeen repeats — that
was an inference from a number, with no measurement of how many size events the drag produced, and it
was wrong: a reproduction that emits exactly 20 size events produces exactly 19 extra copies.

## Reproduction, without Windows and without any project code

`ResizeStatusRepro` (session scratchpad) uses JLine's own `VirtualTerminal` — a real VT interpreter
over a virtual screen — attaches a three-row `Status`, starts `readLine`, types `Hallo`, then walks the
width from 40 to 60 columns raising `WINCH` at each step, and counts `> Hallo` in the screen dump:

| variant | pinned `Status` | terminal type | `> Hallo` on screen |
|---|---|---|---|
| resize | three rows | `xterm` | **7** (stacked on separate rows) |
| resize | three rows | `windows-vtp` | **19** (side by side, as reported) |
| resize | none | `xterm` | 1 |
| typing only, no resize | three rows | either | 1 |

The last row is why the first attempt at a reproduction failed: typing alone is not enough, the size
events are the trigger. The `Status` is equally necessary — the same drag without one is clean.

## Root cause

`LineReaderImpl.handleSignal(WINCH)` takes two different paths, and only one of them keeps the display
model:

```java
if (status == null || status.size() == 0) {
    size = newSize;
    display.resize(size);          // reflow the existing model, emit nothing
} else {
    size = newSize;
    doDisplay();                   // <-- installs a BRAND-NEW Display
    status.resize(size);
    redisplay();
}
```

`doDisplay()` assigns `display = new Display(terminal, false)`. A fresh `Display` has an empty
`oldLines` model and `cursorPos == 0`, i.e. it believes the screen is blank — so the `redisplay()` on
the next line paints the prompt and the buffer as **new content** instead of as a diff against what is
already there. Every size event repeats that, and where the copy lands is decided by the terminal's
capabilities: on `xterm` on the following row, on `windows-vtp` (no `cursor_up`, no `scroll_reverse`)
immediately after the existing text.

`Display.resize(Sized)` is precisely the state-preserving operation this branch needs: it reflows
`oldLines` to the new width and recomputes `cursorPos` from the old cursor offset.

## The fix

One statement in `reader/src/main/java/org/jline/reader/impl/LineReaderImpl.java`:

```diff
-                    doDisplay();
+                    display.resize(size);
```

Verified on this host against the released 4.4.6 jar (JLine's own build needs a JDK 22, only 21 is
installed here, so the patched `LineReaderImpl` was compiled against the jar and overlaid):

- the three new tests in `StatusRedisplayTest` go from **2 failed / 1 passed** to **3 passed**;
- JLine's existing `LineReaderResizeTest`, `StatusTest`, `DisplayTest` and `InBandResizeTest` — 72
  tests — pass **unchanged** both before and after, so the change is not a trade.

The test and the fix are in the local JLine checkout at `X:\Privat\OpenSource\github\jline3`
(`reader/src/test/java/org/jline/reader/impl/StatusRedisplayTest.java` plus the one-line change), ready
to become an upstream pull request.

## Trying it in this project

A patched jar is installed in the local Maven repository as `org.jline:jline:4.4.6-statusfix`, and the
agent's pom carries a `jline.version` property, so:

```bat
cd llama-atmosphere-agent
mvn compile exec:java -Djline.version=4.4.6-statusfix -Dexec.args="--model <model.gguf>"
```

## What was ruled out along the way

| hypothesis | test | result |
|---|---|---|
| Project code (the agent's console) | a probe containing none of it | still reproduces |
| Fixed in a newer JLine | same probe against 4.4.6 | still reproduces |
| Driven by the resize *signal* delivery | `TerminalBuilder.nativeSignals(false)` | still reproduces |
| A Windows-only terminal quirk | virtual VT screen, `xterm` **and** `windows-vtp` | reproduces on both |
| A missing terminfo capability | `windows-vtp.caps` carries `sc`, `rc`, `csr`, `cup`, `el`, `cr` | all present (`ri` is absent) |
| The `lastStatusSize` guard in `redisplay()` | typing with a constant-height bar, no resize | does **not** reproduce |

The `lastStatusSize` row is worth keeping: that guard was the documented hypothesis before the resize
path was measured, and it is not the cause. A bar of constant height renders correctly for as long as
nobody resizes.

One earlier measurement attempt is recorded because it was **inconclusive rather than negative**: a
`WINCH` handler registered by a probe logged nothing, which does *not* mean no size events arrived —
`LineReaderImpl.readLine()` installs its own handler and restores the previous one afterwards, so a
handler registered before the read never runs during it. A second one failed for a reason worth
knowing: teeing `System.out` captured **zero bytes**, because on Windows JLine writes through the
console API rather than through `System.out`.

## Consequences for this project

- Nothing in `llama-atmosphere-agent` causes it, and it cannot be fixed there without reimplementing
  `handleSignal` in a `LineReaderImpl` subclass. An earlier attempt to help by installing a `WINCH`
  handler made the display **worse**, because the line reader owns that signal for the whole session —
  see `CLAUDE.md`, "Do not add a `WINCH` handler".
- No project test asserts the fixed behaviour: it would fail against the JLine release the build
  depends on. `JLineTerminalTest` therefore keeps pinning what *is* true today.
- `--plain` is unaffected and is the escape hatch: it pins nothing and redraws nothing.

## A second defect in the same area, found while verifying this one — not fixed

A bar whose **height changes** while a line is being edited makes the edit line disappear outright:

```
StatusRepro windows-vtp changing   -> "> Hallo" drawn 0 time(s)   (blank screen above the bar)
StatusRepro xterm       changing   -> "> Hallo" drawn 0 time(s)
```

It is a different code path — `redisplay()`'s `lastStatusSize` branch, which also calls `doDisplay()`
(after a `carriage_return`) — and the fix above does **not** address it: both counts stay 0 with the
patched jar. It is not the same symptom the `lastStatusSize` hypothesis predicted (duplication); the
line is lost, not repeated.

Reachable from this project, though rarely: the agent's block is always three rows, so the height
changes only when it is taken down and put back — `/cls` and `/exit`. Both are tested and behave, so
this is recorded as an upstream observation rather than a chased bug.

## Second defect, reproduced minimally: `windows-vtp` loses the first character of the last row

Found while chasing the fragments above, and it needs **no resize, no line reader and no project
code** — a single `status.update()` of a three-row block on a `VirtualTerminal` of type
`windows-vtp` renders the bottom row as `state]` instead of `[state]`. On `xterm` it is intact.

The byte streams say why. At 40 columns `xterm` receives each row as exactly 40 characters written
back to back, relying on the terminal to wrap. `windows-vtp` receives one character more per row,
each followed by `ESC[D`:

```
xterm        ESC7 ESC[8;1H rule<36 spaces>waiting<33 spaces>[state]<33 spaces> ESC8
windows-vtp  ESC7 ESC[8;1H rule<37 spaces>ESC[D waiting<34 spaces>ESC[D [state]<34 spaces>ESC[D ESC8
```

That is `Display`'s compensation for a terminal without `eat_newline_glitch` (delayed wrap), plus its
separate rule of never writing the bottom-right cell (issue #2206). `windows-vtp.caps` declares `am`
but not `xenl`, while `xterm.caps` declares both — yet Windows' virtual-terminal processing delays the
wrap like any other VT, which the report's own screen confirms: on the real console no character is
missing from the block.

**The fix is one word in `terminal/src/main/resources/org/jline/utils/windows-vtp.caps`**: `xenl`
alongside `am`. With it the bottom row renders whole and all of JLine's existing
`LineReaderResizeTest` / `StatusTest` / `DisplayTest` / `InBandResizeTest` stay green (75 tests,
unchanged before and after), plus the three new ones in `StatusDelayedWrapTest` — verified red without
the change (2 of 3 failing on `[state]` vs `state]`) and green with it.

## Third defect: the status bar is sized by the buffer, not by the window

`LineReaderImpl.handleSignal(WINCH)` reads `terminal.getBufferSize()` and hands that size to
`Status.resize(...)`, so every status row is padded to the **buffer** width. JLine's own contract on
`Terminal.getBufferSize()` says the opposite: it exists for line editing, where a buffer wider than
the window is what avoids a wrap, *"while the `getSize()` method should be used when using full screen
mode"* — and a pinned region is a full-screen construct, positioned in window rows. `Status` itself
uses `terminal.getSize()` when it is created; only the resize path disagrees.

On Windows the two differ as a matter of course: `NativeWinSysTerminal.getSize()` returns
`srWindow` (the visible window) and `getBufferSize()` returns `dwSize` (the screen buffer).

Reproduced with a `VirtualTerminal` subclass whose `getBufferSize()` reports 40 columns more than its
screen, one size event, text in the buffer:

```
window 64 cols, buffer 104 cols
7|─────────────────────────────────────────────────────────    |   rule, still in place
8|                                        … waiting for input …|   shifted right by exactly 40
9|                                                             |   the state row is gone
```

The rows are padded past the right edge, each wraps onto a second screen line, and the block smears
across the output — which is what "rule, activity row and state row all on one line, with a gap that
grows" looks like. With `status.resize(terminal.getSize())` the same run puts all three rows back on
their own screen rows with the state row on the last one.

Together the three fixes take JLine's own resize/status/display tests from 74 passing + 5 failing (the
new assertions) to **79 passing**, with the 74 unchanged in both directions.

## What remains on the real console after the fix: fragments of the rule

With the patched jar the prompt is drawn once, which is the defect above. What is still visible on a
real Windows console after dragging a wide window is the **status block's own rows** — runs of the
rule character at several widths, spread across one wrapped logical line, with the block itself
correct at the bottom.

**This does not reproduce in the harness**, and the variants were measured rather than assumed: a
full-width rule (`─` × columns−1) plus the activity and state rows, on `windows-vtp`, dragged from
140 columns down to 60 and back up, both with the block left alone and with all three rows rebuilt at
the new width on **every** size event — every run ends with exactly one row carrying rule characters.

So the emitted sequences are correct for a conformant VT, and what is left is the real console's own
handling of lines drawn at earlier widths: Windows Terminal reflows wrapped lines on widening, and an
`ESC[2J` erase leaves them in the scrollback (the same property that made `/cls` scroll old rows back
into view — see `CLAUDE.md`). Nothing in this project draws those rows a second time: only
`status()` and `clearScreen()` rebuild the block, and neither runs on a resize.

**The workaround is Ctrl-L (or `/cls`)** — it wipes the visible area and redraws the block, which is
already covered by `JLineTerminalTest`. A fix would need the byte stream of a real session, which on
Windows cannot be captured from inside the same JVM (see the note above about JLine writing through
the console API), so it is recorded here rather than guessed at.

## The drift, measured: our render path is innocent, and the screen loses rows

A probe on the reporter's console (`DriftProbe`, session scratchpad) measures the cursor row **twice per
request** — once before rebuilding the pinned block and once after, with the rebuild doing exactly what
the console does. Two results, and both are decisive.

**`cursorAfter` equalled `cursorBefore` in every single measurement**, across every run. That is a
comparison of two readings on the same yardstick, so it holds regardless of whether the expectation was
right — and it rules out this project's render path as the thing that moves the cursor. Together with the
bisection (neither half of JLine's `handleSignal`, nor this console's refresh, nor the screen model) that
leaves the movement happening during the console's own reflow.

**The recovery sequence names the mechanism.** With the window left at one size and Enter pressed
repeatedly, the readings climb one row per printed line:

```
cursorBefore=14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 26, 26, …
```

The cursor is not drifting *up*; each printed line walks it back *down* until it reaches its row and stays
there. So the screen had **lost rows** above the block, and printing restores the layout — which is
exactly what the reporter described: "es scheint als wäre der Bereich darüber invalide und müsste einmal
überschrieben werden", and "nach ganz oft Enter sieht es wieder okay aus".

**The candidate fix, and the obstacle.** The console already owns the primitive: `scrollToBottom()` emits
`rows - 1` newlines once at startup for precisely this reason. Pushing the cursor back down when it sits
above its row would be self-correcting and is entirely within this project. The obstacle is detection: a
cursor-position report is a round trip through the terminal's input, so it cannot be issued from the
polling thread while the reader owns the keyboard — which is why the probe measures only between reads.
Tracking the row by counting what we print is the alternative, and it is fragile in exactly the situation
that matters, a console that is dropping rows on its own.

**Two probe defects, recorded so the earlier readings are not trusted.** The first version reported
`cursorBefore=4` against `expectedRow=26` on every measurement and read as "the console moved it": it
never scrolled to the bottom at startup, so the prompt legitimately sat near the top and the expectation
did not apply there. It also wrote its log lines with `terminal.writer().println` instead of
`printAbove`, which corrupted its own block into three rules at three widths — the defect this project's
console fixed long ago. Both are fixed; the readings above are from the corrected probe.

## The clear was the same defect, and fixing it turned the recovery into a command

Chasing the lost rows produced a second report that turned out to be the same mechanism from the other
end: **after `/cls` the input sat at the top left instead of at the bottom.** It is worth recording here
because it is the one piece of this investigation that had a fix entirely inside this project, and because
the reproduction settled a question the byte-level tests could not.

**Reproduced first, on an interpreted screen.** Four cases in `ScreenUseCasesTest` — a two-row block, the
three-row block the application really pins, a screen with eight lines of output on it, and the same
question asked through Ctrl-L. Every one of them measured the cursor on **row 0** of a 12-row window where
row 9 is right, and — the part that was not expected — **no prompt anywhere on the screen**. The block was
intact and pinned at the bottom, the erased output was gone, no escape tail was visible: every assertion
the three existing `/cls` tests made was satisfied. One of them even looked at the input row and accepted a
blank one (`rows[ROWS - 3].contains(">") || rows[ROWS - 3].isBlank()`), which is precisely the defect.

**Why no prompt at all.** `LineReaderImpl.printAbove` calls `display.update(emptyList(), 0)`, prints, then
`redisplay(false)` — a **diff** against what `Display` believes is on screen. An erase invalidates that
belief exactly the way it invalidates `Status`'s, and `Status` has a documented three-step repair in this
console (`reset`, empty update, render again) while the reader's own display has none that a caller can
reach. So the erase did not merely misplace the prompt; it made the reader draw nothing.

**The fix is to stop erasing.** A clear now prints a window's worth of blank lines through `printAbove`
and does nothing else. Everything is pushed above the window, so the screen is blank; nothing is erased,
so nothing can be pulled back into view (the defect that killed the first attempt, blank rows *after* an
erase); the reader's bookkeeping stays right because printing above the prompt is what `printAbove` is
for; the pinned block is never touched, so all three repair steps went away with the erase; and the cursor
ends on its row **by construction**.

**That last property is the interesting one for the open defect above.** The recovery the reporter found by
hand — "nach ganz oft Enter sieht es wieder korrekt aus" — is the same mechanism: printing walks the cursor
back down one row per line. A clear now performs a window's worth of it in one keystroke, so **a screen
that has lost rows is repaired by `/cls` or Ctrl-L** instead of by holding Enter. That is not a fix for the
lost rows; it is the manual recovery made explicit and reachable, and it needs no cursor-position report,
which is what blocked the detection-based candidate.

**Ctrl-L had to be taken over for this.** JLine's keymap dispatches it by *name* to the widget registered
under `LineReader.CLEAR_SCREEN`, and JLine's own widget is `clear_screen` + `redrawLine()` — i.e. it
reproduced the identical defect, which the harness measured. Replacing the map entry re-points the key
without touching the keymap. It deliberately does not take this console's write lock: a widget runs on the
reader's thread with the reader's own lock held, while `line()` takes the write lock first and the reader's
lock second, so acquiring it there inverts the order and hangs the session. The reader's own lock
serialises the widget against every other `printAbove`, which is what matters; a concurrent block refresh
can still interleave, which is the exposure JLine's own widget has today as well.

## The console changes the screen behind JLine's back, and a diff cannot see it

The report after the clear was fixed: *"Kleiner ziehen sah ganz gut aus, größer macht noch Probleme"*, with a
screen showing a rule of one width and the activity row **cut short beside it on the same line**, the next
rule starting where that left off, and the state row below it looking perfectly correct.

**The mechanism, reproduced deterministically.** JLine pads every row of the pinned region to the width the
terminal *reports* (`Status.update`, against `display.columns`) and writes the rows one after another,
relying on the terminal wrapping at the right margin to begin the next one. When the screen is **wider**
than the reported width, the padding never reaches the margin, no wrap happens, and the next row continues
on the same screen line. A harness whose `getSize()` reports eight columns fewer than its screen has
produces the reported screen exactly:

```
16|>
17|──────────────────────────────────  … waitin
18|g for input …                        [▤ X:/tmp/agent-
19|sandbox · ⏸  manual · ▦ 0/16k · ⚒ 8 · ◆ local-model]
```

Note which row survives: the **last** one. The region addresses its first row and then writes on, so only
the rows in between collapse — which is why the artefact reads as a partial defect and why the state row
looked right in every report.

**Two different causes reach that state, and only one of them can be recovered from.** A console that
reports a width it has not applied yet does it *briefly*, and a rebuild afterwards repairs it — measured:
the transient form recovers with the existing settle redraw, and so does a window that additionally grows a
row taller (the screen model moves its content down with the region, as a real console does). What does
**not** recover is the other cause: **Windows reflows its screen buffer when the window is widened**, joining
rows it had marked as wrapped — which is every row of the region, since each one is padded to the last
column. JLine is never told. Its `Display` still matches what it wrote, so every later update computes an
**empty diff and emits nothing**, and the joined rows stay on screen for the rest of the session. That
accounts for the one thing no earlier theory did: the artefact *persisting* while every redraw runs.

**It also explains the recovery that was found by hand.** Holding Enter repaired it because printing does
not go through the region's diff at all.

**The fix: the settle redraw repaints instead of diffing.** `repaintBlockFromScratch()` hands the region a
block of the **same height** whose rows are blank, and then the real rows. The first pass makes the second
one a real write rather than an empty diff, and erases the rows on the way. Only the settle redraw does
this — a drag reports a size every ~125 ms and a full repaint on each is bytes spent against a screen about
to change again, while the reflow happens when the console applies the *final* size.

**`Status.reset()` is the call that looks right and is wrong.** It clears the display model *and* forgets the
scroll region, so the following update believes it must grow the region and scrolls to make room: the stale
rows were pushed **up** instead of being cleared and the block stood on screen **twice**, four rows apart.
Measured on the interpreted screen, which is the only place that difference is visible. Handing over an
empty block has the same problem for the same reason — it changes the region's height. Same count, different
content, is what invalidates the model without moving anything.

**What this cannot reach, stated plainly.** The **prompt** has a display of its own with the same diff and no
`reset()` a caller can call, so a reflow that damages the prompt's row is still unrepairable from here —
which is the same asymmetry `thePromptItselfStaysVisibleAfterEnlarging` has recorded all along. `/cls` and
Ctrl-L now repair that case, because they only print.

## The fifth fix: `Status.repaint()`, and why it had to be a new method

The repair above needs one thing the library does not offer: a way to tell the pinned region to draw its
rows again without assuming anything about what is on screen. Both candidates were measured and both are
wrong:

- **`Status.redraw()`** is `update(lines)` under another name, so it diffs like it — and it has to, being
  called from `LineReaderImpl.redisplay()` on **every keystroke**, where a diff is exactly what is wanted.
- **`Status.reset()`** clears the display model *and* sets `scrollRegion = display.rows`, so the next
  `update` takes its "we need to scroll up to grow the status bar" branch: the stale rows were pushed **up**
  instead of being cleared and the block stood on screen **twice**, four rows apart. Handing over an empty
  block does the same thing for the same reason — it changes the region's height. Handing over blank rows of
  the same height does work, but makes the block vanish for an instant; two unrelated screen cases caught
  that as a flicker, which on a real console would be a blink of the whole block at the end of every drag.

`Status.resize(Size)` already does the right thing — `display.reset()`, the scroll region, and clearing a
band of old remnants — but its whole body sits behind `if (display.rows != oldRows || display.columns !=
oldColumns)`. After a drag has settled the grid size has not changed, so it is skipped, and the following
update is an empty diff. **That is the precise reason a reflowed screen was never repaired.**

So: `public synchronized void repaint()` — `display.reset()` then `update(lines)`. It clears the model and
nothing else, so the following write covers every reserved row, the scroll region stays put, and nothing
blinks. Four tests in JLine's own style (`terminal/src/test/java/org/jline/utils/StatusRepaintTest.java`)
pin both sides: an `update` with unchanged lines leaves damage on screen, `redraw()` leaves it too,
`repaint()` puts every reserved row back **and** leaves every row above the block untouched (the assertion
that catches the `reset()` variant), and a repaint before anything was ever shown is not an error.

**On the consuming side it is called reflectively**, which none of the other four fixes needs. They change
how the library *behaves*, so the code compiles against the release and merely shows the symptom there; a
new *method* would make the released library fail to compile, and this project must stay buildable with
whatever JLine a copy of it finds (`llama-atmosphere-agent`'s pom names the released version on purpose).
Present: the block is repainted. Absent: nothing is forced, the artefact stays until something prints, and
`/cls` or Ctrl-L repairs it — the same trade the other four make.

**And the project's screen tests now skip themselves without the patched library.** CLAUDE.md has always
required this ("no project test may assert the fixed behaviour while the build depends on an unfixed
release") and it had been broken: CI builds the agent with the released JLine, against which between six and
thirteen of those cases fail — **a different set each run**, because the fourth fix is a data race and its
`ConcurrentModificationException`, once it lands on the reader's signal thread, ends that thread and takes
every later size change with it. There is therefore no fixed list to annotate, and the gate is the whole
class, keyed on `Status.repaint()` being present.

## The sixth fix, and the one that explains the screens full of rule fragments

The report after the repaint: *"größer und kleiner gemacht, cursor ist dann nicht unten"*, *"beim größer
ziehen wieder ganz viele striche unten"*, and after many drags a screen carrying **dozens** of rule
fragments at different widths with pieces of `… waiting for input` between them. `/cls` cleaned it up.

**One measurement settled it.** A status update emits this — on any terminal, at any width:

```
ESC[8;1H  ------------------------------------  <spaces>  ESC[D  working  <spaces>  ESC[D  [state]  <spaces>  ESC[D
```

**One cursor address for the whole bar.** The second row begins on a new screen row only because writing
the last column of the first one made the terminal wrap. That is sound exactly as long as `columns` is the
screen's real width — and while a window is being dragged it is not, nor after a terminal reflows its
buffer. The padding then never reaches the right margin, nothing wraps, and **every reserved row lands on
one screen row, side by side**. With a screen reporting eight columns fewer than it has, JLine's own test
suite shows the rule row reading `------------------------------  working` while the activity row's screen
row holds `[state]`.

**And that is why it accumulated.** The bar is reserved in rows counted from the **bottom**, so what the
collapse pushes past the window does not vanish — it lands in the **output area above** the bar, where
nothing ever writes again. One fragment per drag, and only something that scrolls can clear them, which is
exactly why `/cls` was the one thing that helped.

**The fix: the pinned region addresses each of its rows.** `Display` gains an overridable
`addressesEveryRow()` (false by default, so no other display changes) whose only effect in the update loop
is to switch off the pending-wrap compensation; `Status.MovingCursorDisplay` returns true and turns every
cursor move into an absolute address. Three variants were tried and two were measured wrong, both recorded
in the code:

- **Addressing each row at the top of the update loop.** Then an update whose content has not changed also
  saves and restores the cursor, which interleaves with the line reader's own writes: the typed text came
  out **one character per screen row** (`> H`, `a`, `l`, `l`, `o` down the screen). Deterministic, 2/2.
- **Addressing only the moves that land on a row start.** A row sharing a prefix with what is on screen is
  then not addressed, and since relying on the wrap is off, the pending wrap of the row above is never
  finished: the state row came out shifted one column, reading `" state]"` instead of `"[state]"`.
  Deterministic, 3/3.
- **Every move absolute.** The only variant where the cursor is where the display believes it is. For a bar
  of a few rows it is a handful of bytes per update.

**Verified:** 101 tests in JLine's own suite green (its `DisplayTest` and `ScreenTerminalTest` included),
with `StatusWrongWidthTest` red 2/3 against the unpatched library and green with the fix; the project's 206
tests green against `4.4.6-statusfix6` and green against the released `4.4.6`, where the screen cases skip.

## Two things the sixth fix got wrong, and the row the prompt loses

The report after it: *"nach cls und wieder größer ziehen"*, with the prompt at the left of one row and the
whole block far to the right of that same row. Reproducing the sequence found **two separate things**, and
only one of them was mine to fix.

**1. The sixth fix had an off-by-one per row, and JLine's own tests could not see it.** `Display` counts
positions in **`columns + 1`** per row (`columns1`) throughout, and the new absolute address divided by
`columns`. So row *N* was addressed at column *N*: rule at column 0, activity row at 1, state row at 2.
Every existing test compares **trimmed** rows, which is exactly what hides it — it took an interpreted
screen in the consuming project to see it. `StatusWrongWidthTest` now carries
`everyRowStartsAtColumnZero`, which asserts on the raw row.

**2. The prompt is never addressed at all, and that is the row the user keeps losing.** Bisected on the
interpreted screen, with the block the application really pins:

| after | cursor row | should be |
|---|---|---|
| `/cls` | 16 | 16 |
| **JLine's WINCH handling alone** | **19** | 20 |
| plus this console's row rebuild | 19 | 20 |
| plus this console's repaint | 19 | 20 |

So it is neither of ours. The byte stream for a growing window says why:

```
ESC7  ESC[8;1H ESC[K … ESC[14;1H ESC[K   ESC[1;11r  ESC8   CR
ESC7 ESC[12;1H ------  ESC[13;1H working  ESC[14;1H [state]  ESC8
>
```

The status region clears its band, re-establishes the scroll region and restores the cursor; the block is
addressed row by row (the sixth fix); and then the prompt is written as a bare `>` — **wherever the cursor
happens to be.** Nothing relates it to the region below it. When the window grows, the screen moves its
content down by as many rows as it has scrollback to pull from, which need not equal the number of rows
added, so the prompt ends up a row or two above the rule with a blank row between.

**Why this is not a defect to patch blindly.** A prompt sitting directly above the pinned region is what
*this* application wants — it scrolls to the bottom once at startup so the input is always on the last
usable row. JLine makes no such promise: it draws the prompt at the cursor, which for a half-empty screen is
correctly somewhere in the middle. "Move the cursor to the bottom of the scroll region on a resize" would be
right here and wrong there, so it is a change to the library's contract rather than a bug fix, and it is
recorded as the open question it is. `ScreenUseCasesTest.aGrowingWindowLeavesThePromptOnTheRowTheBlockLeavesForIt`
carries the reproduction with the bisection in its `@Disabled` text.

**What does repair it, and now says so at startup.** Printing moves the cursor back down one row per line, so
`/cls` — a window's worth of blank lines — puts it back on its row in one keystroke, which the measurement
above confirms (16 of 16 after a clear). And because two of these rounds chased a defect whose fix the jar in
use did not contain, the agent now prints the terminal type and the **patch level of the JLine it is actually
running with**, probed by method (`Status.repaint` for the fifth fix, `Display.addressesEveryRow` for the
sixth) rather than by version string — the patched builds overlay classes into the released jar and keep its
manifest version, so the version string cannot tell them apart.
