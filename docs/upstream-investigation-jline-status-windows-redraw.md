<!--
SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>

SPDX-License-Identifier: MIT
-->

# JLine on Windows: a constant-size `Status` makes every input redraw append

**Status:** narrowed to JLine with no project code involved; not yet filed upstream.
**Affects:** JLine 4.4.5 **and** 4.4.6, terminal type `windows-vtp` (Windows Terminal / PowerShell).
**Where it shows in this repo:** `llama-atmosphere-agent`, whose console pins a three-row status block.

## Symptom

Start a session, type a few characters **without** pressing Enter, then widen the window:

```
> Hallo> Hallo> Hallo> Hallo> Hallo> Hallo> Hallo> Hallo> Hallo> Hallo> Hallo> Hallo
```

The prompt and the input buffer appear once per keystroke, side by side on one logical line.

Two details matter for reading the symptom correctly, and both were misread at first:

- **It is not caused by the resize.** The repeats are written while typing. At the original width
  everything past the right edge is clipped and therefore invisible; widening the window reveals what
  was already in the console's line buffer. That is why it was first reported as a resize defect.
- **The count follows the number of keystrokes**, not the number of drag events — five characters gave
  twelve to seventeen repeats across runs, never a number that tracked the dragging.

## Reproduction without any project code

`ResizeProbe` builds a system terminal, attaches a three-row `Status`, and reads one line — that is
all. It reproduces. The same probe with the `Status` block removed does **not** reproduce, on the same
terminal, in the same window, with the same typing:

| variant | pinned `Status` | result |
|---|---|---|
| A | three rows | `> Hallo` repeated per keystroke |
| C | none | one `> Hallo`, as expected |

## What was ruled out, and how

| hypothesis | test | result |
|---|---|---|
| Project code (the agent's console) | run the probe, which contains none of it | still reproduces |
| Fixed in a newer JLine | same probe against 4.4.6 | still reproduces (17 repeats) |
| Driven by resize signals | `TerminalBuilder.nativeSignals(false)` | still reproduces |
| Missing terminfo capability | `windows-vtp.caps` carries `sc`, `rc`, `csr`, `cup`, `el`, `cr` | all present (`ri` is absent) |

One measurement attempt is recorded because it was **inconclusive rather than negative**: a `WINCH`
handler registered by the probe logged nothing, which does *not* mean no size events arrived —
`LineReaderImpl.readLine()` installs its own handler and restores the previous one afterwards, so a
handler registered before the read never runs during it.

A second one failed for a reason worth knowing: teeing `System.out` captured **zero bytes**, because on
Windows JLine writes through the console API rather than through `System.out`. Capturing the emitted
byte stream from inside the same JVM is therefore not straightforward on this platform.

## The mechanism this points at

`LineReaderImpl.redisplay()` (4.4.5, around line 4260) re-synchronises its cursor tracking with the
status block in exactly one place:

```java
Status status = Status.getStatus(terminal, false);
int currentStatusSize = status != null ? status.size() : 0;
if (currentStatusSize != lastStatusSize) {
    // Status bar appeared, grew, or shrank (e.g. from a background
    // thread). Content may have scrolled, invalidating cursor tracking.
    terminal.puts(Capability.carriage_return);
    doDisplay();
    lastStatusSize = currentStatusSize;
}
if (status != null) {
    status.redraw();
}
```

The guard is on the status **size**. A block whose height never changes — three rows, every redraw —
passes that test once and never again, while `Status.redraw()` keeps moving the real cursor on every
call: `Status.MovingCursorDisplay.initCursor()` emits `save_cursor` followed by
`cursor_address(firstLine, 0)`, and the matching `restore_cursor` is emitted only when something was
actually drawn (`cursorPos != -1`). The comment in that guard describes precisely the situation that
then goes unhandled: *content may have scrolled, invalidating cursor tracking*.

That is a hypothesis consistent with every measurement above, not a proven root cause: the remaining
step is to confirm that a status block whose row count **alternates** does not reproduce, which would
make `lastStatusSize` the discriminator outright.

## Consequences for this project

- Nothing in `llama-atmosphere-agent` causes it, and nothing there can honestly fix it. An earlier
  attempt to help by installing a `WINCH` handler made the display **worse**, because the line reader
  owns that signal for the whole session — see `CLAUDE.md`, "Do not add a `WINCH` handler".
- `--plain` is unaffected and is the escape hatch: it pins nothing and redraws nothing.
- The probes live in this session's scratchpad rather than the repo, because they are single-purpose
  harnesses; this document carries what they established.
