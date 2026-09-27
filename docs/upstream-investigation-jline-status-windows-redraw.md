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
