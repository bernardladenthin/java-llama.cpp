// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.atmosphere;

import java.util.concurrent.atomic.AtomicBoolean;

/**
 * A switch that stops the activity line from redrawing while something else owns the terminal.
 *
 * <p>It exists because a turn runs on its own thread. Atmosphere's {@code execute} is synchronous —
 * it returns only when the whole turn including every tool round is done — so the console thread has
 * to drive the spinner while a second thread runs the turn. That is fine until the approval prompt
 * appears: it reads a single key in raw mode on the turn's thread, and a status redraw arriving from
 * the console thread in the middle of that writes escape sequences across the question. So the prompt
 * pauses the redraw for as long as it is waiting for an answer.
 */
public final class TurnActivity {

    private final AtomicBoolean paused = new AtomicBoolean();

    /** Stop redrawing the activity line. */
    public void pause() {
        paused.set(true);
    }

    /** Redraw it again. */
    public void resume() {
        paused.set(false);
    }

    /**
     * Whether redrawing is currently suspended.
     *
     * @return {@code true} while something else owns the terminal
     */
    public boolean isPaused() {
        return paused.get();
    }
}
