// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.atmosphere;

/**
 * How a turn ended: on its own, because it was stopped, or because it ran too long.
 *
 * <p>Three values rather than a boolean, because <em>interrupted</em> must not be reported as the
 * <em>timed out</em> error a {@code false} used to produce.
 */
public enum TurnEnd {
    /** The model produced its final answer (or errored). */
    FINISHED,
    /** The user asked for something else while it was working; the turn was cut short. */
    INTERRUPTED,
    /** Nothing arrived within {@link AgentSession#TURN_TIMEOUT}. */
    TIMED_OUT
}
