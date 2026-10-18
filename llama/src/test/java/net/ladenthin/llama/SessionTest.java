// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT
package net.ladenthin.llama;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.containsString;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

/**
 * Model-free unit tests for {@link Session} construction invariants. A session is pinned to one
 * concrete slot for both inference and {@link Session#save(String)} / {@link Session#restore(String)} /
 * {@link Session#close()}, so a negative slot id is rejected at construction rather than surfacing
 * deep inside the first {@code send()} call (where {@code InferenceParameters.withSlotId} would throw).
 */
public class SessionTest {

    @Test
    public void negativeSlotIdRejectedAtConstruction() {
        // The slot-id guard runs before the model is dereferenced, so a null model still exercises it.
        IllegalArgumentException ex = assertThrows(IllegalArgumentException.class, () -> new Session(null, -1, null));
        assertThat(ex.getMessage(), containsString("slotId must be non-negative"));
    }

    @Test
    public void negativeSlotIdRejectedOnCustomizerConstructor() {
        assertThrows(IllegalArgumentException.class, () -> new Session(null, -7, null, p -> p));
    }

    @Test
    public void forkToANegativeSlotIdRejectedBeforeTheModelIsTouched() {
        // A null model: any model call (saving the slot first) would surface as a NullPointerException.
        Session session = new Session(null, 0, null);
        IllegalArgumentException ex = assertThrows(IllegalArgumentException.class, () -> session.fork(-1, "fork.bin"));
        assertThat(ex.getMessage(), containsString("slotId must be non-negative"));
    }
}
