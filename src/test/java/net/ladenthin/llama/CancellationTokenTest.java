// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import org.junit.Test;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

@ClaudeGenerated(
        purpose = "Verify CancellationToken state transitions (initial, cancel, reset) "
                + "and idempotency of cancel(). Cooperative cancellation behaviour during "
                + "a live inference loop is exercised in LlamaModelTest."
)
public class CancellationTokenTest {

    @Test
    public void initiallyNotCancelled() {
        assertFalse(new CancellationToken().isCancelled());
    }

    @Test
    public void cancelFlipsState() {
        CancellationToken t = new CancellationToken();
        t.cancel();
        assertTrue(t.isCancelled());
    }

    @Test
    public void cancelIsIdempotent() {
        CancellationToken t = new CancellationToken();
        t.cancel();
        t.cancel();
        t.cancel();
        assertTrue(t.isCancelled());
    }

    @Test
    public void resetClearsCancelledFlag() {
        CancellationToken t = new CancellationToken();
        t.cancel();
        assertTrue(t.isCancelled());
        t.reset();
        assertFalse(t.isCancelled());
    }

    @Test
    public void cancelBeforeUseIsObserved() {
        // cancel() before any inference loop sees the token should still flip the flag.
        CancellationToken t = new CancellationToken();
        t.cancel();
        assertTrue(t.isCancelled());
    }
}
