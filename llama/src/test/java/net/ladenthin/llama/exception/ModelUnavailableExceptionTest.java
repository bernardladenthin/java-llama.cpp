// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.exception;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.instanceOf;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.nullValue;
import static org.hamcrest.Matchers.sameInstance;

import net.ladenthin.llama.ClaudeGenerated;
import org.junit.jupiter.api.Test;

@ClaudeGenerated(
        purpose = "Verify the typed-exception unification shape of ModelUnavailableException: the "
                + "(message) and (message, cause) constructor matrix, that it is a typed subclass of "
                + "LlamaException (so callers can catch it by the common base), and that it can be "
                + "thrown and caught.")
public class ModelUnavailableExceptionTest {

    @Test
    public void testMessageIsPreserved() {
        ModelUnavailableException ex = new ModelUnavailableException("model file missing");
        assertThat(ex.getMessage(), is("model file missing"));
    }

    @Test
    public void testMessageAndCausePreserved() {
        Throwable cause = new IllegalStateException("offline set");
        ModelUnavailableException ex = new ModelUnavailableException("model file missing", cause);
        assertThat(ex.getMessage(), is("model file missing"));
        assertThat(ex.getCause(), is(sameInstance(cause)));
    }

    @Test
    public void testIsLlamaException() {
        ModelUnavailableException ex = new ModelUnavailableException("error");
        assertThat(ex, is(instanceOf(LlamaException.class)));
    }

    @Test
    public void testIsRuntimeException() {
        ModelUnavailableException ex = new ModelUnavailableException("error");
        assertThat(ex, is(instanceOf(RuntimeException.class)));
    }

    @Test
    public void testNullMessage() {
        ModelUnavailableException ex = new ModelUnavailableException(null);
        assertThat(ex.getMessage(), is(nullValue()));
    }

    @Test
    public void testCanBeCaughtAsLlamaException() {
        boolean caught = false;
        try {
            throw new ModelUnavailableException("thrown");
        } catch (LlamaException e) {
            assertThat(e.getMessage(), is("thrown"));
            caught = true;
        }
        assertThat("Expected ModelUnavailableException to be catchable as LlamaException", caught, is(true));
    }
}
