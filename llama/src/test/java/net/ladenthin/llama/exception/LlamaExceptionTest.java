// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.exception;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.instanceOf;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.nullValue;

import net.ladenthin.llama.ClaudeGenerated;
import org.junit.jupiter.api.Test;

@ClaudeGenerated(
        purpose = "Verify that LlamaException correctly propagates its message through the "
                + "RuntimeException hierarchy, handles null messages without error, and can "
                + "be thrown and caught as an unchecked exception.")
public class LlamaExceptionTest {

    @Test
    public void testMessageIsPreserved() {
        LlamaException ex = new LlamaException("something went wrong");
        assertThat(ex.getMessage(), is("something went wrong"));
    }

    @Test
    public void testIsRuntimeException() {
        LlamaException ex = new LlamaException("error");
        assertThat(ex, is(instanceOf(RuntimeException.class)));
    }

    @Test
    public void testEmptyMessage() {
        LlamaException ex = new LlamaException("");
        assertThat(ex.getMessage(), is(""));
    }

    @Test
    public void testNullMessage() {
        LlamaException ex = new LlamaException(null);
        assertThat(ex.getMessage(), is(nullValue()));
    }

    @Test
    public void testCanBeThrown() {
        boolean caught = false;
        try {
            throw new LlamaException("thrown");
        } catch (LlamaException e) {
            assertThat(e.getMessage(), is("thrown"));
            caught = true;
        }
        assertThat("Expected LlamaException to be thrown", caught, is(true));
    }
}
