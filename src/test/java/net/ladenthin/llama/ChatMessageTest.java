// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

@ClaudeGenerated(
        purpose = "Verify ChatMessage value class accessors and toString format used by Session.getMessages()."
)
public class ChatMessageTest {

    @Test
    public void accessors() {
        ChatMessage m = new ChatMessage("user", "hi");
        assertEquals("user", m.getRole());
        assertEquals("hi", m.getContent());
    }

    @Test
    public void toStringFormat() {
        assertEquals("assistant: hello", new ChatMessage("assistant", "hello").toString());
    }
}
