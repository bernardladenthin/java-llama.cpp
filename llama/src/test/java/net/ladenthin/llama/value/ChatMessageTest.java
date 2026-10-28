// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.value;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.empty;
import static org.hamcrest.Matchers.hasSize;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.not;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import net.ladenthin.llama.ClaudeGenerated;
import org.junit.jupiter.api.Test;

@ClaudeGenerated(
        purpose = "Pin every ChatMessage path: plain/tool/multimodal constructors and factories, the "
                + "concatText text-joining (newline-joined, image parts skipped, no leading newline), the "
                + "parts validation helpers (null/empty rejection), the three toString branches "
                + "(plain, tool_calls, tool_call_id), and value-based equals/hashCode — full mutation coverage.")
public class ChatMessageTest {

    @Test
    public void plainMessageAccessors() {
        ChatMessage m = new ChatMessage("user", "hi");
        assertThat(m.getRole(), is("user"));
        assertThat(m.getContent(), is("hi"));
        assertThat(m.hasParts(), is(false));
        assertThat(m.getParts().isPresent(), is(false));
        assertThat(m.getToolCalls(), is(empty()));
        assertThat(m.getToolCallId().isPresent(), is(false));
    }

    @Test
    public void toStringPlainBranch() {
        assertThat(new ChatMessage("assistant", "hello").toString(), is("assistant: hello"));
    }

    @Test
    public void toStringToolCallsBranch() {
        ChatMessage m =
                ChatMessage.assistantToolCalls("thinking", Collections.singletonList(new ToolCall("c1", "f", "{}")));
        assertThat(m.toString(), is("assistant (tool_calls=1): thinking"));
    }

    @Test
    public void toolResultWithPartsKeepsThePartsAndTheCallId() {
        ChatMessage m = ChatMessage.toolResult("c0", ContentPart.text("found"), ContentPart.toolReference("buy"));
        assertThat(m.getRole(), is("tool"));
        assertThat(m.getToolCallId().orElseThrow(), is("c0"));
        assertThat(m.hasParts(), is(true));
        List<ContentPart> parts = m.getParts().orElseThrow();
        assertThat(parts, hasSize(2));
        assertThat(parts.get(1).getToolName(), is("buy"));
        // the text of the TEXT parts is the plain content; a tool reference has none
        assertThat(m.getContent(), is("found"));
        assertThrows(UnsupportedOperationException.class, () -> parts.add(ContentPart.text("x")));
    }

    @Test
    public void toolResultWithoutPartsIsRefused() {
        assertThrows(IllegalArgumentException.class, () -> ChatMessage.toolResult("c0", new ContentPart[0]));
    }

    @Test
    public void toStringToolCallIdBranch() {
        ChatMessage m = ChatMessage.toolResult("c1", "42");
        assertThat(m.getRole(), is("tool"));
        assertThat(m.getToolCallId().orElseThrow(), is("c1"));
        assertThat(m.toString(), is("tool (tool_call_id=c1): 42"));
    }

    @Test
    public void assistantToolCallsNullContentBecomesEmpty() {
        // L144 ternary: content == null ? "" : content
        ChatMessage m = ChatMessage.assistantToolCalls(null, Collections.singletonList(new ToolCall("c1", "f", "{}")));
        assertThat(m.getContent(), is(""));
        assertThat(m.getToolCalls(), hasSize(1));
    }

    @Test
    public void toolCallsListIsDefensivelyCopiedAndUnmodifiable() {
        java.util.List<ToolCall> source = new java.util.ArrayList<ToolCall>();
        source.add(new ToolCall("c1", "f", "{}"));
        ChatMessage m = ChatMessage.assistantToolCalls("", source);

        // Mutating the caller's list must not change the (documented-immutable) message.
        source.clear();
        assertThat(m.getToolCalls(), hasSize(1));

        // And the returned list is read-only.
        assertThrows(UnsupportedOperationException.class, () -> m.getToolCalls().add(new ToolCall("c2", "g", "{}")));
    }

    @Test
    public void assistantToolCallsKeepsNonNullContent() {
        ChatMessage m =
                ChatMessage.assistantToolCalls("reason", Collections.singletonList(new ToolCall("c1", "f", "{}")));
        assertThat(m.getContent(), is("reason"));
    }

    @Test
    public void multimodalConcatenatesTextPartsSkippingImagesNoLeadingNewline() {
        // concatText: text parts newline-joined, image parts skipped, first part not prefixed with '\n'.
        ChatMessage m = new ChatMessage(
                "user",
                Arrays.asList(
                        ContentPart.text("describe"),
                        ContentPart.imageUrl("data:image/png;base64,X"),
                        ContentPart.text("please")));
        assertThat(m.getContent(), is("describe\nplease"));
        assertThat(m.hasParts(), is(true));
        assertThat(m.getParts().orElseThrow(), hasSize(3));
    }

    @Test
    public void userMultimodalFactoryBuildsUserMessageWithParts() {
        // L155: factory must return a real instance (not null) carrying the parts.
        ChatMessage m =
                ChatMessage.userMultimodal(ContentPart.text("a"), ContentPart.imageUrl("data:image/png;base64,Y"));
        assertThat(m.getRole(), is("user"));
        assertThat(m.hasParts(), is(true));
        assertThat(m.getParts().orElseThrow(), hasSize(2));
    }

    @Test
    public void nullPartsRejected() {
        assertThrows(IllegalArgumentException.class, () -> new ChatMessage("user", (List<ContentPart>) null));
    }

    @Test
    public void emptyPartsRejected() {
        assertThrows(
                IllegalArgumentException.class, () -> new ChatMessage("user", Collections.<ContentPart>emptyList()));
    }

    @Test
    public void equalsAndHashCodeAreValueBased() {
        assertThat(new ChatMessage("user", "hi"), is(new ChatMessage("user", "hi")));
        assertThat(new ChatMessage("user", "hi").hashCode(), is(new ChatMessage("user", "hi").hashCode()));
    }

    @Test
    public void differingContentBreaksEquality() {
        assertThat(new ChatMessage("user", "hi"), is(not(new ChatMessage("user", "bye"))));
    }
}
