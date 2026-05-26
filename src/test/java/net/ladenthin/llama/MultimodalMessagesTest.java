// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;

import net.ladenthin.llama.json.ParameterJsonSerializer;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

@ClaudeGenerated(
        purpose = "Verify multimodal ChatMessage flow: parts-based constructor, getParts()/hasParts(), "
                + "userMultimodal factory, and that ParameterJsonSerializer.buildMessages(List<ChatMessage>) "
                + "emits the OAI array-form content for parts-bearing messages while keeping the "
                + "string-form content for plain text messages (drop-in compatibility)."
)
public class MultimodalMessagesTest {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    @Test
    public void hasPartsIsFalseForLegacyConstructor() {
        ChatMessage m = new ChatMessage("user", "hello");
        assertFalse(m.hasParts());
        assertEquals(null, m.getParts());
    }

    @Test
    public void hasPartsIsTrueForPartsConstructor() {
        ChatMessage m = new ChatMessage("user",
                Arrays.asList(ContentPart.text("hi"),
                              ContentPart.imageUrl("data:image/png;base64,AAAA")));
        assertTrue(m.hasParts());
        assertEquals(2, m.getParts().size());
    }

    @Test
    public void contentFieldConcatenatesTextPartsForLegacyReaders() {
        ChatMessage m = new ChatMessage("user",
                Arrays.asList(ContentPart.text("describe"),
                              ContentPart.imageUrl("data:image/png;base64,X"),
                              ContentPart.text("please")));
        // Image parts contribute no text; text parts are newline-joined.
        assertEquals("describe\nplease", m.getContent());
    }

    @Test
    public void userMultimodalFactoryBuildsUserMessage() {
        ChatMessage m = ChatMessage.userMultimodal(
                ContentPart.text("what is this?"),
                ContentPart.imageUrl("data:image/jpeg;base64,Y"));
        assertEquals("user", m.getRole());
        assertEquals(2, m.getParts().size());
        assertEquals(ContentPart.Type.TEXT, m.getParts().get(0).getType());
        assertEquals(ContentPart.Type.IMAGE_URL, m.getParts().get(1).getType());
    }

    @Test
    public void emptyPartsListIsRejected() {
        assertThrows(IllegalArgumentException.class, () -> new ChatMessage("user", Collections.<ContentPart>emptyList()));
    }

    @Test
    public void nullPartsListIsRejected() {
        assertThrows(IllegalArgumentException.class, () -> new ChatMessage("user", (List<ContentPart>) null));
    }

    @Test
    public void getPartsListIsUnmodifiable() {
        ChatMessage m = ChatMessage.userMultimodal(ContentPart.text("x"));
        try {
            m.getParts().add(ContentPart.text("y"));
            fail("getParts() must return an unmodifiable list");
        } catch (UnsupportedOperationException expected) {
            // ok
        }
    }

    @Test
    public void serializerEmitsArrayContentForPartsMessage() throws Exception {
        ParameterJsonSerializer s = new ParameterJsonSerializer();
        ChatMessage user = ChatMessage.userMultimodal(
                ContentPart.text("describe"),
                ContentPart.imageUrl("data:image/png;base64,ABCD"));
        ArrayNode arr = s.buildMessages(Collections.singletonList(user));

        assertEquals(1, arr.size());
        JsonNode msg = arr.get(0);
        assertEquals("user", msg.get("role").asText());

        JsonNode content = msg.get("content");
        assertTrue(content.isArray(), "content must be an array when parts are present");
        assertEquals(2, content.size());

        JsonNode p0 = content.get(0);
        assertEquals("text", p0.get("type").asText());
        assertEquals("describe", p0.get("text").asText());

        JsonNode p1 = content.get(1);
        assertEquals("image_url", p1.get("type").asText());
        assertEquals("data:image/png;base64,ABCD", p1.get("image_url").get("url").asText());
    }

    @Test
    public void serializerEmitsStringContentForLegacyMessage() {
        ParameterJsonSerializer s = new ParameterJsonSerializer();
        ChatMessage user = new ChatMessage("user", "plain text");
        ArrayNode arr = s.buildMessages(Collections.singletonList(user));

        assertEquals(1, arr.size());
        JsonNode msg = arr.get(0);
        assertEquals("user", msg.get("role").asText());
        assertTrue(msg.get("content").isTextual(), "content must remain a string for legacy messages");
        assertEquals("plain text", msg.get("content").asText());
    }

    @Test
    public void serializerHandlesMixedMessages() {
        ParameterJsonSerializer s = new ParameterJsonSerializer();
        List<ChatMessage> messages = Arrays.asList(
                new ChatMessage("system", "You are a helper."),
                ChatMessage.userMultimodal(
                        ContentPart.text("what's in here?"),
                        ContentPart.imageUrl("data:image/png;base64,Z")),
                new ChatMessage("assistant", "a cat")
        );
        ArrayNode arr = s.buildMessages(messages);
        assertEquals(3, arr.size());
        assertTrue(arr.get(0).get("content").isTextual());
        assertTrue(arr.get(1).get("content").isArray());
        assertTrue(arr.get(2).get("content").isTextual());
    }

    @Test
    public void inferenceParametersAcceptsMultimodalMessages() {
        InferenceParameters params = new InferenceParameters("");
        params.setMessages(Collections.singletonList(
                ChatMessage.userMultimodal(
                        ContentPart.text("hi"),
                        ContentPart.imageUrl("data:image/png;base64,QQ"))));
        // setMessages encodes into the parameters map under "messages"; verify the
        // resulting JSON has the array form, which is what the upstream OAI chat
        // parser expects for multimodal routing.
        String json = params.toString();
        assertTrue(json.contains("\"messages\""), "messages array must be present");
        assertTrue(json.contains("\"image_url\""), "multimodal part type must be in the serialised JSON");
        assertTrue(json.contains("data:image/png;base64,QQ"), "data URI must round-trip into the request body");
    }
}
