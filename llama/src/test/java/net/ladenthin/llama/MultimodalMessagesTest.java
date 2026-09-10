// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.hasSize;
import static org.hamcrest.Matchers.is;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.fail;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import net.ladenthin.llama.parameters.InferenceParameters;
import net.ladenthin.llama.parameters.ParameterJsonSerializer;
import net.ladenthin.llama.value.ChatMessage;
import net.ladenthin.llama.value.ContentPart;
import org.junit.jupiter.api.Test;

@ClaudeGenerated(
        purpose = "Verify multimodal ChatMessage flow: parts-based constructor, getParts()/hasParts(), "
                + "userMultimodal factory, and that ParameterJsonSerializer.buildMessages(List<ChatMessage>) "
                + "emits the OAI array-form content for parts-bearing messages while keeping the "
                + "string-form content for plain text messages (drop-in compatibility).")
public class MultimodalMessagesTest {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    @Test
    public void hasPartsIsFalseForLegacyConstructor() {
        ChatMessage m = new ChatMessage("user", "hello");
        assertThat(m.hasParts(), is(false));
        assertThat(m.getParts().isPresent(), is(false));
    }

    @Test
    public void hasPartsIsTrueForPartsConstructor() {
        ChatMessage m = new ChatMessage(
                "user", Arrays.asList(ContentPart.text("hi"), ContentPart.imageUrl("data:image/png;base64,AAAA")));
        assertThat(m.hasParts(), is(true));
        assertThat(m.getParts().orElseThrow(), hasSize(2));
    }

    @Test
    public void contentFieldConcatenatesTextPartsForLegacyReaders() {
        ChatMessage m = new ChatMessage(
                "user",
                Arrays.asList(
                        ContentPart.text("describe"),
                        ContentPart.imageUrl("data:image/png;base64,X"),
                        ContentPart.text("please")));
        // Image parts contribute no text; text parts are newline-joined.
        assertThat(m.getContent(), is("describe\nplease"));
    }

    @Test
    public void userMultimodalFactoryBuildsUserMessage() {
        ChatMessage m = ChatMessage.userMultimodal(
                ContentPart.text("what is this?"), ContentPart.imageUrl("data:image/jpeg;base64,Y"));
        assertThat(m.getRole(), is("user"));
        List<ContentPart> parts = m.getParts().orElseThrow();
        assertThat(parts, hasSize(2));
        assertThat(parts.get(0).getType(), is(ContentPart.Type.TEXT));
        assertThat(parts.get(1).getType(), is(ContentPart.Type.IMAGE_URL));
    }

    @Test
    public void emptyPartsListIsRejected() {
        assertThrows(
                IllegalArgumentException.class, () -> new ChatMessage("user", Collections.<ContentPart>emptyList()));
    }

    @Test
    public void nullPartsListIsRejected() {
        assertThrows(IllegalArgumentException.class, () -> new ChatMessage("user", (List<ContentPart>) null));
    }

    @Test
    public void getPartsListIsUnmodifiable() {
        ChatMessage m = ChatMessage.userMultimodal(ContentPart.text("x"));
        try {
            m.getParts().orElseThrow().add(ContentPart.text("y"));
            fail("getParts() must return an unmodifiable list");
        } catch (UnsupportedOperationException expected) {
            // ok
        }
    }

    @Test
    public void serializerEmitsArrayContentForPartsMessage() throws Exception {
        ParameterJsonSerializer s = new ParameterJsonSerializer();
        ChatMessage user = ChatMessage.userMultimodal(
                ContentPart.text("describe"), ContentPart.imageUrl("data:image/png;base64,ABCD"));
        ArrayNode arr = s.buildMessages(Collections.singletonList(user));

        assertThat(arr.size(), is(1));
        JsonNode msg = arr.get(0);
        assertThat(msg.get("role").asText(), is("user"));

        JsonNode content = msg.get("content");
        assertThat("content must be an array when parts are present", content.isArray(), is(true));
        assertThat(content.size(), is(2));

        JsonNode p0 = content.get(0);
        assertThat(p0.get("type").asText(), is("text"));
        assertThat(p0.get("text").asText(), is("describe"));

        JsonNode p1 = content.get(1);
        assertThat(p1.get("type").asText(), is("image_url"));
        assertThat(p1.get("image_url").get("url").asText(), is("data:image/png;base64,ABCD"));
    }

    @Test
    public void serializerEmitsStringContentForLegacyMessage() {
        ParameterJsonSerializer s = new ParameterJsonSerializer();
        ChatMessage user = new ChatMessage("user", "plain text");
        ArrayNode arr = s.buildMessages(Collections.singletonList(user));

        assertThat(arr.size(), is(1));
        JsonNode msg = arr.get(0);
        assertThat(msg.get("role").asText(), is("user"));
        assertThat(
                "content must remain a string for legacy messages",
                msg.get("content").isTextual(),
                is(true));
        assertThat(msg.get("content").asText(), is("plain text"));
    }

    @Test
    public void serializerHandlesMixedMessages() {
        ParameterJsonSerializer s = new ParameterJsonSerializer();
        List<ChatMessage> messages = Arrays.asList(
                new ChatMessage("system", "You are a helper."),
                ChatMessage.userMultimodal(
                        ContentPart.text("what's in here?"), ContentPart.imageUrl("data:image/png;base64,Z")),
                new ChatMessage("assistant", "a cat"));
        ArrayNode arr = s.buildMessages(messages);
        assertThat(arr.size(), is(3));
        assertThat(arr.get(0).get("content").isTextual(), is(true));
        assertThat(arr.get(1).get("content").isArray(), is(true));
        assertThat(arr.get(2).get("content").isTextual(), is(true));
    }

    @Test
    public void inferenceParametersAcceptsMultimodalMessages() {
        InferenceParameters params = new InferenceParameters("")
                .withMessages(Collections.singletonList(ChatMessage.userMultimodal(
                        ContentPart.text("hi"), ContentPart.imageUrl("data:image/png;base64,QQ"))));
        // setMessages encodes into the parameters map under "messages"; verify the
        // resulting JSON has the array form, which is what the upstream OAI chat
        // parser expects for multimodal routing.
        String json = params.toJson();
        assertThat("messages array must be present", json, containsString("\"messages\""));
        assertThat("multimodal part type must be in the serialised JSON", json, containsString("\"image_url\""));
        assertThat("data URI must round-trip into the request body", json, containsString("data:image/png;base64,QQ"));
    }
}
