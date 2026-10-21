// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.parameters;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.empty;
import static org.hamcrest.Matchers.hasSize;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.not;
import static org.hamcrest.Matchers.sameInstance;
import static org.junit.jupiter.api.Assertions.assertThrows;

import net.ladenthin.llama.value.ChatMessage;
import net.ladenthin.llama.value.ContentPart;
import net.ladenthin.llama.value.ToolDefinition;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

/**
 * Running documentation of the {@link ChatRequest} immutability + wither-pattern
 * contract. Every modification method returns a NEW request; the original is
 * never mutated. Two requests with the same content compare equal regardless
 * of identity.
 */
class ChatRequestTest {

    @Nested
    @DisplayName("immutability — every modifier returns a fresh instance")
    class Immutability {

        @Test
        void appendMessageReturnsNewInstance() {
            ChatRequest original = ChatRequest.empty();
            ChatRequest derived = original.appendMessage("user", "hi");
            assertThat(derived, is(not(sameInstance(original))));
            assertThat("original is untouched", original.getMessages(), is(empty()));
            assertThat("derived has the message", derived.getMessages(), hasSize(1));
        }

        @Test
        void appendToolReturnsNewInstance() {
            ChatRequest original = ChatRequest.empty();
            ChatRequest derived = original.appendTool(new ToolDefinition("echo", "Echo", "{}"));
            assertThat(derived, is(not(sameInstance(original))));
            assertThat(original.getTools(), is(empty()));
            assertThat(derived.getTools(), hasSize(1));
        }

        @Test
        void withToolChoiceReturnsNewInstance() {
            ChatRequest original = ChatRequest.empty();
            ChatRequest derived = original.withToolChoice("auto");
            assertThat(derived, is(not(sameInstance(original))));
            assertThat("original toolChoice unset", original.getToolChoice().isPresent(), is(false));
            assertThat(derived.getToolChoice().orElseThrow(), is("auto"));
        }

        @Test
        void withParallelToolCallsReturnsNewInstance() {
            ChatRequest original = ChatRequest.empty();
            ChatRequest derived = original.withParallelToolCalls(Boolean.FALSE);
            assertThat(derived, is(not(sameInstance(original))));
            assertThat("original hint unset", original.getParallelToolCalls().isPresent(), is(false));
            assertThat(derived.getParallelToolCalls().orElseThrow(), is(false));
        }

        @Test
        void withMaxToolRoundsReturnsNewInstance() {
            ChatRequest original = ChatRequest.empty();
            ChatRequest derived = original.withMaxToolRounds(2);
            assertThat(derived, is(not(sameInstance(original))));
            assertThat(original.getMaxToolRounds(), is(ChatRequest.DEFAULT_MAX_TOOL_ROUNDS));
            assertThat(derived.getMaxToolRounds(), is(2));
        }

        @Test
        void withInferenceCustomizerReturnsNewInstance() {
            ChatRequest original = ChatRequest.empty();
            ChatRequest derived = original.withInferenceCustomizer(p -> p.withSeed(42));
            assertThat(derived, is(not(sameInstance(original))));
        }

        @Test
        @DisplayName("chained derivations leave every intermediate untouched")
        void chainedDerivationsLeaveIntermediatesUntouched() {
            ChatRequest a = ChatRequest.empty();
            ChatRequest b = a.appendMessage("user", "hi");
            ChatRequest c = b.appendMessage("assistant", "hello");
            ChatRequest d = c.withMaxToolRounds(3);

            assertThat(a.getMessages(), is(empty()));
            assertThat(b.getMessages(), hasSize(1));
            assertThat(c.getMessages(), hasSize(2));
            assertThat(d.getMessages(), hasSize(2));
            assertThat(c.getMaxToolRounds(), is(ChatRequest.DEFAULT_MAX_TOOL_ROUNDS));
            assertThat(d.getMaxToolRounds(), is(3));
        }

        @Test
        @DisplayName("the messages accessor returns an unmodifiable view")
        void messagesAccessorIsUnmodifiable() {
            ChatRequest req = ChatRequest.empty().appendMessage("user", "hi");
            assertThrows(
                    UnsupportedOperationException.class, () -> req.getMessages().clear());
        }

        @Test
        @DisplayName("the tools accessor returns an unmodifiable view")
        void toolsAccessorIsUnmodifiable() {
            ChatRequest req = ChatRequest.empty().appendTool(new ToolDefinition("e", "d", "{}"));
            assertThrows(
                    UnsupportedOperationException.class, () -> req.getTools().clear());
        }
    }

    @Nested
    @DisplayName("equality — value semantics")
    class Equality {

        @Test
        void twoEmptyRequestsAreEqual() {
            assertThat(ChatRequest.empty(), is(ChatRequest.empty()));
        }

        @Test
        void sameContentSameEquality() {
            ChatRequest a = ChatRequest.empty().appendMessage("user", "hi").withMaxToolRounds(3);
            ChatRequest b = ChatRequest.empty().appendMessage("user", "hi").withMaxToolRounds(3);
            assertThat(a, is(b));
            assertThat(a.hashCode(), is(b.hashCode()));
        }

        @Test
        void differentMessagesNotEqual() {
            ChatRequest a = ChatRequest.empty().appendMessage("user", "hi");
            ChatRequest b = ChatRequest.empty().appendMessage("user", "bye");
            assertThat(a, is(not(b)));
        }

        @Test
        void differentMaxToolRoundsNotEqual() {
            ChatRequest a = ChatRequest.empty().withMaxToolRounds(2);
            ChatRequest b = ChatRequest.empty().withMaxToolRounds(3);
            assertThat(a, is(not(b)));
        }

        @Test
        void differentParallelToolCallsNotEqual() {
            ChatRequest a = ChatRequest.empty().withParallelToolCalls(Boolean.TRUE);
            ChatRequest b = ChatRequest.empty().withParallelToolCalls(Boolean.FALSE);
            assertThat(a, is(not(b)));
        }

        @Test
        @DisplayName(
                "the customiser is excluded from equality — two requests with the same content but different lambdas are equal")
        void customizerExcludedFromEquality() {
            ChatRequest a = ChatRequest.empty().withInferenceCustomizer(p -> p.withSeed(1));
            ChatRequest b = ChatRequest.empty().withInferenceCustomizer(p -> p.withSeed(2));
            assertThat("different lambda identities must NOT make the requests unequal", a, is(b));
        }
    }

    @Nested
    @DisplayName("validation")
    class Validation {

        @Test
        void withMaxToolRoundsRejectsZero() {
            assertThrows(
                    IllegalArgumentException.class, () -> ChatRequest.empty().withMaxToolRounds(0));
        }

        @Test
        void withMaxToolRoundsRejectsNegative() {
            assertThrows(
                    IllegalArgumentException.class, () -> ChatRequest.empty().withMaxToolRounds(-1));
        }

        @Test
        void emptyMessageIsTheCanonicalStartingPoint() {
            assertThat("empty() is a cached singleton", ChatRequest.empty(), is(sameInstance(ChatRequest.empty())));
        }
    }

    @Nested
    @DisplayName("JSON-build helpers stay read-only")
    class JsonHelpers {

        @Test
        void buildMessagesJsonDoesNotMutate() {
            ChatRequest req = ChatRequest.empty().appendMessage("user", "hi");
            String json = req.buildMessagesJson();
            assertThat(json, json, containsString("\"user\""));
            assertThat("build did not mutate the messages list", req.getMessages(), hasSize(1));
        }

        @Test
        void buildMessagesJsonPreservesMultimodalParts() {
            ChatRequest req = ChatRequest.empty()
                    .appendMessage(ChatMessage.userMultimodal(
                            ContentPart.text("describe"), ContentPart.imageUrl("data:image/png;base64,AAAA")));
            String json = req.buildMessagesJson();
            assertThat(json, containsString("\"content\":["));
            assertThat(json, containsString("\"type\":\"image_url\""));
            assertThat(json, containsString("data:image/png;base64,AAAA"));
        }

        @Test
        void buildMessagesJsonEmitsInputAudioParts() {
            ChatRequest req = ChatRequest.empty()
                    .appendMessage(ChatMessage.userMultimodal(
                            ContentPart.text("transcribe"), ContentPart.inputAudio(new byte[] {1, 2, 3}, "wav")));
            String json = req.buildMessagesJson();
            assertThat(json, containsString("\"type\":\"input_audio\""));
            assertThat(json, containsString("\"format\":\"wav\""));
            assertThat(
                    json,
                    containsString("\"data\":\"" + java.util.Base64.getEncoder().encodeToString(new byte[] {1, 2, 3})));
        }

        @Test
        void buildToolsJsonEmptyWhenNoTools() {
            assertThat(ChatRequest.empty().buildToolsJson().isPresent(), is(false));
        }

        @Test
        void buildToolsJsonEmbedsAValidSchemaAsAnObject() {
            String json = ChatRequest.empty()
                    .appendTool(new ToolDefinition("f", "d", "{\"type\":\"object\"}"))
                    .buildToolsJson()
                    .orElseThrow(IllegalStateException::new);
            assertThat(
                    json,
                    is("[{\"type\":\"function\",\"function\":{\"name\":\"f\",\"description\":\"d\","
                            + "\"parameters\":{\"type\":\"object\"}}}]"));
        }

        @ParameterizedTest
        @ValueSource(strings = {"", "  ", "not json{{{", "{\"type\":\"object\"} {\"required\":[\"x\"]}"})
        void appendToolRejectsASchemaThatIsNotExactlyOneJsonValue(String schema) {
            // An empty schema used to be sent as the JSON string "", an unparseable one or one with trailing
            // content as the raw text: llama.cpp then refused the request with "schema must be an object",
            // naming neither the tool nor the cause, and a template rendered without a grammar took it as is.
            ToolDefinition tool = new ToolDefinition("weather", "d", schema);
            IllegalArgumentException e = assertThrows(
                    IllegalArgumentException.class, () -> ChatRequest.empty().appendTool(tool));
            assertThat(
                    e.getMessage(),
                    is("tool 'weather': parameters must be exactly one well-formed JSON value, but was: " + schema));
        }

        @Test
        void appendToolBoundsTheRejectedSchemaInTheMessage() {
            String schema = "x".repeat(300);
            IllegalArgumentException e = assertThrows(
                    IllegalArgumentException.class,
                    () -> ChatRequest.empty().appendTool(new ToolDefinition("f", "d", schema)));
            assertThat(e.getMessage(), not(containsString(schema)));
            assertThat(e.getMessage(), containsString("x".repeat(80) + "..."));
        }
    }
}
