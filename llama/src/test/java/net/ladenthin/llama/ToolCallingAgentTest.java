// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.hasSize;
import static org.hamcrest.Matchers.is;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import net.ladenthin.llama.callback.ToolHandler;
import net.ladenthin.llama.parameters.ChatRequest;
import net.ladenthin.llama.value.ChatChoice;
import net.ladenthin.llama.value.ChatMessage;
import net.ladenthin.llama.value.ChatResponse;
import net.ladenthin.llama.value.Timings;
import net.ladenthin.llama.value.ToolCall;
import net.ladenthin.llama.value.Usage;
import org.junit.jupiter.api.Test;

class ToolCallingAgentTest {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    @Test
    void invokesHandlerAndAppendsAssistantAndToolResultTurns() {
        ToolCall call = new ToolCall("call-1", "weather", "{\"city\":\"Paris\"}");
        List<ChatRequest> requests = new ArrayList<ChatRequest>();
        List<ChatResponse> responses =
                new ArrayList<ChatResponse>(Arrays.asList(toolResponse(call), textResponse("It is sunny.")));
        AtomicInteger invocations = new AtomicInteger();
        Map<String, ToolHandler> handlers = Collections.<String, ToolHandler>singletonMap("weather", args -> {
            invocations.incrementAndGet();
            assertThat(args, is("{\"city\":\"Paris\"}"));
            return "{\"condition\":\"sunny\"}";
        });

        ChatResponse result = ToolCallingAgent.run(
                ChatRequest.empty().appendMessage("user", "Weather?").withMaxToolRounds(2), handlers, request -> {
                    requests.add(request);
                    return responses.remove(0);
                });

        assertThat(result.getFirstContent(), is("It is sunny."));
        assertThat(invocations.get(), is(1));
        assertThat(requests, hasSize(2));
        List<ChatMessage> secondRound = requests.get(1).getMessages();
        assertThat(secondRound, hasSize(3));
        assertThat(secondRound.get(1).getToolCalls().get(0), is(call));
        assertThat(secondRound.get(2).getRole(), is("tool"));
        assertThat(secondRound.get(2).getToolCallId().orElseThrow(), is("call-1"));
        assertThat(secondRound.get(2).getContent(), is("{\"condition\":\"sunny\"}"));
    }

    @Test
    void aChoiceThatForcesACallHoldsForTheFirstRoundOnly() {
        // forced in every round, the model could never answer the tool result: the loop ran to its cap
        ToolCall call = new ToolCall("call-1", "weather", "{}");
        Map<String, ToolHandler> handlers = Collections.<String, ToolHandler>singletonMap("weather", args -> "{}");
        ChatRequest[] forcing = {
            ChatRequest.empty().appendMessage("user", "Weather?").withToolChoice("required"),
            ChatRequest.empty().appendMessage("user", "Weather?").withToolChoiceFunction("weather"),
        };
        for (ChatRequest first : forcing) {
            List<ChatRequest> requests = new ArrayList<ChatRequest>();
            List<ChatResponse> responses =
                    new ArrayList<ChatResponse>(Arrays.asList(toolResponse(call), textResponse("It is sunny.")));
            ToolCallingAgent.run(first.withMaxToolRounds(3), handlers, request -> {
                requests.add(request);
                return responses.remove(0);
            });
            assertThat(requests, hasSize(2));
            assertThat(requests.get(0).getToolChoice(), is(first.getToolChoice()));
            assertThat(requests.get(0).getToolChoiceFunction(), is(first.getToolChoiceFunction()));
            assertThat(requests.get(1).getToolChoice().orElseThrow(), is("auto"));
            assertThat(requests.get(1).getToolChoiceFunction().isPresent(), is(false));
        }
        // a choice that forces nothing stays
        for (String hint : new String[] {"auto", "none"}) {
            ChatRequest request = ChatRequest.empty().withToolChoice(hint);
            assertThat(ToolCallingAgent.afterTheFirstRound(request), is(request));
        }
        assertThat(ToolCallingAgent.afterTheFirstRound(ChatRequest.empty()), is(ChatRequest.empty()));
    }

    @Test
    void invokesEveryToolCallInOneAssistantTurn() {
        ToolCall first = new ToolCall("call-1", "first", "{}");
        ToolCall second = new ToolCall("call-2", "second", "{}");
        Map<String, ToolHandler> handlers = new HashMap<String, ToolHandler>();
        handlers.put("first", args -> "1");
        handlers.put("second", args -> "2");
        List<ChatRequest> requests = new ArrayList<ChatRequest>();
        List<ChatResponse> responses = new ArrayList<ChatResponse>(
                Arrays.asList(toolResponse(Arrays.asList(first, second)), textResponse("done")));

        ToolCallingAgent.run(
                ChatRequest.empty().appendMessage("user", "both").withMaxToolRounds(2), handlers, request -> {
                    requests.add(request);
                    return responses.remove(0);
                });

        List<ChatMessage> secondRound = requests.get(1).getMessages();
        assertThat(secondRound, hasSize(4));
        assertThat(secondRound.get(2).getContent(), is("1"));
        assertThat(secondRound.get(3).getContent(), is("2"));
    }

    @Test
    void unknownToolNameIsReturnedAsValidJson() throws IOException {
        String result = captureToolResult(
                new ToolCall("call-1", "bad\"name", "{}"), Collections.<String, ToolHandler>emptyMap());
        JsonNode parsed = MAPPER.readTree(result);
        assertThat(parsed.path("error").asText(), is("unknown tool: bad\"name"));
    }

    @Test
    void handlerExceptionIsReturnedAsValidJson() throws IOException {
        Map<String, ToolHandler> handlers = Collections.<String, ToolHandler>singletonMap("broken", args -> {
            throw new IllegalStateException("bad \"value\"");
        });
        String result = captureToolResult(new ToolCall("call-1", "broken", "{}"), handlers);
        JsonNode parsed = MAPPER.readTree(result);
        assertThat(parsed.path("error").asText(), is("IllegalStateException: bad \"value\""));
    }

    @Test
    void roundCapStopsBeforeExecutingLastResponseCalls() {
        AtomicInteger chatCalls = new AtomicInteger();
        AtomicInteger toolCalls = new AtomicInteger();
        ToolCall call = new ToolCall("call-1", "echo", "{}");

        ChatResponse result = ToolCallingAgent.run(
                ChatRequest.empty().appendMessage("user", "echo").withMaxToolRounds(1),
                Collections.<String, ToolHandler>singletonMap("echo", args -> {
                    toolCalls.incrementAndGet();
                    return args;
                }),
                request -> {
                    chatCalls.incrementAndGet();
                    return toolResponse(call);
                });

        assertThat(result.getFirstMessage().orElseThrow().getToolCalls(), hasSize(1));
        assertThat(chatCalls.get(), is(1));
        assertThat(toolCalls.get(), is(0));
    }

    @Test
    void responseWithoutToolCallsStopsImmediately() {
        AtomicInteger chatCalls = new AtomicInteger();
        ChatResponse result = ToolCallingAgent.run(
                ChatRequest.empty().appendMessage("user", "hi"),
                Collections.<String, ToolHandler>emptyMap(),
                request -> {
                    chatCalls.incrementAndGet();
                    return textResponse("hello");
                });
        assertThat(result.getFirstContent(), is("hello"));
        assertThat(chatCalls.get(), is(1));
    }

    private String captureToolResult(ToolCall call, Map<String, ToolHandler> handlers) {
        List<ChatRequest> requests = new ArrayList<ChatRequest>();
        List<ChatResponse> responses =
                new ArrayList<ChatResponse>(Arrays.asList(toolResponse(call), textResponse("done")));
        ToolCallingAgent.run(
                ChatRequest.empty().appendMessage("user", "go").withMaxToolRounds(2), handlers, request -> {
                    requests.add(request);
                    return responses.remove(0);
                });
        return requests.get(1).getMessages().get(2).getContent();
    }

    private static ChatResponse toolResponse(ToolCall call) {
        return toolResponse(Collections.singletonList(call));
    }

    private static ChatResponse toolResponse(List<ToolCall> calls) {
        return response(ChatMessage.assistantToolCalls("", calls), "tool_calls");
    }

    private static ChatResponse textResponse(String text) {
        return response(new ChatMessage("assistant", text), "stop");
    }

    private static ChatResponse response(ChatMessage message, String finishReason) {
        return new ChatResponse(
                "id",
                Collections.singletonList(new ChatChoice(0, message, finishReason)),
                new Usage(0, 0),
                new Timings(0, 0, 0.0, 0.0, 0, 0.0, 0.0, 0, 0),
                "{}");
    }
}
