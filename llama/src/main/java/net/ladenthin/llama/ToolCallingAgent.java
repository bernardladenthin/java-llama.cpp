// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import com.fasterxml.jackson.databind.ObjectMapper;
import java.util.Map;
import java.util.Optional;
import java.util.function.Function;
import net.ladenthin.llama.callback.ToolHandler;
import net.ladenthin.llama.parameters.ChatRequest;
import net.ladenthin.llama.value.ChatMessage;
import net.ladenthin.llama.value.ChatResponse;
import net.ladenthin.llama.value.ToolCall;

/** Model-independent orchestration for the tool-calling agent loop. */
final class ToolCallingAgent {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    private ToolCallingAgent() {}

    static ChatResponse run(
            ChatRequest request, Map<String, ToolHandler> handlers, Function<ChatRequest, ChatResponse> chatCall) {
        final int maxRounds = request.getMaxToolRounds();
        if (maxRounds < 1) {
            throw new IllegalArgumentException("ChatRequest.maxToolRounds must be >= 1 (got " + maxRounds + "); "
                    + "chatWithTools always issues at least one chat call.");
        }

        ChatRequest current = request;
        ChatResponse last = chatCall.apply(current);
        for (int round = 1; round < maxRounds; round++) {
            Optional<ChatMessage> assistantOpt = last.getFirstMessage();
            if (!assistantOpt.isPresent() || assistantOpt.get().getToolCalls().isEmpty()) {
                return last;
            }

            ChatMessage assistant = assistantOpt.get();
            current = afterTheFirstRound(current).appendMessage(assistant);
            for (ToolCall call : assistant.getToolCalls()) {
                current = current.appendMessage(ChatMessage.toolResult(call.getId(), invoke(call, handlers)));
            }
            last = chatCall.apply(current);
        }
        return last;
    }

    /**
     * The request of every round after the first: a choice that forces a call -- {@code required} or a
     * named function -- becomes {@code auto}, since forced in every round the model could never answer
     * the tool results, and the loop ran to its cap with a call as its last response.
     */
    static ChatRequest afterTheFirstRound(ChatRequest request) {
        boolean forced = request.getToolChoiceFunction().isPresent()
                || "required".equals(request.getToolChoice().orElse(null));
        return forced ? request.withToolChoice("auto") : request;
    }

    private static String invoke(ToolCall call, Map<String, ToolHandler> handlers) {
        ToolHandler handler = handlers.get(call.getName());
        if (handler == null) {
            return errorJson("unknown tool: " + call.getName());
        }
        try {
            return handler.invoke(call.getArgumentsJson());
        } catch (Exception e) {
            return errorJson(e.getClass().getSimpleName() + ": " + e.getMessage());
        }
    }

    private static String errorJson(String message) {
        return MAPPER.createObjectNode().put("error", message).toString();
    }
}
