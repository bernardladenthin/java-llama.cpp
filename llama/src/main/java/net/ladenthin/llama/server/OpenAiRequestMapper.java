// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import com.fasterxml.jackson.databind.JsonNode;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.BiFunction;
import net.ladenthin.llama.parameters.InferenceParameters;

/**
 * Pure mapping from an OpenAI {@code /v1/chat/completions} request body to {@link InferenceParameters}.
 *
 * <p>The structural fields — {@code messages}, {@code tools}, {@code tool_choice}, and
 * {@code parallel_tool_calls} — are forwarded
 * <em>verbatim</em> as raw JSON so the full OpenAI shape (assistant {@code tool_calls},
 * {@code role:"tool"} results with {@code tool_call_id}, and vision {@code image_url} content parts)
 * round-trips untouched into the native chat-template parser. Sampling fields are translated to the
 * matching {@code InferenceParameters.with*} setter, as are the llama.cpp sampling extensions upstream's
 * request schema reads ({@code min_p}, {@code repeat_penalty}, {@code grammar}, ...) and the reasoning
 * controls upstream's chat
 * parser reads ({@code chat_template_kwargs}, {@code reasoning_effort}, the reasoning budget); unknown
 * fields are ignored.
 *
 * <p>The {@code stream} flag is intentionally not mapped here — streaming is selected by the caller
 * ({@link net.ladenthin.llama.LlamaModel#chatComplete} forces it off,
 * {@link net.ladenthin.llama.LlamaModel#streamChatCompletion} forces it on). Stateless and free of JNI
 * and model dependencies, so it is unit-testable with JSON literals alone.
 */
final class OpenAiRequestMapper {

    /** Upstream's {@code n_probs} for a chat request with {@code logprobs} but no {@code top_logprobs}. */
    private static final int DEFAULT_TOP_LOGPROBS = 20;

    OpenAiRequestMapper() {}

    /**
     * Translate an OpenAI chat request into {@link InferenceParameters}.
     *
     * @param request the parsed OpenAI request object
     * @return inference parameters carrying the verbatim messages and mapped sampling options
     * @throws IllegalArgumentException if {@code messages} is missing or not a non-empty array
     */
    InferenceParameters toInferenceParameters(JsonNode request) {
        JsonNode messages = request.path("messages");
        if (!messages.isArray() || messages.size() == 0) {
            throw new IllegalArgumentException("'messages' must be a non-empty array");
        }

        // cache_prompt=true reuses the slot's KV prefix across turns — the standard llama.cpp-server
        // default and what IDE clients rely on for acceptable repeated-prefix latency. OpenAI requests
        // never carry this llama.cpp-specific flag, so defaulting it here is safe.
        InferenceParameters params = InferenceParameters.empty()
                .withMessagesJson(messages.toString())
                .withCachePrompt(true);

        params = applyCommonFields(params, request);
        params = applyReasoningControls(params, request);

        // Chat logprobs, as upstream's oaicompat_chat_params_parse maps them (the legacy completions
        // "logprobs" is an integer and means something else, so this stays on the chat path).
        if (request.path("logprobs").asBoolean(false)) {
            params = params.withNProbs(request.path("top_logprobs").asInt(DEFAULT_TOP_LOGPROBS));
        }

        // Tools are chat-only.
        JsonNode tools = request.path("tools");
        if (tools.isArray() && tools.size() > 0) {
            params = params.withToolsJson(tools.toString());
            JsonNode toolChoice = request.path("tool_choice");
            if (toolChoice.isTextual()) {
                params = params.withToolChoice(toolChoice.asText());
            } else if ("function".equals(toolChoice.path("type").asText(""))) {
                // A named function: llama.cpp's chat parser reads tool_choice only as a string and would
                // fall back to "auto"; "required" is the closest it allows.
                params = params.withToolChoice("required");
            }
            JsonNode parallelToolCalls = request.path("parallel_tool_calls");
            if (parallelToolCalls.isBoolean()) {
                params = params.withParallelToolCalls(parallelToolCalls.asBoolean());
            }
        }

        return params;
    }

    /**
     * Apply the sampling / KV-cache / output-shaping fields of a chat request (temperature, top_p/top_k, seed, penalties, max tokens, stop, stream_options, response_format,
     * plus the llama.cpp cache extensions). Tools and messages are handled by the caller.
     *
     * @param params the parameters to extend
     * @param request the parsed OpenAI request object
     * @return a new instance with the recognised fields applied
     */
    private InferenceParameters applyCommonFields(InferenceParameters params, JsonNode request) {
        // Upstream reads "n" as n_cmpl (several answers); these paths return one, so more is refused
        // rather than silently answered once.
        JsonNode choices = request.path("n");
        if (choices.isNumber() && choices.asInt() > 1) {
            throw new IllegalArgumentException(
                    "'n' greater than 1 is not supported (one answer is returned) but was " + choices.asInt());
        }
        // logit_bias as upstream's schema reads it: [token, bias] pairs or OpenAI's object form.
        JsonNode logitBias = request.path("logit_bias");
        if (logitBias.isObject() || logitBias.isArray()) {
            params = params.withLogitBiasJson(logitBias.toString());
        }
        JsonNode nProbs = request.path("n_probs");
        if (nProbs.isInt()) {
            params = params.withNProbs(nProbs.asInt());
        }
        JsonNode verbose = request.path("verbose");
        if (verbose.isBoolean()) {
            params = params.withVerbose(verbose.asBoolean());
        }
        JsonNode responseFields = request.path("response_fields");
        if (!responseFields.isMissingNode() && !responseFields.isNull()) {
            params = params.withResponseFields(stringArray(responseFields, "response_fields"));
        }
        // Preserve llama.cpp extensions when advanced clients opt into them.
        JsonNode cachePrompt = request.path("cache_prompt");
        if (cachePrompt.isBoolean()) {
            params = params.withCachePrompt(cachePrompt.asBoolean());
        }
        JsonNode cacheReuse = request.path("n_cache_reuse");
        if (cacheReuse.isIntegralNumber()) {
            params = params.withCacheReuse(cacheReuse.asInt());
        }
        // upstream's default -1 means "any slot"; leaving the field unset is the same request.
        JsonNode slotId = request.path("id_slot");
        if (slotId.isIntegralNumber() && slotId.asInt() >= 0) {
            params = params.withSlotId(slotId.asInt());
        }

        JsonNode temperature = request.path("temperature");
        if (temperature.isNumber()) {
            params = params.withTemperature((float) temperature.asDouble());
        }
        JsonNode topP = request.path("top_p");
        if (topP.isNumber()) {
            params = params.withTopP((float) topP.asDouble());
        }
        JsonNode topK = request.path("top_k");
        if (topK.isNumber()) {
            params = params.withTopK(topK.asInt());
        }
        JsonNode seed = request.path("seed");
        if (seed.isNumber()) {
            params = params.withSeed(seed.asInt());
        }
        JsonNode presencePenalty = request.path("presence_penalty");
        if (presencePenalty.isNumber()) {
            params = params.withPresencePenalty((float) presencePenalty.asDouble());
        }
        JsonNode frequencyPenalty = request.path("frequency_penalty");
        if (frequencyPenalty.isNumber()) {
            params = params.withFrequencyPenalty((float) frequencyPenalty.asDouble());
        }

        // 0 is a valid cap upstream (evaluate the prompt into the cache, generate nothing); -1 is its default.
        int maxTokens = readMaxTokens(request);
        if (maxTokens >= 0) {
            params = params.withNPredict(maxTokens);
        }

        String[] stops = readStops(request);
        if (stops.length > 0) {
            params = params.withStopStrings(stops);
        }

        params = applyLlamaCppSampling(params, request);

        // Forward stream_options verbatim (e.g. {"include_usage":true}) so the native server emits the
        // trailing usage chunk the OpenAI streaming protocol — and the Copilot custom endpoint — expect.
        JsonNode streamOptions = request.path("stream_options");
        if (streamOptions.isObject()) {
            params = params.withStreamOptions(streamOptions.toString());
        }

        // Forward response_format verbatim (json_object / json_schema) so the native server applies the
        // matching grammar constraint — the OpenAI "structured outputs" feature used by strict clients.
        JsonNode responseFormat = request.path("response_format");
        if (responseFormat.isObject()) {
            params = params.withResponseFormat(responseFormat.toString());
        }

        return params;
    }

    /**
     * Apply the llama.cpp sampling extensions upstream's request schema ({@code server-schema.cpp},
     * b11639) reads next to the OpenAI fields. The non-streamed {@code /v1/completions} forwards the body
     * to that schema, so without them the streamed form of the same request would sample differently.
     * {@code json_schema} applies only without a {@code grammar}, as upstream decides.
     *
     * @param params the parameters to extend
     * @param request the parsed request object
     * @return a new instance with the recognised extensions applied
     */
    private InferenceParameters applyLlamaCppSampling(InferenceParameters params, JsonNode request) {
        params = withFloat(params, request, "min_p", InferenceParameters::withMinP);
        params = withFloat(params, request, "typical_p", InferenceParameters::withTypicalP);
        params = withFloat(params, request, "repeat_penalty", InferenceParameters::withRepeatPenalty);
        params = withFloat(params, request, "top_n_sigma", InferenceParameters::withTopNSigma);
        params = withFloat(params, request, "xtc_probability", InferenceParameters::withXtcProbability);
        params = withFloat(params, request, "xtc_threshold", InferenceParameters::withXtcThreshold);
        params = withFloat(params, request, "dynatemp_range", InferenceParameters::withDynamicTemperatureRange);
        params = withFloat(params, request, "dynatemp_exponent", InferenceParameters::withDynamicTemperatureExponent);
        params = withInt(params, request, "repeat_last_n", InferenceParameters::withRepeatLastN);
        params = withInt(params, request, "min_keep", InferenceParameters::withMinKeep);
        params = withInt(params, request, "n_keep", InferenceParameters::withNKeep);
        JsonNode ignoreEos = request.path("ignore_eos");
        if (ignoreEos.isBoolean()) {
            params = params.withIgnoreEos(ignoreEos.asBoolean());
        }
        JsonNode grammar = request.path("grammar");
        if (grammar.isTextual()) {
            return params.withGrammar(grammar.asText());
        }
        JsonNode jsonSchema = request.path("json_schema");
        return jsonSchema.isObject() ? params.withJsonSchema(jsonSchema.toString()) : params;
    }

    /**
     * Apply {@code setter} with the request's number {@code field}, if it is one.
     *
     * @param params the parameters to extend
     * @param request the parsed request object
     * @param field the request field
     * @param setter the matching setter
     * @return the extended parameters, or {@code params} when the field is absent or not a number
     */
    private static InferenceParameters withFloat(
            InferenceParameters params,
            JsonNode request,
            String field,
            BiFunction<InferenceParameters, Float, InferenceParameters> setter) {
        JsonNode value = request.path(field);
        return value.isNumber() ? setter.apply(params, (float) value.asDouble()) : params;
    }

    /**
     * Read a field that must be an array of strings, as upstream's schema reads it; anything else is
     * an invalid request rather than a field silently left out.
     *
     * @param value the field's value
     * @param field the field name, for the message
     * @return the strings
     */
    private static String[] stringArray(JsonNode value, String field) {
        if (!value.isArray()) {
            throw new IllegalArgumentException("'" + field + "' must be an array of strings");
        }
        List<String> strings = new ArrayList<>(value.size());
        for (JsonNode entry : value) {
            if (!entry.isTextual()) {
                throw new IllegalArgumentException("'" + field + "' must be an array of strings");
            }
            strings.add(entry.asText());
        }
        return strings.toArray(new String[0]);
    }

    /**
     * Apply {@code setter} with the request's integer {@code field}, if it is one.
     *
     * @param params the parameters to extend
     * @param request the parsed request object
     * @param field the request field
     * @param setter the matching setter
     * @return the extended parameters, or {@code params} when the field is absent or not an integer
     */
    private static InferenceParameters withInt(
            InferenceParameters params,
            JsonNode request,
            String field,
            BiFunction<InferenceParameters, Integer, InferenceParameters> setter) {
        JsonNode value = request.path(field);
        return value.isIntegralNumber() ? setter.apply(params, value.asInt()) : params;
    }

    /**
     * Read the output-token cap: upstream's {@code n_predict}, else the newer {@code
     * max_completion_tokens}, else the legacy {@code max_tokens} (upstream reads the latter two as
     * aliases of the first).
     *
     * @param request the parsed OpenAI request object
     * @return the requested cap, or {@code -1} when none of the fields is a number
     */
    private int readMaxTokens(JsonNode request) {
        JsonNode nPredict = request.path("n_predict");
        if (nPredict.isNumber()) {
            return nPredict.asInt();
        }
        JsonNode maxCompletion = request.path("max_completion_tokens");
        if (maxCompletion.isNumber()) {
            return maxCompletion.asInt();
        }
        JsonNode maxTokens = request.path("max_tokens");
        if (maxTokens.isNumber()) {
            return maxTokens.asInt();
        }
        return -1;
    }

    /**
     * Read the {@code stop} field, which OpenAI permits as either a single string or an array of
     * strings.
     *
     * @param request the parsed OpenAI request object
     * @return the stop strings (possibly empty, never {@code null})
     */
    private String[] readStops(JsonNode request) {
        JsonNode stop = request.path("stop");
        List<String> stops = new ArrayList<>();
        if (stop.isTextual()) {
            stops.add(stop.asText());
        } else if (stop.isArray()) {
            for (JsonNode entry : stop) {
                if (entry.isTextual()) {
                    stops.add(entry.asText());
                }
            }
        }
        return stops.toArray(new String[0]);
    }

    /**
     * Apply the reasoning controls that upstream's chat parser ({@code oaicompat_chat_params_parse},
     * b11639) reads from a chat request: the request's {@code chat_template_kwargs}, {@code
     * reasoning_effort} ({@code "none"} switches thinking off, any other value becomes the template
     * variable {@code reasoning_effort}) and the reasoning budget ({@code reasoning_budget_tokens},
     * else {@code thinking_budget_tokens}). The Anthropic and Responses translations produce the
     * latter two.
     *
     * @param params the parameters built so far
     * @param request the parsed chat request
     * @return a new instance with the reasoning controls applied
     */
    private InferenceParameters applyReasoningControls(InferenceParameters params, JsonNode request) {
        Map<String, String> kwargs = new LinkedHashMap<>();
        JsonNode requestKwargs = request.path("chat_template_kwargs");
        if (requestKwargs.isObject()) {
            for (Map.Entry<String, JsonNode> entry : requestKwargs.properties()) {
                kwargs.put(entry.getKey(), entry.getValue().toString());
            }
        }
        JsonNode effort = request.path("reasoning_effort");
        if (effort.isTextual()) {
            if ("none".equals(effort.asText())) {
                kwargs.remove("reasoning_effort");
                kwargs.put("enable_thinking", "false");
            } else if (!effort.asText().isEmpty()) {
                kwargs.put("reasoning_effort", effort.toString());
            }
        }
        if (!kwargs.isEmpty()) {
            params = params.withChatTemplateKwargs(kwargs);
        }
        JsonNode budget = request.path("reasoning_budget_tokens");
        if (!budget.isInt()) {
            budget = request.path("thinking_budget_tokens");
        }
        if (budget.isInt()) {
            params = params.withReasoningBudgetTokens(budget.asInt());
        }
        return params;
    }
}
