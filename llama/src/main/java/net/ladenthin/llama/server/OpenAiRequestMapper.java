// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.JsonNodeFactory;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.BiFunction;
import net.ladenthin.llama.parameters.InferenceParameters;
import org.jspecify.annotations.Nullable;

/**
 * Pure mapping from an OpenAI {@code /v1/chat/completions} request body to {@link InferenceParameters}.
 *
 * <p>The structural fields — {@code messages}, {@code tools}, the string form of {@code tool_choice}, and
 * {@code parallel_tool_calls} — are forwarded
 * <em>verbatim</em> as raw JSON so the full OpenAI shape (assistant {@code tool_calls},
 * {@code role:"tool"} results with {@code tool_call_id}, and vision {@code image_url} content parts)
 * round-trips untouched into the native chat-template parser. A {@code tool_choice} naming one function
 * becomes {@code required} with only that tool forwarded (see {@link #namedToolChoice}). Sampling fields are translated to the
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

        params = applyLogprobs(params, request);

        // Tools are chat-only.
        JsonNode tools = request.path("tools");
        if (tools.isArray() && tools.size() > 0) {
            JsonNode toolChoice = request.path("tool_choice");
            String namedTool =
                    namedToolChoice(toolChoice, toolChoice.path("function").path("name"));
            if (namedTool == null) {
                params = params.withToolsJson(tools.toString());
            } else {
                ArrayNode named = JsonNodeFactory.instance.arrayNode();
                for (JsonNode tool : tools) {
                    if (namedTool.equals(tool.path("function").path("name").asText(""))) {
                        named.add(tool);
                    }
                }
                if (named.size() == 0) {
                    throw new IllegalArgumentException("tool_choice names unknown tool '" + namedTool + "'");
                }
                params = params.withToolsJson(named.toString());
            }
            if (toolChoice.isTextual()) {
                params = params.withToolChoice(toolChoice.asText());
            } else if (namedTool != null) {
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
     * The function a {@code tool_choice} object of type {@code function} names, or {@code null} for any
     * other {@code tool_choice}. llama.cpp's chat parser reads {@code tool_choice} only as {@code auto},
     * {@code none} or {@code required}; a named function is enforced by forwarding that tool alone with
     * {@code required}, so the grammar admits no other call. The chat template then does not render the
     * other tool definitions either, which the model may not call anyway (OpenAI keeps them visible and
     * constrains only the output). Shared by the chat mapping ({@code tool_choice.function.name}) and
     * the Responses translation ({@code tool_choice.name}).
     *
     * @param toolChoice the request's {@code tool_choice}
     * @param name the node holding the function name in that request shape
     * @return the named function, or {@code null}
     * @throws IllegalArgumentException for a {@code tool_choice} of type {@code function} without a name
     */
    static @Nullable String namedToolChoice(JsonNode toolChoice, JsonNode name) {
        if (!"function".equals(toolChoice.path("type").asText(""))) {
            return null;
        }
        if (!name.isTextual() || name.asText().isEmpty()) {
            throw new IllegalArgumentException(
                    "tool_choice of type 'function' needs the function's 'name' (got " + toolChoice + ")");
        }
        return name.asText();
    }

    /**
     * Apply {@code logprobs} / {@code top_logprobs} as upstream's chat parser and request schema read
     * them ({@code oaicompat_chat_params_parse} and {@code server-schema.cpp}, b11639): {@code logprobs:
     * true} is the chat form and becomes {@code n_probs = top_logprobs} (20 without it); a
     * {@code top_logprobs} without it is refused with upstream's message; a number is the schema's alias
     * of {@code n_probs} (the legacy completions form), read only without an {@code n_probs}, and any
     * other non-null value read there is refused, as upstream's schema refuses it.
     *
     * @param params the parameters to extend
     * @param request the parsed OpenAI request object
     * @return a new instance with the probabilities applied
     * @throws IllegalArgumentException for a request upstream refuses
     */
    private static InferenceParameters applyLogprobs(InferenceParameters params, JsonNode request) {
        JsonNode logprobs = request.path("logprobs");
        JsonNode topLogprobs = request.path("top_logprobs");
        if (logprobs.isBoolean() && logprobs.asBoolean()) {
            return params.withNProbs(topLogprobs.isNumber() ? topLogprobs.asInt() : DEFAULT_TOP_LOGPROBS);
        }
        if (!topLogprobs.isMissingNode() && !topLogprobs.isNull()) {
            throw new IllegalArgumentException("top_logprobs requires logprobs to be set to true");
        }
        JsonNode nProbs = request.path("n_probs");
        if (logprobs.isMissingNode()
                || logprobs.isNull()
                || logprobs.isBoolean()
                || (!nProbs.isMissingNode() && !nProbs.isNull())) {
            // the schema reads n_probs first, its alias only without it
            return params;
        }
        if (!logprobs.isNumber()) {
            throw new IllegalArgumentException("'logprobs' must be a boolean or a number");
        }
        return params.withNProbs(logprobs.asInt());
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
        // Upstream reads "n" as n_cmpl (several answers, hard limits [1, n_parallel]); these paths return
        // one and "n" never reaches upstream from here, so every other value is refused rather than
        // silently answered once.
        JsonNode choices = request.path("n");
        if (choices.isNumber() && choices.asInt() != 1) {
            throw new IllegalArgumentException("'n' must be 1 (one answer is returned) but was " + choices.asInt());
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

        // -1 is upstream's default (the server's --predict), the same as no field, so it stays unset. Every
        // other value is forwarded: upstream's hard limits [-1, INT32_MAX] refuse -2 and below with 400, as
        // on the verbatim /v1/completions. A 0 is forwarded as upstream reads it, and the server then
        // generates one token (see withNPredict): its budget check runs only after the first token.
        int maxTokens = readMaxTokens(request);
        if (maxTokens != -1) {
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
     * b11639) reads next to the OpenAI fields. {@code /v1/completions} forwards its body to that schema
     * verbatim, streamed or not; the chat paths are built here, so without these fields a chat request
     * would sample differently from a completion carrying the same fields. {@code json_schema} and
     * {@code grammar} are both forwarded: upstream's chat parser refuses a request carrying both
     * ({@code Cannot use both json_schema and grammar}, 400), the answer llama-server gives, so the mapper
     * does not decide for it.
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
            params = params.withGrammar(grammar.asText());
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
