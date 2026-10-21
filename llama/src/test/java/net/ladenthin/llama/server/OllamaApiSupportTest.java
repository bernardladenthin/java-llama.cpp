// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.is;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.IOException;
import org.junit.jupiter.api.Test;

/**
 * Unit tests for {@link OllamaApiSupport}: discovery bodies and the Ollama&harr;OpenAI chat request,
 * response and NDJSON streaming translation. Pure — no model.
 */
public class OllamaApiSupportTest {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    private static JsonNode read(String json) throws IOException {
        return MAPPER.readTree(json);
    }

    @Test
    public void versionJsonCarriesAVersion() throws IOException {
        assertThat(read(OllamaApiSupport.versionJson()).path("version").asText(), is(OllamaApiSupport.OLLAMA_VERSION));
    }

    @Test
    public void tagsJsonListsTheModel() throws IOException {
        JsonNode out = read(OllamaApiSupport.tagsJson("local-qwen"));
        assertThat(out.path("models").get(0).path("name").asText(), is("local-qwen"));
        assertThat(out.path("models").get(0).path("model").asText(), is("local-qwen"));
    }

    @Test
    public void showJsonAdvertisesCapabilitiesAndContextLength() throws IOException {
        JsonNode out = read(OllamaApiSupport.showJson("local-qwen", 8192, false));
        assertThat(out.path("model_info").path("llama.context_length").asInt(), is(8192));
        String capabilities = out.path("capabilities").toString();
        assertThat(capabilities.contains("completion"), is(true));
        assertThat(capabilities.contains("tools"), is(true));
        assertThat(capabilities.contains("vision"), is(false));
    }

    @Test
    public void showJsonAddsVisionCapabilityWhenEnabled() throws IOException {
        JsonNode out = read(OllamaApiSupport.showJson("m", 4096, true));
        assertThat(out.path("capabilities").toString().contains("vision"), is(true));
    }

    @Test
    public void isStreamingDefaultsTrueAndHonoursExplicitFalse() throws IOException {
        assertThat(OllamaApiSupport.isStreaming(read("{}")), is(true));
        assertThat(OllamaApiSupport.isStreaming(read("{\"stream\":false}")), is(false));
        assertThat(OllamaApiSupport.isStreaming(read("{\"stream\":true}")), is(true));
    }

    @Test
    public void toOpenAiChatRequestMapsMessagesToolsAndOptions() throws IOException {
        JsonNode openAi = OllamaApiSupport.toOpenAiChatRequest(read("{\"model\":\"m\","
                + "\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],"
                + "\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"f\"}}],"
                + "\"options\":{\"temperature\":0.4,\"num_predict\":64}}"));
        assertThat(openAi.path("messages").get(0).path("content").asText(), is("hi"));
        assertThat(openAi.path("tools").get(0).path("function").path("name").asText(), is("f"));
        assertThat(openAi.path("temperature").asDouble(), is(0.4));
        // num_predict maps to OpenAI max_tokens.
        assertThat(openAi.path("max_tokens").asInt(), is(64));
    }

    /**
     * Ollama's assistant tool calls carry no {@code type}; llama.cpp's OpenAI message parser rejects a
     * tool call without one ({@code "Missing tool call type"} in {@code common/chat.cpp}), so a chat
     * that replays an earlier tool call in its history failed with a 400 before reaching the model.
     */
    @Test
    public void toOpenAiChatRequestHonoursThinkLikeOllama() throws IOException {
        // Ollama clients switch a model's thinking off with "think": false and pick a level with a
        // string (gpt-oss); the chat mapper turns reasoning_effort into the template switch upstream reads.
        OpenAiRequestMapper mapper = new OpenAiRequestMapper();
        JsonNode off = read(mapper.toInferenceParameters(OllamaApiSupport.toOpenAiChatRequest(
                        read("{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"think\":false}")))
                .toJson());
        assertThat(off.path("chat_template_kwargs").path("enable_thinking").asBoolean(true), is(false));
        JsonNode low = read(mapper.toInferenceParameters(OllamaApiSupport.toOpenAiChatRequest(
                        read("{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"think\":\"low\"}")))
                .toJson());
        assertThat(low.path("chat_template_kwargs").path("reasoning_effort").asText(), is("low"));
    }

    @Test
    public void toOpenAiChatRequestAddsTheFunctionTypeToAssistantToolCalls() throws IOException {
        JsonNode openAi = OllamaApiSupport.toOpenAiChatRequest(
                read(
                        "{\"messages\":["
                                + "{\"role\":\"assistant\",\"tool_calls\":[{\"function\":{\"name\":\"f\",\"arguments\":{\"a\":1}}}]}]}"));

        assertThat(
                openAi.path("messages")
                        .get(0)
                        .path("tool_calls")
                        .get(0)
                        .path("type")
                        .asText(),
                is("function"));
    }

    @Test
    public void toOpenAiChatRequestStringifiesAssistantToolCallArguments() throws IOException {
        JsonNode openAi = OllamaApiSupport.toOpenAiChatRequest(
                read(
                        "{\"messages\":["
                                + "{\"role\":\"assistant\",\"tool_calls\":[{\"function\":{\"name\":\"f\",\"arguments\":{\"a\":1}}}]}]}"));
        JsonNode arguments = openAi.path("messages")
                .get(0)
                .path("tool_calls")
                .get(0)
                .path("function")
                .path("arguments");
        // Ollama sends an object; OpenAI requires a JSON-encoded string.
        assertThat(arguments.isTextual(), is(true));
        assertThat(read(arguments.asText()).path("a").asInt(), is(1));
    }

    @Test
    public void toOpenAiChatRequestTurnsOllamaImagesIntoImageUrlParts() throws IOException {
        // Ollama sends images as a "images" array of bare base64 strings next to "content"; the
        // OpenAI surface only reads image_url content parts, so the images must be converted or
        // the model answers without them (while /api/show advertises "vision").
        String png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==";
        JsonNode openAi = OllamaApiSupport.toOpenAiChatRequest(read("{\"messages\":[{\"role\":\"user\","
                + "\"content\":\"what is this?\",\"images\":[\"" + png + "\"]}]}"));
        JsonNode message = openAi.path("messages").get(0);
        JsonNode content = message.path("content");
        assertThat(content.isArray(), is(true));
        assertThat(content.get(0).path("type").asText(), is("text"));
        assertThat(content.get(0).path("text").asText(), is("what is this?"));
        assertThat(content.get(1).path("type").asText(), is("image_url"));
        assertThat(content.get(1).path("image_url").path("url").asText(), is("data:image/png;base64," + png));
        assertThat(message.has("images"), is(false));
    }

    /**
     * The MIME type of a bare base64 image follows its magic bytes. llama.cpp decodes the payload by
     * content, so a wrong subtype would still work; these cases keep the labels honest.
     */
    @Test
    public void ollamaImageMimeTypeFollowsTheMagicBytes() throws IOException {
        String[][] cases = {
            {"/9j/4AAQ", "image/jpeg"},
            {"R0lGODlh", "image/gif"},
            {"UklGRh4A", "image/webp"},
            {"AAAA", "image/png"}
        };
        for (String[] imageAndMimeType : cases) {
            assertThat(
                    imageAndMimeType[0],
                    imageUrlOf(imageAndMimeType[0]),
                    is("data:" + imageAndMimeType[1] + ";base64," + imageAndMimeType[0]));
        }
    }

    /**
     * An image that already is a data URL passes through unchanged: a second {@code data:} prefix makes
     * llama.cpp reject the whole request ("Invalid uri-encoded base64 value"), so this is the case a
     * client would notice.
     */
    @Test
    public void ollamaImageThatIsAlreadyADataUrlIsPassedThrough() throws IOException {
        String dataUrl = "data:image/jpeg;base64,/9j/4AAQ";
        assertThat(imageUrlOf(dataUrl), is(dataUrl));
    }

    private static String imageUrlOf(String image) throws IOException {
        JsonNode openAi = OllamaApiSupport.toOpenAiChatRequest(read(
                "{\"messages\":[{\"role\":\"user\",\"content\":\"what is this?\",\"images\":[\"" + image + "\"]}]}"));
        JsonNode part = openAi.path("messages").get(0).path("content").get(1);
        assertThat(part.path("type").asText(), is("image_url"));
        return part.path("image_url").path("url").asText();
    }

    @Test
    public void toOpenAiChatRequestMapsFormatJsonToResponseFormat() throws IOException {
        JsonNode openAi = OllamaApiSupport.toOpenAiChatRequest(
                read("{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"format\":\"json\"}"));
        assertThat(openAi.path("response_format").path("type").asText(), is("json_object"));
    }

    @Test
    public void toOpenAiChatRequestMapsFormatSchemaToJsonSchema() throws IOException {
        // Ollama `format` as a JSON Schema object -> OpenAI response_format json_schema.
        JsonNode openAi =
                OllamaApiSupport.toOpenAiChatRequest(read("{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],"
                        + "\"format\":{\"type\":\"object\",\"properties\":{\"a\":{\"type\":\"string\"}}}}"));
        assertThat(openAi.path("response_format").path("type").asText(), is("json_schema"));
        assertThat(
                openAi.path("response_format")
                        .path("json_schema")
                        .path("schema")
                        .path("type")
                        .asText(),
                is("object"));
    }

    @Test
    public void toOpenAiChatRequestForwardsStopFromOptions() throws IOException {
        JsonNode openAi =
                OllamaApiSupport.toOpenAiChatRequest(read("{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],"
                        + "\"options\":{\"stop\":[\"\\n\",\"END\"]}}"));
        assertThat(openAi.path("stop").size(), is(2));
        assertThat(openAi.path("stop").get(1).asText(), is("END"));
    }

    private static void assertSamplingOptionsCopied(JsonNode out) {
        assertThat(out.path("min_p").asDouble(), is(0.05));
        assertThat(out.path("typical_p").asDouble(), is(0.9));
        assertThat(out.path("repeat_penalty").asDouble(), is(1.1));
        assertThat(out.path("repeat_last_n").asInt(), is(32));
        assertThat(out.path("presence_penalty").asDouble(), is(0.2));
        assertThat(out.path("frequency_penalty").asDouble(), is(0.3));
        assertThat(out.path("n_keep").asInt(), is(8));
    }

    @Test
    public void chatCopiesTheLlamaCppSamplingOptions() throws IOException {
        // Ollama's options keep llama.cpp's names; Open WebUI and others set repeat_penalty, min_p, ...
        assertSamplingOptionsCopied(
                OllamaApiSupport.toOpenAiChatRequest(
                        read(
                                "{\"model\":\"m\","
                                        + "\"messages\":[{\"role\":\"user\",\"content\":\"q\"}],\"options\":{\"min_p\":0.05,\"typical_p\":0.9,\"repeat_penalty\":1.1,\"repeat_last_n\":32,\"presence_penalty\":0.2,\"frequency_penalty\":0.3,\"num_keep\":8}}")));
    }

    @Test
    public void generateCopiesTheLlamaCppSamplingOptions() throws IOException {
        assertSamplingOptionsCopied(
                OllamaApiSupport.toOpenAiCompletionRequest(
                        read(
                                "{\"model\":\"m\",\"prompt\":\"p\",\"options\":{\"min_p\":0.05,\"typical_p\":0.9,\"repeat_penalty\":1.1,\"repeat_last_n\":32,\"presence_penalty\":0.2,\"frequency_penalty\":0.3,\"num_keep\":8}}")));
    }

    @Test
    public void infillCopiesTheLlamaCppSamplingOptions() throws IOException {
        assertSamplingOptionsCopied(
                OllamaApiSupport.toInfillRequest(
                        read(
                                "{\"model\":\"m\",\"prompt\":\"p\",\"suffix\":\"s\",\"options\":{\"min_p\":0.05,\"typical_p\":0.9,\"repeat_penalty\":1.1,\"repeat_last_n\":32,\"presence_penalty\":0.2,\"frequency_penalty\":0.3,\"num_keep\":8}}")));
    }

    /**
     * Ollama constrains a raw generation by {@code format} as it does a templated one; the raw route
     * dropped it and completed unconstrained. upstream's completion schema reads {@code json_schema};
     * {@code "json"} is a JSON object, as Ollama's own JSON grammar and upstream's {@code json_object}.
     */
    @Test
    public void aRawGenerationTakesItsFormatAsTheJsonSchema() throws IOException {
        JsonNode json =
                OllamaApiSupport.toOpenAiCompletionRequest(read("{\"prompt\":\"p\",\"raw\":true,\"format\":\"json\"}"));
        assertThat(json.path("json_schema"), is(read("{\"type\":\"object\"}")));
        assertThat(json.has("response_format"), is(false));

        String schema = "{\"type\":\"object\",\"properties\":{\"a\":{\"type\":\"string\"}}}";
        JsonNode schemaRequest = OllamaApiSupport.toOpenAiCompletionRequest(
                read("{\"prompt\":\"p\",\"raw\":true,\"format\":" + schema + "}"));
        assertThat(schemaRequest.path("json_schema"), is(read(schema)));

        for (String unset : new String[] {"\"\"", "null"}) {
            JsonNode none = OllamaApiSupport.toOpenAiCompletionRequest(
                    read("{\"prompt\":\"p\",\"raw\":true,\"format\":" + unset + "}"));
            assertThat(unset, none.has("json_schema"), is(false));
        }
    }

    /**
     * Ollama refuses a {@code format} that is neither {@code "json"} nor a schema object; the routes
     * ignored it and generated unconstrained. An empty string and {@code null} mean "not set", as there.
     */
    @Test
    public void aFormatThatIsNeitherJsonNorASchemaIsRefused() throws IOException {
        for (String format : new String[] {"\"yaml\"", "1", "[\"json\"]", "true"}) {
            IllegalArgumentException raw = assertThrows(
                    IllegalArgumentException.class,
                    () -> OllamaApiSupport.toOpenAiCompletionRequest(
                            read("{\"prompt\":\"p\",\"raw\":true,\"format\":" + format + "}")));
            assertThat(
                    raw.getMessage(), is("invalid format: " + format + "; expected \"json\" or a JSON Schema object"));
            IllegalArgumentException chat = assertThrows(
                    IllegalArgumentException.class,
                    () -> OllamaApiSupport.toOpenAiChatRequest(
                            read("{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"format\":" + format + "}")));
            assertThat(chat.getMessage(), is(raw.getMessage()));
        }
        JsonNode none = OllamaApiSupport.toOpenAiChatRequest(
                read("{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"format\":\"\"}"));
        assertThat(none.has("response_format"), is(false));
    }

    /**
     * Ollama's raw prompt places its images with {@code [img-N]} tags (the runner splits the prompt
     * there); llama.cpp's raw schema wants the server's media marker in their place and the bytes in
     * {@code multimodal_data}, in the order of the markers. The raw route dropped the images.
     */
    @Test
    public void aRawGenerationPutsItsImagesWhereTheirTagsAre() throws IOException {
        JsonNode openAi = OllamaApiSupport.toOpenAiCompletionRequest(
                read("{\"prompt\":\"a [img-1] b [img-0] c\",\"raw\":true,"
                        + "\"images\":[\"AAAA\",\"data:image/png;base64,BBBB\"]}"),
                "<m>");
        assertThat(openAi.path("prompt").path("prompt_string").asText(), is("a <m> b <m> c"));
        assertThat(openAi.path("prompt").path("multimodal_data"), is(read("[\"BBBB\",\"AAAA\"]")));

        JsonNode noImages = OllamaApiSupport.toOpenAiCompletionRequest(
                read("{\"prompt\":\"keep [img-0] as text\",\"raw\":true,\"images\":[]}"), "<m>");
        assertThat(noImages.path("prompt").asText(), is("keep [img-0] as text"));
    }

    /**
     * An image no tag places would go nowhere, and a tag naming no image has nothing to show; Ollama's
     * runner refuses the second ("invalid image index") and ignores the first. Both are refused here.
     */
    @Test
    public void aRawImageWithoutItsTagOrATagWithoutItsImageIsRefused() throws IOException {
        IllegalArgumentException untagged = assertThrows(
                IllegalArgumentException.class,
                () -> OllamaApiSupport.toOpenAiCompletionRequest(
                        read("{\"prompt\":\"a [img-0]\",\"raw\":true,\"images\":[\"AAAA\",\"BBBB\"]}"), "<m>"));
        assertThat(untagged.getMessage(), is("image 1 is not placed in the raw prompt; put [img-1] where it goes"));
        IllegalArgumentException unknown = assertThrows(
                IllegalArgumentException.class,
                () -> OllamaApiSupport.toOpenAiCompletionRequest(
                        read("{\"prompt\":\"a [img-2]\",\"raw\":true,\"images\":[\"AAAA\"]}"), "<m>"));
        assertThat(unknown.getMessage(), is("invalid image index: 2"));
        IllegalArgumentException notText = assertThrows(
                IllegalArgumentException.class,
                () -> OllamaApiSupport.toOpenAiCompletionRequest(
                        read("{\"prompt\":\"a [img-0]\",\"raw\":true,\"images\":[1]}"), "<m>"));
        assertThat(notText.getMessage(), is("'images[0]' must be a base64 string"));
    }

    /**
     * A {@code suffix} makes the request an infill, whose task reads {@code json_schema} as a completion
     * does; its {@code format} was dropped. Images have no place in an infill prompt and are refused
     * rather than dropped.
     */
    @Test
    public void anInfillGenerationTakesItsFormatAndRefusesImages() throws IOException {
        JsonNode infill =
                OllamaApiSupport.toInfillRequest(read("{\"prompt\":\"def f(\",\"suffix\":\")\",\"format\":\"json\"}"));
        assertThat(infill.path("json_schema"), is(read("{\"type\":\"object\"}")));
        IllegalArgumentException images = assertThrows(
                IllegalArgumentException.class,
                () -> OllamaApiSupport.toInfillRequest(
                        read("{\"prompt\":\"def f(\",\"suffix\":\")\",\"images\":[\"AAAA\"]}")));
        assertThat(images.getMessage(), is("images are not supported with a suffix (fill-in-the-middle)"));
    }

    @Test
    public void numPredictMinusTwoFillsTheContextAsUpstreamsMinusOne() throws IOException {
        // Ollama's -2 means "fill the context"; upstream's n_predict hard limit starts at -1 and rejects it.
        JsonNode out = OllamaApiSupport.toOpenAiCompletionRequest(
                read("{\"model\":\"m\",\"prompt\":\"p\",\"options\":{\"num_predict\":-2}}"));
        assertThat(out.path("max_tokens").asInt(), is(-1));
    }

    @Test
    public void repeatLastNMinusOneIsLeftToTheServerDefault() throws IOException {
        // Ollama's -1 means "the context size"; upstream's repeat_last_n accepts 0.. only.
        JsonNode out = OllamaApiSupport.toOpenAiCompletionRequest(
                read("{\"model\":\"m\",\"prompt\":\"p\",\"options\":{\"repeat_last_n\":-1}}"));
        assertThat(out.has("repeat_last_n"), is(false));
    }

    @Test
    public void showParametersCarryTheContextLengthWhereContinueReadsIt() throws IOException {
        // Continue's Ollama provider (core/llm/llms/Ollama.ts) reads the context length only from the
        // "parameters" text, one "<key> <value>" line each, matched by ^(\S+)\s+((?:".*")|\S+)$.
        String parameters = read(OllamaApiSupport.showJson("m", 4096, false))
                .path("parameters")
                .asText();
        java.util.regex.Pattern line = java.util.regex.Pattern.compile("^(\\S+)\\s+((?:\".*\")|\\S+)$");
        String numCtx = null;
        for (String l : parameters.split("\n")) {
            java.util.regex.Matcher m = line.matcher(l);
            if (m.matches() && "num_ctx".equals(m.group(1))) {
                numCtx = m.group(2);
            }
        }
        assertThat(numCtx, is("4096"));
    }

    @Test
    public void versionPassesCopilotsMinimumOllamaVersion() throws IOException {
        // VS Code Copilot's Ollama provider (ollamaProvider.ts, MINIMUM_OLLAMA_VERSION) refuses a server
        // below 0.6.4, comparing the dot-separated parts numerically; "0.1.0" failed it before any request.
        String[] version =
                read(OllamaApiSupport.versionJson()).path("version").asText().split("\\.");
        int[] minimum = {0, 6, 4};
        int cmp = 0;
        for (int i = 0; i < minimum.length && cmp == 0; i++) {
            cmp = Integer.compare(i < version.length ? Integer.parseInt(version[i]) : 0, minimum[i]);
        }
        assertThat(cmp >= 0, is(true));
    }

    @Test
    public void blockingChatAnswerCarriesTheThinkingLikeOllama() throws IOException {
        // Ollama puts a model's thinking into message.thinking.
        JsonNode blocking = read(OllamaApiSupport.toOllamaChatResponse(
                "{\"choices\":[{\"message\":{"
                        + "\"role\":\"assistant\",\"content\":\"42\",\"reasoning_content\":\"6 times 7\"},"
                        + "\"finish_reason\":\"stop\"}]}",
                "m"));
        assertThat(blocking.path("message").path("thinking").asText(), is("6 times 7"));
        assertThat(blocking.path("message").path("content").asText(), is("42"));
    }

    @Test
    public void streamedThinkingDeltaBecomesAThinkingLineLikeOllama() throws IOException {
        // A chunk carrying only reasoning is a line of its own in Ollama's stream, not dropped.
        String line = OllamaApiSupport.toOllamaContentLine(
                "{\"choices\":[{\"delta\":{\"reasoning_content\":\"6 ti\"}}]}", "m");
        assertThat(
                line == null ? "" : read(line).path("message").path("thinking").asText(), is("6 ti"));
    }

    @Test
    public void toolResultNameBecomesTheNameLlamaCppReads() throws IOException {
        // Ollama names the tool of a result "tool_name"; llama.cpp's chat parser reads "name" (chat.cpp:426),
        // which templates use to say which tool answered.
        JsonNode openAi = OllamaApiSupport.toOpenAiChatRequest(read("{\"model\":\"m\",\"messages\":["
                + "{\"role\":\"user\",\"content\":\"q\"},"
                + "{\"role\":\"tool\",\"content\":\"21 C\",\"tool_name\":\"get_weather\"}]}"));
        JsonNode toolMessage = openAi.path("messages").path(1);
        assertThat(toolMessage.path("name").asText(), is("get_weather"));
        assertThat(toolMessage.has("tool_name"), is(false));
    }

    @Test
    public void historyThinkingBecomesReasoningContent() throws IOException {
        // Clients send the thinking back in the history; llama.cpp's chat parser reads reasoning_content.
        JsonNode history = OllamaApiSupport.toOpenAiChatRequest(read("{\"model\":\"m\",\"messages\":["
                + "{\"role\":\"user\",\"content\":\"q\"},"
                + "{\"role\":\"assistant\",\"content\":\"42\",\"thinking\":\"6 times 7\"}]}"));
        assertThat(history.path("messages").path(1).path("reasoning_content").asText(), is("6 times 7"));
        assertThat(history.path("messages").path(1).has("thinking"), is(false));
    }

    @Test
    public void toOllamaChatResponseExtractsContentAndCountsAndDone() throws IOException {
        String openAi = "{\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"hello\"},"
                + "\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":7,\"completion_tokens\":3}}";
        JsonNode out = read(OllamaApiSupport.toOllamaChatResponse(openAi, "m"));
        assertThat(out.path("message").path("content").asText(), is("hello"));
        assertThat(out.path("done").asBoolean(), is(true));
        assertThat(out.path("done_reason").asText(), is("stop"));
        assertThat(out.path("prompt_eval_count").asInt(), is(7));
        assertThat(out.path("eval_count").asInt(), is(3));
    }

    @Test
    public void toOllamaChatResponseConvertsToolCallArgumentsToObject() throws IOException {
        String openAi = "{\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":null,"
                + "\"tool_calls\":[{\"id\":\"c1\",\"type\":\"function\","
                + "\"function\":{\"name\":\"get_weather\",\"arguments\":\"{\\\"city\\\":\\\"Paris\\\"}\"}}]}}]}";
        JsonNode out = read(OllamaApiSupport.toOllamaChatResponse(openAi, "m"));
        JsonNode arguments =
                out.path("message").path("tool_calls").get(0).path("function").path("arguments");
        // OpenAI string arguments become an Ollama object.
        assertThat(arguments.isObject(), is(true));
        assertThat(arguments.path("city").asText(), is("Paris"));
    }

    @Test
    public void toOllamaChatResponseTurnsEmptyToolCallArgumentsIntoAnEmptyObject() throws IOException {
        // a parameterless call can come back with "arguments": "" -- Ollama expects an object
        String openAi = "{\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":null,"
                + "\"tool_calls\":[{\"id\":\"c1\",\"type\":\"function\","
                + "\"function\":{\"name\":\"get_time\",\"arguments\":\"\"}}]}}]}";
        JsonNode out = read(OllamaApiSupport.toOllamaChatResponse(openAi, "m"));
        JsonNode arguments =
                out.path("message").path("tool_calls").get(0).path("function").path("arguments");
        assertThat(arguments.isObject(), is(true));
        assertThat(arguments.size(), is(0));
    }

    @Test
    public void toOllamaContentLineEmitsDeltaAndSkipsEmpty() throws IOException {
        String line = OllamaApiSupport.toOllamaContentLine("{\"choices\":[{\"delta\":{\"content\":\"he\"}}]}", "m");
        assertThat(line.endsWith("\n"), is(true));
        JsonNode parsed = read(line.trim());
        assertThat(parsed.path("message").path("content").asText(), is("he"));
        assertThat(parsed.path("done").asBoolean(), is(false));
        // Role-only / empty-content chunks emit nothing.
        assertThat(
                OllamaApiSupport.toOllamaContentLine("{\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}", "m"),
                is((String) null));
    }

    @Test
    public void generateHasSuffixDetectsFimRequests() throws IOException {
        assertThat(OllamaApiSupport.hasSuffix(read("{\"prompt\":\"a\",\"suffix\":\"b\"}")), is(true));
        assertThat(OllamaApiSupport.hasSuffix(read("{\"prompt\":\"a\"}")), is(false));
    }

    @Test
    public void generateMapsPromptAndOptionsToCompletionRequest() throws IOException {
        JsonNode openAi = OllamaApiSupport.toOpenAiCompletionRequest(
                read("{\"prompt\":\"once\",\"options\":{\"num_predict\":32,\"temperature\":0.5}}"));
        assertThat(openAi.path("prompt").asText(), is("once"));
        assertThat(openAi.path("max_tokens").asInt(), is(32));
        assertThat(openAi.path("temperature").asDouble(), is(0.5));
    }

    @Test
    public void generateWithSuffixMapsToInfillRequest() throws IOException {
        JsonNode infill = OllamaApiSupport.toInfillRequest(
                read("{\"prompt\":\"pre\",\"suffix\":\"suf\",\"options\":{\"num_predict\":16}}"));
        assertThat(infill.path("input_prefix").asText(), is("pre"));
        assertThat(infill.path("input_suffix").asText(), is("suf"));
        assertThat(infill.path("n_predict").asInt(), is(16));
    }

    @Test
    public void generateWithSuffixKeepsTheSamplingOptionsOfAPlainGenerate() throws IOException {
        // FIM clients stop a completion with options.stop; upstream's /infill reads stop, top_p, top_k
        // and seed through the same completion schema as /completions.
        JsonNode infill = OllamaApiSupport.toInfillRequest(read("{\"prompt\":\"pre\",\"suffix\":\"suf\","
                + "\"options\":{\"stop\":[\"\\n\\n\"],\"top_p\":0.9,\"top_k\":40,\"seed\":7}}"));
        assertThat(infill.path("stop").path(0).asText(), is("\n\n"));
        assertThat(infill.path("top_p").asDouble(), is(0.9));
        assertThat(infill.path("top_k").asInt(), is(40));
        assertThat(infill.path("seed").asInt(), is(7));
    }

    @Test
    public void generateTextExtractors() {
        assertThat(OllamaApiSupport.Generated.fromCompletion("{\"choices\":[{\"text\":\"hi\"}]}").text, is("hi"));
        assertThat(OllamaApiSupport.Generated.fromInfill("{\"content\":\"world\"}").text, is("world"));
        // Unexpected bodies yield empty text and "stop" rather than throwing.
        OllamaApiSupport.Generated broken = OllamaApiSupport.Generated.fromCompletion("not json");
        assertThat(broken.text, is(""));
        assertThat(broken.doneReason, is("stop"));
        assertThat(OllamaApiSupport.Generated.fromInfill("not json").doneReason, is("stop"));
    }

    @Test
    public void generateWithoutCountsOmitsThemAndToolCallsAreStop() throws IOException {
        JsonNode out = read(OllamaApiSupport.toOllamaGenerateResponse(
                OllamaApiSupport.Generated.fromCompletion("{\"choices\":[{\"text\":\"hi\"}]}"), "m"));
        assertThat(out.has("prompt_eval_count"), is(false));
        assertThat(out.has("eval_count"), is(false));
        assertThat(OllamaApiSupport.toOllamaDoneReason("tool_calls"), is("stop"));
        assertThat(OllamaApiSupport.toOllamaDoneReason(null), is("stop"));
        assertThat(OllamaApiSupport.finishReasonOf("{\"choices\":[{\"finish_reason\":null}]}"), is((String) null));
        assertThat(OllamaApiSupport.finishReasonOf("not json"), is((String) null));
    }

    @Test
    public void generateResponseAndStreamShapes() throws IOException {
        JsonNode nonStream = read(OllamaApiSupport.toOllamaGenerateResponse(
                new OllamaApiSupport.Generated("hello", "stop", -1L, -1L), "m"));
        assertThat(nonStream.path("response").asText(), is("hello"));
        assertThat(nonStream.path("done").asBoolean(), is(true));

        String stream =
                OllamaApiSupport.toOllamaGenerateStream(new OllamaApiSupport.Generated("hello", "stop", -1L, -1L), "m");
        String[] lines = stream.trim().split("\n");
        assertThat(lines.length, is(2));
        JsonNode first = read(lines[0]);
        assertThat(first.path("response").asText(), is("hello"));
        assertThat(first.path("done").asBoolean(), is(false));
        assertThat(read(lines[1]).path("done").asBoolean(), is(true));
    }

    @Test
    public void toOllamaDoneLineCarriesAccumulatedToolCalls() throws IOException {
        ToolCallDeltaAccumulator accumulator = new ToolCallDeltaAccumulator();
        accumulator.accept("{\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"c1\","
                + "\"function\":{\"name\":\"f\",\"arguments\":\"{\\\"a\\\":1}\"}}]}}]}");
        JsonNode out = read(OllamaApiSupport.toOllamaDoneLine("m", accumulator).trim());
        assertThat(out.path("done").asBoolean(), is(true));
        JsonNode arguments =
                out.path("message").path("tool_calls").get(0).path("function").path("arguments");
        assertThat(arguments.isObject(), is(true));
        assertThat(arguments.path("a").asInt(), is(1));
    }

    @Test
    public void aGenerateRequestWithoutAPromptOrSuffixOnlyLoadsTheModel() throws IOException {
        assertThat(OllamaApiSupport.isLoadOnly(read("{\"model\":\"m\"}")), is(true));
        assertThat(OllamaApiSupport.isLoadOnly(read("{\"prompt\":\"\"}")), is(true));
        assertThat(OllamaApiSupport.isLoadOnly(read("{\"prompt\":\"hi\"}")), is(false));
        // an empty prompt with a suffix is fill-in-the-middle at the start of a file, not a preload
        assertThat(OllamaApiSupport.isLoadOnly(read("{\"prompt\":\"\",\"suffix\":\"x\"}")), is(false));
        JsonNode answer = read(OllamaApiSupport.toOllamaLoadResponse("m"));
        assertThat(answer.path("done_reason").asText(), is("load"));
        assertThat(answer.path("done").asBoolean(false), is(true));
        assertThat(answer.path("response").asText(null), is(""));
    }

    /**
     * Only an absent, {@code null} or empty textual {@code prompt} asks for the load answer. Jackson's
     * {@code asText("")} reads an array or an object as {@code ""}, so a malformed prompt was answered
     * as a successful preload, where Ollama refuses the body with a 400 when it binds the JSON.
     */
    @Test
    public void aNonTextPromptIsNotALoadRequest() throws IOException {
        assertThat(OllamaApiSupport.isLoadOnly(read("{\"prompt\":[\"a\"]}")), is(false));
        assertThat(OllamaApiSupport.isLoadOnly(read("{\"prompt\":{}}")), is(false));
        assertThat(OllamaApiSupport.isLoadOnly(read("{\"prompt\":null}")), is(true));
    }

    @Test
    public void aPromptThatIsNotAStringIsRefused() throws IOException {
        for (String body : new String[] {"{\"prompt\":[\"a\"]}", "{\"prompt\":{}}", "{\"prompt\":1}"}) {
            JsonNode request = read(body);
            IllegalArgumentException e = assertThrows(
                    IllegalArgumentException.class, () -> OllamaApiSupport.requireTextualPrompt(request), body);
            assertThat(body, e.getMessage(), is("'prompt' must be a string"));
        }
        for (String body : new String[] {"{\"prompt\":\"a\"}", "{\"prompt\":\"\"}", "{\"prompt\":null}", "{}"}) {
            OllamaApiSupport.requireTextualPrompt(read(body));
        }
    }

    /**
     * A {@code messages} element that is not an object (a string, {@code null}, an array) failed the
     * translator's cast to an object with a {@link ClassCastException}, which left the HTTP handler
     * without an answer. Upstream refuses it as {@code "Expected 'message' to be an object"}
     * ({@code common/chat.cpp}); the translator now does the same and names the element.
     */
    @Test
    public void aNonObjectMessageIsRejectedWithItsIndex() throws IOException {
        for (String element : new String[] {"\"hi\"", "null", "[1]"}) {
            JsonNode request = read("{\"messages\":[{\"role\":\"user\",\"content\":\"a\"}," + element + "]}");
            IllegalArgumentException e =
                    assertThrows(IllegalArgumentException.class, () -> OllamaApiSupport.toOpenAiChatRequest(request));
            assertThat(
                    element, e.getMessage(), containsString("Expected 'messages[1]' to be an object, got " + element));
        }
    }

    /** A {@code messages} that is present but no array was read as no messages at all. */
    @Test
    public void messagesThatAreNoArrayAreRejected() throws IOException {
        IllegalArgumentException e = assertThrows(
                IllegalArgumentException.class,
                () -> OllamaApiSupport.toOpenAiChatRequest(read("{\"messages\":\"hi\"}")));
        assertThat(e.getMessage(), is("Expected 'messages' to be an array, got \"hi\""));
        // absent and null stay "no messages", which the caller decides about
        assertThat(
                OllamaApiSupport.toOpenAiChatRequest(read("{}"))
                        .path("messages")
                        .size(),
                is(0));
        assertThat(
                OllamaApiSupport.toOpenAiChatRequest(read("{\"messages\":null}"))
                        .path("messages")
                        .size(),
                is(0));
    }

    /**
     * Ollama refuses {@code "raw": true} together with a {@code system}, a {@code template} or a
     * {@code context} ({@code GenerateHandler}: "raw mode does not support template, system, or
     * context"); empty values are no conflict, as {@code req.System != ""} and {@code len(req.Context) > 0}
     * read them there. The server dropped them silently and answered as if they had not been sent.
     */
    @Test
    public void rawWithSystemTemplateOrContextConflictsLikeOllama() throws IOException {
        for (String conflict : new String[] {
            "{\"prompt\":\"p\",\"raw\":true,\"system\":\"x\"}",
            "{\"prompt\":\"p\",\"raw\":true,\"template\":\"{{ .Prompt }}\"}",
            "{\"prompt\":\"p\",\"raw\":true,\"context\":[1,2]}",
            "{\"raw\":true,\"system\":\"x\"}"
        }) {
            assertThat(conflict, OllamaApiSupport.hasRawConflict(read(conflict)), is(true));
        }
        for (String accepted : new String[] {
            "{\"prompt\":\"p\",\"raw\":true}",
            "{\"prompt\":\"p\",\"raw\":true,\"system\":\"\",\"template\":\"\",\"context\":[]}",
            "{\"prompt\":\"p\",\"raw\":false,\"system\":\"x\",\"context\":[1]}",
            "{\"prompt\":\"p\",\"system\":\"x\"}"
        }) {
            assertThat(accepted, OllamaApiSupport.hasRawConflict(read(accepted)), is(false));
        }
    }

    /**
     * Ollama answers an {@code /api/chat} without messages by loading the model ({@code ChatHandler}:
     * {@code len(req.Messages) == 0}, which absent, {@code null} and {@code []} all are), the chat twin
     * of the {@code /api/generate} preload.
     */
    @Test
    public void aChatRequestWithoutMessagesOnlyLoadsTheModel() throws IOException {
        assertThat(OllamaApiSupport.isChatLoadOnly(read("{\"model\":\"m\"}")), is(true));
        assertThat(OllamaApiSupport.isChatLoadOnly(read("{\"messages\":null}")), is(true));
        assertThat(OllamaApiSupport.isChatLoadOnly(read("{\"messages\":[]}")), is(true));
        assertThat(
                OllamaApiSupport.isChatLoadOnly(read("{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}")),
                is(false));
        // a messages value of the wrong type is no preload; the translation refuses it
        assertThat(OllamaApiSupport.isChatLoadOnly(read("{\"messages\":\"hi\"}")), is(false));
        JsonNode answer = read(OllamaApiSupport.toOllamaChatLoadResponse("m"));
        assertThat(answer.path("model").asText(), is("m"));
        assertThat(answer.path("done").asBoolean(false), is(true));
        assertThat(answer.path("done_reason").asText(), is("load"));
        assertThat(answer.path("message").path("role").asText(), is("assistant"));
        assertThat(answer.path("message").path("content").asText(null), is(""));
        assertThat(answer.path("created_at").isTextual(), is(true));
    }

    /**
     * Ollama's {@code "think": true} asks the model to think. It was dropped, so a server whose thinking
     * is off by default ({@code --reasoning off}) ignored it silently, while upstream lets a request's
     * {@code chat_template_kwargs.enable_thinking: true} switch thinking on over that default
     * ({@code server-common.cpp}). The value must stay a JSON boolean: upstream refuses a string.
     */
    @Test
    public void thinkTrueSwitchesThinkingOnExplicitly() throws IOException {
        OpenAiRequestMapper mapper = new OpenAiRequestMapper();
        for (JsonNode openAi : new JsonNode[] {
            OllamaApiSupport.toOpenAiChatRequest(
                    read("{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"think\":true}")),
            OllamaApiSupport.toOpenAiChatRequestFromGenerate(read("{\"prompt\":\"hi\",\"think\":true}"))
        }) {
            JsonNode kwargs =
                    read(mapper.toInferenceParameters(openAi).toJson()).path("chat_template_kwargs");
            assertThat(openAi.toString(), kwargs.path("enable_thinking").isBoolean(), is(true));
            assertThat(openAi.toString(), kwargs.path("enable_thinking").asBoolean(), is(true));
        }
    }

    /**
     * {@code /api/generate} without {@code raw} goes through the chat template as a synthetic
     * {@code /api/chat} request; the request's {@code model}, {@code options}, {@code format} and
     * {@code think} must reach that request, or a generate call silently loses its sampling, its
     * output format and its thinking switch.
     */
    @Test
    public void generateWithoutRawCarriesOptionsFormatAndThinkIntoTheChatRequest() throws IOException {
        JsonNode openAi = OllamaApiSupport.toOpenAiChatRequestFromGenerate(read("{\"model\":\"m\",\"prompt\":\"q\","
                + "\"system\":\"s\",\"format\":\"json\",\"think\":false,"
                + "\"options\":{\"temperature\":0.2,\"num_predict\":16,\"stop\":[\"END\"]}}"));
        assertThat(openAi.path("model").asText(), is("m"));
        assertThat(openAi.path("temperature").asDouble(), is(0.2));
        assertThat(openAi.path("max_tokens").asInt(), is(16));
        assertThat(openAi.path("stop").get(0).asText(), is("END"));
        assertThat(openAi.path("response_format").path("type").asText(), is("json_object"));
        assertThat(openAi.path("reasoning_effort").asText(), is("none"));
        JsonNode messages = openAi.path("messages");
        assertThat(messages.size(), is(2));
        assertThat(messages.get(0).path("role").asText(), is("system"));
        assertThat(messages.get(0).path("content").asText(), is("s"));
        assertThat(messages.get(1).path("role").asText(), is("user"));
        assertThat(messages.get(1).path("content").asText(), is("q"));
    }

    @Test
    public void generateWithASchemaFormatAndNoSystemIsOneUserMessageWithAJsonSchema() throws IOException {
        String schema = "{\"type\":\"object\",\"properties\":{\"a\":{\"type\":\"integer\"}}}";
        JsonNode openAi = OllamaApiSupport.toOpenAiChatRequestFromGenerate(
                read("{\"model\":\"m\",\"prompt\":\"q\",\"format\":" + schema + "}"));
        assertThat(openAi.path("response_format").path("type").asText(), is("json_schema"));
        assertThat(openAi.path("response_format").path("json_schema").path("schema"), is(read(schema)));
        JsonNode messages = openAi.path("messages");
        assertThat(messages.size(), is(1));
        assertThat(messages.get(0).path("role").asText(), is("user"));
        assertThat(messages.get(0).path("content").asText(), is("q"));
    }

    @Test
    public void anOptionThatIsNotANumberIsRefused() throws IOException {
        // Ollama answers a mistyped option with 400; the translation dropped it without a word, so the
        // request ran with the server's defaults (num_predict "8": no cap at all).
        for (String option : new String[] {
            "temperature",
            "top_p",
            "top_k",
            "seed",
            "num_predict",
            "min_p",
            "typical_p",
            "repeat_penalty",
            "presence_penalty",
            "frequency_penalty",
            "num_keep",
            "repeat_last_n"
        }) {
            JsonNode request = read("{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],"
                    + "\"options\":{\"" + option + "\":\"8\"}}");
            IllegalArgumentException e = assertThrows(
                    IllegalArgumentException.class, () -> OllamaApiSupport.toOpenAiChatRequest(request), option);
            assertThat(e.getMessage(), is("Field '" + option + "': type must be number, but is string"));
        }
    }
}
