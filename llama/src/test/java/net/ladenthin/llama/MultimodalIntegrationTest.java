// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.File;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Base64;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.TimeUnit;
import net.ladenthin.llama.parameters.ChatRequest;
import net.ladenthin.llama.parameters.InferenceParameters;
import net.ladenthin.llama.parameters.ModelParameters;
import net.ladenthin.llama.value.ChatMessage;
import net.ladenthin.llama.value.ChatResponse;
import net.ladenthin.llama.value.ContentPart;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

/**
 * End-to-end multimodal regression. Loads a vision-capable model + matching
 * mmproj, sends a real {@link ContentPart#imageFile(java.nio.file.Path)}
 * alongside a text prompt via the typed {@link ChatMessage#userMultimodal}
 * surface, and asserts that the model returns a non-empty reply.
 *
 * <p>The test exercises:</p>
 * <ul>
 *   <li>{@link ModelParameters#setMmproj(String)} wiring through to the native side;</li>
 *   <li>{@link InferenceParameters#withMessages(java.util.List)} serialising
 *       multipart {@code content} as the OAI array-form;</li>
 *   <li>the upstream {@code oaicompat_chat_params_parse} routing
 *       {@code image_url} blocks through the compiled-in {@code mtmd} pipeline.</li>
 * </ul>
 *
 * <p>Self-skips when any of the three system properties is unset or its file
 * is missing, so it runs only in CI (or local dev where the user staged the
 * files explicitly). See {@code .github/workflows/publish.yml} env vars
 * {@code VISION_MODEL_URL} / {@code VISION_MMPROJ_URL}; the image defaults to
 * the committed test resource and can be overridden with
 * {@code net.ladenthin.llama.vision.image}.</p>
 *
 * <p><b>Image source:</b> the CI default is
 * <a href="https://commons.wikimedia.org/wiki/File:20200601_135745_Flowers_and_Bees.jpg">
 * {@code File:20200601_135745_Flowers_and_Bees.jpg}</a> on Wikimedia Commons
 * by Bernard Ladenthin (the project copyright holder); published there under
 * CC-BY-4.0 and additionally granted under MIT to this project by the same
 * author. Any image the test machine can reach works at runtime &#x2014; the
 * URL is just an env var.</p>
 *
 * <p>Implements the vision feature originally requested in the pre-fork upstream repository:
 * <a href="https://github.com/kherud/java-llama.cpp/issues/103">https://github.com/kherud/java-llama.cpp/issues/103</a>
 * and
 * <a href="https://github.com/kherud/java-llama.cpp/issues/34">https://github.com/kherud/java-llama.cpp/issues/34</a>.</p>
 */
@ClaudeGenerated(
        purpose = "End-to-end vision regression: real vision GGUF + mmproj + author-licensed (MIT) "
                + "test image fed through the typed ChatMessage(role, List<ContentPart>) API; "
                + "asserts non-empty reply to prove the OAI multipart content round-trips through "
                + "the upstream mtmd pipeline. Implements the pre-fork upstream vision requests "
                + "https://github.com/kherud/java-llama.cpp/issues/103 and "
                + "https://github.com/kherud/java-llama.cpp/issues/34.")
public class MultimodalIntegrationTest {

    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final byte[] RED_PNG = Base64.getDecoder()
            .decode("iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAACXBIWXMAAAPoAAAD6AG1e1JrAAAAl0lEQVR4nO2S"
                    + "wQkAMBCD3H9pO0QfchDIACpBOD1yAidAXtFdiLsjJ3AC5BXdhbg7cgInQF7RXYi7IydwAuQV3YW4O3ICJ0Be0V2IuyM"
                    + "ncALkFd2FuDtyAidAXtFdiLsjJ3AC5BXdhbg7cgInQF7RXYi7IydwAuQV3YW4O3ICJ0Be0V2IuyMncALkFd2FuDtyAid"
                    + "AXvHnQg+p3PDiuUoi2QAAAABJRU5ErkJggg==");
    private static final byte[] BLUE_PNG = Base64.getDecoder()
            .decode("iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAACXBIWXMAAAPoAAAD6AG1e1JrAAAAlklEQVR4nO2S"
                    + "wQkAMBCD3H9pu0M/chBwACMBPA65gRtAXtFdiLuQG7gB5BXdhbgLuYEbQF7RXYi7kBu4AeQV3YW4C7mBG0Be0V2Iu5Ab"
                    + "uAHkFd2FuAu5gRtAXtFdiLuQG7gB5BXdhbgLuYEbQF7RXYi7kBu4AeQV3YW4C7mBG0Be0V2Iu5AbuAHkFd2FuAu5gRtA"
                    + "XvGfB8f88OJzBsmpAAAAAElFTkSuQmCC");
    private static String modelPath;
    private static String mmprojPath;
    private static String imagePath;
    private static LlamaModel model;

    @BeforeAll
    public static void setup() {
        modelPath = TestConstants.resolveModelProperty(TestConstants.PROP_VISION_MODEL_PATH);
        mmprojPath = TestConstants.resolveModelProperty(TestConstants.PROP_VISION_MMPROJ_PATH);
        // Image path falls back to the committed test resource when the
        // -D property is unset, so the test works on local dev checkouts
        // without any extra wiring. The model / mmproj remain externally
        // staged because their combined size (~600 MB) is too large to
        // commit.
        imagePath = TestConstants.resolveModelProperty(
                TestConstants.PROP_VISION_IMAGE_PATH, TestConstants.DEFAULT_VISION_IMAGE_PATH);

        Assumptions.assumeTrue(
                modelPath != null && !modelPath.isEmpty(),
                "Vision model path not set (-D" + TestConstants.PROP_VISION_MODEL_PATH + "=...)");
        Assumptions.assumeTrue(
                mmprojPath != null && !mmprojPath.isEmpty(),
                "Vision mmproj path not set (-D" + TestConstants.PROP_VISION_MMPROJ_PATH + "=...)");

        Assumptions.assumeTrue(new File(modelPath).exists(), "Vision model file missing: " + modelPath);
        Assumptions.assumeTrue(new File(mmprojPath).exists(), "Vision mmproj file missing: " + mmprojPath);
        Assumptions.assumeTrue(new File(imagePath).exists(), "Vision image file missing: " + imagePath);

        int gpuLayers = Integer.getInteger(TestConstants.PROP_TEST_NGL, TestConstants.DEFAULT_TEST_NGL);

        ModelParameters parameters = new ModelParameters()
                .setCtxSize(2048)
                .setModel(modelPath)
                .setMmproj(mmprojPath)
                .setGpuLayers(gpuLayers)
                .setFit(false);
        if (gpuLayers == 0) {
            parameters.setDevices("none").setMmprojOffload(false);
        }
        model = new LlamaModel(parameters);
        assertTrue(model.getModelMeta().supportsVision(), "loaded model + mmproj must advertise vision input");
        // The native side must emit all three modality keys. ModelMetaTest feeds ModelMeta its own
        // JSON, so nothing there would notice getModelMetaJson() dropping one; this is the only
        // place the real emitter is observed. Presence, not value -- whether this particular model
        // accepts video is the model's business.
        JsonNode modalities = model.getModelMeta().asJson().path("modalities");
        for (String key : new String[] {"vision", "audio", "video"}) {
            assertTrue(modalities.has(key), "getModelMetaJson() must emit modalities." + key + ": " + modalities);
        }
    }

    @AfterAll
    public static void tearDown() {
        if (model != null) {
            model.close();
        }
    }

    /**
     * Build a multimodal user message, send it, and assert the reply is
     * non-empty. We do <em>not</em> assert reply content semantics &#x2014;
     * a 500M vision model can caption an apple inaccurately, and CI must
     * not flap on model quality. The point of the test is that the
     * end-to-end image-bytes &#x2192; base64 data URI &#x2192; OAI multipart
     * JSON &#x2192; upstream mtmd &#x2192; non-empty token stream pipeline
     * works without crashing.
     */
    @Timeout(value = 240_000, unit = TimeUnit.MILLISECONDS)
    @Test
    public void multimodalRequestProducesNonEmptyReply() throws Exception {
        ChatMessage userMsg = ChatMessage.userMultimodal(
                ContentPart.text("Describe what you see in this image in one short sentence."),
                ContentPart.imageFile(Paths.get(imagePath)));

        String reply = model.chatCompleteText(new InferenceParameters("")
                .withMessages(Collections.singletonList(userMsg))
                .withNPredict(48)
                .withTemperature(0.0f));

        assertNotNull(reply, "chatCompleteText must return a string, not null");
        assertFalse(reply.trim().isEmpty(), "reply must be non-empty for a multimodal prompt; got: \"" + reply + "\"");
    }

    /** The typed ChatRequest surface must preserve multipart image content too. */
    @Timeout(value = 240_000, unit = TimeUnit.MILLISECONDS)
    @Test
    public void typedChatRequestProducesNonEmptyReply() throws Exception {
        ChatRequest request = ChatRequest.empty()
                .appendMessage(ChatMessage.userMultimodal(
                        ContentPart.text("Describe this image in one short sentence."),
                        ContentPart.imageFile(Paths.get(imagePath))))
                .withInferenceCustomizer(params -> params.withNPredict(48).withTemperature(0.0f));

        ChatResponse response = model.chat(request);
        assertFalse(response.getFirstContent().trim().isEmpty(), "typed chat response must be non-empty");
    }

    /** Streaming uses the same decoded media buffers and must emit assistant content. */
    @Timeout(value = 240_000, unit = TimeUnit.MILLISECONDS)
    @Test
    public void multimodalStreamingProducesContent() throws Exception {
        ChatMessage userMsg = ChatMessage.userMultimodal(
                ContentPart.text("Describe this image briefly."), ContentPart.imageFile(Paths.get(imagePath)));
        InferenceParameters params = InferenceParameters.empty()
                .withMessages(Collections.singletonList(userMsg))
                .withNPredict(32)
                .withTemperature(0.0f);
        List<String> chunks = new ArrayList<String>();

        model.streamChatCompletion(params, chunks::add);

        StringBuilder content = new StringBuilder();
        for (String chunk : chunks) {
            JsonNode choices = MAPPER.readTree(chunk).path("choices");
            if (choices.isArray() && choices.size() > 0) {
                JsonNode fragment = choices.get(0).path("delta").path("content");
                if (fragment.isTextual()) {
                    content.append(fragment.asText());
                }
            }
        }
        assertFalse(content.toString().trim().isEmpty(), "streamed multimodal content must be non-empty");
    }

    /**
     * Semantic guard against the old false-positive path where only the textual media marker was tokenized.
     * Two synthetic images with the same prompt must be identified by their actual dominant colors.
     */
    @Timeout(value = 240_000, unit = TimeUnit.MILLISECONDS)
    @Test
    public void imageBytesAffectTheAnswer() throws Exception {
        String red = identifyColor(RED_PNG);
        String blue = identifyColor(BLUE_PNG);
        assertTrue(red.contains("red"), "red image must be identified as red; got: " + red);
        assertTrue(blue.contains("blue"), "blue image must be identified as blue; got: " + blue);
    }

    /**
     * Sanity check that a multimodal call followed by a plain text-only call
     * on the same model instance both succeed &#x2014; verifies the parts /
     * legacy split in {@code ParameterJsonSerializer.buildMessages} doesn't
     * poison the inference context for subsequent requests.
     */
    @Timeout(value = 240_000, unit = TimeUnit.MILLISECONDS)
    @Test
    public void multimodalThenTextOnSameModel() throws Exception {
        ChatMessage img = ChatMessage.userMultimodal(
                ContentPart.text("What is this?"), ContentPart.imageFile(Paths.get(imagePath)));
        String firstReply = model.chatCompleteText(new InferenceParameters("")
                .withMessages(Collections.singletonList(img))
                .withNPredict(24)
                .withTemperature(0.0f));
        assertNotNull(firstReply);

        ChatMessage textOnly = new ChatMessage("user", "Reply with the single word: ok");
        String secondReply = model.chatCompleteText(new InferenceParameters("")
                .withMessages(Collections.singletonList(textOnly))
                .withNPredict(8)
                .withTemperature(0.0f));
        assertNotNull(secondReply);
        assertTrue(
                secondReply.trim().length() > 0,
                "text-only call after multimodal must still produce tokens; got: \"" + secondReply + "\"");
    }

    private static String identifyColor(byte[] png) {
        ChatMessage message = ChatMessage.userMultimodal(
                ContentPart.text("What is the dominant color? Reply with only red or blue."),
                ContentPart.imageBytes(png, "image/png"));
        return model.chatCompleteText(InferenceParameters.empty()
                        .withMessages(Collections.singletonList(message))
                        .withNPredict(8)
                        .withTemperature(0.0f))
                .trim()
                .toLowerCase(java.util.Locale.ROOT);
    }
}
