// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Base64;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.TimeUnit;
import net.ladenthin.llama.exception.InvalidRequestException;
import net.ladenthin.llama.parameters.ChatRequest;
import net.ladenthin.llama.parameters.InferenceParameters;
import net.ladenthin.llama.parameters.ModelParameters;
import net.ladenthin.llama.value.ChatMessage;
import net.ladenthin.llama.value.ChatResponse;
import net.ladenthin.llama.value.ContentPart;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

/**
 * Real-model coverage for <b>audio input</b> (llama.cpp discussion #13759). Loads an audio-capable
 * model (Ultravox / Qwen2.5-Omni) with its audio {@code --mmproj} and sends a multipart message
 * carrying a {@link ContentPart#audioFile(java.nio.file.Path)} clip, exercising:
 * <ul>
 *   <li>{@link ModelParameters#setMmproj(String)} wiring an audio encoder;</li>
 *   <li>{@code ParameterJsonSerializer.buildMessages} emitting the OAI {@code input_audio} part;</li>
 *   <li>the upstream {@code oaicompat_chat_params_parse} routing {@code input_audio} through the
 *       compiled-in {@code mtmd} audio pipeline.</li>
 * </ul>
 *
 * <p>The audio prompt defaults to the committed clip
 * {@link TestConstants#DEFAULT_AUDIO_INPUT_PATH} (override via
 * {@link TestConstants#PROP_AUDIO_PATH}). The model and mmproj default to CI's Ultravox pair
 * ({@link TestConstants#DEFAULT_AUDIO_MODEL_PATH}, feature {@code audio} of {@code .github/models.csv};
 * override via {@link TestConstants#PROP_AUDIO_MODEL_PATH} / {@link TestConstants#PROP_AUDIO_MMPROJ_PATH}).
 * Self-skips when a referenced file is missing.
 */
public class AudioInputIntegrationTest {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    /** A 1x1 PNG: an image, which an audio-only projector does not read. */
    private static final byte[] TINY_PNG = Base64.getDecoder()
            .decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC");

    private static LlamaModel model;
    private static String audioPath;

    @BeforeAll
    public static void setup() {
        String modelPath = TestConstants.resolveModelProperty(
                TestConstants.PROP_AUDIO_MODEL_PATH, TestConstants.DEFAULT_AUDIO_MODEL_PATH);
        String mmprojPath = TestConstants.resolveModelProperty(
                TestConstants.PROP_AUDIO_MMPROJ_PATH, TestConstants.DEFAULT_AUDIO_MMPROJ_PATH);
        audioPath = TestConstants.resolveModelProperty(
                TestConstants.PROP_AUDIO_PATH, TestConstants.DEFAULT_AUDIO_INPUT_PATH);

        Assumptions.assumeTrue(new File(modelPath).exists(), "Audio model file missing: " + modelPath);
        Assumptions.assumeTrue(new File(mmprojPath).exists(), "Audio mmproj file missing: " + mmprojPath);
        Assumptions.assumeTrue(new File(audioPath).exists(), "Audio clip missing: " + audioPath);

        int gpuLayers = Integer.getInteger(TestConstants.PROP_TEST_NGL, TestConstants.DEFAULT_TEST_NGL);
        ModelParameters parameters = new ModelParameters()
                .setCtxSize(4096)
                .setModel(modelPath)
                .setMmproj(mmprojPath)
                .setGpuLayers(gpuLayers)
                .setFit(false);
        if (gpuLayers == 0) {
            parameters.setDevices("none").setMmprojOffload(false);
        }
        model = new LlamaModel(parameters);
        assertTrue(model.supportsAudio(), "loaded model + mmproj must advertise audio input");
    }

    @AfterAll
    public static void tearDown() {
        if (model != null) {
            model.close();
        }
    }

    @Test
    @DisplayName("an input_audio content part reaches the model and yields a non-empty reply")
    @Timeout(value = 240_000, unit = TimeUnit.MILLISECONDS)
    public void audioInputProducesNonEmptyReply() throws IOException {
        ChatMessage message = ChatMessage.userMultimodal(
                ContentPart.text("Transcribe the audio."), ContentPart.audioFile(Paths.get(audioPath)));

        String reply = model.chatCompleteText(InferenceParameters.empty()
                .withMessages(Collections.singletonList(message))
                .withNPredict(64));

        assertFalse(reply.trim().isEmpty(), "reply must be non-empty for an audio prompt; got: \"" + reply + "\"");
    }

    @Test
    @DisplayName("the typed ChatRequest surface carries the audio part too")
    @Timeout(value = 240_000, unit = TimeUnit.MILLISECONDS)
    public void typedChatRequestCarriesTheAudio() throws IOException {
        ChatRequest request = ChatRequest.empty()
                .appendMessage(ChatMessage.userMultimodal(
                        ContentPart.text("What do you hear?"), ContentPart.audioFile(Paths.get(audioPath))))
                .withInferenceCustomizer(params -> params.withNPredict(32).withTemperature(0.0f));

        ChatResponse response = model.chat(request);

        assertFalse(response.getFirstContent().trim().isEmpty(), "typed chat response must be non-empty");
    }

    @Test
    @DisplayName("a streamed chat with an audio part emits assistant content")
    @Timeout(value = 240_000, unit = TimeUnit.MILLISECONDS)
    public void streamedAudioChatProducesContent() throws IOException {
        InferenceParameters params = InferenceParameters.empty()
                .withMessages(Collections.singletonList(ChatMessage.userMultimodal(
                        ContentPart.text("What do you hear?"), ContentPart.audioFile(Paths.get(audioPath)))))
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
        assertFalse(content.toString().trim().isEmpty(), "streamed audio content must be non-empty");
    }

    /**
     * The clip reaches the model as embeddings, not as a marker: the prompt with it is longer than the
     * same prompt without. And Whisper reads 30 s per chunk, so a clip of 2.6 s is one chunk and one of
     * 35 s two -- twice the embeddings, which a projector that read only the first 30 s (or each sample
     * once per clip) would not give.
     */
    @Test
    @DisplayName("the clip is counted as embeddings, one Whisper chunk per started 30 seconds")
    @Timeout(value = 480_000, unit = TimeUnit.MILLISECONDS)
    public void theClipIsCountedAsEmbeddingsPerWhisperChunk() throws IOException {
        int textOnly = promptTokens(ChatMessage.userMultimodal(ContentPart.text("Listen.")));
        int shortClip = promptTokens(
                ChatMessage.userMultimodal(ContentPart.text("Listen."), ContentPart.audioFile(Paths.get(audioPath))));
        int longClip = promptTokens(ChatMessage.userMultimodal(
                ContentPart.text("Listen."), ContentPart.inputAudio(toneWav(35, 16000), "wav")));

        assertTrue(shortClip > textOnly, "the clip must add embeddings: " + shortClip + " vs " + textOnly);
        assertEquals(
                2 * (shortClip - textOnly),
                longClip - textOnly,
                "a 35 s clip is two Whisper chunks, a 2.6 s clip one: " + textOnly + " / " + shortClip + " / "
                        + longClip);
    }

    /** llama.cpp's raw prompt object takes audio the way it takes an image, at the media marker. */
    @Test
    @DisplayName("a raw completion takes the clip in the multimodal prompt object")
    @Timeout(value = 240_000, unit = TimeUnit.MILLISECONDS)
    public void aRawCompletionTakesTheClipInTheMultimodalPromptObject() throws IOException {
        String marker = model.getModelMeta().getMediaMarker();
        assertFalse(marker.isEmpty(), "an audio model must report its media marker");
        int textTokens = MAPPER.readTree(
                        model.handleCompletionsOai("{\"prompt\":\"Hear: \",\"max_tokens\":1,\"temperature\":0}"))
                .path("usage")
                .path("prompt_tokens")
                .asInt();
        ObjectNode request = MAPPER.createObjectNode();
        request.putObject("prompt")
                .put("prompt_string", "Hear: " + marker + " Transcript:")
                .putArray("multimodal_data")
                .add(Base64.getEncoder().encodeToString(Files.readAllBytes(Paths.get(audioPath))));
        request.put("max_tokens", 4).put("temperature", 0);

        JsonNode answer = MAPPER.readTree(model.handleCompletionsOai(request.toString()));

        assertTrue(
                answer.path("usage").path("prompt_tokens").asInt() > textTokens,
                "the clip must add its embeddings to the raw prompt; got: " + answer);
    }

    @Test
    @DisplayName("an audio request followed by a text request on the same model both answer")
    @Timeout(value = 240_000, unit = TimeUnit.MILLISECONDS)
    public void audioThenTextOnTheSameModel() throws IOException {
        model.chatCompleteText(InferenceParameters.empty()
                .withMessages(Collections.singletonList(ChatMessage.userMultimodal(
                        ContentPart.text("What is this?"), ContentPart.audioFile(Paths.get(audioPath)))))
                .withNPredict(16)
                .withTemperature(0.0f));

        String text = model.chatCompleteText(InferenceParameters.empty()
                .withMessages(Collections.singletonList(new ChatMessage("user", "Reply with the single word: ok")))
                .withNPredict(8)
                .withTemperature(0.0f));

        assertFalse(text.trim().isEmpty(), "a text request after an audio one must still answer");
    }

    /** Two greedy answers to one clip, each computed whole, are the same: the audio path is deterministic. */
    @Test
    @DisplayName("the same clip gives the same greedy answer twice")
    @Timeout(value = 240_000, unit = TimeUnit.MILLISECONDS)
    public void theSameClipGivesTheSameAnswer() throws IOException {
        InferenceParameters params = InferenceParameters.empty()
                .withMessages(Collections.singletonList(ChatMessage.userMultimodal(
                        ContentPart.text("Transcribe the audio."), ContentPart.audioFile(Paths.get(audioPath)))))
                .withNPredict(24)
                .withTemperature(0.0f)
                .withCachePrompt(false);

        assertEquals(model.chatCompleteText(params), model.chatCompleteText(params));
    }

    @Test
    @DisplayName("the audio projector reads no image: such a request is the caller's fault")
    @Timeout(value = 120_000, unit = TimeUnit.MILLISECONDS)
    public void anImageIsRefusedByTheAudioProjector() {
        assertFalse(model.supportsVision(), "Ultravox's projector takes audio only");
        InferenceParameters params = InferenceParameters.empty()
                .withMessages(Collections.singletonList(ChatMessage.userMultimodal(
                        ContentPart.text("What is this?"), ContentPart.imageBytes(TINY_PNG, "image/png"))))
                .withNPredict(4);

        assertThrows(InvalidRequestException.class, () -> model.chatCompleteText(params));
        // and the model still answers afterwards
        assertFalse(model.chatCompleteText(InferenceParameters.empty()
                        .withMessages(Collections.singletonList(new ChatMessage("user", "Say ok")))
                        .withNPredict(4))
                .isEmpty());
    }

    private static int promptTokens(ChatMessage message) throws IOException {
        String json = model.chatComplete(InferenceParameters.empty()
                .withMessages(Collections.singletonList(message))
                .withNPredict(1)
                .withTemperature(0.0f)
                .withCachePrompt(false));
        return MAPPER.readTree(json).path("usage").path("prompt_tokens").asInt();
    }

    /** A mono 16-bit WAV of a 440 Hz tone, {@code seconds} long. */
    private static byte[] toneWav(int seconds, int rate) throws IOException {
        int samples = seconds * rate;
        ByteBuffer pcm = ByteBuffer.allocate(samples * 2).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < samples; i++) {
            pcm.putShort((short) (8000 * Math.sin(2 * Math.PI * 440 * i / rate)));
        }
        ByteBuffer header = ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN);
        header.put("RIFF".getBytes(StandardCharsets.US_ASCII)).putInt(36 + samples * 2);
        header.put("WAVEfmt ".getBytes(StandardCharsets.US_ASCII)).putInt(16).putShort((short) 1);
        header.putShort((short) 1)
                .putInt(rate)
                .putInt(rate * 2)
                .putShort((short) 2)
                .putShort((short) 16);
        header.put("data".getBytes(StandardCharsets.US_ASCII)).putInt(samples * 2);
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        out.write(header.array());
        out.write(pcm.array());
        return out.toByteArray();
    }
}
