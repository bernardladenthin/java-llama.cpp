// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.greaterThan;
import static org.hamcrest.Matchers.greaterThanOrEqualTo;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.lessThan;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.fail;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
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
import net.ladenthin.llama.server.OpenAiCompatServer;
import net.ladenthin.llama.server.OpenAiServerConfig;
import net.ladenthin.llama.value.ChatMessage;
import net.ladenthin.llama.value.ChatResponse;
import net.ladenthin.llama.value.ContentPart;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.api.io.TempDir;

/**
 * Video input end to end: a video in a chat message reaches the model as its frames.
 *
 * <p>llama.cpp decodes a video with the {@code ffmpeg} and {@code ffprobe} programs of the machine
 * and gives the sampled frames to the vision projector as images. Nothing here judges what the
 * model says about the clip; what is asserted is what the pipeline did, read from the prompt
 * tokens of the answer: a video costs as many tokens as its frames, more frames cost more, and the
 * typed, streamed and HTTP paths count the same.
 *
 * <p>The cases that decode need the two programs. Without them they are skipped, unless the
 * environment variable {@code JLLAMA_REQUIRE_FFMPEG} is set (the CI job that installs ffmpeg sets
 * it), where a skip would hide that nothing ran. The other cases run everywhere.
 */
@ClaudeGenerated(
        purpose = "End-to-end test of video input through ContentPart.videoFile / videoBytes / videoUrl: the "
                + "frames ffmpeg decodes reach the vision projector (prompt tokens grow with the frame count), on the "
                + "blocking, typed, streamed and OpenAI-compatible HTTP paths; a machine without ffmpeg refuses the "
                + "request as invalid and the model keeps answering.")
public class VideoInputIntegrationTest {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    /** Frames of the committed clip: two seconds at four frames per second. */
    private static final int CLIP_FRAMES = 8;

    private static String modelPath;
    private static String mmprojPath;
    private static Path clip;
    private static LlamaModel model;
    private static boolean ffmpeg;

    @BeforeAll
    public static void setup() {
        modelPath = TestConstants.resolveModelProperty(
                TestConstants.PROP_VISION_MODEL_PATH, TestConstants.DEFAULT_VISION_MODEL_PATH);
        mmprojPath = TestConstants.resolveModelProperty(
                TestConstants.PROP_VISION_MMPROJ_PATH, TestConstants.DEFAULT_VISION_MMPROJ_PATH);
        clip = Paths.get(TestConstants.DEFAULT_VIDEO_INPUT_PATH);
        Assumptions.assumeTrue(new File(modelPath).exists(), "Vision model file missing: " + modelPath);
        Assumptions.assumeTrue(new File(mmprojPath).exists(), "Vision mmproj file missing: " + mmprojPath);
        Assumptions.assumeTrue(Files.exists(clip), "Video clip missing: " + clip);
        ffmpeg = runs("ffmpeg") && runs("ffprobe");
        model = new LlamaModel(parameters());
    }

    @AfterAll
    public static void tearDown() {
        if (model != null) {
            model.close();
        }
    }

    /** The vision model with room for the frames of the clip. */
    private static ModelParameters parameters() {
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
        return parameters;
    }

    private static boolean runs(String program) {
        try {
            Process process = new ProcessBuilder(program, "-version")
                    .redirectErrorStream(true)
                    .start();
            try (InputStream out = process.getInputStream()) {
                byte[] sink = new byte[4096];
                while (out.read(sink) >= 0) {
                    // drained so the program cannot block on a full pipe
                }
            }
            return process.waitFor(30, TimeUnit.SECONDS) && process.exitValue() == 0;
        } catch (IOException e) {
            return false;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            return false;
        }
    }

    private static void requireFfmpeg() {
        if (!ffmpeg && System.getenv("JLLAMA_REQUIRE_FFMPEG") != null) {
            fail("JLLAMA_REQUIRE_FFMPEG is set, and ffmpeg / ffprobe are not on PATH");
        }
        Assumptions.assumeTrue(ffmpeg, "ffmpeg / ffprobe are not on PATH");
    }

    /** The prompt tokens a chat request with these parts costs on {@code target}. */
    private static int promptTokens(LlamaModel target, ContentPart... parts) throws IOException {
        String answer = target.chatComplete(new InferenceParameters("")
                .withMessages(Collections.singletonList(ChatMessage.userMultimodal(parts)))
                .withNPredict(4)
                .withTemperature(0.0f)
                .withCachePrompt(false));
        return MAPPER.readTree(answer).path("usage").path("prompt_tokens").asInt(-1);
    }

    @Test
    public void aModelWithAVisionProjectorReportsVideoInput() {
        assertThat(model.supportsVision(), is(true));
        assertThat(model.supportsVideo(), is(true));
        assertThat(model.getModelMeta().supportsVideo(), is(true));
    }

    @Timeout(value = 600_000, unit = TimeUnit.MILLISECONDS)
    @Test
    public void aVideoCostsThePromptTokensOfItsFrames() throws IOException {
        requireFfmpeg();
        ContentPart question = ContentPart.text("What happens in this video?");
        int textOnly = promptTokens(model, question);
        assertThat(textOnly, greaterThan(0));
        int withVideo = promptTokens(model, question, ContentPart.videoFile(clip));
        // every frame is at least one embedding; a model that never saw the video would count the text alone
        assertThat(withVideo, greaterThanOrEqualTo(textOnly + CLIP_FRAMES));

        // the same bytes under an explicit type, and as a data URI the caller formed
        byte[] bytes = Files.readAllBytes(clip);
        assertThat(promptTokens(model, question, ContentPart.videoBytes(bytes, "video/mp4")), is(withVideo));
        assertThat(
                promptTokens(
                        model,
                        question,
                        ContentPart.videoUrl(
                                "data:video/mp4;base64," + Base64.getEncoder().encodeToString(bytes))),
                is(withVideo));

        // the video twice: twice the frames
        int twice = promptTokens(model, question, ContentPart.videoFile(clip), ContentPart.videoFile(clip));
        assertThat(twice, greaterThanOrEqualTo(withVideo + CLIP_FRAMES));
    }

    @Timeout(value = 600_000, unit = TimeUnit.MILLISECONDS)
    @Test
    public void theFrameRateDecidesHowManyFramesTheModelSees() throws IOException {
        requireFfmpeg();
        ContentPart question = ContentPart.text("What happens in this video?");
        int atDefault = promptTokens(model, question, ContentPart.videoFile(clip));
        // one frame per second instead of four: a quarter of the frames
        try (LlamaModel slower = new LlamaModel(parameters().setVideoFps(1.0f))) {
            int atOne = promptTokens(slower, question, ContentPart.videoFile(clip));
            assertThat(atOne, greaterThan(promptTokens(slower, question)));
            assertThat(atOne, lessThan(atDefault));
        }
    }

    @Timeout(value = 600_000, unit = TimeUnit.MILLISECONDS)
    @Test
    public void theTypedAndTheStreamedPathTakeAVideo() throws IOException {
        requireFfmpeg();
        ChatRequest request = ChatRequest.empty()
                .appendMessage(ChatMessage.userMultimodal(
                        ContentPart.text("Describe this video in one short sentence."), ContentPart.videoFile(clip)))
                .withInferenceCustomizer(params -> params.withNPredict(8).withTemperature(0.0f));
        ChatResponse response = model.chat(request);
        assertThat(response.getUsage().getPromptTokens(), greaterThanOrEqualTo((long) CLIP_FRAMES));

        List<String> chunks = new ArrayList<String>();
        model.streamChatCompletion(
                InferenceParameters.empty()
                        .withMessages(Collections.singletonList(ChatMessage.userMultimodal(
                                ContentPart.text("Describe this video in one short sentence."),
                                ContentPart.videoFile(clip))))
                        .withNPredict(8)
                        .withTemperature(0.0f),
                chunks::add);
        assertThat(chunks.isEmpty(), is(false));
    }

    @Timeout(value = 600_000, unit = TimeUnit.MILLISECONDS)
    @Test
    public void theOpenAiCompatibleServerForwardsAVideoPartAndAdvertisesVideo() throws IOException {
        OpenAiServerConfig config = OpenAiServerConfig.builder()
                .port(0)
                .supportsVision(model.supportsVision())
                .supportsVideo(model.supportsVideo())
                .build();
        try (OpenAiCompatServer server = new OpenAiCompatServer(model, config).start()) {
            String base = "http://127.0.0.1:" + server.getPort();
            JsonNode props = MAPPER.readTree(http("GET", base + "/props", null, 200));
            assertThat(props.path("modalities").path("video").asBoolean(), is(true));

            // a part that is no media at all is refused whether or not ffmpeg is here
            String garbage = Base64.getEncoder().encodeToString("not a video".getBytes(StandardCharsets.UTF_8));
            http("POST", base + "/v1/chat/completions", chatBody("input_video", "data", garbage), 400);

            if (!ffmpeg) {
                return; // decoding is covered where the programs are
            }
            String encoded = Base64.getEncoder().encodeToString(Files.readAllBytes(clip));
            JsonNode own = MAPPER.readTree(
                    http("POST", base + "/v1/chat/completions", chatBody("input_video", "data", encoded), 200));
            int ownTokens = own.path("usage").path("prompt_tokens").asInt(-1);
            assertThat(ownTokens, greaterThanOrEqualTo(CLIP_FRAMES));
            JsonNode alias = MAPPER.readTree(http(
                    "POST",
                    base + "/v1/chat/completions",
                    chatBody("video_url", "url", "data:video/mp4;base64," + encoded),
                    200));
            assertThat(alias.path("usage").path("prompt_tokens").asInt(-1), is(ownTokens));
        }
    }

    @Timeout(value = 600_000, unit = TimeUnit.MILLISECONDS)
    @Test
    public void aMachineWithoutFfmpegRefusesAVideoAndKeepsAnswering(@TempDir Path empty) throws IOException {
        // A directory without the two programs is a machine without ffmpeg (--video-ffmpeg-dir). The
        // request must fail as the caller's, and the process and the model must go on.
        try (LlamaModel without = new LlamaModel(parameters().setVideoFfmpegDir(empty.toString()))) {
            ContentPart question = ContentPart.text("What happens in this video?");
            InvalidRequestException refused = assertThrows(
                    InvalidRequestException.class, () -> promptTokens(without, question, ContentPart.videoFile(clip)));
            assertThat(String.valueOf(refused.getMessage()), containsString("Failed to load"));
            assertThat(promptTokens(without, question), greaterThan(0));
        }
    }

    /** A chat request whose one message carries a text part and a video part of {@code type}. */
    private static String chatBody(String type, String key, String value) {
        ObjectNode root = MAPPER.createObjectNode();
        ArrayNode content =
                root.putArray("messages").addObject().put("role", "user").putArray("content");
        content.addObject().put("type", "text").put("text", "What happens in this video?");
        content.addObject().put("type", type).putObject(type).put(key, value);
        root.put("max_tokens", 4);
        root.put("temperature", 0);
        root.put("cache_prompt", false);
        return root.toString();
    }

    private static String http(String method, String url, String body, int expectedStatus) throws IOException {
        HttpURLConnection connection = (HttpURLConnection) new URL(url).openConnection();
        connection.setRequestMethod(method);
        connection.setConnectTimeout(10_000);
        connection.setReadTimeout(600_000);
        if (body != null) {
            connection.setDoOutput(true);
            connection.setRequestProperty("Content-Type", "application/json");
            try (OutputStream out = connection.getOutputStream()) {
                out.write(body.getBytes(StandardCharsets.UTF_8));
            }
        }
        int status = connection.getResponseCode();
        InputStream stream = status >= 400 ? connection.getErrorStream() : connection.getInputStream();
        ByteArrayOutputStream text = new ByteArrayOutputStream();
        if (stream != null) {
            try (InputStream in = stream) {
                byte[] buffer = new byte[8192];
                for (int n = in.read(buffer); n >= 0; n = in.read(buffer)) {
                    text.write(buffer, 0, n);
                }
            }
        }
        String answer = new String(text.toByteArray(), StandardCharsets.UTF_8);
        assertThat(method + " " + url + " answered " + answer, status, is(expectedStatus));
        return answer;
    }
}
