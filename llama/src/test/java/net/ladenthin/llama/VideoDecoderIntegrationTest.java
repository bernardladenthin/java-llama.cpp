// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.greaterThan;
import static org.hamcrest.Matchers.greaterThanOrEqualTo;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.lessThan;
import static org.junit.jupiter.api.Assertions.assertThrows;

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
import java.net.ServerSocket;
import java.net.URL;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Base64;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import net.ladenthin.llama.callback.VideoDecoder;
import net.ladenthin.llama.callback.VideoStream;
import net.ladenthin.llama.exception.InvalidRequestException;
import net.ladenthin.llama.parameters.InferenceParameters;
import net.ladenthin.llama.parameters.ModelParameters;
import net.ladenthin.llama.server.NativeServer;
import net.ladenthin.llama.value.ChatMessage;
import net.ladenthin.llama.value.ContentPart;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

/**
 * A model loaded with a {@link VideoDecoder}: the videos of its requests are read through the
 * decoder, not through the {@code ffmpeg} program, so this test needs none.
 *
 * <p>The decoder here is a stand-in that reads a video "file" of its own making (eight bytes: a
 * magic, the frame count, the size and the frame rate) and paints every frame in one colour. What
 * is asserted is the conversation between llama.cpp and the Java side: which calls arrive, in which
 * order, with which buffer, on which thread, and that the frames it hands over are the prompt
 * tokens the model evaluates.
 */
@ClaudeGenerated(
        purpose = "End-to-end test of LlamaModel(parameters, progress, VideoDecoder): a Java decoder is opened "
                + "with the request's bytes, asked for the frames llama.cpp samples (skipping by a null buffer), "
                + "closed once, and its frames cost prompt tokens; a refusing or throwing decoder fails the request "
                + "as invalid and the model goes on; an attached native server calls the decoder from its own "
                + "threads.")
public class VideoDecoderIntegrationTest {

    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final int WIDTH = 64;
    private static final int HEIGHT = 48;

    /** What the stand-in decoder was asked, across every stream it opened. */
    private static final class Recorder {
        final AtomicInteger opened = new AtomicInteger();
        final AtomicInteger started = new AtomicInteger();
        final AtomicInteger framesRead = new AtomicInteger();
        final AtomicInteger framesSkipped = new AtomicInteger();
        final AtomicInteger closed = new AtomicInteger();
        final AtomicInteger badBuffers = new AtomicInteger();
        final List<String> threads = Collections.synchronizedList(new ArrayList<String>());
        volatile boolean resamples;
        volatile boolean failToStart;
        volatile boolean throwOnOpen;
        volatile float rateAsked = -1.0f;

        void reset() {
            opened.set(0);
            started.set(0);
            framesRead.set(0);
            framesSkipped.set(0);
            closed.set(0);
            badBuffers.set(0);
            threads.clear();
            resamples = false;
            failToStart = false;
            throwOnOpen = false;
            rateAsked = -1.0f;
        }
    }

    /** Reads {@link #fakeVideo(int, int)}: every frame one colour, as many frames as the file says. */
    private static final class FakeDecoder implements VideoDecoder {
        private final Recorder recorder;

        FakeDecoder(Recorder recorder) {
            this.recorder = recorder;
        }

        @Override
        public VideoStream open(byte[] data) throws IOException {
            recorder.threads.add(Thread.currentThread().getName());
            if (recorder.throwOnOpen) {
                throw new IOException("this decoder cannot open anything right now");
            }
            if (data.length != 8 || data[0] != 'J' || data[1] != 'L' || data[2] != 'V' || data[3] != '1') {
                return null;
            }
            recorder.opened.incrementAndGet();
            final int frames = data[4];
            final float rate = data[7];
            return new VideoStream() {
                private int next;
                private int width = -1;
                private int height = -1;

                @Override
                public int getWidth() {
                    return data[5];
                }

                @Override
                public int getHeight() {
                    return data[6];
                }

                @Override
                public float getFramesPerSecond() {
                    return rate;
                }

                @Override
                public int getFrameCount() {
                    return frames;
                }

                @Override
                public float getDurationSeconds() {
                    return frames / rate;
                }

                @Override
                public boolean start(int width, int height, float framesPerSecond) throws IOException {
                    recorder.started.incrementAndGet();
                    recorder.rateAsked = framesPerSecond;
                    this.width = width;
                    this.height = height;
                    if (recorder.failToStart) {
                        throw new IOException("this stream cannot start");
                    }
                    return recorder.resamples;
                }

                @Override
                public boolean readFrame(ByteBuffer rgb, int width, int height) {
                    if (next >= frames) {
                        return false;
                    }
                    if (rgb == null) {
                        recorder.framesSkipped.incrementAndGet();
                    } else {
                        if (!rgb.isDirect()
                                || rgb.remaining() != width * height * 3
                                || width != this.width
                                || height != this.height) {
                            recorder.badBuffers.incrementAndGet();
                        }
                        byte colour = (byte) (40 + 20 * next);
                        while (rgb.hasRemaining()) {
                            rgb.put(colour);
                        }
                        recorder.framesRead.incrementAndGet();
                    }
                    next++;
                    return true;
                }

                @Override
                public void close() {
                    recorder.closed.incrementAndGet();
                }
            };
        }
    }

    private static final Recorder RECORDER = new Recorder();
    private static String modelPath;
    private static String mmprojPath;
    private static LlamaModel model;

    /** A video only {@link FakeDecoder} reads: {@code frames} frames at {@code rate} per second. */
    private static byte[] fakeVideo(int frames, int rate) {
        return new byte[] {'J', 'L', 'V', '1', (byte) frames, WIDTH, HEIGHT, (byte) rate};
    }

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

    @BeforeAll
    public static void setup() {
        modelPath = TestConstants.resolveModelProperty(
                TestConstants.PROP_VISION_MODEL_PATH, TestConstants.DEFAULT_VISION_MODEL_PATH);
        mmprojPath = TestConstants.resolveModelProperty(
                TestConstants.PROP_VISION_MMPROJ_PATH, TestConstants.DEFAULT_VISION_MMPROJ_PATH);
        Assumptions.assumeTrue(new File(modelPath).exists(), "Vision model file missing: " + modelPath);
        Assumptions.assumeTrue(new File(mmprojPath).exists(), "Vision mmproj file missing: " + mmprojPath);
        model = new LlamaModel(parameters(), null, new FakeDecoder(RECORDER));
    }

    @AfterAll
    public static void tearDown() {
        if (model != null) {
            model.close();
        }
    }

    @BeforeEach
    public void resetRecorder() {
        RECORDER.reset();
    }

    private static int promptTokens(LlamaModel target, ContentPart... parts) throws IOException {
        String answer = target.chatComplete(new InferenceParameters("")
                .withMessages(Collections.singletonList(ChatMessage.userMultimodal(parts)))
                .withNPredict(4)
                .withTemperature(0.0f)
                .withCachePrompt(false));
        return MAPPER.readTree(answer).path("usage").path("prompt_tokens").asInt(-1);
    }

    @Test
    public void aDecoderIsRequired() {
        assertThrows(NullPointerException.class, () -> new LlamaModel(parameters(), null, null));
    }

    @Timeout(value = 600_000, unit = TimeUnit.MILLISECONDS)
    @Test
    public void theModelReadsAVideoThroughTheDecoder() throws IOException {
        assertThat(model.supportsVideo(), is(true));
        ContentPart question = ContentPart.text("What happens in this video?");
        int textOnly = promptTokens(model, question);
        assertThat(RECORDER.opened.get(), is(0));

        // eight frames at four per second, the rate the model samples at: every frame is read
        int withVideo = promptTokens(model, question, ContentPart.videoBytes(fakeVideo(8, 4), "video/mp4"));
        assertThat(RECORDER.opened.get(), is(1));
        assertThat(RECORDER.started.get(), is(1));
        assertThat(RECORDER.framesRead.get(), is(8));
        assertThat(RECORDER.closed.get(), is(1));
        assertThat(
                "a frame buffer that is not a direct buffer of width x height x 3", RECORDER.badBuffers.get(), is(0));
        assertThat(withVideo, greaterThanOrEqualTo(textOnly + 8));
        // The decoder is not called on the thread that made the request: llama.cpp reads the media of
        // a request on its own thread, which is attached to the JVM for that. An implementation must
        // not rely on being on the caller's thread (no thread-locals, no caller's class loader).
        assertThat(RECORDER.threads, is(Collections.singletonList("jllama-video")));

        // twice the frames cost more
        int longer = promptTokens(model, question, ContentPart.videoBytes(fakeVideo(16, 4), "video/mp4"));
        assertThat(RECORDER.framesRead.get(), is(8 + 16));
        assertThat(longer, greaterThanOrEqualTo(withVideo + 8));
        assertThat(RECORDER.closed.get(), is(2));
    }

    @Timeout(value = 600_000, unit = TimeUnit.MILLISECONDS)
    @Test
    public void theFrameRateDecidesWhichFramesAreReadAndWhichSkipped() throws IOException {
        ContentPart question = ContentPart.text("What happens in this video?");
        try (LlamaModel half = new LlamaModel(parameters().setVideoFps(2.0f), null, new FakeDecoder(RECORDER))) {
            // an 8 frame video of 4 per second at 2 per second: every second frame, the rest skipped unread
            int sampled = promptTokens(half, question, ContentPart.videoBytes(fakeVideo(8, 4), "video/mp4"));
            assertThat(RECORDER.framesRead.get(), is(4));
            assertThat(RECORDER.framesSkipped.get(), greaterThanOrEqualTo(3));

            // a stream that resamples itself is told the rate and asked for every frame it delivers
            RECORDER.reset();
            RECORDER.resamples = true;
            int all = promptTokens(half, question, ContentPart.videoBytes(fakeVideo(8, 4), "video/mp4"));
            assertThat(RECORDER.rateAsked, is(2.0f));
            assertThat(RECORDER.framesRead.get(), is(8));
            assertThat(RECORDER.framesSkipped.get(), is(0));
            assertThat(sampled, lessThan(all));
        }
    }

    @Timeout(value = 600_000, unit = TimeUnit.MILLISECONDS)
    @Test
    public void aVideoTheDecoderDoesNotOpenIsTheCallersFault() throws IOException {
        ContentPart question = ContentPart.text("What happens in this video?");
        // bytes the decoder answers with null
        assertThrows(
                InvalidRequestException.class,
                () -> promptTokens(
                        model,
                        question,
                        ContentPart.videoBytes("not a video".getBytes(StandardCharsets.UTF_8), "video/mp4")));
        assertThat(RECORDER.opened.get(), is(0));

        // an exception out of open() stays on the Java side of the call
        RECORDER.throwOnOpen = true;
        assertThrows(
                InvalidRequestException.class,
                () -> promptTokens(model, question, ContentPart.videoBytes(fakeVideo(8, 4), "video/mp4")));
        RECORDER.throwOnOpen = false;

        // a stream that fails to start is closed, and the request refused
        RECORDER.failToStart = true;
        assertThrows(
                InvalidRequestException.class,
                () -> promptTokens(model, question, ContentPart.videoBytes(fakeVideo(8, 4), "video/mp4")));
        assertThat(RECORDER.opened.get(), is(1));
        assertThat(RECORDER.closed.get(), is(1));
        RECORDER.failToStart = false;

        // and the model goes on
        assertThat(
                promptTokens(model, question, ContentPart.videoBytes(fakeVideo(4, 4), "video/mp4")),
                greaterThan(promptTokens(model, question)));
    }

    @Timeout(value = 600_000, unit = TimeUnit.MILLISECONDS)
    @Test
    public void aServerAttachedToTheModelDecodesOnItsOwnThreads() throws Exception {
        int port;
        try (ServerSocket socket = new ServerSocket(0)) {
            port = socket.getLocalPort();
        }
        // server before model on the way out: try-with-resources closes it first
        try (NativeServer server =
                new NativeServer(model, "--host", "127.0.0.1", "--port", Integer.toString(port)).start()) {
            String base = "http://127.0.0.1:" + port;
            awaitHealthy(base);
            JsonNode props = MAPPER.readTree(http("GET", base + "/props", null, 200));
            assertThat(props.path("modalities").path("video").asBoolean(), is(true));

            ObjectNode root = MAPPER.createObjectNode();
            ArrayNode content =
                    root.putArray("messages").addObject().put("role", "user").putArray("content");
            content.addObject().put("type", "text").put("text", "What happens in this video?");
            content.addObject()
                    .put("type", "input_video")
                    .putObject("input_video")
                    .put("data", Base64.getEncoder().encodeToString(fakeVideo(8, 4)));
            root.put("max_tokens", 4);
            root.put("temperature", 0);
            JsonNode answer = MAPPER.readTree(http("POST", base + "/v1/chat/completions", root.toString(), 200));
            assertThat(answer.path("usage").path("prompt_tokens").asInt(-1), greaterThanOrEqualTo(8));
            assertThat(RECORDER.framesRead.get(), is(8));
            assertThat(RECORDER.closed.get(), is(1));
            // the library thread again, not one of the JVM
            assertThat(RECORDER.threads, is(Collections.singletonList("jllama-video")));
        }
    }

    private static void awaitHealthy(String base) throws Exception {
        long deadline = System.currentTimeMillis() + 60_000L;
        IOException last = null;
        while (System.currentTimeMillis() < deadline) {
            try {
                http("GET", base + "/health", null, 200);
                return;
            } catch (IOException | AssertionError e) {
                last = e instanceof IOException ? (IOException) e : new IOException(e.getMessage());
                Thread.sleep(100);
            }
        }
        throw new IOException("the attached server did not become healthy", last);
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
