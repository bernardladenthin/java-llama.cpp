// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT
package net.ladenthin.llama.ffm;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.greaterThanOrEqualTo;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.file.Path;
import java.util.Base64;
import java.util.Collections;
import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import net.ladenthin.llama.callback.VideoDecoder;
import net.ladenthin.llama.callback.VideoStream;
import net.ladenthin.llama.exception.InvalidRequestException;
import net.ladenthin.llama.exception.LlamaException;
import net.ladenthin.llama.parameters.ModelParameters;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

/**
 * {@link FfmLlamaModel#load(ModelParameters, net.ladenthin.llama.callback.LoadProgressCallback, VideoDecoder)}
 * on CI's vision pair: the videos of a request go to the Java decoder through {@code jllama_model_load_video}
 * and its upcall stubs, the way {@code VideoDecoderIntegrationTest} holds it for {@code LlamaModel} over JNI.
 * The decoder reads a "video" of its own making (eight bytes: a magic, the frame count, the size, the rate) and
 * paints every frame in one colour; no ffmpeg is needed.
 */
class FfmVideoDecoderIntegrationTest {

    /** CI's vision model, as the core's {@code TestConstants} names it. */
    static final String VISION_MODEL_PROPERTY = "net.ladenthin.llama.ffm.vision.model";

    static final String DEFAULT_VISION_MODEL = "models/SmolVLM-500M-Instruct-Q8_0.gguf";

    static final String VISION_MMPROJ_PROPERTY = "net.ladenthin.llama.ffm.vision.mmproj";

    static final String DEFAULT_VISION_MMPROJ = "models/mmproj-SmolVLM-500M-Instruct-Q8_0.gguf";

    private static final ObjectMapper JSON = new ObjectMapper();
    private static final int WIDTH = 64;
    private static final int HEIGHT = 48;

    private static final AtomicInteger OPENED = new AtomicInteger();
    private static final AtomicInteger READ = new AtomicInteger();
    private static final AtomicInteger CLOSED = new AtomicInteger();
    private static final Set<String> THREADS = Collections.synchronizedSet(new HashSet<>());

    private static Path model;
    private static Path mmproj;
    private static FfmLlamaModel vision;

    /** Reads {@link #fakeVideo(int, int)}: every frame one colour, as many frames as the file says. */
    static final VideoDecoder DECODER = data -> {
        THREADS.add(Thread.currentThread().getName());
        if (data.length != 8 || data[0] != 'J' || data[1] != 'L' || data[2] != 'V' || data[3] != '1') {
            return null;
        }
        OPENED.incrementAndGet();
        int frames = data[4];
        float rate = data[7];
        return new VideoStream() {
            private int next;

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
            public boolean readFrame(ByteBuffer rgb, int width, int height) {
                if (next >= frames) {
                    return false;
                }
                if (rgb != null) {
                    byte colour = (byte) (40 + 20 * next);
                    while (rgb.hasRemaining()) {
                        rgb.put(colour);
                    }
                    READ.incrementAndGet();
                }
                next++;
                return true;
            }

            @Override
            public void close() {
                CLOSED.incrementAndGet();
            }
        };
    };

    private static byte[] fakeVideo(int frames, int rate) {
        return new byte[] {'J', 'L', 'V', '1', (byte) frames, WIDTH, HEIGHT, (byte) rate};
    }

    private static ModelParameters parameters(Path text) {
        return new ModelParameters()
                .setModel(text.toString())
                .setMmproj(mmproj.toString())
                .setCtxSize(4096)
                .setGpuLayers(0)
                .setThreads(2)
                .setDevices("none")
                .setMmprojOffload(false)
                .setFit(false);
    }

    @BeforeAll
    static void load() {
        model = FfmTestSupport.assumeFile(VISION_MODEL_PROPERTY, DEFAULT_VISION_MODEL);
        mmproj = FfmTestSupport.assumeFile(VISION_MMPROJ_PROPERTY, DEFAULT_VISION_MMPROJ);
        vision = FfmLlamaModel.load(parameters(model), null, DECODER);
    }

    @AfterAll
    static void free() {
        if (vision != null) {
            vision.close();
        }
    }

    /** The prompt tokens of a chat request with the question and, unless null, the video. */
    private static int promptTokens(byte[] video) throws IOException {
        ObjectNode request = JSON.createObjectNode();
        ArrayNode content = request.putArray("messages").addObject().put("role", "user").putArray("content");
        content.addObject().put("type", "text").put("text", "What happens in this video?");
        if (video != null) {
            content.addObject()
                    .put("type", "video_url")
                    .putObject("video_url")
                    .put("url", "data:video/mp4;base64," + Base64.getEncoder().encodeToString(video));
        }
        request.put("max_tokens", 2).put("temperature", 0).put("cache_prompt", false);
        return JSON.readTree(vision.chatCompletions(request.toString()))
                .path("usage")
                .path("prompt_tokens")
                .asInt(-1);
    }

    @Test
    @Timeout(value = 10, unit = TimeUnit.MINUTES)
    void theVideoOfARequestIsReadThroughTheJavaDecoder() throws IOException {
        assertTrue(JSON.readTree(vision.meta()).path("modalities").path("video").asBoolean(), vision.meta());
        int opened = OPENED.get();
        int read = READ.get();
        int closed = CLOSED.get();
        int textOnly = promptTokens(null);
        // eight frames at four per second, the rate the model samples at: every frame is read
        int withVideo = promptTokens(fakeVideo(8, 4));
        assertEquals(opened + 1, OPENED.get());
        assertEquals(read + 8, READ.get());
        assertEquals(closed + 1, CLOSED.get());
        assertThat(withVideo, greaterThanOrEqualTo(textOnly + 8));
        // the decoder runs on a thread of the library, not on this one
        assertFalse(THREADS.contains(Thread.currentThread().getName()), THREADS.toString());
    }

    @Test
    @Timeout(value = 10, unit = TimeUnit.MINUTES)
    void bytesTheDecoderDoesNotOpenAreTheCallersFaultAndTheModelGoesOn() throws IOException {
        int opened = OPENED.get();
        assertThrows(InvalidRequestException.class, () -> promptTokens(new byte[] {1, 2, 3}));
        assertEquals(opened, OPENED.get());
        assertThat(promptTokens(null), greaterThanOrEqualTo(1));
    }

    @Test
    void aFailedLoadReleasesTheDecoderWithoutHarm() {
        Path missing = model.resolveSibling("no-such-vision-model.gguf");
        assertThrows(LlamaException.class, () -> FfmLlamaModel.load(parameters(missing), null, DECODER));
        assertThrows(NullPointerException.class, () -> FfmLlamaModel.load(parameters(model), null, null));
    }
}
