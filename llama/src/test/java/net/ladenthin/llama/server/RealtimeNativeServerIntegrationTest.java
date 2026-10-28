// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.greaterThan;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.not;
import static org.junit.jupiter.api.Assertions.fail;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import java.io.File;
import java.io.IOException;
import java.net.ServerSocket;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.WebSocket;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Base64;
import java.util.List;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.CompletionStage;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import net.ladenthin.llama.ClaudeGenerated;
import net.ladenthin.llama.TestConstants;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

/**
 * The WebSocket {@code /v1/realtime} of the embedded C++ server ({@link NativeServer}) with an
 * audio-in/audio-out conversation model, LFM3-Audio (llama.cpp b11946, #31269; tool calls b11960,
 * #31287): the session and its audio formats, a text and a spoken answer, audio input, and a tool
 * call. The route is upstream's, served by {@code server-realtime.cpp}, which this build compiles in
 * since b11946; this test pins that the embedded server answers it end to end. A Java API over the
 * same session is open (TODO.md, "Realtime conversations").
 *
 * <p>Not in CI's model set (1.4&nbsp;GB): self-skips unless {@link TestConstants#PROP_REALTIME_MODEL}
 * and {@link TestConstants#PROP_REALTIME_MMPROJ} name the pair, e.g. {@code LFM3-Audio-2B-Q4_0.gguf}
 * and {@code mmproj-LFM3-Audio-2B-Q8_0.gguf} of {@code LiquidAI/LFM3-Audio-2B-GGUF}. The client is
 * the JDK's {@link java.net.http.WebSocket} (the tests compile for Java 21, the library for 8).</p>
 */
@ClaudeGenerated(
        purpose = "Prove the embedded server's /v1/realtime end to end with a real audio conversation "
                + "model: session formats, text and audio responses, audio input, tool calls.")
public class RealtimeNativeServerIntegrationTest extends OpenAiServerTestSupport {

    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final long EVENT_TIMEOUT_SECONDS = 180;

    private static NativeServer server;
    private static int port;

    @BeforeAll
    public static void setup() throws Exception {
        String model = TestConstants.resolveModelProperty(TestConstants.PROP_REALTIME_MODEL);
        String mmproj = TestConstants.resolveModelProperty(TestConstants.PROP_REALTIME_MMPROJ);
        Assumptions.assumeTrue(
                model != null && new File(model).exists(),
                "realtime model not set or missing (-D" + TestConstants.PROP_REALTIME_MODEL + "=...)");
        Assumptions.assumeTrue(
                mmproj != null && new File(mmproj).exists(),
                "its mmproj not set or missing (-D" + TestConstants.PROP_REALTIME_MMPROJ + "=...)");
        int gpuLayers = Integer.getInteger(TestConstants.PROP_TEST_NGL, TestConstants.DEFAULT_TEST_NGL);
        try (ServerSocket socket = new ServerSocket(0)) {
            port = socket.getLocalPort();
        }
        server = new NativeServer(
                        "-m",
                        model,
                        "--mmproj",
                        mmproj,
                        "--host",
                        "127.0.0.1",
                        "--port",
                        Integer.toString(port),
                        "-c",
                        "4096",
                        "-ngl",
                        Integer.toString(gpuLayers))
                .start();
        long deadline = System.currentTimeMillis() + 240_000L;
        while (true) {
            try {
                if (new RealtimeNativeServerIntegrationTest().get(port, "/health", "").code == 200) {
                    break;
                }
            } catch (IOException e) {
                // not listening yet
            }
            if (System.currentTimeMillis() > deadline) {
                fail("the server did not become healthy");
            }
            Thread.sleep(250L);
        }
    }

    @AfterAll
    public static void tearDown() {
        if (server != null) {
            server.close();
        }
    }

    @Test
    @Timeout(value = 5, unit = TimeUnit.MINUTES)
    public void theSessionReportsTheInputAndOutputAudioFormats() throws Exception {
        try (Session session = Session.open(port, "")) {
            JsonNode created = session.await("session.created");
            JsonNode audio = created.path("session").path("audio");
            assertThat(created.toString(), created.path("session").path("type").asText(), is("realtime"));
            assertThat(
                    audio.toString(),
                    audio.path("input").path("format").path("rate").asInt(),
                    greaterThan(0));
            assertThat(
                    audio.toString(),
                    audio.path("output").path("format").path("rate").asInt(),
                    greaterThan(0));
        }
    }

    @Test
    @Timeout(value = 5, unit = TimeUnit.MINUTES)
    public void aTextMessageGetsATextResponse() throws Exception {
        try (Session session = Session.open(port, "")) {
            session.await("session.created");
            session.userText("What is two plus two? Answer with one word.");
            session.send(responseCreate("text", 32));
            Response response = session.collectResponse();
            assertThat(
                    response.done.toString(),
                    response.done.path("response").path("status").asText(),
                    is("completed"));
            assertThat("text of " + response.done, response.text.toString().trim(), not(is("")));
        }
    }

    @Test
    @Timeout(value = 5, unit = TimeUnit.MINUTES)
    public void anAudioResponseStreamsPcm16() throws Exception {
        try (Session session = Session.open(port, "")) {
            session.await("session.created");
            session.userText("Say hello.");
            session.send(responseCreate("audio", 48));
            Response response = session.collectResponse();
            assertThat(
                    response.done.toString(),
                    response.done.path("response").path("status").asText(),
                    is("completed"));
            assertThat("audio bytes of " + response.done, response.audioBytes, greaterThan(0));
            assertThat("PCM16 comes in whole samples", response.audioBytes % 2, is(0));
        }
    }

    @Test
    @Timeout(value = 5, unit = TimeUnit.MINUTES)
    public void spokenInputIsAnswered() throws Exception {
        try (Session session = Session.open(port, "")) {
            int rate = session.await("session.created")
                    .path("session")
                    .path("audio")
                    .path("input")
                    .path("format")
                    .path("rate")
                    .asInt();
            session.sendClip(rate);
            session.send(MAPPER.createObjectNode().put("type", "input_audio_buffer.commit"));
            session.await("input_audio_buffer.committed");
            session.send(responseCreate("text", 48));
            Response response = session.collectResponse();
            assertThat(
                    response.done.toString(),
                    response.done.path("response").path("status").asText(),
                    is("completed"));
            assertThat("text of " + response.done, response.text.toString().trim(), not(is("")));
        }
    }

    @Test
    @Timeout(value = 5, unit = TimeUnit.MINUTES)
    public void aTranscriptionSessionTranscribesTheCommittedAudio() throws Exception {
        try (Session session = Session.open(port, "")) {
            int rate = session.await("session.created")
                    .path("session")
                    .path("audio")
                    .path("input")
                    .path("format")
                    .path("rate")
                    .asInt();
            ObjectNode update = MAPPER.createObjectNode().put("type", "session.update");
            update.putObject("session").put("type", "transcription");
            session.send(update);
            JsonNode updated = session.await("session.updated");
            assertThat(updated.toString(), updated.path("session").path("type").asText(), is("transcription"));
            session.sendClip(rate);
            session.send(MAPPER.createObjectNode().put("type", "input_audio_buffer.commit"));
            JsonNode completed = session.await("conversation.item.input_audio_transcription.completed");
            assertThat(
                    completed.toString(), completed.path("transcript").asText().trim(), not(is("")));
        }
    }

    @Test
    @Timeout(value = 5, unit = TimeUnit.MINUTES)
    public void aRequiredToolIsCalledWithItsArguments() throws Exception {
        try (Session session = Session.open(port, "")) {
            session.await("session.created");
            ObjectNode update = MAPPER.createObjectNode().put("type", "session.update");
            ObjectNode config = update.putObject("session");
            ArrayNode tools = config.putArray("tools");
            ObjectNode tool = tools.addObject().put("type", "function").put("name", "get_weather");
            tool.put("description", "Get the current weather of a city");
            ObjectNode parameters = tool.putObject("parameters").put("type", "object");
            parameters.putObject("properties").putObject("city").put("type", "string");
            parameters.putArray("required").add("city");
            config.put("tool_choice", "required");
            session.send(update);
            session.await("session.updated");
            session.userText("What is the weather in Berlin?");
            session.send(responseCreate("text", 64));
            JsonNode arguments = session.await("response.function_call_arguments.done");
            assertThat(arguments.toString(), arguments.path("name").asText(), is("get_weather"));
            JsonNode parsed = MAPPER.readTree(arguments.path("arguments").asText());
            assertThat(arguments.toString(), parsed.path("city").isTextual(), is(true));
            session.await("response.done");
        }
    }

    /** {@code response.create} with one output modality and a token bound. */
    private static ObjectNode responseCreate(String modality, int maxOutputTokens) {
        ObjectNode create = MAPPER.createObjectNode().put("type", "response.create");
        ObjectNode response = create.putObject("response");
        response.putArray("output_modalities").add(modality);
        response.put("max_output_tokens", maxOutputTokens);
        return create;
    }

    /**
     * The 16-bit PCM of a WAV file as mono at {@code rate}: the channels averaged, resampled by
     * linear interpolation (the server takes PCM16 mono at the model's input rate).
     */
    static byte[] monoPcm16At(byte[] wav, int rate) {
        ByteBuffer in = ByteBuffer.wrap(wav).order(ByteOrder.LITTLE_ENDIAN);
        // RIFF chunks in any order: "fmt " for the layout, "data" for the samples
        int channels = 0;
        int sourceRate = 0;
        int bits = 0;
        int dataOffset = -1;
        int dataSize = 0;
        for (int offset = 12; offset + 8 <= wav.length; ) {
            String id = new String(wav, offset, 4, java.nio.charset.StandardCharsets.US_ASCII);
            int size = in.getInt(offset + 4);
            if (id.equals("fmt ")) {
                channels = in.getShort(offset + 10);
                sourceRate = in.getInt(offset + 12);
                bits = in.getShort(offset + 22);
            } else if (id.equals("data")) {
                dataOffset = offset + 8;
                dataSize = Math.min(size, wav.length - dataOffset);
                break;
            }
            offset += 8 + size + (size & 1);
        }
        if (bits != 16 || channels <= 0 || dataOffset < 0) {
            throw new IllegalArgumentException(
                    "16-bit PCM WAV expected: bits=" + bits + ", channels=" + channels + ", data at " + dataOffset);
        }
        int frames = dataSize / (2 * channels);
        double[] mono = new double[frames];
        for (int f = 0; f < frames; f++) {
            double sum = 0;
            for (int c = 0; c < channels; c++) {
                sum += in.getShort(dataOffset + (f * channels + c) * 2);
            }
            mono[f] = sum / channels;
        }
        int outFrames = (int) ((long) frames * rate / sourceRate);
        ByteBuffer out = ByteBuffer.allocate(outFrames * 2).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < outFrames; i++) {
            double position = (double) i * sourceRate / rate;
            int left = (int) position;
            int right = Math.min(left + 1, frames - 1);
            double value = mono[left] + (mono[right] - mono[left]) * (position - left);
            out.putShort((short) Math.max(Short.MIN_VALUE, Math.min(Short.MAX_VALUE, Math.round(value))));
        }
        return out.array();
    }

    /** A collected response: its text deltas, the size of its audio and the {@code response.done}. */
    private static final class Response {
        final StringBuilder text = new StringBuilder();
        int audioBytes;
        JsonNode done;
    }

    /** One realtime session: a WebSocket that queues every server event as JSON. */
    private static final class Session implements AutoCloseable {
        private final BlockingQueue<JsonNode> events = new LinkedBlockingQueue<>();
        private final List<JsonNode> seen = new ArrayList<>();
        private WebSocket socket;

        static Session open(int port, String query) throws Exception {
            Session session = new Session();
            StringBuilder frame = new StringBuilder();
            session.socket = HttpClient.newHttpClient()
                    .newWebSocketBuilder()
                    .buildAsync(
                            URI.create("ws://127.0.0.1:" + port + "/v1/realtime" + query), new WebSocket.Listener() {
                                @Override
                                public CompletionStage<?> onText(WebSocket webSocket, CharSequence data, boolean last) {
                                    frame.append(data);
                                    if (last) {
                                        try {
                                            session.events.add(MAPPER.readTree(frame.toString()));
                                        } catch (IOException e) {
                                            session.events.add(MAPPER.createObjectNode()
                                                    .put("type", "unparsable")
                                                    .put("raw", frame.toString()));
                                        }
                                        frame.setLength(0);
                                    }
                                    webSocket.request(1);
                                    return null;
                                }
                            })
                    .get(30, TimeUnit.SECONDS);
            return session;
        }

        void send(JsonNode event) throws Exception {
            socket.sendText(MAPPER.writeValueAsString(event), true).get(30, TimeUnit.SECONDS);
        }

        /** The committed test clip as PCM16 mono at {@code rate}, in 200 ms appends. */
        void sendClip(int rate) throws Exception {
            byte[] pcm = monoPcm16At(Files.readAllBytes(Paths.get(TestConstants.DEFAULT_AUDIO_INPUT_PATH)), rate);
            int chunk = rate / 5 * 2;
            for (int offset = 0; offset < pcm.length; offset += chunk) {
                int length = Math.min(chunk, pcm.length - offset);
                ObjectNode append = MAPPER.createObjectNode().put("type", "input_audio_buffer.append");
                append.put(
                        "audio",
                        Base64.getEncoder().encodeToString(java.util.Arrays.copyOfRange(pcm, offset, offset + length)));
                send(append);
            }
        }

        void userText(String text) throws Exception {
            ObjectNode create = MAPPER.createObjectNode().put("type", "conversation.item.create");
            ObjectNode item = create.putObject("item").put("type", "message").put("role", "user");
            item.putArray("content").addObject().put("type", "input_text").put("text", text);
            send(create);
        }

        /** The next event of {@code type}; fails on an {@code error} event or after the timeout. */
        JsonNode await(String type) throws Exception {
            long deadline = System.currentTimeMillis() + TimeUnit.SECONDS.toMillis(EVENT_TIMEOUT_SECONDS);
            while (true) {
                long left = deadline - System.currentTimeMillis();
                JsonNode event = left > 0 ? events.poll(left, TimeUnit.MILLISECONDS) : null;
                if (event == null) {
                    fail("no " + type + " within " + EVENT_TIMEOUT_SECONDS + " s; seen: " + types());
                }
                seen.add(event);
                String eventType = event.path("type").asText();
                if (eventType.equals("error")) {
                    fail("error event while waiting for " + type + ": " + event);
                }
                if (eventType.equals(type)) {
                    return event;
                }
            }
        }

        /** Every event up to {@code response.done}: the text deltas and the size of the audio deltas. */
        Response collectResponse() throws Exception {
            Response response = new Response();
            long deadline = System.currentTimeMillis() + TimeUnit.SECONDS.toMillis(EVENT_TIMEOUT_SECONDS);
            while (true) {
                long left = deadline - System.currentTimeMillis();
                JsonNode event = left > 0 ? events.poll(left, TimeUnit.MILLISECONDS) : null;
                if (event == null) {
                    fail("no response.done within " + EVENT_TIMEOUT_SECONDS + " s; seen: " + types());
                }
                seen.add(event);
                switch (event.path("type").asText()) {
                    case "error":
                        fail("error event in the response: " + event);
                        break;
                    case "response.output_text.delta":
                        response.text.append(event.path("delta").asText());
                        break;
                    case "response.output_audio.delta":
                        response.audioBytes +=
                                Base64.getDecoder().decode(event.path("delta").asText()).length;
                        break;
                    case "response.done":
                        response.done = event;
                        return response;
                    default:
                        break;
                }
            }
        }

        private List<String> types() {
            List<String> types = new ArrayList<>();
            for (JsonNode event : seen) {
                types.add(event.path("type").asText());
            }
            return types;
        }

        @Override
        public void close() {
            if (socket != null) {
                socket.sendClose(WebSocket.NORMAL_CLOSURE, "done").exceptionally(e -> null);
            }
        }
    }
}
