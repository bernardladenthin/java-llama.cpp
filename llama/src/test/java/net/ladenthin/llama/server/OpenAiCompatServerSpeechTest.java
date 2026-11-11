// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.is;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Base64;
import java.util.List;
import net.ladenthin.llama.exception.InvalidRequestException;
import org.jspecify.annotations.Nullable;
import org.junit.jupiter.api.Test;

/**
 * {@code POST /v1/audio/speech} of {@link OpenAiCompatServer} over a real socket with a fake
 * {@link SpeechBackend}, and the pure rules of {@link SpeechApiSupport}. No native library, no model.
 */
public class OpenAiCompatServerSpeechTest {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    private static OpenAiServerConfig config() {
        return OpenAiServerConfig.builder().host("127.0.0.1").port(0).build();
    }

    /** A canonical 44-byte WAV header for 16-bit mono at {@code rate}, followed by {@code samples}. */
    private static byte[] wav(int rate, byte[] samples) {
        byte[] out = new byte[SpeechApiSupport.WAV_HEADER_BYTES + samples.length];
        byte[] riff = "RIFF".getBytes(StandardCharsets.US_ASCII);
        System.arraycopy(riff, 0, out, 0, 4);
        out[24] = (byte) rate;
        out[25] = (byte) (rate >>> 8);
        out[26] = (byte) (rate >>> 16);
        out[27] = (byte) (rate >>> 24);
        System.arraycopy(samples, 0, out, SpeechApiSupport.WAV_HEADER_BYTES, samples.length);
        return out;
    }

    /** Records what it was asked and answers a fixed WAV file with bytes no text encoding survives. */
    private static final class FakeSpeech implements SpeechBackend {
        final List<String> calls = new ArrayList<>();
        final byte[] samples = {(byte) 0x00, (byte) 0x80, (byte) 0xFF, (byte) 0x7F, (byte) 0xC3, (byte) 0x28};

        @Override
        public byte[] synthesize(String text, @Nullable String voice, @Nullable String language) {
            calls.add(text + "|" + voice + "|" + language);
            if ("nobody".equals(voice)) {
                throw new InvalidRequestException("unknown voice 'nobody'");
            }
            if ("broken".equals(text)) {
                throw new IllegalStateException("the engine failed");
            }
            return wav(24000, samples);
        }

        final List<SpeechOptions> options = new ArrayList<>();

        @Override
        public byte[] synthesize(String text, @Nullable String voice, @Nullable String language, SpeechOptions set) {
            options.add(set);
            return synthesize(text, voice, language);
        }

        @Override
        public List<String> voices() {
            return Arrays.asList("female", "male");
        }
    }

    private static final class Answer {
        final int code;
        final byte[] body;
        final @Nullable String contentType;

        Answer(int code, byte[] body, @Nullable String contentType) {
            this.code = code;
            this.body = body;
            this.contentType = contentType;
        }

        String text() {
            return new String(body, StandardCharsets.UTF_8);
        }
    }

    private static Answer send(int port, String method, String path, @Nullable String json) throws IOException {
        HttpURLConnection conn = (HttpURLConnection) new URL("http://127.0.0.1:" + port + path).openConnection();
        conn.setRequestMethod(method);
        if (json != null) {
            conn.setDoOutput(true);
            conn.setRequestProperty("Content-Type", "application/json");
            try (OutputStream os = conn.getOutputStream()) {
                os.write(json.getBytes(StandardCharsets.UTF_8));
            }
        }
        int code = conn.getResponseCode();
        InputStream is = code >= 400 ? conn.getErrorStream() : conn.getInputStream();
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        if (is != null) {
            byte[] buffer = new byte[4096];
            int n;
            while ((n = is.read(buffer)) > 0) {
                bytes.write(buffer, 0, n);
            }
            is.close();
        }
        return new Answer(code, bytes.toByteArray(), conn.getContentType());
    }

    private static OpenAiCompatServer server(@Nullable SpeechBackend speech) throws IOException {
        return new OpenAiCompatServer(new OpenAiCompatServerHttpTest.FakeBackend(), config(), speech).start();
    }

    /**
     * The server's default language and voice stand in for what a request leaves out (llama.cpp
     * b12283, {@code --tts-lang} / {@code --tts-voice}); a speaker reference in the request
     * overrides the default voice, and what the request names wins.
     */
    @Test
    public void theServersDefaultLanguageAndVoiceApplyWhereARequestNamesNone() throws IOException {
        FakeSpeech speech = new FakeSpeech();
        OpenAiServerConfig config = OpenAiServerConfig.builder()
                .host("127.0.0.1")
                .port(0)
                .speechLanguage("de")
                .speechVoice("female")
                .build();
        try (OpenAiCompatServer server =
                new OpenAiCompatServer(new OpenAiCompatServerHttpTest.FakeBackend(), config, speech).start()) {
            assertThat(send(server.getPort(), "POST", "/v1/audio/speech", "{\"input\":\"a\"}").code, is(200));
            assertThat(send(server.getPort(), "POST", "/tts", "{\"input\":\"b\"}").code, is(200));
            assertThat(
                    send(server.getPort(), "POST", "/tts", "{\"input\":\"c\",\"voice\":\"male\",\"lang\":\"en\"}").code,
                    is(200));
            String reference = Base64.getEncoder().encodeToString(new byte[] {1, 2, 3});
            assertThat(
                    send(
                                    server.getPort(),
                                    "POST",
                                    "/tts",
                                    "{\"input\":\"d\",\"speaker_ref_b64\":\"" + reference + "\"}")
                            .code,
                    is(200));
        }
        assertThat(speech.calls, is(Arrays.asList("a|female|de", "b|female|de", "c|male|en", "d|null|de")));
    }

    @Test
    public void aSpeechRequestAnswersTheWavFileByteForByte() throws IOException {
        FakeSpeech speech = new FakeSpeech();
        try (OpenAiCompatServer server = server(speech)) {
            for (String path : new String[] {"/v1/audio/speech", "/audio/speech"}) {
                Answer answer = send(
                        server.getPort(),
                        "POST",
                        path,
                        "{\"model\":\"tts\",\"input\":\"Hello there.\",\"voice\":\"male\",\"lang\":\"en\"}");
                assertThat(path, answer.code, is(200));
                assertThat(answer.contentType, is("audio/wav"));
                assertArrayEquals(wav(24000, speech.samples), answer.body, path);
            }
            assertThat(speech.calls, is(Arrays.asList("Hello there.|male|en", "Hello there.|male|en")));
        }
    }

    @Test
    public void ttsHandsTheSamplingAndTheSpeakerReferenceToTheEngine() throws IOException {
        FakeSpeech speech = new FakeSpeech();
        byte[] reference = {(byte) 0x52, (byte) 0x00, (byte) 0xFF, (byte) 0x80};
        try (OpenAiCompatServer server = server(speech)) {
            Answer answer = send(
                    server.getPort(),
                    "POST",
                    "/tts",
                    "{\"input\":\"Guten Morgen.\",\"voice\":\"male\",\"lang\":\"de\",\"n_predict\":200,"
                            + "\"top_k\":7,\"seed\":42,\"temperature\":0.5,\"top_p\":0.9,\"speaker_ref_b64\":\""
                            + Base64.getEncoder().encodeToString(reference) + "\"}");
            assertThat(answer.text(), answer.code, is(200));
            assertThat(answer.contentType, is("audio/wav"));
            assertArrayEquals(wav(24000, speech.samples), answer.body);
            assertThat(speech.calls, is(Arrays.asList("Guten Morgen.|male|de")));
            SpeechOptions set = speech.options.get(0);
            assertThat(set.getMaxFrames(), is(200));
            assertThat(set.getTopK(), is(7));
            assertThat(set.getSeed(), is(42));
            assertThat(set.getTemperature(), is(0.5f));
            assertArrayEquals(reference, set.getSpeakerReference());
        }
    }

    @Test
    public void ttsTakesPromptAndMaxTokensAsUpstreamsAliasesAndLeavesTheRestUnset() throws IOException {
        FakeSpeech speech = new FakeSpeech();
        try (OpenAiCompatServer server = server(speech)) {
            Answer answer =
                    send(server.getPort(), "POST", "/tts", "{\"prompt\":\"Hi\",\"max_tokens\":64,\"seed\":null}");
            assertThat(answer.text(), answer.code, is(200));
            assertThat(speech.calls, is(Arrays.asList("Hi|null|null")));
            SpeechOptions set = speech.options.get(0);
            assertThat(set.getMaxFrames(), is(64));
            assertThat(set.getTopK(), is((Integer) null));
            assertThat(set.getSeed(), is((Integer) null));
            assertThat(set.getTemperature(), is((Float) null));
            assertThat(set.getSpeakerReference() == null, is(true));

            // a null "n_predict" is an absent one: the alias still counts
            answer = send(server.getPort(), "POST", "/tts", "{\"input\":\"Hi\",\"n_predict\":null,\"max_tokens\":32}");
            assertThat(answer.text(), answer.code, is(200));
            assertThat(speech.options.get(1).getMaxFrames(), is(32));
        }
    }

    @Test
    public void ttsAnswersFloatSamplesForPcmAndSixteenBitOnesForPcm16() throws IOException {
        FakeSpeech speech = new FakeSpeech();
        try (OpenAiCompatServer server = server(speech)) {
            Answer floats = send(server.getPort(), "POST", "/tts", "{\"input\":\"Hi\",\"response_format\":\"pcm\"}");
            assertThat(floats.code, is(200));
            assertThat(floats.contentType, is("audio/pcm;rate=24000;encoding=float;bits=32"));
            ByteBuffer read = ByteBuffer.wrap(floats.body).order(ByteOrder.LITTLE_ENDIAN);
            // the fake's samples: 0x8000 = -32768, 0x7FFF = 32767, 0x28C3 = 10435
            assertThat(floats.body.length, is(12));
            assertThat(read.getFloat(), is(-1.0f));
            assertThat(read.getFloat(), is(32767 / 32768.0f));
            assertThat(read.getFloat(), is(10435 / 32768.0f));

            Answer ints = send(server.getPort(), "POST", "/tts", "{\"input\":\"Hi\",\"response_format\":\"pcm16\"}");
            assertThat(ints.contentType, is("audio/pcm;rate=24000;encoding=signed-int;bits=16"));
            assertArrayEquals(speech.samples, ints.body);
        }
    }

    @Test
    public void aTtsRequestThisRouteDoesNotServeIs400AndNeverReachesTheEngine() throws IOException {
        FakeSpeech speech = new FakeSpeech();
        try (OpenAiCompatServer server = server(speech)) {
            String[] refused = {
                "{}",
                "{\"input\":\"\"}",
                "{\"input\":42}",
                "{\"prompt\":true}",
                "{\"input\":\"Hi\",\"stream\":true}",
                "{\"input\":\"Hi\",\"response_format\":\"mp3\"}",
                "{\"input\":\"Hi\",\"speaker_ref_b64\":\"not base64!\"}",
                "{\"input\":\"Hi\",\"top_k\":\"many\"}",
                "{\"input\":\"Hi\",\"n_predict\":1.5}",
                "{\"input\":\"Hi\",\"temperature\":-1}"
            };
            for (String body : refused) {
                Answer answer = send(server.getPort(), "POST", "/tts", body);
                assertThat(body + " -> " + answer.text(), answer.code, is(400));
            }
            assertThat(speech.calls, is(new ArrayList<String>()));
        }
        try (OpenAiCompatServer server = server(null)) {
            assertThat(send(server.getPort(), "POST", "/tts", "{\"input\":\"Hi\"}").code, is(501));
        }
    }

    @Test
    public void aBackendThatKnowsNoOptionsIsAskedWithoutThem() throws IOException {
        List<String> calls = new ArrayList<>();
        SpeechBackend plain = (text, voice, language) -> {
            calls.add(text + "|" + voice + "|" + language);
            return wav(16000, new byte[] {1, 0});
        };
        try (OpenAiCompatServer server = server(plain)) {
            Answer answer = send(server.getPort(), "POST", "/tts", "{\"input\":\"Hi\",\"top_k\":3,\"voice\":\"alba\"}");
            assertThat(answer.code, is(200));
            assertThat(calls, is(Arrays.asList("Hi|alba|null")));
        }
        assertThat(
                SpeechOptions.defaults().toString(),
                is("SpeechOptions{speakerReference=none, maxFrames=null, topK=null, seed=null, temperature=null}"));
        assertThat(
                new SpeechOptions(new byte[3], 1, 2, 3, 0.25f).toString(),
                is("SpeechOptions{speakerReference=3 bytes, maxFrames=1, topK=2, seed=3, temperature=0.25}"));
    }

    @Test
    public void pcmAnswersTheSamplesAloneAndNamesTheirRate() throws IOException {
        FakeSpeech speech = new FakeSpeech();
        try (OpenAiCompatServer server = server(speech)) {
            Answer answer = send(
                    server.getPort(),
                    "POST",
                    "/v1/audio/speech",
                    "{\"input\":\"Hi\",\"voice\":{\"id\":\"female\"},\"response_format\":\"pcm\"}");
            assertThat(answer.code, is(200));
            assertThat(answer.contentType, is("audio/pcm;rate=24000;encoding=signed-int;bits=16"));
            assertArrayEquals(speech.samples, answer.body);
            assertThat(speech.calls, is(Arrays.asList("Hi|female|null")));
        }
    }

    @Test
    public void aServerWithoutASpeechEngineAnswers501() throws IOException {
        try (OpenAiCompatServer server = server(null)) {
            Answer answer = send(server.getPort(), "POST", "/v1/audio/speech", "{\"input\":\"Hi\"}");
            assertThat(answer.code, is(501));
            assertThat(answer.text(), containsString("does not support speech generation"));
            assertThat(answer.text(), containsString("not_supported_error"));
        }
    }

    @Test
    public void aRequestThisRouteDoesNotServeIs400AndNeverReachesTheEngine() throws IOException {
        FakeSpeech speech = new FakeSpeech();
        try (OpenAiCompatServer server = server(speech)) {
            String[][] refused = {
                {"{}", "'input'"},
                {"{\"input\":\"\"}", "'input'"},
                {"{\"input\":\"Hi\",\"response_format\":\"mp3\"}", "response_format"},
                {"{\"input\":\"Hi\",\"stream_format\":\"sse\"}", "stream_format"},
                {"{\"input\":\"Hi\",\"speed\":1.5}", "'speed'"},
            };
            for (String[] row : refused) {
                Answer answer = send(server.getPort(), "POST", "/v1/audio/speech", row[0]);
                assertThat(row[0], answer.code, is(400));
                assertThat(row[0], answer.text(), containsString(row[1]));
            }
            assertThat(speech.calls.size(), is(0));
        }
    }

    @Test
    public void whatTheEngineRefusesIs400AndWhatItFailsAtIs500() throws IOException {
        try (OpenAiCompatServer server = server(new FakeSpeech())) {
            Answer unknownVoice =
                    send(server.getPort(), "POST", "/v1/audio/speech", "{\"input\":\"Hi\",\"voice\":\"nobody\"}");
            assertThat(unknownVoice.code, is(400));
            assertThat(unknownVoice.text(), containsString("unknown voice 'nobody'"));
            Answer failed = send(server.getPort(), "POST", "/v1/audio/speech", "{\"input\":\"broken\"}");
            assertThat(failed.code, is(500));
            assertThat(failed.text(), containsString("the engine failed"));
        }
    }

    @Test
    public void onlyPostIsServed() throws IOException {
        try (OpenAiCompatServer server = server(new FakeSpeech())) {
            assertThat(send(server.getPort(), "GET", "/v1/audio/speech", null).code, is(405));
        }
    }

    @Test
    public void propsListTheVoicesOfTheSpeechEngineAndNoneWithoutOne() throws IOException {
        try (OpenAiCompatServer server = server(new FakeSpeech())) {
            JsonNode props = MAPPER.readTree(
                    send(server.getPort(), "GET", "/props", null).text());
            assertThat(props.path("voices").size(), is(2));
            assertThat(props.path("voices").get(1).asText(), is("male"));
        }
        try (OpenAiCompatServer server = server(null)) {
            JsonNode props = MAPPER.readTree(
                    send(server.getPort(), "GET", "/props", null).text());
            assertThat(props.has("voices"), is(false));
        }
    }

    // ----- SpeechApiSupport, without a server -----

    private static SpeechApiSupport.Request parse(String json) throws IOException {
        return SpeechApiSupport.parse(MAPPER.readTree(json));
    }

    @Test
    public void theDefaultsAreAWavFileInTheModelsVoiceAndLanguage() throws IOException {
        SpeechApiSupport.Request request = parse("{\"input\":\"Hi\",\"instructions\":\"cheerful\",\"speed\":1.0}");
        assertThat(request.input(), is("Hi"));
        assertThat(request.voice() == null, is(true));
        assertThat(request.language() == null, is(true));
        assertThat(request.pcm(), is(false));
        assertThat(
                parse("{\"input\":\"Hi\",\"stream_format\":\"audio\",\"speed\":1}")
                        .pcm(),
                is(false));
        assertThat(parse("{\"input\":\"Hi\",\"response_format\":\"wav\"}").pcm(), is(false));
    }

    @Test
    public void aSpeedThatIsNoNumberIsRefused() {
        assertThrows(IllegalArgumentException.class, () -> parse("{\"input\":\"Hi\",\"speed\":\"fast\"}"));
        assertThrows(IllegalArgumentException.class, () -> parse("{\"input\":7}"));
    }

    @Test
    public void theSampleRateIsReadLittleEndianAndBytesThatAreNoWavAreRefused() {
        assertThat(SpeechApiSupport.sampleRate(wav(24000, new byte[0])), is(24000));
        assertThat(SpeechApiSupport.sampleRate(wav(0x01020304, new byte[0])), is(0x01020304));
        assertThat(SpeechApiSupport.pcm(wav(16000, new byte[] {1, 2, 3})), is(new byte[] {1, 2, 3}));
        assertThrows(IllegalArgumentException.class, () -> SpeechApiSupport.sampleRate(new byte[43]));
        byte[] notRiff = wav(24000, new byte[4]);
        for (int i = 0; i < 4; i++) {
            byte[] broken = notRiff.clone();
            broken[i] = 'x';
            assertThrows(IllegalArgumentException.class, () -> SpeechApiSupport.pcm(broken));
        }
        assertThat(SpeechApiSupport.pcmContentType(48000), is("audio/pcm;rate=48000;encoding=signed-int;bits=16"));
    }
}
