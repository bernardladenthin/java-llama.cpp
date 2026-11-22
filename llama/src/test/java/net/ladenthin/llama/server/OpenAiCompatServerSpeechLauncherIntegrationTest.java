// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import net.ladenthin.llama.ClaudeGenerated;
import net.ladenthin.llama.TestConstants;
import net.ladenthin.llama.loader.NativeLibraryPresence;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

/**
 * The standalone launcher with a speech model: {@code OpenAiCompatServer.main} with
 * {@code --tts-model} / {@code --tts-mmproj} loads a {@code TextToSpeech} beside the chat model and
 * serves the speech routes from it. {@code OpenAiServerCliTest} holds the parsing only; this starts
 * {@code main} in a child JVM, as a user does, and asks it to speak. The chat model is CI's draft
 * model, the speech model CI's tiny-voca pair (the smallest of the set); either missing skips.
 */
@ClaudeGenerated(
        purpose = "End-to-end guard that the launcher's --tts-model / --tts-mmproj options make a running "
                + "OpenAiCompatServer speak over /v1/audio/speech and list the speech model's voices in /props.")
public class OpenAiCompatServerSpeechLauncherIntegrationTest {

    private static final Pattern READY = Pattern.compile("OpenAI-compatible endpoint ready: http://([^:/]+):(\\d+)/");

    private static final int WAV_HEADER_BYTES = 44;

    private static final String LIB_PATH_PROPERTY = "net.ladenthin.llama.lib.path";

    @Test
    @Timeout(value = 15, unit = TimeUnit.MINUTES)
    public void theLauncherWithASpeechModelSpeaksAndListsItsVoices() throws Exception {
        assumeTrue(NativeLibraryPresence.onClasspath(), "libjllama not on classpath");
        String model = TestConstants.DRAFT_MODEL_PATH;
        String ttsModel = TestConstants.resolveModelProperty(
                TestConstants.PROP_TINY_VOCA_MODEL, TestConstants.DEFAULT_TINY_VOCA_MODEL_PATH);
        String ttsMmproj = TestConstants.resolveModelProperty(
                TestConstants.PROP_TINY_VOCA_MMPROJ, TestConstants.DEFAULT_TINY_VOCA_MMPROJ_PATH);
        for (String file : Arrays.asList(model, ttsModel, ttsMmproj)) {
            assumeTrue(new File(file).isFile(), "model not found: " + file);
        }

        List<String> command = new ArrayList<>();
        command.add(Paths.get(System.getProperty("java.home"), "bin", "java").toString());
        // the natives the test JVM was told about, for a job that has them outside the classpath
        for (String property : Arrays.asList(LIB_PATH_PROPERTY, "java.library.path")) {
            String value = System.getProperty(property);
            if (value != null && !value.isEmpty()) {
                command.add("-D" + property + "=" + value);
            }
        }
        command.add("-cp");
        command.add(System.getProperty("java.class.path"));
        command.add(OpenAiCompatServer.class.getName());
        command.addAll(Arrays.asList(
                "-m",
                model,
                "-c",
                "512",
                "-ngl",
                "0",
                "-t",
                "2",
                "--host",
                "127.0.0.1",
                "--port",
                "0",
                "--tts-model",
                ttsModel,
                "--tts-mmproj",
                ttsMmproj));
        ProcessBuilder builder = new ProcessBuilder(command);
        builder.redirectErrorStream(true);
        Process child = builder.start();
        StringBuilder output = new StringBuilder();
        CountDownLatch ready = new CountDownLatch(1);
        String[] endpoint = new String[2];
        Thread reader = new Thread(
                () -> {
                    try (BufferedReader lines =
                            new BufferedReader(new InputStreamReader(child.getInputStream(), StandardCharsets.UTF_8))) {
                        String line;
                        while ((line = lines.readLine()) != null) {
                            synchronized (output) {
                                output.append(line).append('\n');
                            }
                            Matcher matcher = READY.matcher(line);
                            if (matcher.find()) {
                                endpoint[0] = matcher.group(1);
                                endpoint[1] = matcher.group(2);
                                ready.countDown();
                            }
                        }
                    } catch (IOException e) {
                        // the child ended; what it printed is in output
                    } finally {
                        ready.countDown(); // at the end of the output too: a child that died is a failure now
                    }
                },
                "launcher-output");
        reader.setDaemon(true);
        reader.start();
        Throwable failure = null;
        try {
            assertTrue(
                    ready.await(10, TimeUnit.MINUTES) && endpoint[1] != null && child.isAlive(),
                    () -> "the launcher did not start:\n" + snapshot(output));
            String base = "http://" + endpoint[0] + ":" + endpoint[1];

            byte[] wav = post(base + "/v1/audio/speech", "{\"input\":\"Hello there.\",\"response_format\":\"wav\"}");
            assertTrue(wav.length > WAV_HEADER_BYTES, "a WAV with samples, got " + wav.length + " bytes");
            assertEquals("RIFF", new String(wav, 0, 4, StandardCharsets.US_ASCII));
            assertEquals("WAVE", new String(wav, 8, 4, StandardCharsets.US_ASCII));
            ByteBuffer header = ByteBuffer.wrap(wav).order(ByteOrder.LITTLE_ENDIAN);
            assertEquals(24000, header.getInt(24), "tiny-voca speaks at 24 kHz");
            boolean audible = false;
            for (int i = WAV_HEADER_BYTES; i < wav.length && !audible; i++) {
                audible = wav[i] != 0;
            }
            assertTrue(audible, "the speech must not be silence");

            JsonNode props = new ObjectMapper().readTree(get(base + "/props"));
            JsonNode voices = props.path("voices");
            assertTrue(voices.isArray() && voices.size() > 0, () -> "the speech model's voices in /props: " + props);
        } catch (Throwable e) {
            failure = e;
            throw e;
        } finally {
            child.destroy();
            boolean ended = child.waitFor(2, TimeUnit.MINUTES);
            if (!ended) {
                child.destroyForcibly();
                AssertionError notEnded =
                        new AssertionError("the launcher did not end on SIGTERM:\n" + snapshot(output));
                if (failure != null) {
                    failure.addSuppressed(notEnded); // the first failure is the one to read
                } else {
                    throw notEnded;
                }
            }
        }
        assertFalse(child.isAlive());
    }

    private static String snapshot(StringBuilder output) {
        synchronized (output) {
            return output.toString();
        }
    }

    private static byte[] post(String url, String body) throws IOException {
        HttpURLConnection connection = (HttpURLConnection) new URL(url).openConnection();
        connection.setRequestMethod("POST");
        connection.setDoOutput(true);
        connection.setConnectTimeout(30_000);
        connection.setReadTimeout(600_000);
        connection.setRequestProperty("Content-Type", "application/json");
        try (OutputStream out = connection.getOutputStream()) {
            out.write(body.getBytes(StandardCharsets.UTF_8));
        }
        int status = connection.getResponseCode();
        InputStream in = status < 400 ? connection.getInputStream() : connection.getErrorStream();
        assertNotNull(in, "no body for status " + status);
        byte[] response = readAll(in);
        assertEquals(
                200, status, () -> url + " answered " + status + ": " + new String(response, StandardCharsets.UTF_8));
        return response;
    }

    private static byte[] get(String url) throws IOException {
        HttpURLConnection connection = (HttpURLConnection) new URL(url).openConnection();
        connection.setConnectTimeout(30_000);
        connection.setReadTimeout(60_000);
        assertEquals(200, connection.getResponseCode(), url);
        return readAll(connection.getInputStream());
    }

    private static byte[] readAll(InputStream in) throws IOException {
        try (InputStream stream = in) {
            ByteArrayOutputStream bytes = new ByteArrayOutputStream();
            byte[] buffer = new byte[8192];
            int n;
            while ((n = stream.read(buffer)) != -1) {
                bytes.write(buffer, 0, n);
            }
            return bytes.toByteArray();
        }
    }
}
