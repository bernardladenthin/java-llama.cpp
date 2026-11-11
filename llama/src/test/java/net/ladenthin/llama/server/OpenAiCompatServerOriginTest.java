// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.not;
import static org.hamcrest.Matchers.startsWith;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.InetAddress;
import java.net.Socket;
import java.nio.charset.StandardCharsets;
import org.junit.jupiter.api.Test;

/**
 * {@link OpenAiCompatServer} refusing requests from somewhere else once {@code corsAllowOrigin} is not
 * {@code "*"}: the {@link OriginGuard} rules over real HTTP. The requests are written to a socket,
 * because {@code HttpURLConnection} sets {@code Host} and {@code Origin} itself. No native library, no
 * model.
 */
public class OpenAiCompatServerOriginTest {

    private static OpenAiServerConfig config(String corsAllowOrigin) {
        return OpenAiServerConfig.builder()
                .host("127.0.0.1")
                .port(0)
                .modelId("test-model")
                .corsAllowOrigin(corsAllowOrigin)
                .build();
    }

    /** The whole response (status line, headers, body) to {@code method path} with the given header lines. */
    private static String request(int port, String method, String path, String... headerLines) throws IOException {
        try (Socket socket = new Socket(InetAddress.getByName("127.0.0.1"), port)) {
            socket.setSoTimeout(10_000);
            StringBuilder request =
                    new StringBuilder(method).append(' ').append(path).append(" HTTP/1.1\r\n");
            boolean host = false;
            for (String line : headerLines) {
                host |= line.regionMatches(true, 0, "Host:", 0, 5);
                request.append(line).append("\r\n");
            }
            if (!host) {
                request.append("Host: 127.0.0.1:").append(port).append("\r\n");
            }
            request.append("Connection: close\r\n\r\n");
            OutputStream out = socket.getOutputStream();
            out.write(request.toString().getBytes(StandardCharsets.UTF_8));
            out.flush();
            ByteArrayOutputStream buffer = new ByteArrayOutputStream();
            InputStream in = socket.getInputStream();
            byte[] chunk = new byte[1024];
            int read;
            while ((read = in.read(chunk)) != -1) {
                buffer.write(chunk, 0, read);
            }
            return new String(buffer.toByteArray(), StandardCharsets.UTF_8);
        }
    }

    private static String get(int port, String... headerLines) throws IOException {
        return request(port, "GET", "/health", headerLines);
    }

    @Test
    public void anOriginListRefusesAForeignOriginAndHost() throws IOException {
        try (OpenAiCompatServer server = new OpenAiCompatServer(
                        new OpenAiCompatServerHttpTest.FakeBackend(), config("http://app.example"))
                .start()) {
            int port = server.getPort();
            assertThat(get(port), startsWith("HTTP/1.1 200"));

            String allowed = get(port, "Origin: http://app.example");
            assertThat(allowed, startsWith("HTTP/1.1 200"));
            assertThat(allowed, containsString("Access-control-allow-origin: http://app.example"));
            assertThat(allowed, containsString("Vary: Origin"));

            String foreign = get(port, "Origin: http://evil.example");
            assertThat(foreign, startsWith("HTTP/1.1 403"));
            assertThat(foreign, containsString("\"type\":\"permission_error\""));
            assertThat(foreign, containsString("corsAllowOrigin"));
            assertThat(foreign.toLowerCase(java.util.Locale.ROOT), not(containsString("access-control-allow-origin")));

            // DNS rebinding: the connection is loopback, the name in Host is somebody else's
            assertThat(get(port, "Host: evil.example:" + port), startsWith("HTTP/1.1 403"));
            assertThat(get(port, "Host: localhost:" + port), startsWith("HTTP/1.1 200"));
            assertThat(get(port, "Host: app.example"), startsWith("HTTP/1.1 200"));

            assertThat(get(port, "Sec-Fetch-Site: cross-site"), startsWith("HTTP/1.1 403"));
            assertThat(get(port, "Sec-Fetch-Site: same-origin"), startsWith("HTTP/1.1 200"));
        }
    }

    @Test
    public void theRefusalComesBeforeTheRouteRuns() throws IOException {
        try (OpenAiCompatServer server = new OpenAiCompatServer(
                        new OpenAiCompatServerHttpTest.FakeBackend(), config("http://app.example"))
                .start()) {
            int port = server.getPort();
            // a generating route, a route that does not exist and the catch-all answer alike
            assertThat(
                    request(port, "POST", "/v1/chat/completions", "Origin: http://evil.example", "Content-Length: 0"),
                    startsWith("HTTP/1.1 403"));
            assertThat(request(port, "GET", "/nowhere", "Origin: http://evil.example"), startsWith("HTTP/1.1 403"));
        }
    }

    @Test
    public void aPreflightIsAnsweredForAnyOriginAndNamesOnlyAnAllowedOne() throws IOException {
        try (OpenAiCompatServer server = new OpenAiCompatServer(
                        new OpenAiCompatServerHttpTest.FakeBackend(), config("http://app.example"))
                .start()) {
            int port = server.getPort();
            String allowed = request(port, "OPTIONS", "/v1/chat/completions", "Origin: http://app.example");
            assertThat(allowed, startsWith("HTTP/1.1 204"));
            assertThat(allowed, containsString("Access-control-allow-origin: http://app.example"));
            // the browser reads the missing header as a refusal and never sends the request
            String foreign = request(port, "OPTIONS", "/v1/chat/completions", "Origin: http://evil.example");
            assertThat(foreign, startsWith("HTTP/1.1 204"));
            assertThat(foreign.toLowerCase(java.util.Locale.ROOT), not(containsString("access-control-allow-origin")));
        }
    }

    @Test
    public void localhostServesLocalOriginsOnly() throws IOException {
        try (OpenAiCompatServer server =
                new OpenAiCompatServer(new OpenAiCompatServerHttpTest.FakeBackend(), config("localhost")).start()) {
            int port = server.getPort();
            String local = get(port, "Origin: http://localhost:5173");
            assertThat(local, startsWith("HTTP/1.1 200"));
            assertThat(local, containsString("Access-control-allow-origin: http://localhost:5173"));
            assertThat(get(port, "Origin: http://evil.example"), startsWith("HTTP/1.1 403"));
            assertThat(get(port, "Host: evil.example"), startsWith("HTTP/1.1 403"));
        }
    }

    @Test
    public void theDefaultServesEveryOriginAndHost() throws IOException {
        try (OpenAiCompatServer server =
                new OpenAiCompatServer(new OpenAiCompatServerHttpTest.FakeBackend(), config("*")).start()) {
            int port = server.getPort();
            String foreign = get(port, "Origin: http://evil.example", "Sec-Fetch-Site: cross-site");
            assertThat(foreign, startsWith("HTTP/1.1 200"));
            assertThat(foreign, containsString("Access-control-allow-origin: *"));
            assertThat(foreign, not(containsString("Vary: Origin")));
            assertThat(get(port, "Host: evil.example:" + port), startsWith("HTTP/1.1 200"));
        }
    }
}
