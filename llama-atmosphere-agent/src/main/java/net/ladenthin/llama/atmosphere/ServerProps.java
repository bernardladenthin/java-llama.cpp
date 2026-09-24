// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.atmosphere;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Reads the context size of a running server from llama.cpp's {@code GET /props}, so the status line
 * can show a percentage in {@code --base-url} mode too.
 *
 * <p>Both java-llama.cpp's {@code OpenAiCompatServer} and upstream {@code llama-server} answer with
 * {@code {"default_generation_settings":{"n_ctx":…}}}. The endpoint sits next to the OpenAI routes
 * rather than under {@code /v1}, and this project's server serves it under both, so both are tried.
 * Every failure — a server without the route, a foreign OpenAI-compatible endpoint, a timeout — is
 * reported as "unknown"; the status line then shows the token count without a percentage rather than
 * a made-up denominator.
 */
public final class ServerProps {

    private static final Pattern N_CTX = Pattern.compile("\"n_ctx\"\\s*:\\s*(\\d+)");
    private static final Duration TIMEOUT = Duration.ofSeconds(3);

    private ServerProps() {}

    /**
     * Look the context size up.
     *
     * @param baseUrl the OpenAI-compatible base URL, e.g. {@code http://127.0.0.1:8080/v1}
     * @param apiKey the bearer token to send
     * @return the context size in tokens, or {@link StatusLine#UNKNOWN_CONTEXT} when it cannot be read
     */
    public static int contextSize(String baseUrl, String apiKey) {
        HttpClient client = HttpClient.newBuilder().connectTimeout(TIMEOUT).build();
        for (String url : candidates(baseUrl)) {
            int size = read(client, url, apiKey);
            if (size != StatusLine.UNKNOWN_CONTEXT) {
                return size;
            }
        }
        return StatusLine.UNKNOWN_CONTEXT;
    }

    /**
     * The {@code /props} URLs tried, in order.
     *
     * @param baseUrl the base URL
     * @return the candidate URLs
     */
    static List<String> candidates(String baseUrl) {
        String trimmed = baseUrl.endsWith("/") ? baseUrl.substring(0, baseUrl.length() - 1) : baseUrl;
        if (trimmed.endsWith("/v1")) {
            String root = trimmed.substring(0, trimmed.length() - "/v1".length());
            return List.of(root + "/props", trimmed + "/props");
        }
        return List.of(trimmed + "/props");
    }

    /**
     * Extract {@code n_ctx} from a {@code /props} body.
     *
     * @param body the response body
     * @return the context size, or {@link StatusLine#UNKNOWN_CONTEXT} when the field is absent
     */
    static int parseContextSize(String body) {
        Matcher matcher = N_CTX.matcher(body);
        if (!matcher.find()) {
            return StatusLine.UNKNOWN_CONTEXT;
        }
        try {
            return Integer.parseInt(matcher.group(1));
        } catch (NumberFormatException e) {
            return StatusLine.UNKNOWN_CONTEXT;
        }
    }

    private static int read(HttpClient client, String url, String apiKey) {
        try {
            HttpRequest request = HttpRequest.newBuilder(URI.create(url))
                    .timeout(TIMEOUT)
                    .header("Authorization", "Bearer " + apiKey)
                    .GET()
                    .build();
            HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString());
            return response.statusCode() == 200 ? parseContextSize(response.body()) : StatusLine.UNKNOWN_CONTEXT;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            return StatusLine.UNKNOWN_CONTEXT;
        } catch (RuntimeException | java.io.IOException e) {
            return StatusLine.UNKNOWN_CONTEXT;
        }
    }
}
