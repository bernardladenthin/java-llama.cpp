// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.atmosphere;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import jakarta.servlet.Filter;
import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.ServletRequest;
import jakarta.servlet.ServletResponse;
import jakarta.servlet.http.HttpServlet;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.security.SecureRandom;
import java.util.Base64;

/**
 * The browser page: Atmosphere's own prebuilt AI Console, served without Spring.
 *
 * <p>The console is a finished single-page app shipped inside {@code atmosphere-spring-boot-starter}
 * under {@code META-INF/resources/atmosphere/console/}; that jar is on the classpath for its resources only,
 * with every dependency excluded. The app needs exactly two things from a server, and this class provides
 * both — which is what the Spring starter otherwise does:
 *
 * <ul>
 *   <li>its files, with the {@code __ATMO_CSP_NONCE__} placeholder in {@code index.html} replaced by a fresh
 *       nonce and a matching strict Content-Security-Policy header ({@link Files});
 *   <li>{@code GET /api/console/info}, which tells it which endpoint to connect to and which optional tabs
 *       exist ({@link Info}) — all of them off here, since none of those server features are running.
 * </ul>
 *
 * <p>The console's contract is not documented; its info fields were read from the starter's
 * {@code AtmosphereConsoleInfoEndpoint}. {@code WebServerTest} pins that the page loads and connects, so an
 * Atmosphere upgrade that changes the contract is noticed rather than shipped.
 */
final class WebConsole {

    /** The URL path the console is served under. */
    static final String PATH = "/atmosphere/console";

    /** Where the console's files are inside the starter jar. */
    private static final String RESOURCES = "META-INF/resources/atmosphere/console/";

    private WebConsole() {}

    /**
     * Whether the console's files are on the classpath at all.
     *
     * @return {@code true} when {@code index.html} can be found
     */
    static boolean available() {
        return WebConsole.class.getClassLoader().getResource(RESOURCES + "index.html") != null;
    }

    /**
     * Serves the console's files. A filter rather than a servlet because the path lies inside the Atmosphere
     * servlet's own mapping, which a servlet could not win against.
     */
    static final class Files implements Filter {

        private static final SecureRandom RANDOM = new SecureRandom();

        @Override
        public void doFilter(ServletRequest request, ServletResponse response, FilterChain chain)
                throws IOException, ServletException {
            HttpServletRequest http = (HttpServletRequest) request;
            HttpServletResponse out = (HttpServletResponse) response;
            String relative = http.getRequestURI().substring(PATH.length());
            if (relative.isEmpty() || "/".equals(relative)) {
                relative = "/index.html";
            }
            if (relative.contains("..") || relative.contains("\\")) {
                out.sendError(400);
                return;
            }
            String name = relative.substring(1);
            try (InputStream in = WebConsole.class.getClassLoader().getResourceAsStream(RESOURCES + name)) {
                if (in == null) {
                    out.sendError(404);
                    return;
                }
                out.setContentType(contentType(name));
                out.setHeader("X-Content-Type-Options", "nosniff");
                if (!"index.html".equals(name)) {
                    in.transferTo(out.getOutputStream());
                    return;
                }
                byte[] random = new byte[16];
                RANDOM.nextBytes(random);
                String nonce = Base64.getEncoder().encodeToString(random);
                String html =
                        new String(in.readAllBytes(), StandardCharsets.UTF_8).replace("__ATMO_CSP_NONCE__", nonce);
                out.setHeader(
                        "Content-Security-Policy",
                        "default-src 'self'; script-src 'nonce-" + nonce
                                + "' 'strict-dynamic'; style-src 'self' 'nonce-"
                                + nonce + "'; img-src 'self' data:; font-src 'self' data:; connect-src 'self' ws: wss:;"
                                + " frame-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none';"
                                + " form-action 'self'");
                out.setHeader("Cache-Control", "no-store");
                out.getOutputStream().write(html.getBytes(StandardCharsets.UTF_8));
            }
        }

        static String contentType(String name) {
            if (name.endsWith(".html")) {
                return "text/html; charset=utf-8";
            }
            if (name.endsWith(".js")) {
                return "text/javascript; charset=utf-8";
            }
            if (name.endsWith(".css")) {
                return "text/css; charset=utf-8";
            }
            if (name.endsWith(".svg")) {
                return "image/svg+xml";
            }
            if (name.endsWith(".png")) {
                return "image/png";
            }
            if (name.endsWith(".woff2")) {
                return "font/woff2";
            }
            return "application/octet-stream";
        }
    }

    /** {@code GET /api/console/info}: which endpoint to connect to, and which tabs to show. */
    static final class Info extends HttpServlet {

        private static final long serialVersionUID = 1L;

        /** The path the console asks. */
        static final String PATH = "/api/console/info";

        private final String subtitle;

        Info(String subtitle) {
            this.subtitle = subtitle;
        }

        @Override
        protected void doGet(HttpServletRequest request, HttpServletResponse response) throws IOException {
            ObjectNode info = new ObjectMapper().createObjectNode();
            info.put("subtitle", subtitle);
            info.put("endpoint", WebServer.AGENT_PATH);
            info.put("runtime", "built-in");
            info.put("mode", "ai");
            info.put("transport", "atmosphere");
            for (String feature : new String[] {
                "hasInteractions",
                "hasVerifier",
                "hasCheckpoints",
                "hasRooms",
                "hasActuator",
                "hasMetrics",
                "hasAdmin",
                "hasWorkspace",
                "hasTape"
            }) {
                info.put(feature, false);
            }
            response.setContentType("application/json; charset=utf-8");
            response.setHeader("Cache-Control", "no-store");
            response.getOutputStream().write(info.toString().getBytes(StandardCharsets.UTF_8));
        }
    }
}
