// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.atmosphere;

import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.Locale;
import java.util.Set;
import org.eclipse.jetty.http.HttpCookie;
import org.eclipse.jetty.http.HttpHeader;
import org.eclipse.jetty.server.Handler;
import org.eclipse.jetty.server.Request;
import org.eclipse.jetty.server.Response;
import org.eclipse.jetty.util.Callback;
import org.eclipse.jetty.util.Fields;
import org.jspecify.annotations.Nullable;

/**
 * The door in front of the browser front end: nothing reaches the agent without the token.
 *
 * <p>This is not a formality. The agent can write files and, with {@code --allow-shell}, run any command
 * line, so whoever can talk to this port can run commands as the user who started it. Three checks, all in
 * front of every servlet so that the WebSocket upgrade is covered as well as the pages:
 *
 * <ol>
 *   <li><b>The token.</b> Printed once at startup as {@code /?token=…}. Visiting that address exchanges it
 *       for an {@code HttpOnly}, {@code SameSite=Strict} cookie and redirects to the console, so the token
 *       leaves the address bar at once. Scripts may send {@code Authorization: Bearer <token>} instead.
 *   <li><b>The origin.</b> A request that carries an {@code Origin} header must come from this very
 *       server. That is what stops another web page open in the same browser from driving the agent
 *       (cross-site WebSocket hijacking); {@code SameSite=Strict} stops it from riding on the cookie.
 *   <li><b>The host name.</b> On a loopback address the {@code Host} header must name loopback too, which
 *       is what defeats DNS rebinding: an attacker's name that resolves to 127.0.0.1 still says so in
 *       {@code Host}.
 * </ol>
 *
 * <p>The comparison of the token is constant-time, so its length and prefix cannot be probed.
 */
final class WebAccessGuard extends Handler.Wrapper {

    /** The cookie the token is exchanged for. */
    static final String COOKIE = "jllama_agent";

    /** Where a successful token exchange, and a plain visit to {@code /}, lands. */
    private final String home;

    private final byte[] token;
    private final boolean loopbackOnly;

    private static final Set<String> LOOPBACK_NAMES = Set.of("127.0.0.1", "localhost", "[::1]", "::1");

    /**
     * Guard the handler that will be set as this wrapper's child.
     *
     * @param token the access token
     * @param bindHost the address the server binds; on loopback the {@code Host} header must be loopback too
     * @param home where to send the browser after the token exchange
     */
    WebAccessGuard(String token, String bindHost, String home) {
        this.token = token.getBytes(StandardCharsets.UTF_8);
        this.loopbackOnly = isLoopback(bindHost);
        this.home = home;
    }

    @Override
    public boolean handle(Request request, Response response, Callback callback) throws Exception {
        String host = request.getHeaders().get(HttpHeader.HOST);
        if (loopbackOnly && !isLoopback(hostName(host))) {
            Response.writeError(request, response, callback, 403, "this agent answers on a loopback name only");
            return true;
        }
        String origin = request.getHeaders().get(HttpHeader.ORIGIN);
        if (origin != null && !sameOrigin(origin, host)) {
            Response.writeError(request, response, callback, 403, "cross-origin requests are refused");
            return true;
        }
        Fields query = Request.extractQueryParameters(request);
        String offered = query.getValue("token");
        if (offered != null) {
            if (!matches(offered)) {
                Response.writeError(request, response, callback, 401, "wrong token");
                return true;
            }
            response.getHeaders()
                    .add(
                            HttpHeader.SET_COOKIE,
                            COOKIE + "=" + new String(token, StandardCharsets.UTF_8)
                                    + "; Path=/; HttpOnly; SameSite=Strict");
            Response.sendRedirect(request, response, callback, home);
            return true;
        }
        if (!authorized(request)) {
            Response.writeError(
                    request,
                    response,
                    callback,
                    401,
                    "open the address the agent printed at startup (it carries ?token=...)");
            return true;
        }
        if ("/".equals(Request.getPathInContext(request))) {
            Response.sendRedirect(request, response, callback, home);
            return true;
        }
        return super.handle(request, response, callback);
    }

    private boolean authorized(Request request) {
        String authorization = request.getHeaders().get(HttpHeader.AUTHORIZATION);
        if (authorization != null && authorization.regionMatches(true, 0, "Bearer ", 0, 7)) {
            return matches(authorization.substring(7).strip());
        }
        for (HttpCookie cookie : Request.getCookies(request)) {
            if (COOKIE.equals(cookie.getName()) && matches(cookie.getValue())) {
                return true;
            }
        }
        return false;
    }

    private boolean matches(String offered) {
        return MessageDigest.isEqual(token, offered.getBytes(StandardCharsets.UTF_8));
    }

    /**
     * Whether an {@code Origin} header names the server the request was sent to.
     *
     * @param origin the {@code Origin} header, e.g. {@code http://127.0.0.1:8787}
     * @param host the {@code Host} header, e.g. {@code 127.0.0.1:8787}
     * @return {@code true} for the same scheme-less authority
     */
    static boolean sameOrigin(String origin, @Nullable String host) {
        if (host == null) {
            return false;
        }
        try {
            URI uri = URI.create(origin);
            String authority = uri.getRawAuthority();
            return ("http".equals(uri.getScheme()) || "https".equals(uri.getScheme()))
                    && authority != null
                    && authority.equalsIgnoreCase(host);
        } catch (IllegalArgumentException e) {
            return false;
        }
    }

    /**
     * The host name of a {@code Host} header, without the port.
     *
     * @param host the header, or {@code null}
     * @return the name, lower case; empty for a missing header
     */
    static String hostName(@Nullable String host) {
        if (host == null) {
            return "";
        }
        String name = host.startsWith("[")
                ? host.substring(0, host.indexOf(']') + 1)
                : (host.lastIndexOf(':') > 0 ? host.substring(0, host.lastIndexOf(':')) : host);
        return name.toLowerCase(Locale.ROOT);
    }

    /**
     * Whether a host name or address is loopback.
     *
     * @param host the name, e.g. {@code localhost} or {@code 127.0.0.1}
     * @return {@code true} for loopback
     */
    static boolean isLoopback(String host) {
        return LOOPBACK_NAMES.contains(host.toLowerCase(Locale.ROOT)) || host.startsWith("127.");
    }
}
