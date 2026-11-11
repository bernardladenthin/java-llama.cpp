// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import java.util.Locale;
import org.jspecify.annotations.Nullable;

/**
 * Decides whether a request may run under the configured CORS origins, and which
 * {@code Access-Control-Allow-Origin} value answers it.
 *
 * <p>An {@code Access-Control-Allow-Origin} header alone protects nothing: without it a browser hides the
 * <em>response</em> from a foreign page, but it still sends the request, and a completion or a slot erase
 * runs. So for every configured value other than {@code "*"} a request is refused when it comes from
 * somewhere else, by the rules of llama.cpp's own server (its {@code --cors-origins}, upstream #32098):
 *
 * <ul>
 *   <li><b>DNS rebinding.</b> On a loopback connection the {@code Host} header must name loopback
 *       ({@code localhost}, {@code 127.0.0.0/8}, {@code ::1}, the unspecified addresses) or the host of an
 *       allowed origin. A page whose own name resolves to {@code 127.0.0.1} reaches the server with that
 *       name in {@code Host}, and is same-origin to itself.
 *   <li><b>A foreign {@code Origin}.</b> An {@code Origin} is served when it is the request's own
 *       {@code Host} (a page this server delivered), an entry of the list, or, for the value
 *       {@code "localhost"}, any origin on {@code localhost}, {@code 127.0.0.1} or {@code ::1}.
 *   <li><b>No {@code Origin}.</b> A client that is no browser is served. A browser's request without one
 *       says where it comes from in {@code Sec-Fetch-Site}; {@code cross-site} and {@code same-site} are
 *       refused (this server delivers no page a navigation could want).
 * </ul>
 *
 * <p>Stateless and free of the HTTP server's types, so every rule is unit-testable with literals.
 */
final class OriginGuard {

    /** The value that serves every origin and checks nothing. */
    static final String ANY = "*";

    /** The value that serves the origins of this machine, whatever their port. */
    static final String LOCALHOST = "localhost";

    private static final String SCHEME_SEPARATOR = "://";
    private static final String IPV4_MAPPED_PREFIX = "::ffff:";
    private static final int IPV4_PARTS = 4;
    private static final int IPV4_PART_MAX = 255;
    private static final int LOOPBACK_FIRST_PART = 127;

    private OriginGuard() {}

    /**
     * Whether the request may run.
     *
     * @param allowedOrigins the configured value: {@code "*"}, {@code "localhost"} or a comma-separated
     *     list of origins ({@code scheme://host[:port]})
     * @param loopbackConnection whether the request arrived on a loopback address of this machine
     * @param host the request's {@code Host} header
     * @param origin the request's {@code Origin} header
     * @param secFetchSite the request's {@code Sec-Fetch-Site} header
     * @return {@code false} when the request is to be refused with {@code 403}
     */
    static boolean isAllowed(
            String allowedOrigins,
            boolean loopbackConnection,
            @Nullable String host,
            @Nullable String origin,
            @Nullable String secFetchSite) {
        if (ANY.equals(allowedOrigins)) {
            return true;
        }
        final String hostValue = host == null ? "" : host.trim();
        if (loopbackConnection && !isAllowedHost(allowedOrigins, hostValue)) {
            return false;
        }
        if (origin != null && !origin.isEmpty()) {
            return (!hostValue.isEmpty() && hostValue.equalsIgnoreCase(authorityOf(origin)))
                    || isListedOrigin(allowedOrigins, origin);
        }
        return !"cross-site".equals(secFetchSite) && !"same-site".equals(secFetchSite);
    }

    /**
     * The {@code Access-Control-Allow-Origin} value for a response, or {@code null} for none.
     *
     * <p>The header takes one origin (or {@code "*"}), never a list, so a list is answered with the
     * request's own origin when it is on it; {@code "localhost"} likewise names the local origin that asked.
     *
     * @param allowedOrigins the configured value
     * @param origin the request's {@code Origin} header
     * @return the header value, or {@code null} when the response carries none
     */
    static @Nullable String allowOriginHeader(String allowedOrigins, @Nullable String origin) {
        if (ANY.equals(allowedOrigins)) {
            return ANY;
        }
        if (origin == null || origin.isEmpty()) {
            return LOCALHOST.equals(allowedOrigins) || allowedOrigins.indexOf(',') >= 0 ? null : allowedOrigins;
        }
        return isListedOrigin(allowedOrigins, origin) ? origin : null;
    }

    /** Whether {@code origin} is one the configured value names. */
    private static boolean isListedOrigin(String allowedOrigins, String origin) {
        if (LOCALHOST.equals(allowedOrigins)) {
            final String name = nameOf(authorityOf(origin));
            return "localhost".equals(name) || "127.0.0.1".equals(name) || "::1".equals(name);
        }
        for (final String entry : allowedOrigins.split(",", -1)) {
            if (entry.trim().equals(origin)) {
                return true;
            }
        }
        return false;
    }

    /** Whether a {@code Host} on a loopback connection names loopback or the host of an allowed origin. */
    private static boolean isAllowedHost(String allowedOrigins, String host) {
        final String name = nameOf(host);
        if ("localhost".equals(name) || "0.0.0.0".equals(name) || "::".equals(name) || isLoopbackAddress(name)) {
            return true;
        }
        if (LOCALHOST.equals(allowedOrigins)) {
            return false;
        }
        for (final String entry : allowedOrigins.split(",", -1)) {
            final String allowed = entry.trim();
            if (allowed.contains(SCHEME_SEPARATOR) && authorityOf(allowed).equalsIgnoreCase(host)) {
                return true;
            }
        }
        return false;
    }

    /** What follows {@code scheme://} in an origin; empty when it has no scheme. */
    private static String authorityOf(String origin) {
        final int separator = origin.indexOf(SCHEME_SEPARATOR);
        return separator < 0 ? "" : origin.substring(separator + SCHEME_SEPARATOR.length());
    }

    /** The name of a {@code host[:port]} value, lower case, without port and IPv6 brackets. */
    private static String nameOf(String host) {
        if (host.startsWith("[")) {
            final int end = host.indexOf(']');
            return end < 0 ? "" : host.substring(1, end).toLowerCase(Locale.ROOT);
        }
        final int colon = host.indexOf(':');
        return (colon < 0 ? host : host.substring(0, colon)).toLowerCase(Locale.ROOT);
    }

    /** Whether {@code address} is {@code 127.0.0.0/8}, {@code ::1} or an IPv4-mapped {@code 127.0.0.0/8}. */
    private static boolean isLoopbackAddress(String address) {
        if (address.startsWith(IPV4_MAPPED_PREFIX)) {
            return isLoopbackIpv4(address.substring(IPV4_MAPPED_PREFIX.length()));
        }
        return "::1".equals(address) || isLoopbackIpv4(address);
    }

    /** Whether {@code text} is a dotted IPv4 address whose first part is 127. */
    private static boolean isLoopbackIpv4(String text) {
        final String[] parts = text.split("\\.", -1);
        if (parts.length != IPV4_PARTS) {
            return false;
        }
        for (int i = 0; i < parts.length; i++) {
            final int value = decimal(parts[i]);
            if (value < 0 || value > IPV4_PART_MAX || (i == 0 && value != LOOPBACK_FIRST_PART)) {
                return false;
            }
        }
        return true;
    }

    /** The value of one to three decimal digits, or {@code -1} for anything else. */
    private static int decimal(String digits) {
        if (digits.isEmpty() || digits.length() > 3) {
            return -1;
        }
        int value = 0;
        for (int i = 0; i < digits.length(); i++) {
            final char c = digits.charAt(i);
            if (c < '0' || c > '9') {
                return -1;
            }
            value = value * 10 + (c - '0');
        }
        return value;
    }
}
