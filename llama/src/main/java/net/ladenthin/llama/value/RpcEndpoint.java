// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.value;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import lombok.EqualsAndHashCode;

/**
 * The address of a llama.cpp RPC server: {@code host:port}.
 *
 * <p>Validated here, before it reaches native code, because llama.cpp's own parser is minimal: it
 * splits at the first {@code ':'} and resolves the host as IPv4 only. An IPv6 literal would be cut
 * in the middle and a missing port would reach the socket layer as garbage, so both are rejected
 * with a message instead. Accepted hosts are an IPv4 address or a host name.
 *
 * <p>The protocol has no authentication and no encryption; see {@link #DEFAULT_PORT} and the
 * README section "Distributed inference over RPC" before pointing it at anything but a trusted
 * network.
 */
@EqualsAndHashCode
public final class RpcEndpoint {

    /** The port upstream's {@code rpc-server} and {@code RpcServer} listen on by default. */
    public static final int DEFAULT_PORT = 50052;

    /** A host name may not be longer than this (RFC 1035). */
    static final int MAX_HOST_LENGTH = 253;

    private final String host;
    private final int port;

    private RpcEndpoint(String host, int port) {
        this.host = host;
        this.port = port;
    }

    /**
     * Creates an endpoint from its parts.
     *
     * @param host an IPv4 address or a host name
     * @param port a TCP port, 1 to 65535
     * @return the endpoint
     * @throws IllegalArgumentException when the host or the port is not valid
     */
    public static RpcEndpoint of(String host, int port) {
        return new RpcEndpoint(validateHost(host), validatePort(port));
    }

    /**
     * Parses {@code host:port}.
     *
     * @param endpoint the endpoint text
     * @return the endpoint
     * @throws IllegalArgumentException when the text is not {@code host:port}
     */
    public static RpcEndpoint parse(String endpoint) {
        String text = endpoint.trim();
        int colon = text.indexOf(':');
        if (colon < 0) {
            throw new IllegalArgumentException("RPC endpoint '" + endpoint + "' has no port; expected host:port");
        }
        if (text.indexOf(':', colon + 1) >= 0) {
            throw new IllegalArgumentException("RPC endpoint '" + endpoint
                    + "' contains more than one ':'; llama.cpp's RPC transport is IPv4-only, so an IPv6"
                    + " address cannot be used (expected host:port)");
        }
        String portText = text.substring(colon + 1);
        if (portText.isEmpty() || portText.length() > 5 || !isDigits(portText)) {
            throw new IllegalArgumentException("RPC endpoint '" + endpoint + "' has no valid port: '" + portText + "'");
        }
        return of(text.substring(0, colon), Integer.parseInt(portText));
    }

    /**
     * Parses a comma-separated list, the form {@code --rpc} takes on the command line.
     *
     * @param endpoints e.g. {@code "10.0.0.2:50052,10.0.0.3:50052"}
     * @return the endpoints, in order
     * @throws IllegalArgumentException when the list is empty or an entry is not valid
     */
    public static List<RpcEndpoint> parseList(String endpoints) {
        List<RpcEndpoint> out = new ArrayList<>();
        for (String part : endpoints.split(",", -1)) {
            if (!part.trim().isEmpty()) {
                out.add(parse(part));
            }
        }
        if (out.isEmpty()) {
            throw new IllegalArgumentException(
                    "no RPC endpoint in '" + endpoints + "'; expected host:port[,host:port...]");
        }
        return Collections.unmodifiableList(out);
    }

    /**
     * The host part.
     *
     * @return an IPv4 address or a host name
     */
    public String getHost() {
        return host;
    }

    /**
     * The port part.
     *
     * @return a TCP port, 1 to 65535
     */
    public int getPort() {
        return port;
    }

    /**
     * The endpoint as llama.cpp expects it.
     *
     * @return {@code host:port}
     */
    @Override
    public String toString() {
        return host + ":" + port;
    }

    private static String validateHost(String host) {
        int length = host.length();
        if (length == 0) {
            throw new IllegalArgumentException("RPC endpoint host '" + host + "' is empty");
        }
        if (length > MAX_HOST_LENGTH) {
            throw new IllegalArgumentException(
                    "RPC endpoint host is " + length + " characters long, more than " + MAX_HOST_LENGTH);
        }
        for (int i = 0; i < length; i++) {
            char c = host.charAt(i);
            boolean allowed = (c >= 'a' && c <= 'z')
                    || (c >= 'A' && c <= 'Z')
                    || (c >= '0' && c <= '9')
                    || c == '.'
                    || c == '-'
                    || c == '_';
            if (!allowed) {
                throw new IllegalArgumentException("RPC endpoint host '" + host + "' contains '" + c
                        + "'; expected an IPv4 address or a host name");
            }
        }
        return host;
    }

    private static int validatePort(int port) {
        if (port < 1 || port > 65535) {
            throw new IllegalArgumentException("RPC endpoint port " + port + " is outside 1..65535");
        }
        return port;
    }

    private static boolean isDigits(String text) {
        for (int i = 0; i < text.length(); i++) {
            char c = text.charAt(i);
            if (c < '0' || c > '9') {
                return false;
            }
        }
        return true;
    }
}
