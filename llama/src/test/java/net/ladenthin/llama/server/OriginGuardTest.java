// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.nullValue;

import org.junit.jupiter.api.Test;

/**
 * Unit tests for {@link OriginGuard}: which requests a configured origin list serves, and which
 * {@code Access-Control-Allow-Origin} value answers them. Pure — no server, no model. The rules are
 * llama.cpp's ({@code --cors-origins}, upstream #32098); {@code test_native_server_cors.cpp} holds the
 * same cases against the embedded server.
 */
public class OriginGuardTest {

    private static final String LIST = "http://app.example, https://other.example:8443";

    private static boolean loopback(String allowed, String host, String origin, String site) {
        return OriginGuard.isAllowed(allowed, true, host, origin, site);
    }

    @Test
    public void anyOriginChecksNothing() {
        assertThat(loopback("*", "evil.example", "http://evil.example", "cross-site"), is(true));
        assertThat(OriginGuard.isAllowed("*", false, null, null, null), is(true));
    }

    @Test
    public void aLoopbackConnectionNeedsAHostThatNamesLoopback() {
        assertThat(loopback(LIST, "127.0.0.1:9931", null, null), is(true));
        assertThat(loopback(LIST, "localhost:9931", null, null), is(true));
        assertThat(loopback(LIST, "LOCALHOST", null, null), is(true));
        assertThat(loopback(LIST, "127.8.9.10", null, null), is(true));
        assertThat(loopback(LIST, "[::1]:9931", null, null), is(true));
        assertThat(loopback(LIST, "[::ffff:127.0.0.1]:9931", null, null), is(true));
        assertThat(loopback(LIST, "0.0.0.0:9931", null, null), is(true));
        assertThat(loopback(LIST, "[::]:9931", null, null), is(true));
    }

    @Test
    public void aLoopbackConnectionRefusesAForeignHost() {
        assertThat(loopback(LIST, "evil.example:9931", null, null), is(false));
        assertThat(loopback(LIST, "128.0.0.1", null, null), is(false));
        assertThat(loopback(LIST, "127.0.0.256", null, null), is(false));
        assertThat(loopback(LIST, "127.0.0", null, null), is(false));
        assertThat(loopback(LIST, "127.0.0.1.evil.example", null, null), is(false));
        assertThat(loopback(LIST, "127.0.0.x", null, null), is(false));
        assertThat(loopback(LIST, "127.0.0.0001", null, null), is(false));
        assertThat(loopback(LIST, "127..0.1", null, null), is(false));
        assertThat(loopback(LIST, "[::1", null, null), is(false));
        assertThat(loopback(LIST, "[::2]", null, null), is(false));
        assertThat(loopback(LIST, "[::ffff:10.0.0.1]", null, null), is(false));
        assertThat(loopback(LIST, "", null, null), is(false));
        assertThat(loopback(LIST, null, null, null), is(false));
    }

    @Test
    public void theHostOfAnAllowedOriginMayNameTheServer() {
        assertThat(loopback(LIST, "app.example", null, null), is(true));
        assertThat(loopback(LIST, "APP.example", null, null), is(true));
        assertThat(loopback(LIST, "other.example:8443", null, null), is(true));
        // the port is part of what an origin names
        assertThat(loopback(LIST, "other.example", null, null), is(false));
        assertThat(loopback(LIST, "app.example:9931", null, null), is(false));
        // "localhost" names no host besides loopback
        assertThat(loopback("localhost", "app.example", null, null), is(false));
        // an entry without a scheme is no origin
        assertThat(loopback("app.example", "app.example", null, null), is(false));
    }

    @Test
    public void aConnectionThatIsNotLoopbackIsNotCheckedForItsHost() {
        assertThat(OriginGuard.isAllowed(LIST, false, "box.lan:9931", null, null), is(true));
        assertThat(OriginGuard.isAllowed("localhost", false, "box.lan:9931", null, null), is(true));
        // its Origin still is
        assertThat(OriginGuard.isAllowed(LIST, false, "box.lan:9931", "http://evil.example", null), is(false));
    }

    @Test
    public void anOriginIsServedWhenItIsListed() {
        assertThat(loopback(LIST, "127.0.0.1", "http://app.example", null), is(true));
        assertThat(loopback(LIST, "127.0.0.1", "https://other.example:8443", null), is(true));
        assertThat(loopback(LIST, "127.0.0.1", "http://evil.example", null), is(false));
        // scheme, port and case are part of an origin
        assertThat(loopback(LIST, "127.0.0.1", "https://app.example", null), is(false));
        assertThat(loopback(LIST, "127.0.0.1", "http://app.example:81", null), is(false));
        assertThat(loopback(LIST, "127.0.0.1", "http://APP.example", null), is(false));
        assertThat(loopback(LIST, "127.0.0.1", "null", null), is(false));
    }

    @Test
    public void anOriginThatIsTheRequestsOwnHostIsServed() {
        assertThat(OriginGuard.isAllowed(LIST, false, "box.lan:9931", "http://box.lan:9931", null), is(true));
        assertThat(OriginGuard.isAllowed(LIST, false, "BOX.lan:9931", "http://box.lan:9931", null), is(true));
        assertThat(loopback(LIST, "127.0.0.1:9931", "http://127.0.0.1:9931", null), is(true));
        assertThat(OriginGuard.isAllowed(LIST, false, "box.lan:9931", "http://box.lan:1", null), is(false));
        // no Host, no same origin: an origin without a scheme has an empty authority too
        assertThat(OriginGuard.isAllowed(LIST, false, null, "box.lan", null), is(false));
        assertThat(OriginGuard.isAllowed(LIST, false, "", "box.lan", null), is(false));
    }

    @Test
    public void localhostServesEveryLocalOrigin() {
        assertThat(loopback("localhost", "127.0.0.1", "http://localhost:5173", null), is(true));
        assertThat(loopback("localhost", "127.0.0.1", "https://127.0.0.1", null), is(true));
        assertThat(loopback("localhost", "127.0.0.1", "http://[::1]:3000", null), is(true));
        assertThat(loopback("localhost", "127.0.0.1", "http://LocalHost", null), is(true));
        assertThat(loopback("localhost", "127.0.0.1", "http://evil.example", null), is(false));
        assertThat(loopback("localhost", "127.0.0.1", "http://localhost.evil.example", null), is(false));
        assertThat(loopback("localhost", "127.0.0.1", "localhost", null), is(false));
    }

    @Test
    public void aRequestWithoutAnOriginIsServedUnlessABrowserSaysItIsForeign() {
        assertThat(loopback(LIST, "127.0.0.1", null, null), is(true));
        assertThat(loopback(LIST, "127.0.0.1", "", null), is(true));
        assertThat(loopback(LIST, "127.0.0.1", null, "same-origin"), is(true));
        assertThat(loopback(LIST, "127.0.0.1", null, "none"), is(true));
        assertThat(loopback(LIST, "127.0.0.1", null, "cross-site"), is(false));
        assertThat(loopback(LIST, "127.0.0.1", null, "same-site"), is(false));
        // with an Origin, the Origin decides
        assertThat(loopback(LIST, "127.0.0.1", "http://app.example", "cross-site"), is(true));
    }

    @Test
    public void theAllowOriginHeaderNamesOneOriginOrNone() {
        assertThat(OriginGuard.allowOriginHeader("*", null), is("*"));
        assertThat(OriginGuard.allowOriginHeader("*", "http://evil.example"), is("*"));
        // one configured origin is the header's value, as before the guard existed
        assertThat(OriginGuard.allowOriginHeader("http://app.example", null), is("http://app.example"));
        assertThat(OriginGuard.allowOriginHeader("http://app.example", ""), is("http://app.example"));
        assertThat(OriginGuard.allowOriginHeader("http://app.example", "http://app.example"), is("http://app.example"));
        assertThat(OriginGuard.allowOriginHeader("http://app.example", "http://evil.example"), is(nullValue()));
        // a list is not a header value: the origin that asked, when it is on it
        assertThat(OriginGuard.allowOriginHeader(LIST, null), is(nullValue()));
        assertThat(OriginGuard.allowOriginHeader(LIST, "https://other.example:8443"), is("https://other.example:8443"));
        assertThat(OriginGuard.allowOriginHeader(LIST, "http://evil.example"), is(nullValue()));
        assertThat(OriginGuard.allowOriginHeader("localhost", null), is(nullValue()));
        assertThat(OriginGuard.allowOriginHeader("localhost", "http://localhost:5173"), is("http://localhost:5173"));
        assertThat(OriginGuard.allowOriginHeader("localhost", "http://evil.example"), is(nullValue()));
    }
}
