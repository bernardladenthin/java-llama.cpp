// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.value;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.not;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.Test;

public class RpcEndpointTest {

    @Test
    public void ofKeepsHostAndPort() {
        RpcEndpoint endpoint = RpcEndpoint.of("10.0.0.2", 50052);
        assertThat(endpoint.getHost(), is("10.0.0.2"));
        assertThat(endpoint.getPort(), is(50052));
        assertThat(endpoint.toString(), is("10.0.0.2:50052"));
    }

    @Test
    public void defaultPortIsUpstreamRpcServersDefault() {
        assertThat(RpcEndpoint.DEFAULT_PORT, is(50052));
    }

    @Test
    public void parseAcceptsAnIpv4AddressAndAHostName() {
        assertThat(RpcEndpoint.parse("192.168.1.10:1"), is(RpcEndpoint.of("192.168.1.10", 1)));
        assertThat(RpcEndpoint.parse("gpu-box_2.lan:65535"), is(RpcEndpoint.of("gpu-box_2.lan", 65535)));
    }

    @Test
    public void parseTrimsSurroundingWhitespace() {
        assertThat(RpcEndpoint.parse("  host:7 "), is(RpcEndpoint.of("host", 7)));
    }

    @Test
    public void parseRejectsAMissingPort() {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> RpcEndpoint.parse("host"));
        assertThat(e.getMessage(), containsString("no port"));
    }

    @Test
    public void parseRejectsAnIpv6AddressWithAReason() {
        IllegalArgumentException e =
                assertThrows(IllegalArgumentException.class, () -> RpcEndpoint.parse("[::1]:50052"));
        assertThat(e.getMessage(), containsString("IPv4-only"));
        assertThrows(IllegalArgumentException.class, () -> RpcEndpoint.parse("::1"));
    }

    @Test
    public void parseRejectsAnEmptyOrNonNumericPort() {
        // the message matters: Integer.parseInt would also throw (NumberFormatException is an
        // IllegalArgumentException), so only the message proves the digit check itself ran
        for (String bad : new String[] {"host:", "host:5x", "host:x5", "host:-1", "host: 1", "host:1/", "host:1:"}) {
            IllegalArgumentException e =
                    assertThrows(IllegalArgumentException.class, () -> RpcEndpoint.parse(bad), bad);
            assertThat(bad, e, is(not(org.hamcrest.Matchers.instanceOf(NumberFormatException.class))));
        }
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> RpcEndpoint.parse("host:5x"));
        assertThat(e.getMessage(), containsString("no valid port: '5x'"));
    }

    @Test
    public void everyDigitIsAcceptedInAPort() {
        assertThat(RpcEndpoint.parse("h:1234").getPort(), is(1234));
        assertThat(RpcEndpoint.parse("h:5678").getPort(), is(5678));
        assertThat(RpcEndpoint.parse("h:10").getPort(), is(10));
        assertThat(RpcEndpoint.parse("h:9").getPort(), is(9));
    }

    @Test
    public void parseRejectsAPortWithMoreThanFiveDigitsBeforeParsingIt() {
        // would overflow Integer.parseInt; must be the port message, not a NumberFormatException
        IllegalArgumentException e =
                assertThrows(IllegalArgumentException.class, () -> RpcEndpoint.parse("host:99999999999"));
        assertThat(e.getMessage(), containsString("no valid port"));
        assertThat(e, is(not(org.hamcrest.Matchers.instanceOf(NumberFormatException.class))));
    }

    @Test
    public void portBoundariesAreOneAndSixtyFiveThousandFiveHundredThirtyFive() {
        assertThat(RpcEndpoint.of("h", 1).getPort(), is(1));
        assertThat(RpcEndpoint.of("h", 65535).getPort(), is(65535));
        IllegalArgumentException low = assertThrows(IllegalArgumentException.class, () -> RpcEndpoint.of("h", 0));
        assertThat(low.getMessage(), containsString("outside 1..65535"));
        assertThrows(IllegalArgumentException.class, () -> RpcEndpoint.of("h", 65536));
        assertThrows(IllegalArgumentException.class, () -> RpcEndpoint.parse("h:65536"));
    }

    @Test
    public void emptyHostIsRejected() {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> RpcEndpoint.parse(":1"));
        assertThat(e.getMessage(), containsString("is empty"));
    }

    @Test
    public void hostWithAnInvalidCharacterIsRejected() {
        for (String host : new String[] {"a b", "a,b", "a/b", "a@b", "ä"}) {
            IllegalArgumentException e =
                    assertThrows(IllegalArgumentException.class, () -> RpcEndpoint.of(host, 1), host);
            assertThat(e.getMessage(), containsString("expected an IPv4 address or a host name"));
        }
    }

    @Test
    public void everyAllowedHostCharacterIsAccepted() {
        String host = "abcxyzABCXYZ0189.-_";
        assertThat(RpcEndpoint.of(host, 1).getHost(), is(host));
    }

    @Test
    public void hostLengthBoundary() {
        String longest = repeat('a', RpcEndpoint.MAX_HOST_LENGTH);
        assertThat(RpcEndpoint.of(longest, 1).getHost(), is(longest));
        IllegalArgumentException e =
                assertThrows(IllegalArgumentException.class, () -> RpcEndpoint.of(longest + "a", 1));
        assertThat(e.getMessage(), containsString("more than 253"));
    }

    @Test
    public void parseListKeepsOrderAndSkipsEmptyEntries() {
        List<RpcEndpoint> endpoints = RpcEndpoint.parseList("a:1, b:2,,c:3,");
        assertThat(
                endpoints, is(Arrays.asList(RpcEndpoint.of("a", 1), RpcEndpoint.of("b", 2), RpcEndpoint.of("c", 3))));
    }

    @Test
    public void parseListOfNothingIsRejected() {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> RpcEndpoint.parseList(" , "));
        assertThat(e.getMessage(), containsString("no RPC endpoint in"));
    }

    @Test
    public void parseListIsUnmodifiable() {
        List<RpcEndpoint> endpoints = RpcEndpoint.parseList("a:1");
        assertThrows(UnsupportedOperationException.class, () -> endpoints.add(RpcEndpoint.of("b", 2)));
    }

    @Test
    public void equalityIsByHostAndPort() {
        assertThat(RpcEndpoint.of("a", 1), is(RpcEndpoint.of("a", 1)));
        assertThat(RpcEndpoint.of("a", 1).hashCode(), is(RpcEndpoint.of("a", 1).hashCode()));
        assertThat(RpcEndpoint.of("a", 1), is(not(RpcEndpoint.of("a", 2))));
        assertThat(RpcEndpoint.of("a", 1), is(not(RpcEndpoint.of("b", 1))));
    }

    private static String repeat(char c, int n) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < n; i++) {
            sb.append(c);
        }
        return sb.toString();
    }
}
