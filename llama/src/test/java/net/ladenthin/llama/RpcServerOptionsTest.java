// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.greaterThanOrEqualTo;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.nullValue;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.nio.file.Paths;
import net.ladenthin.llama.value.RpcEndpoint;
import org.junit.jupiter.api.Test;

/**
 * The command line and the bind-address rule of {@link RpcServer}. Model-free and native-free:
 * {@link RpcServer.Options} is a nested class of its own, so none of this loads libjllama.
 */
public class RpcServerOptionsTest {

    @Test
    public void defaultsMatchUpstreamRpcServer() {
        RpcServer.Options options = RpcServer.Options.parse(new String[0]);
        assertThat(options.host, is("127.0.0.1"));
        assertThat(options.port, is(RpcEndpoint.DEFAULT_PORT));
        assertThat(options.threads, is(RpcServer.Options.defaultThreads()));
        assertThat(options.cacheDir, is(nullValue()));
        assertThat(options.help, is(false));
    }

    @Test
    public void defaultThreadsIsHalfTheProcessorsAndAtLeastOne() {
        int expected = Math.max(1, Runtime.getRuntime().availableProcessors() / 2);
        assertThat(RpcServer.Options.defaultThreads(), is(expected));
        assertThat(RpcServer.Options.defaultThreads(), greaterThanOrEqualTo(1));
    }

    @Test
    public void everyOptionInBothSpellings() {
        RpcServer.Options longForm = RpcServer.Options.parse(
                new String[] {"--host", "0.0.0.0", "--port", "6000", "--threads", "3", "--cache", "c"});
        assertThat(longForm.host, is("0.0.0.0"));
        assertThat(longForm.port, is(6000));
        assertThat(longForm.threads, is(3));
        assertThat(longForm.cacheDir, is(Paths.get("c")));

        RpcServer.Options shortForm =
                RpcServer.Options.parse(new String[] {"-H", "10.1.2.3", "-p", "7", "-t", "1", "-c", "d"});
        assertThat(shortForm.host, is("10.1.2.3"));
        assertThat(shortForm.port, is(7));
        assertThat(shortForm.threads, is(1));
        assertThat(shortForm.cacheDir, is(Paths.get("d")));
    }

    @Test
    public void helpInBothSpellings() {
        assertThat(RpcServer.Options.parse(new String[] {"-h"}).help, is(true));
        assertThat(RpcServer.Options.parse(new String[] {"--help"}).help, is(true));
    }

    @Test
    public void unknownArgumentNamesItAndShowsTheUsage() {
        IllegalArgumentException e =
                assertThrows(IllegalArgumentException.class, () -> RpcServer.Options.parse(new String[] {"--bogus"}));
        assertThat(e.getMessage(), containsString("--bogus"));
        assertThat(e.getMessage(), containsString(RpcServer.Options.USAGE));
    }

    @Test
    public void aFlagWithoutItsValueIsRejected() {
        IllegalArgumentException e =
                assertThrows(IllegalArgumentException.class, () -> RpcServer.Options.parse(new String[] {"--port"}));
        assertThat(e.getMessage(), containsString("--port needs a value"));
    }

    @Test
    public void aNonNumericNumberIsRejected() {
        IllegalArgumentException e = assertThrows(
                IllegalArgumentException.class, () -> RpcServer.Options.parse(new String[] {"--threads", "many"}));
        assertThat(e.getMessage(), containsString("--threads expects a number"));
    }

    @Test
    public void aHostNameIsRejectedAsBindAddress() {
        // the native server binds with inet_addr, which knows no names
        IllegalArgumentException e = assertThrows(
                IllegalArgumentException.class, () -> RpcServer.Options.parse(new String[] {"--host", "localhost"}));
        assertThat(e.getMessage(), containsString("not an IPv4 literal"));
    }

    @Test
    public void ipv4LiteralRule() {
        assertDoesNotThrow(() -> RpcServer.Options.requireIpv4Literal("0.0.0.0"));
        assertDoesNotThrow(() -> RpcServer.Options.requireIpv4Literal("255.255.255.255"));
        for (String bad :
                new String[] {"256.0.0.1", "1.2.3", "1.2.3.4.5", "1..3.4", "a.b.c.d", "1.2.3.1000", "", "::1"}) {
            assertThrows(IllegalArgumentException.class, () -> RpcServer.Options.requireIpv4Literal(bad), bad);
        }
    }
}
