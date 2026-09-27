// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.atmosphere;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.is;

import java.util.List;
import org.junit.jupiter.api.Test;

class ServerPropsTest {

    @Test
    void propsIsTriedNextToTheV1RoutesFirst() {
        // llama-server serves /props at the root; this project's server serves it under both.
        assertThat(
                ServerProps.candidates("http://127.0.0.1:8080/v1"),
                is(List.of("http://127.0.0.1:8080/props", "http://127.0.0.1:8080/v1/props")));
        assertThat(
                ServerProps.candidates("http://127.0.0.1:8080/v1/"),
                is(List.of("http://127.0.0.1:8080/props", "http://127.0.0.1:8080/v1/props")));
        assertThat(ServerProps.candidates("http://host/api"), is(List.of("http://host/api/props")));
    }

    @Test
    void theContextSizeIsReadFromTheGenerationDefaults() {
        assertThat(
                ServerProps.parseContextSize("{\"default_generation_settings\":{\"n_ctx\":16384,\"model\":\"m\"}}"),
                is(16384));
    }

    @Test
    void anythingElseIsUnknownRatherThanAGuess() {
        assertThat(ServerProps.parseContextSize("{}"), is(StatusLine.UNKNOWN_CONTEXT));
        assertThat(ServerProps.parseContextSize("not json"), is(StatusLine.UNKNOWN_CONTEXT));
        assertThat(ServerProps.parseContextSize("{\"n_ctx\":\"many\"}"), is(StatusLine.UNKNOWN_CONTEXT));
    }

    @Test
    void anUnreachableServerIsUnknownAndDoesNotThrow() {
        assertThat(ServerProps.contextSize("http://127.0.0.1:1/v1", "k"), is(StatusLine.UNKNOWN_CONTEXT));
    }
}
