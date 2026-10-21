// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.atmosphere;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.is;

import com.fasterxml.jackson.databind.ObjectMapper;
import java.util.List;
import org.junit.jupiter.api.Test;

/**
 * The scripted backend's own chunks must be JSON, which is a property the tests driving it cannot see:
 * an invalid chunk reaches the client as a dropped stream, so the turn simply produces nothing and the
 * assertion that fails is about the answer, several layers away from the cause.
 *
 * <p>That is what happened. {@code quote} escaped only {@code "}, {@code \} and {@code \n}, so a text
 * carrying {@code System.lineSeparator()} put a raw CR inside the JSON string on Windows --
 * {@code Illegal unquoted character ((CTRL-CHAR, code 13))}, behind the client's
 * {@code HTTP/1.1 header parser received no bytes} --
 * and {@link AgentSessionTest#aLoopRunsItsStepsThroughTheFrontEndUntilTheMarker()} failed
 * deterministically with a stalled loop. The three agent jobs run on Linux, where the separator is a
 * bare LF, so CI never saw it.
 */
class ScriptedBackendTest {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    private static void assertEveryChunkIsJson(String what, List<String> chunks) {
        assertThat("a turn is at least role + text + finish", chunks.size() >= 3, is(true));
        for (String chunk : chunks) {
            try {
                MAPPER.readTree(chunk);
            } catch (Exception e) {
                throw new AssertionError(what + ": chunk is not JSON: " + chunk, e);
            }
        }
    }

    /** The text a turn's chunks carry: the {@code delta.content} of every chunk, in order. */
    private static String contentOf(List<String> chunks) throws Exception {
        StringBuilder content = new StringBuilder();
        for (String chunk : chunks) {
            var delta = MAPPER.readTree(chunk).path("choices").path(0).path("delta");
            if (delta.path("content").isTextual()) {
                content.append(delta.path("content").asText());
            }
        }
        return content.toString();
    }

    @Test
    void aTextTurnCarryingTheLineSeparatorIsJsonOnEveryPlatform() {
        assertEveryChunkIsJson(
                "System.lineSeparator()",
                ScriptedBackend.textTurn("working" + System.lineSeparator() + TaskLoop.SENTINEL));
        assertEveryChunkIsJson("CRLF", ScriptedBackend.textTurn("working\r\n" + TaskLoop.SENTINEL));
        assertEveryChunkIsJson("LF", ScriptedBackend.textTurn("working\n" + TaskLoop.SENTINEL));
        assertEveryChunkIsJson("bare CR", ScriptedBackend.textTurn("working\r" + TaskLoop.SENTINEL));
    }

    /**
     * The default branch of {@code quote}, not the named cases: a control character nobody listed is
     * what reopened the gap, so every one of them is checked rather than the four with a short escape.
     * Valid JSON is not enough -- any hexadecimal or short escape parses -- so each must also come back
     * as itself: a wrong code point, or a CR written with the escape of an LF, parses just as well.
     */
    @Test
    void everyControlCharacterSurvivesAsJson() throws Exception {
        for (int c = 0; c < 0x20; c++) {
            String what = "U+" + String.format("%04X", c);
            String text = "a" + (char) c + "b";
            List<String> chunks = ScriptedBackend.textTurn(text);
            assertEveryChunkIsJson(what, chunks);
            assertThat(what, contentOf(chunks), is(text));
        }
        String quoted = "say \"hi\"\\ now";
        assertEveryChunkIsJson("quote and backslash", ScriptedBackend.textTurn(quoted));
        assertThat("quote and backslash", contentOf(ScriptedBackend.textTurn(quoted)), is(quoted));
        String beyond = "a\u007fb\u0085c d";
        assertEveryChunkIsJson("DEL and beyond", ScriptedBackend.textTurn(beyond));
        assertThat("DEL and beyond", contentOf(ScriptedBackend.textTurn(beyond)), is(beyond));
    }

    /** A chunk must also carry the text back unchanged -- escaping may not lose or alter it. */
    @Test
    void theTextComesBackOutOfTheChunkUnchanged() throws Exception {
        String text = "working" + System.lineSeparator() + "\tdone \"quoted\" \\ U+0001:\u0001";
        assertThat(contentOf(ScriptedBackend.textTurn(text)), is(text));
    }
}
