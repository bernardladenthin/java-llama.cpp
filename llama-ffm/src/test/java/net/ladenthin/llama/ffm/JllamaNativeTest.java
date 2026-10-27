// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT
package net.ladenthin.llama.ffm;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.hasItem;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.not;
import static org.hamcrest.Matchers.startsWith;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_LONG;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import net.ladenthin.llama.exception.LlamaException;
import net.ladenthin.llama.value.LlamaCppVersion;
import net.ladenthin.llama.value.LogLevel;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/**
 * The model-free half of the C API through FFM: the library loads through the core's loader, the
 * symbols resolve, the version and build facts match the build, a failure arrives as the core's
 * exception with the C API's message, and the log callback is reached from the logger's worker.
 */
class JllamaNativeTest {

    @BeforeAll
    static void nativesLoadable() {
        FfmTestSupport.assumeNativesLoadable();
    }

    /**
     * {@code jllama_rpc_server_devices} allocates an array for zero entries too; returning early for
     * {@code count == 0} never freed it (measured: a deliberately invalid pointer that glibc's
     * {@code free()} aborts on passed through without a call).
     */
    @Test
    void takeStringsFreesANonNullArrayEvenWhenItIsEmpty() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment array = arena.allocate(ADDRESS);
            MemorySegment out = JllamaNative.pointerSlot(arena);
            out.set(ADDRESS, 0, array);
            MemorySegment length = arena.allocate(JAVA_LONG);
            length.set(JAVA_LONG, 0, 0L);
            List<String> freed = new ArrayList<>();

            List<String> strings = JllamaNative.takeStrings(
                    out, length, (pointer, count) -> freed.add(pointer.address() + "/" + count));

            assertThat(strings, is(List.of()));
            assertThat(freed, is(List.of(array.address() + "/0")));
        }
    }

    /**
     * A populated array is read entry by entry in order and freed with its element count: a wrong
     * count only leaks the element strings inside {@code jllama_strings_free}, which the JVM cannot
     * see, and the one native caller with a test returns a single entry, which hides the order.
     */
    @Test
    void takeStringsReadsEveryEntryInOrderAndFreesTheArrayWithItsCount() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment array = arena.allocate(ADDRESS, 2);
            array.setAtIndex(ADDRESS, 0, arena.allocateFrom("alpha", StandardCharsets.UTF_8));
            array.setAtIndex(ADDRESS, 1, arena.allocateFrom("beta", StandardCharsets.UTF_8));
            MemorySegment out = JllamaNative.pointerSlot(arena);
            out.set(ADDRESS, 0, array);
            MemorySegment length = arena.allocate(JAVA_LONG);
            length.set(JAVA_LONG, 0, 2L);
            List<String> freed = new ArrayList<>();

            List<String> strings = JllamaNative.takeStrings(
                    out, length, (pointer, count) -> freed.add(pointer.address() + "/" + count));

            assertThat(strings, is(List.of("alpha", "beta")));
            assertThat(freed, is(List.of(array.address() + "/2")));
        }
    }

    @Test
    void takeStringsDoesNotFreeANullArray() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment out = JllamaNative.pointerSlot(arena);
            out.set(ADDRESS, 0, MemorySegment.NULL);
            MemorySegment length = arena.allocate(JAVA_LONG);
            length.set(JAVA_LONG, 0, 0L);
            List<String> freed = new ArrayList<>();

            List<String> strings = JllamaNative.takeStrings(out, length, (pointer, count) -> freed.add("freed"));

            assertThat(strings, is(List.of()));
            assertThat(freed, is(List.of()));
        }
    }

    /**
     * llama/patches/0022's switch through the FFM binding: it reaches the CPU backend's entry points
     * (the same process-wide switch as {@code LlamaModel.setCpuMoeBatched}) and round-trips.
     */
    @Test
    void cpuMoeBatchedSwitchRoundTrips() {
        boolean initial = Jllama.isCpuMoeBatched();
        try {
            Jllama.setCpuMoeBatched(false);
            assertThat(Jllama.isCpuMoeBatched(), is(false));
            Jllama.setCpuMoeBatched(true);
            assertThat(Jllama.isCpuMoeBatched(), is(true));
        } finally {
            Jllama.setCpuMoeBatched(initial);
        }
    }

    @Test
    void apiVersionIsTheVersionThisModuleWasBuiltAs() {
        String expected = System.getProperty("net.ladenthin.llama.ffm.expected.version");
        assertTrue(expected != null && !expected.isEmpty(), "surefire passes the pom version");
        assertEquals(expected, Jllama.apiVersion());
    }

    @Test
    void buildInfoNamesThePinnedLlamaCppTag() {
        assertThat(Jllama.buildInfo(), startsWith(LlamaCppVersion.LLAMA_CPP_VERSION + "-"));
    }

    @Test
    void jsonSchemaToGrammarRendersARootRule() {
        String grammar = Jllama.jsonSchemaToGrammar("{\"type\": \"object\"}");
        assertThat(grammar, containsString("root ::="));
    }

    @Test
    void aModelThatDoesNotExistFailsWithTheLibrarysMessage() {
        LlamaException e = assertThrows(LlamaException.class,
                () -> FfmLlamaModel.load(new String[] {"jllama", "-m", "/nonexistent/model.gguf", "--no-warmup"}, null));
        assertThat(e.getMessage(), not(is("")));
    }

    @Test
    void logCallbackSeesTheLinesOfAFailingLoadAndNullRestoresTheConsole() {
        List<String> lines = new CopyOnWriteArrayList<>();
        Jllama.setLogCallback((level, text) -> lines.add(level + " " + text));
        try {
            assertThrows(LlamaException.class, () -> FfmLlamaModel.load(
                    new String[] {"jllama", "-m", "/nonexistent/model.gguf", "--no-warmup"}, null));
        } finally {
            Jllama.setLogCallback(null); // flushes synchronously before returning
        }
        assertThat(lines, not(org.hamcrest.Matchers.empty()));
        assertThat(lines.toString(), containsString(LogLevel.ERROR.name()));
    }

    @Test
    void anErrorThrownByTheLogCallbackLosesTheLineAndNothingElse() {
        // An Error crossing an FFM upcall would end the JVM; the bridge swallows it like a RuntimeException.
        List<String> lines = new CopyOnWriteArrayList<>();
        Jllama.setLogCallback((level, text) -> {
            lines.add(text);
            throw new AssertionError("from the callback");
        });
        try {
            assertThrows(LlamaException.class, () -> FfmLlamaModel.load(
                    new String[] {"jllama", "-m", "/nonexistent/model.gguf", "--no-warmup"}, null));
        } finally {
            Jllama.setLogCallback(null);
        }
        assertThat(lines, not(org.hamcrest.Matchers.empty()));
    }

    @Test
    void rpcServerDevicesResolveTheCpuAndRefuseAnUnknownName() {
        assertThat(FfmRpcServer.devices("CPU"), hasItem("CPU"));
        LlamaException e = assertThrows(LlamaException.class, () -> FfmRpcServer.devices("NoSuchDevice9"));
        assertThat(e.getMessage(), containsString("CPU"));
    }

    /**
     * The speaker-reference query resolves and reaches the library without a TTS model: a null
     * handle is the caller's mistake, refused before the out-slot is written. The answer for a real
     * pair is FfmTextToSpeechIntegrationTest's.
     */
    @Test
    void ttsSupportsSpeakerReferenceRefusesANullHandle() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment supported = JllamaNative.booleanSlot(arena);
            IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                    () -> JllamaNative.call(JllamaNative.TTS_SUPPORTS_SPEAKER_REFERENCE, MemorySegment.NULL, supported));
            assertThat(e.getMessage(), containsString("TextToSpeech handle is null"));
        }
    }
}
