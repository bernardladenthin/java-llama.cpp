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
import static org.junit.jupiter.api.Assertions.assertTrue;

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
    void rpcServerDevicesResolveTheCpuAndRefuseAnUnknownName() {
        assertThat(FfmRpcServer.devices("CPU"), hasItem("CPU"));
        LlamaException e = assertThrows(LlamaException.class, () -> FfmRpcServer.devices("NoSuchDevice9"));
        assertThat(e.getMessage(), containsString("CPU"));
    }
}
