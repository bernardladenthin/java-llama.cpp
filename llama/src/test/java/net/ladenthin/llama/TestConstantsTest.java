// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

@ClaudeGenerated(
        purpose = "Pin TestConstants.resolveModelPath — the fix for model-gated tests silently "
                + "self-skipping in CI because Surefire's working directory is the llama/ module "
                + "while the shared GGUF cache is restored to the reactor root.")
public class TestConstantsTest {

    @Test
    public void nullAndEmptyPassThrough() {
        assertNull(TestConstants.resolveModelPath(null));
        assertEquals("", TestConstants.resolveModelPath(""));
    }

    @Test
    public void absolutePathIsReturnedUnchanged(@TempDir Path tempDir) throws Exception {
        Path file = Files.createFile(tempDir.resolve("model.gguf"));
        String absolute = file.toAbsolutePath().toString();
        assertEquals(absolute, TestConstants.resolveModelPath(absolute));
    }

    @Test
    public void existingRelativePathIsReturnedUnchanged() {
        // pom.xml exists relative to the module basedir, which is Surefire's working directory.
        assertTrue(new File("pom.xml").exists(), "precondition: the working directory is the module basedir");
        assertEquals("pom.xml", TestConstants.resolveModelPath("pom.xml"));
    }

    @Test
    public void pathOnlyPresentInTheParentResolvesToAnExistingAbsolutePath() {
        // The reactor root holds the aggregator pom; the module directory does not hold a
        // "../pom.xml"-shaped sibling of that name, so this is exactly the models/ situation.
        String resolved = TestConstants.resolveModelPath("llama/pom.xml");
        assertTrue(
                new File(resolved).exists(), "a path that only exists from the reactor root must resolve: " + resolved);
        assertTrue(Paths.get(resolved).isAbsolute(), "resolved parent-relative paths are absolute: " + resolved);
    }

    @Test
    public void unresolvablePathIsReturnedUnchangedSoSkipMessagesStayReadable() {
        String missing = "models/definitely-not-present-" + TestConstantsTest.class.getSimpleName() + ".gguf";
        assertEquals(missing, TestConstants.resolveModelPath(missing));
    }

    @Test
    public void propertyLookupResolvesAndHonoursTheDefault() {
        String key = "net.ladenthin.llama.test.resolver.probe";
        assertNull(System.getProperty(key), "precondition: probe property must be unset");
        assertNull(TestConstants.resolveModelProperty(key));
        assertEquals("pom.xml", TestConstants.resolveModelProperty(key, "pom.xml"));

        System.setProperty(key, "llama/pom.xml");
        try {
            assertTrue(new File(TestConstants.resolveModelProperty(key)).exists());
        } finally {
            System.clearProperty(key);
        }
    }

    @Test
    public void theShippedModelConstantsGoThroughTheResolver() {
        // Guards the wiring itself: if a future edit drops the resolveModelPath(...) wrapper from a
        // constant, that constant stops matching its resolved literal and every test gated on it
        // silently self-skips again in CI. Holds in both layouts — with the GGUF present the
        // resolver returns the same absolute path on both sides, without it the same literal.
        assertEquals(TestConstants.resolveModelPath("models/codellama-7b.Q2_K.gguf"), TestConstants.MODEL_PATH);
        assertEquals(
                TestConstants.resolveModelPath("models/AMD-Llama-135m-code.Q2_K.gguf"), TestConstants.DRAFT_MODEL_PATH);
        assertEquals(
                TestConstants.resolveModelPath("models/Qwen3-0.6B-Q4_K_M.gguf"), TestConstants.REASONING_MODEL_PATH);
        assertEquals(
                TestConstants.resolveModelPath("models/jina-reranker-v1-tiny-en-Q4_0.gguf"),
                TestConstants.RERANKING_MODEL_PATH);
        assertEquals(
                TestConstants.resolveModelPath("models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"),
                TestConstants.DEFAULT_TOOL_MODEL_PATH);
        assertEquals(
                TestConstants.resolveModelPath("src/test/resources/images/test-image.jpg"),
                TestConstants.DEFAULT_VISION_IMAGE_PATH);
        assertEquals(
                TestConstants.resolveModelPath("src/test/resources/audios/sample.wav"),
                TestConstants.DEFAULT_AUDIO_INPUT_PATH);
    }
}
