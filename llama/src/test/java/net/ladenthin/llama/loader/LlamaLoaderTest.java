// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.loader;

import static org.junit.jupiter.api.Assertions.*;

import java.io.BufferedInputStream;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.attribute.FileTime;
import java.util.jar.JarOutputStream;
import java.util.zip.CRC32;
import java.util.zip.ZipEntry;
import net.ladenthin.llama.ClaudeGenerated;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

@ClaudeGenerated(
        purpose =
                "Verify the helper statics extracted from LlamaLoader without requiring any "
                        + "native library: shouldCleanPath detects jllama/llama/ggml-prefixed files for "
                        + "cleanup (ggml covers the extracted macOS ggml-metal.metal); "
                        + "contentsEquals performs a correct byte-level stream comparison "
                        + "including BufferedInputStream wrapping and length mismatches; getTempDir "
                        + "honours the 'net.ladenthin.llama.tmpdir' system-property override; and "
                        + "getNativeResourcePath produces the expected classpath resource prefix; parseExtras reads a "
                        + "backend extras file; BACKEND_PRIORITY ends with msvc then cpu; and "
                        + "resourceMatchesFile compares a classpath resource to an on-disk file byte-for-byte; extractFile extracts a resource, reuses an already-identical copy without rewriting it, and replaces one whose content differs; and extractionKey derives the per-build key of the extraction directory from a jar entry's CRC and size, a file's size and mtime, or the URL.")
public class LlamaLoaderTest {

    private static final String TMPDIR_PROP = LlamaSystemProperties.PREFIX + ".tmpdir";

    /** A small file present on the test classpath, used as a byte-comparison fixture. */
    private static final String EXISTING_TEST_RESOURCE = "images/test-image.jpg";

    private String previousTmpDir;

    @BeforeEach
    public void saveTmpDirProp() {
        previousTmpDir = System.getProperty(TMPDIR_PROP);
    }

    @AfterEach
    public void restoreTmpDirProp() {
        if (previousTmpDir == null) {
            System.clearProperty(TMPDIR_PROP);
        } else {
            System.setProperty(TMPDIR_PROP, previousTmpDir);
        }
    }

    // -------------------------------------------------------------------------
    // shouldCleanPath
    // -------------------------------------------------------------------------

    @Test
    public void testShouldCleanPathJllamaPrefix() {
        assertTrue(LlamaLoader.shouldCleanPath(Paths.get("/tmp/jllama.so")));
    }

    @Test
    public void testShouldCleanPathJllamaWithSuffix() {
        assertTrue(LlamaLoader.shouldCleanPath(Paths.get("/tmp/jllama-abc123.dylib")));
    }

    @Test
    public void testShouldCleanPathLlamaPrefix() {
        assertTrue(LlamaLoader.shouldCleanPath(Paths.get("/tmp/llama.dll")));
    }

    @Test
    public void testShouldCleanPathLlamaWithSuffix() {
        assertTrue(LlamaLoader.shouldCleanPath(Paths.get("/tmp/llama-model.so")));
    }

    @Test
    public void testShouldCleanPathUnrelatedFile() {
        assertFalse(LlamaLoader.shouldCleanPath(Paths.get("/tmp/somefile.so")));
    }

    @Test
    public void testShouldCleanPathEmptyFilename() {
        assertFalse(LlamaLoader.shouldCleanPath(Paths.get("/tmp/")));
    }

    @Test
    public void testShouldCleanPathPartialMatchInMiddle() {
        // "myJllama" does not start with "jllama" so should not be cleaned
        assertFalse(LlamaLoader.shouldCleanPath(Paths.get("/tmp/myjllama.so")));
    }

    @Test
    public void testShouldCleanPathCaseSensitive() {
        // "Jllama" does not start with lowercase "jllama"
        assertFalse(LlamaLoader.shouldCleanPath(Paths.get("/tmp/Jllama.so")));
    }

    @Test
    public void testShouldCleanPathGgmlMetalFile() {
        // Regression: initialize() extracts ggml-metal.metal on macOS, but cleanup() never
        // matched the "ggml" prefix, so a stale extracted copy was left in the temp dir forever.
        assertTrue(LlamaLoader.shouldCleanPath(Paths.get("/tmp/ggml-metal.metal")));
    }

    @Test
    public void testShouldCleanPathGgmlPrefix() {
        assertTrue(LlamaLoader.shouldCleanPath(Paths.get("/tmp/ggml.tmp")));
    }

    // -------------------------------------------------------------------------
    // parseExtras / BACKEND_PRIORITY
    // -------------------------------------------------------------------------

    private static java.util.List<String> parseExtras(String content) throws IOException {
        return LlamaLoader.parseExtras(new java.io.BufferedReader(new java.io.StringReader(content)));
    }

    @Test
    public void testParseExtrasKeepsOrderAndSkipsCommentsBlankLinesAndCrLf() throws IOException {
        assertEquals(
                java.util.Arrays.asList("OpenCL.dll", "second.dll"),
                parseExtras("# loaded first\r\n\n  OpenCL.dll \r\n\tsecond.dll\n# tail\n"));
    }

    @Test
    public void testParseExtrasEmptyContent() throws IOException {
        assertTrue(parseExtras("").isEmpty());
    }

    @Test
    public void testBackendPriorityTriesAcceleratorsBeforeCpuAndMsvcBeforeCpu() {
        java.util.List<String> priority = LlamaLoader.BACKEND_PRIORITY;
        assertEquals("cpu", priority.get(priority.size() - 1));
        assertEquals("msvc", priority.get(priority.size() - 2));
        assertEquals("cuda13", priority.get(0));
    }

    @Test
    public void testBackendTempDirPrefixMatchesCleanup() {
        // The per-backend extraction directories must be picked up by the temp-dir cleanup.
        assertTrue(
                LlamaLoader.shouldCleanPath(Paths.get("/tmp/" + LlamaLoader.extractionDirectoryName("cuda13", null))));
    }

    // -------------------------------------------------------------------------
    // extractionKey / extractionDirectoryName
    // -------------------------------------------------------------------------

    @Test
    public void testExtractionKeyOfAJarEntryIsItsCrcAndSize(@TempDir Path dir) throws IOException {
        byte[] content = "not really a library".getBytes(StandardCharsets.UTF_8);
        Path jar = dir.resolve("natives.jar");
        try (JarOutputStream out = new JarOutputStream(Files.newOutputStream(jar))) {
            out.putNextEntry(new ZipEntry("net/ladenthin/llama/Linux/x86_64/cpu/libjllama.so"));
            out.write(content);
            out.closeEntry();
        }
        CRC32 crc = new CRC32();
        crc.update(content, 0, content.length);
        URL entry = new URL("jar:" + jar.toUri() + "!/net/ladenthin/llama/Linux/x86_64/cpu/libjllama.so");
        assertEquals(
                Long.toHexString(crc.getValue()) + "-" + Long.toHexString(content.length),
                LlamaLoader.extractionKey(entry));
        // the same bytes in another jar give the same key: the key names the build, not the jar
        Path copy = dir.resolve("copy.jar");
        Files.copy(jar, copy);
        assertEquals(
                LlamaLoader.extractionKey(entry),
                LlamaLoader.extractionKey(
                        new URL("jar:" + copy.toUri() + "!/net/ladenthin/llama/Linux/x86_64/cpu/libjllama.so")));
        // the jar is not left open: it can be deleted (this fails on Windows while it is cached)
        Files.delete(jar);
        Files.delete(copy);
    }

    @Test
    public void testExtractionKeyOfAFileFollowsItsSizeAndModificationTime(@TempDir Path dir) throws IOException {
        Path file = dir.resolve("libjllama.so");
        Files.write(file, new byte[] {1, 2, 3});
        Files.setLastModifiedTime(file, FileTime.fromMillis(1_700_000_000_000L));
        assertEquals(
                Long.toHexString(1_700_000_000_000L) + "-3",
                LlamaLoader.extractionKey(file.toUri().toURL()));
        Files.write(file, new byte[] {1, 2, 3, 4});
        Files.setLastModifiedTime(file, FileTime.fromMillis(1_700_000_000_000L));
        assertEquals(
                Long.toHexString(1_700_000_000_000L) + "-4",
                LlamaLoader.extractionKey(file.toUri().toURL()));
    }

    @Test
    public void testExtractionKeyFallsBackToTheUrlAndToNone() throws IOException {
        URL elsewhere = new URL("https://example.invalid/natives.jar!/libjllama.so");
        assertEquals(Integer.toHexString(elsewhere.toExternalForm().hashCode()), LlamaLoader.extractionKey(elsewhere));
        // a jar URL whose entry does not exist
        URL missing = new URL("jar:file:/nonexistent/natives.jar!/libjllama.so");
        assertEquals(Integer.toHexString(missing.toExternalForm().hashCode()), LlamaLoader.extractionKey(missing));
        assertEquals("none", LlamaLoader.extractionKey(null));
    }

    @Test
    public void testExtractionDirectoryNameCarriesBackendAndKey() {
        assertEquals(
                LlamaLoader.BACKEND_TEMP_DIR_PREFIX + "cuda13-none",
                LlamaLoader.extractionDirectoryName("cuda13", null));
    }

    // -------------------------------------------------------------------------
    // contentsEquals
    // -------------------------------------------------------------------------

    @Test
    public void testContentsEqualsIdenticalContent() throws IOException {
        byte[] data = {1, 2, 3, 4, 5};
        assertTrue(LlamaLoader.contentsEquals(new ByteArrayInputStream(data), new ByteArrayInputStream(data)));
    }

    @Test
    public void resourceMatchesFileFalseWhenResourceAbsent() throws IOException {
        java.nio.file.Path tmp = java.nio.file.Files.createTempFile("llama-loader-test", ".bin");
        try {
            java.nio.file.Files.write(tmp, new byte[] {1, 2, 3});
            // A missing classpath resource must compare as "not matching", never throw.
            assertFalse(LlamaLoader.resourceMatchesFile("net/ladenthin/llama/does-not-exist.bin", tmp));
        } finally {
            java.nio.file.Files.deleteIfExists(tmp);
        }
    }

    @Test
    public void resourceMatchesFileTrueWhenBytesIdentical() throws IOException {
        // The fast-path reuse predicate: a present resource and a byte-identical on-disk copy match.
        java.nio.file.Path tmp = java.nio.file.Files.createTempFile("llama-loader-test", ".bin");
        try {
            try (java.io.InputStream in =
                    LlamaLoader.class.getClassLoader().getResourceAsStream(EXISTING_TEST_RESOURCE)) {
                assertNotNull(in, "fixture must be on the test classpath: " + EXISTING_TEST_RESOURCE);
                java.nio.file.Files.copy(in, tmp, java.nio.file.StandardCopyOption.REPLACE_EXISTING);
            }
            assertTrue(LlamaLoader.resourceMatchesFile(EXISTING_TEST_RESOURCE, tmp));
        } finally {
            java.nio.file.Files.deleteIfExists(tmp);
        }
    }

    @Test
    public void resourceMatchesFileFalseWhenContentDiffers() throws IOException {
        // A present resource whose on-disk copy diverges (here: one extra trailing byte) must NOT match,
        // so a stale/partial file is never mistaken for the shipped library on the reuse fast path.
        java.nio.file.Path tmp = java.nio.file.Files.createTempFile("llama-loader-test", ".bin");
        try {
            try (java.io.InputStream in =
                    LlamaLoader.class.getClassLoader().getResourceAsStream(EXISTING_TEST_RESOURCE)) {
                assertNotNull(in, "fixture must be on the test classpath: " + EXISTING_TEST_RESOURCE);
                java.nio.file.Files.copy(in, tmp, java.nio.file.StandardCopyOption.REPLACE_EXISTING);
            }
            java.nio.file.Files.write(tmp, new byte[] {0}, java.nio.file.StandardOpenOption.APPEND);
            assertFalse(LlamaLoader.resourceMatchesFile(EXISTING_TEST_RESOURCE, tmp));
        } finally {
            java.nio.file.Files.deleteIfExists(tmp);
        }
    }

    @Test
    public void testContentsEqualsBothEmpty() throws IOException {
        assertTrue(LlamaLoader.contentsEquals(
                new ByteArrayInputStream(new byte[0]), new ByteArrayInputStream(new byte[0])));
    }

    @Test
    public void testContentsEqualsDifferentContent() throws IOException {
        assertFalse(LlamaLoader.contentsEquals(
                new ByteArrayInputStream(new byte[] {1, 2, 3}), new ByteArrayInputStream(new byte[] {1, 2, 4})));
    }

    @Test
    public void testContentsEqualsFirstLonger() throws IOException {
        assertFalse(LlamaLoader.contentsEquals(
                new ByteArrayInputStream(new byte[] {1, 2, 3}), new ByteArrayInputStream(new byte[] {1, 2})));
    }

    @Test
    public void testContentsEqualsSecondLonger() throws IOException {
        assertFalse(LlamaLoader.contentsEquals(
                new ByteArrayInputStream(new byte[] {1, 2}), new ByteArrayInputStream(new byte[] {1, 2, 3})));
    }

    @Test
    public void testContentsEqualsAlreadyBuffered() throws IOException {
        // Passes BufferedInputStreams directly — should not double-wrap
        byte[] data = {10, 20, 30};
        assertTrue(LlamaLoader.contentsEquals(
                new BufferedInputStream(new ByteArrayInputStream(data)),
                new BufferedInputStream(new ByteArrayInputStream(data))));
    }

    @Test
    public void testContentsEqualsDifferentAtFirstByte() throws IOException {
        assertFalse(LlamaLoader.contentsEquals(
                new ByteArrayInputStream(new byte[] {0}), new ByteArrayInputStream(new byte[] {1})));
    }

    @Test
    public void testContentsEqualsSingleByteMatch() throws IOException {
        assertTrue(LlamaLoader.contentsEquals(
                new ByteArrayInputStream(new byte[] {42}), new ByteArrayInputStream(new byte[] {42})));
    }

    // -------------------------------------------------------------------------
    // getTempDir
    // -------------------------------------------------------------------------

    @Test
    public void testGetTempDirDefaultsToJavaIoTmpdir() {
        System.clearProperty(TMPDIR_PROP);
        File expected = new File(System.getProperty("java.io.tmpdir"));
        assertEquals(expected, LlamaLoader.getTempDir());
    }

    @Test
    public void testGetTempDirUsesOverrideProperty() {
        // Build path with platform separator so File.getPath() round-trips correctly
        String customPath = new File(System.getProperty("java.io.tmpdir"), "llama-test-custom").getPath();
        System.setProperty(TMPDIR_PROP, customPath);
        assertEquals(new File(customPath), LlamaLoader.getTempDir());
    }

    // -------------------------------------------------------------------------
    // getNativeResourcePath
    // -------------------------------------------------------------------------

    @Test
    public void testGetNativeResourcePathIsClassLoaderRelative() {
        // Resolved through the ClassLoader, which rejects a leading slash (see LlamaLoader.resource).
        String path = LlamaLoader.getNativeResourcePath();
        assertFalse(path.startsWith("/"), "Resource path must not start with '/': " + path);
    }

    @Test
    public void testGetNativeResourcePathContainsPackage() {
        String path = LlamaLoader.getNativeResourcePath();
        // Package net.ladenthin.llama maps to net/ladenthin/llama
        assertTrue(path.contains("net/ladenthin/llama"), "Resource path should contain package");
    }

    @Test
    public void testGetNativeResourcePathContainsOsAndArch() {
        String path = LlamaLoader.getNativeResourcePath();
        // Should end with OS/arch from OSInfo
        String osArch = OSInfo.getNativeLibFolderPathForCurrentOS();
        assertTrue(path.endsWith(osArch), "Resource path should end with OS/arch: " + path);
    }

    /**
     * Regression for the layered-restructure bug: the native-library classpath
     * root is fixed at {@code net/ladenthin/llama/<os>/<arch>} by CMakeLists +
     * the publish workflow, so it must NOT track the loader's own Java package
     * (which moved to {@code net.ladenthin.llama.loader}). Deriving it from
     * {@code LlamaLoader.class.getPackage()} produced {@code .../llama/loader/...},
     * one level too deep, so {@code getResource(...)} returned null and every
     * native-backed test failed with "No native library found".
     */
    @Test
    public void testGetNativeResourcePathIsPackageIndependent() {
        String path = LlamaLoader.getNativeResourcePath();
        String osArch = OSInfo.getNativeLibFolderPathForCurrentOS();
        assertEquals("net/ladenthin/llama/" + osArch, path);
        assertFalse(
                path.contains("/loader/"),
                "Resource path must not include the loader subpackage — the native libs live at "
                        + "net/ladenthin/llama/<os>/<arch>, not under the loader package: " + path);
    }

    // -------------------------------------------------------------------------
    // extractFile
    // -------------------------------------------------------------------------
    //
    // These drive extractFile directly rather than through initialize(). They have to: the
    // cleanup pass initialize() runs first deletes every temp entry whose name starts with
    // "jllama" — which is exactly the per-backend extraction directory a test would seed — so
    // the reuse/replace decision is unreachable from there. Measured, not assumed: seeding a
    // byte-identical file and calling initialize() re-extracts it with a fresh mtime.

    @TempDir
    Path extractDir;

    private static byte[] testResourceBytes() throws IOException {
        try (InputStream in = LlamaLoader.class.getClassLoader().getResourceAsStream(EXISTING_TEST_RESOURCE)) {
            assertNotNull(in, "fixture must be on the test classpath: " + EXISTING_TEST_RESOURCE);
            java.io.ByteArrayOutputStream out = new java.io.ByteArrayOutputStream();
            byte[] buffer = new byte[8192];
            int read;
            while ((read = in.read(buffer)) != -1) {
                out.write(buffer, 0, read);
            }
            return out.toByteArray();
        }
    }

    @Test
    public void extractFileWritesTheResourceIntoTheTargetDirectory() throws IOException {
        Path extracted = LlamaLoader.extractFile("images", "test-image.jpg", extractDir.toString());
        assertNotNull(extracted);
        assertEquals(extractDir.resolve("test-image.jpg"), extracted);
        assertArrayEquals(testResourceBytes(), Files.readAllBytes(extracted));
    }

    @Test
    public void extractFileReturnsNullWhenTheResourceIsAbsent() {
        assertNull(LlamaLoader.extractFile("images", "no-such-file.bin", extractDir.toString()));
    }

    @Test
    public void extractFileLeavesNoTemporaryFileBehind() throws IOException {
        LlamaLoader.extractFile("images", "test-image.jpg", extractDir.toString());
        try (java.util.stream.Stream<Path> entries = Files.list(extractDir)) {
            assertEquals(
                    1L,
                    entries.count(),
                    "the per-attempt .tmp file must be consumed by the move or deleted in the finally block");
        }
    }

    @Test
    public void extractFileReusesAnIdenticalCopyWithoutRewritingIt() throws IOException {
        byte[] resourceBytes = testResourceBytes();
        Path target = extractDir.resolve("test-image.jpg");
        Files.write(target, resourceBytes);
        // A rewrite goes through createTempFile + move, which necessarily lands a fresh mtime, so
        // an unchanged distinctive past value is positive evidence the fast path returned the
        // existing file untouched. That matters beyond tidiness: on Windows a library another
        // process has already loaded cannot be replaced at all, and an in-place rewrite would
        // expose a half-written library to a concurrent loader.
        FileTime seeded = FileTime.fromMillis(1_000_000_000L);
        Files.setLastModifiedTime(target, seeded);

        Path extracted = LlamaLoader.extractFile("images", "test-image.jpg", extractDir.toString());

        assertEquals(target, extracted);
        assertEquals(seeded, Files.getLastModifiedTime(target));
        assertArrayEquals(resourceBytes, Files.readAllBytes(target));
    }

    @Test
    public void extractFileReplacesACopyWhoseContentDiffers() throws IOException {
        // A same-named file whose content differs is what a previous release's extraction leaves
        // behind in a shared tmpdir. Reusing it would load a stale library forever.
        Path target = extractDir.resolve("test-image.jpg");
        Files.write(target, "stale content from an older release".getBytes(StandardCharsets.UTF_8));

        Path extracted = LlamaLoader.extractFile("images", "test-image.jpg", extractDir.toString());

        assertEquals(target, extracted);
        assertArrayEquals(testResourceBytes(), Files.readAllBytes(target));
    }

    // The nested call is what JNI_OnLoad makes: GetFieldID on LlamaModel initializes it, and its
    // static block calls initialize() again on the loading thread. It must not run a second load
    // (with a multi-backend jar that deleted and re-extracted the library still being loaded).
    @Test
    public void aNestedInitializeOnTheLoadingThreadDoesNotRunTheBodyAgain() {
        java.util.concurrent.atomic.AtomicInteger runs = new java.util.concurrent.atomic.AtomicInteger();
        LlamaLoader.runOnceOnThisThread(() -> {
            runs.incrementAndGet();
            LlamaLoader.runOnceOnThisThread(runs::incrementAndGet);
        });
        assertEquals(1, runs.get());
    }

    @Test
    public void aLaterInitializeStillRunsTheBody() {
        java.util.concurrent.atomic.AtomicInteger runs = new java.util.concurrent.atomic.AtomicInteger();
        LlamaLoader.runOnceOnThisThread(runs::incrementAndGet);
        LlamaLoader.runOnceOnThisThread(runs::incrementAndGet);
        assertEquals(2, runs.get(), "only a nested call is skipped; tests and later classes re-run it");
    }

    @Test
    public void aFailingBodyDoesNotLeaveTheGuardSet() {
        assertThrows(
                IllegalStateException.class,
                () -> LlamaLoader.runOnceOnThisThread(() -> {
                    throw new IllegalStateException("load failed");
                }));
        java.util.concurrent.atomic.AtomicInteger runs = new java.util.concurrent.atomic.AtomicInteger();
        LlamaLoader.runOnceOnThisThread(runs::incrementAndGet);
        assertEquals(1, runs.get(), "a failed load must be retryable");
    }
}
