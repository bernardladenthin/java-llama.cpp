package de.kherud.llama;

import java.io.ByteArrayInputStream;
import java.io.BufferedInputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Paths;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

@ClaudeGenerated(
        purpose = "Verify the helper statics extracted from LlamaLoader without requiring any " +
                  "native library: shouldCleanPath detects jllama/llama-prefixed files for " +
                  "cleanup; contentsEquals performs a correct byte-level stream comparison " +
                  "including BufferedInputStream wrapping and length mismatches; getTempDir " +
                  "honours the 'de.kherud.llama.tmpdir' system-property override; and " +
                  "getNativeResourcePath produces the expected classpath resource prefix."
)
public class LlamaLoaderTest {

	private static final String TMPDIR_PROP = "de.kherud.llama.tmpdir";
	private String previousTmpDir;

	@Before
	public void saveTmpDirProp() {
		previousTmpDir = System.getProperty(TMPDIR_PROP);
	}

	@After
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

	// -------------------------------------------------------------------------
	// contentsEquals
	// -------------------------------------------------------------------------

	@Test
	public void testContentsEqualsIdenticalContent() throws IOException {
		byte[] data = {1, 2, 3, 4, 5};
		assertTrue(LlamaLoader.contentsEquals(
				new ByteArrayInputStream(data),
				new ByteArrayInputStream(data)
		));
	}

	@Test
	public void testContentsEqualsBothEmpty() throws IOException {
		assertTrue(LlamaLoader.contentsEquals(
				new ByteArrayInputStream(new byte[0]),
				new ByteArrayInputStream(new byte[0])
		));
	}

	@Test
	public void testContentsEqualsDifferentContent() throws IOException {
		assertFalse(LlamaLoader.contentsEquals(
				new ByteArrayInputStream(new byte[]{1, 2, 3}),
				new ByteArrayInputStream(new byte[]{1, 2, 4})
		));
	}

	@Test
	public void testContentsEqualsFirstLonger() throws IOException {
		assertFalse(LlamaLoader.contentsEquals(
				new ByteArrayInputStream(new byte[]{1, 2, 3}),
				new ByteArrayInputStream(new byte[]{1, 2})
		));
	}

	@Test
	public void testContentsEqualsSecondLonger() throws IOException {
		assertFalse(LlamaLoader.contentsEquals(
				new ByteArrayInputStream(new byte[]{1, 2}),
				new ByteArrayInputStream(new byte[]{1, 2, 3})
		));
	}

	@Test
	public void testContentsEqualsAlreadyBuffered() throws IOException {
		// Passes BufferedInputStreams directly — should not double-wrap
		byte[] data = {10, 20, 30};
		assertTrue(LlamaLoader.contentsEquals(
				new BufferedInputStream(new ByteArrayInputStream(data)),
				new BufferedInputStream(new ByteArrayInputStream(data))
		));
	}

	@Test
	public void testContentsEqualsDifferentAtFirstByte() throws IOException {
		assertFalse(LlamaLoader.contentsEquals(
				new ByteArrayInputStream(new byte[]{0}),
				new ByteArrayInputStream(new byte[]{1})
		));
	}

	@Test
	public void testContentsEqualsSingleByteMatch() throws IOException {
		assertTrue(LlamaLoader.contentsEquals(
				new ByteArrayInputStream(new byte[]{42}),
				new ByteArrayInputStream(new byte[]{42})
		));
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
	public void testGetNativeResourcePathStartsWithSlash() {
		String path = LlamaLoader.getNativeResourcePath();
		assertTrue("Resource path should start with '/'", path.startsWith("/"));
	}

	@Test
	public void testGetNativeResourcePathContainsPackage() {
		String path = LlamaLoader.getNativeResourcePath();
		// Package de.kherud.llama maps to de/kherud/llama
		assertTrue("Resource path should contain package", path.contains("de/kherud/llama"));
	}

	@Test
	public void testGetNativeResourcePathContainsOsAndArch() {
		String path = LlamaLoader.getNativeResourcePath();
		// Should end with OS/arch from OSInfo
		String osArch = OSInfo.getNativeLibFolderPathForCurrentOS();
		assertTrue("Resource path should end with OS/arch: " + path,
				path.endsWith(osArch));
	}
}
