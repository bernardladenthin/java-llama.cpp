// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.PrintStream;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

@ClaudeGenerated(
		purpose = "Verify NativeLibraryPermissionSetter.apply(File): returns the AND of " +
				"the three File setter calls (setReadable, setWritable, setExecutable) and " +
				"emits a descriptive warning to the injected PrintStream when any setter " +
				"returns false. Uses a StubFile subclass so the test does not touch disk."
)
public class NativeLibraryPermissionSetterTest {

	/** Stub File whose setReadable/setWritable/setExecutable returns are configurable. */
	private static class StubFile extends File {
		final boolean readable;
		final boolean writable;
		final boolean executable;

		StubFile(boolean readable, boolean writable, boolean executable) {
			super("stub-native-lib");
			this.readable = readable;
			this.writable = writable;
			this.executable = executable;
		}

		@Override
		public boolean setReadable(boolean r) {
			return readable;
		}

		@Override
		public boolean setWritable(boolean w, boolean ownerOnly) {
			return writable;
		}

		@Override
		public boolean setExecutable(boolean x) {
			return executable;
		}
	}

	private ByteArrayOutputStream sink;
	private NativeLibraryPermissionSetter setter;

	private void setUp() {
		sink = new ByteArrayOutputStream();
		setter = new NativeLibraryPermissionSetter(new PrintStream(sink));
	}

	@Test
	public void testApplyAllSucceed() {
		setUp();
		assertTrue(setter.apply(new StubFile(true, true, true)), "expected success when all setters return true");
		assertEquals("", sink.toString(), "no warning expected on success");
	}

	@Test
	public void testApplyReadableFails() {
		setUp();
		assertFalse(setter.apply(new StubFile(false, true, true)));
		String out = sink.toString();
		assertTrue(out.contains("readable=false"), "warning should mention readable=false: " + out);
		assertTrue(out.contains("writable=true"), "warning should mention writable=true: " + out);
		assertTrue(out.contains("executable=true"), "warning should mention executable=true: " + out);
		assertTrue(out.contains("stub-native-lib"), "warning should mention file path: " + out);
	}

	@Test
	public void testApplyWritableFails() {
		setUp();
		assertFalse(setter.apply(new StubFile(true, false, true)));
		assertTrue(sink.toString().contains("writable=false"));
	}

	@Test
	public void testApplyExecutableFails() {
		setUp();
		assertFalse(setter.apply(new StubFile(true, true, false)));
		assertTrue(sink.toString().contains("executable=false"));
	}

	@Test
	public void testApplyAllFail() {
		setUp();
		assertFalse(setter.apply(new StubFile(false, false, false)));
		String out = sink.toString();
		assertTrue(out.contains("readable=false"));
		assertTrue(out.contains("writable=false"));
		assertTrue(out.contains("executable=false"));
	}

	@Test
	public void testConstructorRejectsNullSink() {
	assertThrows(NullPointerException.class, () -> new NativeLibraryPermissionSetter(null));
	}
}
