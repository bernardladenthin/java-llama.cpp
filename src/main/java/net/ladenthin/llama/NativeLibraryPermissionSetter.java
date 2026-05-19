// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import java.io.File;
import java.io.PrintStream;
import java.util.Objects;

/**
 * Applies the read / write (owner-only) / execute permissions required for the
 * JVM to load an extracted native library file.
 *
 * <p>The three {@link File} setter calls each return a {@code boolean}; this
 * class observes those return values and writes a descriptive warning to a
 * configurable {@link PrintStream} when any permission change is rejected by
 * the platform. Both the warning sink and the entry point are instance members
 * so the behaviour can be unit-tested without touching {@link System#err}.
 */
final class NativeLibraryPermissionSetter {

	private final PrintStream warningSink;

	NativeLibraryPermissionSetter(PrintStream warningSink) {
		this.warningSink = Objects.requireNonNull(warningSink, "warningSink");
	}

	/**
	 * Sets read, owner-only write, and execute permissions on {@code file}.
	 *
	 * @param file the extracted native library file
	 * @return {@code true} if all three permission changes succeeded
	 */
	boolean apply(File file) {
		boolean readable = file.setReadable(true);
		boolean writable = file.setWritable(true, true);
		boolean executable = file.setExecutable(true);
		if (!readable || !writable || !executable) {
			warningSink.println("Warning: could not set permissions on " + file
					+ " (readable=" + readable
					+ ", writable=" + writable
					+ ", executable=" + executable + ")");
			return false;
		}
		return true;
	}
}
