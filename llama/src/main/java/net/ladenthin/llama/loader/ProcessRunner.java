// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.loader;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.TimeUnit;
import lombok.ToString;

/**
 * Runs a short system command and returns what it printed -- {@link OSInfo} asks {@code uname} for the
 * machine and the operating system. JDK only, on purpose: for two {@code uname} calls a process library
 * such as Apache Commons Exec would be a runtime dependency out of all proportion, in a library that keeps
 * those to a minimum.
 *
 * <p>Meant for commands that print a few lines: the output is read once the command has ended, which a pipe
 * buffer covers many times over. A command that does not end in time is killed, so a hanging probe can
 * never hold up class loading.
 */
@ToString
class ProcessRunner {

    /** How long {@link #runAndWaitFor(String)} waits; {@code uname} answers in milliseconds. */
    static final long DEFAULT_TIMEOUT_SECONDS = 10;

    /**
     * Runs a command with the {@linkplain #DEFAULT_TIMEOUT_SECONDS default timeout}.
     *
     * @param command the program and its arguments, separated by whitespace
     * @return what the command wrote to standard output
     * @throws IOException if the command cannot be started or does not end in time
     * @throws InterruptedException if the calling thread is interrupted while waiting
     */
    String runAndWaitFor(String command) throws IOException, InterruptedException {
        return runAndWaitFor(command, DEFAULT_TIMEOUT_SECONDS, TimeUnit.SECONDS);
    }

    /**
     * Runs a command and returns its standard output. The command is split at whitespace and started
     * directly, without a shell, so nothing in it is interpreted; its standard error is not read.
     *
     * @param command the program and its arguments, separated by whitespace
     * @param timeout how long to wait for the command to end
     * @param unit the unit of {@code timeout}
     * @return what the command wrote to standard output
     * @throws IOException if the command cannot be started or does not end in time (it is then killed)
     * @throws InterruptedException if the calling thread is interrupted while waiting
     */
    String runAndWaitFor(String command, long timeout, TimeUnit unit) throws IOException, InterruptedException {
        Process process = new ProcessBuilder(command.trim().split("\\s+")).start();
        try {
            if (!process.waitFor(timeout, unit)) {
                throw new IOException("'" + command + "' did not end within " + timeout + " " + unit);
            }
            return readAll(process.getInputStream());
        } finally {
            // a no-op for a process that has ended; kills one that timed out or whose wait was interrupted
            process.destroyForcibly();
        }
    }

    private static String readAll(InputStream in) throws IOException {
        try (InputStream stream = in) {
            ByteArrayOutputStream bytes = new ByteArrayOutputStream();
            byte[] chunk = new byte[256];
            for (int n = stream.read(chunk); n >= 0; n = stream.read(chunk)) {
                bytes.write(chunk, 0, n);
            }
            // ByteArrayOutputStream#toString(Charset) is Java 10+; this artifact targets Java 8.
            return new String(bytes.toByteArray(), StandardCharsets.UTF_8);
        }
    }
}
