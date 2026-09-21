// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.atmosphere;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.time.Duration;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import org.atmosphere.ai.tool.ToolDefinition;

/**
 * The {@code run_command} tool: runs any shell command (starting in the workspace) and returns its exit code and
 * (merged, truncated) output. Opt-in via {@code --allow-shell} — a model-driven shell is exactly as
 * powerful as the user account it runs under.
 */
public final class ShellTool {

    /** Tool name as offered to the model. */
    public static final String TOOL_NAME = "run_command";

    private static final String PARAM_COMMAND = "command";
    private static final String PARAM_TIMEOUT = "timeout_seconds";

    private ShellTool() {}

    /**
     * Build the tool definition.
     *
     * @param workspace the working directory of every command
     * @param defaultTimeout the timeout applied when the model does not pass {@code timeout_seconds}
     * @param maxOutputChars output is truncated to this many characters (tail kept, head marked)
     * @return the definition
     */
    public static ToolDefinition definition(Path workspace, Duration defaultTimeout, int maxOutputChars) {
        return ToolDefinition.builder(
                        TOOL_NAME,
                        "Run any command line on this machine through " + shellName() + " and return its exit"
                                + " code and output (stdout and stderr merged). It starts in the workspace"
                                + " directory but is not limited to it: use it for every task a terminal can do,"
                                + " e.g. docker, git, package managers, build tools or system information.")
                .parameter(PARAM_COMMAND, "The command line to run through " + shellName(), "string", true)
                .parameter(PARAM_TIMEOUT, "Seconds to wait before the command is killed", "integer", false)
                .executor(args -> {
                    Object command = args.get(PARAM_COMMAND);
                    if (command == null || command.toString().isBlank()) {
                        return "Error: '" + PARAM_COMMAND + "' is required";
                    }
                    Duration timeout = timeoutOf(args.get(PARAM_TIMEOUT), defaultTimeout);
                    return run(workspace, command.toString(), timeout, maxOutputChars);
                })
                .build();
    }

    /**
     * Whether commands run through {@code cmd.exe} rather than {@code sh}.
     *
     * @return {@code true} on Windows
     */
    static boolean isWindows() {
        return System.getProperty("os.name", "")
                .toLowerCase(java.util.Locale.ROOT)
                .contains("win");
    }

    /**
     * The shell {@link #run} uses, named for the model so it writes commands in the right syntax.
     *
     * @return {@code "cmd.exe on Windows"} or {@code "sh"}
     */
    static String shellName() {
        return isWindows() ? "cmd.exe on Windows" : "sh";
    }

    private static Duration timeoutOf(Object raw, Duration fallback) {
        if (raw instanceof Number n && n.longValue() > 0) {
            return Duration.ofSeconds(n.longValue());
        }
        if (raw instanceof String s && !s.isBlank()) {
            try {
                long seconds = Long.parseLong(s.trim());
                if (seconds > 0) {
                    return Duration.ofSeconds(seconds);
                }
            } catch (NumberFormatException ignored) {
                // fall through to the default
            }
        }
        return fallback;
    }

    /**
     * Run one command through the platform shell ({@code sh -c} / {@code cmd.exe /c}).
     *
     * @param workspace the working directory
     * @param command the command line
     * @param timeout kill the process after this long
     * @param maxOutputChars truncate the captured output to this many characters
     * @return a text block starting with {@code exit code: N}, followed by the output
     * @throws IOException if the process cannot be started
     * @throws InterruptedException if interrupted while waiting
     */
    static String run(Path workspace, String command, Duration timeout, int maxOutputChars)
            throws IOException, InterruptedException {
        ProcessBuilder builder =
                isWindows() ? new ProcessBuilder("cmd.exe", "/c", command) : new ProcessBuilder("sh", "-c", command);
        builder.directory(workspace.toFile());
        builder.redirectErrorStream(true);
        Process process = builder.start();
        process.getOutputStream().close();
        CompletableFuture<byte[]> output = CompletableFuture.supplyAsync(() -> {
            try {
                return process.getInputStream().readAllBytes();
            } catch (IOException e) {
                return ("[output unreadable: " + e.getMessage() + "]").getBytes(StandardCharsets.UTF_8);
            }
        });
        boolean finished = process.waitFor(timeout.toMillis(), TimeUnit.MILLISECONDS);
        if (!finished) {
            // Kill the shell AND its children: `sh -c "sleep 30"` forks sleep, which would otherwise
            // keep the output pipe open (and the read below blocked) for its full duration.
            process.toHandle().descendants().forEach(ProcessHandle::destroyForcibly);
            process.destroyForcibly();
            process.waitFor(5, TimeUnit.SECONDS);
        }
        String text;
        try {
            text = new String(output.get(5, TimeUnit.SECONDS), StandardCharsets.UTF_8);
        } catch (ExecutionException | TimeoutException e) {
            text = "[output unavailable: " + e.getMessage() + "]";
        }
        StringBuilder result = new StringBuilder();
        if (finished) {
            result.append("exit code: ").append(process.exitValue()).append('\n');
        } else {
            result.append("exit code: (killed after ")
                    .append(timeout.getSeconds())
                    .append(" s)\n");
        }
        if (text.length() > maxOutputChars) {
            result.append("[output truncated to the last ")
                    .append(maxOutputChars)
                    .append(" of ")
                    .append(text.length())
                    .append(" characters]\n")
                    .append(text, text.length() - maxOutputChars, text.length());
        } else {
            result.append(text);
        }
        return result.toString();
    }
}
