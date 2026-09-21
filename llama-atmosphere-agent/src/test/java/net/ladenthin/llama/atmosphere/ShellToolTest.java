// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.atmosphere;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.startsWith;

import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.Map;
import org.atmosphere.ai.tool.ToolDefinition;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * Runs on every platform: each test picks its command line with {@link ShellTool#isWindows()}, the same
 * detection {@link ShellTool#run} uses to choose between {@code cmd.exe /c} and {@code sh -c}, so the test
 * always speaks the shell the tool actually starts.
 */
class ShellToolTest {

    @TempDir
    Path workspace;

    /**
     * The command line for the shell {@link ShellTool} starts on this platform.
     *
     * @param posix the {@code sh} form
     * @param windows the {@code cmd.exe} form
     * @return the form matching {@link ShellTool#isWindows()}
     */
    private static String shell(String posix, String windows) {
        return ShellTool.isWindows() ? windows : posix;
    }

    @Test
    void runsInTheWorkspaceAndReportsExitCodeAndOutput() throws Exception {
        Files.writeString(workspace.resolve("marker.txt"), "x");
        ToolDefinition tool = ShellTool.definition(workspace, Duration.ofSeconds(30), 10_000);

        Object result = tool.executor().execute(Map.of("command", shell("ls", "dir /b")));

        assertThat(tool.name(), is(ShellTool.TOOL_NAME));
        assertThat(String.valueOf(result), startsWith("exit code: 0"));
        assertThat(String.valueOf(result), containsString("marker.txt"));
    }

    @Test
    void nonZeroExitAndStderrAreReturnedNotThrown() throws Exception {
        ToolDefinition tool = ShellTool.definition(workspace, Duration.ofSeconds(30), 10_000);

        Object result =
                tool.executor().execute(Map.of("command", shell("echo boom 1>&2; exit 3", "echo boom 1>&2 & exit 3")));

        assertThat(String.valueOf(result), startsWith("exit code: 3"));
        assertThat(String.valueOf(result), containsString("boom"));
    }

    @Test
    void missingCommandIsAnErrorString() throws Exception {
        ToolDefinition tool = ShellTool.definition(workspace, Duration.ofSeconds(30), 10_000);

        assertThat(String.valueOf(tool.executor().execute(Map.of())), containsString("'command' is required"));
    }

    @Test
    void outputIsTruncatedToTheTail() throws Exception {
        // echo is the one output command both shells share; it ends the line with the platform's
        // separator (\n from sh, \r\n from cmd.exe), which is part of the counted output.
        String newline = ShellTool.isWindows() ? "\r\n" : "\n";
        String result = ShellTool.run(workspace, "echo aaaaaaaaaaaaaaaaaaaaZZ", Duration.ofSeconds(30), 5);

        assertThat(
                result,
                containsString("[output truncated to the last 5 of " + (22 + newline.length()) + " characters]"));
        assertThat(result, containsString("ZZ" + newline));
    }

    @Test
    void timeoutKillsTheProcess() throws Exception {
        // cmd.exe has no sleep; ping waits about one second per echo request. Like `sh -c "sleep 30"`,
        // it is a child of the shell, so this also covers killing the descendants.
        String result = ShellTool.run(
                workspace, shell("sleep 30", "ping -n 30 127.0.0.1 >nul"), Duration.ofMillis(300), 10_000);

        assertThat(result, startsWith("exit code: (killed after 0 s)"));
    }
}
