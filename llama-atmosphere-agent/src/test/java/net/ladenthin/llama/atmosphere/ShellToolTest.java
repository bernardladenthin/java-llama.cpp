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

class ShellToolTest {

    @TempDir
    Path workspace;

    @Test
    void runsInTheWorkspaceAndReportsExitCodeAndOutput() throws Exception {
        Files.writeString(workspace.resolve("marker.txt"), "x");
        ToolDefinition tool = ShellTool.definition(workspace, Duration.ofSeconds(30), 10_000);

        Object result = tool.executor().execute(Map.of("command", "ls"));

        assertThat(tool.name(), is(ShellTool.TOOL_NAME));
        assertThat(String.valueOf(result), startsWith("exit code: 0"));
        assertThat(String.valueOf(result), containsString("marker.txt"));
    }

    @Test
    void nonZeroExitAndStderrAreReturnedNotThrown() throws Exception {
        ToolDefinition tool = ShellTool.definition(workspace, Duration.ofSeconds(30), 10_000);

        Object result = tool.executor().execute(Map.of("command", "echo boom 1>&2; exit 3"));

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
        String result = ShellTool.run(workspace, "printf 'aaaaaaaaaaaaaaaaaaaaZZ'", Duration.ofSeconds(30), 5);

        assertThat(result, containsString("[output truncated to the last 5 of 22 characters]"));
        assertThat(result, containsString("aaaZZ"));
    }

    @Test
    void timeoutKillsTheProcess() throws Exception {
        String result = ShellTool.run(workspace, "sleep 30", Duration.ofMillis(300), 10_000);

        assertThat(result, startsWith("exit code: (killed after 0 s)"));
    }
}
