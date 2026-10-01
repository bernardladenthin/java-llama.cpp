// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.loader;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.lessThan;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assumptions.assumeFalse;

import java.io.IOException;
import java.util.concurrent.TimeUnit;
import net.ladenthin.llama.ClaudeGenerated;
import org.junit.jupiter.api.Test;

@ClaudeGenerated(
        purpose = "Pin ProcessRunner's contract: standard output is returned, an unknown command "
                + "and a command that does not end in time are IOExceptions, and the latter is killed.")
public class ProcessRunnerTest {

    private static final boolean WINDOWS = System.getProperty("os.name", "").startsWith("Windows");

    @Test
    public void returnsStandardOutput() throws Exception {
        assumeFalse(WINDOWS, "uname is a POSIX command");
        String output = new ProcessRunner().runAndWaitFor("  uname   -s ");
        assertThat(output.trim().isEmpty(), is(false));
    }

    @Test
    public void unknownCommandIsAnIOException() {
        assertThrows(IOException.class, () -> new ProcessRunner().runAndWaitFor("jllama-no-such-command-4711"));
    }

    @Test
    public void commandThatDoesNotEndInTimeIsKilledAndReported() {
        assumeFalse(WINDOWS, "sleep is a POSIX command");
        long start = System.nanoTime();
        IOException e = assertThrows(
                IOException.class, () -> new ProcessRunner().runAndWaitFor("sleep 30", 200, TimeUnit.MILLISECONDS));
        assertThat(e.getMessage(), containsString("'sleep 30' did not end within 200 MILLISECONDS"));
        assertThat(TimeUnit.NANOSECONDS.toSeconds(System.nanoTime() - start), lessThan(10L));
    }
}
