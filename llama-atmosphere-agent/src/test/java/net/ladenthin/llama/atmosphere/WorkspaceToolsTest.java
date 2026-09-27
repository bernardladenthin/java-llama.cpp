// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.atmosphere;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.contains;
import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.not;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import org.atmosphere.ai.fs.AgentFileSystem;
import org.atmosphere.ai.fs.WorkspaceAgentFileSystem;
import org.atmosphere.ai.tool.ToolDefinition;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** The three replaced tools, driven the way Atmosphere drives them. */
class WorkspaceToolsTest {

    @TempDir
    Path workspace;

    private Map<Class<?>, Object> scope;
    private WorkspaceTools.ReadTracker tracker;

    @BeforeEach
    void bindFileSystem() {
        AgentFileSystem fs = new WorkspaceAgentFileSystem(workspace, AgentFileSystem.Limits.defaults());
        scope = Map.of(AgentFileSystem.class, fs);
        tracker = new WorkspaceTools.ReadTracker();
    }

    private String call(ToolDefinition tool, Map<String, Object> arguments) throws Exception {
        return String.valueOf(tool.executor().execute(arguments, scope));
    }

    private void write(String name, String content) throws Exception {
        Files.writeString(workspace.resolve(name), content, StandardCharsets.UTF_8);
    }

    // ----- read_file -----

    @Test
    void readingReturnsNumberedLinesAndSaysWhatItLeftOut() throws Exception {
        write("big.txt", "l1\nl2\nl3\nl4\nl5\n");

        String all = call(WorkspaceTools.readFile(tracker), Map.of("file_path", "big.txt"));
        assertThat(all, containsString("  1: l1"));
        assertThat("a file that fits is shown without a footer", all, not(containsString("showing lines")));

        String window = call(WorkspaceTools.readFile(tracker), Map.of("file_path", "big.txt", "offset", 2, "limit", 2));
        assertThat(window, containsString("  2: l2"));
        assertThat(window, containsString("  3: l3"));
        assertThat(window, not(containsString("l4")));
        assertThat(window, containsString("showing lines 2–3 of 5"));
    }

    @Test
    void readingPastTheEndAndReadingNothingAreExplained() throws Exception {
        write("small.txt", "only\n");
        write("empty.txt", "");

        assertThat(
                call(WorkspaceTools.readFile(tracker), Map.of("file_path", "small.txt", "offset", 99)),
                containsString("past the end"));
        assertThat(call(WorkspaceTools.readFile(tracker), Map.of("file_path", "empty.txt")), containsString("empty"));
        assertThat(call(WorkspaceTools.readFile(tracker), Map.of()), containsString("file_path is required"));
    }

    // ----- edit_file -----

    @Test
    void anEditIsRefusedUntilTheFileWasRead() throws Exception {
        write("code.txt", "alpha\n");

        String refused = call(
                WorkspaceTools.editFile(tracker),
                Map.of("file_path", "code.txt", "old_string", "alpha", "new_string", "beta"));
        assertThat(refused, containsString("read code.txt before editing"));
        assertThat("nothing was written", Files.readString(workspace.resolve("code.txt")), is("alpha\n"));

        call(WorkspaceTools.readFile(tracker), Map.of("file_path", "code.txt"));
        String edited = call(
                WorkspaceTools.editFile(tracker),
                Map.of("file_path", "code.txt", "old_string", "alpha", "new_string", "beta"));
        assertThat(edited, containsString("Edited code.txt"));
        assertThat(Files.readString(workspace.resolve("code.txt")), is("beta\n"));
    }

    @Test
    void theAnswerOfASuccessfulEditShowsTheChangedRegion() throws Exception {
        write("code.txt", "a\nb\nc\nd\n");
        call(WorkspaceTools.readFile(tracker), Map.of("file_path", "code.txt"));

        String answer = call(
                WorkspaceTools.editFile(tracker),
                Map.of("file_path", "code.txt", "old_string", "c", "new_string", "CHANGED"));

        assertThat(answer, containsString("3: CHANGED"));
    }

    @Test
    void severalEditsArriveAsAListAndAreAllOrNothing() throws Exception {
        write("code.txt", "one\ntwo\nthree\n");
        call(WorkspaceTools.readFile(tracker), Map.of("file_path", "code.txt"));

        String failed = call(
                WorkspaceTools.editFile(tracker),
                Map.of(
                        "file_path",
                        "code.txt",
                        "edits",
                        List.of(
                                Map.of("old_string", "one", "new_string", "1"),
                                Map.of("old_string", "absent", "new_string", "x"))));
        assertThat(failed, containsString("edit 2 of 2"));
        assertThat(
                "a failed batch leaves the file untouched",
                Files.readString(workspace.resolve("code.txt")),
                is("one\ntwo\nthree\n"));

        String applied = call(
                WorkspaceTools.editFile(tracker),
                Map.of(
                        "file_path",
                        "code.txt",
                        "edits",
                        List.of(
                                Map.of("old_string", "one", "new_string", "1"),
                                Map.of("old_string", "three", "new_string", "3"))));
        assertThat(applied, containsString("2 edits"));
        assertThat(Files.readString(workspace.resolve("code.txt")), is("1\ntwo\n3\n"));
    }

    @Test
    void aFailedEditExplainsItselfInsteadOfSayingNo() throws Exception {
        write("code.txt", "public void handle(String name) {\n}\n");
        call(WorkspaceTools.readFile(tracker), Map.of("file_path", "code.txt"));

        String answer = call(
                WorkspaceTools.editFile(tracker),
                Map.of("file_path", "code.txt", "old_string", "public void handle(String value) {", "new_string", "x"));

        assertThat(answer, containsString("closest lines"));
        assertThat(answer, containsString("1: public void handle(String name) {"));
    }

    // ----- grep -----

    @Test
    void theSearchSkipsBuildOutputAndRepositoryInternals() throws Exception {
        Files.createDirectories(workspace.resolve("src"));
        Files.createDirectories(workspace.resolve("target/classes"));
        Files.createDirectories(workspace.resolve(".git"));
        Files.writeString(workspace.resolve("src/Main.java"), "class Main { void needle() {} }\n");
        Files.writeString(workspace.resolve("target/classes/Main.txt"), "needle in build output\n");
        Files.writeString(workspace.resolve(".git/config"), "needle in the repository\n");

        String answer = call(WorkspaceTools.grep(), Map.of("pattern", "needle"));

        assertThat(answer, containsString("src/Main.java"));
        assertThat("build output is not source", answer, not(containsString("target/")));
        assertThat("the repository's own files are not source", answer, not(containsString(".git")));
        assertThat(answer, containsString("1 matches in 1 files"));
    }

    @Test
    void theSearchCanBeNarrowedAndCanListFilesOnly() throws Exception {
        Files.writeString(workspace.resolve("A.java"), "needle\n");
        Files.writeString(workspace.resolve("B.txt"), "needle\n");

        assertThat(
                call(WorkspaceTools.grep(), Map.of("pattern", "needle", "glob", "*.java")),
                not(containsString("B.txt")));
        String filesOnly = call(WorkspaceTools.grep(), Map.of("pattern", "needle", "files_only", true));
        assertThat(filesOnly, containsString("A.java"));
        assertThat("files_only means no line content", filesOnly, not(containsString("  1: needle")));
    }

    @Test
    void anInvalidPatternIsAnErrorMessageNotAnException() throws Exception {
        assertThat(call(WorkspaceTools.grep(), Map.of("pattern", "[unclosed")), containsString("Invalid regular"));
        assertThat(call(WorkspaceTools.grep(), Map.of()), containsString("pattern is required"));
    }

    @Test
    void theToolSetReplacesTheFrameworksReadEditAndGrepWithoutDuplicates() {
        List<String> names =
                WorkspaceTools.all(tracker).stream().map(ToolDefinition::name).toList();

        assertThat(names, contains("ls", "read_file", "write_file", "edit_file", "glob", "grep", "delete", "rename"));
        assertThat(
                "no tool name may appear twice",
                names.size(),
                is(names.stream().distinct().count() == 8L ? 8 : -1));
    }
}
