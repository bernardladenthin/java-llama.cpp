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
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * The record of what was said, which is a different thing from the conversation the model is sent.
 */
class TranscriptTest {

    @TempDir
    Path directory;

    @Test
    void entriesComeBackInTheOrderTheyWereSaid() {
        Transcript transcript = new Transcript();

        transcript.add(Transcript.Kind.USER, "first");
        transcript.add(Transcript.Kind.TOOL, "ls {} -> a b");
        transcript.add(Transcript.Kind.AGENT, "second");

        assertThat(
                transcript.entries().stream().map(Transcript.Entry::text).toList(),
                contains("first", "ls {} -> a b", "second"));
    }

    @Test
    void twoEntriesInTheSameMomentAreBothKept() {
        // The reason this is a list and not a map keyed by the time: a tool result and the answer that
        // follows it regularly land in the same millisecond, and a map would keep one and lose the
        // other without saying so.
        Transcript transcript = new Transcript();

        for (int i = 0; i < 50; i++) {
            transcript.add(Transcript.Kind.AGENT, "entry " + i);
        }

        assertThat(transcript.size(), is(50));
    }

    @Test
    void blankTextIsNotWorthALine() {
        Transcript transcript = new Transcript();

        transcript.add(Transcript.Kind.AGENT, "");
        transcript.add(Transcript.Kind.AGENT, "   ");
        transcript.add(Transcript.Kind.AGENT, null);

        assertThat("a session of interrupted turns would otherwise fill the file", transcript.size(), is(0));
    }

    @Test
    void everyLineCarriesTheTimeAndWhoSaidIt() {
        Transcript transcript = new Transcript();
        transcript.add(Transcript.Kind.USER, "what time is it");

        String rendered = transcript.render();

        assertThat(rendered, containsString("you: what time is it"));
        assertThat(
                "a date and a clock time",
                rendered.matches("(?s)\\[\\d{4}-\\d{2}-\\d{2} \\d{2}:\\d{2}:\\d{2}\\].*"),
                is(true));
    }

    @Test
    void savingWritesEverythingToTheNamedFile() throws Exception {
        Transcript transcript = new Transcript();
        transcript.add(Transcript.Kind.USER, "hello");
        transcript.add(Transcript.Kind.AGENT, "hi");

        Path written = transcript.save(directory, "session.txt");

        assertThat(written.getFileName().toString(), is("session.txt"));
        String text = Files.readString(written, StandardCharsets.UTF_8);
        assertThat(text, containsString("you: hello"));
        assertThat(text, containsString("agent: hi"));
    }

    @Test
    void savingWithoutANameUsesTheTime() throws Exception {
        Transcript transcript = new Transcript();
        transcript.add(Transcript.Kind.USER, "hello");

        Path written = transcript.save(directory, null);

        assertThat(written.getFileName().toString(), containsString("transcript-"));
        assertThat(written.getFileName().toString().endsWith(".txt"), is(true));
    }

    @Test
    void aConfiguredFileIsAppendedToAsThingsAreSaid() throws Exception {
        // The point of it: a session that is killed still leaves what it had. Nothing is written at
        // the end, so there is no end to miss.
        Path live = directory.resolve("logs").resolve("live.txt");
        Transcript transcript = new Transcript(live);

        transcript.add(Transcript.Kind.USER, "one");
        assertThat("written already, not at the end", Files.readString(live), containsString("one"));

        transcript.add(Transcript.Kind.AGENT, "two");

        List<String> lines = Files.readAllLines(live, StandardCharsets.UTF_8);
        assertThat(lines.size(), is(2));
        assertThat(lines.get(0), containsString("you: one"));
        assertThat(lines.get(1), containsString("agent: two"));
    }

    @Test
    void aFileThatCannotBeWrittenDoesNotEndTheSession() {
        // It exists to survive a bad ending, so it may not cause one.
        Transcript transcript = new Transcript(directory);

        transcript.add(Transcript.Kind.USER, "the path is a directory, not a file");

        assertThat("kept in memory regardless", transcript.size(), is(1));
    }

    @Test
    void clearingForgetsTheSession() {
        Transcript transcript = new Transcript();
        transcript.add(Transcript.Kind.USER, "hello");

        transcript.clear();

        assertThat(transcript.size(), is(0));
        assertThat(transcript.render(), not(containsString("hello")));
    }
}
