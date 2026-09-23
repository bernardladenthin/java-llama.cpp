// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.atmosphere;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.not;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.List;
import org.junit.jupiter.api.Test;

/** The matching and rewriting rules that decide whether an edit lands. */
class TextEditsTest {

    private static TextEdits.Edit edit(String oldString, String newString) {
        return new TextEdits.Edit(oldString, newString, false);
    }

    @Test
    void aCrlfFileIsEditedByAModelThatOnlyWritesLineFeeds() {
        // THE bug this class exists for: the framework compares raw content, so on Windows the
        // model's "a\nb" never matches the file's "a\r\nb" and every edit silently fails.
        String file = "int a = 1;\r\nint b = 2;\r\n";

        String edited = TextEdits.apply(file, List.of(edit("int a = 1;\nint b = 2;", "int a = 3;\nint b = 4;")));

        assertThat(edited, is("int a = 3;\r\nint b = 4;\r\n"));
    }

    @Test
    void theFilesOwnLineEndingAndByteOrderMarkSurvive() {
        assertThat(TextEdits.apply("﻿alpha\r\nbeta\r\n", List.of(edit("beta", "gamma"))), is("﻿alpha\r\ngamma\r\n"));
        assertThat(TextEdits.apply("alpha\nbeta\n", List.of(edit("beta", "gamma"))), is("alpha\ngamma\n"));
        assertThat(TextEdits.lineEnding("a\r\nb"), is("\r\n"));
        assertThat(TextEdits.lineEnding("a\nb"), is("\n"));
        assertThat(TextEdits.lineEnding("no newline at all"), is("\n"));
    }

    @Test
    void anAmbiguousMatchNamesTheLineNumbers() {
        String file = "x();\ny();\nx();\n";

        TextEdits.EditException error =
                assertThrows(TextEdits.EditException.class, () -> TextEdits.apply(file, List.of(edit("x();", "z();"))));

        assertThat(error.getMessage(), containsString("occurs 2 times"));
        assertThat(
                "naming the lines is what makes the next attempt possible",
                error.getMessage(),
                containsString("lines 1, 3"));
        assertThat(error.getMessage(), containsString("replace_all"));
    }

    @Test
    void replaceAllTakesEveryOccurrence() {
        assertThat(
                TextEdits.apply("x();\ny();\nx();\n", List.of(new TextEdits.Edit("x();", "z();", true))),
                is("z();\ny();\nz();\n"));
    }

    @Test
    void aMissShowsTheNearestLinesInsteadOfJustSayingNo() {
        String file = "public void handle(String name) {\n    log(name);\n}\n";

        TextEdits.EditException error = assertThrows(
                TextEdits.EditException.class,
                () -> TextEdits.apply(file, List.of(edit("public void handle(String value) {", "x"))));

        assertThat(error.getMessage(), containsString("not found"));
        assertThat(error.getMessage(), containsString("closest lines"));
        assertThat(error.getMessage(), containsString("1: public void handle(String name) {"));
    }

    @Test
    void aMissWithNothingSimilarSaysToReadTheFileAgain() {
        TextEdits.EditException error = assertThrows(
                TextEdits.EditException.class,
                () -> TextEdits.apply("alpha\nbeta\n", List.of(edit("zzzzzzzzzzzzzzzz", "x"))));

        assertThat(error.getMessage(), containsString("read the file again"));
        // and it warns about the trap that causes many of these misses
        assertThat(error.getMessage(), containsString("line numbers"));
    }

    @Test
    void severalEditsAreAllAppliedOrNoneAreAtAll() {
        String file = "one\ntwo\nthree\n";

        String edited = TextEdits.apply(file, List.of(edit("one", "1"), edit("three", "3")));
        assertThat(edited, is("1\ntwo\n3\n"));

        TextEdits.EditException error = assertThrows(
                TextEdits.EditException.class,
                () -> TextEdits.apply(file, List.of(edit("one", "1"), edit("absent", "x"))));
        assertThat(error.getMessage(), containsString("edit 2 of 2"));
    }

    @Test
    void anEmptyOrUnchangedEditIsRejectedRatherThanSilentlyDoingNothing() {
        assertThat(
                assertThrows(TextEdits.EditException.class, () -> TextEdits.apply("a\n", List.of(edit("", "x"))))
                        .getMessage(),
                containsString("must not be empty"));
        assertThat(
                assertThrows(TextEdits.EditException.class, () -> TextEdits.apply("a\n", List.of(edit("a", "a"))))
                        .getMessage(),
                containsString("identical"));
        assertThat(
                assertThrows(TextEdits.EditException.class, () -> TextEdits.apply("a\n", List.of()))
                        .getMessage(),
                containsString("No edits"));
    }

    @Test
    void theAnswerShowsTheEditedRegionWithLineNumbers() {
        String edited = TextEdits.apply("a\nb\nc\nd\ne\n", List.of(edit("c", "CHANGED")));

        String snippet = TextEdits.snippet(TextEdits.normalize(edited), "c", "CHANGED", 1);

        assertThat(snippet, containsString("3: CHANGED"));
        assertThat(snippet, containsString("2: b"));
        assertThat("the whole file is never echoed back", snippet, not(containsString("5: e")));
    }

    @Test
    void similarityIsBetweenZeroAndOneAndOrdersTheCandidates() {
        assertThat(TextEdits.similarity("abc", "abc"), is(1.0));
        assertThat(TextEdits.similarity("", "abc"), is(0.0));
        assertThat(TextEdits.similarity("int a = 1;", "int a = 2;") > TextEdits.CANDIDATE_MIN_SIMILARITY, is(true));
        assertThat(
                TextEdits.similarity("int a = 1;", "completely different") < TextEdits.CANDIDATE_MIN_SIMILARITY,
                is(true));
    }
}
