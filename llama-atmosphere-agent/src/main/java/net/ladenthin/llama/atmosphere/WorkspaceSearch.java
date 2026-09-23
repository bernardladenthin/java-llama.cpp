// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.atmosphere;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.FileVisitResult;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.SimpleFileVisitor;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.regex.PatternSyntaxException;

/**
 * Searching the workspace for text.
 *
 * <p><b>Why this exists instead of the framework's grep.</b> Atmosphere walks the workspace
 * <em>alphabetically</em> and spends one global 2-second deadline and one global 500-hit budget on
 * that walk. In any real project {@code .git} sorts before {@code src}, so both budgets are consumed
 * by the repository's own object store, the build output and {@code node_modules} before the source
 * is reached — the tool then reports a truncated result that does not contain the code at all.
 * Excluding those directories is the entire fix, and it is a filter in the walk.
 *
 * <p>The second reason is what the model reads. Results are grouped by file with line numbers (so a
 * ranged read can follow), every line is cut at {@value #MAX_LINE_CHARS} characters (a minified file
 * is otherwise a single 200 KB "line"), and truncation is always <em>stated</em>: a search that
 * silently returns part of the matches is read as a complete answer, which is the documented way this
 * class of tool misleads an agent.
 */
public final class WorkspaceSearch {

    /** Directories never searched, whatever the pattern. */
    public static final Set<String> EXCLUDED_DIRECTORIES = Set.of(
            ".git",
            ".hg",
            ".svn",
            "node_modules",
            "target",
            "build",
            "dist",
            "out",
            ".gradle",
            ".idea",
            ".mvn",
            ".venv",
            "venv",
            "__pycache__",
            ".cache",
            ".next",
            "vendor");

    /** Files larger than this are skipped: they are generated, not written. */
    public static final long MAX_FILE_BYTES = 2_000_000;

    /** A matching line is cut here. */
    public static final int MAX_LINE_CHARS = 200;

    /** Matches shown per file before the rest is summarized. */
    public static final int MAX_MATCHES_PER_FILE = 20;

    /** The whole search stops here. */
    public static final int MAX_TOTAL_MATCHES = 100;

    /** A pattern that backtracks is cut off after this. */
    public static final long DEADLINE_MILLIS = 5_000;

    private WorkspaceSearch() {}

    /**
     * One matching line.
     *
     * @param path the path relative to the workspace root, with {@code /} separators
     * @param line the 1-based line number
     * @param text the line, already cut to {@value #MAX_LINE_CHARS} characters
     */
    public record Hit(String path, int line, String text) {}

    /**
     * The result of a search.
     *
     * @param hits the matches, in walk order
     * @param filesWithMatches how many files matched, including those not shown
     * @param truncated whether the search stopped before the end
     */
    public record Result(List<Hit> hits, int filesWithMatches, boolean truncated) {}

    /**
     * Search below {@code root}.
     *
     * @param root the workspace root
     * @param pattern the regular expression
     * @param subdirectory the directory to search, relative to the root, or {@code null} for all
     * @param glob an optional file-name glob such as {@code *.java}, or {@code null} for all files
     * @return the matches
     * @throws IllegalArgumentException when the pattern is not a valid regular expression
     */
    public static Result search(Path root, String pattern, String subdirectory, String glob) {
        Pattern regex;
        try {
            regex = Pattern.compile(pattern);
        } catch (PatternSyntaxException e) {
            throw new IllegalArgumentException("Invalid regular expression: " + e.getMessage(), e);
        }
        Path start = subdirectory == null || subdirectory.isBlank()
                ? root
                : root.resolve(subdirectory).normalize();
        if (!start.startsWith(root) || !Files.isDirectory(start)) {
            return new Result(List.of(), 0, false);
        }
        java.nio.file.PathMatcher nameMatcher =
                glob == null || glob.isBlank() ? null : start.getFileSystem().getPathMatcher("glob:" + glob);

        List<Hit> hits = new ArrayList<>();
        Set<String> matchedFiles = new java.util.LinkedHashSet<>();
        long deadline = System.currentTimeMillis() + DEADLINE_MILLIS;
        boolean[] truncated = {false};
        try {
            Files.walkFileTree(start, Set.of(), Integer.MAX_VALUE, new SimpleFileVisitor<>() {
                @Override
                public FileVisitResult preVisitDirectory(Path directory, BasicFileAttributes attributes) {
                    String name = directory.getFileName() == null
                            ? ""
                            : directory.getFileName().toString();
                    return EXCLUDED_DIRECTORIES.contains(name) || (!directory.equals(start) && name.startsWith("."))
                            ? FileVisitResult.SKIP_SUBTREE
                            : FileVisitResult.CONTINUE;
                }

                @Override
                public FileVisitResult visitFile(Path file, BasicFileAttributes attributes) {
                    if (hits.size() >= MAX_TOTAL_MATCHES || System.currentTimeMillis() > deadline) {
                        truncated[0] = true;
                        return FileVisitResult.TERMINATE;
                    }
                    // Use the attributes the walk already has: a separate isRegularFile/size call per
                    // file goes through CreateFileW on Windows and dominates the walk.
                    if (!attributes.isRegularFile() || attributes.size() > MAX_FILE_BYTES) {
                        return FileVisitResult.CONTINUE;
                    }
                    if (nameMatcher != null && !nameMatcher.matches(file.getFileName())) {
                        return FileVisitResult.CONTINUE;
                    }
                    searchFile(root, file, regex, hits, matchedFiles, truncated);
                    return FileVisitResult.CONTINUE;
                }

                @Override
                public FileVisitResult visitFileFailed(Path file, IOException e) {
                    return FileVisitResult.CONTINUE;
                }
            });
        } catch (IOException e) {
            throw new IllegalArgumentException("Search failed: " + e.getMessage(), e);
        }
        return new Result(List.copyOf(hits), matchedFiles.size(), truncated[0]);
    }

    private static void searchFile(
            Path root, Path file, Pattern regex, List<Hit> hits, Set<String> matchedFiles, boolean[] truncated) {
        List<String> lines;
        try {
            // Binary content fails to decode as UTF-8 and is skipped -- the cheap equivalent of
            // ripgrep's "a file with a NUL byte is binary".
            lines = Files.readAllLines(file, StandardCharsets.UTF_8);
        } catch (IOException | RuntimeException e) {
            return;
        }
        String relative = root.relativize(file).toString().replace('\\', '/');
        int inThisFile = 0;
        for (int i = 0; i < lines.size(); i++) {
            if (hits.size() >= MAX_TOTAL_MATCHES) {
                truncated[0] = true;
                return;
            }
            Matcher matcher = regex.matcher(lines.get(i));
            if (!matcher.find()) {
                continue;
            }
            matchedFiles.add(relative);
            inThisFile++;
            if (inThisFile > MAX_MATCHES_PER_FILE) {
                truncated[0] = true;
                return;
            }
            String text = lines.get(i);
            hits.add(new Hit(
                    relative,
                    i + 1,
                    text.length() <= MAX_LINE_CHARS ? text : text.substring(0, MAX_LINE_CHARS) + " …[cut]"));
        }
    }

    /**
     * Render a result for the model: grouped by file, line-numbered, and honest about truncation.
     *
     * @param result the search result
     * @param filesOnly whether to list only the file names
     * @return the text the tool returns
     */
    public static String format(Result result, boolean filesOnly) {
        if (result.hits().isEmpty()) {
            return "(no matches)";
        }
        Map<String, List<Hit>> byFile = new LinkedHashMap<>();
        for (Hit hit : result.hits()) {
            byFile.computeIfAbsent(hit.path(), key -> new ArrayList<>()).add(hit);
        }
        StringBuilder text = new StringBuilder();
        if (filesOnly) {
            byFile.keySet().forEach(path -> text.append(path).append(System.lineSeparator()));
        } else {
            for (Map.Entry<String, List<Hit>> file : byFile.entrySet()) {
                text.append(file.getKey()).append(System.lineSeparator());
                for (Hit hit : file.getValue()) {
                    text.append("  ")
                            .append(hit.line())
                            .append(": ")
                            .append(hit.text())
                            .append(System.lineSeparator());
                }
                text.append(System.lineSeparator());
            }
        }
        text.append(result.hits().size())
                .append(" matches in ")
                .append(result.filesWithMatches())
                .append(" files");
        if (result.truncated()) {
            text.append(" (TRUNCATED — there are more; narrow the search with `glob`, `dir`"
                    + " or a more specific pattern)");
        }
        return text.toString();
    }
}
