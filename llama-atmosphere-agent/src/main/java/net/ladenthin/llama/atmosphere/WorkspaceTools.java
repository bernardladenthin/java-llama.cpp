// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.atmosphere;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.atmosphere.ai.fs.AgentFileSystem;
import org.atmosphere.ai.fs.FileSystemTools;
import org.atmosphere.ai.fs.WorkspaceAgentFileSystem;
import org.atmosphere.ai.tool.ToolDefinition;
import org.atmosphere.ai.tool.ToolExecutor;
import org.atmosphere.ai.tool.ToolKind;
import org.atmosphere.ai.tool.ToolParameter;
import org.jspecify.annotations.Nullable;

/**
 * The three tools this agent provides itself, replacing the framework's versions of the same names:
 * {@code read_file}, {@code edit_file} and {@code grep}. Everything else — {@code ls},
 * {@code write_file}, {@code glob}, {@code delete}, {@code rename} — stays Atmosphere's.
 *
 * <p>They are replacements, not additions, on purpose: two tools that both claim to "read a file" is
 * the worst outcome for tool selection. Underneath they use the same {@link AgentFileSystem} the
 * framework binds to the session, so path validation, the workspace confinement and the size limits
 * are unchanged; only the parameters the model sees and the text it gets back are ours.
 *
 * <p>Why each one differs from the framework's:
 *
 * <ul>
 *   <li><b>read_file</b> takes {@code offset}/{@code limit} and prints line numbers. Reading a whole
 *       file costs context for nothing and measurably lowers task success (SWE-agent: 12.7 % with
 *       whole files against 18.0 % with a 100-line window).
 *   <li><b>edit_file</b> normalizes line endings (the framework compares raw content, so no edit ever
 *       matches in a CRLF file), explains a miss with the nearest lines, names the line numbers of an
 *       ambiguous match, and can apply several edits at once — see {@link TextEdits}.
 *   <li><b>grep</b> skips {@code .git}, build output and dependency directories — the framework's walk
 *       is alphabetical with global budgets, so those directories eat the result before the source is
 *       reached — and states when it truncated; see {@link WorkspaceSearch}.
 * </ul>
 */
public final class WorkspaceTools {

    /** Lines returned by {@code read_file} when the model gives no limit. */
    public static final int DEFAULT_READ_LIMIT = 400;

    /** Lines of context shown around a completed edit. */
    private static final int EDIT_SNIPPET_CONTEXT = 3;

    private WorkspaceTools() {}

    /**
     * Remembers which files were read, so an edit can insist on it.
     *
     * <p>Not a staleness check: an {@code old_string} that matches the current content exactly and
     * unambiguously is safe whether or not the file changed meanwhile. What this prevents is the
     * other case — a model inventing the text it wants to replace. One instance per session.
     */
    public static final class ReadTracker {
        private final Set<String> read = new LinkedHashSet<>();

        /**
         * Note that a file was read.
         *
         * @param path the path as the model wrote it
         */
        public void markRead(String path) {
            read.add(normalizePath(path));
        }

        /**
         * Whether a file was read in this session.
         *
         * @param path the path as the model wrote it
         * @return {@code true} when it was read before
         */
        public boolean wasRead(String path) {
            return read.contains(normalizePath(path));
        }

        private static String normalizePath(String path) {
            return path.replace('\\', '/').replaceAll("^\\./", "");
        }
    }

    /**
     * The tool set: Atmosphere's tools where they are fine, ours where they are not.
     *
     * @param tracker the read tracker shared by {@code read_file} and {@code edit_file}
     * @return the tools to offer the model
     */
    public static List<ToolDefinition> all(ReadTracker tracker) {
        return List.of(
                FileSystemTools.ls(),
                readFile(tracker),
                FileSystemTools.writeFile(),
                editFile(tracker),
                FileSystemTools.glob(),
                grep(),
                FileSystemTools.delete(),
                FileSystemTools.rename());
    }

    /**
     * {@code read_file}: a window of a file, with line numbers.
     *
     * @param tracker records the read so an edit is allowed afterwards
     * @return the tool
     */
    public static ToolDefinition readFile(ReadTracker tracker) {
        return ToolDefinition.builder(
                        FileSystemTools.READ_FILE,
                        "Read a file from the workspace. Returns the lines numbered as `  12: text`."
                                + " Reads at most " + DEFAULT_READ_LIMIT + " lines at a time; use offset and limit to"
                                + " page through a longer file. The line numbers are display only — never include them"
                                + " in old_string when you edit.")
                .parameter("file_path", "File path relative to the workspace root", "string", true)
                .parameter("offset", "First line to show, 1-based (default 1)", "integer", false)
                .parameter("limit", "How many lines to show (default " + DEFAULT_READ_LIMIT + ")", "integer", false)
                .returnType("string")
                .executor(withFileSystem((arguments, injectables) -> {
                    AgentFileSystem fs = fileSystem(injectables);
                    if (fs == null) {
                        return "File tools unavailable: no agent filesystem is bound to this session.";
                    }
                    String path = string(arguments, "file_path");
                    if (path == null) {
                        return "Error: file_path is required";
                    }
                    try {
                        List<String> lines = fs.read(path).lines().toList();
                        int offset = Math.max(1, integer(arguments, "offset", 1));
                        int limit = Math.max(1, integer(arguments, "limit", DEFAULT_READ_LIMIT));
                        tracker.markRead(path);
                        return window(lines, offset, limit);
                    } catch (IllegalArgumentException e) {
                        return "Error: " + e.getMessage();
                    }
                }))
                .kind(ToolKind.READ)
                .build();
    }

    /**
     * Render the requested window with line numbers and say what was left out.
     *
     * @param lines the file's lines
     * @param offset the first line, 1-based
     * @param limit how many lines
     * @return the text for the model
     */
    static String window(List<String> lines, int offset, int limit) {
        if (lines.isEmpty()) {
            return "(empty file)";
        }
        if (offset > lines.size()) {
            return "(offset " + offset + " is past the end; the file has " + lines.size() + " lines)";
        }
        int last = Math.min(lines.size(), offset + limit - 1);
        StringBuilder text = new StringBuilder();
        for (int i = offset; i <= last; i++) {
            text.append("  ").append(i).append(": ").append(lines.get(i - 1)).append(System.lineSeparator());
        }
        if (offset > 1 || last < lines.size()) {
            text.append("(showing lines ")
                    .append(offset)
                    .append("–")
                    .append(last)
                    .append(" of ")
                    .append(lines.size())
                    .append("; use offset/limit for the rest)");
        }
        return text.toString();
    }

    /**
     * {@code edit_file}: exact replacement with line-ending handling, corrective errors, and several
     * edits in one call.
     *
     * @param tracker enforces that the file was read first
     * @return the tool
     */
    public static ToolDefinition editFile(ReadTracker tracker) {
        ToolParameter edits = ToolParameter.ofArray(
                "edits",
                "Several replacements applied to the same file, in order. Use instead of"
                        + " old_string/new_string. All of them must match, or the file is left untouched.",
                false,
                new ToolParameter(
                        "edit",
                        "One replacement",
                        "object",
                        false,
                        List.of(),
                        null,
                        List.of(
                                new ToolParameter("old_string", "The exact text to replace", "string", true),
                                new ToolParameter("new_string", "The replacement", "string", true),
                                new ToolParameter("replace_all", "Replace every occurrence", "boolean", false))));
        return ToolDefinition.builder(
                        FileSystemTools.EDIT_FILE,
                        "Edit a file by replacing exact text. Read the file first and copy old_string from it"
                                + " (without the line numbers the read tool prints). old_string must match exactly"
                                + " once unless replace_all is set. Line endings are handled for you.")
                .parameter("file_path", "File path relative to the workspace root", "string", true)
                .parameter("old_string", "The exact text to replace", "string", false)
                .parameter("new_string", "The replacement text", "string", false)
                .parameter("replace_all", "Replace every occurrence instead of requiring exactly one", "boolean", false)
                .parameter(edits)
                .returnType("string")
                .executor(withFileSystem((arguments, injectables) -> {
                    AgentFileSystem fs = fileSystem(injectables);
                    if (fs == null) {
                        return "File tools unavailable: no agent filesystem is bound to this session.";
                    }
                    String path = string(arguments, "file_path");
                    if (path == null) {
                        return "Error: file_path is required";
                    }
                    List<TextEdits.Edit> list;
                    try {
                        list = parseEdits(arguments);
                    } catch (IllegalArgumentException e) {
                        return "Error: " + e.getMessage();
                    }
                    if (!tracker.wasRead(path)) {
                        return "Error: read " + path + " before editing it, so old_string comes from the file"
                                + " rather than from memory.";
                    }
                    try {
                        String original = fs.read(path);
                        String edited = TextEdits.apply(original, list);
                        fs.write(path, edited);
                        return "Edited " + path + " (" + list.size() + (list.size() == 1 ? " edit)" : " edits)")
                                + describe(edited, list);
                    } catch (TextEdits.EditException e) {
                        return "Error: " + e.getMessage();
                    } catch (IllegalArgumentException e) {
                        return "Error: " + e.getMessage();
                    }
                }))
                .kind(ToolKind.EDIT)
                .build();
    }

    private static String describe(String edited, List<TextEdits.Edit> edits) {
        TextEdits.Edit last = edits.get(edits.size() - 1);
        String snippet = TextEdits.snippet(
                TextEdits.normalize(edited), last.oldString(), last.newString(), EDIT_SNIPPET_CONTEXT);
        return snippet == null ? "" : System.lineSeparator() + snippet;
    }

    /**
     * Read the edits from the call, in either shape.
     *
     * @param arguments the tool arguments
     * @return the edits, in order
     * @throws IllegalArgumentException when neither shape is present or an entry is incomplete
     */
    static List<TextEdits.Edit> parseEdits(Map<String, Object> arguments) {
        Object many = arguments == null ? null : arguments.get("edits");
        if (many instanceof Iterable<?> entries) {
            List<TextEdits.Edit> edits = new ArrayList<>();
            for (Object entry : entries) {
                if (!(entry instanceof Map<?, ?> map)) {
                    throw new IllegalArgumentException(
                            "every entry of edits must be an object with old_string" + " and new_string");
                }
                Object oldString = map.get("old_string");
                Object newString = map.get("new_string");
                if (oldString == null) {
                    throw new IllegalArgumentException("every entry of edits needs old_string");
                }
                edits.add(new TextEdits.Edit(
                        oldString.toString(),
                        newString == null ? "" : newString.toString(),
                        Boolean.TRUE.equals(map.get("replace_all"))
                                || "true".equalsIgnoreCase(String.valueOf(map.get("replace_all")))));
            }
            if (edits.isEmpty()) {
                throw new IllegalArgumentException("edits was empty");
            }
            return edits;
        }
        String oldString = string(arguments, "old_string");
        if (oldString == null) {
            throw new IllegalArgumentException("old_string is required (or pass edits)");
        }
        String newString = string(arguments, "new_string");
        return List.of(
                new TextEdits.Edit(oldString, newString == null ? "" : newString, bool(arguments, "replace_all")));
    }

    /**
     * {@code grep}: search the workspace, skipping what is not source.
     *
     * @return the tool
     */
    public static ToolDefinition grep() {
        return ToolDefinition.builder(
                        FileSystemTools.GREP,
                        "Search the workspace with a regular expression. Returns matching lines grouped by file"
                                + " as `  12: text`, capped at " + WorkspaceSearch.MAX_TOTAL_MATCHES + " matches;"
                                + " .git, build output and dependency directories are skipped. Says so when the"
                                + " result is truncated.")
                .parameter("pattern", "The regular expression to search for", "string", true)
                .parameter("dir", "Directory to search under, relative to the workspace root", "string", false)
                .parameter("glob", "Only search files matching this name pattern, e.g. *.java", "string", false)
                .parameter("files_only", "Return just the file names instead of the matching lines", "boolean", false)
                .returnType("string")
                .executor(withFileSystem((arguments, injectables) -> {
                    AgentFileSystem fs = fileSystem(injectables);
                    if (!(fs instanceof WorkspaceAgentFileSystem workspace)) {
                        return "Search unavailable: this session has no workspace directory.";
                    }
                    String pattern = string(arguments, "pattern");
                    if (pattern == null) {
                        return "Error: pattern is required";
                    }
                    try {
                        Path root = workspace.root();
                        WorkspaceSearch.Result result = WorkspaceSearch.search(
                                root, pattern, string(arguments, "dir"), string(arguments, "glob"));
                        return WorkspaceSearch.format(result, bool(arguments, "files_only"));
                    } catch (IllegalArgumentException e) {
                        return "Error: " + e.getMessage();
                    }
                }))
                .kind(ToolKind.READ)
                .build();
    }

    /**
     * Adapt a two-argument function to {@link ToolExecutor}, whose single abstract method takes only
     * the arguments — the injectables arrive through its default overload, and that is where the
     * session's filesystem lives.
     *
     * @param body what the tool does
     * @return the executor
     */
    private static ToolExecutor withFileSystem(ToolBody body) {
        return new ToolExecutor() {
            @Override
            public Object execute(Map<String, Object> arguments) {
                return execute(arguments, Map.of());
            }

            @Override
            public Object execute(Map<String, Object> arguments, Map<Class<?>, Object> injectables) {
                return body.run(arguments, injectables);
            }
        };
    }

    /** The body of a tool: arguments plus the session's injectables. */
    @FunctionalInterface
    private interface ToolBody {
        /**
         * Run the tool.
         *
         * @param arguments the call's arguments
         * @param injectables the session scope, carrying the filesystem
         * @return what the model sees
         */
        Object run(Map<String, Object> arguments, Map<Class<?>, Object> injectables);
    }

    private static @Nullable AgentFileSystem fileSystem(@Nullable Map<Class<?>, Object> injectables) {
        return FileSystemTools.resolveFileSystem(injectables == null ? Map.of() : injectables)
                .orElse(null);
    }

    private static @Nullable String string(@Nullable Map<String, Object> arguments, String name) {
        Object value = arguments == null ? null : arguments.get(name);
        return value == null || value.toString().isEmpty() ? null : value.toString();
    }

    private static boolean bool(@Nullable Map<String, Object> arguments, String name) {
        Object value = arguments == null ? null : arguments.get(name);
        return Boolean.TRUE.equals(value) || "true".equalsIgnoreCase(String.valueOf(value));
    }

    private static int integer(@Nullable Map<String, Object> arguments, String name, int fallback) {
        Object value = arguments == null ? null : arguments.get(name);
        if (value instanceof Number number) {
            return number.intValue();
        }
        try {
            return value == null ? fallback : Integer.parseInt(value.toString().trim());
        } catch (NumberFormatException e) {
            return fallback;
        }
    }
}
