// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.atmosphere;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import org.jspecify.annotations.Nullable;

/**
 * What was said in this session, in the order it was said, with the time.
 *
 * <p>It is not the conversation the model is sent. That one is rewritten by {@code /compact} — a
 * summary replaces the turns it summarises — and it never carried a time at all, so it cannot answer
 * "what did I ask before lunch" or "what did it actually reply". This record only ever grows.
 * {@code /compact} adds a note to it and changes nothing else; {@code /clear} empties it, because
 * that command means "forget this session" and leaving the text behind would make that untrue.
 *
 * <p><b>A list, not a map keyed by the time.</b> Two entries can share a millisecond — a tool result
 * and the answer that follows it regularly do — and a map would keep one of them and silently drop
 * the other. Insertion order already is time order, which is the only ordering anyone wants here.
 *
 * <p>With a file configured, every entry is also appended as it happens, so a session that is killed
 * still leaves what it had. Without one, {@code /save} writes the whole thing on request.
 */
public final class Transcript {

    private static final DateTimeFormatter STAMP = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");

    private static final DateTimeFormatter FILE_STAMP = DateTimeFormatter.ofPattern("yyyy-MM-dd_HH-mm-ss");

    /** Who said it. */
    public enum Kind {
        /** What the user typed. */
        USER("you"),
        /** What the model answered. */
        AGENT("agent"),
        /** A tool call and what it returned. */
        TOOL("tool"),
        /** Something the session did: compacted, interrupted, mode changed. */
        NOTE("note");

        private final String label;

        Kind(String label) {
            this.label = label;
        }

        /**
         * The word written in the file.
         *
         * @return the label
         */
        public String label() {
            return label;
        }
    }

    /**
     * One thing that was said.
     *
     * @param at when
     * @param kind who
     * @param text what, verbatim and uncut
     */
    public record Entry(LocalDateTime at, Kind kind, String text) {}

    private final List<Entry> entries = new CopyOnWriteArrayList<>();
    private final @Nullable Path liveFile;

    /**
     * Keep a session transcript in memory only.
     */
    public Transcript() {
        this(null);
    }

    /**
     * Keep a session transcript, and append every entry to a file as it happens.
     *
     * @param liveFile the file to append to, or {@code null} to keep it in memory
     */
    public Transcript(@Nullable Path liveFile) {
        this.liveFile = liveFile;
    }

    /**
     * Record something.
     *
     * <p>Blank text is dropped: an empty answer is not worth a line, and the file would fill with
     * them on a session of interrupted turns.
     *
     * @param kind who said it
     * @param text what was said
     */
    public void add(Kind kind, String text) {
        if (text == null || text.isBlank()) {
            return;
        }
        Entry entry = new Entry(LocalDateTime.now(), kind, text.strip());
        entries.add(entry);
        appendLive(entry);
    }

    /**
     * Everything recorded, oldest first.
     *
     * @return the entries
     */
    public List<Entry> entries() {
        return List.copyOf(entries);
    }

    /**
     * How many entries were recorded.
     *
     * @return the count
     */
    public int size() {
        return entries.size();
    }

    /** Forget the session, as {@code /clear} means it. */
    public void clear() {
        entries.clear();
    }

    /**
     * The whole transcript as text.
     *
     * @return one block per entry, oldest first
     */
    public String render() {
        StringBuilder text = new StringBuilder();
        for (Entry entry : entries) {
            text.append(format(entry));
        }
        return text.toString();
    }

    /**
     * Write the transcript to a file.
     *
     * @param directory where it goes, normally the workspace
     * @param name the file name, or {@code null} for one named after the time it was written
     * @return the file that was written
     * @throws IOException if it cannot be written
     */
    public Path save(Path directory, @Nullable String name) throws IOException {
        String fileName = name == null || name.isBlank()
                ? "transcript-" + LocalDateTime.now().format(FILE_STAMP) + ".txt"
                : name.strip();
        Path file = directory.resolve(fileName);
        Files.createDirectories(file.toAbsolutePath().getParent());
        Files.writeString(file, render(), StandardCharsets.UTF_8);
        return file;
    }

    private void appendLive(Entry entry) {
        if (liveFile == null) {
            return;
        }
        try {
            Path parent = liveFile.toAbsolutePath().getParent();
            if (parent != null) {
                Files.createDirectories(parent);
            }
            Files.writeString(
                    liveFile,
                    format(entry),
                    StandardCharsets.UTF_8,
                    StandardOpenOption.CREATE,
                    StandardOpenOption.APPEND);
        } catch (IOException e) {
            // A transcript that cannot be written must not end the session: the point of it is to
            // survive a bad ending, so it may not cause one.
        }
    }

    /**
     * One entry as it appears in the file.
     *
     * @param entry the entry
     * @return the block, ending in a newline
     */
    private static String format(Entry entry) {
        String separator = System.lineSeparator();
        return "[" + entry.at().format(STAMP) + "] " + entry.kind().label() + ": " + entry.text() + separator;
    }
}
