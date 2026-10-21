// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.loader;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.is;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.InvalidPathException;
import java.nio.file.Path;
import java.nio.file.Paths;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * {@link OfflineModelGuard#countsAsMissing}: which configured model paths the offline guard reports
 * as missing. The path parser is injected, because on Linux {@code Paths.get} throws for a name the
 * JVM's {@code sun.jnu.encoding} cannot encode (a POSIX-locale JVM and a non-ASCII name) although the
 * file exists and the native loader, which receives the UTF-8 bytes, opens it.
 */
public class OfflineModelGuardTest {

    /** What {@code UnixPath.encode} throws when {@code sun.jnu.encoding} cannot encode the name. */
    private static Path unencodable(String model) {
        throw new InvalidPathException(model, "Malformed input or input contains unmappable characters");
    }

    @Test
    public void countsAsMissing_aNameTheJvmEncodingCannotRepresent_isLeftToTheNativeLoader() {
        // An existing /models/größe.gguf in a container without LANG used to be reported as
        // "the configured model file does not exist", although it loads without --offline.
        assertThat(
                OfflineModelGuard.countsAsMissing(
                        "models/fehlt-😀.gguf", StandardCharsets.US_ASCII, OfflineModelGuardTest::unencodable),
                is(false));
    }

    @Test
    public void countsAsMissing_aNameTheFileSystemRejects_isMissing() {
        // the JVM could spell it, so the rejection is the file system's (a character Windows forbids)
        assertThat(
                OfflineModelGuard.countsAsMissing(
                        "models/bad?name.gguf", StandardCharsets.UTF_8, OfflineModelGuardTest::unencodable),
                is(true));
    }

    @Test
    public void countsAsMissing_aNulCharacter_isMissing() {
        assertThat(
                OfflineModelGuard.countsAsMissing("models/bad\u0000name.gguf", StandardCharsets.UTF_8, Paths::get),
                is(true));
    }

    @Test
    public void countsAsMissing_aNulCharacterWinsOverAnUnencodableName() {
        // native code would cut the path at the NUL, so it is never handed on
        assertThat(
                OfflineModelGuard.countsAsMissing(
                        "models/bad\u0000😀.gguf", StandardCharsets.US_ASCII, OfflineModelGuardTest::unencodable),
                is(true));
    }

    @Test
    public void countsAsMissing_anExistingAndAMissingFile(@TempDir Path dir) throws IOException {
        Path model = Files.createFile(dir.resolve("model.gguf"));

        assertThat(OfflineModelGuard.countsAsMissing(model.toString(), StandardCharsets.UTF_8, Paths::get), is(false));
        assertThat(
                OfflineModelGuard.countsAsMissing(
                        dir.resolve("absent.gguf").toString(), StandardCharsets.UTF_8, Paths::get),
                is(true));
    }

    @Test
    public void fileNameEncoding_isTheJnuEncodingExceptOnWindows() {
        // WindowsPath never consults sun.jnu.encoding, so a name it rejects is the file system's verdict
        assertThat(OfflineModelGuard.fileNameEncoding(false, "ANSI_X3.4-1968"), is(StandardCharsets.US_ASCII));
        assertThat(OfflineModelGuard.fileNameEncoding(true, "Cp1252"), is(StandardCharsets.UTF_8));
        assertThat(
                OfflineModelGuard.countsAsMissing(
                        "models/bad?\uD83D\uDE00.gguf",
                        OfflineModelGuard.fileNameEncoding(true, "Cp1252"),
                        OfflineModelGuardTest::unencodable),
                is(true));
    }

    @Test
    public void jnuEncoding_isTheCharsetOfTheProperty() {
        assertThat(OfflineModelGuard.jnuEncoding("UTF-8"), is(StandardCharsets.UTF_8));
        assertThat(OfflineModelGuard.jnuEncoding("ANSI_X3.4-1968"), is(StandardCharsets.US_ASCII));
    }

    @Test
    public void jnuEncoding_anAbsentOrUnknownNameFallsBackToTheDefaultCharset() {
        assertThat(OfflineModelGuard.jnuEncoding(null), is(java.nio.charset.Charset.defaultCharset()));
        assertThat(OfflineModelGuard.jnuEncoding("no-such-charset"), is(java.nio.charset.Charset.defaultCharset()));
        assertThat(OfflineModelGuard.jnuEncoding("bad name!"), is(java.nio.charset.Charset.defaultCharset()));
    }
}
