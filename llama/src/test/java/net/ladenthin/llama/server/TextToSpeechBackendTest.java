// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.server;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.equalTo;
import static org.hamcrest.Matchers.is;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.attribute.PosixFilePermissions;
import org.junit.jupiter.api.Test;

/** The temporary file a speaker reference crosses to the native engine in; no native library. */
class TextToSpeechBackendTest {

    private static final byte[] CLIP = "RIFF-not-really".getBytes(StandardCharsets.US_ASCII);

    @Test
    void theReferenceFileHoldsTheClipAndIsGoneAfterTheDelete() throws IOException {
        Path file = TextToSpeechBackend.writeReferenceFile(CLIP);
        try {
            assertThat(Files.readAllBytes(file), is(equalTo(CLIP)));
        } finally {
            TextToSpeechBackend.deleteReferenceFile(file);
        }
        assertThat(Files.exists(file), is(false));
    }

    @Test
    void theReferenceFileIsReadableByItsOwnerOnly() throws IOException {
        assumeTrue(FileSystems.getDefault().supportedFileAttributeViews().contains("posix"));
        Path file = TextToSpeechBackend.writeReferenceFile(CLIP);
        try {
            assertThat(PosixFilePermissions.toString(Files.getPosixFilePermissions(file)), is(equalTo("rw-------")));
        } finally {
            TextToSpeechBackend.deleteReferenceFile(file);
        }
    }

    @Test
    void deletingAFileThatIsAlreadyGoneIsNotAnError() throws IOException {
        Path file = TextToSpeechBackend.writeReferenceFile(CLIP);
        TextToSpeechBackend.deleteReferenceFile(file);
        TextToSpeechBackend.deleteReferenceFile(file);
        assertThat(Files.exists(file), is(false));
    }
}
