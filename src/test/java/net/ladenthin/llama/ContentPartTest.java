// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Base64;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

@ClaudeGenerated(
        purpose = "Factory contracts and data-URI shape for the multimodal ContentPart value type."
)
public class ContentPartTest {

    @TempDir
    Path tmp;

    @Test
    public void textPartCarriesText() {
        ContentPart p = ContentPart.text("hello");
        assertEquals(ContentPart.Type.TEXT, p.getType());
        assertEquals("hello", p.getText());
        assertNull(p.getImageUrl());
    }

    @Test
    public void imageUrlPartCarriesUrl() {
        ContentPart p = ContentPart.imageUrl("https://example.com/a.png");
        assertEquals(ContentPart.Type.IMAGE_URL, p.getType());
        assertEquals("https://example.com/a.png", p.getImageUrl());
        assertNull(p.getText());
    }

    @Test
    public void imageBytesProducesDataUri() {
        byte[] bytes = {1, 2, 3, 4, 5};
        ContentPart p = ContentPart.imageBytes(bytes, "image/png");
        String expected = "data:image/png;base64," + Base64.getEncoder().encodeToString(bytes);
        assertEquals(expected, p.getImageUrl());
    }

    @Test
    public void textRejectsNull() {
        assertThrows(NullPointerException.class, () -> ContentPart.text(null));
    }

    @Test
    public void imageUrlRejectsNull() {
        assertThrows(NullPointerException.class, () -> ContentPart.imageUrl(null));
    }

    @Test
    public void imageBytesRejectsNullBytes() {
        assertThrows(NullPointerException.class, () -> ContentPart.imageBytes(null, "image/png"));
    }

    @Test
    public void imageBytesRejectsNullMimeType() {
        assertThrows(NullPointerException.class, () -> ContentPart.imageBytes(new byte[]{0}, null));
    }

    @Test
    public void imageBytesRejectsEmptyMimeType() {
        assertThrows(IllegalArgumentException.class, () -> ContentPart.imageBytes(new byte[]{0}, ""));
    }

    @Test
    public void imageFileDetectsPngMime() throws IOException {
        Path file = tmp.resolve("logo.PNG");
        Files.write(file, new byte[]{(byte) 0x89, 0x50, 0x4E, 0x47});
        ContentPart p = ContentPart.imageFile(file);
        assertTrue(p.getImageUrl().startsWith("data:image/png;base64,"));
    }

    @Test
    public void imageFileDetectsJpegFromJpgExtension() throws IOException {
        Path file = tmp.resolve("photo.jpg");
        Files.write(file, new byte[]{(byte) 0xFF, (byte) 0xD8, (byte) 0xFF});
        ContentPart p = ContentPart.imageFile(file);
        assertTrue(p.getImageUrl().startsWith("data:image/jpeg;base64,"));
    }

    @Test
    public void imageFileDetectsJpegFromJpegExtension() throws IOException {
        Path file = tmp.resolve("photo.jpeg");
        Files.write(file, new byte[]{(byte) 0xFF, (byte) 0xD8, (byte) 0xFF});
        ContentPart p = ContentPart.imageFile(file);
        assertTrue(p.getImageUrl().startsWith("data:image/jpeg;base64,"));
    }

    @Test
    public void imageFileDetectsWebp() throws IOException {
        Path file = tmp.resolve("img.webp");
        Files.write(file, new byte[]{0x52, 0x49, 0x46, 0x46});
        ContentPart p = ContentPart.imageFile(file);
        assertTrue(p.getImageUrl().startsWith("data:image/webp;base64,"));
    }

    @Test
    public void imageFileDetectsGif() throws IOException {
        Path file = tmp.resolve("anim.gif");
        Files.write(file, new byte[]{0x47, 0x49, 0x46, 0x38});
        ContentPart p = ContentPart.imageFile(file);
        assertTrue(p.getImageUrl().startsWith("data:image/gif;base64,"));
    }

    @Test
    public void imageFileRejectsUnknownExtension() throws IOException {
        Path file = tmp.resolve("doc.txt");
        Files.write(file, "hello".getBytes());
        try {
            ContentPart.imageFile(file);
            fail("expected IllegalArgumentException for unknown extension");
        } catch (IllegalArgumentException expected) {
            assertNotNull(expected.getMessage());
        }
    }
}
