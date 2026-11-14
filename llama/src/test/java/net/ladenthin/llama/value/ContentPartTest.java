// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

package net.ladenthin.llama.value;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.notNullValue;
import static org.hamcrest.Matchers.nullValue;
import static org.hamcrest.Matchers.startsWith;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.fail;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Base64;
import net.ladenthin.llama.ClaudeGenerated;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

@ClaudeGenerated(purpose = "Factory contracts and data-URI shape for the multimodal ContentPart value type.")
public class ContentPartTest {

    @TempDir
    Path tmp;

    @Test
    public void textPartCarriesText() {
        ContentPart p = ContentPart.text("hello");
        assertThat(p.getType(), is(ContentPart.Type.TEXT));
        assertThat(p.getText(), is("hello"));
        assertThat(p.getImageUrl(), is(nullValue()));
    }

    @Test
    public void imageUrlPartCarriesUrl() {
        ContentPart p = ContentPart.imageUrl("https://example.com/a.png");
        assertThat(p.getType(), is(ContentPart.Type.IMAGE_URL));
        assertThat(p.getImageUrl(), is("https://example.com/a.png"));
        assertThat(p.getText(), is(nullValue()));
    }

    @Test
    public void imageBytesProducesDataUri() {
        byte[] bytes = {1, 2, 3, 4, 5};
        ContentPart p = ContentPart.imageBytes(bytes, "image/png");
        String expected = "data:image/png;base64," + Base64.getEncoder().encodeToString(bytes);
        assertThat(p.getImageUrl(), is(expected));
    }

    @Test
    public void inputAudioBase64EncodesAndNormalisesFormat() {
        byte[] bytes = {1, 2, 3, 4, 5};
        ContentPart p = ContentPart.inputAudio(bytes, "WAV");
        assertThat(p.getType(), is(ContentPart.Type.INPUT_AUDIO));
        assertThat(p.getAudioData(), is(Base64.getEncoder().encodeToString(bytes)));
        assertThat(p.getAudioFormat(), is("wav"));
        assertThat(p.getImageUrl(), is(nullValue()));
        assertThat(p.getText(), is(nullValue()));
    }

    @Test
    public void toolReferenceCarriesTheToolNameOnly() {
        ContentPart p = ContentPart.toolReference("buy");
        assertThat(p.getType(), is(ContentPart.Type.TOOL_REFERENCE));
        assertThat(p.getToolName(), is("buy"));
        assertThat(p.getText(), is(nullValue()));
        assertThat(p.getImageUrl(), is(nullValue()));
        assertThat(p.getAudioData(), is(nullValue()));
    }

    @Test
    public void toolReferenceRejectsAnEmptyOrMissingName() {
        assertThrows(IllegalArgumentException.class, () -> ContentPart.toolReference(""));
        assertThrows(NullPointerException.class, () -> ContentPart.toolReference(null));
    }

    @Test
    public void otherPartsCarryNoToolName() {
        assertThat(ContentPart.text("x").getToolName(), is(nullValue()));
    }

    @Test
    public void inputAudioRejectsUnsupportedFormat() {
        assertThrows(IllegalArgumentException.class, () -> ContentPart.inputAudio(new byte[] {1}, "ogg"));
    }

    @Test
    public void inputAudioRejectsNullBytes() {
        assertThrows(NullPointerException.class, () -> ContentPart.inputAudio(null, "wav"));
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
        assertThrows(NullPointerException.class, () -> ContentPart.imageBytes(new byte[] {0}, null));
    }

    @Test
    public void imageBytesRejectsEmptyMimeType() {
        assertThrows(IllegalArgumentException.class, () -> ContentPart.imageBytes(new byte[] {0}, ""));
    }

    @Test
    public void imageFileDetectsPngMime() throws IOException {
        Path file = tmp.resolve("logo.PNG");
        Files.write(file, new byte[] {(byte) 0x89, 0x50, 0x4E, 0x47});
        ContentPart p = ContentPart.imageFile(file);
        assertThat(p.getImageUrl(), startsWith("data:image/png;base64,"));
    }

    @Test
    public void imageFileDetectsJpegFromJpgExtension() throws IOException {
        Path file = tmp.resolve("photo.jpg");
        Files.write(file, new byte[] {(byte) 0xFF, (byte) 0xD8, (byte) 0xFF});
        ContentPart p = ContentPart.imageFile(file);
        assertThat(p.getImageUrl(), startsWith("data:image/jpeg;base64,"));
    }

    @Test
    public void imageFileDetectsJpegFromJpegExtension() throws IOException {
        Path file = tmp.resolve("photo.jpeg");
        Files.write(file, new byte[] {(byte) 0xFF, (byte) 0xD8, (byte) 0xFF});
        ContentPart p = ContentPart.imageFile(file);
        assertThat(p.getImageUrl(), startsWith("data:image/jpeg;base64,"));
    }

    @Test
    public void imageFileDetectsWebp() throws IOException {
        Path file = tmp.resolve("img.webp");
        Files.write(file, new byte[] {0x52, 0x49, 0x46, 0x46});
        ContentPart p = ContentPart.imageFile(file);
        assertThat(p.getImageUrl(), startsWith("data:image/webp;base64,"));
    }

    @Test
    public void imageFileDetectsGif() throws IOException {
        Path file = tmp.resolve("anim.gif");
        Files.write(file, new byte[] {0x47, 0x49, 0x46, 0x38});
        ContentPart p = ContentPart.imageFile(file);
        assertThat(p.getImageUrl(), startsWith("data:image/gif;base64,"));
    }

    @Test
    public void imageFileRejectsUnknownExtension() throws IOException {
        Path file = tmp.resolve("doc.txt");
        Files.writeString(file, "hello");
        try {
            ContentPart.imageFile(file);
            fail("expected IllegalArgumentException for unknown extension");
        } catch (IllegalArgumentException expected) {
            assertThat(expected.getMessage(), is(notNullValue()));
        }
    }

    @Test
    public void inputAudioAcceptsMp3Format() {
        byte[] bytes = new byte[] {9, 8, 7};
        ContentPart p = ContentPart.inputAudio(bytes, "MP3");
        assertThat(p.getAudioFormat(), is("mp3"));
        assertThat(p.getAudioData(), is(Base64.getEncoder().encodeToString(bytes)));
    }

    @Test
    public void audioFileDetectsWavFromExtension() throws IOException {
        Path file = tmp.resolve("clip.WAV");
        byte[] bytes = new byte[] {1, 2, 3, 4};
        Files.write(file, bytes);
        ContentPart p = ContentPart.audioFile(file);
        assertThat(p.getAudioFormat(), is("wav"));
        assertThat(p.getAudioData(), is(Base64.getEncoder().encodeToString(bytes)));
    }

    @Test
    public void audioFileDetectsMp3FromExtension() throws IOException {
        Path file = tmp.resolve("clip.mp3");
        byte[] bytes = new byte[] {5, 6, 7, 8};
        Files.write(file, bytes);
        ContentPart p = ContentPart.audioFile(file);
        assertThat(p.getAudioFormat(), is("mp3"));
        assertThat(p.getAudioData(), is(Base64.getEncoder().encodeToString(bytes)));
    }

    @Test
    public void audioFileRejectsUnknownExtension() throws IOException {
        Path file = tmp.resolve("clip.ogg");
        Files.write(file, new byte[] {0});
        try {
            ContentPart.audioFile(file);
            fail("expected IllegalArgumentException for unknown audio extension");
        } catch (IllegalArgumentException expected) {
            assertThat(expected.getMessage(), is(notNullValue()));
        }
    }

    @Test
    public void videoUrlKeepsTheUrlAndNothingElse() {
        ContentPart p = ContentPart.videoUrl("https://example.org/clip.mp4");
        assertThat(p.getType(), is(ContentPart.Type.VIDEO_URL));
        assertThat(p.getVideoUrl(), is("https://example.org/clip.mp4"));
        assertThat(p.getImageUrl(), is(nullValue()));
        assertThat(p.getText(), is(nullValue()));
        assertThat(p.getAudioData(), is(nullValue()));
        assertThat(p.getToolName(), is(nullValue()));
        assertThrows(NullPointerException.class, () -> ContentPart.videoUrl(null));
    }

    @Test
    public void aPartThatIsNoVideoHasNoVideoUrl() {
        assertThat(ContentPart.text("hello").getVideoUrl(), is(nullValue()));
        assertThat(ContentPart.imageUrl("data:image/png;base64,AAAA").getVideoUrl(), is(nullValue()));
        assertThat(ContentPart.toolReference("search").getVideoUrl(), is(nullValue()));
    }

    @Test
    public void videoBytesWrapsTheBytesInADataUri() {
        byte[] bytes = new byte[] {0, 1, 2, (byte) 0xFF};
        ContentPart p = ContentPart.videoBytes(bytes, "video/mp4");
        assertThat(p.getType(), is(ContentPart.Type.VIDEO_URL));
        assertThat(
                p.getVideoUrl(),
                is("data:video/mp4;base64," + Base64.getEncoder().encodeToString(bytes)));
    }

    @Test
    public void videoBytesRejectsATypeThatIsNoVideo() {
        byte[] bytes = new byte[] {1};
        assertThrows(IllegalArgumentException.class, () -> ContentPart.videoBytes(bytes, "image/png"));
        assertThrows(IllegalArgumentException.class, () -> ContentPart.videoBytes(bytes, ""));
        // the prefix alone names no type; one more character does
        assertThrows(IllegalArgumentException.class, () -> ContentPart.videoBytes(bytes, "video/"));
        assertThat(ContentPart.videoBytes(bytes, "video/x").getVideoUrl(), startsWith("data:video/x;base64,"));
        assertThrows(NullPointerException.class, () -> ContentPart.videoBytes(null, "video/mp4"));
        assertThrows(NullPointerException.class, () -> ContentPart.videoBytes(bytes, null));
    }

    @Test
    public void videoFileDetectsTheTypeFromTheExtension() throws IOException {
        byte[] bytes = new byte[] {9, 8, 7};
        String encoded = Base64.getEncoder().encodeToString(bytes);
        String[][] expected = {
            {"clip.mp4", "video/mp4"},
            {"clip.M4V", "video/mp4"},
            {"clip.mov", "video/quicktime"},
            {"clip.webm", "video/webm"},
            {"clip.MKV", "video/x-matroska"},
            {"clip.avi", "video/x-msvideo"},
        };
        for (String[] row : expected) {
            Path file = tmp.resolve(row[0]);
            Files.write(file, bytes);
            assertThat(row[0], ContentPart.videoFile(file).getVideoUrl(), is("data:" + row[1] + ";base64," + encoded));
        }
    }

    @Test
    public void videoFileRejectsAnUnknownExtensionAndAPathWithoutAName() throws IOException {
        Path file = tmp.resolve("clip.txt");
        Files.write(file, new byte[] {0});
        IllegalArgumentException unknown =
                assertThrows(IllegalArgumentException.class, () -> ContentPart.videoFile(file));
        assertThat(unknown.getMessage(), startsWith("Cannot infer MIME type from extension: "));
        Path root = tmp.getRoot();
        IllegalArgumentException nameless =
                assertThrows(IllegalArgumentException.class, () -> ContentPart.videoFile(root));
        assertThat(nameless.getMessage(), startsWith("videoPath has no file name component: "));
        assertThrows(NullPointerException.class, () -> ContentPart.videoFile(null));
    }
}
