# Test resource audio clips

## `sample.wav`

A ~2.6 s stereo 16-bit 48 kHz WAV (~488 KB) used by
`AudioInputIntegrationTest` as the committed default audio prompt, so the
audio-input path can be exercised end-to-end without staging a clip
out-of-band. The model pair defaults to CI's Ultravox v0.5 over
Llama-3.2-1B (feature `audio` of `.github/models.csv`); the
`net.ladenthin.llama.audio.model` / `.mmproj` properties override it.

### Provenance

Recorded and provided by Bernard Ladenthin (project copyright holder)
specifically for use as a test fixture in this project.

### License

The author grants this file for use in this project under the project's
MIT license. The `SPDX-FileCopyrightText: 2026 Bernard Ladenthin
<bernard.ladenthin@gmail.com>` annotation in `REUSE.toml` records that
grant (WAV carries no comment/metadata channel for an in-file header, so
the annotation lives in `REUSE.toml`, same as `images/test-image.jpg`).

### Override

To point the test at a different clip without overwriting this one, set
the `net.ladenthin.llama.audio.input` system property on the `mvn test`
command line. `.wav` and `.mp3` work; the file extension drives format
detection in `ContentPart.audioFile(Path)`.
