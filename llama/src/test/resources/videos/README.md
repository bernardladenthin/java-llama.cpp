# Test resource video clips

## `sample.mp4`

Two seconds of video at four frames per second (8 frames, 128x96, H.264 in MP4, about 3.5 KB),
used by `VideoInputIntegrationTest` as the clip it sends to the vision model. The test counts the
prompt tokens the clip costs, so the frame count is part of the fixture: a clip with another length
or frame rate needs `CLIP_FRAMES` in the test changed with it.

### Provenance

Generated for this project with FFmpeg 6.1.1; the picture is FFmpeg's synthetic test source
(`testsrc`), not a recording:

```bash
ffmpeg -f lavfi -i "testsrc=duration=2:size=128x96:rate=4" -pix_fmt yuv420p -c:v libx264 \
       -preset veryslow -crf 30 -map_metadata -1 -fflags +bitexact -flags:v +bitexact sample.mp4
```

### License

The file is the output of that command; the project's author grants it under the project's MIT
license. The `SPDX-FileCopyrightText` annotation in `REUSE.toml` records that (an MP4 has no place
for an in-file header, same as `images/test-image.jpg` and `audios/sample.wav`).

### What the test needs besides it

The `ffmpeg` and `ffprobe` programs on `PATH`: llama.cpp decodes video by running them. Without
them the decoding cases skip themselves; with `JLLAMA_REQUIRE_FFMPEG` set they fail instead, which
is how the CI job that installs ffmpeg makes sure they ran.
