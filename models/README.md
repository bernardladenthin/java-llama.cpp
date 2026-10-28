# Test models

The model-backed tests look for their GGUF files here, as `models/<file>`. Nothing downloads them
automatically on a local checkout: a test whose model is missing skips itself, so the suite stays
green with any subset of the models present.

**The list is [`.github/models.csv`](../.github/models.csv)** — one
`filename,feature,sha256,url` row per model, each URL pinned to a Hugging Face commit. It is the
single source for both sides:

- **CI**: every job that needs models runs `.github/fetch-models.sh` (the composite action
  `.github/actions/fetch-models`) for the features it needs; a missing or changed model fails the
  job there instead of turning into a skip.
- **The tests**: each default path in `TestConstants` names a file of that list
  (`TestConstantsTest` checks that the two are the same set). A `-Dnet.ladenthin.llama.*` property
  only overrides a default; see the
  [System Properties Reference](../README.md#system-properties-reference).

To run the tests of one model locally, fetch its feature (the second column) with the same script,
from the repository root:

```bash
bash .github/fetch-models.sh draft          # the small draft model most tests use
bash .github/fetch-models.sh vision tts     # several features
bash .github/fetch-models.sh                # all of them (about 6.3 GB)
```

A file that is already here with the right SHA-256 is kept. With `HF_TOKEN` set in the environment
the downloads use that Hugging Face account's rate limit.

The tests find this directory both from the repository root and from the `llama/` module (they
also accept `llama/models/`). `*.gguf` files here are git-ignored.
