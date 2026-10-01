# Test models

The model-backed tests look for their GGUF files here, as `models/<file>`. Nothing downloads them
automatically on a local checkout: a test whose model is missing skips itself, so the suite stays
green with any subset of the models present.

**The list is [`.github/models.csv`](../.github/models.csv)** — one `filename,url` row per model.
It is the single source for both sides:

- **CI**: the `download-models` job of `publish.yml` fetches every row into a shared cache, which
  every test job restores; a missing model fails the job there instead of turning into a skip.
- **The tests**: each default path in `TestConstants` names a file of that list
  (`TestConstantsTest` checks that the two are the same set). A `-Dnet.ladenthin.llama.*` property
  only overrides a default; see the
  [System Properties Reference](../README.md#system-properties-reference).

To run the tests of one model locally, download its row:

```bash
# e.g. the small draft model most tests use
grep '^AMD-Llama-135m-code' .github/models.csv | while IFS=, read -r file url; do
  curl -L --fail -o "models/$file" "$url"
done
# or all of them (several GB)
grep -v '^#' .github/models.csv | while IFS=, read -r file url; do
  [ -n "$file" ] && curl -L --fail -C - -o "models/$file" "$url"
done
```

The tests find this directory both from the repository root and from the `llama/` module (they
also accept `llama/models/`). `*.gguf` files here are git-ignored.
