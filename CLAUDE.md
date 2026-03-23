# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Java bindings for [llama.cpp](https://github.com/ggerganov/llama.cpp) via JNI, providing a high-level API for LLM inference in Java. The Java layer communicates with a native C++ library through JNI.

Current llama.cpp pinned version: **b6545**

## Upgrading/Downgrading llama.cpp Version

To change the llama.cpp version, update the following **three** files:

1. **CMakeLists.txt** — Line 30: `GIT_TAG        b6545`
2. **README.md** — Line 2: Badge and link with version number
3. **CLAUDE.md** — Line 9: This documentation

Example: To upgrade from b5022 to b6545:
```bash
# Edit CMakeLists.txt, line 30: change b5022 to b6545
# Edit README.md, line 2: change b5022 to b6545 (in both badge and link)
# Edit CLAUDE.md, line 9: change b5022 to b6545
git add CMakeLists.txt README.md CLAUDE.md
git commit -m "Upgrade llama.cpp from b5022 to b6545"
git push -u origin claude/upgrade-llama-cpp-<session-id>
```

**Note:** Always test the build with `cmake -B build && cmake --build build --config Release` after version changes to catch compatibility issues early.

## Comparing llama.cpp Versions

To inspect what changed between two versions, use the GitHub compare URL:
```
https://github.com/ggml-org/llama.cpp/compare/<old>...<new>
```
Example: https://github.com/ggml-org/llama.cpp/compare/b5022...b6545

**Note:** Not every build number exists as a tag. To narrow down changes, compare smaller ranges (e.g. `b5022...b5025`). If a range is too large, GitHub may show "This comparison is taking too long to generate" — use smaller increments in that case.

### Major API Changes in Recent Versions (b5022→b6545)

**From b5022 to b5043:**
1. `common_grammar_trigger` removed `to_json`/`from_json` template methods
2. `common_params.model` changed from `std::string` to `common_params_model` struct
3. Model fields (`hf_file`, `hf_repo`, `model_url`) moved into nested `model` struct

**From b5043 to b5688 and beyond (b6545):**
1. Introduced wrapper classes (`server_grammar_trigger`, `server_tokens`) for cleaner abstraction
2. Added `common_chat_syntax` struct replacing `common_chat_format` enum
3. New sampling parameters (`top_n_sigma`) and reasoning format support
4. Expanded tool calling and speculative decoding support
5. Major refactoring of server code for better maintainability

## Build Commands

### Java (Maven)
```bash
mvn compile          # Compiles Java and generates JNI headers
mvn test             # Run all tests (requires native library and model files)
mvn package          # Build JAR
mvn test -Dtest=LlamaModelTest#testGenerate  # Run a single test method
```

### Native Library (CMake)
Must run `mvn compile` first to generate JNI headers, then:
```bash
# CPU only
cmake -B build
cmake --build build --config Release

# CUDA (Linux)
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release

# Metal (macOS)
cmake -B build -DLLAMA_METAL=ON
cmake --build build --config Release

# Optional: enable model downloading via URL
cmake -B build -DLLAMA_CURL=ON
```

Built libraries are placed in `src/main/resources/de/kherud/llama/{OS}/{ARCH}/`.

### Code Formatting
```bash
clang-format -i src/main/cpp/*.cpp src/main/cpp/*.hpp   # Format C++ code
```

## Architecture

### Two-Layer Design

**Java layer** (`src/main/java/de/kherud/llama/`):
- `LlamaModel` — Main API class (AutoCloseable). Wraps native context for inference, embeddings, re-ranking, and tokenization.
- `ModelParameters` / `InferenceParameters` — Builder-pattern parameter classes that serialize to JSON (extend `JsonParameters`) for passing to native code.
- `LlamaIterator` / `LlamaIterable` — Streaming generation via Java `Iterator`/`Iterable`.
- `LlamaLoader` — Extracts the platform-specific native library from the JAR to a temp directory, or finds it on `java.library.path`.
- `OSInfo` — Detects OS and architecture for library resolution.

**Native layer** (`src/main/cpp/`):
- `jllama.cpp` — JNI implementation bridging Java calls to llama.cpp.
- `server.hpp` — Inference server logic (adapted from llama.cpp's server).
- `utils.hpp` — Helper utilities.
- Uses `nlohmann/json` for JSON deserialization of parameters.

### Parameter Flow
Java parameters are serialized to JSON strings and passed to native code, which deserializes them using nlohmann/json. This avoids complex JNI field mapping for the many llama.cpp parameters.

### Native Library Resolution
`LlamaLoader` tries in order:
1. System property `de.kherud.llama.lib.path`
2. `java.library.path`
3. Extracts from JAR resources at `de/kherud/llama/{os}/{arch}/`

### Cross-compilation
Docker-based cross-compilation scripts are in `.github/dockcross/` for ARM/Android targets. CI workflows use these for non-x86 Linux builds.

## Testing

Tests require a model file. The CI downloads models from HuggingFace:
- **LlamaModel tests**: CodeLlama-7B-GGUF (`codellama-7b.Q2_K.gguf`)
- **RerankingModel tests**: Jina-Reranker model

Set the model path via system property or environment variable (see test files for exact property names).

Test files are in `src/test/java/de/kherud/llama/` and `src/test/java/examples/`.

## Key Constraints

- **Java 11+** required.
- Native memory allocated by llama.cpp is not GC-managed — always use `LlamaModel` in try-with-resources or call `close()` explicitly.
- The `server.hpp` file is adapted from llama.cpp upstream — minimize modifications to ease future upgrades.
- Platform-specific native libraries must be pre-built and placed under `src/main/resources/` before packaging for distribution.
