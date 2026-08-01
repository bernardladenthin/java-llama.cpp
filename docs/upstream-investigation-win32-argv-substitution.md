# `common_params_parse` can silently discard the caller's argv on Windows

Technical findings for llama.cpp. Everything below was verified against
`ggml-org/llama.cpp` `master` @ [`ddd4ec142`](https://github.com/ggml-org/llama.cpp/commit/ddd4ec1428a6201e18975ea52b07c71e0f9aef26)
(`chat : enable tool call in thinking for DS4 (#26269)`).

Downstream context: this is patch `0001` of [java-llama.cpp](https://github.com/bernardladenthin/java-llama.cpp),
which embeds `llama-server` in a JVM process and therefore builds its own `argv`.

Reported upstream as [ggml-org/llama.cpp#26416](https://github.com/ggml-org/llama.cpp/issues/26416)
(2026-08-01), which links back to this document and asks which of the two directions below the
maintainers prefer before a pull request is opened.

## Summary

On Windows, `common_params_parse` replaces the `argv` it was passed with one
reconstructed from the process command line, whenever the element counts happen to
match. The contents are never compared. A caller that constructs its own `argv` can
therefore have completely unrelated arguments parsed, with no warning and no way to
detect it from the return value.

This is observable today in llama.cpp's own test suite, without any patch.

## Affected code

[`common/arg.cpp:1203-1209`](https://github.com/ggml-org/llama.cpp/blob/ddd4ec1428a6201e18975ea52b07c71e0f9aef26/common/arg.cpp#L1203-L1209)

```c
bool common_params_parse(int argc, char ** argv, common_params & params, llama_example ex, void(*print_usage)(int, char **)) {
#ifdef _WIN32
    auto utf8 = make_utf8_argv();
    // repair argv only when it matches the process command line
    if (static_cast<int>(utf8.buf.size()) == argc) {
        argv = utf8.ptrs.data();
    }
#endif
```

`make_utf8_argv()` ([`common/arg.cpp:1180-1200`](https://github.com/ggml-org/llama.cpp/blob/ddd4ec1428a6201e18975ea52b07c71e0f9aef26/common/arg.cpp#L1180-L1200))
builds the UTF-8 argv from `GetCommandLineW()` via `CommandLineToArgvW()`. It reads the
**process** command line, independently of what the caller passed in.

Origin of the code:

- PR [#24779](https://github.com/ggml-org/llama.cpp/pull/24779) - `mtmd, arg: fix utf8 handling on windows`, merged 2026-06-19 by @ngxson
- Issue [#18571](https://github.com/ggml-org/llama.cpp/issues/18571) - the UTF-8 problems it addressed

The recovery itself is correct and needed. Only its placement inside the general parsing
entry point is the problem.

## Reproducer

No JNI, no patch, no special configuration - just `master` and the existing test binary,
invoked two different ways.

```
> test-arg-parser.exe
test-arg-parser: test invalid usage
test-arg-parser: all tests OK
exit 0

> test-arg-parser.exe -m spoofed.gguf
to show complete usage, run with -h
Assertion failed: false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON),
file X:\...\llama.cpp\tests\test-arg-parser.cpp, line 96
exit 0xC0000409
```

The failing assertion is a negative test,
[`tests/test-arg-parser.cpp:95-96`](https://github.com/ggml-org/llama.cpp/blob/ddd4ec1428a6201e18975ea52b07c71e0f9aef26/tests/test-arg-parser.cpp#L95-L96):

```c
// wrong value (int)
argv = {"binary_name", "-ngl", "hello"};
assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
```

Three elements. The process command line also had three tokens, so the constructed argv
was replaced by `{test-arg-parser.exe, -m, spoofed.gguf}` - which parses successfully,
while the test expects a parse failure.

Note that `test-arg-parser.cpp` declares `int main(void)` and never touches its own
arguments. `GetCommandLineW()` sees them regardless.

That this does not surface in CI is only due to the token count the test binary happens
to be invoked with there.

### Environment

| | |
|---|---|
| llama.cpp | `master` @ `ddd4ec142`, unmodified |
| OS | Windows 11 Pro 26200 |
| Compiler | MSVC 19.44.35228 (VS Build Tools 2022 17.14), Ninja |

```
cmake -S llama.cpp -B build -G Ninja -DCMAKE_BUILD_TYPE=Release ^
  -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF ^
  -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_UI=OFF ^
  -DLLAMA_OPENSSL=OFF -DLLAMA_SUBPROCESS=OFF -DGGML_OPENMP=OFF
cmake --build build --target test-arg-parser
```

## Who is affected

- `tests/test-arg-parser.cpp`, as shown above
- any embedded user of the `common` library that supplies its own `argv` rather than the
  process command line

## Possible directions

### 1. Separate the two meanings

`common_params_parse` parses exactly what it is given. A new `common_params_parse_main`
performs the Windows recovery first and is what the standalone tools' `main()` calls.

- the intent becomes explicit at the call site, and the library function stops second
  guessing its own parameters
- cost: 37 files, because every `main()` has to be moved over. The substantive part is
  about 30 lines in `common/arg.cpp` and `common/arg.h`; the rest is mechanical.

Reference implementation, based on `ddd4ec142`:

- branch: [`fix/win32-arg-parse-honor-caller-argv`](https://github.com/bernardladenthin/llama.cpp/tree/fix/win32-arg-parse-honor-caller-argv)
- diff: <https://github.com/ggml-org/llama.cpp/compare/master...bernardladenthin:llama.cpp:fix/win32-arg-parse-honor-caller-argv?expand=1>
- single commit `6468993c2`, 37 files, +67 / -42
- adds a regression test to `tests/test-arg-parser.cpp`

### 2. Tighten the condition

Compare the contents rather than only the element count, and substitute only on a full
match.

- much smaller, no API change, no call site churn
- risk: the comparison between the CRT `argv` and the ANSI round-trip of the wide argv
  may not hold under every codepage. Where it does not, the substitution stops happening
  and the fix from #24779 silently stops applying - breaking exactly what it protects.

Direction 1 avoids that risk because it never has to guess whether the caller's argv is
the process command line: the call site says so.
