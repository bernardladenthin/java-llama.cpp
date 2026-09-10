<!--
SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>

SPDX-License-Identifier: MIT
-->

# The parameter wire surface: what was wrong, why it was wrong, and what replaced it

This library sends names on three wires — CLI options in the argv that loads a model, JSON keys in a
completion request, and JSON keys in a fine-tuning configuration. For most of its recorded history
nothing checked any of them against the code that reads them.

This file records the measurements behind the rework that changed that, so a later reader does not
have to re-derive them. Every number here was produced by running something, not by reading
code; the commands are given so they can be re-run.

## 1. Eleven names were dead, and most of them were born that way

The rework deleted eleven wire names in its first pass (a twelfth, `chat_template`, was found later by
the guard itself — see section 5a). The interesting part is not that they were dead — it is **when**
they died.

This repository's recorded history starts at commit `38f00b2`, which has **no parent**: the tree was
squashed at the fork from [`kherud/java-llama.cpp`](https://github.com/kherud/java-llama.cpp)
(fork point `49be664`, "bump pom.xml version 4.1.0 -> 4.20"). All eleven names are present at both
points. Checking each against the llama.cpp version pinned at the time:

| Name | at **b4916** (kherud's own pin) | at **b9994** (this repo's first commit) | at **b10883** |
|---|---|---|---|
| `tfs_z`, `penalize_nl`, `penalty_prompt`, `use_jinja` | **0 occurrences** in `examples/server` + `common` | 0 occurrences | 0 occurrences |
| `--grp-attn-n`, `--grp-attn-w` | present, `set_examples({MAIN, PASSKEY})` | `{COMPLETION, PASSKEY}` | `{COMPLETION, PASSKEY}` |
| `--dump-kv-cache` | present, no `set_examples` (so server-visible) | **gone** | gone |
| `--hf-repo-v`, `--hf-file-v` | present | gone (by b10456) | gone |
| `--mlock`, `--no-mmap` | present | present | **gone at b10878** |

**Six of the eleven were already non-functional at the fork point's own pin.** Not one of them ever
worked in this repository. Only two — `--mlock` / `--no-mmap` — are the "upstream moved and we lagged"
story that the b10878 bump told; the rest are older than that, and older than this repo.

The two example-scoped ones deserve their own note, because they defeat the obvious check.
`--grp-attn-n` and `--grp-attn-w` are **present in `common/arg.cpp` at every tag this project has ever
pinned**, so any textual sweep of upstream reports them alive. `common_params_parser_init`'s `add_opt`
filters by example at registration time, so they are never registered for `LLAMA_EXAMPLE_SERVER` and
the server parser rejects them exactly like a deleted option. Only the real option table knows this.

Reproducing the table:

```bash
# request keys at a given tag
git -C <llama.cpp> grep -c '"tfs_z"' b4916 -- examples/server common

# example scoping of a CLI option
git -C <llama.cpp> show b4916:common/arg.cpp | grep -A6 '"--grp-attn-n"' | grep set_examples
```

## 2. The design that made it possible, and where it came from

The fork point already contains the mechanism, in `de/kherud/llama/JsonParameters.java`:

```java
abstract class JsonParameters {
    // We save parameters directly as a String map here, to re-use as much as possible of the
    // (json-based) C++ code. The JNI code for a proper Java-typed data object is comparatively
    // too complex and hard to maintain.
    final Map<String, String> parameters = new HashMap<>();

    @Override public String toString() {
        builder.append("\t\"").append(key).append("\": ").append(value);
```

That `toString()` is character-for-character the one this project shipped until the rework, and the
comment is the documented trade: type-safety for JNI simplicity. Two consequences follow directly.

**A `String` key makes "a name with no counterpart" representable.** With ~200 builder setters each
writing a literal, the set of names that can reach a wire is whatever the sources happen to contain —
knowable only by scanning them, and never checked against the receiver. That is the shape of the
eleven above.

**A map of pre-serialized text has nowhere to validate.** The value is already a string by the time
it is stored, so a method that accepts a JSON *fragment* has no place to check it. That did not matter
at the fork point, where every value came from an internal serializer. It began to matter when this
fork added five methods that take a fragment from the caller.

## 3. A demonstrated field-injection defect

`withJsonSchema`, `withResponseFormat`, `withStreamOptions`, `withMessagesJson` and `withToolsJson`
each accepted a caller-supplied JSON fragment and stored it verbatim in that map. None of the five
exists at the fork point — the unsafe serializer is inherited, the entry points that exploit it are
this fork's own.

Because the document was assembled by string concatenation, a fragment carrying a top-level comma did
not only set its own field. Run against the built classes:

```java
InferenceParameters.of("hi").withNPredict(16).withCachePrompt(true)
    .withResponseFormat("{\"type\":\"json_object\"}, \"n_predict\": 999999, \"cache_prompt\": false")
```

```json
{
	"n_predict": 16,
	"response_format": {"type":"json_object"}, "n_predict": 999999, "cache_prompt": false,
	"prompt": "hi",
	"cache_prompt": true
}
```

`n_predict` and `cache_prompt` each appear twice. nlohmann resolves duplicate keys **last-wins**
(verified directly: `n_predict=999999 cache_prompt=0`), so the injected values won over the ones the
application had set. `InferenceParameters.toString()` was the request body — `LlamaModel` passed it
straight to `requestCompletion` / `handleChatCompletions` / `requestChatCompletionStream` — so any
host assembling tools, `response_format` or messages JSON from something it did not fully control
could have its own limits overridden.

**One detail worth keeping.** The obvious fix — parse the fragment with Jackson's `readTree` — is not
enough: `readTree` parses the first value and *ignores* what follows, so the payload above would have
been silently truncated to its first object. Rejecting it needs `FAIL_ON_TRAILING_TOKENS`. One quiet
wrong answer traded for another is not a fix.

## 4. What replaced it

Every wire name is now an enum constant carrying the contract it must satisfy, and the base classes
accept nothing else:

| Registry | Receiver it is checked against | Contract kinds |
|---|---|---|
| `args.ModelFlag` + `args.ModelOption` | `common_params_parser_init(params, LLAMA_EXAMPLE_SERVER).options` | `SERVER_PARSER`, `PROJECT_PSEUDO` |
| `parameters.RequestField` | `server_schema::make_llama_cmpl_schema(...)` | `SCHEMA`, `OAI_LAYER` |
| `parameters.TrainingField` | `jllama_train::config_keys()` | — |

`cmake/extract-java-wire-names.cmake` reads the registries at configure time and emits `{name,
contract}` pairs; `src/test/cpp/test_model_flags.cpp` and `src/test/cpp/test_wire_contracts.cpp` feed
them to the receivers above. Those run in `C++ Tests` on every platform.

Three properties are worth stating explicitly, because each closes a specific hole:

- **The contract lives on the constant, not in the test.** A list of exemptions inside a checking test
  is the thing that goes stale. `--vocab-only` declares itself `PROJECT_PSEUDO` where it is defined,
  and the eleven OAI-layer request keys declare themselves `OAI_LAYER` the same way.
- **The exemption checks are inverted.** Such a name cannot go stale by outliving its constant — it
  lives on it. It can go stale the other way: upstream may later register a name we exempted, at which
  point the exemption hides a real check. Both tests assert exactly that, plus that the exempt set is
  non-empty, so a generator that lost the contract column would not silently exempt everything.
- **Reachability is checked in the other direction too.** `WireNameRegistryTest` drives every public
  builder method reflectively and asserts every declared constant is actually emitted by one. A
  constant nothing emits is invisible to the C++ check as well — it would be fed to the receiver
  forever with no caller able to reach it.

Values are no longer raw text either: `JsonParameters` enforces that **every stored value is exactly
one well-formed JSON value**, checked centrally on write, and renders the body through Jackson with
keys in sorted order. `toString()` became a redacted, deliberately non-JSON debug view — a parameter
set carries the prompt, the message history and the tool definitions, so a log line built from one
used to leak the whole payload, and a caller who still passes it to the native layer now fails at the
parser instead of quietly sending a different body.

The invariant found a defect it was not written for on its first run: `JsonParameters.withEnum` stored
`getArgValue()` **unquoted** (`q8_0`, not `"q8_0"`), which is not a JSON value at all. It had no
production caller — the CLI side has its own `putEnum`, where a bare string is correct because argv is
not JSON — so it was deleted rather than fixed.

## 5. The trainer surface, which looked safest and had no guard at all

`TrainingParameters` → `train_engine.cpp` is a contract where both ends are ours, which is exactly why
it had nothing checking it. It reads with `j.value(key, default)`: a rename on either side does not
fail, it silently reverts one knob to its default. And `LlamaTrainerIntegrationTest` is gated on a
system property no CI job sets, so nothing runnable covered it.

At the time of the rework the two sides agreed exactly (15/15) — there was no live defect. The parser
now goes through a `jllama_train::keys` constant per field and `config_keys()` returns those same
constants, so a changed spelling moves both at once, and `JavaTrainingFieldContract` asserts the Java
registry and the engine list are equal in both directions.

## 5a. The exemption that proved a key dead, one commit after it was written

Section 4's `OAI_LAYER` contract says: this key is consumed by `oaicompat_*_params_parse` or the task
layer before `make_llama_cmpl_schema` ever sees the body, so do not expect the schema to know it. The
test asserted exactly that — the key is **not** in the schema — with the inverted-check reasoning in
rule 3: an exemption cannot rot by outliving its constant, only by upstream later adopting the name.

That reasoning was incomplete. Absence from the schema is satisfied equally well by a key **nothing
reads at all**, so the exemption was a hole exactly the size of the problem the registry was built to
close. `chat_template` sat in it: a public `InferenceParameters.withChatTemplate`, writing a key that
appears in upstream sources only where the server *emits* it, in the `/props` payload.

It could not be closed by driving the parser, because the eleven keys have three different consumers
(`oaicompat_chat_params_parse`, the completion/task layer, and `server-context.cpp`'s infill path) and
two of them need a live `server_context`. The oracle chosen instead is a **reader-shaped** sweep of the
receiver's own sources, run at configure time by the same generator:

```bash
# what the generator does, per OAI_LAYER key, over tools/server/*.cpp + common/*.cpp
grep -rEn 'json_value\([A-Za-z_.]+, *"<key>"|\.contains\("<key>"\)|\.at\("<key>"\)'
```

The *shape* is the whole point. `chat_template` does occur as a bare literal upstream, so a token grep
would have called it live; requiring it to appear in a position that reads it from a body does not.
Measured at b10883:

| key | readers | key | readers |
|---|---|---|---|
| `chat_template` | **0** | `parallel_tool_calls` | 1 |
| `chat_template_kwargs` | 1 | `prompt` | 8 |
| `id_slot` | 1 | `response_format` | 3 |
| `input_prefix` | 2 | `tool_choice` | 3 |
| `input_suffix` | 2 | `tools` | 9 |
| `messages` | 4 | | |

`JavaRequestFieldContract.EveryOaiLayerKeyIsReadSomewhereUpstream` fails on a count of zero, naming the
key and the remedy. Run against the tree that declared `chat_template`, that is exactly what it printed;
the constant and its builder method were then deleted under rule 2 (a name with no counterpart is
deleted, never deprecated).

One consumer was affected: the Android "LLM Service" app passed its chat-template override per request,
where llama.cpp discarded it. It now sets it at load time via `ModelParameters.setChatTemplate`. Two
tests had also pinned the dead key — a `ChatAdvancedTest` case asserting only that `applyTemplate` did
not throw (its own Javadoc explained the missing behavioural assertion with the wrong cause: the model's
built-in template winning, rather than the field never being read), and an `InferenceParametersTest`
case asserting the string mapping. Both were deleted with the method. This is the same shape as every
other entry in section 1: a green test pinning the mapping, never the contract.

## 6. What this does not cover

- The **failure path** of a dead name is still only checkable by the receiver's own tables. If
  upstream stops populating one of those tables, the guard degrades to a vacuous pass — which is why
  each test also asserts its oracle is populated before trusting its verdict.
- `test_wire_contracts.cpp` checks the request schema, not the OAI/task layer above it. The
  `OAI_LAYER` exemption is now checked from both sides — absent from the schema, and read by
  *something* upstream (section 5a) — but the reader sweep is a source pattern, not the parser. It
  proves a key is read from some request body; it does not prove *this* endpoint reads it, nor that
  it is read with the meaning the builder method documents.
- The **Java↔C++ trainer contract** is guarded; the C++↔C++ pairing inside `train_engine.cpp` rests on
  both sides being written against the same `keys` constants, not on a test.
