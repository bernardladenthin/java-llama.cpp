// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT
//
// Contract test: every CLI flag the Java layer can emit must be one the llama.cpp server arg
// parser actually registers.
//
// WHY THIS IS NOT REDUNDANT WITH THE JAVA TESTS
// ---------------------------------------------
// `LlamaModel.loadModel(parameters.toArray())` hands the ModelParameters map to
// `common_params_parse(..., LLAMA_EXAMPLE_SERVER)` as argv. An option that parser does not
// know is a hard error, not a warning: `arg.cpp` throws, `common_params_parse` returns false,
// and `load_model_impl` throws `LlamaException("Failed to parse model parameters")`. So a flag
// upstream removed does not degrade the call -- it makes the model unloadable.
//
// The Java tests cannot see this. `ModelFlagTest` and `ModelParametersExtendedTest` assert the
// *string mapping* (`assertThat(p.parameters, hasKey("--mlock"))`), never that llama.cpp still
// accepts the string, so they pass forever while the flag is dead. That is how `--mlock` and
// `--no-mmap` survived their removal at b10878, and `--dump-kv-cache` / `--hf-repo-v` /
// `--hf-file-v` survived theirs for far longer.
//
// WHY A GREP OVER common/arg.cpp IS NOT A SUBSTITUTE
// ---------------------------------------------------
// `--grp-attn-n` and `--grp-attn-w` are *present* in `arg.cpp` at every tag this project has
// pinned, so any textual sweep reports them alive. They carry
// `.set_examples({LLAMA_EXAMPLE_COMPLETION, ...})`, and `common_params_parser_init`'s `add_opt`
// filters by example at registration time -- so they are never registered for
// `LLAMA_EXAMPLE_SERVER` and the server parser rejects them exactly like a deleted flag. Only
// the real option table knows this, which is why the oracle here is
// `common_params_parser_init(params, LLAMA_EXAMPLE_SERVER).options` rather than a file scan.
//
// The Java-side list is generated at configure time from ModelFlag.java + ModelParameters.java
// (cmake/extract-java-cli-flags.cmake), so the two halves cannot drift: adding a builder method
// with a bad flag string reds this test without anyone remembering to update it.
//
// Hermetic: no model, no JVM, no network. `common_params_parser_init` only fills a struct.

#include "arg.h"
#include "common.h"

#include "jllama_java_cli_flags.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <set>
#include <string>
#include <vector>

namespace {

// Flags the Java layer emits on purpose that llama.cpp is NOT expected to know.
//
// `--vocab-only` is this project's own pseudo-flag: `jllama.cpp`'s `loadModel` calls
// `strip_flag_from_argv(argv, argc, "--vocab-only", &vocab_only)` and removes it *before*
// `common_params_parse` ever sees the argv, using it to select the vocab-only path that owns
// its model directly and never starts a server_context. Adding an entry here is a deliberate
// statement that some project code removes the flag from argv -- never a way to silence this
// test for a flag that is simply dead.
const std::set<std::string> &project_only_flags() {
    static const std::set<std::string> flags = {"--vocab-only"};
    return flags;
}

// Every option string the server example registers, positive and negated forms alike.
std::set<std::string> server_registered_flags() {
    common_params params;
    common_params_context ctx = common_params_parser_init(params, LLAMA_EXAMPLE_SERVER);

    std::set<std::string> registered;
    for (const auto &opt : ctx.options) {
        for (const auto &arg : opt.get_args()) {
            registered.insert(arg);
        }
    }
    return registered;
}

} // namespace

// The extractor must never hand us an empty or obviously truncated list: a vacuous pass is the
// failure mode this whole file exists to prevent. cmake/extract-java-cli-flags.cmake enforces a
// floor of its own at configure time; this repeats it at the consuming end so a hand-edited or
// stale generated header cannot slip through either.
TEST(JavaCliFlagContract, GeneratedFlagListIsPopulated) {
    ASSERT_GT(JLLAMA_JAVA_CLI_FLAG_COUNT, 50)
        << "the generated Java CLI flag list is empty or truncated -- the extractor is broken";
    ASSERT_EQ(JLLAMA_JAVA_CLI_FLAG_COUNT,
              static_cast<int>(sizeof(JLLAMA_JAVA_CLI_FLAGS) / sizeof(JLLAMA_JAVA_CLI_FLAGS[0])));
}

// Sanity-check the oracle itself before trusting its verdict: if `common_params_parser_init`
// ever returned an empty table, every flag would "pass" the exemption path below instead.
TEST(JavaCliFlagContract, ServerOptionTableIsPopulated) {
    const std::set<std::string> registered = server_registered_flags();
    ASSERT_GT(registered.size(), 100u)
        << "common_params_parser_init(LLAMA_EXAMPLE_SERVER) returned an implausibly small option "
           "table -- the oracle is broken, not the Java layer";
    // Spot-check a flag every llama.cpp version this project supports has had.
    EXPECT_EQ(registered.count("--model"), 1u);
}

// THE contract. A failure here means a ModelParameters/ModelFlag member produces an argv that
// makes loadModel() throw -- fix the Java side (repoint, deprecate or remove it), do not add an
// exemption.
TEST(JavaCliFlagContract, EveryJavaEmittedFlagIsAcceptedByTheServerParser) {
    const std::set<std::string> registered = server_registered_flags();

    std::vector<std::string> rejected;
    for (int i = 0; i < JLLAMA_JAVA_CLI_FLAG_COUNT; ++i) {
        const std::string flag = JLLAMA_JAVA_CLI_FLAGS[i];
        if (project_only_flags().count(flag) != 0) {
            continue;
        }
        if (registered.count(flag) == 0) {
            rejected.push_back(flag);
        }
    }

    std::string message;
    for (const auto &flag : rejected) {
        message += "\n  " + flag;
    }
    EXPECT_TRUE(rejected.empty())
        << "The Java layer emits " << rejected.size()
        << " flag(s) that llama.cpp's server arg parser does not register. Every caller of the "
           "matching builder method gets 'Failed to parse model parameters' instead of a loaded "
           "model:"
        << message
        << "\nCheck whether upstream removed the option or narrowed its set_examples() away from "
           "LLAMA_EXAMPLE_SERVER, then repoint or retire the Java member.";
}

// The exemption list is not allowed to rot either: an entry that upstream later *does* register
// would silently stop being checked. (It is fine for a project-only flag to stay unknown to
// llama.cpp -- that is the point -- so this only asserts each entry is still emitted by Java.)
TEST(JavaCliFlagContract, EveryExemptedFlagIsStillEmittedByJava) {
    std::set<std::string> emitted;
    for (int i = 0; i < JLLAMA_JAVA_CLI_FLAG_COUNT; ++i) {
        emitted.insert(JLLAMA_JAVA_CLI_FLAGS[i]);
    }
    for (const auto &flag : project_only_flags()) {
        EXPECT_EQ(emitted.count(flag), 1u)
            << flag
            << " is exempted from the contract but no longer emitted by the Java layer -- "
               "drop the exemption";
    }
}
