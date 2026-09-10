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
// The Java-side list is generated at configure time from the two registry enums, ModelFlag.java
// and ModelOption.java (cmake/extract-java-wire-names.cmake), so the two halves cannot drift:
// adding a builder method with a bad flag string reds this test without anyone remembering to
// update it. Each name carries the contract its own constant declares, so the exemption below is
// read from the registry rather than repeated here -- a list inside a test is exactly the thing
// that goes stale.
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

// A name the Java layer emits on purpose that llama.cpp is NOT expected to know declares itself
// as PROJECT_PSEUDO on its own enum constant. Today that is `--vocab-only`: `jllama.cpp`'s
// `loadModel` calls `strip_flag_from_argv(argv, argc, "--vocab-only", &vocab_only)` and removes it
// *before* `common_params_parse` ever sees the argv, using it to select the vocab-only path that
// owns its model directly and never starts a server_context. Marking a constant PROJECT_PSEUDO is
// a deliberate statement that some project code removes it from argv -- never a way to silence
// this test for a name that is simply dead.
bool is_project_pseudo(int index) { return std::string(JLLAMA_JAVA_CLI_CONTRACTS[index]) == "PROJECT_PSEUDO"; }

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
// failure mode this whole file exists to prevent. cmake/extract-java-wire-names.cmake enforces a
// floor of its own at configure time; this repeats it at the consuming end so a hand-edited or
// stale generated header cannot slip through either.
TEST(JavaCliFlagContract, GeneratedFlagListIsPopulated) {
    ASSERT_GT(JLLAMA_JAVA_CLI_COUNT, 50)
        << "the generated Java CLI flag list is empty or truncated -- the extractor is broken";
    ASSERT_EQ(JLLAMA_JAVA_CLI_COUNT,
              static_cast<int>(sizeof(JLLAMA_JAVA_CLI_NAMES) / sizeof(JLLAMA_JAVA_CLI_NAMES[0])));
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
    for (int i = 0; i < JLLAMA_JAVA_CLI_COUNT; ++i) {
        const std::string flag = JLLAMA_JAVA_CLI_NAMES[i];
        if (is_project_pseudo(i)) {
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

// The exemption is not allowed to rot either. It cannot go stale by naming something Java no
// longer emits -- it lives on the constant, so it disappears with it -- but it CAN go stale the
// other way: upstream may later register a name we exempted, at which point the exemption is
// hiding a real check rather than describing a real strip. Also assert the set is not empty, so a
// generator that dropped the contract column would not turn every name into a silent exemption.
TEST(JavaCliFlagContract, ProjectPseudoFlagsAreStillUnknownToTheServerParser) {
    const std::set<std::string> registered = server_registered_flags();

    int pseudo_count = 0;
    for (int i = 0; i < JLLAMA_JAVA_CLI_COUNT; ++i) {
        if (!is_project_pseudo(i)) {
            continue;
        }
        ++pseudo_count;
        const std::string flag = JLLAMA_JAVA_CLI_NAMES[i];
        EXPECT_EQ(registered.count(flag), 0u)
            << flag
            << " is declared PROJECT_PSEUDO but llama.cpp's server parser now registers it -- "
               "drop the contract override so the name is checked like every other";
    }
    EXPECT_EQ(pseudo_count, 1) << "expected exactly one PROJECT_PSEUDO name (--vocab-only); a "
                                  "different count means a contract was added, removed, or the "
                                  "generated contract column is missing";
}
