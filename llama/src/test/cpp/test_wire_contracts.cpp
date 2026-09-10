// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT
//
// Contract tests for the two wire surfaces that are NOT argv: the request body llama.cpp's server
// schema reads, and the fine-tuning configuration this project's own train_engine.cpp reads.
//
// WHY THESE ARE THE QUIETER HALF
// ------------------------------
// The CLI contract (test_model_flags.cpp) is guarded because an unregistered option is a hard
// parse error -- loud, if late. These two are worse: both receivers *silently ignore* a key they
// do not know. `server_schema::eval_llama_cmpl_schema` walks its field table and skips anything
// absent from it, and `train_engine.cpp` reads with `j.value(key, default)`, which falls back to
// the default. So a field that stops being read produces no error at all, anywhere -- the
// parameter simply stops having an effect, and every Java test that asserts the string mapping
// keeps passing.
//
// That is not hypothetical. Four request keys -- tfs_z, penalize_nl, penalty_prompt and use_jinja
// -- were emitted by this library for its entire recorded history while appearing nowhere in
// llama.cpp's server or common sources; six of the eleven names retired in this series were
// already dead at the fork point's own pin. Review found them, twice. A gate finds them once.
//
// Hermetic: no model, no JVM, no network. `make_llama_cmpl_schema` only builds a field table, and
// `jllama_train::config_keys()` is a header-only list.

#include "server-common.h"
#include "server-task.h"

#include "server-schema.h"

#include "train_engine.h"

#include "jllama_java_request_fields.h"
#include "jllama_java_training_fields.h"

#include <gtest/gtest.h>

#include <memory>
#include <set>
#include <string>
#include <vector>

namespace {

// Every key llama.cpp's completion-request schema accepts: each field's primary name plus its
// aliases, and the subfields of a nested field under their dotted path.
void collect_schema_keys(const std::vector<std::unique_ptr<server_schema::field>> &fields, const std::string &prefix,
                         std::set<std::string> &out) {
    for (const auto &f : fields) {
        for (const char *name : f->name) {
            out.insert(prefix + name);
        }
        if (auto *nested = dynamic_cast<server_schema::field_nested *>(f.get())) {
            collect_schema_keys(nested->subfields, prefix + f->name.at(0) + ".", out);
        }
    }
}

std::set<std::string> schema_keys() {
    common_params params_base;
    task_params params;
    auto schema = server_schema::make_llama_cmpl_schema(params_base, params);

    std::set<std::string> keys;
    collect_schema_keys(schema, "", keys);
    return keys;
}

bool is_oai_layer(int index) { return std::string(JLLAMA_JAVA_REQUEST_CONTRACTS[index]) == "OAI_LAYER"; }

} // namespace

// The extractor must never hand us an empty or truncated list: a vacuous pass is the failure mode
// these files exist to prevent.
TEST(JavaRequestFieldContract, GeneratedFieldListIsPopulated) {
    ASSERT_GT(JLLAMA_JAVA_REQUEST_COUNT, 30)
        << "the generated request-key list is empty or truncated -- the extractor is broken";
    ASSERT_EQ(JLLAMA_JAVA_REQUEST_COUNT,
              static_cast<int>(sizeof(JLLAMA_JAVA_REQUEST_NAMES) / sizeof(JLLAMA_JAVA_REQUEST_NAMES[0])));
}

// Sanity-check the oracle before trusting its verdict: an empty field table would make every key
// below look accepted-by-omission rather than checked.
TEST(JavaRequestFieldContract, SchemaFieldTableIsPopulated) {
    const std::set<std::string> keys = schema_keys();
    ASSERT_GT(keys.size(), 40u) << "make_llama_cmpl_schema returned an implausibly small field "
                                   "table -- the oracle is broken, not the Java layer";
    // Spot-check a field every llama.cpp version this project supports has read.
    EXPECT_EQ(keys.count("temperature"), 1u);
}

// THE contract. A failure here means an InferenceParameters wither writes a key the server throws
// away, so the setter looks like configuration and behaves like a no-op.
TEST(JavaRequestFieldContract, EverySchemaKeyIsAcceptedByTheRequestSchema) {
    const std::set<std::string> keys = schema_keys();

    std::vector<std::string> rejected;
    for (int i = 0; i < JLLAMA_JAVA_REQUEST_COUNT; ++i) {
        if (is_oai_layer(i)) {
            continue;
        }
        const std::string key = JLLAMA_JAVA_REQUEST_NAMES[i];
        if (keys.count(key) == 0) {
            rejected.push_back(key);
        }
    }

    std::string message;
    for (const auto &key : rejected) {
        message += "\n  " + key;
    }
    EXPECT_TRUE(rejected.empty())
        << "The Java layer writes " << rejected.size()
        << " request key(s) llama.cpp's schema does not read. The server discards an unknown key "
           "without a word, so the matching wither is a silent no-op:"
        << message
        << "\nEither upstream removed the field (retire the RequestField constant) or it moved to "
           "the OAI/task layer (declare it RequestContract.OAI_LAYER).";
}

// The OAI_LAYER exemption must stay honest. A key declared as "consumed before the schema" that
// the schema now DOES read is a contract that has quietly stopped describing reality -- harmless
// today, but it means the constant is no longer being checked against anything. Assert the set is
// non-empty too, so a generator that dropped the contract column cannot turn every key into a
// silent exemption.
TEST(JavaRequestFieldContract, OaiLayerKeysAreStillOutsideTheSchema) {
    const std::set<std::string> keys = schema_keys();

    int oai_count = 0;
    for (int i = 0; i < JLLAMA_JAVA_REQUEST_COUNT; ++i) {
        if (!is_oai_layer(i)) {
            continue;
        }
        ++oai_count;
        const std::string key = JLLAMA_JAVA_REQUEST_NAMES[i];
        EXPECT_EQ(keys.count(key), 0u)
            << key
            << " is declared OAI_LAYER but the request schema now reads it -- drop the override so "
               "it is checked like every other key";
    }
    EXPECT_GT(oai_count, 0) << "no key is declared OAI_LAYER; the generated contract column is "
                               "missing, which would exempt nothing and check everything by luck";
}

// The trainer contract. Both ends are ours, which makes it easy to assume it cannot drift -- but
// the parser reads with `j.value(key, default)`, so a rename on either side turns a configured
// knob into its default with no error, and LlamaTrainerIntegrationTest is gated on a system
// property no CI job sets. This is the only runnable check that the two agree.
TEST(JavaTrainingFieldContract, JavaAndEngineAgreeOnEveryKey) {
    ASSERT_GT(JLLAMA_JAVA_TRAINING_COUNT, 10)
        << "the generated trainer-key list is empty or truncated -- the extractor is broken";

    std::set<std::string> java_keys;
    for (int i = 0; i < JLLAMA_JAVA_TRAINING_COUNT; ++i) {
        java_keys.insert(JLLAMA_JAVA_TRAINING_NAMES[i]);
    }

    const std::vector<std::string> engine_list = jllama_train::config_keys();
    const std::set<std::string> engine_keys(engine_list.begin(), engine_list.end());
    ASSERT_EQ(engine_keys.size(), engine_list.size()) << "jllama_train::config_keys() lists a key twice";

    std::string only_java;
    for (const auto &key : java_keys) {
        if (engine_keys.count(key) == 0) {
            only_java += "\n  " + key;
        }
    }
    std::string only_engine;
    for (const auto &key : engine_keys) {
        if (java_keys.count(key) == 0) {
            only_engine += "\n  " + key;
        }
    }

    EXPECT_TRUE(only_java.empty()) << "TrainingParameters writes key(s) train_engine.cpp never "
                                      "reads; they fall back to the engine default silently:"
                                   << only_java;
    EXPECT_TRUE(only_engine.empty()) << "train_engine.cpp reads key(s) TrainingParameters never "
                                        "writes; they always take the engine default:"
                                     << only_engine;
}
