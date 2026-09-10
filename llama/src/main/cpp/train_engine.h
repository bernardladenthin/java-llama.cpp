// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT
//
// Native fine-tuning engine (proof-of-concept): a self-contained wrapper over llama.cpp's
// ggml-opt training path (llama_opt_init / llama_opt_epoch), mirroring upstream
// examples/training/finetune.cpp. Loads its own model + context (independent of the inference
// server_context in jllama.cpp), fine-tunes on a text corpus, and writes a new GGUF via
// llama_model_save_to_file. Kept out of jllama.cpp so the JNI layer stays thin.

#ifndef JLLAMA_TRAIN_ENGINE_H
#define JLLAMA_TRAIN_ENGINE_H

#include <string>
#include <vector>

namespace jllama_train {

// The keys the configuration parser reads, in one place so the Java registry can be checked
// against them.
//
// This pairing needs a guard more than it looks like it does. The parser reads every key with
// `j.value(key, default)`, which falls back to the default when the key is absent -- so a rename
// on either side does not fail, it silently reverts one knob to its default. Java's
// `parameters.TrainingField` declares the same names and `test_wire_contracts.cpp` asserts the two
// sets are equal; without that, nothing runnable covers this boundary at all, because
// `LlamaTrainerIntegrationTest` is gated on a system property no CI job sets.
namespace keys {
inline constexpr const char *MODEL_PATH    = "model_path";
inline constexpr const char *TRAINING_TEXT = "training_text";
inline constexpr const char *TRAINING_FILE = "training_file";
inline constexpr const char *OUTPUT_PATH   = "output_path";
inline constexpr const char *EPOCHS        = "epochs";
inline constexpr const char *LEARNING_RATE = "learning_rate";
inline constexpr const char *LR_MIN        = "lr_min";
inline constexpr const char *DECAY_EPOCHS  = "decay_epochs";
inline constexpr const char *WEIGHT_DECAY  = "weight_decay";
inline constexpr const char *OPTIMIZER     = "optimizer";
inline constexpr const char *N_CTX         = "n_ctx";
inline constexpr const char *N_GPU_LAYERS  = "n_gpu_layers";
inline constexpr const char *VAL_SPLIT     = "val_split";
inline constexpr const char *N_BATCH       = "n_batch";
inline constexpr const char *N_UBATCH      = "n_ubatch";
} // namespace keys

// Every key `config_keys()` lists is one the parser reads, and vice versa: both are written
// against the `keys` constants above, so a changed spelling moves together.
inline std::vector<std::string> config_keys() {
    return {keys::MODEL_PATH,   keys::TRAINING_TEXT, keys::TRAINING_FILE, keys::OUTPUT_PATH,
            keys::EPOCHS,       keys::LEARNING_RATE, keys::LR_MIN,        keys::DECAY_EPOCHS,
            keys::WEIGHT_DECAY, keys::OPTIMIZER,     keys::N_CTX,         keys::N_GPU_LAYERS,
            keys::VAL_SPLIT,    keys::N_BATCH,       keys::N_UBATCH};
}

// One fine-tuning run's inputs.
struct finetune_config {
    std::string model_path;    // base GGUF to fine-tune
    std::string training_text; // corpus supplied inline (used when training_file is empty)
    std::string training_file; // corpus read from this path instead of training_text
    std::string output_path;   // where the fine-tuned GGUF is written
    int         epochs;        // number of passes over the corpus (>= 1)
    float       learning_rate; // lr at the first epoch
    float       lr_min;        // minimum lr for decay; < 0 = no decay
    float       decay_epochs;  // decay lr0 -> lr_min over this many epochs; <= 0 = disabled
    float       weight_decay;  // weight decay; 0 = disabled
    int         optimizer;     // ggml_opt_optimizer_type: 0 = AdamW, 1 = SGD
    int         n_ctx;         // context size; 0 = the model's trained context
    int         n_gpu_layers;  // layers offloaded to the GPU; -1 = auto
    float       val_split;     // fraction of the corpus held out for validation
    int         n_batch;       // logical batch size; 0 = native default
    int         n_ubatch;      // physical (micro) batch size; 0 = native default
};

// Run one fine-tuning job end to end. Returns true on success; on failure returns false and sets
// `err`. Not re-entrant; intended to be called off the JVM's critical threads (it blocks for the
// full training run).
bool finetune(const finetune_config &cfg, std::string &err);

} // namespace jllama_train

#endif // JLLAMA_TRAIN_ENGINE_H
