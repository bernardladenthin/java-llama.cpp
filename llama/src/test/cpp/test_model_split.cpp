// Runnable guard for patches/0012 (the layer-split fix in llama.cpp's src/llama-model.cpp).
//
// The patch also ships an upstream test (tests/test-model-split.cpp), but a FetchContent
// subproject builds with LLAMA_BUILD_TESTS=OFF, so that one is applied-but-never-compiled here.
// This file drives the same two functions from jllama_test, which every platform runs in CI --
// so a llama.cpp bump that drops the patch reds "C++ Tests" everywhere instead of surfacing as
// one red macOS Java job with the message "error loading model: vector".
//
// Note that the failure it guards against is NOT reproducible on this side: it needs a GPU
// backend that reports zero free memory, and without one `devices` is empty and the mapping is
// never reached. What is testable -- and is what actually broke -- is the arithmetic itself.

#include "llama-model.h"

#include <gtest/gtest.h>

#include <cmath>
#include <string>
#include <vector>

namespace {

// The lookup load_tensors() performs for every offloaded layer.
void expect_every_layer_maps_into_range(const std::vector<float> &splits, int n_layers) {
    for (int il = 0; il < n_layers; ++il) {
        int idx = -1;
        ASSERT_NO_THROW(idx = llama_model_splits_select_device(splits, il, n_layers)) << "layer " << il;
        EXPECT_GE(idx, 0) << "layer " << il;
        EXPECT_LT((size_t) idx, splits.size()) << "layer " << il;
    }
}

} // namespace

TEST(LlamaModelSplits, NormalizeIsProportionalToTheWeights) {
    std::vector<float> splits = {1.0f, 3.0f};
    llama_model_splits_normalize(splits);

    EXPECT_NEAR(splits[0], 0.25f, 1e-6f);
    EXPECT_NEAR(splits[1], 1.00f, 1e-6f);
}

TEST(LlamaModelSplits, NormalizeSingleDeviceTakesEverything) {
    std::vector<float> splits = {42.0f};
    llama_model_splits_normalize(splits);

    ASSERT_EQ(splits.size(), 1u);
    EXPECT_NEAR(splits[0], 1.0f, 1e-6f);
}

// The regression. A device reporting zero free memory -- e.g. a Metal device whose
// currentAllocatedSize has grown past its recommendedMaxWorkingSetSize -- makes the sum of the
// weights zero. Dividing by it put a NaN in every split point.
TEST(LlamaModelSplits, ZeroSumDoesNotProduceNaNSplitPoints) {
    for (size_t n_devices : {(size_t) 1, (size_t) 2, (size_t) 4}) {
        std::vector<float> splits(n_devices, 0.0f);
        llama_model_splits_normalize(splits);

        ASSERT_EQ(splits.size(), n_devices);
        for (size_t i = 0; i < splits.size(); ++i) {
            EXPECT_TRUE(std::isfinite(splits[i])) << "n_devices=" << n_devices << " i=" << i;
        }
        EXPECT_NEAR(splits.back(), 1.0f, 1e-6f) << "n_devices=" << n_devices;
    }
}

// NaN compares false against everything, so std::upper_bound returned the end iterator and
// load_tensors() indexed one past the last device -- an std::out_of_range whose libc++ what() is
// the bare string "vector". This is the assertion that would have failed before the fix.
TEST(LlamaModelSplits, ZeroSumStillMapsEveryLayerToARealDevice) {
    for (size_t n_devices : {(size_t) 1, (size_t) 2, (size_t) 4}) {
        std::vector<float> splits(n_devices, 0.0f);
        llama_model_splits_normalize(splits);
        expect_every_layer_maps_into_range(splits, 32);
    }
}

TEST(LlamaModelSplits, ProportionalSplitsMapEveryLayerToARealDevice) {
    std::vector<float> splits = {1.0f, 3.0f};
    llama_model_splits_normalize(splits);
    expect_every_layer_maps_into_range(splits, 32);
}

// The diagnostic half: malformed split points must name themselves rather than reaching the
// device list and coming back out as libc++'s content-free "vector".
TEST(LlamaModelSplits, SelectDeviceNamesMalformedSplitPoints) {
    const std::vector<float> nan_splits = {std::nanf(""), std::nanf("")};

    try {
        llama_model_splits_select_device(nan_splits, 0, 32);
        FAIL() << "expected malformed split points to throw";
    } catch (const std::exception &e) {
        const std::string what = e.what();
        EXPECT_NE(what, "vector");
        EXPECT_NE(what.find("llama_model_splits_select_device"), std::string::npos) << what;
        EXPECT_NE(what.find("layer 0 of 32"), std::string::npos) << what;
        EXPECT_NE(what.find("device index 2"), std::string::npos) << what;
        EXPECT_NE(what.find("nan"), std::string::npos) << what;
    }
}
