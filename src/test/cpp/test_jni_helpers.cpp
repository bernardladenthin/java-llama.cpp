// Tests for get_server_context_impl() in jni_helpers.hpp.
//
// The function relies on JNIEnv, which is normally only available when a JVM
// is running.  To keep this test self-contained (no JVM), we exploit the fact
// that JNIEnv_ in C++ mode is a thin class whose every method dispatches
// through a JNINativeInterface_ function-pointer table.  We zero-initialise
// that table, then patch the two slots we actually call (GetLongField and
// ThrowNew) with small lambda-backed stubs.
//
// Covered scenarios:
//   - handle == 0  →  ThrowNew("Model is not loaded") called, nullptr returned
//   - handle != 0  →  no throw, correct server_context* returned
//   - ThrowNew is NOT called when the handle is valid

#include <gtest/gtest.h>

#include <cstring>

// jni_helpers.hpp is the unit under test; it includes jni.h which defines
// JNIEnv_ and JNINativeInterface_.
#include "jni_helpers.hpp"

// ============================================================
// Minimal mock JNI environment
// ============================================================

namespace {

// Mutable globals written by the stub functions, read by the tests.
static jlong        g_mock_handle       = 0;
static bool         g_throw_called      = false;
static std::string  g_throw_message;

// Stub that satisfies the JNINativeInterface_::GetLongField signature.
static jlong JNICALL stub_GetLongField(JNIEnv * /*env*/, jobject /*obj*/,
                                       jfieldID /*id*/) {
    return g_mock_handle;
}

// Stub that satisfies the JNINativeInterface_::ThrowNew signature.
static jint JNICALL stub_ThrowNew(JNIEnv * /*env*/, jclass /*clazz*/,
                                  const char *msg) {
    g_throw_called  = true;
    g_throw_message = msg ? msg : "";
    return 0;
}

// Build a JNIEnv that routes GetLongField and ThrowNew through our stubs.
// All other slots remain null; any unexpected call will crash, acting as an
// assertion that we only touch the two operations we intend to.
JNIEnv *make_mock_env(JNINativeInterface_ &table, JNIEnv_ &env_obj) {
    std::memset(&table, 0, sizeof(table));
    table.GetLongField = stub_GetLongField;
    table.ThrowNew     = stub_ThrowNew;
    env_obj.functions  = &table;
    return &env_obj;
}

// Convenience: reset all mock state before each test.
struct MockJniFixture : ::testing::Test {
    JNINativeInterface_ table{};
    JNIEnv_             env_obj{};
    JNIEnv             *env = nullptr;

    // Dummy field/class handles — their values are never dereferenced by the
    // stubs, so any non-null sentinel is fine.
    jfieldID dummy_field = reinterpret_cast<jfieldID>(0x1);
    jclass   dummy_class = reinterpret_cast<jclass>(0x2);

    void SetUp() override {
        env             = make_mock_env(table, env_obj);
        g_mock_handle   = 0;
        g_throw_called  = false;
        g_throw_message.clear();
    }
};

} // namespace

// ============================================================
// Test: null handle → ThrowNew + nullptr
// ============================================================

TEST_F(MockJniFixture, NullHandle_ThrowsAndReturnsNullptr) {
    g_mock_handle = 0; // model not loaded

    server_context *result =
        get_server_context_impl(env, /*obj=*/nullptr, dummy_field, dummy_class);

    EXPECT_EQ(result, nullptr)
        << "Expected nullptr when the model handle is 0";
    EXPECT_TRUE(g_throw_called)
        << "Expected ThrowNew to be called for a null handle";
    EXPECT_EQ(g_throw_message, "Model is not loaded");
}

// ============================================================
// Test: valid handle → correct server_context* returned, no throw
// ============================================================

TEST_F(MockJniFixture, ValidHandle_ReturnsServerContextAndDoesNotThrow) {
    // Build a minimal jllama_context on the stack.  We only need server to be
    // set; worker and worker_ready are never touched by the helper.
    // server_context is forward-declared in jni_helpers.hpp, so we can legally
    // hold a pointer to it without a full definition.
    server_context *sentinel = reinterpret_cast<server_context *>(0xDEADBEEF);
    jllama_context  fake_ctx;
    fake_ctx.server = sentinel;

    g_mock_handle = reinterpret_cast<jlong>(&fake_ctx);

    server_context *result =
        get_server_context_impl(env, /*obj=*/nullptr, dummy_field, dummy_class);

    EXPECT_EQ(result, sentinel)
        << "Expected the server pointer embedded in jllama_context";
    EXPECT_FALSE(g_throw_called)
        << "ThrowNew must not be called for a valid handle";
}

// ============================================================
// Test: ThrowNew message is exactly "Model is not loaded"
//       (guards against future typo regressions)
// ============================================================

TEST_F(MockJniFixture, NullHandle_ErrorMessageIsExact) {
    g_mock_handle = 0;

    get_server_context_impl(env, nullptr, dummy_field, dummy_class);

    ASSERT_TRUE(g_throw_called);
    EXPECT_EQ(g_throw_message, "Model is not loaded");
}

// ============================================================
// Test: ThrowNew is NOT called when handle is valid
// ============================================================

TEST_F(MockJniFixture, ValidHandle_NeverCallsThrowNew) {
    server_context *sentinel = reinterpret_cast<server_context *>(0xCAFEBABE);
    jllama_context  fake_ctx;
    fake_ctx.server = sentinel;
    g_mock_handle = reinterpret_cast<jlong>(&fake_ctx);

    get_server_context_impl(env, nullptr, dummy_field, dummy_class);

    EXPECT_FALSE(g_throw_called);
}
