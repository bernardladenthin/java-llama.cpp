// Tests for jni_helpers.hpp.
//
// This file covers all functions in jni_helpers.hpp — both Layer A (JNI handle
// management) and Layer B (JNI + server orchestration).
//
// Pure JSON transform tests live in test_json_helpers.cpp.
//
// Layer A tests (no server.hpp needed for the functions under test, but
// server.hpp is included here for Layer B and to satisfy the TU convention):
//   get_server_context_impl, get_jllama_context_impl,
//   require_single_task_id_impl, require_json_field_impl,
//   jint_array_to_tokens_impl
//
// Layer B tests (need server.hpp + mock JNIEnv + pre-seeded server_response):
//   json_to_jstring_impl, results_to_jstring_impl,
//   build_completion_tasks_impl, recv_slot_task_result_impl,
//   collect_task_results_impl, embedding_to_jfloat_array_impl
//
// JNIEnv is mocked via a zero-filled JNINativeInterface_ table with only the
// slots exercised by each test patched.  server_response is used directly:
// results are pre-seeded via send() before recv() is called, so the condvar
// is satisfied immediately without blocking.

#include <gtest/gtest.h>

#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <unordered_set>

// server.hpp must precede jni_helpers.hpp (no include guard in server.hpp).
#include "server.hpp"
#include "jni_helpers.hpp"

// embedding_to_jfloat_array_impl is also tested in this file (see bottom).

// ============================================================
// Shared fake result types
// ============================================================

namespace {

struct fake_ok_result : server_task_result {
    std::string msg;
    explicit fake_ok_result(int id_, std::string m) : msg(std::move(m)) { id = id_; }
    json to_json() override { return {{"content", msg}}; }
};

static server_task_result_ptr make_error(int id_, const std::string &msg) {
    auto r      = std::make_unique<server_task_result_error>();
    r->id       = id_;
    r->err_msg  = msg;
    r->err_type = ERROR_TYPE_SERVER;
    return r;
}

static server_task_result_ptr make_ok(int id_, const std::string &msg = "ok") {
    return std::make_unique<fake_ok_result>(id_, msg);
}

// ============================================================
// Mock JNI environment helpers
// ============================================================

// State captured by stubs — reset in each fixture's SetUp().
static bool        g_throw_called        = false;
static std::string g_throw_message;
static std::string g_new_string_utf_value;
static jlong       g_mock_handle         = 0;

static jstring g_new_string_utf_sentinel = reinterpret_cast<jstring>(0xBEEF);

static jint    JNICALL stub_ThrowNew(JNIEnv *, jclass, const char *msg) {
    g_throw_called  = true;
    g_throw_message = msg ? msg : "";
    return 0;
}
static jlong   JNICALL stub_GetLongField(JNIEnv *, jobject, jfieldID) {
    return g_mock_handle;
}
static jstring JNICALL stub_NewStringUTF(JNIEnv *, const char *utf) {
    g_new_string_utf_value = utf ? utf : "";
    return g_new_string_utf_sentinel;
}

// Minimal env: ThrowNew + GetLongField + NewStringUTF.
JNIEnv *make_mock_env(JNINativeInterface_ &table, JNIEnv_ &env_obj) {
    std::memset(&table, 0, sizeof(table));
    table.ThrowNew     = stub_ThrowNew;
    table.GetLongField = stub_GetLongField;
    table.NewStringUTF = stub_NewStringUTF;
    env_obj.functions  = &table;
    return &env_obj;
}

// Base fixture: resets all mock state.
struct MockJniFixture : ::testing::Test {
    JNINativeInterface_ table{};
    JNIEnv_             env_obj{};
    JNIEnv             *env          = nullptr;
    jfieldID            dummy_field  = reinterpret_cast<jfieldID>(0x1);
    jclass              dummy_class  = reinterpret_cast<jclass>(0x2);

    void SetUp() override {
        env                    = make_mock_env(table, env_obj);
        g_mock_handle          = 0;
        g_throw_called         = false;
        g_throw_message.clear();
        g_new_string_utf_value.clear();
    }
};

// Extends MockJniFixture with a fresh server_response queue.
struct ServerFixture : MockJniFixture {
    server_response queue;
};

} // namespace

// ============================================================
// get_server_context_impl
// ============================================================

TEST_F(MockJniFixture, GetServerContext_NullHandle_ThrowsAndReturnsNull) {
    g_mock_handle = 0;

    server_context *result =
        get_server_context_impl(env, nullptr, dummy_field, dummy_class);

    EXPECT_EQ(result, nullptr);
    EXPECT_TRUE(g_throw_called);
    EXPECT_EQ(g_throw_message, "Model is not loaded");
}

TEST_F(MockJniFixture, GetServerContext_ValidHandle_ReturnsServerContextNoThrow) {
    server_context *sentinel = reinterpret_cast<server_context *>(0xDEADBEEF);
    jllama_context  fake_ctx;
    fake_ctx.server = sentinel;
    g_mock_handle   = reinterpret_cast<jlong>(&fake_ctx);

    server_context *result =
        get_server_context_impl(env, nullptr, dummy_field, dummy_class);

    EXPECT_EQ(result, sentinel);
    EXPECT_FALSE(g_throw_called);
}

TEST_F(MockJniFixture, GetServerContext_ErrorMessageIsExact) {
    g_mock_handle = 0;
    (void)get_server_context_impl(env, nullptr, dummy_field, dummy_class);
    ASSERT_TRUE(g_throw_called);
    EXPECT_EQ(g_throw_message, "Model is not loaded");
}

TEST_F(MockJniFixture, GetServerContext_ValidHandle_NeverCallsThrowNew) {
    server_context *sentinel = reinterpret_cast<server_context *>(0xCAFEBABE);
    jllama_context  fake_ctx;
    fake_ctx.server = sentinel;
    g_mock_handle   = reinterpret_cast<jlong>(&fake_ctx);
    (void)get_server_context_impl(env, nullptr, dummy_field, dummy_class);
    EXPECT_FALSE(g_throw_called);
}

// ============================================================
// get_jllama_context_impl
// ============================================================

TEST_F(MockJniFixture, GetJllamaContext_NullHandle_ReturnsNullWithoutThrow) {
    g_mock_handle = 0;

    jllama_context *result = get_jllama_context_impl(env, nullptr, dummy_field);

    EXPECT_EQ(result, nullptr);
    EXPECT_FALSE(g_throw_called);
}

TEST_F(MockJniFixture, GetJllamaContext_ValidHandle_ReturnsWrapper) {
    jllama_context fake_ctx;
    fake_ctx.server = nullptr;
    g_mock_handle   = reinterpret_cast<jlong>(&fake_ctx);

    jllama_context *result = get_jllama_context_impl(env, nullptr, dummy_field);

    EXPECT_EQ(result, &fake_ctx);
    EXPECT_FALSE(g_throw_called);
}

TEST_F(MockJniFixture, GetJllamaContext_ReturnsWrapperNotInnerServer) {
    server_context *sentinel = reinterpret_cast<server_context *>(0xDEADBEEF);
    jllama_context  fake_ctx;
    fake_ctx.server = sentinel;
    g_mock_handle   = reinterpret_cast<jlong>(&fake_ctx);

    jllama_context *result = get_jllama_context_impl(env, nullptr, dummy_field);

    EXPECT_EQ(result, &fake_ctx);
    EXPECT_NE(static_cast<void *>(result), static_cast<void *>(sentinel));
}

TEST_F(MockJniFixture, GetJllamaContext_NullHandle_WhileGetServerContextThrows) {
    g_mock_handle = 0;

    (void)get_server_context_impl(env, nullptr, dummy_field, dummy_class);
    EXPECT_TRUE(g_throw_called);

    g_throw_called = false;
    (void)get_jllama_context_impl(env, nullptr, dummy_field);
    EXPECT_FALSE(g_throw_called);
}

// ============================================================
// require_single_task_id_impl
// ============================================================

TEST_F(MockJniFixture, RequireSingleTaskId_ExactlyOne_ReturnsIdNoThrow) {
    std::unordered_set<int> ids = {42};
    EXPECT_EQ(require_single_task_id_impl(env, ids, dummy_class), 42);
    EXPECT_FALSE(g_throw_called);
}

TEST_F(MockJniFixture, RequireSingleTaskId_Empty_ReturnsZeroAndThrows) {
    std::unordered_set<int> ids;
    EXPECT_EQ(require_single_task_id_impl(env, ids, dummy_class), 0);
    EXPECT_TRUE(g_throw_called);
    EXPECT_EQ(g_throw_message, "multitasking currently not supported");
}

TEST_F(MockJniFixture, RequireSingleTaskId_Multiple_ReturnsZeroAndThrows) {
    std::unordered_set<int> ids = {1, 2, 3};
    EXPECT_EQ(require_single_task_id_impl(env, ids, dummy_class), 0);
    EXPECT_TRUE(g_throw_called);
}

// ============================================================
// require_json_field_impl
// ============================================================

TEST_F(MockJniFixture, RequireJsonField_PresentField_ReturnsTrueNoThrow) {
    nlohmann::json data = {{"input_prefix", "hello"}};
    EXPECT_TRUE(require_json_field_impl(env, data, "input_prefix", dummy_class));
    EXPECT_FALSE(g_throw_called);
}

TEST_F(MockJniFixture, RequireJsonField_MissingField_ReturnsFalseAndThrows) {
    nlohmann::json data = {{"other", 1}};
    EXPECT_FALSE(require_json_field_impl(env, data, "input_prefix", dummy_class));
    EXPECT_TRUE(g_throw_called);
    EXPECT_EQ(g_throw_message, "\"input_prefix\" is required");
}

TEST_F(MockJniFixture, RequireJsonField_EmptyJson_ReturnsFalseAndThrows) {
    EXPECT_FALSE(require_json_field_impl(
        env, nlohmann::json::object(), "input_suffix", dummy_class));
    EXPECT_TRUE(g_throw_called);
    EXPECT_EQ(g_throw_message, "\"input_suffix\" is required");
}

// ============================================================
// jint_array_to_tokens_impl
// ============================================================

namespace {

static jint  g_array_data[8]  = {};
static jsize g_array_length   = 0;
static bool  g_release_called = false;
static jint  g_release_mode   = -1;

static jsize JNICALL stub_GetArrayLength(JNIEnv *, jarray) { return g_array_length; }
static jint *JNICALL stub_GetIntArrayElements(JNIEnv *, jintArray, jboolean *) {
    return g_array_data;
}
static void  JNICALL stub_ReleaseIntArrayElements(JNIEnv *, jintArray, jint *, jint mode) {
    g_release_called = true;
    g_release_mode   = mode;
}

JNIEnv *make_array_env(JNINativeInterface_ &table, JNIEnv_ &env_obj) {
    std::memset(&table, 0, sizeof(table));
    table.GetArrayLength          = stub_GetArrayLength;
    table.GetIntArrayElements     = stub_GetIntArrayElements;
    table.ReleaseIntArrayElements = stub_ReleaseIntArrayElements;
    env_obj.functions             = &table;
    return &env_obj;
}

struct ArrayFixture : ::testing::Test {
    JNINativeInterface_ table{};
    JNIEnv_             env_obj{};
    JNIEnv             *env = nullptr;

    void SetUp() override {
        env              = make_array_env(table, env_obj);
        g_release_called = false;
        g_release_mode   = -1;
        std::memset(g_array_data, 0, sizeof(g_array_data));
        g_array_length   = 0;
    }
};

} // namespace

TEST_F(ArrayFixture, JintArrayToTokens_EmptyArray_ReturnsEmptyVector) {
    g_array_length = 0;
    auto tokens = jint_array_to_tokens_impl(env, nullptr);
    EXPECT_TRUE(tokens.empty());
    EXPECT_TRUE(g_release_called);
    EXPECT_EQ(g_release_mode, JNI_ABORT);
}

TEST_F(ArrayFixture, JintArrayToTokens_ThreeElements_CopiedCorrectly) {
    g_array_data[0] = 10; g_array_data[1] = 20; g_array_data[2] = 30;
    g_array_length  = 3;
    auto tokens = jint_array_to_tokens_impl(env, nullptr);
    ASSERT_EQ(tokens.size(), 3u);
    EXPECT_EQ(tokens[0], 10);
    EXPECT_EQ(tokens[1], 20);
    EXPECT_EQ(tokens[2], 30);
}

TEST_F(ArrayFixture, JintArrayToTokens_ReleasesWithAbortFlag) {
    g_array_length = 1; g_array_data[0] = 42;
    (void)jint_array_to_tokens_impl(env, nullptr);
    EXPECT_TRUE(g_release_called);
    EXPECT_EQ(g_release_mode, JNI_ABORT);
}

// ============================================================
// json_to_jstring_impl
// ============================================================

TEST_F(MockJniFixture, JsonToJstring_Object_RoundTrips) {
    json j = {{"key", "value"}, {"n", 42}};
    jstring js = json_to_jstring_impl(env, j);
    EXPECT_NE(js, nullptr);
    json parsed = json::parse(g_new_string_utf_value);
    EXPECT_TRUE(parsed.is_object());
    EXPECT_EQ(parsed.value("key", ""), "value");
    EXPECT_EQ(parsed.value("n", 0), 42);
}

TEST_F(MockJniFixture, JsonToJstring_Array_RoundTrips) {
    json j = json::array({1, 2, 3});
    jstring js = json_to_jstring_impl(env, j);
    EXPECT_NE(js, nullptr);
    json parsed = json::parse(g_new_string_utf_value);
    EXPECT_TRUE(parsed.is_array());
    ASSERT_EQ(parsed.size(), 3u);
}

TEST_F(MockJniFixture, JsonToJstring_ReturnsSentinel) {
    jstring js = json_to_jstring_impl(env, {{"ok", true}});
    EXPECT_EQ(js, reinterpret_cast<jstring>(0xBEEF));
}

// ============================================================
// results_to_jstring_impl
// ============================================================

TEST_F(MockJniFixture, ResultsToJstring_SingleResult_ReturnsBareObject) {
    std::vector<server_task_result_ptr> results;
    results.push_back(make_ok(1, "hello"));

    jstring js = results_to_jstring_impl(env, results);

    EXPECT_NE(js, nullptr);
    json parsed = json::parse(g_new_string_utf_value);
    EXPECT_TRUE(parsed.is_object());
    EXPECT_EQ(parsed.value("content", ""), "hello");
}

TEST_F(MockJniFixture, ResultsToJstring_MultipleResults_ReturnsArray) {
    std::vector<server_task_result_ptr> results;
    results.push_back(make_ok(2, "first"));
    results.push_back(make_ok(3, "second"));

    jstring js = results_to_jstring_impl(env, results);

    EXPECT_NE(js, nullptr);
    json parsed = json::parse(g_new_string_utf_value);
    EXPECT_TRUE(parsed.is_array());
    ASSERT_EQ(parsed.size(), 2u);
    EXPECT_EQ(parsed[0].value("content", ""), "first");
    EXPECT_EQ(parsed[1].value("content", ""), "second");
}

TEST_F(MockJniFixture, ResultsToJstring_EmptyVector_ReturnsEmptyArray) {
    std::vector<server_task_result_ptr> results;
    jstring js = results_to_jstring_impl(env, results);
    EXPECT_NE(js, nullptr);
    json parsed = json::parse(g_new_string_utf_value);
    EXPECT_TRUE(parsed.is_array());
    EXPECT_TRUE(parsed.empty());
}

// ============================================================
// collect_task_results_impl
// ============================================================

TEST_F(ServerFixture, CollectResults_SingleOk_ReturnsTrueAndFillsOut) {
    queue.add_waiting_task_id(1);
    queue.send(make_ok(1, "hello"));

    std::unordered_set<int> ids = {1};
    std::vector<server_task_result_ptr> out;

    EXPECT_TRUE(collect_task_results_impl(env, queue, ids, out, dummy_class));
    ASSERT_EQ(out.size(), 1u);
    EXPECT_EQ(out[0]->to_json()["content"], "hello");
    EXPECT_FALSE(g_throw_called);
}

TEST_F(ServerFixture, CollectResults_SingleError_ReturnsFalseAndThrows) {
    queue.add_waiting_task_id(2);
    queue.send(make_error(2, "something went wrong"));

    std::unordered_set<int> ids = {2};
    std::vector<server_task_result_ptr> out;

    EXPECT_FALSE(collect_task_results_impl(env, queue, ids, out, dummy_class));
    EXPECT_TRUE(out.empty());
    EXPECT_TRUE(g_throw_called);
    EXPECT_EQ(g_throw_message, "something went wrong");
}

TEST_F(ServerFixture, CollectResults_MultipleOk_AllCollected) {
    for (int i = 10; i < 13; ++i) { queue.add_waiting_task_id(i); queue.send(make_ok(i)); }

    std::unordered_set<int> ids = {10, 11, 12};
    std::vector<server_task_result_ptr> out;

    EXPECT_TRUE(collect_task_results_impl(env, queue, ids, out, dummy_class));
    EXPECT_EQ(out.size(), 3u);
    EXPECT_FALSE(g_throw_called);
}

TEST_F(ServerFixture, CollectResults_SecondError_StopsAndThrows) {
    queue.add_waiting_task_id(20); queue.send(make_ok(20));
    queue.add_waiting_task_id(21); queue.send(make_error(21, "task 21 failed"));

    std::unordered_set<int> ids = {20, 21};
    std::vector<server_task_result_ptr> out;

    EXPECT_FALSE(collect_task_results_impl(env, queue, ids, out, dummy_class));
    EXPECT_TRUE(g_throw_called);
    EXPECT_EQ(g_throw_message, "task 21 failed");
}

TEST_F(ServerFixture, CollectResults_SuccessPath_WaitingIdsRemoved) {
    queue.add_waiting_task_id(30); queue.send(make_ok(30));
    std::unordered_set<int> ids = {30};
    std::vector<server_task_result_ptr> out;
    (void)collect_task_results_impl(env, queue, ids, out, dummy_class);
    EXPECT_FALSE(queue.waiting_task_ids.count(30));
}

TEST_F(ServerFixture, CollectResults_ErrorPath_WaitingIdsRemoved) {
    queue.add_waiting_task_id(40); queue.send(make_error(40, "err"));
    std::unordered_set<int> ids = {40};
    std::vector<server_task_result_ptr> out;
    (void)collect_task_results_impl(env, queue, ids, out, dummy_class);
    EXPECT_FALSE(queue.waiting_task_ids.count(40));
}

// ============================================================
// recv_slot_task_result_impl
// ============================================================

TEST_F(ServerFixture, RecvSlotResult_Success_ReturnsNonNullNoThrow) {
    queue.add_waiting_task_id(50); queue.send(make_ok(50, "slot-ok"));

    jstring result = recv_slot_task_result_impl(env, queue, 50, dummy_class);

    EXPECT_NE(result, nullptr);
    EXPECT_FALSE(g_throw_called);
    EXPECT_NE(g_new_string_utf_value.find("slot-ok"), std::string::npos);
}

TEST_F(ServerFixture, RecvSlotResult_Error_ReturnsNullAndThrows) {
    queue.add_waiting_task_id(51); queue.send(make_error(51, "slot operation failed"));

    jstring result = recv_slot_task_result_impl(env, queue, 51, dummy_class);

    EXPECT_EQ(result, nullptr);
    EXPECT_TRUE(g_throw_called);
    EXPECT_EQ(g_throw_message, "slot operation failed");
}

TEST_F(ServerFixture, RecvSlotResult_Success_WaitingIdRemoved) {
    queue.add_waiting_task_id(52); queue.send(make_ok(52));
    (void)recv_slot_task_result_impl(env, queue, 52, dummy_class);
    EXPECT_FALSE(queue.waiting_task_ids.count(52));
}

TEST_F(ServerFixture, RecvSlotResult_Error_WaitingIdRemoved) {
    queue.add_waiting_task_id(53); queue.send(make_error(53, "err"));
    (void)recv_slot_task_result_impl(env, queue, 53, dummy_class);
    EXPECT_FALSE(queue.waiting_task_ids.count(53));
}

// ============================================================
// build_completion_tasks_impl — error path only
// (success path requires a live server_context with vocab/ctx)
// ============================================================

TEST_F(MockJniFixture, BuildTasks_MissingPrompt_ReturnsFalseAndThrows) {
    json data = {{"n_predict", 1}};
    std::vector<server_task> tasks;

    bool ok = build_completion_tasks_impl(env, /*ctx_server=*/nullptr, data,
                                          "test-cmpl-id",
                                          SERVER_TASK_TYPE_COMPLETION,
                                          OAICOMPAT_TYPE_NONE,
                                          tasks, dummy_class);

    EXPECT_FALSE(ok);
    EXPECT_TRUE(g_throw_called);
    EXPECT_TRUE(tasks.empty());
}

TEST_F(MockJniFixture, BuildTasks_MissingPrompt_InfillTypeHasSameBehaviour) {
    json data = {{"input_prefix", "def f():"}, {"input_suffix", "return 1"}};
    std::vector<server_task> tasks;

    bool ok = build_completion_tasks_impl(env, nullptr, data, "infill-id",
                                          SERVER_TASK_TYPE_INFILL,
                                          OAICOMPAT_TYPE_NONE,
                                          tasks, dummy_class);

    EXPECT_FALSE(ok);
    EXPECT_TRUE(g_throw_called);
    EXPECT_TRUE(tasks.empty());
}

// ============================================================
// embedding_to_jfloat_array_impl
// ============================================================

namespace {

static bool  g_float_new_called  = false;
static jsize g_float_alloc_size  = -1;
static jsize g_float_copied_size = -1;

static jfloatArray JNICALL stub_NewFloatArray(JNIEnv *, jsize n) {
    g_float_new_called = true;
    g_float_alloc_size = n;
    return reinterpret_cast<jfloatArray>(0xF1);
}
static void JNICALL stub_SetFloatArrayRegion(JNIEnv *, jfloatArray, jsize, jsize n, const jfloat *) {
    g_float_copied_size = n;
}

struct FloatArrayFixture : MockJniFixture {
    void SetUp() override {
        MockJniFixture::SetUp();
        g_float_new_called  = false;
        g_float_alloc_size  = -1;
        g_float_copied_size = -1;
        table.NewFloatArray       = stub_NewFloatArray;
        table.SetFloatArrayRegion = stub_SetFloatArrayRegion;
    }
};

} // namespace

TEST_F(FloatArrayFixture, EmbeddingToJfloatArray_ReturnsSentinel) {
    std::vector<float> v = {1.0f, 2.0f, 3.0f};
    auto *result = embedding_to_jfloat_array_impl(env, v, dummy_class);
    EXPECT_EQ(result, reinterpret_cast<jfloatArray>(0xF1));
}

TEST_F(FloatArrayFixture, EmbeddingToJfloatArray_AllocatesCorrectSize) {
    std::vector<float> v = {0.1f, 0.2f};
    embedding_to_jfloat_array_impl(env, v, dummy_class);
    EXPECT_EQ(g_float_alloc_size, 2);
}

TEST_F(FloatArrayFixture, EmbeddingToJfloatArray_CopiesAllElements) {
    std::vector<float> v(5, 0.5f);
    embedding_to_jfloat_array_impl(env, v, dummy_class);
    EXPECT_EQ(g_float_copied_size, 5);
}

TEST_F(FloatArrayFixture, EmbeddingToJfloatArray_EmptyVector_AllocatesZeroLen) {
    std::vector<float> v;
    embedding_to_jfloat_array_impl(env, v, dummy_class);
    EXPECT_EQ(g_float_alloc_size, 0);
    EXPECT_FALSE(g_throw_called);
}

TEST_F(FloatArrayFixture, EmbeddingToJfloatArray_AllocFails_ThrowsOomAndReturnsNull) {
    table.NewFloatArray = [](JNIEnv *, jsize) -> jfloatArray { return nullptr; };
    std::vector<float> v = {1.0f};
    auto *result = embedding_to_jfloat_array_impl(env, v, dummy_class);
    EXPECT_EQ(result, nullptr);
    EXPECT_TRUE(g_throw_called);
    EXPECT_EQ(g_throw_message, "could not allocate embedding");
}
