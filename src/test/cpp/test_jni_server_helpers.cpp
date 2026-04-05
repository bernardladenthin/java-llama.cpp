// Tests for jni_server_helpers.hpp:
//   - build_completion_tasks_impl
//   - collect_task_results_impl
//   - recv_slot_task_result_impl  (added by Finding 5)
//
// build_completion_tasks_impl needs:
//   - JNIEnv  — used only for ThrowNew on the error path
//   - server_context*  — NOT accessed when "prompt" is absent (exception thrown
//     first), so nullptr is safe for error-path tests.
//   - json data — provided inline in the test
//
// collect_task_results_impl and recv_slot_task_result_impl need:
//   - server_response  (from server.hpp) — provides recv() / remove_waiting_task_ids()
//   - JNIEnv           — used only for ThrowNew on the error path
//
// server_response is used directly: we pre-seed it with results via send()
// before calling collect_task_results_impl().  Because recv() checks the
// queue under a mutex+condvar, pre-seeding lets us call recv() from the same
// thread without blocking.
//
// JNIEnv is mocked with the same stub technique used in test_jni_helpers.cpp:
// a zero-filled JNINativeInterface_ table with only GetLongField (unused here)
// and ThrowNew patched so we can observe whether an exception was raised.
//
// Covered scenarios:
//   - single success result → out filled, no throw, returns true
//   - single error result   → out empty, ThrowNew called with correct message, returns false
//   - multiple success results → all collected in order, returns true
//   - first result ok, second is error → cleanup, ThrowNew, returns false
//   - waiting ids are removed on success (remove_waiting_task_ids called)
//   - waiting ids are removed on error  (remove_waiting_task_ids called)

#include <gtest/gtest.h>

#include <cstring>
#include <memory>
#include <thread>
#include <unordered_set>

// server.hpp must come before jni_server_helpers.hpp (no include guard in server.hpp).
#include "server.hpp"
#include "jni_server_helpers.hpp"

// ============================================================
// Minimal concrete server_task_result subtypes for testing
// ============================================================

namespace {

// A success result whose to_json() returns {"content": "<msg>"}.
struct fake_ok_result : server_task_result {
    std::string msg;
    explicit fake_ok_result(int id_, std::string m) : msg(std::move(m)) { id = id_; }
    json to_json() override { return {{"content", msg}}; }
};

// An error result — reuses the real server_task_result_error so that
// to_json() → format_error_response() → {"message": err_msg, ...} matches
// the exact JSON key that collect_task_results_impl reads.
static server_task_result_ptr make_error(int id_, const std::string &msg) {
    auto r       = std::make_unique<server_task_result_error>();
    r->id        = id_;
    r->err_msg   = msg;
    r->err_type  = ERROR_TYPE_SERVER;
    return r;
}

static server_task_result_ptr make_ok(int id_, const std::string &msg = "ok") {
    return std::make_unique<fake_ok_result>(id_, msg);
}

// ============================================================
// Mock JNI environment (same pattern as test_jni_helpers.cpp)
// ============================================================

static bool        g_throw_called  = false;
static std::string g_throw_message;

// NewStringUTF stub: stores the string so tests can inspect it, returns a
// non-null sentinel so callers can distinguish success from nullptr (error).
static std::string g_new_string_utf_value;
static jstring     g_new_string_utf_sentinel = reinterpret_cast<jstring>(0xBEEF);

static jint JNICALL stub_ThrowNew(JNIEnv *, jclass, const char *msg) {
    g_throw_called  = true;
    g_throw_message = msg ? msg : "";
    return 0;
}

static jlong JNICALL stub_GetLongField(JNIEnv *, jobject, jfieldID) { return 0; }

static jstring JNICALL stub_NewStringUTF(JNIEnv *, const char *utf) {
    g_new_string_utf_value = utf ? utf : "";
    return g_new_string_utf_sentinel;
}

JNIEnv *make_mock_env(JNINativeInterface_ &table, JNIEnv_ &env_obj) {
    std::memset(&table, 0, sizeof(table));
    table.ThrowNew     = stub_ThrowNew;
    table.GetLongField = stub_GetLongField; // unused but avoids a null slot crash
    table.NewStringUTF = stub_NewStringUTF;
    env_obj.functions  = &table;
    return &env_obj;
}

// Test fixture: fresh mock env + fresh server_response per test.
struct CollectResultsFixture : ::testing::Test {
    JNINativeInterface_ table{};
    JNIEnv_             env_obj{};
    JNIEnv             *env        = nullptr;
    jclass              dummy_eclass = reinterpret_cast<jclass>(0x1);

    server_response queue;

    void SetUp() override {
        env            = make_mock_env(table, env_obj);
        g_throw_called = false;
        g_throw_message.clear();
        g_new_string_utf_value.clear();
    }
};

} // namespace

// ============================================================
// Single success result
// ============================================================

TEST_F(CollectResultsFixture, SingleOkResult_ReturnsTrueAndFillsOut) {
    queue.add_waiting_task_id(1);
    queue.send(make_ok(1, "hello"));

    std::unordered_set<int> ids = {1};
    std::vector<server_task_result_ptr> out;

    bool ok = collect_task_results_impl(env, queue, ids, out, dummy_eclass);

    EXPECT_TRUE(ok);
    EXPECT_EQ(out.size(), 1u);
    EXPECT_EQ(out[0]->to_json()["content"], "hello");
    EXPECT_FALSE(g_throw_called);
}

// ============================================================
// Single error result
// ============================================================

TEST_F(CollectResultsFixture, SingleErrorResult_ReturnsFalseAndThrows) {
    queue.add_waiting_task_id(2);
    queue.send(make_error(2, "something went wrong"));

    std::unordered_set<int> ids = {2};
    std::vector<server_task_result_ptr> out;

    bool ok = collect_task_results_impl(env, queue, ids, out, dummy_eclass);

    EXPECT_FALSE(ok);
    EXPECT_TRUE(out.empty()) << "out must not be populated on error";
    EXPECT_TRUE(g_throw_called);
    EXPECT_EQ(g_throw_message, "something went wrong");
}

// ============================================================
// Multiple success results
// ============================================================

TEST_F(CollectResultsFixture, MultipleOkResults_AllCollected) {
    for (int i = 10; i < 13; ++i) {
        queue.add_waiting_task_id(i);
        queue.send(make_ok(i, "msg" + std::to_string(i)));
    }

    std::unordered_set<int> ids = {10, 11, 12};
    std::vector<server_task_result_ptr> out;

    bool ok = collect_task_results_impl(env, queue, ids, out, dummy_eclass);

    EXPECT_TRUE(ok);
    EXPECT_EQ(out.size(), 3u);
    EXPECT_FALSE(g_throw_called);
}

// ============================================================
// First ok, second error — error path cleans up remaining ids
// ============================================================

TEST_F(CollectResultsFixture, SecondResultIsError_StopsAndThrows) {
    queue.add_waiting_task_id(20);
    queue.add_waiting_task_id(21);
    queue.send(make_ok(20));
    queue.send(make_error(21, "task 21 failed"));

    std::unordered_set<int> ids = {20, 21};
    std::vector<server_task_result_ptr> out;

    bool ok = collect_task_results_impl(env, queue, ids, out, dummy_eclass);

    EXPECT_FALSE(ok);
    EXPECT_TRUE(g_throw_called);
    EXPECT_EQ(g_throw_message, "task 21 failed");
}

// ============================================================
// Waiting ids are removed from the queue on the success path
// ============================================================

TEST_F(CollectResultsFixture, SuccessPath_WaitingIdsRemovedAfterCollect) {
    queue.add_waiting_task_id(30);
    queue.send(make_ok(30));

    std::unordered_set<int> ids = {30};
    std::vector<server_task_result_ptr> out;
    collect_task_results_impl(env, queue, ids, out, dummy_eclass);

    // After collect, the id must no longer be in the waiting set.
    // We verify indirectly: sending a second result for id=30 should
    // NOT be returned by a subsequent recv for a different id — the
    // simplest check is that waiting_task_ids no longer contains 30.
    EXPECT_FALSE(queue.waiting_task_ids.count(30))
        << "remove_waiting_task_ids must clear the id on success";
}

// ============================================================
// Waiting ids are removed from the queue on the error path
// ============================================================

TEST_F(CollectResultsFixture, ErrorPath_WaitingIdsRemovedAfterError) {
    queue.add_waiting_task_id(40);
    queue.send(make_error(40, "err"));

    std::unordered_set<int> ids = {40};
    std::vector<server_task_result_ptr> out;
    collect_task_results_impl(env, queue, ids, out, dummy_eclass);

    EXPECT_FALSE(queue.waiting_task_ids.count(40))
        << "remove_waiting_task_ids must clear the id on error";
}

// ============================================================
// Tests for build_completion_tasks_impl
//
// Only the error path is unit-testable here: the function reads
// data["prompt"] in its own statement BEFORE accessing ctx_server, so
// passing nullptr for ctx_server is safe when "prompt" is absent.
//
// The success path requires a live server_context (llama vocab + context
// pointers) and is covered by LlamaModelTest Java integration tests.
// ============================================================

TEST_F(CollectResultsFixture, BuildTasks_MissingPrompt_ReturnsFalseAndThrows) {
    json data = {{"n_predict", 1}}; // deliberately no "prompt" key
    std::string completion_id = "test-cmpl-id";
    std::vector<server_task> tasks;

    // ctx_server is nullptr — safe because data.at("prompt") throws before
    // any ctx_server member is accessed.
    bool ok = build_completion_tasks_impl(env,
                                          /*ctx_server=*/nullptr,
                                          data, completion_id,
                                          SERVER_TASK_TYPE_COMPLETION,
                                          OAICOMPAT_TYPE_NONE,
                                          tasks,
                                          dummy_eclass);

    EXPECT_FALSE(ok)           << "Missing 'prompt' must return false";
    EXPECT_TRUE(g_throw_called) << "ThrowNew must be called for missing 'prompt'";
    EXPECT_TRUE(tasks.empty()) << "tasks must remain empty on error";
}

TEST_F(CollectResultsFixture, BuildTasks_MissingPrompt_TaskTypeDoesNotAffectErrorBehaviour) {
    // The error path is identical regardless of task_type / oaicompat.
    // Verify INFILL behaves the same way.
    json data = {{"input_prefix", "def f():"}, {"input_suffix", "return 1"}};
    std::string completion_id = "infill-cmpl-id";
    std::vector<server_task> tasks;

    bool ok = build_completion_tasks_impl(env,
                                          /*ctx_server=*/nullptr,
                                          data, completion_id,
                                          SERVER_TASK_TYPE_INFILL,
                                          OAICOMPAT_TYPE_NONE,
                                          tasks,
                                          dummy_eclass);

    EXPECT_FALSE(ok);
    EXPECT_TRUE(g_throw_called);
    EXPECT_TRUE(tasks.empty());
}

// ============================================================
// Tests for recv_slot_task_result_impl
//
// Pre-seed the server_response queue (same technique as collect tests):
// calling send() before recv() satisfies the condvar predicate immediately,
// so recv() returns without blocking even from the same thread.
//
// NewStringUTF is stubbed to return a sentinel non-null jstring and capture
// the serialised JSON so we can verify the success path.
// ============================================================

TEST_F(CollectResultsFixture, RecvSlotResult_SuccessResult_ReturnsNonNullAndDoesNotThrow) {
    queue.add_waiting_task_id(50);
    queue.send(make_ok(50, "slot-ok"));

    jstring result = recv_slot_task_result_impl(env, queue, 50, dummy_eclass);

    EXPECT_NE(result, nullptr)       << "success result must return non-null jstring";
    EXPECT_FALSE(g_throw_called)     << "ThrowNew must not be called on success";
    // The stub captures the serialised JSON passed to NewStringUTF.
    EXPECT_FALSE(g_new_string_utf_value.empty())
        << "NewStringUTF must be called with the result JSON";
    EXPECT_NE(g_new_string_utf_value.find("slot-ok"), std::string::npos)
        << "result JSON must contain the content from the fake result";
}

TEST_F(CollectResultsFixture, RecvSlotResult_ErrorResult_ReturnsNullAndThrows) {
    queue.add_waiting_task_id(51);
    queue.send(make_error(51, "slot operation failed"));

    jstring result = recv_slot_task_result_impl(env, queue, 51, dummy_eclass);

    EXPECT_EQ(result, nullptr)      << "error result must return nullptr";
    EXPECT_TRUE(g_throw_called)     << "ThrowNew must be called on error";
    EXPECT_EQ(g_throw_message, "slot operation failed");
}

TEST_F(CollectResultsFixture, RecvSlotResult_WaitingIdRemovedAfterSuccess) {
    queue.add_waiting_task_id(52);
    queue.send(make_ok(52));

    recv_slot_task_result_impl(env, queue, 52, dummy_eclass);

    EXPECT_FALSE(queue.waiting_task_ids.count(52))
        << "remove_waiting_task_id must clear the id on success";
}

TEST_F(CollectResultsFixture, RecvSlotResult_WaitingIdRemovedAfterError) {
    queue.add_waiting_task_id(53);
    queue.send(make_error(53, "err"));

    recv_slot_task_result_impl(env, queue, 53, dummy_eclass);

    EXPECT_FALSE(queue.waiting_task_ids.count(53))
        << "remove_waiting_task_id must clear the id on error";
}
