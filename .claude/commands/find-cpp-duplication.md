---
name: find-cpp-duplication
description: Scan C++ source files in src/main/cpp/ for all categories of duplication and boilerplate — short repeated expressions, copy-pasted blocks, pipeline compositions, near-identical switch arms, mixed JNI/logic concerns, inconsistent helper usage, local cleanup sequences, and dead code. Reports findings only, no modifications.
---

Review the C++ source files in src/main/cpp/ (jllama.cpp, jni_helpers.hpp,
jni_server_helpers.hpp, server.hpp, utils.hpp) and identify all duplication
opportunities. Cast a wider net than start/end boilerplate — include patterns
anywhere inside a function body.

Look for ALL of the following categories:

1. SHORT REPEATED EXPRESSIONS (1–2 lines, 3+ sites)
   Any single expression or two-line sequence that appears verbatim in three or
   more places, even if surrounded by different context.
   Example: result->to_json()["message"].get<std::string>()

2. REPEATED MULTI-LINE BLOCKS (3+ lines, 2+ sites)
   Any block of 3 or more consecutive lines that is copy-pasted with at most
   minor variation (different variable names or one differing string literal).
   Example: four-line jintArray → vector<llama_token> read pattern.

3. PIPELINE COMPOSITIONS
   Any sequence of 2+ function calls that always appear chained together in the
   same order at every call site. The chain itself is a candidate for wrapping.
   Example: build_completion_tasks → dispatch_tasks → collect_and_serialize

4. NEAR-IDENTICAL SWITCH CASES OR IF-ELSE ARMS
   Two or more switch cases / if-else branches whose bodies differ only by one
   variable, constant, or string literal.

5. MIXED CONCERNS (JNI + LOGIC)
   Functions where pure computation (no JNI calls) is interleaved with JNI
   serialisation. The pure part is a candidate for extraction to a separately
   testable _impl function.
   Example: single-vs-array JSON construction inside results_to_jstring.

6. INCONSISTENT HELPER USAGE
   Places where an already-extracted helper exists but is not used — either
   because the helper was added after the call site was written, or the call
   site is inside a header that the helper lives in.
   Example: a header function still using dump()+NewStringUTF after
   json_to_jstring_impl was extracted.

7. LOCAL CLEANUP SEQUENCES
   Repeated tear-down sequences inside a single function (delete X; delete Y;
   free(); ThrowNew()) that differ only in the error message — candidate for a
   local lambda.

8. DEAD CODE
   Commented-out blocks that duplicate active code immediately above or below.

For each finding report:
  - Category (from the list above)
  - Exact file names and line numbers of every occurrence
  - The minimal signature of the helper that would eliminate the duplication
  - Whether the extraction is unit-testable without a real JVM or llama model
    (i.e., can all llama.h / server.hpp dependencies be passed as parameters)
  - Estimated line savings across all call sites

Do not modify any files. Report only.
