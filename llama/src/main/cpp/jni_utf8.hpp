// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

#pragma once

// jni_utf8.hpp — text from native code to Java without Modified UTF-8.
//
// JNI's NewStringUTF and ThrowNew take *Modified* UTF-8: a supplementary-plane character (every
// 4-byte emoji) is spelled as two 3-byte surrogates there, and a 4-byte standard UTF-8 sequence is
// invalid input -- HotSpot turns it into garbage, Android's CheckJNI aborts the process. The text
// these helpers carry is not ours: an error message from llama.cpp quotes a file path or a piece
// of the request, and both can hold any character. So a message with a byte above 0x7F crosses as
// bytes through String(byte[], "UTF-8"), where Java's decoder is the one that decides (and replaces
// an invalid sequence with U+FFFD instead of failing). Pure ASCII is identical in both encodings
// and keeps the one-call path.
//
// Self-contained on purpose (jni.h only, no cached ids): every JNI translation unit uses it,
// including train_engine.cpp, which deliberately does not include jni_helpers.hpp.

#include "jni.h"

#include <cstring>

[[nodiscard]] inline bool jni_utf8_is_ascii(const char *s) {
    for (const char *p = s; *p != '\0'; ++p) {
        if (static_cast<unsigned char>(*p) >= 0x80) {
            return false;
        }
    }
    return true;
}

// A java.lang.String from standard UTF-8 bytes; nullptr with a pending Java exception on failure.
// Called with an exception already pending, it builds nothing (the JNI spec forbids the calls) and
// leaves that exception in place.
[[nodiscard]] inline jstring new_string_utf8_impl(JNIEnv *env, const char *utf8) {
    if (env->ExceptionCheck()) {
        return nullptr;
    }
    if (utf8 == nullptr) {
        utf8 = "";
    }
    if (jni_utf8_is_ascii(utf8)) {
        return env->NewStringUTF(utf8);
    }
    jclass string_class = env->FindClass("java/lang/String");
    if (string_class == nullptr) {
        return nullptr;
    }
    jstring result = nullptr;
    jmethodID ctor = env->GetMethodID(string_class, "<init>", "([BLjava/lang/String;)V");
    jstring charset = ctor != nullptr ? env->NewStringUTF("UTF-8") : nullptr;
    if (charset != nullptr) {
        const jsize length = static_cast<jsize>(std::strlen(utf8));
        jbyteArray bytes = env->NewByteArray(length);
        if (bytes != nullptr) {
            env->SetByteArrayRegion(bytes, 0, length, reinterpret_cast<const jbyte *>(utf8));
            result = static_cast<jstring>(env->NewObject(string_class, ctor, bytes, charset));
            env->DeleteLocalRef(bytes);
        }
        env->DeleteLocalRef(charset);
    }
    env->DeleteLocalRef(string_class);
    return result;
}

// Throws a new exception_class(message), message read as standard UTF-8. exception_class must
// have a (String) constructor; every exception class of this library has one. When the message
// cannot be built, the exception that stopped it (OutOfMemoryError) is the one left pending. An
// exception already pending is the more precise error and is never replaced, as jni_guard_impl does.
inline void throw_utf8_impl(JNIEnv *env, jclass exception_class, const char *message) {
    if (env->ExceptionCheck()) {
        return;
    }
    if (message == nullptr) {
        message = "";
    }
    if (jni_utf8_is_ascii(message)) {
        env->ThrowNew(exception_class, message);
        return;
    }
    jstring text = new_string_utf8_impl(env, message);
    if (text == nullptr) {
        return;
    }
    jmethodID ctor = env->GetMethodID(exception_class, "<init>", "(Ljava/lang/String;)V");
    jobject exception = ctor != nullptr ? env->NewObject(exception_class, ctor, text) : nullptr;
    env->DeleteLocalRef(text);
    if (exception != nullptr) {
        env->Throw(static_cast<jthrowable>(exception));
        env->DeleteLocalRef(exception);
    }
}
