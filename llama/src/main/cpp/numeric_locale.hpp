// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT

#ifndef JLLAMA_NUMERIC_LOCALE_HPP
#define JLLAMA_NUMERIC_LOCALE_HPP

#include <clocale>
#include <string>

#if defined(__APPLE__)
#include <xlocale.h> // newlocale / uselocale
#elif !defined(_WIN32)
#include <locale.h>
#endif

namespace jllama {

// The "C" numeric locale on the calling thread while it lives. llama.cpp's option parser reads
// numbers with std::stof / std::stoi, and its Jinja reads and writes the numbers of a chat template
// with std::stod and the C library, all of which follow LC_NUMERIC, and a JVM leaves the locale of
// its environment in effect: under de_DE "--temp 0.7" read as 0, "--repeat-penalty 0.9" was refused
// and a template rendered {{ 0.5 }} as "0," (measured). llama_server() sets the "C" numeric locale
// for the whole process, but only once it runs -- after the engine's parse, and never for a model
// load. Per thread, so the host process keeps its own: every C API call runs under one
// (jllama_api.cpp), and so do the load and the server's worker thread (jllama_engine.cpp). The one
// exception is the embedded server: llama_server_parsed() / llama_server_attach_parsed() (patches/0006,
// 0007) set the "C" numeric locale for the whole process, as llama_server() does, because its HTTP
// threads read and render numbers and no guard of ours runs on them.
class c_numeric_locale {
  public:
    c_numeric_locale(const c_numeric_locale &) = delete;
    c_numeric_locale &operator=(const c_numeric_locale &) = delete;
#if defined(_WIN32)
    c_numeric_locale() : previous_mode_(_configthreadlocale(_ENABLE_PER_THREAD_LOCALE)) {
        const char *current = std::setlocale(LC_NUMERIC, nullptr);
        previous_ = current != nullptr ? current : "C";
        std::setlocale(LC_NUMERIC, "C");
    }
    ~c_numeric_locale() {
        std::setlocale(LC_NUMERIC, previous_.c_str());
        if (previous_mode_ != -1) {
            _configthreadlocale(previous_mode_);
        }
    }

  private:
    int previous_mode_;
    std::string previous_;
#elif defined(__ANDROID__)
    // bionic formats and reads every number in the "C" locale whatever the locale is, and its
    // newlocale() ignores the base it is given (a fresh object each time, the duplocale() below leaked
    // one per call) and drops MB_CUR_MAX to 1 for a "C" locale -- nothing to do, and nothing to gain.
    c_numeric_locale() = default;
#else
    c_numeric_locale() : previous_(uselocale(static_cast<locale_t>(0))) {
#if defined(__APPLE__)
        // Apple's newlocale() copies the base and leaves it to the caller (Libc, locale/xlocale.c:
        // _duplocale(base), never a release), where POSIX lets it consume the base: a duplocale() of
        // the base leaked one locale object per C API call (a stream: one per token). Given directly;
        // LC_GLOBAL_LOCALE is a base it takes.
        c_locale_ = newlocale(LC_NUMERIC_MASK, "C", previous_);
#else
        // glibc and musl consume the base on success (and may return it modified), so they get a copy
        // of the thread's locale, which LC_GLOBAL_LOCALE itself must never be.
        locale_t base = duplocale(previous_);
        if (base != static_cast<locale_t>(0)) {
            c_locale_ = newlocale(LC_NUMERIC_MASK, "C", base);
            if (c_locale_ == static_cast<locale_t>(0)) {
                freelocale(base);
            }
        }
#endif
        if (c_locale_ != static_cast<locale_t>(0)) {
            uselocale(c_locale_);
        }
    }
    ~c_numeric_locale() {
        if (c_locale_ != static_cast<locale_t>(0)) {
            uselocale(previous_);
            freelocale(c_locale_);
        }
    }

  private:
    locale_t previous_;
    locale_t c_locale_ = static_cast<locale_t>(0);
#endif
};

} // namespace jllama

#endif // JLLAMA_NUMERIC_LOCALE_HPP
