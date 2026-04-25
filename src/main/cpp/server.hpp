// server.hpp — thin include shim; all implementations come from upstream
// llama.cpp translation units compiled via CMakeLists.txt target_sources.
// Replaces the former 3,780-line hand-ported server copy.
#pragma once
#include "server-context.h"
#include "server-queue.h"
#include "server-task.h"
#include "server-common.h"
#include "server-chat.h"
#include "utils.hpp"
