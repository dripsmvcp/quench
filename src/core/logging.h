#pragma once

#include <cstdio>
#include <cstdarg>
#include <cstdlib>
#include <atomic>

namespace quench {

enum class LogLevel : int {
    DEBUG = 0,
    INFO = 1,
    WARN = 2,
    ERROR = 3,
    FATAL = 4,
};

void log_set_level(LogLevel level);

// "debug" | "info" | "warn" | "error" | "fatal" (case-insensitive) → level.
// Returns false and leaves `out` untouched on anything else, so the caller can
// warn instead of silently picking a level nobody asked for.
bool log_level_from_string(const char* s, LogLevel& out);

// Inline for zero-overhead log level check in hot paths.
// The atomic is defined in logging.cpp; declared here for inlining.
extern std::atomic<LogLevel> g_log_level;
inline LogLevel log_get_level() { return g_log_level.load(std::memory_order_relaxed); }

// printf-style format checking. Without it a %s/%d mismatch in one of the ~1400
// call sites compiles clean and prints garbage at runtime — which is how the raw
// fprintf sites migrated in this change could have gone wrong silently.
void log_message(LogLevel level, const char* file, int line, const char* fmt, ...)
    __attribute__((format(printf, 4, 5)));

}  // namespace quench

#define QUENCH_LOG_DEBUG(...)                                                               \
    do {                                                                                 \
        if (::quench::log_get_level() <= ::quench::LogLevel::DEBUG)                            \
            ::quench::log_message(::quench::LogLevel::DEBUG, __FILE__, __LINE__, __VA_ARGS__); \
    } while (0)
#define QUENCH_LOG_INFO(...)                                                               \
    do {                                                                                \
        if (::quench::log_get_level() <= ::quench::LogLevel::INFO)                            \
            ::quench::log_message(::quench::LogLevel::INFO, __FILE__, __LINE__, __VA_ARGS__); \
    } while (0)
#define QUENCH_LOG_WARN(...)                                                               \
    do {                                                                                \
        if (::quench::log_get_level() <= ::quench::LogLevel::WARN)                            \
            ::quench::log_message(::quench::LogLevel::WARN, __FILE__, __LINE__, __VA_ARGS__); \
    } while (0)
#define QUENCH_LOG_ERROR(...)                                                               \
    do {                                                                                 \
        if (::quench::log_get_level() <= ::quench::LogLevel::ERROR)                            \
            ::quench::log_message(::quench::LogLevel::ERROR, __FILE__, __LINE__, __VA_ARGS__); \
    } while (0)
#define QUENCH_LOG_FATAL(...) ::quench::log_message(::quench::LogLevel::FATAL, __FILE__, __LINE__, __VA_ARGS__)

// --- Precondition check ---
// QUENCH_CHECK is the production-safe replacement for <cassert> assert(). Unlike
// assert(), it does NOT vanish under NDEBUG. On failure it logs at FATAL and
// aborts the process, surfacing internal-invariant violations in Release
// builds the same way they would in Debug.
//
// Use for internal-API preconditions where violation = programmer error.
// Do NOT use for user-input validation — return an QuenchError code instead.
#define QUENCH_CHECK(cond, ...)                       \
    do {                                           \
        if (!(cond)) {                             \
            QUENCH_LOG_FATAL(__VA_ARGS__);            \
            std::abort();                          \
        }                                          \
    } while (0)

// --- CUDA error checking macros ---
// Log-only: reports CUDA errors without affecting control flow.
// Use in cleanup paths or where failure is non-fatal.
#define QUENCH_CUDA_CHECK_LOG(call)                                                     \
    do {                                                                             \
        cudaError_t err_ = (call);                                                   \
        if (err_ != cudaSuccess) {                                                   \
            QUENCH_LOG_ERROR("CUDA error: %s at %s:%d — %s", #call, __FILE__, __LINE__, \
                          cudaGetErrorString(err_));                                 \
        }                                                                            \
    } while (0)

// Check + return false: for bool-returning init/setup functions.
#define QUENCH_CUDA_CHECK_BOOL(call)                                                    \
    do {                                                                             \
        cudaError_t err_ = (call);                                                   \
        if (err_ != cudaSuccess) {                                                   \
            QUENCH_LOG_ERROR("CUDA error: %s at %s:%d — %s", #call, __FILE__, __LINE__, \
                          cudaGetErrorString(err_));                                 \
            return false;                                                            \
        }                                                                            \
    } while (0)

// Post-launch check: place immediately after a kernel `<<<>>>` launch. Surfaces
// launch-time failures (invalid configuration, missing kernel image, OOM at
// launch) at the launch site instead of at the next synchronizing call.
// Uses cudaPeekAtLastError() so the sticky error is NOT cleared — existing
// downstream QUENCH_CUDA_CHECK_* handling still sees and propagates it.
#define QUENCH_CUDA_CHECK_LAUNCH()                                                      \
    do {                                                                             \
        cudaError_t err_ = cudaPeekAtLastError();                                    \
        if (err_ != cudaSuccess) {                                                   \
            QUENCH_LOG_ERROR("CUDA kernel launch failed at %s:%d — %s", __FILE__,       \
                          __LINE__, cudaGetErrorString(err_));                       \
        }                                                                            \
    } while (0)

// Check + return void: for void-returning functions.
#define QUENCH_CUDA_CHECK_VOID(call)                                                    \
    do {                                                                             \
        cudaError_t err_ = (call);                                                   \
        if (err_ != cudaSuccess) {                                                   \
            QUENCH_LOG_ERROR("CUDA error: %s at %s:%d — %s", #call, __FILE__, __LINE__, \
                          cudaGetErrorString(err_));                                 \
            return;                                                                  \
        }                                                                            \
    } while (0)
