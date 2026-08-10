#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    QUENCH_SUCCESS = 0,
    QUENCH_ERROR_INVALID_ARG = -1,
    QUENCH_ERROR_OUT_OF_MEMORY = -2,
    QUENCH_ERROR_CUDA = -3,
    QUENCH_ERROR_FILE_NOT_FOUND = -4,
    QUENCH_ERROR_INVALID_MODEL = -5,
    QUENCH_ERROR_UNSUPPORTED = -6,
    QUENCH_ERROR_INTERNAL = -7,
    QUENCH_ERROR_CANCELLED = -8,
    // The engine refused to admit the request because it cannot fit — the KV
    // pool is smaller than the prompt needs, and no eviction can change that.
    // Distinct from CANCELLED (client went away, engine-internal abort) because
    // it is the one cancellation a caller can act on: shorten the prompt, or
    // give the process more VRAM. Invariant I6, "OOM is typed and recoverable".
    QUENCH_ERROR_CAPACITY = -9,
} QuenchError;

const char* quench_error_string(QuenchError err);

#ifdef __cplusplus
}
#endif
