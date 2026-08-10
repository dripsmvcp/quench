#pragma once

// CUDA-presence gate shared across the test suite.
//
// This is the single source of truth: do not re-define `HasCudaDevice()` or
// `SKIP_IF_NO_CUDA()` in individual test files — divergent bodies (a fully
// qualified call, a raw cudaGetDevice check, an unqualified call) are exactly
// how skip behavior drifts between suites.
//
// Kept separate from test_model_builder.h on purpose: that header pulls in
// model/model.h + cuda_fp16/half, which host-compiled .cpp tests should not be
// forced to include just to gate on a CUDA device. This header needs only
// <cuda_runtime.h>.

#include <cuda_runtime.h>

namespace quench {
namespace test {

inline bool HasCudaDevice() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    return err == cudaSuccess && count > 0;
}

}  // namespace test
}  // namespace quench

#define SKIP_IF_NO_CUDA()                               \
    do {                                                \
        if (!::quench::test::HasCudaDevice()) {            \
            GTEST_SKIP() << "No CUDA device available"; \
        }                                               \
    } while (0)
