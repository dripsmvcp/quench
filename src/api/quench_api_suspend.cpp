// Suspend-to-RAM C API (weight snapshot + GPU release). Split out of
// quench_api.cpp as its own logical unit — see include/quench/quench.h for the
// documented suspend/resume flow and src/memory/weight_snapshot.h for the
// snapshot machinery.

#include "api/quench_internal.h"
#include "core/cuda_static_reset.h"
#include "core/logging.h"
#include "memory/mem_account.h"  // trim_device_mempool

#include <cuda_runtime.h>

#include <exception>
#include <new>

QuenchError quench_weights_snapshot_capture(QuenchModel model, size_t host_ram_headroom_mb,
                                      QuenchWeightSnapshot* out_snap) {
    if (!model || !model->model || !out_snap)
        return QUENCH_ERROR_INVALID_ARG;
    *out_snap = nullptr;
    try {
        auto snap = quench::WeightSnapshot::capture(*model->model,
                                                 host_ram_headroom_mb * (1024ull * 1024ull));
        auto handle = new (std::nothrow) QuenchWeightSnapshot_T();
        if (!handle)
            return QUENCH_ERROR_OUT_OF_MEMORY;
        handle->snap = std::move(snap);
        *out_snap = handle;
        return QUENCH_SUCCESS;
    } catch (const quench::SnapshotUnsupportedError& e) {
        QUENCH_LOG_ERROR("quench_weights_snapshot_capture: %s", e.what());
        return QUENCH_ERROR_UNSUPPORTED;
    } catch (const quench::SnapshotHostOomError& e) {
        QUENCH_LOG_ERROR("quench_weights_snapshot_capture: %s", e.what());
        return QUENCH_ERROR_OUT_OF_MEMORY;
    } catch (const std::bad_alloc&) {
        return QUENCH_ERROR_OUT_OF_MEMORY;
    } catch (const std::exception& e) {
        QUENCH_LOG_ERROR("quench_weights_snapshot_capture: %s", e.what());
        return QUENCH_ERROR_INTERNAL;
    }
}

QuenchError quench_weights_snapshot_arm(QuenchWeightSnapshot snap) {
    if (!snap || !snap->snap)
        return QUENCH_ERROR_INVALID_ARG;
    quench::weight_snapshot_arm(snap->snap.get());
    return QUENCH_SUCCESS;
}

void quench_weights_snapshot_free(QuenchWeightSnapshot snap) {
    if (!snap)
        return;
    if (snap->snap)
        quench::weight_snapshot_disarm(snap->snap.get());
    delete snap;
}

size_t quench_weights_snapshot_bytes(QuenchWeightSnapshot snap) {
    return (snap && snap->snap) ? snap->snap->total_bytes() : 0;
}

int quench_weights_snapshot_hits(QuenchWeightSnapshot snap) {
    return (snap && snap->snap) ? snap->snap->hits() : 0;
}

QuenchError quench_gpu_release(int device_reset) {
    cudaError_t sync = cudaDeviceSynchronize();
    if (sync != cudaSuccess) {
        QUENCH_LOG_WARN("quench_gpu_release: device sync failed (%s) — continuing", cudaGetErrorString(sync));
        (void)cudaGetLastError();
    }
    quench::trim_device_mempool();
    if (device_reset) {
        // Free + re-arm every lazily-created module-static CUDA resource
        // (cuBLAS handles, workspaces, scratch) while the context is still
        // valid — their `if (!ptr)` guards would otherwise hand out dangling
        // handles to the next engine after the reset.
        quench::reset_static_cuda_state();
        cudaError_t r = cudaDeviceReset();
        if (r != cudaSuccess) {
            QUENCH_LOG_ERROR("quench_gpu_release: cudaDeviceReset failed: %s", cudaGetErrorString(r));
            return QUENCH_ERROR_CUDA;
        }
        QUENCH_LOG_INFO("quench_gpu_release: CUDA primary context reset — process holds no GPU resources");
    }
    return QUENCH_SUCCESS;
}
