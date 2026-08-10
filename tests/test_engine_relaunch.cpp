// Regression tests for engine relaunch — the quench-server model auto-swap path.
//
// Two production bugs (server [auto-swap] Qwen3.6-35B → Gemma-4-31B, 2026-06):
//
//  1. SIGSEGV on engine re-init after inference: the process-global
//     attention-cuBLAS handle kept the destroyed engine's stream bound
//     (cublasSetStream in the batched-attention path). The next engine's
//     attention_cublas_prewarm() issued its dummy GemmBatchedEx on that
//     dangling stream → cuBLAS algo heuristics → cuStreamGetGreenCtx →
//     segfault inside libcuda. The server died with no error output
//     (exit 139); docker restart-policy masked it as connection drops.
//
//  2. VRAM starvation on swap: weights are allocated via cudaMallocAsync from
//     the device default mempool, whose release threshold Engine init raises
//     to UINT64_MAX. Freeing model+context parked ~weights-sized memory in
//     the pool instead of returning it to the driver — the next model load
//     (plain-cudaMalloc paths, cudaMemGetInfo-based sizing and the upload
//     oversubscription gate) saw ~1.5 GB free on a 32 GB card and failed
//     ("Failed to upload token embedding").
//
// Requires a real model on disk: QUENCH_TEST_MODEL or the default
// /models/Qwen3-8B-Q8_0.gguf, matching test_api_generate.cpp.

#include <gtest/gtest.h>
#include "quench/quench.h"
#include "api/quench_internal.h"
#include "test_models.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace {

static const char* get_model_path() {
    return quench_test::env_cstr_or(quench_test::kEnvModel, "/models/Qwen3-8B-Q8_0.gguf");
}

static bool model_exists() {
    FILE* f = fopen(get_model_path(), "r");
    if (f) {
        fclose(f);
        return true;
    }
    return false;
}

#define SKIP_IF_NO_MODEL()                                           \
    do {                                                             \
        if (!model_exists())                                         \
            GTEST_SKIP() << "Model not found: " << get_model_path(); \
    } while (0)

static size_t device_free_mib() {
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess)
        return 0;
    return free_b >> 20;
}

// One full lifecycle: load → create context → generate (binds the global
// attention-cuBLAS handle to this engine's stream) → free everything.
static void run_one_cycle(const char* path) {
    QuenchModel model = nullptr;
    ASSERT_EQ(quench_model_load(path, QUENCH_FORMAT_GGUF, &model), QUENCH_SUCCESS);

    QuenchConfig config = quench_config_default();
    config.max_seq_len = 1024;
    config.max_batch_size = 1;

    QuenchContext ctx = nullptr;
    QuenchError err = quench_context_create(model, &config, &ctx);
    if (err != QUENCH_SUCCESS) {
        quench_model_free(model);
        FAIL() << "Context creation failed: " << quench_error_string(err);
    }

    QuenchGenerateParams params = quench_generate_params_default();
    params.seed = 42;
    params.max_tokens = 8;
    params.temperature = 0.7f;
    params.apply_chat_template = 1;

    char buf[1024] = {};
    size_t n = 0;
    EXPECT_EQ(quench_generate(ctx, "Say hi.", &params, buf, sizeof(buf), &n), QUENCH_SUCCESS);
    EXPECT_GT(n, 0u);

    quench_context_free(ctx);
    quench_model_free(model);
}

}  // namespace

TEST(EngineRelaunchTest, ReloadAfterInferenceReleasesVramAndDoesNotCrash) {
    SKIP_IF_NO_MODEL();

    size_t free_before = device_free_mib();
    ASSERT_GT(free_before, 0u);

    // Pool "used" is a PROCESS-wide counter, so an absolute bound on it is only
    // true when this test runs first. It did not: in the full test-e2e run
    // earlier tests leave blocks in the default pool and the figure read 1792
    // MiB against a 1024 MiB bound, while the same test passed in isolation.
    // That is the same category error the comment below rejects for the DEVICE
    // figure, one level down — so take a baseline and assert on the delta,
    // which is the part this cycle actually owns.
    unsigned long long pool_used_before = 0;
    {
        cudaMemPool_t p0 = nullptr;
        int d0 = 0;
        cudaGetDevice(&d0);
        if (cudaDeviceGetDefaultMemPool(&p0, d0) == cudaSuccess)
            (void)cudaMemPoolGetAttribute(p0, cudaMemPoolAttrUsedMemCurrent, &pool_used_before);
    }

    run_one_cycle(get_model_path());
    if (::testing::Test::HasFatalFailure())
        return;

    // Teardown must hand the weights back to the default mempool rather than
    // parking them there — that is the real regression risk, and freeing via
    // cudaFreeAsync (not cudaFree) is what makes the trim able to reclaim
    // them. Assert it at POOL level, which is the level at which it is true.
    //
    // Do NOT assert this at device level instead, by cudaMalloc'ing the
    // apparently-missing amount and treating success as proof that the memory
    // was merely under-reported. That check is unsound on WSL2/WDDM: the
    // driver oversubscribes into host memory and returns cudaSuccess, so the
    // probe passes whether or not the memory is really there.
    // Measured: a 28 GiB allocation succeeds on a 32 GB card with 22.6 GiB
    // reported free, and runs at 237 GB/s against 1531 GB/s resident.
    //
    // The device-level figure genuinely does not return here — WSL2/WDDM keeps
    // a process's peak commitment for the process lifetime, no matter what the
    // pool does — so asserting on it would encode a platform
    // property as a leak. What still guards is the second full cycle
    // below: the next load must succeed.
    size_t free_between = device_free_mib();
    cudaMemPool_t pool = nullptr;
    int dev = 0;
    cudaGetDevice(&dev);
    if (cudaDeviceGetDefaultMemPool(&pool, dev) == cudaSuccess) {
        // UsedMemCurrent, not ReservedMemCurrent. Reserved drops to 0 on the
        // trim even when the blocks were never returned — verified by
        // substituting cudaFree for cudaFreeAsync: reserved still
        // reads 0 while used stays at the full weight footprint and climbs to
        // 16600 MiB on the second cycle. Asserting on reserved would pass
        // straight through that failure.
        unsigned long long used = 0;
        ASSERT_EQ(cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemCurrent, &used), cudaSuccess);
        const unsigned long long retained = used > pool_used_before ? used - pool_used_before : 0;
        EXPECT_LT(retained >> 20, 1024u)
            << "this cycle left " << (retained >> 20) << " MiB in the default mempool as USED ("
            << (pool_used_before >> 20) << " -> " << (used >> 20)
            << " MiB) — the weights were freed with an API that does not return stream-ordered "
            << "blocks to the pool, so the trim can reclaim nothing. Device-reported "
            << "free went " << free_before << " -> " << free_between << " MiB, which is expected "
            << "on this platform and is NOT what this assertion is about.";
    }

    // Re-init after inference: a dangling stream on the global
    // attention-cuBLAS handle segfaults here unless the prewarm re-binds it.
    run_one_cycle(get_model_path());
}

//: a SECOND engine on the SAME model handle (load once, create/free/create
// context) must not CUDA-IMA. Some models free their source weight tensors for
// VRAM during the first engine's pre-dequant (Phase-4b), leaving dangling
// pointers a second build would read. The engine now rejects that up front
// (clean error), while models that never drop sources (dense Q8_0) still support
// create/free/create. Either outcome is acceptable here — the invariant is "no
// illegal access / no crash", and the process stays usable afterward.
TEST(EngineRelaunchTest, SecondEngineOnSameModelHandleNeverIMAs) {
    SKIP_IF_NO_MODEL();

    QuenchModel model = nullptr;
    ASSERT_EQ(quench_model_load(get_model_path(), QUENCH_FORMAT_GGUF, &model), QUENCH_SUCCESS);

    // Lifecycle diagnostic: dump the model's key tensor pointers/qtypes
    // at each phase so a mutation by engine #1 (which engine #2 then reads as
    // dangling) shows up as a diff in the test log.
    auto dump_state = [&](const char* tag) {
        quench::Model* m = model->model.get();
        fprintf(stderr,
                "[TENSORDUMP %s] tok_emb{d=%p q=%d sc=%p dev=%d} out_proj{d=%p q=%d sc=%p} "
                "allocs=%zu consumed=%d\n",
                tag, m->tok_emb_.data, (int)m->tok_emb_.qtype, m->tok_emb_.scales,
                (int)m->tok_emb_.on_device, m->out_proj_.data, (int)m->out_proj_.qtype,
                m->out_proj_.scales, m->gpu_allocations_.size(), (int)m->sources_consumed());
        for (int i = 0; i < 2 && i < m->n_layers(); ++i) {
            const auto& L = m->layer(i);
            fprintf(stderr,
                    "[TENSORDUMP %s] L%d wq{d=%p q=%d sc=%p} w_gate{d=%p q=%d sc=%p} "
                    "w_up{d=%p q=%d} ssm_in{d=%p q=%d sc=%p} ssm_out{d=%p q=%d}\n",
                    tag, i, L.wq.data, (int)L.wq.qtype, L.wq.scales, L.w_gate.data,
                    (int)L.w_gate.qtype, L.w_gate.scales, L.w_up.data, (int)L.w_up.qtype,
                    L.ssm_in.data, (int)L.ssm_in.qtype, L.ssm_in.scales, L.ssm_out.data,
                    (int)L.ssm_out.qtype);
        }
    };
    dump_state("post-load");

    QuenchConfig config = quench_config_default();
    config.max_seq_len = 1024;
    config.max_batch_size = 1;

    QuenchGenerateParams params = quench_generate_params_default();
    params.seed = 42;
    params.max_tokens = 8;
    params.temperature = 0.7f;
    params.apply_chat_template = 1;

    // First engine on the handle: create, generate, free.
    QuenchContext ctx1 = nullptr;
    ASSERT_EQ(quench_context_create(model, &config, &ctx1), QUENCH_SUCCESS);
    dump_state("post-ctx1-init");
    char buf1[1024] = {};
    size_t n1 = 0;
    EXPECT_EQ(quench_generate(ctx1, "Say hi.", &params, buf1, sizeof(buf1), &n1), QUENCH_SUCCESS);
    EXPECT_GT(n1, 0u);
    quench_context_free(ctx1);
    dump_state("post-ctx1-free");

    // Second engine on the SAME handle. Must return cleanly — success or a
    // plain error — never an illegal memory access.
    QuenchContext ctx2 = nullptr;
    QuenchError err = quench_context_create(model, &config, &ctx2);
    if (err == QUENCH_SUCCESS) {
        // Accepted (sources not dropped) → it must actually work.
        ASSERT_NE(ctx2, nullptr);
        char buf2[1024] = {};
        size_t n2 = 0;
        EXPECT_EQ(quench_generate(ctx2, "Say hi.", &params, buf2, sizeof(buf2), &n2), QUENCH_SUCCESS);
        EXPECT_GT(n2, 0u);
        quench_context_free(ctx2);
    } else {
        // Rejected (sources consumed) → clean error, no context handed out.
        EXPECT_EQ(ctx2, nullptr);
    }

    // The process must still be usable: a freshly LOADED model works regardless
    // of which branch above ran (proves the CUDA context was not poisoned).
    quench_model_free(model);
    run_one_cycle(get_model_path());
}
