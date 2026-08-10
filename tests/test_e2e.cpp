#include <gtest/gtest.h>
#include "quench/quench.h"
#include "api/quench_internal.h"
#include "gguf_stub.h"
#include "test_models.h"
#include "runtime/engine.h"

#include <cuda_runtime.h>

#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <unistd.h>

namespace {

// Helper: get model path from environment variable QUENCH_TEST_MODEL.
// Tests that require a model are skipped if not set.
static std::string test_model_path() { return quench_test::env_path(quench_test::kEnvModel); }

// --- API sanity tests (no model required) ---

TEST(EndToEndTest, VersionString) {
    const char* version = quench_version();
    EXPECT_NE(version, nullptr);
    EXPECT_GT(strlen(version), 0u);
}

TEST(EndToEndTest, ConfigDefault) {
    QuenchConfig config = quench_config_default();
    EXPECT_GE(config.max_batch_size, 0);  // 0 = auto-detect
    EXPECT_GE(config.max_seq_len, 0);     // 0 = auto-detect
    EXPECT_EQ(config.compute_dtype, QUENCH_DTYPE_FP16);
    EXPECT_EQ(config.enable_pdl, 1);
    EXPECT_EQ(config.enable_cuda_graphs, 1);
    EXPECT_EQ(config.gpu_layers, -1);
}

TEST(EndToEndTest, GenerateParamsDefault) {
    QuenchGenerateParams params = quench_generate_params_default();
    EXPECT_GT(params.temperature, 0.0f);
    EXPECT_GT(params.top_p, 0.0f);
    EXPECT_GE(params.top_k, 0);
    EXPECT_GT(params.max_tokens, 0);
    EXPECT_EQ(params.seed, -1);
    EXPECT_EQ(params.apply_chat_template, 1);
}

TEST(EndToEndTest, ErrorStrings) {
    EXPECT_STREQ(quench_error_string(QUENCH_SUCCESS), "success");
    EXPECT_STREQ(quench_error_string(QUENCH_ERROR_INVALID_ARG), "invalid argument");
    EXPECT_STREQ(quench_error_string(QUENCH_ERROR_OUT_OF_MEMORY), "out of memory");
    EXPECT_STREQ(quench_error_string(QUENCH_ERROR_CUDA), "CUDA error");
    EXPECT_STREQ(quench_error_string(QUENCH_ERROR_FILE_NOT_FOUND), "file not found");
    EXPECT_STREQ(quench_error_string(QUENCH_ERROR_INVALID_MODEL), "invalid or corrupt model file");
}

TEST(EndToEndTest, LoadNonexistentModel) {
    QuenchModel model = nullptr;
    QuenchError err = quench_model_load("/nonexistent/path/model.gguf", QUENCH_FORMAT_GGUF, &model);
    EXPECT_NE(err, QUENCH_SUCCESS);
    EXPECT_EQ(model, nullptr);
}

TEST(EndToEndTest, NullArguments) {
    // model_load with null path
    QuenchModel model = nullptr;
    EXPECT_EQ(quench_model_load(nullptr, QUENCH_FORMAT_GGUF, &model), QUENCH_ERROR_INVALID_ARG);

    // model_load with null output
    EXPECT_EQ(quench_model_load("test.gguf", QUENCH_FORMAT_GGUF, nullptr), QUENCH_ERROR_INVALID_ARG);

    // tokenize with null model
    int32_t tokens[64];
    int n_tokens = 0;
    EXPECT_EQ(quench_tokenize(nullptr, "hello", tokens, &n_tokens, 64), QUENCH_ERROR_INVALID_ARG);

    // context_create with null model
    QuenchConfig cfg = quench_config_default();
    QuenchContext ctx = nullptr;
    EXPECT_EQ(quench_context_create(nullptr, &cfg, &ctx), QUENCH_ERROR_INVALID_ARG);

    // context_reset with null
    EXPECT_EQ(quench_context_reset(nullptr), QUENCH_ERROR_INVALID_ARG);

    // generate with null context
    QuenchGenerateParams params = quench_generate_params_default();
    char buf[256];
    size_t len;
    EXPECT_EQ(quench_generate(nullptr, "test", &params, buf, sizeof(buf), &len), QUENCH_ERROR_INVALID_ARG);

    // decode_step with null context
    int32_t tok;
    EXPECT_EQ(quench_decode_step(nullptr, &params, &tok), QUENCH_ERROR_INVALID_ARG);
}

// --- Model-dependent tests (require QUENCH_TEST_MODEL env var) ---

TEST(EndToEndModelTest, LoadModel) {
    const std::string path = test_model_path();
    if (path.empty())
        GTEST_SKIP() << "Set QUENCH_TEST_MODEL to run model tests";

    QuenchModel model = nullptr;
    QuenchError err = quench_model_load(path.c_str(), QUENCH_FORMAT_GGUF, &model);
    ASSERT_EQ(err, QUENCH_SUCCESS);
    ASSERT_NE(model, nullptr);

    EXPECT_GT(quench_model_n_layers(model), 0);
    EXPECT_GT(quench_model_d_model(model), 0);
    EXPECT_GT(quench_model_vocab_size(model), 0);

    quench_model_free(model);
}

TEST(EndToEndModelTest, Tokenize) {
    const std::string path = test_model_path();
    if (path.empty())
        GTEST_SKIP() << "Set QUENCH_TEST_MODEL to run model tests";

    QuenchModel model = nullptr;
    ASSERT_EQ(quench_model_load(path.c_str(), QUENCH_FORMAT_GGUF, &model), QUENCH_SUCCESS);

    int32_t tokens[256];
    int n_tokens = 0;
    ASSERT_EQ(quench_tokenize(model, "Hello world", tokens, &n_tokens, 256), QUENCH_SUCCESS);
    EXPECT_GT(n_tokens, 0);
    EXPECT_LE(n_tokens, 256);

    // Roundtrip: detokenize should produce something non-empty
    char buf[1024];
    ASSERT_EQ(quench_detokenize(model, tokens, n_tokens, buf, sizeof(buf)), QUENCH_SUCCESS);
    EXPECT_GT(strlen(buf), 0u);

    quench_model_free(model);
}

TEST(EndToEndModelTest, CreateContextAndGenerate) {
    const std::string path = test_model_path();
    if (path.empty())
        GTEST_SKIP() << "Set QUENCH_TEST_MODEL to run model tests";

    QuenchModel model = nullptr;
    ASSERT_EQ(quench_model_load(path.c_str(), QUENCH_FORMAT_GGUF, &model), QUENCH_SUCCESS);

    QuenchConfig config = quench_config_default();
    config.max_seq_len = 512;
    config.max_batch_size = 1;
    config.enable_cuda_graphs = 0;  // Simpler for testing

    QuenchContext ctx = nullptr;
    ASSERT_EQ(quench_context_create(model, &config, &ctx), QUENCH_SUCCESS);
    ASSERT_NE(ctx, nullptr);

    // Generate a short completion
    QuenchGenerateParams params = quench_generate_params_default();
    params.max_tokens = 16;
    params.temperature = 0.0f;  // Greedy for determinism
    params.apply_chat_template = 0;

    char output[4096];
    size_t output_len = 0;
    QuenchError err = quench_generate(ctx, "The capital of France is", &params, output, sizeof(output),
                                &output_len);
    ASSERT_EQ(err, QUENCH_SUCCESS);
    EXPECT_GT(output_len, 0u);

    quench_context_free(ctx);
    quench_model_free(model);
}

TEST(EndToEndModelTest, PrefillDecodeStep) {
    const std::string path = test_model_path();
    if (path.empty())
        GTEST_SKIP() << "Set QUENCH_TEST_MODEL to run model tests";

    QuenchModel model = nullptr;
    ASSERT_EQ(quench_model_load(path.c_str(), QUENCH_FORMAT_GGUF, &model), QUENCH_SUCCESS);

    QuenchConfig config = quench_config_default();
    config.max_seq_len = 256;
    config.max_batch_size = 1;
    config.enable_cuda_graphs = 0;

    QuenchContext ctx = nullptr;
    ASSERT_EQ(quench_context_create(model, &config, &ctx), QUENCH_SUCCESS);

    // Tokenize a prompt
    int32_t tokens[128];
    int n_tokens = 0;
    ASSERT_EQ(quench_tokenize(model, "Hello", tokens, &n_tokens, 128), QUENCH_SUCCESS);
    ASSERT_GT(n_tokens, 0);

    // Prefill
    ASSERT_EQ(quench_prefill(ctx, tokens, n_tokens), QUENCH_SUCCESS);

    // Decode a few tokens
    QuenchGenerateParams params = quench_generate_params_default();
    params.max_tokens = 8;
    params.temperature = 0.0f;

    for (int i = 0; i < 4; i++) {
        int32_t token = 0;
        QuenchError err = quench_decode_step(ctx, &params, &token);
        if (err != QUENCH_SUCCESS)
            break;  // Request may finish early (EOS)
        EXPECT_GT(token, 0);
    }

    // Reset and verify we can reuse the context
    ASSERT_EQ(quench_context_reset(ctx), QUENCH_SUCCESS);

    quench_context_free(ctx);
    quench_model_free(model);
}

// --- Long-context tests (real model required) ---

TEST(EndToEndModelTest, LongContext8k) {
    const std::string path = test_model_path();
    if (path.empty())
        GTEST_SKIP() << "Set QUENCH_TEST_MODEL to run model tests";

    QuenchModel model = nullptr;
    ASSERT_EQ(quench_model_load(path.c_str(), QUENCH_FORMAT_GGUF, &model), QUENCH_SUCCESS);

    QuenchConfig config = quench_config_default();
    config.max_seq_len = 16384;
    config.max_batch_size = 1;
    config.enable_cuda_graphs = 0;

    QuenchContext ctx = nullptr;
    ASSERT_EQ(quench_context_create(model, &config, &ctx), QUENCH_SUCCESS);

    // Build a long prompt (~6k tokens by repeating text)
    std::string prompt;
    while (prompt.size() < 24000)
        prompt += "The quick brown fox jumps over the lazy dog. ";
    prompt += "Summarize everything above in one sentence:";

    QuenchGenerateParams params = quench_generate_params_default();
    params.max_tokens = 32;
    params.temperature = 0.0f;
    params.apply_chat_template = 0;

    char output[4096];
    size_t output_len = 0;
    QuenchError err = quench_generate(ctx, prompt.c_str(), &params, output, sizeof(output), &output_len);
    ASSERT_EQ(err, QUENCH_SUCCESS) << "8k context generation failed: " << quench_error_string(err);
    EXPECT_GT(output_len, 5u) << "Output too short — likely degenerate";

    quench_context_free(ctx);
    quench_model_free(model);
}

TEST(EndToEndModelTest, LongContext16k_NVFP4KV) {
    const std::string path = test_model_path();
    if (path.empty())
        GTEST_SKIP() << "Set QUENCH_TEST_MODEL to run model tests";

    QuenchModel model = nullptr;
    ASSERT_EQ(quench_model_load(path.c_str(), QUENCH_FORMAT_GGUF, &model), QUENCH_SUCCESS);

    QuenchConfig config = quench_config_default();
    config.max_seq_len = 16384;
    config.max_batch_size = 1;
    config.enable_cuda_graphs = 0;
    config.kv_cache_dtype = QUENCH_DTYPE_NVFP4;

    QuenchContext ctx = nullptr;
    QuenchError err = quench_context_create(model, &config, &ctx);
    if (err != QUENCH_SUCCESS) {
        quench_model_free(model);
        GTEST_SKIP() << "Context creation failed (NVFP4 KV may not fit): " << quench_error_string(err);
    }

    // Build a ~16k token prompt
    std::string prompt;
    while (prompt.size() < 60000)
        prompt += "The quick brown fox jumps over the lazy dog. ";
    prompt += "What animal was mentioned?";

    QuenchGenerateParams params = quench_generate_params_default();
    params.max_tokens = 16;
    params.temperature = 0.0f;
    params.apply_chat_template = 0;

    char output[2048];
    size_t output_len = 0;
    err = quench_generate(ctx, prompt.c_str(), &params, output, sizeof(output), &output_len);
    EXPECT_EQ(err, QUENCH_SUCCESS) << "16k context generation failed: " << quench_error_string(err);
    if (err == QUENCH_SUCCESS) {
        EXPECT_GT(output_len, 0u);
    }

    quench_context_free(ctx);
    quench_model_free(model);
}

// --- Stub GGUF tests (no real model required, uses generated ~200 KB GGUF) ---

class StubModelTest : public ::testing::Test {
protected:
    void SetUp() override {
        stub_path_ = quench::test::generate_gguf_stub("llama");
        ASSERT_FALSE(stub_path_.empty()) << "Failed to generate stub GGUF";
    }

    void TearDown() override {
        if (!stub_path_.empty())
            unlink(stub_path_.c_str());
        stub_path_.clear();
    }

    std::string stub_path_;
};

TEST_F(StubModelTest, LoadStubModel) {
    QuenchModel model = nullptr;
    QuenchError err = quench_model_load(stub_path_.c_str(), QUENCH_FORMAT_GGUF, &model);
    ASSERT_EQ(err, QUENCH_SUCCESS) << "Failed to load stub GGUF: " << quench_error_string(err);
    ASSERT_NE(model, nullptr);

    EXPECT_EQ(quench_model_n_layers(model), 1);
    EXPECT_EQ(quench_model_d_model(model), 64);
    EXPECT_EQ(quench_model_vocab_size(model), 256);

    quench_model_free(model);
}

TEST_F(StubModelTest, TokenizeStub) {
    QuenchModel model = nullptr;
    ASSERT_EQ(quench_model_load(stub_path_.c_str(), QUENCH_FORMAT_GGUF, &model), QUENCH_SUCCESS);

    // Stub tokenizer has 256 byte-tokens but no BPE merge rules.
    // quench_tokenize may return 0 tokens (no merges to apply) or
    // fall back to byte-level encoding. Either is acceptable.
    int32_t tokens[256];
    int n_tokens = 0;
    QuenchError err = quench_tokenize(model, "Hello", tokens, &n_tokens, 256);
    // Success or graceful failure — no crash
    EXPECT_TRUE(err == QUENCH_SUCCESS || n_tokens == 0);

    quench_model_free(model);
}

TEST_F(StubModelTest, CreateContextAndInfer) {
    QuenchModel model = nullptr;
    ASSERT_EQ(quench_model_load(stub_path_.c_str(), QUENCH_FORMAT_GGUF, &model), QUENCH_SUCCESS);

    QuenchConfig config = quench_config_default();
    config.max_seq_len = 64;
    config.max_batch_size = 1;
    config.enable_cuda_graphs = 0;
    config.enable_pdl = 0;

    QuenchContext ctx = nullptr;
    QuenchError err = quench_context_create(model, &config, &ctx);
    // Context creation involves GPU weight upload; this may fail if CUDA is not
    // available or if the tiny model trips some validation. Either outcome is
    // acceptable — the key check is no crash.
    if (err != QUENCH_SUCCESS) {
        // "Expected without GPU" is one reason context creation fails. A CUDA
        // context left in an error state is a different one, and skipping on it
        // is how this test poisoned the ~57 tests that run after it in test-e2e
        // without ever going red. Distinguish the two before skipping.
        cudaError_t sticky = cudaDeviceSynchronize();
        if (sticky == cudaSuccess)
            sticky = cudaGetLastError();
        quench_model_free(model);
        ASSERT_EQ(sticky, cudaSuccess)
            << "context creation failed AND left the CUDA context in an error state ("
            << cudaGetErrorString(sticky) << "). Every later test in this binary will fail on a "
            << "context it did not break. This guard exists because that is exactly what happened: "
            << "~Engine closed the T2 arena without re-arming the module statics that had taken a "
            << "slice from it, so the second engine matmul'd into freed memory. Fixed by calling "
            << "reset_static_cuda_state() after engine_arena_close(); this assertion is what makes "
            << "a regression of that class loud instead of silent.";
        GTEST_SKIP() << "Context creation failed (expected without GPU): " << quench_error_string(err);
    }
    ASSERT_NE(ctx, nullptr);

    // Attempt a short generation (random weights = garbage output, but no crash)
    QuenchGenerateParams params = quench_generate_params_default();
    params.max_tokens = 4;
    params.temperature = 0.0f;
    params.apply_chat_template = 0;

    char output[1024];
    size_t output_len = 0;
    err = quench_generate(ctx, "AB", &params, output, sizeof(output), &output_len);
    // Generation may fail with tiny random weights; we just check no crash/abort
    (void)err;

    quench_context_free(ctx);
    quench_model_free(model);

    // The contract is stronger than "no crash": the test must also leave the
    // CUDA context clean. A run that leaves it in an error state (every
    // teardown free failing with an illegal memory access, swallowed by the
    // engine's guard) kills the ~57 tests that run after this one in test-e2e
    // on a context they did not break. The classic mechanism: ~Engine closing
    // the T2 arena without re-arming the module statics holding a slice of
    // it, so the SECOND engine in a process matmuls into freed memory. A
    // "cublasLtMatmul failed (status 14)" line preceding the IMA is the same
    // symptom, not a separate defect: it appears in every run that has the
    // dangling pointer and in none that does not.
    // This assertion stays because the class is silent by construction.
    {
        cudaError_t sticky = cudaDeviceSynchronize();
        if (sticky == cudaSuccess)
            sticky = cudaGetLastError();
        EXPECT_EQ(sticky, cudaSuccess)
            << "the stub model left the CUDA context in an error state (" << cudaGetErrorString(sticky)
            << "). Every later test in this binary will fail on a context it did not break.";
    }
}

TEST_F(StubModelTest, PrefillDecodeStub) {
    QuenchModel model = nullptr;
    ASSERT_EQ(quench_model_load(stub_path_.c_str(), QUENCH_FORMAT_GGUF, &model), QUENCH_SUCCESS);

    QuenchConfig config = quench_config_default();
    config.max_seq_len = 64;
    config.max_batch_size = 1;
    config.enable_cuda_graphs = 0;
    config.enable_pdl = 0;

    QuenchContext ctx = nullptr;
    QuenchError err = quench_context_create(model, &config, &ctx);
    if (err != QUENCH_SUCCESS) {
        quench_model_free(model);
        GTEST_SKIP() << "Context creation failed: " << quench_error_string(err);
    }

    // Tokenize — stub has no BPE merges, use raw token IDs instead
    int32_t tokens[] = {72, 105};  // ASCII 'H', 'i'
    int n_tokens = 2;

    // Prefill
    err = quench_prefill(ctx, tokens, n_tokens);
    if (err != QUENCH_SUCCESS) {
        // Prefill may fail with tiny model; acceptable if no crash
        quench_context_free(ctx);
        quench_model_free(model);
        return;
    }

    // Decode a couple tokens
    QuenchGenerateParams params = quench_generate_params_default();
    params.max_tokens = 4;
    params.temperature = 0.0f;

    for (int i = 0; i < 2; i++) {
        int32_t token = 0;
        err = quench_decode_step(ctx, &params, &token);
        if (err != QUENCH_SUCCESS)
            break;
    }

    // Reset should always succeed
    EXPECT_EQ(quench_context_reset(ctx), QUENCH_SUCCESS);

    quench_context_free(ctx);
    quench_model_free(model);
}

TEST_F(StubModelTest, VRAMLeakDetection) {
    QuenchModel model = nullptr;
    ASSERT_EQ(quench_model_load(stub_path_.c_str(), QUENCH_FORMAT_GGUF, &model), QUENCH_SUCCESS);

    QuenchConfig config = quench_config_default();
    config.max_seq_len = 64;
    config.max_batch_size = 1;
    config.enable_cuda_graphs = 0;
    config.enable_pdl = 0;

    QuenchContext ctx = nullptr;
    QuenchError err = quench_context_create(model, &config, &ctx);
    if (err != QUENCH_SUCCESS) {
        quench_model_free(model);
        GTEST_SKIP() << "Context creation failed: " << quench_error_string(err);
    }

    // Run one warm-up request so lazy CUDA allocations are settled
    {
        int32_t warmup_tokens[] = {1, 2, 3};
        err = quench_prefill(ctx, warmup_tokens, 3);
        if (err == QUENCH_SUCCESS) {
            QuenchGenerateParams p = quench_generate_params_default();
            p.max_tokens = 2;
            p.temperature = 0.0f;
            p.seed = 42;
            int32_t tok;
            quench_decode_step(ctx, &p, &tok);
        }
        quench_context_reset(ctx);
    }

    // Measure VRAM baseline after warm-up
    cudaDeviceSynchronize();
    size_t free_before = 0, total = 0;
    cudaMemGetInfo(&free_before, &total);

    // Run 20 prefill+decode+reset cycles
    constexpr int kNumRequests = 20;
    for (int i = 0; i < kNumRequests; i++) {
        int32_t tokens[] = {1, 2, 3, 4, 5};
        err = quench_prefill(ctx, tokens, 5);
        if (err != QUENCH_SUCCESS)
            break;

        QuenchGenerateParams params = quench_generate_params_default();
        params.max_tokens = 4;
        params.temperature = 0.0f;
        params.seed = 100 + i;

        for (int j = 0; j < 2; j++) {
            int32_t out_token = 0;
            err = quench_decode_step(ctx, &params, &out_token);
            if (err != QUENCH_SUCCESS)
                break;
        }

        err = quench_context_reset(ctx);
        if (err != QUENCH_SUCCESS)
            break;
    }

    // Measure VRAM after all requests
    cudaDeviceSynchronize();
    size_t free_after = 0;
    cudaMemGetInfo(&free_after, &total);

    size_t leak = (free_before > free_after) ? (free_before - free_after) : 0;
    float leak_pct = 100.0f * static_cast<float>(leak) / static_cast<float>(total);
    EXPECT_LT(leak_pct, 5.0f) << "VRAM leak detected: " << (leak / (1024 * 1024)) << " MiB after "
                              << kNumRequests << " requests (" << leak_pct << "% of total "
                              << (total / (1024 * 1024)) << " MiB)";

    quench_context_free(ctx);
    quench_model_free(model);
}

// ---------------------------------------------------------------------------
// Multi-decode output isolation correctness test
// Verifies that 2 requests running concurrently produce non-empty, distinct
// output — proving that their KV caches and logit buffers are isolated.
// ---------------------------------------------------------------------------

TEST(EndToEndModelTest, MultiDecodeOutputIsolation) {
    const std::string path = test_model_path();
    if (path.empty())
        GTEST_SKIP() << "Set QUENCH_TEST_MODEL to run model tests";

    QuenchModel model = nullptr;
    ASSERT_EQ(quench_model_load(path.c_str(), QUENCH_FORMAT_GGUF, &model), QUENCH_SUCCESS);
    ASSERT_NE(model, nullptr);

    QuenchConfig config = quench_config_default();
    config.max_seq_len = 256;
    config.max_batch_size = 4;
    config.enable_cuda_graphs = 0;

    QuenchContext ctx = nullptr;
    ASSERT_EQ(quench_context_create(model, &config, &ctx), QUENCH_SUCCESS);
    ASSERT_NE(ctx, nullptr);

    quench::Engine* engine = ctx->engine.get();
    ASSERT_NE(engine, nullptr);

    // Tokenize two different prompts
    const char* prompts[2] = {
        "The capital of France is",
        "The capital of Germany is",
    };

    std::shared_ptr<quench::Request> reqs[2];
    for (int i = 0; i < 2; i++) {
        int32_t tokens[128];
        int n_tokens = 0;
        ASSERT_EQ(quench_tokenize(model, prompts[i], tokens, &n_tokens, 128), QUENCH_SUCCESS);
        ASSERT_GT(n_tokens, 0);

        auto req = std::make_shared<quench::Request>();
        req->input_tokens.assign(tokens, tokens + n_tokens);
        req->max_tokens = 16;
        req->temperature = 0.0f;  // greedy for determinism
        req->top_p = 1.0f;
        req->top_k = 0;
        req->ignore_eos = false;
        req->status = quench::RequestStatus::PENDING;
        reqs[i] = req;
        engine->add_request(req);
    }

    // Step until both requests finish
    constexpr int kMaxSteps = 512;
    for (int step = 0; step < kMaxSteps; step++) {
        bool all_done = true;
        for (int i = 0; i < 2; i++) {
            if (reqs[i]->status != quench::RequestStatus::FINISHED &&
                reqs[i]->status != quench::RequestStatus::CANCELLED) {
                all_done = false;
                break;
            }
        }
        if (all_done) break;
        (void)engine->step();
    }

    // Both requests must have finished (not cancelled)
    for (int i = 0; i < 2; i++) {
        EXPECT_EQ(reqs[i]->status, quench::RequestStatus::FINISHED)
            << "Request " << i << " did not finish (status=" << static_cast<int>(reqs[i]->status) << ")";
    }

    // Detokenize output tokens for each request
    std::string outputs[2];
    for (int i = 0; i < 2; i++) {
        ASSERT_FALSE(reqs[i]->output_tokens.empty())
            << "Request " << i << " produced no output tokens";

        char buf[1024] = {};
        ASSERT_EQ(quench_detokenize(model, reqs[i]->output_tokens.data(),
                                 static_cast<int>(reqs[i]->output_tokens.size()),
                                 buf, sizeof(buf)),
                  QUENCH_SUCCESS);
        outputs[i] = std::string(buf);
        EXPECT_GT(outputs[i].size(), 0u) << "Request " << i << " has empty decoded output";
    }

    // Outputs must be different — proves request isolation
    EXPECT_NE(outputs[0], outputs[1])
        << "Both requests produced identical output '" << outputs[0]
        << "' — KV cache isolation may be broken";

    quench_context_free(ctx);
    quench_model_free(model);
}

}  // namespace
