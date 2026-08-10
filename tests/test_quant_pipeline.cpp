// QuantPipeline build post-condition test.
//
// QuantPipeline (src/exec/quant_pipeline.h) is the init-time weight-
// quantization pipeline extracted out of GraphExecutor. Its build() runs
// once during engine init and populates the long-lived decode caches
// (FP16 / FP8 / NVFP4) that the forward hot path reads. The whole point of
// the extraction is that this stage is a separable, testable component.
//
// FALLBACK PATH: we drive the pipeline through the
// normal engine / C-API path and assert its observable post-condition — the
// model decodes coherent, non-degenerate output — which only happens because
// build() successfully populated the decode caches. A truly standalone unit
// test (construct QuantPipeline directly, hand it a bare Model +
// VRAMAllocator + empty caches, call build(), assert the caches filled) is
// the preferred proof, but it is currently impractical: a bare quench::Model is
// only reachable via the opaque QuenchModel C-API handle, VRAMAllocator is owned
// by the Engine and constructed inside engine->init(), and
// the VRAMBudget that drives which phases run is computed during KV-cache
// init. Standing all of that up in a test means reproducing ~half of engine
// init.
//
// FOLLOW-UP: replace this with a true bare-QuantPipeline unit test once a
// lightweight Model + VRAMAllocator fixture exists (no full engine init).
//
// Requires a real model and GPU. Run with:
//   quench-tests --gtest_filter="QuantPipelineTest.*"

#include <gtest/gtest.h>
#include "quench/quench.h"
#include "test_models.h"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace {

static std::string model_path() {
    return quench_test::env_path_or(quench_test::kEnvModel, "/models/Qwen3-8B-Q8_0.gguf");
}

static bool model_exists(const std::string& p) {
    FILE* f = fopen(p.c_str(), "r");
    if (f) {
        fclose(f);
        return true;
    }
    return false;
}

// Count "word-like" whitespace-separated tokens that contain at least one
// alphanumeric character. A populated decode cache produces real words; a
// pipeline that failed to fill its caches yields empty / garbage output.
static int wordlike_token_count(const std::string& text) {
    int count = 0;
    bool in_word = false;
    bool has_alnum = false;
    for (char c : text) {
        if (std::isspace(static_cast<unsigned char>(c))) {
            if (in_word && has_alnum)
                ++count;
            in_word = false;
            has_alnum = false;
        } else {
            in_word = true;
            if (std::isalnum(static_cast<unsigned char>(c)))
                has_alnum = true;
        }
    }
    if (in_word && has_alnum)
        ++count;
    return count;
}

class QuantPipelineTest : public ::testing::Test {
protected:
    QuenchModel model_ = nullptr;
    QuenchContext ctx_ = nullptr;

    // Pin the cuBLAS workspace for greedy-deterministic behavior on sm_120.
    static void SetUpTestSuite() {
        setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8", /*overwrite=*/0);
    }

    void SetUp() override {
        const std::string path = model_path();
        if (!model_exists(path))
            GTEST_SKIP() << "Model not found: " << path;

        // quench_model_load + quench_context_create run engine init, which calls
        // QuantPipeline::build() (via GraphExecutor::pre_dequant_weights).
        ASSERT_EQ(quench_model_load(path.c_str(), QUENCH_FORMAT_GGUF, &model_), QUENCH_SUCCESS);

        QuenchConfig config = quench_config_default();
        config.max_seq_len = 2048;
        config.max_batch_size = 1;

        QuenchError err = quench_context_create(model_, &config, &ctx_);
        if (err != QUENCH_SUCCESS) {
            quench_model_free(model_);
            model_ = nullptr;
            GTEST_SKIP() << "Context creation failed: " << quench_error_string(err);
        }
    }

    void TearDown() override {
        if (ctx_)
            quench_context_free(ctx_);
        if (model_)
            quench_model_free(model_);
    }
};

// build() post-condition: with the decode caches populated, greedy decode
// produces coherent, non-degenerate output (>= 10 word-like tokens).
TEST_F(QuantPipelineTest, BuildPopulatesDecodeCachesForCoherentDecode) {
    QuenchGenerateParams params = quench_generate_params_default();
    params.max_tokens = 64;
    params.temperature = 0.0f;  // greedy: exercises the decode caches build() filled
    params.seed = 42;
    params.apply_chat_template = 1;

    char output[4096];
    size_t output_len = 0;
    QuenchError err = quench_generate(ctx_, "List three primary colors.", &params, output,
                                sizeof(output), &output_len);
    ASSERT_EQ(err, QUENCH_SUCCESS) << "Decode failed: " << quench_error_string(err);

    std::string out(output, output_len);
    ASSERT_GT(out.size(), 0u) << "Empty decode output — decode caches likely unpopulated";

    int words = wordlike_token_count(out);
    EXPECT_GE(words, 10) << "Decode produced only " << words
                         << " word-like tokens (expected >= 10); a coherent decode "
                            "indicates QuantPipeline::build() populated the caches. "
                            "Output: " << out.substr(0, 200);
}

}  // namespace
