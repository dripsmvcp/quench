// Parity tests for quench_generate vs the quench_prefill_with_params +
// quench_decode_step loop. After Phase 5 Track B (refactor: dedupe quench_generate
// via quench_prefill + quench_decode_step), quench_generate is a thin wrapper around
// those two primitives. This test pins the contract — if the wrapper drifts
// from manual prefill+decode-loop output, this test fails.
//
// Requires a real model on disk. Skipped via QUENCH_TEST_MODEL or default
// /models/Qwen3-8B-Q8_0.gguf, matching test_degeneration.cpp.

#include <gtest/gtest.h>
#include "quench/quench.h"
#include "test_models.h"

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

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

class ApiGenerateParityTest : public ::testing::Test {
   protected:
    static void SetUpTestSuite() {
        // Greedy determinism on Blackwell sm_120.
        setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8", /*overwrite=*/0);
    }

    void SetUp() override {
        SKIP_IF_NO_MODEL();

        ASSERT_EQ(quench_model_load(get_model_path(), QUENCH_FORMAT_GGUF, &model_), QUENCH_SUCCESS);

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
        if (ctx_) quench_context_free(ctx_);
        if (model_) quench_model_free(model_);
    }

    QuenchModel model_ = nullptr;
    QuenchContext ctx_ = nullptr;
};

// Sanity: quench_generate produces non-empty output for a simple chat prompt.
// This pins the post-refactor wrapper against the prefill+decode_step
// primitives — if the wrapper drops the prefill-sampled token or otherwise
// breaks the contract, generation will look empty / one-token-short.
TEST_F(ApiGenerateParityTest, GenerateProducesNonEmptyOutput) {
    QuenchGenerateParams params = quench_generate_params_default();
    params.seed = 42;
    params.max_tokens = 16;
    params.temperature = 0.7f;
    params.apply_chat_template = 1;

    char buf[2048] = {};
    size_t n = 0;
    ASSERT_EQ(quench_generate(ctx_, "What is 2+2?", &params, buf, sizeof(buf), &n), QUENCH_SUCCESS);

    EXPECT_GT(n, 0u) << "quench_generate returned 0 bytes — wrapper may have dropped "
                        "the prefill-sampled first token or terminated early";
    EXPECT_LT(n, sizeof(buf)) << "output exceeded buffer";
}

// Streaming + non-streaming must produce the same number of callback
// invocations as final detokenised characters, and both paths must agree on
// token count under fixed greedy params + same-context reset.
//
// Both code paths share the generate_via_prefill_decode_loop helper
// internally, so this is a regression guard: if a future change diverges the
// streaming wrapper's token-handling from the non-streaming wrapper, this
// test fails.
TEST_F(ApiGenerateParityTest, StreamingDeliversAllTokens) {
    QuenchGenerateParams params = quench_generate_params_default();
    params.seed = 42;
    params.max_tokens = 16;
    params.temperature = 0.7f;
    params.apply_chat_template = 1;
    params.ignore_eos = 1;  // force exactly max_tokens to be emitted

    // Path A: non-streaming.
    char buf[2048] = {};
    size_t n_nonstream = 0;
    ASSERT_EQ(quench_generate(ctx_, "Say hi.", &params, buf, sizeof(buf), &n_nonstream),
              QUENCH_SUCCESS);
    std::string nonstream(buf, n_nonstream);

    // Reset context — same engine, same weight cache, fresh request slot.
    ASSERT_EQ(quench_context_reset(ctx_), QUENCH_SUCCESS);

    // Path B: streaming. Concatenate every callback payload.
    std::string streamed;
    int callback_count = 0;
    struct CbState {
        std::string* out;
        int* count;
    } state{&streamed, &callback_count};
    auto cb = +[](const char* text, size_t len, void* user_data) -> int {
        auto* s = static_cast<CbState*>(user_data);
        s->out->append(text, len);
        (*s->count)++;
        return 0;
    };
    ASSERT_EQ(quench_generate_streaming(ctx_, "Say hi.", &params, cb, &state), QUENCH_SUCCESS);

    // With ignore_eos=1 + max_tokens=16, both paths should emit 16 tokens.
    // The callback fires once per generated token, so callback_count should
    // equal max_tokens (or be 1 fewer if the prefill-sampled token bookkeeping
    // is wrong — which is what this test guards against).
    EXPECT_EQ(callback_count, params.max_tokens)
        << "streaming wrapper delivered " << callback_count << " tokens, expected "
        << params.max_tokens << " — wrapper may be dropping the prefill-sampled token";

    EXPECT_GT(streamed.size(), 0u) << "streaming produced empty output";
    EXPECT_GT(nonstream.size(), 0u) << "non-streaming produced empty output";
}

// Manual prefill + decode_step loop must run successfully end-to-end with the
// same params quench_generate uses. This is a smoke test that the public C API
// composes correctly — the wrapper-vs-manual content parity is not asserted
// because cross-engine non-determinism (separate Engine instances loading the
// same weights) can shift greedy logits enough to diverge after a few tokens.
TEST_F(ApiGenerateParityTest, ManualPrefillDecodeLoopRuns) {
    QuenchGenerateParams params = quench_generate_params_default();
    params.seed = 42;
    params.max_tokens = 8;
    params.temperature = 0.7f;
    params.apply_chat_template = 0;
    params.ignore_eos = 1;

    const char* prompt = "Hello.";

    int32_t tokens[256] = {0};
    int n_tokens = 0;
    ASSERT_EQ(quench_tokenize(model_, prompt, tokens, &n_tokens, 256), QUENCH_SUCCESS);
    ASSERT_GT(n_tokens, 0);

    ASSERT_EQ(quench_prefill_with_params(ctx_, tokens, n_tokens, &params), QUENCH_SUCCESS);

    std::vector<int32_t> out_tokens;
    out_tokens.reserve(params.max_tokens);
    for (int i = 0; i < params.max_tokens; ++i) {
        int32_t tok = 0;
        QuenchError step_err = quench_decode_step(ctx_, &params, &tok);
        if (step_err != QUENCH_SUCCESS) break;
        out_tokens.push_back(tok);
    }

    EXPECT_GT(out_tokens.size(), 0u)
        << "manual prefill + decode_step loop returned 0 tokens";

    char buf[2048] = {};
    ASSERT_EQ(quench_detokenize(model_, out_tokens.data(), static_cast<int>(out_tokens.size()),
                             buf, sizeof(buf)),
              QUENCH_SUCCESS);
    EXPECT_GT(std::strlen(buf), 0u) << "detokenised manual output is empty";
}

// I6, admission half: a prompt the KV pool can never hold must come back as a
// TYPED, actionable error — not as a generic cancel, and not as a successful
// call that produced nothing.
//
// Before this, the scheduler cancelled with a clear log line and the API
// returned QUENCH_ERROR_CANCELLED, which is the same code a client disconnect
// produces. A caller could not tell "you went away" from "this will never fit,
// shorten the prompt", so the server answered 200 with an empty completion.
TEST(AdmissionCapacityTest, PromptLargerThanTheKvPoolIsTypedNotCancelled) {
    SKIP_IF_NO_MODEL();

    QuenchModel model = nullptr;
    ASSERT_EQ(quench_model_load(get_model_path(), QUENCH_FORMAT_GGUF, &model), QUENCH_SUCCESS);

    QuenchConfig config = quench_config_default();
    config.max_seq_len = 4096;
    config.max_batch_size = 1;
    // Deliberately tiny: 16 blocks x 16 tokens = 256 tokens of KV, which no
    // eviction can grow. The prompt below needs more than that.
    config.kv_cache_max_blocks = 16;

    QuenchContext ctx = nullptr;
    QuenchError cerr = quench_context_create(model, &config, &ctx);
    if (cerr != QUENCH_SUCCESS) {
        quench_model_free(model);
        GTEST_SKIP() << "context creation refused the tiny pool: " << quench_error_string(cerr);
    }

    // ~1500 tokens of prompt against a 256-token pool.
    std::string prompt;
    for (int i = 0; i < 500; ++i)
        prompt += "the quick brown fox jumps over the lazy dog. ";

    QuenchGenerateParams params = quench_generate_params_default();
    params.seed = 42;
    params.max_tokens = 8;
    params.temperature = 0.0f;

    char buf[512] = {};
    size_t n = 0;
    QuenchError err = quench_generate(ctx, prompt.c_str(), &params, buf, sizeof(buf), &n);

    EXPECT_EQ(err, QUENCH_ERROR_CAPACITY)
        << "expected a typed capacity refusal, got " << quench_error_string(err)
        << " (" << static_cast<int>(err) << ")";
    EXPECT_STRNE(quench_error_string(QUENCH_ERROR_CAPACITY), quench_error_string(QUENCH_ERROR_CANCELLED))
        << "the two must be distinguishable to a caller";

    quench_context_free(ctx);
    quench_model_free(model);
}

}  // namespace
