#include "api/quench_internal.h"
#include "exec/executor.h"
#include "runtime/engine.h"
#include "model/gguf_loader.h"
#include "model/safetensors_loader.h"
#include "model/tokenizer.h"
#include "model/chat_template.h"
#include "memory/kv_cache.h"
#include "memory/mem_account.h"  // trim_device_mempool

#include "core/logging.h"

#include <cstring>
#include <memory>
#include <vector>
#include <new>
#include <exception>
#include <filesystem>

#include <cuda_runtime.h>

// Out-of-line so quench_internal.h can forward-declare quench::Engine: this is the
// single point where unique_ptr<Engine> needs the complete type.
QuenchContext_T::QuenchContext_T() = default;
QuenchContext_T::~QuenchContext_T() = default;



// --- Error string ---

const char* quench_error_string(QuenchError err) {
    switch (err) {
        case QUENCH_SUCCESS:
            return "success";
        case QUENCH_ERROR_INVALID_ARG:
            return "invalid argument";
        case QUENCH_ERROR_OUT_OF_MEMORY:
            return "out of memory";
        case QUENCH_ERROR_CUDA:
            return "CUDA error";
        case QUENCH_ERROR_FILE_NOT_FOUND:
            return "file not found";
        case QUENCH_ERROR_INVALID_MODEL:
            return "invalid or corrupt model file";
        case QUENCH_ERROR_UNSUPPORTED:
            return "unsupported operation";
        case QUENCH_ERROR_INTERNAL:
            return "internal error";
        case QUENCH_ERROR_CANCELLED:
            return "cancelled";
        case QUENCH_ERROR_CAPACITY:
            return "insufficient KV capacity for this request";
        default:
            return "unknown error";
    }
}

// --- Config defaults ---

QuenchConfig quench_config_default(void) {
    QuenchConfig config;
    config.device_id = 0;
    config.kv_cache_max_blocks = 0;  // auto
    config.max_batch_size = 0;        // auto (engine detects from model size)
    config.max_seq_len = 0;           // auto (engine detects from model metadata + VRAM)
    config.compute_dtype = QUENCH_DTYPE_FP16;
    config.temperature = 0.6f;
    config.top_p = 0.95f;
    config.top_k = 0;
    config.max_tokens = 256;
    config.enable_green_contexts = 0;
    config.green_ctx_prefill_ratio = 0.8f;
    config.enable_pdl = 1;
    config.enable_cuda_graphs = 1;
    config.gpu_layers = -1;  // all on GPU
    config.kv_cache_dtype = QUENCH_DTYPE_FP16;
    config.ssm_state_dtype = QUENCH_DTYPE_FP32;
    config.vram_budget_mb = 0;                // use all available
    config.prefill_chunk_size = -1;           // per-arch default (2048 if supported, 0 otherwise)
    config.use_fp8_prefill = 0;               // FP16 weight cache by default
    config.use_nvfp4_decode = -1;             // auto (by quant/MoE/GDN, not by arch)
    config.use_mxfp4_prefill = 0;             // off by default
    config.dual_path_quant = 0;               // off by default
    config.min_kv_tokens = 0;                 // auto (pick reasonable minimum based on model)
    config.use_prefix_caching = 0;            // off by default
    config.prefix_cache_path[0] = '\0';       // no persistence by default
    config.prefix_pin_budget_pct = 25;        // cache_control pins capped at 25% of the KV pool
    config.mmproj_path = NULL;                // no vision model
    config.streaming_kv_enabled = 0;          // off by default (opt-in)
    config.streaming_kv_auto = 1;             // auto-enable when KV cache >90% full
    config.streaming_kv_n_sinks = 4;          // StreamingLLM paper default
    config.streaming_kv_window = 0;           // 0 = derive from ModelConfig::sliding_window
    config.streaming_kv_threshold = 0;        // 0 = auto
    return config;
}

// --- Generate params defaults ---

QuenchGenerateParams quench_generate_params_default(void) {
    QuenchGenerateParams params;
    params.temperature = 1.0f;
    params.top_p = 1.0f;
    params.top_k = 0;
    params.max_tokens = 256;
    params.seed = -1;
    params.min_p = 0.0f;
    params.typical_p = 1.0f;
    params.repetition_penalty = 1.0f;
    params.frequency_penalty = 0.0f;
    params.presence_penalty = 0.0f;
    params.repeat_last_n = 0;
    params.dry_multiplier = 0.0f;
    params.dry_base = 1.75f;
    params.dry_allowed_length = 2;
    params.dry_penalty_last_n = 0;
    params.mirostat = 0;
    params.mirostat_tau = 5.0f;
    params.mirostat_eta = 0.1f;
    params.apply_chat_template = 1;
    params.ignore_eos = 0;
    params.logprobs = 0;
    params.top_logprobs = 0;
    params.json_mode = 0;
    return params;
}

// --- Version ---

// Single-sourced from the CMake project version (-DQUENCH_VERSION_STRING=...).
// Falls back to a sentinel for non-CMake/ad-hoc builds so it never silently
// drifts from CMakeLists.txt again.
#ifndef QUENCH_VERSION_STRING
#define QUENCH_VERSION_STRING "0.0.0-dev"
#endif

const char* quench_version(void) { return QUENCH_VERSION_STRING; }

// --- Helper: map QuenchDType to quench::QType ---

static quench::QType map_dtype(QuenchDType dt) {
    switch (dt) {
        case QUENCH_DTYPE_FP32:
            return quench::QType::F32;
        case QUENCH_DTYPE_FP16:
            return quench::QType::F16;
        case QUENCH_DTYPE_BF16:
            return quench::QType::BF16;
        case QUENCH_DTYPE_FP8_E4M3:
            return quench::QType::FP8_E4M3;
        case QUENCH_DTYPE_FP8_E5M2:
            return quench::QType::FP8_E5M2;
        case QUENCH_DTYPE_INT8:
            return quench::QType::INT8;
        case QUENCH_DTYPE_INT4:
            return quench::QType::INT4;
        case QUENCH_DTYPE_INT32:
            return quench::QType::INT32;
        case QUENCH_DTYPE_FP4_E2M1:
            return quench::QType::FP4_E2M1;
        case QUENCH_DTYPE_NVFP4:
            return quench::QType::NVFP4;
        case QUENCH_DTYPE_MXFP4_KV:
            return quench::QType::MXFP4_KV;
        default:
            return quench::QType::F16;
    }
}

// --- Model Loading ---

QuenchError quench_model_load_ex(const char* path, QuenchModelFormat format, int load_mtp_head,
                           QuenchModel* out_model) {
    if (!path || !out_model) {
        return QUENCH_ERROR_INVALID_ARG;
    }
    *out_model = nullptr;

    try {
        std::unique_ptr<quench::Model> model;

        switch (format) {
            case QUENCH_FORMAT_GGUF:
                model = quench::load_gguf(path);
                break;
            case QUENCH_FORMAT_SAFETENSORS:
                model = quench::load_safetensors(path, load_mtp_head != 0);
                break;
            default:
                return QUENCH_ERROR_INVALID_ARG;
        }

        if (!model) {
            // Distinguish a genuinely missing path from one that exists but
            // failed to parse (bad GGUF magic, truncated SafeTensors, …). The
            // loaders return nullptr for both; reporting "file not found" for a
            // file that is present but corrupt is misleading.
            std::error_code ec;
            if (std::filesystem::exists(path, ec))
                return QUENCH_ERROR_INVALID_MODEL;
            return QUENCH_ERROR_FILE_NOT_FOUND;
        }

        auto handle = new (std::nothrow) QuenchModel_T();
        if (!handle) {
            return QUENCH_ERROR_OUT_OF_MEMORY;
        }

        handle->model = std::move(model);
        *out_model = handle;
        return QUENCH_SUCCESS;
    } catch (const std::bad_alloc&) {
        return QUENCH_ERROR_OUT_OF_MEMORY;
    } catch (const std::exception& e) {
        QUENCH_LOG_ERROR("quench_model_load: %s", e.what());
        return QUENCH_ERROR_INTERNAL;
    } catch (...) {
        return QUENCH_ERROR_INTERNAL;
    }
}

QuenchError quench_model_load(const char* path, QuenchModelFormat format, QuenchModel* out_model) {
    // Default load path: do NOT load the MTP head (saves ~1.57 GiB VRAM on
    // Qwen3.6). Callers that want spec-decode use quench_model_load_ex(.., 1, ..).
    return quench_model_load_ex(path, format, /*load_mtp_head=*/0, out_model);
}

void quench_model_free(QuenchModel model) {
    if (!model)
        return;
    delete model;
    quench::trim_device_mempool();
}

QuenchModelArch quench_model_arch(QuenchModel model) {
    if (!model || !model->model) {
        return QUENCH_ARCH_GENERIC;
    }
    return static_cast<QuenchModelArch>(quench::model_arch_c_api_id(model->model->config().arch));
}

int quench_model_n_layers(QuenchModel model) {
    if (!model || !model->model) {
        return 0;
    }
    return model->model->config().n_layers;
}

int quench_model_d_model(QuenchModel model) {
    if (!model || !model->model) {
        return 0;
    }
    return model->model->config().d_model;
}

int quench_model_vocab_size(QuenchModel model) {
    if (!model || !model->model) {
        return 0;
    }
    return model->model->config().vocab_size;
}

int32_t quench_model_bos_token(QuenchModel model) {
    if (!model || !model->model || !model->model->tokenizer()) {
        return -1;
    }
    const auto* tok = model->model->tokenizer();
    return tok->add_bos() ? tok->bos_id() : -1;
}

QuenchError quench_lora_load(QuenchContext ctx, const char* path, int32_t* out_id) {
    if (!ctx || !ctx->engine || !path || !out_id)
        return QUENCH_ERROR_INVALID_ARG;
    try {
        int id = ctx->engine->lora_load(path);
        if (id <= 0)
            return QUENCH_ERROR_INVALID_MODEL;
        *out_id = id;
        return QUENCH_SUCCESS;
    } catch (const std::exception& e) {
        QUENCH_LOG_ERROR("quench_lora_load: %s", e.what());
        return QUENCH_ERROR_INTERNAL;
    }
}

QuenchError quench_lora_set(QuenchContext ctx, int32_t adapter_id) {
    if (!ctx || !ctx->engine)
        return QUENCH_ERROR_INVALID_ARG;
    try {
        return ctx->engine->lora_set(adapter_id) ? QUENCH_SUCCESS : QUENCH_ERROR_INVALID_ARG;
    } catch (const std::exception& e) {
        QUENCH_LOG_ERROR("quench_lora_set: %s", e.what());
        return QUENCH_ERROR_INTERNAL;
    }
}

int quench_model_max_seq_len(QuenchModel model) {
    if (!model || !model->model) {
        return 0;
    }
    return model->model->config().max_seq_len;
}

// Effective context window the engine actually allocated (VRAM-aware auto-size).
// May be smaller than quench_model_max_seq_len when VRAM is tight — gate prompt
// length on THIS, not the model's declared max, to avoid KV/position overruns.
int quench_context_max_seq_len(QuenchContext ctx) {
    if (!ctx || !ctx->engine) {
        return 0;
    }
    return ctx->engine->max_seq_len();
}

// --- Context / Runtime ---

QuenchError quench_context_create(QuenchModel model, const QuenchConfig* config, QuenchContext* out_ctx) {
    if (!model || !config || !out_ctx) {
        return QUENCH_ERROR_INVALID_ARG;
    }

    *out_ctx = nullptr;

    if (!model->model) {
        return QUENCH_ERROR_INVALID_MODEL;
    }

    try {
        // Build EngineConfig from QuenchConfig
        quench::EngineConfig ecfg;
        ecfg.max_batch_size = config->max_batch_size;
        ecfg.max_seq_len = config->max_seq_len;
        ecfg.kv_cache_max_blocks = static_cast<int>(config->kv_cache_max_blocks);
        ecfg.compute_dtype = map_dtype(config->compute_dtype);
        ecfg.use_green_contexts = (config->enable_green_contexts != 0);
        ecfg.green_ctx_prefill_ratio = config->green_ctx_prefill_ratio;
        ecfg.use_cuda_graphs = (config->enable_cuda_graphs != 0);
        ecfg.use_pdl = (config->enable_pdl != 0);
        ecfg.gpu_layers = config->gpu_layers;
        ecfg.kv_cache_dtype = map_dtype(config->kv_cache_dtype);
        ecfg.ssm_state_dtype = map_dtype(config->ssm_state_dtype);
        ecfg.vram_budget_mb = config->vram_budget_mb;
        ecfg.temperature = config->temperature;
        ecfg.top_p = config->top_p;
        ecfg.top_k = config->top_k;
        ecfg.prefill_chunk_size = config->prefill_chunk_size;
        ecfg.use_fp8_prefill = (config->use_fp8_prefill != 0);
        ecfg.use_nvfp4_decode = config->use_nvfp4_decode;
        ecfg.min_kv_tokens = config->min_kv_tokens;
        ecfg.use_mxfp4_prefill = (config->use_mxfp4_prefill != 0);
        ecfg.dual_path_quant = (config->dual_path_quant != 0);
        ecfg.use_prefix_caching = (config->use_prefix_caching != 0);
        if (config->prefix_cache_path[0] != '\0')
            ecfg.prefix_cache_path = config->prefix_cache_path;
        ecfg.prefix_pin_budget_pct = config->prefix_pin_budget_pct;
        if (config->mmproj_path)
            ecfg.mmproj_path = config->mmproj_path;
        ecfg.streaming_kv_enabled = (config->streaming_kv_enabled != 0);
        ecfg.streaming_kv_auto = (config->streaming_kv_auto != 0);
        ecfg.streaming_kv_n_sinks = config->streaming_kv_n_sinks;
        ecfg.streaming_kv_window = config->streaming_kv_window;
        ecfg.streaming_kv_threshold = config->streaming_kv_threshold;

        // Create and initialize the engine
        auto engine = std::make_unique<quench::Engine>();
        if (!engine->init(model->model, ecfg)) {
            return QUENCH_ERROR_INTERNAL;
        }

        // Create the context handle
        auto ctx = new (std::nothrow) QuenchContext_T();
        if (!ctx) {
            return QUENCH_ERROR_OUT_OF_MEMORY;
        }

        ctx->model_handle = model;
        ctx->engine = std::move(engine);
        ctx->active_request = nullptr;

        *out_ctx = ctx;
        return QUENCH_SUCCESS;
    } catch (const std::bad_alloc&) {
        return QUENCH_ERROR_OUT_OF_MEMORY;
    } catch (const std::exception& e) {
        QUENCH_LOG_ERROR("quench_context_create: %s", e.what());
        return QUENCH_ERROR_INTERNAL;
    } catch (...) {
        return QUENCH_ERROR_INTERNAL;
    }
}

void quench_context_free(QuenchContext ctx) {
    if (!ctx)
        return;
    delete ctx;
    quench::trim_device_mempool();
}

// --- Helper: tokenize a prompt using chat template or raw encoding ---

namespace {

static std::vector<int32_t> tokenize_prompt(QuenchContext ctx, const char* prompt,
                                            const QuenchGenerateParams* params) {
    auto* tok = ctx->model_handle->model->tokenizer();
    const auto& tmpl = ctx->engine->chat_template();
    bool has_img = ctx->engine->has_vision() && ctx->engine->has_vision_input();

    // Tokenize the prompt, injecting image tokens if a vision image is set.
    if (params->apply_chat_template && !tmpl.is_raw()) {
        std::vector<quench::ChatMessage> messages = {{"user", prompt}};
        if (has_img) {
            return tmpl.apply_with_image(*tok, messages, 256);
        } else {
            return tmpl.apply(*tok, messages);
        }
    }

    auto tokens = tok->encode(prompt);
    if (tok->add_bos() && (tokens.empty() || tokens[0] != tok->bos_id())) {
        tokens.insert(tokens.begin(), static_cast<int32_t>(tok->bos_id()));
    }
    return tokens;
}

// --- Helper: apply sampling params from QuenchGenerateParams to a Request ---

static void apply_sampling_params(quench::Request& req, const QuenchGenerateParams* params) {
    req.max_tokens = params->max_tokens;
    req.temperature = params->temperature;
    req.top_p = params->top_p;
    req.top_k = params->top_k;
    req.seed = params->seed;
    req.min_p = params->min_p;
    req.typical_p = params->typical_p;
    req.repetition_penalty = params->repetition_penalty;
    req.frequency_penalty = params->frequency_penalty;
    req.presence_penalty = params->presence_penalty;
    req.repeat_last_n = params->repeat_last_n;
    req.dry_multiplier = params->dry_multiplier;
    req.dry_base = params->dry_base;
    req.dry_allowed_length = params->dry_allowed_length;
    req.dry_penalty_last_n = params->dry_penalty_last_n;
    req.mirostat = params->mirostat;
    req.mirostat_tau = params->mirostat_tau;
    req.mirostat_eta = params->mirostat_eta;
    if (params->mirostat == 2 && req.mirostat_mu == 0.0f)
        req.mirostat_mu = 2.0f * params->mirostat_tau;
}

}  // anonymous namespace

// --- Generation ---

// Thin wrapper helper: tokenise the prompt, prefill, then loop decode_step.
// `on_token` is invoked for every newly decoded token; return true to keep
// going, false to stop early (the caller's request gets cancelled).
//
// This is the shared body of quench_generate / quench_generate_streaming. Both
// public entry points stay ABI-stable; only their bodies collapse into this
// helper + quench_prefill_with_params + quench_decode_step.
namespace {

template <typename OnToken>
static QuenchError generate_via_prefill_decode_loop(QuenchContext ctx, const char* prompt,
                                                 const QuenchGenerateParams* params,
                                                 OnToken&& on_token) {
    auto* tok = ctx->model_handle->model->tokenizer();
    if (!tok)
        return QUENCH_ERROR_INVALID_MODEL;

    auto tokens = tokenize_prompt(ctx, prompt, params);

    // Empty token stream would walk into executor_forward.cu's "n_tokens must
    // be positive" guard and then trip a FATAL via the slice() of an
    // uninitialised logits tensor. Bail out cleanly here instead of crashing.
    if (tokens.empty()) {
        QUENCH_LOG_WARN(
            "quench_generate: prompt tokenised to 0 tokens (prompt='%.80s%s', "
            "model may lack vocab coverage or chat-template guard rejected it)",
            prompt, std::strlen(prompt) > 80 ? "…" : "");
        return QUENCH_ERROR_INVALID_ARG;
    }

    // Prefill: quench_prefill_with_params handles request lifecycle, sampling
    // params for the first-token sample, and the engine->step() loop until
    // PREFILLING completes.
    QuenchError err = quench_prefill_with_params(ctx, tokens.data(),
                                           static_cast<int>(tokens.size()), params);
    if (err != QUENCH_SUCCESS)
        return err;

    // The prefill last-chunk sampler emits the FIRST generation token into
    // req->output_tokens. quench_prefill_with_params marks those as "already
    // consumed" so a token-level (prefill+decode_step) caller doesn't get
    // them — but the high-level quench_generate / quench_generate_streaming
    // contract says every generated token reaches the caller. Reset the
    // cursor so quench_decode_step drains the prefill-sampled token(s) on its
    // first call(s) before stepping the engine again.
    ctx->consumed_output = 0;

    // Decode loop: quench_decode_step handles per-step sampling params,
    // multi-token (self-spec) consumption, and clears active_request when the
    // engine marks FINISHED (EOS or its own max_tokens guard fires).
    const int max_tokens = params->max_tokens > 0 ? params->max_tokens : 1;
    for (int i = 0; i < max_tokens; ++i) {
        int32_t token = 0;
        QuenchError step_err = quench_decode_step(ctx, params, &token);
        if (step_err != QUENCH_SUCCESS) {
            // INTERNAL after natural finish (active_request was cleared) is
            // the normal stop signal — anything else propagates.
            if (step_err == QUENCH_ERROR_INTERNAL && !ctx->active_request)
                break;
            return step_err;
        }

        if (!on_token(token))
            return QUENCH_ERROR_CANCELLED;

        // Engine signalled FINISHED (EOS / its own max_tokens) — decode_step
        // already cleaned up and set active_request = nullptr.
        if (!ctx->active_request)
            break;
    }

    return QUENCH_SUCCESS;
}

}  // namespace

QuenchError quench_generate_streaming(QuenchContext ctx, const char* prompt, const QuenchGenerateParams* params,
                                QuenchTokenCallback cb, void* user_data) {
    if (!ctx || !prompt || !params || !cb) {
        return QUENCH_ERROR_INVALID_ARG;
    }
    if (!ctx->engine) {
        return QUENCH_ERROR_INTERNAL;
    }

    try {
        auto* tok = ctx->model_handle->model->tokenizer();
        if (!tok)
            return QUENCH_ERROR_INVALID_MODEL;

        return generate_via_prefill_decode_loop(
            ctx, prompt, params, [&](int32_t token) -> bool {
                std::string text = tok->decode({token});
                int stop = cb(text.c_str(), text.size(), user_data);
                return stop == 0;
            });
    } catch (const std::bad_alloc&) {
        return QUENCH_ERROR_OUT_OF_MEMORY;
    } catch (const std::exception& e) {
        QUENCH_LOG_ERROR("quench_generate_streaming: %s", e.what());
        return QUENCH_ERROR_INTERNAL;
    } catch (...) {
        return QUENCH_ERROR_INTERNAL;
    }
}

QuenchError quench_generate(QuenchContext ctx, const char* prompt, const QuenchGenerateParams* params, char* output_buf,
                      size_t output_buf_size, size_t* output_len) {
    if (!ctx || !prompt || !params || !output_buf || output_buf_size == 0) {
        return QUENCH_ERROR_INVALID_ARG;
    }

    if (!ctx->engine) {
        return QUENCH_ERROR_INTERNAL;
    }

    try {
        auto* tok = ctx->model_handle->model->tokenizer();
        if (!tok)
            return QUENCH_ERROR_INVALID_MODEL;

        std::vector<int32_t> output_tokens;
        output_tokens.reserve(params->max_tokens > 0 ? params->max_tokens : 256);

        QuenchError err = generate_via_prefill_decode_loop(
            ctx, prompt, params, [&](int32_t token) -> bool {
                output_tokens.push_back(token);
                return true;
            });
        if (err != QUENCH_SUCCESS) {
            if (output_len) *output_len = 0;
            if (output_buf_size > 0) output_buf[0] = '\0';
            return err;
        }

        // Detokenise the accumulated tokens.
        std::string result = tok->decode(output_tokens);

        // Copy result to output buffer.
        size_t copy_len = result.size();
        if (copy_len >= output_buf_size) {
            copy_len = output_buf_size - 1;
        }
        std::memcpy(output_buf, result.data(), copy_len);
        output_buf[copy_len] = '\0';

        if (output_len) {
            *output_len = copy_len;
        }

        return QUENCH_SUCCESS;
    } catch (const std::bad_alloc&) {
        return QUENCH_ERROR_OUT_OF_MEMORY;
    } catch (const std::exception& e) {
        QUENCH_LOG_ERROR("quench_generate: %s", e.what());
        return QUENCH_ERROR_INTERNAL;
    } catch (...) {
        return QUENCH_ERROR_INTERNAL;
    }
}

QuenchError quench_tokenize(QuenchModel model, const char* text, int32_t* tokens, int* n_tokens, int max_tokens) {
    if (!model || !text || !tokens || !n_tokens || max_tokens <= 0) {
        return QUENCH_ERROR_INVALID_ARG;
    }

    auto* tok = model->model ? model->model->tokenizer() : nullptr;
    if (!tok || tok->vocab_size() == 0) {
        *n_tokens = 0;
        return QUENCH_ERROR_INVALID_MODEL;
    }

    try {
        auto ids = tok->encode(text);
        int count = static_cast<int>(ids.size());
        if (count > max_tokens)
            count = max_tokens;

        for (int i = 0; i < count; i++) {
            tokens[i] = ids[i];
        }
        *n_tokens = count;
        return QUENCH_SUCCESS;
    } catch (const std::bad_alloc&) {
        return QUENCH_ERROR_OUT_OF_MEMORY;
    } catch (const std::exception& e) {
        QUENCH_LOG_ERROR("quench_tokenize: %s", e.what());
        return QUENCH_ERROR_INTERNAL;
    } catch (...) {
        return QUENCH_ERROR_INTERNAL;
    }
}

QuenchError quench_detokenize(QuenchModel model, const int32_t* tokens, int n_tokens, char* output_buf,
                        size_t output_buf_size) {
    if (!model || !tokens || !output_buf || output_buf_size == 0 || n_tokens < 0) {
        return QUENCH_ERROR_INVALID_ARG;
    }

    auto* tok = model->model ? model->model->tokenizer() : nullptr;
    if (!tok || tok->vocab_size() == 0) {
        output_buf[0] = '\0';
        return QUENCH_ERROR_INVALID_MODEL;
    }

    try {
        std::vector<int32_t> ids(tokens, tokens + n_tokens);
        std::string text = tok->decode(ids);

        size_t copy_len = text.size();
        if (copy_len >= output_buf_size)
            copy_len = output_buf_size - 1;
        std::memcpy(output_buf, text.data(), copy_len);
        output_buf[copy_len] = '\0';
        return QUENCH_SUCCESS;
    } catch (const std::bad_alloc&) {
        return QUENCH_ERROR_OUT_OF_MEMORY;
    } catch (const std::exception& e) {
        QUENCH_LOG_ERROR("quench_detokenize: %s", e.what());
        return QUENCH_ERROR_INTERNAL;
    } catch (...) {
        return QUENCH_ERROR_INTERNAL;
    }
}

QuenchError quench_prefill_with_params(QuenchContext ctx, const int32_t* tokens, int n_tokens,
                                 const QuenchGenerateParams* params) {
    if (!ctx || !tokens || n_tokens <= 0) {
        return QUENCH_ERROR_INVALID_ARG;
    }

    if (!ctx->engine) {
        return QUENCH_ERROR_INTERNAL;
    }

    try {
        // If there is an existing active request, free its KV cache and mark
        // cancelled so the scheduler removes it from active_ on next schedule().
        if (ctx->active_request) {
            ctx->engine->kv_manager()->free_sequence(ctx->active_request->id);
            ctx->engine->reset_ssm_state(ctx->active_request->id);
            ctx->active_request->status = quench::RequestStatus::CANCELLED;
            ctx->active_request = nullptr;
        }

        // Create a request with the input tokens
        auto req = std::make_shared<quench::Request>();
        req->input_tokens.assign(tokens, tokens + n_tokens);
        req->max_tokens = 4096;  // Large default; decode_step controls actual stopping
        // Apply caller-supplied sampling params so the prefill last-chunk
        // sampler honours top_p / top_k / temperature for the FIRST token.
        // Without this Gemma-4-NVFP4 (and other noisy-logit-tail quants)
        // can sample garbage like <|end_of_text|> on token #0 and never
        // recover, even with temperature == 0.7 + properly-loaded
        // generation_config.json defaults.
        if (params) {
            req->temperature = params->temperature;
            req->top_p = params->top_p;
            req->top_k = params->top_k;
            req->seed = params->seed;
            req->min_p = params->min_p;
            req->typical_p = params->typical_p;
            req->repetition_penalty = params->repetition_penalty;
            req->frequency_penalty = params->frequency_penalty;
            req->presence_penalty = params->presence_penalty;
            req->repeat_last_n = params->repeat_last_n;
            req->dry_multiplier = params->dry_multiplier;
            req->dry_base = params->dry_base;
            req->dry_allowed_length = params->dry_allowed_length;
            req->dry_penalty_last_n = params->dry_penalty_last_n;
            req->mirostat = params->mirostat;
            req->mirostat_tau = params->mirostat_tau;
            req->mirostat_eta = params->mirostat_eta;
            if (params->mirostat == 2 && req->mirostat_mu == 0.0f)
                req->mirostat_mu = 2.0f * params->mirostat_tau;
        }
        req->ignore_eos = true;  // Don't stop during prefill — decode_step controls stopping
        req->status = quench::RequestStatus::PENDING;

        // Add to engine (assigns request id)
        ctx->engine->add_request(req);

        // Store as the active request for subsequent decode_step calls
        ctx->active_request = req;
        ctx->consumed_output = 0;

        // Run steps until prefill completes (may take multiple steps with chunked prefill)
        do {
            (void)ctx->engine->step();
        } while (req->status == quench::RequestStatus::PREFILLING);

        // Verify the request was prefilled. A cancellation here reported a flat
        // OUT_OF_MEMORY whatever the cause, which is where the admission
        // refusal lost its identity: the scheduler logs "needs N KV blocks but
        // cache capacity is M" and the caller saw "out of memory", i.e. a
        // transient-looking condition for something retrying will never fix.
        if (req->status == quench::RequestStatus::CANCELLED) {
            const bool capacity = req->cancel_reason == quench::CancelReason::KvCapacity;
            ctx->active_request = nullptr;
            return capacity ? QUENCH_ERROR_CAPACITY : QUENCH_ERROR_OUT_OF_MEMORY;
        }

        // After prefill, any tokens already in output_tokens are "consumed"
        // by the prefill path (the first decode token).
        ctx->consumed_output = req->output_tokens.size();

        return QUENCH_SUCCESS;
    } catch (const std::bad_alloc&) {
        return QUENCH_ERROR_OUT_OF_MEMORY;
    } catch (const std::exception& e) {
        QUENCH_LOG_ERROR("quench_prefill: %s", e.what());
        return QUENCH_ERROR_INTERNAL;
    } catch (...) {
        return QUENCH_ERROR_INTERNAL;
    }
}

// Legacy entry point — defaults to no caller-supplied sampling, leaves the
// first-token sample at end of prefill on Request struct defaults
// (top_p=1, top_k=0). Kept for ABI; new callers should use
// quench_prefill_with_params and pass the same params they'll use in
// quench_decode_step.
QuenchError quench_prefill(QuenchContext ctx, const int32_t* tokens, int n_tokens) {
    return quench_prefill_with_params(ctx, tokens, n_tokens, nullptr);
}

QuenchError quench_perplexity(QuenchContext ctx, const int32_t* tokens, int n_tokens, double* out_ppl) {
    if (!ctx || !tokens || n_tokens < 2 || !out_ppl) {
        return QUENCH_ERROR_INVALID_ARG;
    }
    if (!ctx->engine) {
        return QUENCH_ERROR_INTERNAL;
    }
    *out_ppl = -1.0;
    try {
        // Fresh context so the prefill covers exactly this corpus.
        quench_context_reset(ctx);
        // Chunked-prefill-aware: the engine accumulates per-position NLL
        // after every chunk it forwards. (The executor's hidden_ only
        // retains the most recent chunk, so the historical post-hoc
        // executor()->perplexity_nll() silently scored stale positions
        // whenever the resolved prefill chunk size was smaller than the
        // corpus — which is the C-API DEFAULT: prefill_chunk_size=-1
        // resolves to 512 on dense archs.)
        if (!ctx->engine->begin_perplexity_capture(tokens, n_tokens))
            return QUENCH_ERROR_INTERNAL;
        QuenchError e = quench_prefill(ctx, tokens, n_tokens);
        double ppl = -1.0;
        const bool reduced = ctx->engine->end_perplexity_capture(&ppl);  // always frees buffers
        // Release the prefill request's KV + recurrent slot. NOTE: do NOT
        // null active_request first — quench_context_reset only cleans up
        // (free_sequence / reset_ssm_state / slot release) when it still
        // sees the request; nulling early leaked the KV sequence AND the
        // SSM/GDN slot on every quench_perplexity call, so repeated scoring
        // on hybrid models drifted (stale recurrent state, slot pool decay).
        quench_context_reset(ctx);
        if (e != QUENCH_SUCCESS)
            return e;
        if (!reduced || ppl < 0.0)
            return QUENCH_ERROR_INTERNAL;
        *out_ppl = ppl;
        return QUENCH_SUCCESS;
    } catch (const std::exception& ex) {
        QUENCH_LOG_ERROR("quench_perplexity: %s", ex.what());
        return QUENCH_ERROR_INTERNAL;
    }
}

QuenchError quench_calibration_write(QuenchContext ctx, const char* path) {
    if (!ctx || !path || !*path)
        return QUENCH_ERROR_INVALID_ARG;
    if (!ctx->engine || !ctx->engine->executor())
        return QUENCH_ERROR_INTERNAL;
    try {
        const quench::ActivationCalibrator* calib = ctx->engine->executor()->calibration();
        if (!calib || calib->empty()) {
            QUENCH_LOG_ERROR(
                "quench_calibration_write: nothing collected — is [calibration] enabled set, and did a "
                "forward pass run?");
            return QUENCH_ERROR_INVALID_ARG;
        }
        const std::string model_id =
            (ctx->model_handle && ctx->model_handle->model) ? ctx->model_handle->model->source_path() : "";
        quench::CalibrationStats stats = calib->snapshot(model_id);
        std::string err = quench::write_calibration_stats(path, stats);
        if (!err.empty()) {
            QUENCH_LOG_ERROR("quench_calibration_write: %s", err.c_str());
            return QUENCH_ERROR_INTERNAL;
        }
        QUENCH_LOG_INFO("Calibration written: %s (%zu entries)", path, stats.entries.size());
        return QUENCH_SUCCESS;
    } catch (const std::exception& ex) {
        QUENCH_LOG_ERROR("quench_calibration_write: %s", ex.what());
        return QUENCH_ERROR_INTERNAL;
    }
}

QuenchError quench_decode_step(QuenchContext ctx, const QuenchGenerateParams* params, int32_t* out_token) {
    if (!ctx || !params || !out_token) {
        return QUENCH_ERROR_INVALID_ARG;
    }

    *out_token = 0;

    if (!ctx->engine || !ctx->active_request) {
        return QUENCH_ERROR_INTERNAL;
    }

    try {
        auto& req = ctx->active_request;

        // Apply ignore_eos BEFORE the finished check — when benchmarking with
        // synthetic tokens, prefill may produce EOS as the first output token
        // (e.g. Gemma-3), marking the request FINISHED.  If the caller wants
        // to ignore EOS, we must reset the request back to GENERATING.
        bool caller_ignore_eos = (params->ignore_eos != 0);
        if (caller_ignore_eos && req->status == quench::RequestStatus::FINISHED) {
            req->status = quench::RequestStatus::DECODING;
        }

        // Drain unconsumed tokens BEFORE the finished check — a multi-token
        // step (spec-ngram verify chunk) can emit several tokens AND finish
        // the request in the same engine step; returning the error here would
        // drop the queued tail (: CLI output ended mid-chunk).
        if (ctx->consumed_output < req->output_tokens.size()) {
            *out_token = req->output_tokens[ctx->consumed_output++];
            if (req->status == quench::RequestStatus::FINISHED &&
                ctx->consumed_output >= req->output_tokens.size()) {
                ctx->active_request = nullptr;
            }
            return QUENCH_SUCCESS;
        }

        // Check if already finished. FINISHED keeps returning INTERNAL — the
        // quench_generate loop relies on "INTERNAL + cleared active_request" as
        // its natural end-of-stream signal. CANCELLED is reported as such:
        // the engine cancels requests it cannot serve (KV pool exhausted at
        // decode — the scheduler logs the reason), and surfacing that as a
        // bare "internal error" hid the cause.
        if (req->status == quench::RequestStatus::FINISHED || req->status == quench::RequestStatus::CANCELLED) {
            bool cancelled = req->status == quench::RequestStatus::CANCELLED;
            // A capacity refusal is the one cancellation the caller can act on,
            // so it gets its own code rather than being folded into the generic
            // cancel (I6). The server maps it to 503.
            const bool capacity =
                cancelled && req->cancel_reason == quench::CancelReason::KvCapacity;
            ctx->active_request = nullptr;
            if (capacity)
                return QUENCH_ERROR_CAPACITY;
            return cancelled ? QUENCH_ERROR_CANCELLED : QUENCH_ERROR_INTERNAL;
        }

        // Update sampling params on the request for this step
        apply_sampling_params(*req, params);
        req->ignore_eos = caller_ignore_eos;
        req->logprobs = (params->logprobs != 0);
        req->top_logprobs = std::max(0, std::min(20, params->top_logprobs));
        req->json_mode = (params->json_mode != 0);

        // Need a new engine step (the multi-token drain above returned any
        // leftovers). A step may legitimately yield ZERO new tokens when it
        // only launches an async graph-loop burst (n-gram speculation miss
        // path) — the burst's tokens arrive on the next step's drain. Retry a
        // bounded number of times; a persistent zero-token stream is still an
        // internal error.
        size_t prev_output_size = req->output_tokens.size();
        for (int attempts = 0;
             attempts < 8 && req->output_tokens.size() == prev_output_size &&
             req->status == quench::RequestStatus::DECODING;
             ++attempts) {
            (void)ctx->engine->step();
        }

        if (req->output_tokens.size() <= prev_output_size) {
            // The engine cancelled the request mid-step (KV pool exhausted at
            // decode — reject-newest; the scheduler already logged the reason
            // and freed the sequence). Report CANCELLED, not INTERNAL.
            if (req->status == quench::RequestStatus::CANCELLED) {
                ctx->active_request = nullptr;
                return QUENCH_ERROR_CANCELLED;
            }
            QUENCH_LOG_ERROR("quench_decode_step: engine produced no token in 8 steps (status=%s, %zu outputs)",
                          quench::request_status_name(req->status), req->output_tokens.size());
            return QUENCH_ERROR_INTERNAL;
        }
        ctx->consumed_output = prev_output_size;
        *out_token = req->output_tokens[ctx->consumed_output++];

        // If the request finished (eos or max_tokens), clean up — but only
        // once every queued token has been handed to the caller (a verify
        // chunk can emit several tokens and finish in one step).
        if (req->status == quench::RequestStatus::FINISHED &&
            ctx->consumed_output >= req->output_tokens.size()) {
            // KV cache is already freed by engine step() on FINISHED
            ctx->active_request = nullptr;
        }

        return QUENCH_SUCCESS;
    } catch (const std::bad_alloc&) {
        return QUENCH_ERROR_OUT_OF_MEMORY;
    } catch (const std::exception& e) {
        QUENCH_LOG_ERROR("quench_decode_step: %s", e.what());
        return QUENCH_ERROR_INTERNAL;
    } catch (...) {
        return QUENCH_ERROR_INTERNAL;
    }
}

QuenchError quench_context_reset(QuenchContext ctx) {
    if (!ctx) {
        return QUENCH_ERROR_INVALID_ARG;
    }

    if (!ctx->engine) {
        return QUENCH_ERROR_INTERNAL;
    }

    // Free the active request's KV cache and mark it cancelled so the
    // scheduler removes it from active_ on the next schedule() call.
    if (ctx->active_request) {
        ctx->engine->kv_manager()->free_sequence(ctx->active_request->id);
        // Evict all cached blocks to prevent prefix cache hits with stale data
        while (ctx->engine->kv_manager()->evict_cached_block()) {}
        // Drop recurrent-state snapshots for the same reason (hybrid models)
        ctx->engine->clear_recurrent_snapshots();
        // Reset SSM state for hybrid models (Mamba2)
        ctx->engine->reset_ssm_state(ctx->active_request->id);
        ctx->active_request->status = quench::RequestStatus::CANCELLED;
        ctx->active_request = nullptr;
        ctx->consumed_output = 0;
    }

    // Sync GPU to ensure all async operations from the previous request complete
    // before resetting state. Without this, stale async graph loops or pending
    // kernel launches can corrupt the next request's data.
    cudaDeviceSynchronize();

    // Invalidate cached CUDA graphs — stale graph captures from the previous
    // request can produce non-deterministic output if replayed for a new request.
    ctx->engine->invalidate_graphs();

    // Reset batch pool upload cache — the next request may reuse the same
    // physical KV cache blocks, and stale cached block table pointers would
    // cause the GPU to read from old KV data.
    ctx->engine->reset_batch_pool_cache();

    // Reset MTP-side KV cache + accuracy telemetry for a clean new session.
    ctx->engine->mtp_accuracy_reset();

    return QUENCH_SUCCESS;
}

// --- MTP spec-decode ---

QuenchError quench_enable_mtp_spec_decode(QuenchContext ctx, int k) {
    if (!ctx) return QUENCH_ERROR_INVALID_ARG;
    if (!ctx->engine) return QUENCH_ERROR_INTERNAL;
    if (k <= 0) return QUENCH_ERROR_INVALID_ARG;
    return ctx->engine->enable_mtp_spec_decode(k) ? QUENCH_SUCCESS
                                                  : QUENCH_ERROR_INVALID_ARG;
}

// --- Vision (Multimodal) ---

