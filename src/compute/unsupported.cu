// Entry points for features this build does not ship (vision, MoE, GDN/SSM,
// MLA, MXFP4, K-quant dense GEMM, LoRA, encoder, SafeTensors). The dispatch
// layer is written against the full interface surface; here every optional
// entry point either no-ops / reports "unavailable" (capability probes,
// teardown) or logs FATAL and aborts (compute paths that are unreachable for
// a llama-family dense GGUF model).
#include "core/logging.h"
#include "core/dispatch_policy.h"
#include "exec/executor.h"
#include "exec/expert_cache.h"
#include "exec/moe_workspace.h"
#include "exec/quant_pipeline.h"
#include "runtime/engine.h"
#include "model/model.h"
#include "model/image_placeholders.h"
#include "model/mrope_positions.h"
#include "model/safetensors_loader.h"
#include "lora/lora_adapter.h"
#include "quant/dequant_gptq.h"
#include "quant/calibration_stats.h"
#include "quant/mxfp4_gemm.h"
#include "quant/cutlass_mxfp4_weight.h"
#include "compute/attention_paged.h"
#include "compute/attention_mxfp4_prefill.h"
#include "compute/attention_fmha_mxfp4_sm120.h"
#include "compute/gemm_cutlass_mxfp4_sm120.h"
#include "compute/gemm_cutlass_grouped_3x.h"
#include "compute/gemm_grouped.h"
#include "compute/gemm_q4k.h"
#include "compute/mmq_q4k_hmma.h"
#include "compute/moe_routing.h"
#include "compute/ffn_sparsity_probe.h"
#include "compute/ffn_sparsity_mask.h"
#include "compute/encoder_forward.h"
#include "compute/hadamard.h"
#include "compute/ssm.h"
#include "compute/mla_absorb.h"
#include "compute/mla_kv_assemble.h"
#include "vision/vision_pipeline.h"
#include "vision/qwen3vl_pipeline.h"
#include "vision/vision_model.h"
#include "vision/vision_encoder.h"
#include "vision/qwen3vl_encoder.h"
#include "vision/image_processor.h"
#include "quench/quench.h"
#include <cstdlib>

namespace quench {

[[noreturn]] static void unsupported(const char* what) {
    QUENCH_LOG_FATAL("feature not available in this build: %s", what);
    std::abort();
}

// ---- vision -----------------------------------------------------------------
VisionPipeline::~VisionPipeline() = default;
bool VisionPipeline::init(const std::string&, int, Model*, cudaStream_t) {
    QUENCH_LOG_ERROR("vision support is not available in this build (--mmproj ignored)");
    return false;
}
size_t VisionPipeline::demand_bytes(const VisionConfig&, int) { return 0; }
size_t VisionPipeline::taken_bytes() const { return 0; }
bool VisionPipeline::set_image(const std::string&, cudaStream_t) { return false; }
bool VisionPipeline::set_image_from_memory(const uint8_t*, size_t, cudaStream_t) { return false; }
bool VisionPipeline::preprocess(const uint8_t*, size_t, ImageData&) const { return false; }
bool VisionPipeline::encode_to(const ImageData&, half*, cudaStream_t) { return false; }

VisionModel::~VisionModel() = default;

Qwen3VLPipeline::~Qwen3VLPipeline() = default;
bool Qwen3VLPipeline::init(VisionModel&, int) { return false; }
int Qwen3VLPipeline::patch_budget(const VisionModel&, int) { return 0; }

size_t vision_mmproj_arena_bytes(const std::string&, int) { return 0; }
size_t qwen3vl_vision_arena_bytes(VisionModel&, int) { return 0; }

bool Engine::attach_qwen_image_(Request&) { return false; }
bool Engine::encode_qwen_image_for_(Request&, cudaStream_t) { return false; }
bool Engine::encode_image_for(Request&) { return false; }
bool Engine::preprocess_image_qwen(const uint8_t*, size_t, QwenPatches&) const { return false; }
int Engine::image_tokens_of(const QwenPatches&) const { return 0; }
std::vector<int> Engine::pending_image_token_counts() const { return {}; }
void Engine::bind_mrope_single_(InferenceState&, const Request&, cudaStream_t) {}
void Engine::bind_mrope_prefill_(InferenceState&, const Request&, int, int, cudaStream_t) {}
void Engine::bind_mrope_decode_(InferenceState&, const std::vector<std::shared_ptr<Request>>&,
                                cudaStream_t) {}
void Engine::free_mrope_buffers_() {}

bool expand_image_placeholders(std::vector<int32_t>&, int32_t, const std::vector<int>&,
                               std::string&) { return false; }
int image_tokens_before(const std::vector<int32_t>&, int32_t, int) { return 0; }
size_t image_content_hash(const uint8_t*, size_t) { return 0; }
size_t combine_image_hash(size_t running, size_t) { return running; }

// ---- SafeTensors loader -----------------------------------------------------
std::unique_ptr<Model> load_safetensors(const std::string&, bool) {
    QUENCH_LOG_ERROR("SafeTensors loading is not available in this build — use a GGUF checkpoint");
    return nullptr;
}

// ---- LoRA -------------------------------------------------------------------
LoraAdapter::~LoraAdapter() = default;
bool LoraAdapter::load(const std::string&, int) {
    QUENCH_LOG_ERROR("LoRA support is not available in this build");
    return false;
}
void GraphExecutor::set_lora(const LoraAdapter*) {}
void GraphExecutor::lora_delta_(const LoraWeights&, const void*, void*, int, cudaStream_t) {
    unsupported("lora_delta_");
}

// ---- MoE --------------------------------------------------------------------
void MoEWorkspace::free(VRAMAllocator*) {}
MoeRoutingBuffers::~MoeRoutingBuffers() = default;
void MoeRoutingBuffers::allocate(int, int, int) { unsupported("MoeRoutingBuffers::allocate"); }
bool ExpertLRUCache::init(size_t, size_t, VRAMAllocator*, int, int, bool) {
    QUENCH_LOG_ERROR("expert offload is not available in this build");
    return false;
}
void ExpertLRUCache::destroy() {}
bool GraphExecutor::moe_prefill_uncapturable() const { return false; }
void gemm_grouped_cleanup() {}
void gemm_moe_batched(const void*, void*, const int32_t*, const void* const*, int, int, QType,
                      int, cudaStream_t, void**, QType, const float*, const float*) {
    unsupported("gemm_moe_batched");
}
bool cutlass_grouped_3x_nvfp4_available() { return false; }
void gemm_grouped_3x_nvfp4_prewarm() {}
void QuantPipeline::nvfp4_decode_cache_moe_experts_(const ModelConfig&, const VRAMBudget&,
                                                    size_t&, cudaStream_t, Nvfp4DecodeContext&) {
    // Dense models have no expert tensors — the full implementation is a no-op
    // for them, so the phase hook stays callable unconditionally.
}
void QuantPipeline::gpt_oss_convert_moe_experts_(const ModelConfig&, Nvfp4DecodeContext&) {
    unsupported("gpt_oss_convert_moe_experts_");
}

// NVFP4 MoE GEMV kernels (launched from retained FFN dispatch, MoE-only path).
__global__ void gemv_nvfp4_moe_decode_kernel(const uint8_t*, const uint8_t*, const float*,
                                             const int*, const half*, half*, int, int, size_t,
                                             size_t, int, int) {}
__global__ void gemv_nvfp4_moe_gate_up_fused_kernel(const uint8_t*, const uint8_t*, const float*,
                                                    const uint8_t*, const uint8_t*, const float*,
                                                    const int*, const half*, half*, half*, int,
                                                    int, size_t, size_t, int) {}

// ---- GDN/SSM helpers referenced from retained code --------------------------
void sigmoid_mul(const Tensor&, const Tensor&, Tensor&, cudaStream_t) { unsupported("sigmoid_mul"); }

// ---- FFN sparsity -----------------------------------------------------------
void probe_ffn_silu_sparsity(int, const half*, const half*, int, cudaStream_t) {}
void flush_ffn_sparsity_probe_log() {}
void build_swiglu_block_mask(const half*, const half*, uint32_t*, int, float, cudaStream_t) {
    unsupported("build_swiglu_block_mask");
}

// ---- encoder / embeddings ---------------------------------------------------
bool encoder_workspace_init(EncoderWorkspace&, const Model&, int, cudaStream_t) { return false; }
void encoder_workspace_free(EncoderWorkspace&) {}
bool encoder_embed(const Model&, EncoderWorkspace&, const int32_t*, int, float*, cudaStream_t) {
    unsupported("encoder_embed");
}

// ---- GPTQ / calibration -----------------------------------------------------
void dequant_gptq4(half*, const int*, const int*, const half*, const int*, int, int, int,
                   cudaStream_t) { unsupported("dequant_gptq4"); }
std::string write_calibration_stats(const std::string&, const CalibrationStats&) { return {}; }

// ---- K-quant dense GEMM (dp4a/HMMA prefill; GEMV family is retained) --------
void gemm_q4k_dp4a_dense(const void*, const half*, half*, void*, float*, int, int, int,
                         cudaStream_t) { unsupported("gemm_q4k_dp4a_dense"); }
void gemm_q5k_dp4a_dense(const void*, const half*, half*, void*, float*, int, int, int,
                         cudaStream_t) { unsupported("gemm_q5k_dp4a_dense"); }
bool try_q4k_hmma_dispatch(const void*, const void*, void*, int, int, int, cudaStream_t) {
    return false;
}

// ---- MLA (DeepSeek latent attention) ----------------------------------------
void mla_reorder_q(half*, int, int, int, int, cudaStream_t) { unsupported("mla_reorder_q"); }
void mla_assemble_kv(const half*, const half*, half*, half*, int, int, int, int, int,
                     cudaStream_t, int) { unsupported("mla_assemble_kv"); }
void mla_compact_attn_output(const half*, half*, int, int, int, int, cudaStream_t) {
    unsupported("mla_compact_attn_output");
}
void mla_latent_cache_write(const half*, const half*, half*, const int*, int, int, int, int, int,
                            int, cudaStream_t) { unsupported("mla_latent_cache_write"); }
void mla_absorbed_decode(const half*, const half*, const half*, half*, float*, const int*, int,
                         int, int, int, int, int, int, float, cudaStream_t) {
    unsupported("mla_absorbed_decode");
}

// ---- MXFP4 ------------------------------------------------------------------
bool attention_mxfp4_available() { return false; }
bool cutlass_sm120_mxfp4_available() { return false; }
size_t cutlass_mxfp4_sf_size(int, int) { return 0; }
size_t gemm_mxfp4_cutlass_sm120_workspace(int, int, int) { return 0; }
bool gemm_mxfp4_cutlass_sm120(const void*, const void*, const CutlassMxFP4Weight&, void*, int,
                              int, int, void*, size_t, cudaStream_t) {
    unsupported("gemm_mxfp4_cutlass_sm120");
}
void quantize_fp16_to_mxfp4_cutlass(const void*, void*, void*, int, int, cudaStream_t) {
    unsupported("quantize_fp16_to_mxfp4_cutlass");
}
void convert_nvfp4_to_mxfp4_cutlass(const NvFP4QuantResult&, CutlassMxFP4Weight&, cudaStream_t) {
    unsupported("convert_nvfp4_to_mxfp4_cutlass");
}
bool unpack_mxfp4_gguf(const void*, int64_t, int64_t, CutlassMxFP4Weight&, cudaStream_t) {
    return false;
}
void free_cutlass_mxfp4_weight(CutlassMxFP4Weight&) {}
void dequant_mxfp4_to_fp16(const void*, int64_t, int64_t, void*, cudaStream_t) {
    unsupported("dequant_mxfp4_to_fp16");
}
void mxfp4_gemv_set_l1_carveout() {}
void hadamard_transform_fp16(const half*, half*, int, int, int, cudaStream_t) {
    unsupported("hadamard_transform_fp16");
}
void gemv_mxfp4_kpar(const CutlassMxFP4Weight&, const half*, half*, int, int, cudaStream_t) {
    unsupported("gemv_mxfp4_kpar");
}
void gemv_mxfp4_kpar_fp32(const CutlassMxFP4Weight&, const half*, float*, int, int,
                          cudaStream_t) { unsupported("gemv_mxfp4_kpar_fp32"); }
void gemv_mxfp4_residual(const CutlassMxFP4Weight&, const half*, half*, const half*, int, int,
                         cudaStream_t) { unsupported("gemv_mxfp4_residual"); }
void gemv_mxfp4_swiglu_residual(const CutlassMxFP4Weight&, const half*, const half*, half*,
                                const half*, int, int, cudaStream_t) {
    unsupported("gemv_mxfp4_swiglu_residual");
}
void gemv_mxfp4_geglu_residual(const CutlassMxFP4Weight&, const half*, const half*, half*,
                               const half*, int, int, cudaStream_t) {
    unsupported("gemv_mxfp4_geglu_residual");
}
void gemv_mxfp4_gate_up_fused(const CutlassMxFP4Weight&, const CutlassMxFP4Weight&, const half*,
                              half*, half*, int, int, cudaStream_t) {
    unsupported("gemv_mxfp4_gate_up_fused");
}
void gemv_mxfp4_qkv_fused(const CutlassMxFP4Weight&, const CutlassMxFP4Weight&,
                          const CutlassMxFP4Weight&, const half*, half*, half*, half*, int, int,
                          int, int, cudaStream_t) { unsupported("gemv_mxfp4_qkv_fused"); }
bool fmha_sm120_mxfp4_prefill(const Tensor&, const Tensor&, const Tensor&, Tensor&, float, bool,
                              int, float, cudaStream_t, bool, int) {
    unsupported("fmha_sm120_mxfp4_prefill");
}
bool fmha_sm120_mxfp4_prefill_paged(const Tensor&, Tensor&, const half*, const half*,
                                    const uint8_t*, const uint8_t*, const uint8_t*,
                                    const uint8_t*, const int*, int, int, int, float, bool, int,
                                    float, cudaStream_t, int, float) {
    unsupported("fmha_sm120_mxfp4_prefill_paged");
}

// ---- non-FP16 paged-attention KV variants -----------------------------------
void paged_attention_decode_fp8(const Tensor&, const Tensor&, const Tensor&, Tensor&, const int*,
                                const int*, int, float, float, int, int, float, cudaStream_t,
                                int, int) { unsupported("paged_attention_decode_fp8"); }
void paged_attention_decode_int8(const Tensor&, const Tensor&, const Tensor&, Tensor&,
                                 const half*, const half*, const int*, const int*, int, float,
                                 int, int, float, cudaStream_t, int, int) {
    unsupported("paged_attention_decode_int8");
}
void paged_attention_decode_int4(const Tensor&, const Tensor&, const Tensor&, Tensor&,
                                 const half*, const half*, const int*, const int*, int, float,
                                 int, int, float, cudaStream_t, int, int) {
    unsupported("paged_attention_decode_int4");
}
void paged_attention_decode_nvfp4(const Tensor&, const Tensor&, const Tensor&, Tensor&,
                                  const uint8_t*, const uint8_t*, const int*, const int*, int,
                                  float, int, int, float, cudaStream_t, int, int) {
    unsupported("paged_attention_decode_nvfp4");
}
void paged_attention_decode_mxfp4_kv(const Tensor&, const Tensor&, const Tensor&, Tensor&,
                                     const uint8_t*, const uint8_t*, const int*, const int*, int,
                                     float, int, int, float, cudaStream_t, int, int) {
    unsupported("paged_attention_decode_mxfp4_kv");
}
void paged_attention_decode_nvfp4_tc(const Tensor&, const Tensor&, const Tensor&, Tensor&,
                                     const uint8_t*, const uint8_t*, const int*, const int*, int,
                                     float, int, int, float, cudaStream_t, int, int, const half*,
                                     const half*, int, int, int, const half*, const half*, int,
                                     const int*, const int*, const int*, const int*, const int*) {
    unsupported("paged_attention_decode_nvfp4_tc");
}

// ---- QuantPipeline MXFP4 phase ---------------------------------------------
void QuantPipeline::pre_dequant_phase3c_standalone_mxfp4_(const ModelConfig&, cudaStream_t) {
    // Self-guarded phase: only active for native-MXFP4 checkpoints, which this
    // build does not load.
}

// ---- misc ------------------------------------------------------------------
VisionEncoder::~VisionEncoder() = default;
Qwen3VLEncoder::~Qwen3VLEncoder() = default;
void MoeRoutingBuffers::free() {}
void moe_weighted_sum_residual(const void*, const float*, const void*, void*, int, int,
                               cudaStream_t) { unsupported("moe_weighted_sum_residual"); }
void moe_gate_topk_fused(const void*, const void*, int, int, int, MoeRoutingBuffers&,
                         MoeRoutingResult&, cudaStream_t, bool, bool, const void*) {
    unsupported("moe_gate_topk_fused");
}
void gemv_q8_0_q8_1_residual_masked(const void*, const block_q8_1*, const float*,
                                    const uint32_t*, __half*, const __half*, int, int,
                                    cudaStream_t) { unsupported("gemv_q8_0_q8_1_residual_masked"); }

}  // namespace quench

// ---- C API vision entry points ----------------------------------------------
extern "C" {
QuenchError quench_set_image(QuenchContext, const char*) { return QUENCH_ERROR_UNSUPPORTED; }
QuenchError quench_set_image_from_memory(QuenchContext, const uint8_t*, size_t) {
    return QUENCH_ERROR_UNSUPPORTED;
}
QuenchError quench_add_image(QuenchContext, const char*) { return QUENCH_ERROR_UNSUPPORTED; }
QuenchError quench_add_image_from_memory(QuenchContext, const uint8_t*, size_t) {
    return QUENCH_ERROR_UNSUPPORTED;
}
int quench_pending_image_tokens(QuenchContext) { return 0; }
}
