#include "exec/gemm_kernel_registry.h"

#include "compute/gemm_cutlass_mxfp4_sm120.h"  // CutlassMxFP4Weight, gemm_mxfp4_cutlass_sm120 (dual-cache path)
#include "compute/gemm_cutlass_sm120.h"  // CutlassNvFP4Weight, gemm_nvfp4_cutlass_sm120, quantize_fp16_to_nvfp4_cutlass
#include "core/logging.h"
#include "core/tensor.h"

namespace quench {

// ---------------------------------------------------------------------------
// CUTLASS_NVFP4 tier (dual-cache integration).
//
// Adapter that wraps the CUTLASS sm_120 block-scaled NVFP4
// GEMM: quantize the FP16 activation into the
// pre-allocated CUTLASS NVFP4 scratch (`cutlass_act_data` + `cutlass_act_sf`),
// then run `gemm_nvfp4_cutlass_sm120` against the pre-converted CUTLASS
// weight payload. This is the *preferred* NVFP4 GEMM path — used when the
// loader has populated `cutlass_nvfp4_cache` AND the workspace pointers are
// present; otherwise the dequant fallback (gemm_nvfp4) takes over.
//
// Strategy: tier=CUTLASS_NVFP4, weight_qtype=F16 (engine observes FP16 source
// qtype for an NVFP4-cached weight; the CUTLASS conversion happened at load
// time and lives in the CutlassNvFP4Weight), m_is_one=false. The (M==1)
// decode case never reaches the CUTLASS path — it goes to the
// faster gemv_nvfp4_kpar — so we only register the prefill slot.
//
// Dual-cache MXFP4 branch: when `--mxfp4-prefill` is enabled
// the loader builds `cutlass_mxfp4` alongside `cutlass_nvfp4` (executor_pre_
// dequant.cu iterates every NVFP4 entry with K%32==0 and converts
// the scales to UE8M0 MXFP4). When BOTH caches hit the same `weight.data`,
// the handler tries MXFP4 CUTLASS first and falls back to NVFP4 CUTLASS
// on failure, using
// the optional `mxfp4_payload` + `mxfp4_act_sf` + `mxfp4_workspace` fields
// forwarded from the dispatch site. nullptr `mxfp4_payload` = no dual-cache
// hit, skip straight to the NVFP4 path.
//
// Preconditions checked here (and at the dispatch site for fast fail):
// - input/output/weight_payload non-null
// - input qtype == F16
// - cutlass_act_data + cutlass_act_sf non-null (mandatory; the FP16
//   activation has to land somewhere). cutlass_workspace is *optional* —
//   gemm_nvfp4_cutlass_sm120 has its own static-fallback workspace alloc.
// If the activation scratch is missing we return PreconditionFail so the
// caller can fall back to the dequant path. Same drop-through
// applies if gemm_nvfp4_cutlass_sm120 itself returns false (CUTLASS
// can_implement rejected the dims).
// ---------------------------------------------------------------------------
static GemmDispatchResult cutlass_nvfp4_gemm_kernel(const GemmKernelArgs& args) {
    QUENCH_CHECK(args.input != nullptr, "cutlass_nvfp4_gemm_kernel: input is null");
    QUENCH_CHECK(args.output != nullptr, "cutlass_nvfp4_gemm_kernel: output is null");
    QUENCH_CHECK(args.weight_payload != nullptr, "cutlass_nvfp4_gemm_kernel: weight_payload is null");
    QUENCH_CHECK(args.input->qtype == QType::F16, "cutlass_nvfp4_gemm_kernel: input qtype must be F16");

    // Workspace precondition. Loud-but-soft: return PreconditionFail so the
    // dispatch site can fall back to the dequant kernel (or legacy).
    // Only the activation scratch (act_data + act_sf) is mandatory — the GEMM
    // workspace may legitimately be null, because allocate_auxiliary_buffers()
    // sizes it with gemm_nvfp4_cutlass_sm120_workspace() at the MAX shape and
    // allocates nothing when that is 0. Since A7 step 8 the GEMM refuses rather
    // than growing a static workspace if it ever needs more, and a refusal
    // arrives here as PreconditionFail → the dequant fallback.
    if (args.cutlass_act_data == nullptr || args.cutlass_act_sf == nullptr) {
        return GemmDispatchResult::PreconditionFail;
    }

    const CutlassNvFP4Weight& payload =
        *static_cast<const CutlassNvFP4Weight*>(args.weight_payload);

    const int M = static_cast<int>(args.input->shape[0]);
    const int K = static_cast<int>(args.input->shape[1]);
    const int N = static_cast<int>(payload.N);

    // Dual-cache MXFP4 branch. Fires only when (a) the dispatch site found a matching
    // entry in `cutlass_mxfp4` for the same weight.data and (b) the MXFP4
    // activation scratch is allocated and (c) K is a multiple of the UE8M0
    // SfAtom group (32). We re-use the NVFP4 activation data buffer because
    // the packed E2M1 layout is the same — only the scale factor layout
    // differs (UE8M0 per 32 vs UE4M3 per 16 for NVFP4).
    if (args.mxfp4_payload != nullptr && args.mxfp4_act_sf != nullptr && K % 32 == 0) {
        const CutlassMxFP4Weight& mx_payload =
            *static_cast<const CutlassMxFP4Weight*>(args.mxfp4_payload);
        quantize_fp16_to_mxfp4_cutlass(args.input->data, args.cutlass_act_data, args.mxfp4_act_sf, M, K,
                                       args.stream);
        bool mx_ok = gemm_mxfp4_cutlass_sm120(args.cutlass_act_data, args.mxfp4_act_sf, mx_payload,
                                              args.output->data, M, N, K, args.mxfp4_workspace,
                                              args.mxfp4_workspace_size, args.stream);
        if (mx_ok)
            return GemmDispatchResult::Ok;
        // MXFP4 CUTLASS rejected the dims — fall through to NVFP4 CUTLASS
        // below. Mirrors the legacy `if (ok) return;` drop-through. The
        // activation buffer is reusable because quantize_fp16_to_nvfp4_cutlass
        // overwrites it (separate scale buffer too — cutlass_act_sf vs
        // mxfp4_act_sf).
    }

    // Mirror executor_kernels.cu:2147 + 2179-2181 verbatim — same activation
    // quantization step, same CUTLASS GEMM call, same arg order.
    // act_prequantized: the scratch already holds quantize(input) from a
    // prior dispatch on the same input (QKV / gate-up dedupe) — skip the
    // re-quantize; the buffer contents are bit-identical by construction.
    // (mxfp4_payload != nullptr means the dual-cache branch above may have
    // clobbered the scratch with MXFP4 scale layout — never skip then.)
    if (!args.act_prequantized || args.mxfp4_payload != nullptr)
        quantize_fp16_to_nvfp4_cutlass(args.input->data, args.cutlass_act_data, args.cutlass_act_sf, M, K,
                                       args.stream);
    bool ok = gemm_nvfp4_cutlass_sm120(args.cutlass_act_data, args.cutlass_act_sf, payload,
                                       args.output->data, M, N, K, args.cutlass_workspace,
                                       args.cutlass_workspace_size, args.stream);
    if (!ok) {
        // A failed CUTLASS run drops through to the dequant fallback.
        // Signal PreconditionFail so the caller invokes it.
        // gemm_nvfp4_cutlass_sm120 already logs QUENCH_LOG_WARN/ERROR on the
        // can_implement / initialize / run failures internally — no need to
        // double-log here.
        return GemmDispatchResult::PreconditionFail;
    }
    return GemmDispatchResult::Ok;
}

// Static registration: only the (M>1) prefill slot — the
// (M==1) decode case always picks the gemv_nvfp4_kpar fast path.
namespace {
struct CutlassNvFP4Registration {
    CutlassNvFP4Registration() {
        GemmKernelRegistry::instance().register_kernel(
            GemmStrategy{StorageTier::CUTLASS_NVFP4, QType::F16, /*m_is_one=*/false},
            &cutlass_nvfp4_gemm_kernel);
    }
};
static CutlassNvFP4Registration s_cutlass_nvfp4_registration;
}  // namespace

}  // namespace quench
