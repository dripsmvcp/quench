#pragma once

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    QUENCH_DTYPE_FP32 = 0,
    QUENCH_DTYPE_FP16 = 1,
    QUENCH_DTYPE_BF16 = 2,
    QUENCH_DTYPE_FP8_E4M3 = 3,
    QUENCH_DTYPE_FP8_E5M2 = 4,
    QUENCH_DTYPE_INT8 = 5,
    QUENCH_DTYPE_INT4 = 6,
    QUENCH_DTYPE_INT32 = 7,
    QUENCH_DTYPE_FP4_E2M1 = 8,
    // Values 9 and 10 are intentionally left unassigned.
    QUENCH_DTYPE_NVFP4 = 11,            // NVFP4 KV cache: packed FP4 + UE4M3 per-group_of_16 scales
    QUENCH_DTYPE_MXFP4_KV = 12,        // MXFP4-KV cache: packed FP4 + UE8M0 per-group_of_16 scales (Path A retirement target)
} QuenchDType;

typedef enum {
    QUENCH_ARCH_LLAMA = 0,
    QUENCH_ARCH_MISTRAL = 1,
    QUENCH_ARCH_MIXTRAL = 2,
    QUENCH_ARCH_DEEPSEEK = 3,
    QUENCH_ARCH_NEMOTRON_H_MOE = 4,
    QUENCH_ARCH_QWEN3 = 5,
    QUENCH_ARCH_QWEN3_MOE = 6,
    QUENCH_ARCH_GEMMA3 = 7,
    QUENCH_ARCH_LLAMA4 = 8,
    QUENCH_ARCH_GENERIC = 9,
    QUENCH_ARCH_QWEN35 = 10,
    QUENCH_ARCH_QWEN35_MOE = 11,
    QUENCH_ARCH_GEMMA4 = 12,
    QUENCH_ARCH_QWEN36_MOE = 13,
    QUENCH_ARCH_GPT_OSS = 14,
    QUENCH_ARCH_NOMIC_BERT = 15,
} QuenchModelArch;

typedef enum {
    QUENCH_QUANT_NONE = 0,
    QUENCH_QUANT_Q4_0 = 1,
    QUENCH_QUANT_Q4_K_M = 2,
    QUENCH_QUANT_Q8_0 = 3,
    QUENCH_QUANT_FP8 = 4,
    QUENCH_QUANT_FP8_E4M3 = 5,
    QUENCH_QUANT_NVFP4 = 6,
} QuenchQuantType;

typedef enum {
    QUENCH_FORMAT_GGUF = 0,
    QUENCH_FORMAT_SAFETENSORS = 1,
} QuenchModelFormat;

#ifdef __cplusplus
}
#endif
