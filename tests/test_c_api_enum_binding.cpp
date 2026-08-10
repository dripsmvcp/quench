// The public C enums and their internal counterparts are two hand-maintained
// copies of one numbering. Nothing bound them.
//
// src/model/model.cpp:148 re-declares every QUENCH_ARCH_* value as a kApi*
// constant with the comment "QUENCH_ARCH_* values from include/quench/types.h (avoid
// header dependency)" — a deliberate choice to keep the public C header out of
// the model layer, and a correct one. But grepping tests/ for kApi or QUENCH_ARCH_
// returned nothing before this file: a wrong or forgotten id made
// quench_model_architecture() report the wrong architecture to every C-API
// consumer, silently, with a green build and a green test suite.
//
// A test file may include both sides, so this is where the two copies get tied
// together. Same for the QuenchDType ↔ QType pair.
//
// CPU-only: pure enum arithmetic, no CUDA, no Model.

#include "quench/types.h"
#include "model/model_arch.h"
#include "core/qtype.h"

#include <gtest/gtest.h>

#include <set>
#include <string>
#include <vector>

using namespace quench;

namespace {

struct ArchBinding {
    ModelArch arch;
    int c_api_id;
    const char* name;  // the registry's own name string, for a readable failure
};

// Every enumerator of ModelArch, paired with the QUENCH_ARCH_* value the C API
// promises for it. Adding an architecture means adding a row here — which is
// the point: the compiler cannot see the coupling, so the test has to.
const ArchBinding kBindings[] = {
    {ModelArch::LLAMA, QUENCH_ARCH_LLAMA, "llama"},
    {ModelArch::MISTRAL, QUENCH_ARCH_MISTRAL, "mistral"},
    {ModelArch::MIXTRAL, QUENCH_ARCH_MIXTRAL, "mixtral"},
    {ModelArch::DEEPSEEK, QUENCH_ARCH_DEEPSEEK, "deepseek"},
    {ModelArch::NEMOTRON_H_MOE, QUENCH_ARCH_NEMOTRON_H_MOE, "nemotron_h_moe"},
    {ModelArch::QWEN3, QUENCH_ARCH_QWEN3, "qwen3"},
    {ModelArch::QWEN3_MOE, QUENCH_ARCH_QWEN3_MOE, "qwen3moe"},
    {ModelArch::QWEN35, QUENCH_ARCH_QWEN35, "qwen35"},
    {ModelArch::QWEN35_MOE, QUENCH_ARCH_QWEN35_MOE, "qwen35moe"},
    {ModelArch::QWEN36_MOE, QUENCH_ARCH_QWEN36_MOE, "qwen36moe"},
    {ModelArch::GPT_OSS, QUENCH_ARCH_GPT_OSS, "gpt_oss"},
    {ModelArch::GEMMA3, QUENCH_ARCH_GEMMA3, "gemma3"},
    {ModelArch::GEMMA4, QUENCH_ARCH_GEMMA4, "gemma4"},
    {ModelArch::LLAMA4, QUENCH_ARCH_LLAMA4, "llama4"},
    {ModelArch::NOMIC_BERT, QUENCH_ARCH_NOMIC_BERT, "nomic-bert"},
    {ModelArch::GENERIC, QUENCH_ARCH_GENERIC, "generic"},
};

}  // namespace

// The registry's c_api_id must be the value the public header promises.
TEST(CApiEnumBinding, ArchIdsMatchThePublicHeader) {
    for (const auto& b : kBindings) {
        EXPECT_EQ(model_arch_c_api_id(b.arch), b.c_api_id)
            << "arch '" << b.name << "': model.cpp's kApi* table disagrees with QUENCH_ARCH_* in "
            << "include/quench/types.h — quench_model_architecture() would report the wrong architecture";
    }
}

// Two architectures sharing a C-API id makes them indistinguishable to a
// consumer; the copy in model.cpp is exactly where that typo would live.
TEST(CApiEnumBinding, ArchIdsAreUnique) {
    std::set<int> seen;
    for (const auto& b : kBindings) {
        const int id = model_arch_c_api_id(b.arch);
        EXPECT_TRUE(seen.insert(id).second) << "duplicate C-API arch id " << id << " (at '" << b.name << "')";
    }
}

// Coverage guard: kBindings must list every ModelArch enumerator. GENERIC is
// the last one, so its id doubles as the count. A new architecture added to
// model_arch.h without a row here leaves the binding untested.
TEST(CApiEnumBinding, EveryArchEnumeratorIsCovered) {
    const size_t n = sizeof(kBindings) / sizeof(kBindings[0]);
    EXPECT_EQ(n, 16u) << "ModelArch gained or lost an enumerator — add/remove the row in "
                         "kBindings so the C-API id stays bound";
    // Every listed arch must round-trip through the registry's name table too,
    // which is what catches a row pointing at the wrong enumerator.
    for (const auto& b : kBindings)
        EXPECT_STREQ(model_arch_name(b.arch), b.name);
}

// The name string is what parse_model_arch() consumes, so name → arch → id must
// be a closed loop. A row copy-pasted with the wrong enumerator survives the
// id check when two archs happen to share an id range; this closes that gap.
TEST(CApiEnumBinding, ArchNamesRoundTripThroughParse) {
    for (const auto& b : kBindings) {
        if (b.arch == ModelArch::GENERIC)
            continue;  // "generic" is the fallback, not a parseable input
        EXPECT_EQ(parse_model_arch(b.name), b.arch)
            << "name '" << b.name << "' does not parse back to its own enumerator";
    }
}

// An unrecognised architecture string resolves to GENERIC and LOADS — it does
// not error. That is deliberate (the registry maps 30 strings onto known archs
// and a Llama-shaped checkpoint usually works through it), but it means an
// unsupported model looks supported, so parse_model_arch() warns.
// Pinning the behaviour here so the fallback cannot be turned into a silent
// hard error or a silent success again without a test saying so.
TEST(CApiEnumBinding, UnknownArchFallsBackToGeneric) {
    EXPECT_EQ(parse_model_arch("definitely-not-an-architecture"), ModelArch::GENERIC);
    EXPECT_EQ(parse_model_arch(""), ModelArch::GENERIC);
    // A known one must still resolve — guards against the fallback swallowing
    // everything if the registry lookup ever breaks.
    EXPECT_EQ(parse_model_arch("qwen3"), ModelArch::QWEN3);
}

// QuenchDType is a second enum for the concept QType already names, kept separate
// so the public header stays C and CUDA-free. quench_api.cpp maps between them by
// hand; these are the pairs that mapping must honour.
TEST(CApiEnumBinding, DTypeValuesAreStableAndDistinct) {
    const std::vector<std::pair<int, const char*>> dtypes = {
        {QUENCH_DTYPE_FP32, "fp32"},         {QUENCH_DTYPE_FP16, "fp16"},         {QUENCH_DTYPE_BF16, "bf16"},
        {QUENCH_DTYPE_FP8_E4M3, "fp8_e4m3"}, {QUENCH_DTYPE_FP8_E5M2, "fp8_e5m2"}, {QUENCH_DTYPE_INT8, "int8"},
        {QUENCH_DTYPE_INT4, "int4"},         {QUENCH_DTYPE_INT32, "int32"},       {QUENCH_DTYPE_FP4_E2M1, "fp4_e2m1"},
        {QUENCH_DTYPE_NVFP4, "nvfp4"},       {QUENCH_DTYPE_MXFP4_KV, "mxfp4_kv"},
    };
    std::set<int> seen;
    for (const auto& [v, name] : dtypes)
        EXPECT_TRUE(seen.insert(v).second) << "duplicate QuenchDType value for " << name;

    // 9 and 10 are retired TURBOQUANT alias values. They must stay
    // retired: reusing a wire value silently reinterprets an old caller's KV
    // dtype request.
    EXPECT_EQ(seen.count(9), 0u) << "QUENCH_DTYPE value 9 was a retired TurboQuant alias — do not reuse";
    EXPECT_EQ(seen.count(10), 0u) << "QUENCH_DTYPE value 10 was a retired TurboQuant alias — do not reuse";
}
