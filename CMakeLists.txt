cmake_minimum_required(VERSION 3.25 FATAL_ERROR)
project(quench LANGUAGES C CXX CUDA VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 23)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CMake's NVIDIA-CUDA module (through 4.3.x) tops out at c++20 — it defines no
# CUDA23 dialect flag, so `CMAKE_CUDA_STANDARD 23` fails with "CMake does not know
# the flags to enable it". nvcc 13.3 does accept `-std=c++23`, so teach CMake the
# flag directly. Drop this once the module ships a native CUDA23 mapping.
if(NOT DEFINED CMAKE_CUDA23_STANDARD_COMPILE_OPTION)
  set(CMAKE_CUDA23_STANDARD_COMPILE_OPTION  "-std=c++23")
  set(CMAKE_CUDA23_EXTENSION_COMPILE_OPTION "-std=c++23")
endif()
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Custom cmake modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(CompilerFlags)
include(quench-deps)  # third-party dependency pins (single source; mirrored by Dockerfile)

# First-party warning flags as an INTERFACE target. Linked PRIVATE by every
# quench target (lib, tools, server, tests) so warnings apply to OUR host C++ TUs
# only — never nvcc, never FetchContent deps. (Was a global CMAKE_CXX_FLAGS.)
add_library(quench_warnings INTERFACE)
target_compile_options(quench_warnings INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>)

# --- CUDA Toolkit ---
find_package(CUDAToolkit 13.2 REQUIRED)
message(STATUS "CUDA ${CUDAToolkit_VERSION} detected")

# --- CUDA Architecture ---
# Primary target: sm_120a (RTX 5090, RTX PRO 6000 — GB202).
# 'a' suffix = architecture-specific (FP8 MMA .kind::f8f6f4, NVFP4 block-scaled
# MMA, TMA warp-specialized grouped GEMM tactics).
#
# Fallback PTX: compute_120f (family-portable). JIT-compiles on load for GB203
# (RTX 5080, 5070 Ti). Loses TMA-WS grouped GEMM tactics → CUTLASS NVFP4
# fast-path on Mamba2 shapes is degraded. Set QUENCH_DISABLE_120F_FALLBACK=ON to
# strip the PTX entry and shrink the fatbin.
#
# CMake < 3.31 doesn't parse the 'a'/'f' suffix in CMAKE_CUDA_ARCHITECTURES,
# so we pass gencode directly.
option(QUENCH_DISABLE_120F_FALLBACK "Skip compute_120f PTX fallback (RTX 5090-only fatbin)" OFF)
set(QUENCH_SM120_FLAGS "--generate-code=arch=compute_120a,code=sm_120a")
if(NOT QUENCH_DISABLE_120F_FALLBACK)
    set(QUENCH_SM120_FLAGS "${QUENCH_SM120_FLAGS} --generate-code=arch=compute_120f,code=compute_120f")
    message(STATUS "CUDA architectures: sm_120a (SASS, RTX 5090 / RTX PRO 6000) + compute_120f (PTX, RTX 5080 / 5070 Ti)")
else()
    message(STATUS "CUDA architectures: sm_120a (RTX 5090 only — fallback disabled)")
endif()
set(CMAKE_CUDA_ARCHITECTURES OFF)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${QUENCH_SM120_FLAGS}")

# --- Options ---
option(QUENCH_BUILD_TESTS "Build tests" ON)
option(QUENCH_BUILD_TOOLS "Build CLI and benchmark tools" ON)
option(QUENCH_BUILD_BENCH "Build benchmarks" ON)
option(QUENCH_BUILD_SERVER "Build quench-server (OpenAI-compatible HTTP server)" ON)
# Link-time interposition on the CUDA allocation symbols, so acceptance
# criterion 3 can be measured against EVERY device allocation rather than only
# the ones routed through Backend (docs/MEMORY_ARCHITECTURE.md A6).
# Off by default: --wrap belongs on a measurement build, not a shipping one.
option(QUENCH_ALLOC_INTERPOSE "Wrap cudaMalloc/cudaMallocAsync to attribute steady-state allocations" OFF)
set(QUENCH_WRAP_LINK_FLAGS "")
if(QUENCH_ALLOC_INTERPOSE)
    set(QUENCH_WRAP_LINK_FLAGS
        "-Wl,--wrap=cudaMalloc,--wrap=cudaFree,--wrap=cudaMallocAsync,--wrap=cudaFreeAsync,--wrap=cudaMallocHost,--wrap=cudaHostAlloc")
    message(STATUS "Allocation interposition ENABLED (measurement build)")
endif()
option(QUENCH_SANITIZERS "Enable AddressSanitizer + UndefinedBehaviorSanitizer" OFF)

if(QUENCH_SANITIZERS)
    set(SANITIZER_FLAGS "-fsanitize=address,undefined -fno-omit-frame-pointer -g")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SANITIZER_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${SANITIZER_FLAGS}")
    # Note: CUDA code is NOT sanitized (nvcc doesn't support ASAN).
    # ASAN catches host-side memory errors in the C++ runtime/API layer.
    message(STATUS "Sanitizers enabled: ASAN + UBSAN (host code only)")
endif()

# --- Third-party dependencies ---
include(FetchContent)

if(QUENCH_BUILD_TESTS)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG ${QUENCH_DEP_GOOGLETEST_TAG}
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()

option(QUENCH_USE_CUTLASS "Use CUTLASS for MoE GEMM (requires sm_90+)" ON)
if(QUENCH_USE_CUTLASS)
    FetchContent_Declare(cutlass
        GIT_REPOSITORY https://github.com/NVIDIA/cutlass.git
        GIT_TAG ${QUENCH_DEP_CUTLASS_TAG}
        GIT_SHALLOW TRUE
        EXCLUDE_FROM_ALL
    )
    # CUTLASS is header-only for our use — disable all build targets
    set(CUTLASS_ENABLE_HEADERS_ONLY ON CACHE BOOL "" FORCE)
    set(CUTLASS_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
    set(CUTLASS_ENABLE_TOOLS OFF CACHE BOOL "" FORCE)
    set(CUTLASS_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(cutlass)
endif()

# --- Core library sources ---
set(QUENCH_CORE_SOURCES
    src/core/tensor.cpp
    src/core/qtype.cpp
    src/core/buffer.cpp
    src/core/logging.cpp
    src/core/cuda_static_reset.cpp
)

set(QUENCH_MEMORY_SOURCES
    src/memory/backend.cpp
    $<$<BOOL:${QUENCH_ALLOC_INTERPOSE}>:src/memory/alloc_interpose.cpp>
    src/memory/arena.cpp
    src/memory/engine_arena.cpp
    src/memory/graph_slots.cpp
    src/memory/host_pinned.cpp
    src/memory/library_reserve_cache.cpp
    src/memory/block_pool.cpp
    src/memory/scratch_stack.cpp
    src/memory/plan.cpp
    src/memory/vram_allocator.cu
    src/memory/mem_account.cu
    src/memory/kv_cache.cu
    src/memory/kv_cache_manager.cpp
    src/memory/ssm_state.cu
    src/memory/recurrent_snapshot_store.cpp
    src/memory/layer_offload.cu
    src/memory/weight_snapshot.cpp
    src/memory/weight_cache_file.cpp
    src/memory/vram_query.cpp
)

set(QUENCH_MODEL_SOURCES
    src/model/model.cpp
    src/model/model_profile.cpp
    src/model/gguf_loader.cpp
    src/model/gguf_parse.cpp
    src/model/gguf_tensor_assign.cpp
    src/model/json_util.cpp
    src/model/hf_hub.cpp
    src/model/weight_map.cpp
    src/model/weight_upload.cu
    src/model/tokenizer.cpp
    src/model/tensor_kind_name.cpp
    src/model/tensor_kind_table.cu
    src/model/tensor_kind_matcher.cpp
    src/model/chat_template.cpp
    src/model/chat_template_families.cpp
    src/model/jinja.cpp
)

set(QUENCH_QUANT_SOURCES
    src/quant/dequant_fp16.cu
    src/quant/dequant_gpu.cu
    src/quant/fp8_utils.cu
    src/quant/quant_gemm.cu
    src/quant/fp8_quant.cu
    src/quant/nvfp4_quant.cu
    src/quant/nvfp4_gemm.cu
    src/quant/nvfp4_gemv_dense.cu
    src/quant/nvfp4_gemv_fused.cu
)

set(QUENCH_COMPUTE_SOURCES
    src/compute/dispatch_paths.cpp
    src/compute/gemm.cu
    src/compute/gemm_dp4a.cu
    src/compute/ggml_mmvq.cu
    src/compute/mmq_q8_imma_q4k.cu
    src/compute/mmq_q8_imma_q51.cu
    src/compute/mmq_q8_imma_q6k.cu
    src/compute/rowwise_topm.cu
    src/compute/gemm_capture_fp16_sm120.cu
    src/compute/weight_dispatch.cu
    src/compute/kv_gather.cu
    src/compute/attention_paged.cu
    src/compute/attention_blackwell.cu
    src/compute/attention_cublas.cu
    src/compute/attention_dispatch.cu
    src/compute/rope.cu
    src/compute/layernorm.cu
    src/compute/activation.cu
    src/compute/embedding.cu
    src/compute/sampling.cu
    src/compute/sampling_topk_topp.cu
    src/compute/sampling_penalties.cu
    src/compute/sampling_filters.cu
    src/compute/gemm_gemv_dtype.cu
    src/compute/gemm_batched.cu
    src/compute/softmax.cu
    src/compute/regex_constrain.cu
    src/compute/grammar_constrain.cu
    src/compute/gbnf_grammar.cpp
    src/compute/gbnf_parser.cpp
    src/compute/json_constrain.cu
    src/compute/json_schema.cpp
    src/compute/schema_constrain.cu
    src/compute/reduce.cu
    src/compute/mmq_q8_imma.cu
    src/compute/mmq_q8_imma_scratch.cu
)

if(QUENCH_USE_CUTLASS)
    list(APPEND QUENCH_COMPUTE_SOURCES src/compute/gemm_cutlass_sm120.cu)
    list(APPEND QUENCH_COMPUTE_SOURCES src/compute/attention_fmha_sm120.cu)
    list(APPEND QUENCH_COMPUTE_SOURCES src/compute/nvfp4_quant_hw.cu)
    # Bench/probe TUs were relocated to tests/bench/ (P3 §10 #12, P5 §3).
    # Validate/ref TUs that still serve internal call sites stay in src/compute/.
    if(QUENCH_BUILD_TESTS OR QUENCH_BUILD_BENCH)
        list(APPEND QUENCH_COMPUTE_SOURCES src/compute/nvfp4_quant_ref.cu)
    endif()
endif()

# Note: TurboQuant paged attention previously required a fast-math override
# (ptxas C7600 on sm_120a). This is resolved by using sm_120f instead.

set(QUENCH_RUNTIME_SOURCES
    src/runtime/config.cpp
    src/runtime/process_diag.cpp
    src/runtime/graph_eligibility.cpp
    src/runtime/green_ctx.cu
    src/runtime/scheduler.cpp
    src/runtime/request.cpp
    src/runtime/batch.cpp
    src/runtime/engine.cpp
    src/runtime/engine_init_resolver.cpp
    src/runtime/engine_weight_upload.cpp
    src/runtime/engine_kv_cache_init.cpp
    src/runtime/engine_workspace_warmup.cpp
    src/runtime/engine_graph_decode.cpp
    src/runtime/engine_scheduler.cpp
    src/runtime/engine_spec_ngram.cpp
    src/runtime/engine_spec_capture.cpp
    src/runtime/engine_spec_mtp.cpp
    src/runtime/ngram_draft.cpp
    src/runtime/suffix_draft.cpp
    src/runtime/token_recycle_draft.cpp
    src/compute/mtp_forward.cu
    src/runtime/engine_sampling_stop.cpp
    src/runtime/constraint_manager.cpp
    src/runtime/vram_budget.cpp
    src/runtime/plan_shadow.cpp
    src/runtime/cuda_graph.cu
    src/runtime/pdl.cu
    src/runtime/presets.cpp
    src/runtime/storage_planner.cpp
)

set(QUENCH_EXEC_SOURCES
    src/exec/executor.cu
    src/exec/executor_kernels.cu
    src/exec/activation_calibrator.cu
    src/compute/unsupported.cu
    src/exec/executor_kv_write_kernels.cu
    src/exec/executor_elementwise.cu
    src/exec/decode_pipeline_advance.cu
    src/exec/pre_dequant_phase3_fp8.cu
    src/exec/pre_dequant_phase3_cutlass.cu
    src/exec/executor_gemm_dispatch.cu
    src/exec/gemm_kernel_registry.cu
    src/exec/gemm_kernel_fp8.cu
    src/exec/gemm_kernel_nvfp4_gemv.cu
    src/exec/gemm_kernel_nvfp4_gemm.cu
    src/exec/gemm_kernel_cutlass_nvfp4.cu
    src/exec/gemm_kernel_gguf.cu
    src/exec/gemm_scratch.cu
    src/exec/workspace_sizes.cpp
    src/exec/executor_workspace.cu
    src/exec/executor_workspace_buffers.cu
    src/exec/executor_kv_write.cu
    src/exec/executor_pre_dequant.cu
    src/exec/pre_dequant_phase0_nvfp4_loader.cu
    src/exec/pre_dequant_phase1_fp16_cache.cu
    src/exec/pre_dequant_phase2_fp8_cache.cu
    src/exec/pre_dequant_phase3_nvfp4_decode.cu
    src/exec/pre_dequant_phase4_tensor_registry.cu
    src/exec/executor_workspace_config.cu
    src/exec/executor_attention.cu
    src/exec/executor_forward.cu
    src/exec/executor_perplexity.cu
    src/exec/executor_ffn.cu
    src/exec/weight_handle.cu
    src/exec/quant_scratch.cu
)

set(QUENCH_VISION_SOURCES
)

set(QUENCH_API_SOURCES
    src/api/quench_api.cpp
    src/api/quench_api_suspend.cpp
)

# --- Main library target ---
add_library(quench STATIC
    ${QUENCH_CORE_SOURCES}
    ${QUENCH_MEMORY_SOURCES}
    ${QUENCH_MODEL_SOURCES}
    ${QUENCH_QUANT_SOURCES}
    ${QUENCH_COMPUTE_SOURCES}
    ${QUENCH_RUNTIME_SOURCES}
    ${QUENCH_EXEC_SOURCES}
    ${QUENCH_VISION_SOURCES}
    ${QUENCH_API_SOURCES}
)

target_include_directories(quench
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/tests  # for tests/bench/ headers (test/bench TUs only)
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/stb
)

# Single-source the runtime version string from the project version so
# quench_version() / `quench-cli --version` can never drift from CMakeLists.
target_compile_definitions(quench PRIVATE QUENCH_VERSION_STRING="${PROJECT_VERSION}")

target_link_libraries(quench
    PUBLIC
        CUDA::cudart
        CUDA::cuda_driver
        CUDA::cublas
        CUDA::cublasLt
    PRIVATE
        pthread
        quench_warnings
)

if(QUENCH_USE_CUTLASS)
    target_include_directories(quench PRIVATE
        ${cutlass_SOURCE_DIR}/include
        ${cutlass_SOURCE_DIR}/tools/util/include
    )
    target_compile_definitions(quench PRIVATE QUENCH_USE_CUTLASS=1)
endif()

# Enable separable compilation for device-side launches (PDL)
set_target_properties(quench PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
)

# --- Tools ---
if(QUENCH_BUILD_TOOLS)
    # quench-cli
    add_executable(quench-cli
        tools/quench-cli/main.cpp
    tools/common/args_common.cpp
    tools/quench-cli/args.cpp
    )
    target_link_libraries(quench-cli PRIVATE quench quench_warnings)
    if(QUENCH_ALLOC_INTERPOSE)
        target_link_options(quench-cli PRIVATE ${QUENCH_WRAP_LINK_FLAGS})
    endif()
    target_include_directories(quench-cli PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
                                               ${CMAKE_CURRENT_SOURCE_DIR}/tools)

endif()

# --- Server ---
if(QUENCH_BUILD_SERVER)
    FetchContent_Declare(httplib
        GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
        GIT_TAG ${QUENCH_DEP_HTTPLIB_TAG}
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(httplib)

    FetchContent_Declare(nlohmann_json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG ${QUENCH_DEP_NLOHMANN_JSON_TAG}
        GIT_SHALLOW TRUE
    )
    set(JSON_BuildTests OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(nlohmann_json)

    # Web UI: generated header, regenerated whenever the page changes.
    set(QUENCH_WEBUI_HEADER ${CMAKE_CURRENT_BINARY_DIR}/generated/webui_asset.h)
    add_custom_command(
        OUTPUT ${QUENCH_WEBUI_HEADER}
        COMMAND ${CMAKE_COMMAND} -DIN=${QUENCH_WEBUI_PAGE} -DOUT=${QUENCH_WEBUI_HEADER}
                -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_webui.cmake
        DEPENDS ${QUENCH_WEBUI_PAGE} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_webui.cmake
        COMMENT "Embedding web UI into quench-server"
        VERBATIM)

    add_executable(quench-server
        ${QUENCH_WEBUI_HEADER}
        tools/quench-server/main.cpp
    tools/common/args_common.cpp
    tools/quench-server/args.cpp
        tools/quench-server/utils.cpp
        tools/quench-server/tool_call.cpp
        tools/quench-server/tool_call_gemma.cpp
        tools/quench-server/anthropic.cpp
        tools/quench-server/responses.cpp
        tools/quench-server/handlers.cpp
        tools/quench-server/constraint_validation.cpp
        tools/quench-server/handlers_chat.cpp
        tools/quench-server/handlers_chat_core.cpp
        tools/quench-server/handlers_chat_params.cpp
        tools/quench-server/handlers_chat_stream.cpp
        tools/quench-server/stream_driver.cpp
        tools/quench-server/handlers_messages.cpp
        tools/quench-server/handlers_responses.cpp
        tools/quench-server/handlers_misc.cpp
        tools/quench-server/metrics_memory.cpp
        tools/quench-server/handlers_admin.cpp
        tools/quench-server/batching_engine.cpp
    )
    target_link_libraries(quench-server PRIVATE quench httplib::httplib nlohmann_json::nlohmann_json quench_warnings)
    if(QUENCH_ALLOC_INTERPOSE)
        target_link_options(quench-server PRIVATE ${QUENCH_WRAP_LINK_FLAGS})
    endif()
    target_include_directories(quench-server PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
                                                 ${CMAKE_CURRENT_SOURCE_DIR}/tools
                                                  ${CMAKE_CURRENT_BINARY_DIR}/generated)
    # Optional gcov instrumentation of the server TUs only (the CUDA quench lib is
    # left uninstrumented — nvcc can't take --coverage). Measures real line
    # coverage of tools/quench-server/ over an E2E run. Off by default.
    option(QUENCH_COVERAGE "Instrument quench-server with gcov" OFF)
    if(QUENCH_COVERAGE)
        target_compile_options(quench-server PRIVATE --coverage -O0 -g)
        target_link_options(quench-server PRIVATE --coverage)
    endif()
endif()

# --- Tests ---
# Tests are split into per-module binaries so a kernel change only relinks the
# affected binary instead of one monolithic 60-file quench-tests target.
# A wrapper shell script `quench-tests` is generated for backwards-compat with
# Makefile, CI, and smoke_test.sh (forwards args to every binary in sequence;
# gtest_filter mismatches return success, so only matching tests run).
if(QUENCH_BUILD_TESTS)
    enable_testing()
    include(GoogleTest)

    function(quench_add_test_module target)
        cmake_parse_arguments(ARG "WITH_STUB" "" "SOURCES" ${ARGN})
        set(_sources ${ARG_SOURCES})
        if(ARG_WITH_STUB)
            list(APPEND _sources tests/gguf_stub.cpp)
        endif()
        add_executable(${target} ${_sources})
        target_link_libraries(${target} PRIVATE quench GTest::gtest GTest::gtest_main quench_warnings)
        target_include_directories(${target} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_SOURCE_DIR}/tests
        )
        set_target_properties(${target} PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
        )
        # NOTE: deliberately NOT gtest_discover_tests() — it would register every
        # test individually IN ADDITION to the unit/gpu/perf label aggregates
        # below, so a plain `ctest` (no label) ran the whole suite twice (R5/).
        # The label aggregates are the single ctest registration; discovery would
        # also require a GPU at configure-time (the binaries enumerate CUDA tests).
    endfunction()

    quench_add_test_module(test-core SOURCES
        tests/test_tensor.cpp
        tests/test_tensor_kind_table.cpp
        tests/test_tensor_kind_matcher.cpp
        tests/test_tensor_kind_coverage.cpp
        tests/test_kv_cache.cpp
        tests/test_gguf_loader.cpp
        tests/test_gguf_fault_injection.cpp
        tests/test_gguf_half.cpp
        tests/test_config.cpp
        # RuntimeConfig → process_diag snapshot. CPU-only by design:
        # process_diag exists so leaf kernels can read config without carrying
        # one, so the coupling is testable with no CUDA, Model or Engine.
        tests/test_process_diag.cpp
        # Resolved-dispatch recorder. CPU-only: plain POD + name tables,
        # and it guards the "new tier prints '?' in the summary" gap.
        tests/test_dispatch_record.cpp
        # Graph-demotion reason enum (F-14). CPU-only: the enum↔name binding is
        # what makes "graphs=0(reason)" readable, and a new reason without a
        # name would silently print "?".
        tests/test_graph_eligibility.cpp
        # Public C enums vs their internal counterparts. A test file may
        # include both sides, which is the only place the two hand-maintained
        # copies of the arch/dtype numbering can be tied together.
        tests/test_c_api_enum_binding.cpp
        # Pre-cudaDeviceReset hook registry: self-registration replaced a
        # hand-maintained call list, so the new failure mode is an EMPTY registry.
        tests/test_cuda_static_reset.cpp
        # Generative FSM battery for the schema-less json_object grammar. Lives
        # in the CPU lane on purpose: the FSM is CUDA-free without init(), and
        # every past grammar bug escaped CI because its tests need a GPU.
        tests/test_json_constrain_property.cpp
        tests/test_schema_constrain_property.cpp
        # GBNF grammar engine (parser + pushdown simulator + UTF-8 assembly).
        # CPU-only by construction — see the file header.
        tests/test_gbnf_grammar.cpp
        # Pre-dequant VRAM budget split. CPU-only: pure arithmetic in
        # runtime/vram_budget.h, and it belongs in the lane CI actually runs.
        tests/test_pre_dequant_budget.cpp
        # Memory subsystem L1 (docs/MEMORY_ARCHITECTURE.md). FakeBackend is the
        # substitution seam that keeps the whole allocator stack testable in the
        # CPU lane; it is test scaffolding, so it is compiled here rather than
        # shipped in the quench library.
        src/memory/fake_backend.cpp
        tests/test_memory_backend.cpp
        tests/test_memory_allocators.cpp
        tests/test_memory_plan.cpp
        # The KV-pool sizing arithmetic. Pure integers, so it belongs in
        # the lane CI runs — it used to be inline in engine_kv_cache_init.cpp,
        # unreachable without a GPU and a 22 GB checkpoint.
        tests/test_kv_residual_sizing.cpp
        tests/test_graph_slots.cpp
        # T5's engine-persistent half. The property that matters — freed exactly
        # once, never after a move — is ownership, not CUDA, so it belongs in the
        # lane CI actually runs.
        tests/test_host_pinned.cpp
        # Arena sizing arithmetic (A7 step 4b). Pure and Model-free via ExecShape,
        # so the numbers the T2 arena is reserved from are pinned in the lane CI
        # actually runs.
        tests/test_workspace_sizes.cpp
        # Persisted library-reserve measurement: a keyed text file whose number
        # decides how much VRAM the plan hands out, so its failure modes are
        # pinned in the lane CI actually runs.
        tests/test_library_reserve_cache.cpp
        tests/test_weight_snapshot.cpp
        tests/test_rope_override.cpp
        tests/test_think_stop_logic.cpp
        tests/test_stream_pipeline.cpp
    )
    if(QUENCH_BUILD_SERVER)
        # Anthropic /v1/messages transform tests reuse the server's
        # self-contained anthropic.cpp (json-only dependencies).
        # test_sse_stream_utils.cpp covers the SSE chunk envelopes +
        # reasoning/channel splits in tools/quench-server/utils.cpp (issue).
        target_sources(test-core PRIVATE
            tools/quench-server/anthropic.cpp
            tools/quench-server/responses.cpp
            tools/quench-server/utils.cpp
            tools/quench-server/tool_call.cpp
            tools/quench-server/tool_call_gemma.cpp
        tools/quench-server/tool_call_gemma.cpp
        tools/common/args_common.cpp
    tools/quench-server/args.cpp
            tests/test_anthropic_transform.cpp
            tests/test_responses_transform.cpp
            tests/test_sse_stream_utils.cpp
            tests/test_tool_call.cpp
        tests/test_regex_constrain.cpp
            tests/test_tool_stream_filter.cpp
            tests/test_reasoning_reconcile.cpp
            tests/test_stream_reasoning_split.cpp
            tests/test_server_auth.cpp
            # Admission-time rejection of unenforceable constraints. The
            # alias coverage is the part that rots: a new spelling in the
            # parameter parser without a line here is a validation bypass.
            tools/quench-server/constraint_validation.cpp
            tests/test_constraint_validation.cpp
            tests/test_server_args.cpp
            tests/test_base64.cpp)
        target_include_directories(test-core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tools/quench-server
                                                     ${CMAKE_CURRENT_SOURCE_DIR}/tools)
        target_link_libraries(test-core PRIVATE nlohmann_json::nlohmann_json httplib::httplib)
    endif()
    # test_stream_pipeline.cpp includes the self-contained pure header
    # tools/quench-server/stream_pipeline.h (no httplib/json deps).
    target_include_directories(test-core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tools/quench-server
                                                 ${CMAKE_CURRENT_SOURCE_DIR}/tools)

    quench_add_test_module(test-text SOURCES
        tests/test_tokenizer.cpp
        tests/test_tokenizer_robustness.cpp
        tests/test_tokenizer_compat.cpp
        tests/test_chat_template.cpp
        tests/test_jinja.cpp
    )

    quench_add_test_module(test-compute SOURCES
        tests/test_rope.cu
        tests/test_layernorm.cu
        tests/test_activation.cu
        tests/test_embedding.cu
        tests/test_gemm.cu
        tests/test_gemm_capture_fp16_sm120.cu
        tests/test_gemm_kernel_registry.cu
        tests/test_cluster_launch.cu
        tests/test_fp8_gemm.cu
        tests/test_reduce.cu
        tests/test_softmax.cu
        tests/test_sampling.cu
        tests/test_executor_kernels.cu
    )

    quench_add_test_module(test-attention SOURCES
        tests/test_attention_fmha_sm120.cu
        tests/test_attention_crosspath.cu
        tests/test_paged_attention.cu
        tests/test_attention_paged_oracle.cu
        tests/test_attention_chunked.cu
    )

    quench_add_test_module(test-quant SOURCES
        tests/test_quant.cu
        tests/test_quant_integration.cu
        tests/test_gguf_dequant_ref.cu
        tests/test_nvfp4_quant.cu
        tests/test_nvfp4_quant_ref.cu
        tests/test_nvfp4_quant_hw.cu
        tests/test_nvfp4_compressed_tensors_ref.cu
        tests/test_nvfp4_outlier_ref.cu
        tests/test_nvfp4_gemv_kpar_loop.cu
        tests/test_nvfp4_gemm_batched.cu
        tests/test_nvfp4_gemv_gemma3_dims.cu
        tests/test_nvfp4_gemm_graph_capture.cu
        tests/test_cutlass_nvfp4_alpha.cu
        tests/test_mmq_q8_imma.cu
        tests/test_weight_dispatch.cu
    )

    quench_add_test_module(test-kv SOURCES
        tests/test_kv_cache_write.cu
        tests/test_kv_gather.cu
        tests/test_kv_block_copy.cu
        tests/test_green_ctx.cu
        tests/test_capture_abort.cu
        tests/test_prefix_cache_equiv.cpp
        tests/test_vram_query.cpp
    )

    quench_add_test_module(test-moe-gdn SOURCES
        tests/test_json_constrain.cu
        tests/test_schema_constrain.cu
        tests/test_tool_call_constrain.cu
    )

    quench_add_test_module(test-e2e WITH_STUB SOURCES
        tests/test_forward_pass.cu
        tests/test_engine_integration.cu
        tests/test_e2e.cpp
        tests/test_e2e_models.cpp
        tests/test_e2e_greedy_lock.cpp
        tests/test_continuous_batching.cpp
        tests/test_degeneration.cpp
        tests/test_quant_pipeline.cpp
        tests/test_weight_registry_preservation.cpp
        tests/test_vram_budget_reserve.cpp
        tests/test_chunked_prefill.cu
        tests/test_api_generate.cpp
        tests/test_engine_relaunch.cpp
        tests/test_suspend_resume.cu
        tests/test_warm_cache.cu
        tests/test_prefix_cache_e2e.cpp
        tests/test_determinism_e2e.cpp
    )

    set(QUENCH_TEST_BINARIES
        test-core test-text test-compute test-attention
        test-quant test-kv test-moe-gdn test-e2e
    )

    # --- CTest label aggregates ---
    # ctest -L unit  — CPU-only tests (no GPU required)
    # ctest -L gpu   — GPU tests (require NVIDIA GPU)
    # ctest -L perf  — Performance regression tests
    #
    # Source of truth for the unit/gpu split. The gtest filter only applies to
    # test-e2e (which contains both stub-only unit tests and CUDA-backed E2E
    # tests in the same binary). All other binaries are clean unit-or-GPU.
    #
    # The CPU and GPU tests are interleaved inside test_e2e.cpp / test_continuous_
    # batching.cpp (e.g. StubModelTest has both CPU and GPU subtests, sharing one
    # fixture), so they cannot be split into a separate binary without duplicating
    # fixtures. Instead the split stays a filter, but `guard_e2e_lane_split` below
    # asserts the filter still resolves to the exact expected CPU-only set — a
    # rename can no longer silently shift a test into the wrong lane (R5/).
    # NOTE: scripts/verify.sh carries a literal copy of this filter (container
    # path has no build/ dir) — keep both in sync when changing it.
    set(_unit_e2e_filter "BatchBuilderTest.*:SchedulerTest.*:RequestTest.*:EndToEndTest.*:StubModelTest.LoadStubModel:StubModelTest.TokenizeStub")

    add_test(NAME unit_core       COMMAND test-core)
    add_test(NAME unit_text       COMMAND test-text)
    add_test(NAME unit_e2e_subset COMMAND test-e2e "--gtest_filter=${_unit_e2e_filter}")
    add_test(NAME guard_e2e_lane_split
        COMMAND ${CMAKE_COMMAND} -E env
                ${CMAKE_CURRENT_SOURCE_DIR}/scripts/check_e2e_lane_split.sh
                $<TARGET_FILE:test-e2e> "${_unit_e2e_filter}")
    set_tests_properties(unit_core unit_text unit_e2e_subset guard_e2e_lane_split
        PROPERTIES LABELS "unit")

    # PR eval verdict policy (scripts/eval_scorer.py via scripts/pr_eval_bot.py):
    # pure Python, stdlib-only, GPU-free. In the unit lane so a policy change
    # cannot land without its tests running.
    add_test(NAME unit_eval_policy
        COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/tests/eval/test_pr_eval_bot.py)
    set_tests_properties(unit_eval_policy PROPERTIES LABELS "unit")

    add_test(NAME gpu_compute   COMMAND test-compute)
    add_test(NAME gpu_attention COMMAND test-attention)
    add_test(NAME gpu_quant     COMMAND test-quant)
    add_test(NAME gpu_kv        COMMAND test-kv)
    add_test(NAME gpu_moe_gdn   COMMAND test-moe-gdn)
    add_test(NAME gpu_e2e       COMMAND test-e2e "--gtest_filter=-${_unit_e2e_filter}")
    set_tests_properties(gpu_compute gpu_attention gpu_quant gpu_kv gpu_moe_gdn gpu_e2e
        PROPERTIES LABELS "gpu")

    set(_perf_filter "*Perf*:*Bench*:*Throughput*")
    add_test(NAME perf_compute   COMMAND test-compute   "--gtest_filter=${_perf_filter}")
    add_test(NAME perf_attention COMMAND test-attention "--gtest_filter=${_perf_filter}")
    add_test(NAME perf_quant     COMMAND test-quant     "--gtest_filter=${_perf_filter}")
    add_test(NAME perf_e2e       COMMAND test-e2e       "--gtest_filter=${_perf_filter}")
    set_tests_properties(perf_compute perf_attention perf_quant perf_e2e
        PROPERTIES LABELS "perf")

    # --- Convenience wrappers for environments without the build tree ---
    # The Docker runtime image only carries the test binaries (no
    # CTestTestfile.cmake), so `ctest -L unit` cannot be used there. These
    # shell wrappers run the same partitions inside the runtime container.
    # On a host build, prefer `ctest -L unit|gpu|perf` over the wrappers.

    # quench-tests: run all GTest binaries sequentially. Matches `ctest` (no label).
    file(WRITE ${CMAKE_BINARY_DIR}/quench-tests
"#!/bin/sh
DIR=$(cd \"$(dirname \"$0\")\" && pwd)
status=0
for bin in test-core test-text test-compute test-attention test-quant test-kv test-moe-gdn test-e2e; do
    if [ -x \"$DIR/$bin\" ]; then
        \"$DIR/$bin\" \"$@\" || status=$?
    fi
done
exit $status
")

    # quench-tests-unit: CPU-only subset. Matches `ctest -L unit`. Sourced from the
    # same _unit_e2e_filter as the unit_e2e_subset ctest entry above.
    file(WRITE ${CMAKE_BINARY_DIR}/quench-tests-unit
"#!/bin/sh
DIR=$(cd \"$(dirname \"$0\")\" && pwd)
status=0
if [ -x \"$DIR/test-core\" ]; then \"$DIR/test-core\" \"$@\" || status=$?; fi
if [ -x \"$DIR/test-text\" ]; then \"$DIR/test-text\" \"$@\" || status=$?; fi
if [ -x \"$DIR/test-e2e\" ];  then \"$DIR/test-e2e\" '--gtest_filter=${_unit_e2e_filter}' \"$@\" || status=$?; fi
exit $status
")

    foreach(_w quench-tests quench-tests-unit)
        file(CHMOD ${CMAKE_BINARY_DIR}/${_w}
            PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                        GROUP_READ GROUP_EXECUTE
                        WORLD_READ WORLD_EXECUTE)
    endforeach()
endif()

# --- Install ---
include(GNUInstallDirs)

install(TARGETS quench
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(DIRECTORY include/quench/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/quench
    FILES_MATCHING PATTERN "*.h"
)

if(QUENCH_BUILD_TOOLS)
    install(TARGETS quench-cli RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

if(QUENCH_BUILD_SERVER)
    install(TARGETS quench-server RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
