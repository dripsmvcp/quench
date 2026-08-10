#pragma once

#include "quench/quench.h"
#include "memory/weight_snapshot.h"
#include "model/model.h"
#include "runtime/request.h"

#include <memory>

namespace quench {
// Forward-declared on purpose. QuenchContext_T only stores an Engine, and pulling
// runtime/engine.h in here put it in front of every TU that includes this header
// — most of which never touch Engine at all. The out-of-line destructor below is
// what makes the forward declaration legal: std::unique_ptr needs the complete
// type where the deleter is instantiated, and that is now quench_api.cpp alone.
class Engine;
}  // namespace quench

// Internal handle types backing the opaque C API handles.
// Shared between quench_api.cpp and tool binaries that need
// direct access to the engine (quench-cli, quench-server).

struct QuenchModel_T {
    std::shared_ptr<quench::Model> model;
};

struct QuenchWeightSnapshot_T {
    std::unique_ptr<quench::WeightSnapshot> snap;
};

struct QuenchContext_T {
    QuenchContext_T();
    ~QuenchContext_T();

    QuenchModel model_handle = nullptr;
    std::unique_ptr<quench::Engine> engine;

    // State for token-level prefill/decode API
    std::shared_ptr<quench::Request> active_request;

    // Multi-token step consumption (self-speculative decode produces N tokens per step)
    size_t consumed_output = 0;
};
