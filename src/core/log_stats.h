#pragma once

// Log statistics, split from logging.h so consumers do not pull the macro
// header into their recompile set (logging.h is included by every TU).

#include <cstdint>

namespace quench {

// Process-wide count of ERROR/FATAL log lines since start, independent of the
// visibility filter. "Zero" is the only value with a guarantee: nothing has
// been reported wrong. Use it to gate decisions that must not be made from a
// degraded state — e.g. persisting a measurement taken during an errored init.
uint64_t log_error_count();

}  // namespace quench
