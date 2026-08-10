# Turns the web UI into a header the server links in, so the binary carries the
# page with it — no runtime asset path to get wrong inside the container image.
#
# Invoked as a script: cmake -DIN=<html> -DOUT=<header> -P embed_webui.cmake

file(READ "${IN}" QUENCH_WEBUI_PAYLOAD)

# The delimiter must not occur in the payload, or the raw string ends early.
if(QUENCH_WEBUI_PAYLOAD MATCHES "\\)IMPUI\"")
    message(FATAL_ERROR "web UI contains the raw-string delimiter )IMPUI\" — change the delimiter in ${CMAKE_CURRENT_LIST_FILE}")
endif()

file(WRITE "${OUT}"
"// Generated from tools/quench-server/webui/index.html — do not edit.
#pragma once

inline constexpr const char* QUENCH_WEBUI_HTML = R\"IMPUI(${QUENCH_WEBUI_PAYLOAD})IMPUI\";
")
