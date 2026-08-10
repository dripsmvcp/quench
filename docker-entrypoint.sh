#!/bin/bash
set -e

# Translate environment variables to quench-server CLI flags.
# Usage: docker run ... -e QUENCH_MODEL=/models/model.gguf quench:latest

# If the first argument is a flag (e.g. --help, --model, --version), it is meant
# for the default command, not a command name. Prepend quench-server so the flag
# reaches the real binary instead of being exec'd as a command.
if [ "${1#-}" != "$1" ]; then
    set -- quench-server "$@"
fi

CMD="$1"
shift 2>/dev/null || true

# Default to quench-server if no command given
if [ -z "$CMD" ]; then
    CMD="quench-server"
fi

# If the command is not quench-server or quench-cli, exec directly (e.g. bash, sh)
case "$CMD" in
    quench-server|quench-cli) ;;
    *) exec "$CMD" "$@" ;;
esac

args=()

# Model path
if [ -n "$QUENCH_MODEL" ]; then
    args+=(--model "$QUENCH_MODEL")
fi

# Host — default 0.0.0.0 inside container
if [ -n "$QUENCH_HOST" ]; then
    args+=(--host "$QUENCH_HOST")
elif [ "$CMD" = "quench-server" ]; then
    args+=(--host "0.0.0.0")
fi

# Port
if [ -n "$QUENCH_PORT" ]; then
    args+=(--port "$QUENCH_PORT")
fi

# Max tokens
if [ -n "$QUENCH_MAX_TOKENS" ]; then
    args+=(--max-tokens "$QUENCH_MAX_TOKENS")
fi

# GPU layers
if [ -n "$QUENCH_GPU_LAYERS" ]; then
    args+=(--gpu-layers "$QUENCH_GPU_LAYERS")
fi

# Device ID
if [ -n "$QUENCH_DEVICE" ]; then
    args+=(--device "$QUENCH_DEVICE")
fi

# Chat template
if [ -n "$QUENCH_CHAT_TEMPLATE" ]; then
    args+=(--chat-template "$QUENCH_CHAT_TEMPLATE")
fi

# Boolean flags — accept 1 or true
is_true() { [ "$1" = "1" ] || [ "$1" = "true" ] || [ "$1" = "TRUE" ]; }

if is_true "$QUENCH_KV_FP8"; then
    args+=(--kv-fp8)
fi

if is_true "$QUENCH_KV_INT8"; then
    args+=(--kv-int8)
fi

if [ "$QUENCH_DECODE_NVFP4" = "1" ]; then
    args+=(--decode-nvfp4)
elif [ "$QUENCH_DECODE_NVFP4" = "2" ]; then
    args+=(--decode-nvfp4-only)
elif [ "$QUENCH_DECODE_NVFP4" = "0" ]; then
    args+=(--no-nvfp4)
fi

if is_true "$QUENCH_DECODE_NVFP4_ONLY"; then
    args+=(--decode-nvfp4-only)
fi

if is_true "$QUENCH_NO_NVFP4"; then
    args+=(--no-nvfp4)
fi

if is_true "$QUENCH_NO_CUDA_GRAPHS"; then
    args+=(--no-cuda-graphs)
fi

if is_true "$QUENCH_SSM_FP16"; then
    args+=(--ssm-fp16)
fi

# Vision encoder
if [ -n "$QUENCH_MMPROJ" ]; then
    args+=(--mmproj "$QUENCH_MMPROJ")
fi

# Prefill chunk size
if [ -n "$QUENCH_PREFILL_CHUNK_SIZE" ]; then
    args+=(--prefill-chunk-size "$QUENCH_PREFILL_CHUNK_SIZE")
fi

# Think budget
if [ -n "$QUENCH_THINK_BUDGET" ]; then
    args+=(--think-budget "$QUENCH_THINK_BUDGET")
fi

# Models directory
if [ -n "$QUENCH_MODELS_DIR" ]; then
    args+=(--models-dir "$QUENCH_MODELS_DIR")
elif [ "$CMD" = "quench-server" ] && [ -d "/models" ]; then
    args+=(--models-dir "/models")
fi

exec "$CMD" "${args[@]}" "$@"
