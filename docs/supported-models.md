# Supported models

quench runs **dense llama-family transformers from GGUF files**. Model families with a known-working, end-to-end verified code path are listed below. Throughput methodology notes live in [`performance.md`](performance.md). VRAM figures are model weights only. The KV cache is sized *on top* of that from whatever is left once the weight caches are built, so it scales with free VRAM and the configured context rather than sitting in a fixed band. Bound it with `--max-seq-len` / `kv_cache` settings and read the actual split with `--mem-report`.

Anything not on this list may still load — the GGUF path covers most LLaMA-derived dense architectures — but it has not been verified end-to-end.

## Verified models

**Q8_0 is the verified GGUF quantization.** Dequant paths for other GGUF quants exist but are unverified; prefer Q8_0.

| Model | Quant | VRAM (weights) | Format |
|---|---|---:|---|
| [Mistral-Nemo-12B-Instruct](https://huggingface.co/bartowski/Mistral-Nemo-Instruct-2407-GGUF) | Q8_0 | 12.4 GB | GGUF |
| [Mistral-7B-Instruct](https://huggingface.co/bartowski/Mistral-7B-Instruct-v0.3-GGUF) | Q8_0 | 7.7 GB | GGUF |
| [Llama-3.2-3B-Instruct](https://huggingface.co/unsloth/Llama-3.2-3B-Instruct-GGUF) | Q8_0 | 3.2 GB | GGUF |
| Llama 3.x dense (8B-class) | Q8_0 | ~8.5 GB | GGUF |
| [Qwen3-4B](https://huggingface.co/unsloth/Qwen3-4B-GGUF) | Q8_0 | 4.0 GB | GGUF |
| [Qwen3-8B](https://huggingface.co/unsloth/Qwen3-8B-GGUF) | Q8_0 | 8.2 GB | GGUF |

## Format notes

- **GGUF** — standard llama.cpp format, loaded directly from a single file. Q8_0 is verified; other quant types in the file are handled by generic dequant paths but are not verified. Most community quants come from [unsloth](https://huggingface.co/unsloth) or [bartowski](https://huggingface.co/bartowski).
- At load time the engine builds its own device-side caches from the GGUF payload: an FP8 E4M3 weight cache for prefill and an NVFP4 decode cache. See [`quantization.md`](quantization.md).

## Loading

```bash
# GGUF — file path
quench-cli --model models/Mistral-Nemo-Instruct-2407-Q8_0.gguf --prompt "Hello"
```

quench can resolve a Hugging Face repo id to a GGUF file (`--model <repo-id>` with optional `--revision`), or you can stage weights yourself:

```bash
huggingface-cli download bartowski/Mistral-Nemo-Instruct-2407-GGUF \
  Mistral-Nemo-Instruct-2407-Q8_0.gguf --local-dir models/
```
