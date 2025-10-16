# StepAudio Integration Test Plan

## Overview
This document describes how to validate the StepAudio2 text-to-speech integration inside SGLang.

## Prerequisites
- GPU-enabled environment, e.g. `CUDA_VISIBLE_DEVICES=1`
- Step-Audio-2-mini weights downloaded to `/data/cache/huggingface/hub/Step-Audio-2-mini`
- Token2Wav assets available under `/data/cache/huggingface/hub/Step-Audio-2-mini/token2wav`
- Prompt audio file for voice cloning (default `assets/prompt.wav`)

## Smoke Test (token2wav only)
Run the standalone TTS smoke test:

```bash
CUDA_VISIBLE_DEVICES=1 NUMBA_DISABLE_JIT=1 PYTHONPATH=python \
  python scripts/test_stepaudio_tts.py \
  --model-path /data/cache/huggingface/hub/Step-Audio-2-mini/token2wav \
  --prompt-wav assets/prompt.wav \
  --output temp/test_stepaudio_token2wav.wav
```

Expected: the script prints `Wrote temp/test_stepaudio_token2wav.wav (...) bytes`.

## Full HTTP Server Test
1. Ensure caches route to `/data/cache` via the built-in defaults (or set `SGLANG_CACHE_ROOT` manually).
2. Launch the server:

```bash
CUDA_VISIBLE_DEVICES=1 NUMBA_DISABLE_JIT=1 PYTHONPATH=python \
  python -m sglang.srt.entrypoints.http_server \
  --model-path /data/cache/huggingface/hub/Step-Audio-2-mini \
  --served-model-name Step-Audio-2-mini \
  --host 0.0.0.0 --port 30000 \
  --enable-tts-engine \
  --tts-model-path /data/cache/huggingface/hub/Step-Audio-2-mini/token2wav \
  --default-prompt-wav assets/prompt.wav
```

3. Send an OpenAI-compatible request that includes `"modalities": ["audio"]` and verify the response contains synthesized audio bytes/base64.

## Notes
- Caches fall back to `/data/cache/<component>` when corresponding env vars are unset or pointing at `/root/.cache`.
- Replace `assets/prompt.wav` with a real speaker sample as needed.
