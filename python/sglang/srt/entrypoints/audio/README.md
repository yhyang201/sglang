# Audio Server Implementation for SGLang

This directory contains the audio generation functionality for SGLang, enabling text-to-speech (TTS) capabilities integrated with the SGLang inference engine.

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│              Audio Server (FastAPI)                  │
│                                                      │
│  ┌─────────────────────┐  ┌────────────────────┐   │
│  │  SGLang Engine      │  │  TTS Engine        │   │
│  │  (Text Generation)  │  │  (Token2Wav)       │   │
│  │                     │  │                    │   │
│  │  - Model Inference  │  │  - Audio Synthesis │   │
│  │  - Token Generation │  │  - Voice Cloning   │   │
│  └─────────────────────┘  └────────────────────┘   │
│           ↓                         ↓               │
│  ┌──────────────────────────────────────────────┐  │
│  │     Audio Token Parser                       │  │
│  │     Extracts <audio_XXX> tokens              │  │
│  └──────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

## Components

### 1. **TTS Engine Abstraction** (`/tts/`)

Base abstraction for TTS engines:
- `base_tts_engine.py`: Abstract base class for all TTS engines
- `step_audio2/`: Step-Audio2 TTS implementation
  - `step_audio2_tts.py`: Implements BaseTTSEngine
  - `token2wav.py`: Core TTS model (from Step-Audio2)
  - Supporting modules (flashcosyvoice, cosyvoice2, etc.)

### 2. **Audio Token Parser** (`/parser/`)

Extracts audio tokens from generation output:
- Pattern: `<audio_123>` where 123 is the token ID
- Similar architecture to function call parser
- Supports both streaming and non-streaming modes

### 3. **Audio Server** (`audio_server.py`)

Orchestrates the entire audio generation pipeline:
1. Receives chat messages
2. Generates text with SGLang Engine
3. Extracts audio tokens from output
4. Synthesizes audio with TTS Engine
5. Returns combined text + audio response

### 4. **OpenAI Audio API** (`/openai/serving_audio.py`)

OpenAI-compatible API endpoints:
- `/v1/chat/completions` with audio modality support
- Follows OpenAI Audio API specification
- Reference: https://platform.openai.com/docs/guides/audio

## File Structure

```
sglang/srt/
├── tts/
│   ├── __init__.py
│   ├── base_tts_engine.py              # TTS abstract base class
│   └── step_audio2/                    # Step-Audio2 implementation
│       ├── __init__.py
│       ├── step_audio2_tts.py          # TTS Engine implementation
│       ├── token2wav.py                # Core TTS model
│       ├── flashcosyvoice/             # Fast inference backend
│       ├── cosyvoice2/                 # Audio processing modules
│       └── utils.py                    # Utilities
│
├── parser/
│   └── audio_token_parser.py           # Extract audio tokens
│
├── entrypoints/
│   ├── audio/
│   │   ├── __init__.py
│   │   ├── audio_server.py             # Main audio server
│   │   └── example.py                  # Usage example
│   │
│   └── openai/
│       └── serving_audio.py            # OpenAI Audio API
```

## Usage Example

### Basic Usage

```python
import asyncio
from sglang.srt.entrypoints.engine import Engine
from sglang.srt.tts.step_audio2.step_audio2_tts import StepAudio2TTS
from sglang.srt.entrypoints.audio.audio_server import AudioServer

async def main():
    # Initialize SGLang Engine
    sglang_engine = Engine(
        model_path="stepfun-ai/Step-Audio-2-mini",
    )

    # Initialize TTS Engine
    tts_engine = StepAudio2TTS(
        model_path="Step-Audio-2-mini/token2wav",
        float16=True,
    )

    # Initialize Audio Server
    audio_server = AudioServer(
        sglang_engine=sglang_engine,
        tts_engine=tts_engine,
        default_prompt_wav="assets/default_female.wav",
    )

    # Generate audio response
    result = await audio_server.generate_audio_response(
        messages="Please introduce yourself.",
        sampling_params={"temperature": 0.7, "max_new_tokens": 1024}
    )

    print(f"Text: {result['text']}")
    print(f"Audio bytes: {len(result['audio'])}")

    # Cleanup
    audio_server.shutdown()

asyncio.run(main())
```

### With FastAPI (Production)

```python
from fastapi import FastAPI
from sglang.srt.entrypoints.openai.serving_audio import OpenAIServingAudio
from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest

app = FastAPI()

# Initialize at startup
audio_server = AudioServer(...)
audio_handler = OpenAIServingAudio(..., audio_server)

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if 'audio' in (request.modalities or []):
        return await audio_handler.create_chat_completion_with_audio(request)
    else:
        # Regular text completion
        pass
```

## Key Features

✅ **Modular Design**: Easily swap TTS engines
✅ **OpenAI Compatible**: Follow OpenAI Audio API spec
✅ **In-Process**: Both engines run in same process for efficiency
✅ **Extensible**: Easy to add new TTS implementations

## Current Limitations (TODOs)

### High Priority
- [ ] Implement streaming support for audio generation
- [ ] Add proper error handling and retry logic
- [ ] Parse audio input from OpenAI request format
- [ ] Format response according to OpenAI Audio API spec
- [ ] Get audio token IDs from model config (currently hardcoded pattern)
- [ ] Implement proper message format handling (OpenAI chat format)

### Medium Priority
- [ ] Add audio format conversion (MP3, Opus, AAC, etc.)
- [ ] Implement memory management (load/unload TTS model)
- [ ] Add caching for prompt_wav embeddings
- [ ] Add Base64 encoding options
- [ ] Implement batch processing support

### Low Priority
- [ ] Add performance monitoring and metrics
- [ ] Add unit tests for all components
- [ ] Add GPU memory optimization (custom_mem configuration)
- [ ] Add dedicated `/v1/audio/speech` endpoint
- [ ] Add audio transcription endpoint

## Testing

```bash
# Run the example
python -m sglang.srt.entrypoints.audio.example

# TODO: Add unit tests
# pytest tests/audio/
```

## Dependencies

- `torch`: Deep learning framework
- `torchaudio`: Audio processing
- `s3tokenizer`: Audio tokenizer (from Step-Audio2)
- `onnxruntime`: Speaker embedding model
- `hyperpyyaml`: Config loading

See `Step-Audio2/` for complete dependencies.

## References

- OpenAI Audio API: https://platform.openai.com/docs/guides/audio
- Step-Audio2: https://huggingface.co/stepfun-ai/Step-Audio-2-mini
- SGLang Documentation: https://github.com/sgl-project/sglang

## Contributing

This is an initial framework implementation. Areas for improvement:
1. Streaming support
2. Better error handling
3. Performance optimization
4. More TTS engine implementations
5. Comprehensive testing

## License

Same as SGLang main project (Apache 2.0).
