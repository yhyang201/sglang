# Step-Audio2 Integration Guide

This guide explains how to use the Step-Audio2 parsers with SGLang for TTS (Text-to-Speech) and tool calling functionality.

## Quick Start

### 1. Server Configuration

Start the SGLang server with Step-Audio2 parser configurations:

```bash
python -m sglang.launch_server \
    --model-path stepfun-ai/Step-Audio-2-mini \
    --audio-parser step_audio_2 \
    --tool-call-parser step_audio_2 \
    --reasoning-parser step-audio2 \
    --port 30000
```

**Configuration Parameters:**
- `--audio-parser step_audio_2`: Enable TTS content parsing
- `--tool-call-parser step_audio_2`: Enable tool call parsing
- `--reasoning-parser step-audio2`: Enable thinking content parsing

### 2. Client Usage

#### Text-to-Speech (TTS) Example

```python
import requests

url = "http://localhost:8000/v1/chat/completions"

# Request with TTS output
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Tell me about Paris."},
    {"role": "assistant", "content": "<tts_start>", "eot": False}
]

response = requests.post(
    url,
    json={
        "model": "step-audio-2-mini",
        "messages": messages,
        "max_tokens": 2048,
        "temperature": 0.7
    }
)

result = response.json()

# Access TTS content
tts_content = result["choices"][0]["message"]["tts_content"]
print("TTS Text:", tts_content["tts_text"])
print("TTS Audio Tokens:", tts_content["tts_audio"])
```

**Expected Response Structure:**

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "step-audio-2-mini",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "Additional text after TTS",
      "tts_content": {
        "tts_text": "Paris is the capital of France...",
        "tts_audio": "<audio_0><audio_1><audio_2>...<audio_500>"
      }
    },
    "finish_reason": "stop"
  }],
  "usage": {...}
}
```

#### Tool Calling Example

```python
import requests

url = "http://localhost:8000/v1/chat/completions"

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather information",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name"
                }
            },
            "required": ["location"]
        }
    }
}]

messages = [
    {"role": "user", "content": "What's the weather in Shanghai?"}
]

response = requests.post(
    url,
    json={
        "model": "step-audio-2-mini",
        "messages": messages,
        "tools": tools,
        "max_tokens": 1024
    }
)

result = response.json()
tool_calls = result["choices"][0]["message"]["tool_calls"]
print("Tool calls:", tool_calls)
```

**Expected Tool Call Format:**

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "Let me check the weather for you.",
      "tool_calls": [{
        "id": "call_123",
        "type": "function",
        "function": {
          "name": "get_weather",
          "arguments": "{\"location\": \"Shanghai\"}"
        }
      }]
    }
  }]
}
```

#### Thinking/Reasoning Example

```python
import requests

url = "http://localhost:8000/v1/chat/completions"

messages = [
    {"role": "user", "content": "Solve this math problem: 123 + 456"}
]

response = requests.post(
    url,
    json={
        "model": "step-audio-2-mini",
        "messages": messages,
        "separate_reasoning": True,  # Enable reasoning separation
        "max_tokens": 1024
    }
)

result = response.json()
print("Reasoning:", result["choices"][0]["message"]["reasoning_content"])
print("Answer:", result["choices"][0]["message"]["content"])
```

## TTS Content Details

### Format

The `tts_content` field contains:
- `tts_text`: The text content that should be spoken
- `tts_audio`: Audio tokens in format `<audio_0><audio_1>...`

### Audio Token Format

Audio tokens are represented as strings like `<audio_123>` where:
- Token range: `<audio_0>` to `<audio_6560>`
- Each token represents a quantized audio unit
- These can be converted to actual audio using the Token2wav model

### Converting Audio Tokens to WAV

```python
from sglang.srt.tts.step_audio2.token2wav import Token2wav
import re

# Extract audio token IDs from tts_audio string
audio_str = tts_content["tts_audio"]  # e.g., "<audio_0><audio_1><audio_2>"
audio_token_ids = [int(x) for x in re.findall(r'<audio_(\d+)>', audio_str)]

# Load Token2wav model
token2wav = Token2wav('path/to/token2wav/model')

# Generate audio
audio_bytes = token2wav(audio_token_ids, prompt_wav='assets/default_female.wav')

# Save to file
with open('output.wav', 'wb') as f:
    f.write(audio_bytes)
```

## Parser Configuration

### Audio Parser

The audio parser extracts TTS content from model output:

```python
# Server-side configuration
from sglang.srt.parser.audio_parser import AudioParserManager

parser = AudioParserManager.create_parser("step_audio_2", tokenizer)

# Check if TTS output expected
is_tts = parser.is_tts_output(prompt_token_ids)

# Parse TTS content
text_tokens, audio_tokens, other_tokens = parser.extract_tts_content_nonstreaming(
    output_token_ids,
    is_tts_ta4_output=is_tts
)
```

### Tool Parser

The tool parser handles function calls:

```python
from sglang.srt.function_call.function_call_parser import FunctionCallParser

parser = FunctionCallParser(tools=tools, tool_call_parser="step_audio_2")

# Parse tool calls
normal_text, tool_calls = parser.parse_non_stream(full_text)
```

### Reasoning Parser

The reasoning parser separates thinking content:

```python
from sglang.srt.parser.reasoning_parser import ReasoningParser

parser = ReasoningParser(
    model_type="step-audio2",
    stream_reasoning=False
)

# Parse reasoning
reasoning_text, normal_text = parser.parse_non_stream(full_text)
```

## Complete Example: Speech-to-Speech with Tool Calling

```python
import requests
import re

url = "http://localhost:8000/v1/chat/completions"

# Define tools
tools = [{
    "type": "function",
    "function": {
        "name": "search",
        "description": "Search for information",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"}
            },
            "required": ["query"]
        }
    }
}]

# First request: User asks a question
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Search for information about the Eiffel Tower"},
    {"role": "assistant", "content": "<tts_start>", "eot": False}
]

response = requests.post(
    url,
    json={
        "model": "step-audio-2-mini",
        "messages": messages,
        "tools": tools,
        "max_tokens": 4096,
        "temperature": 0.7
    }
)

result = response.json()
message = result["choices"][0]["message"]

print("TTS Content:", message.get("tts_content"))
print("Tool Calls:", message.get("tool_calls"))

# Extract audio tokens if present
if message.get("tts_content") and message["tts_content"]["tts_audio"]:
    audio_str = message["tts_content"]["tts_audio"]
    audio_tokens = [int(x) for x in re.findall(r'<audio_(\d+)>', audio_str)]
    print(f"Extracted {len(audio_tokens)} audio tokens")

    # Convert to audio (requires Token2wav model)
    # token2wav = Token2wav('path/to/model')
    # audio_bytes = token2wav(audio_tokens, prompt_wav='default_voice.wav')
```

## Troubleshooting

### Parser Not Activated

If parsers are not working:
1. Check server args include the parser configuration
2. Verify the parser is registered in the manager
3. Check logs for parser initialization errors

```bash
# Check if parsers are loaded
grep "Parser initialized" server.log
```

### TTS Content Not Parsed

If `tts_content` is always `null`:
1. Ensure `--audio-parser step_audio_2` is set
2. Verify the prompt ends with `<tts_start>`
3. Check that the model output includes TTS tokens

### Tool Calls Not Detected

If tool calls are not parsed:
1. Ensure `--tool-call-parser step_audio_2` is set
2. Verify tools are provided in the request
3. Check the model output format matches `<tool_call>function\n...`

## Limitations (Current MVP)

1. **No Streaming for Audio Parser**: TTS content parsing only works in non-streaming mode
2. **Re-tokenization Overhead**: Output text is re-tokenized for audio parsing (workaround until token IDs are directly accessible)
3. **Simplified Streaming for Tool Parser**: Tool parser has basic streaming support

## Next Steps

1. Implement full streaming support for audio parser
2. Optimize by accessing output token IDs directly without re-tokenization
3. Add comprehensive integration tests
4. Add audio generation endpoint (Token2wav integration)

## API Reference

### Server Arguments

- `--audio-parser <name>`: Audio parser to use (default: None)
  - Options: `step_audio_2`

- `--tool-call-parser <name>`: Tool call parser to use
  - Options: `step_audio_2`, `llama3`, `qwen25`, etc.

- `--reasoning-parser <name>`: Reasoning parser to use
  - Options: `step-audio2`, `deepseek-r1`, `qwen3`, etc.

### Response Fields

#### ChatMessage

- `content`: Main text content (after TTS extraction)
- `tts_content`: TTS content object
  - `tts_text`: Text to be spoken
  - `tts_audio`: Audio tokens as string
- `tool_calls`: Array of tool call objects
- `reasoning_content`: Thinking/reasoning text

## Support

For issues or questions:
- Check the main documentation: `STEP_AUDIO2_PARSERS.md`
- Review parser test files for examples
- Open an issue on GitHub with detailed logs

## References

- [Step-Audio2 Official Repo](https://github.com/stepfun-ai/Step-Audio2)
- [vLLM Implementation](https://github.com/stepfun-ai/vllm/tree/step-audio2-mini)
- [SGLang Documentation](https://sgl-project.github.io/)
