# Step-Audio2 Parser Fix Summary

## Problem Statement

The Step-Audio2 model integration had critical parser conflicts that caused:
1. **TTS content appearing in `reasoning_content`**: Audio tokens were incorrectly parsed as reasoning
2. **Tool calls appearing in `reasoning_content`**: Tool call JSON was treated as reasoning
3. **User confusion**: Required manual `<tts_start>` token insertion in messages

## Root Cause

The `DeepSeekR1Detector` reasoning parser used `force_reasoning=True`, which caused it to:
- Consume ALL content when no `</think>` tag was present
- Run BEFORE audio and tool parsers
- Treat audio tokens and tool calls as reasoning content

## Solution Implemented

### 1. Created Custom `StepAudio2Detector`
**File**: `/sgl-workspace/sglang/python/sglang/srt/parser/reasoning_parser.py:195-279`

```python
class StepAudio2Detector(BaseReasoningFormatDetector):
    def __init__(self, stream_reasoning: bool = True, force_reasoning: bool = False):
        super().__init__(
            "<think>",
            "</think>",
            force_reasoning=False,  # Key fix: don't be greedy
            stream_reasoning=stream_reasoning,
        )
```

**Key Features**:
- `force_reasoning=False`: Only treats content between `<think></think>` as reasoning
- `_remove_protected_content()`: Skips tool call and TTS markers
- Proper null checking to avoid crashes

### 2. Fixed Parser Execution Order
**File**: `/sgl-workspace/sglang/python/sglang/srt/entrypoints/openai/serving_chat.py:957-1009`

**Old Order** (Broken):
```
Reasoning Parser (consumes everything) →
Tool Parser (gets nothing) →
Audio Parser (gets nothing)
```

**New Order** (Fixed):
```
Audio Parser (extracts TTS, returns remaining) →
Tool Parser (extracts tools, returns remaining) →
Reasoning Parser (processes remaining only)
```

### 3. Added `tts_output` Parameter
**File**: `/sgl-workspace/sglang/python/sglang/srt/entrypoints/openai/protocol.py:498-501`

```python
class ChatCompletionRequest(BaseModel):
    ...
    tts_output: bool = Field(
        default=False,
        description="Enable text-to-speech output"
    )
```

**Auto-append Logic** (`serving_chat.py:555-563`):
```python
if request.tts_output:
    tts_start_token = "<tts_start>"
    tts_start_id = self.tokenizer_manager.tokenizer.vocab.get(tts_start_token)
    if tts_start_id is not None:
        prompt_ids.append(tts_start_id)
```

### 4. Improved TTS Detection
**File**: `/sgl-workspace/sglang/python/sglang/srt/entrypoints/openai/serving_chat.py:110-135`

**Priority**:
1. Check `request.tts_output` parameter (recommended)
2. Fallback to checking if prompt ends with `<tts_start>` token (legacy)

## API Changes

### Before (Manual Token - Still Works)
```python
response = requests.post(url, json={
    "messages": [
        {"role": "user", "content": "Say hello"},
        {"role": "assistant", "content": "<tts_start>", "eot": False}
    ],
    ...
})
```

### After (Recommended - New Parameter)
```python
response = requests.post(url, json={
    "messages": [
        {"role": "user", "content": "Say hello"}
    ],
    "tts_output": True,  # Automatically adds <tts_start>
    ...
})
```

## Files Modified

1. **`reasoning_parser.py`**
   - Added `StepAudio2Detector` class (lines 195-279)
   - Registered detector for "step-audio2" (line 360)

2. **`serving_chat.py`**
   - Reordered parser execution (lines 957-1009)
   - Added auto-append logic for TTS token (lines 555-563)
   - Improved TTS detection (lines 110-135)
   - Updated text extraction logic (line 1009)

3. **`protocol.py`**
   - Added `tts_output` parameter (lines 498-501)

4. **`STEP_AUDIO2_INTEGRATION_GUIDE.md`**
   - Updated TTS examples with new parameter
   - Added API reference for `tts_output`
   - Updated troubleshooting section
   - Added "Recent Improvements" section

## Testing

### TTS Output Test
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "step-audio-2-mini",
    "messages": [{"role": "user", "content": "Hello"}],
    "tts_output": true,
    "max_tokens": 100
  }'
```

**Expected Result**:
```json
{
  "choices": [{
    "message": {
      "content": "remaining text",
      "tts_content": {
        "tts_text": "Hello! How can I help you?",
        "tts_audio": "<audio_0><audio_1>..."
      },
      "reasoning_content": null
    }
  }]
}
```

### Tool Call Test
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "step-audio-2-mini",
    "messages": [{"role": "user", "content": "What is the weather?"}],
    "tools": [...],
    "max_tokens": 100
  }'
```

**Expected Result**:
```json
{
  "choices": [{
    "message": {
      "content": "Let me check",
      "tool_calls": [{
        "function": {
          "name": "get_weather",
          "arguments": "{...}"
        }
      }],
      "reasoning_content": null
    }
  }]
}
```

## Benefits

1. **✅ Correct Content Separation**: Audio, tool calls, and reasoning are properly separated
2. **✅ Simpler API**: No need to manually add `<tts_start>` token
3. **✅ Backward Compatible**: Legacy method still works
4. **✅ Mixed Content Support**: Can handle responses with multiple content types
5. **✅ Better Error Messages**: Clear warnings when TTS token not found

## Limitations

1. **No Streaming for TTS**: Audio parsing only works in non-streaming mode
2. **Re-tokenization Overhead**: Output text is re-encoded for audio parsing
3. **Fixed Parser Order**: Cannot customize parser execution order per request

## Future Improvements

1. Add streaming support for audio parser
2. Access output token IDs directly (avoid re-tokenization)
3. Support mixed-mode outputs (reasoning + TTS simultaneously)
4. Add integration tests for all parser combinations
5. Optimize `_remove_protected_content()` performance

## Migration Guide

### For Users Currently Using Manual `<tts_start>`

**Option 1: Switch to `tts_output` (Recommended)**
```python
# Old
messages.append({"role": "assistant", "content": "<tts_start>", "eot": False})

# New
# Remove the assistant message, add tts_output=True
request["tts_output"] = True
```

**Option 2: Keep Using Manual Token**
- No changes needed
- Your existing code will continue to work

### For Server Operators

**Required**:
- Update SGLang to the version with these fixes
- Restart server (no config changes needed)

**Optional**:
- Update documentation to recommend `tts_output` parameter
- Add examples using the new parameter

## Verification

To verify the fix is working:

1. Check server logs for:
   ```
   TTS output enabled: appended <tts_start> token
   [TTS] is_tts_output: True, from_request: True
   ```

2. Check response has correct structure:
   - `tts_content` is populated (not null)
   - `reasoning_content` is null (unless using `separate_reasoning`)
   - Audio tokens are in `tts_content.tts_audio`, not in `content`

3. Tool calls are in `tool_calls` array, not in `reasoning_content`

## Support

For issues or questions:
- Check logs for parser initialization errors
- Verify server arguments include: `--audio-parser step_audio_2 --reasoning-parser step-audio2`
- Review `/sgl-workspace/sglang/STEP_AUDIO2_INTEGRATION_GUIDE.md`
