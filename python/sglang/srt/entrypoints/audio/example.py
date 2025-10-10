"""
Audio Server Integration Example

This example demonstrates how to initialize and use the AudioServer
with SGLang Engine and Step-Audio2 TTS.
"""

import asyncio
import logging

from sglang.srt.entrypoints.audio.audio_server import AudioServer
from sglang.srt.entrypoints.engine import Engine
from sglang.srt.tts.step_audio2.step_audio2_tts import StepAudio2TTS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    """
    Example usage of AudioServer with SGLang and Step-Audio2 TTS.
    """

    # Step 1: Initialize SGLang Engine
    logger.info("Initializing SGLang Engine...")
    sglang_engine = Engine(
        model_path="stepfun-ai/Step-Audio-2-mini",
        # TODO: Add appropriate server args
        # mem_fraction_static=0.7,  # Reserve memory for TTS
    )

    # Step 2: Initialize TTS Engine
    logger.info("Initializing Step-Audio2 TTS Engine...")
    tts_engine = StepAudio2TTS(
        model_path="Step-Audio-2-mini/token2wav",
        float16=True,  # Use FP16 for efficiency
    )

    # Step 3: Initialize AudioServer
    logger.info("Initializing AudioServer...")
    audio_server = AudioServer(
        sglang_engine=sglang_engine,
        tts_engine=tts_engine,
        default_prompt_wav="assets/default_female.wav",  # TODO: Update path
    )

    # Step 4: Test audio generation
    logger.info("Testing audio generation...")

    # Example 1: Simple text prompt
    result = await audio_server.generate_audio_response(
        messages="Please introduce yourself.",
        sampling_params={
            "temperature": 0.7,
            "max_new_tokens": 1024,
        }
    )

    logger.info(f"Generated text: {result['text']}")
    logger.info(f"Generated {len(result['audio'])} bytes of audio")
    logger.info(f"Audio tokens: {result['audio_tokens'][:10]}...")  # First 10 tokens

    # Example 2: Chat messages format
    # TODO: Implement proper message format handling
    # messages = [
    #     {"role": "system", "content": "You are a helpful assistant."},
    #     {"role": "user", "content": "Tell me about the Great Wall."},
    # ]
    # result = await audio_server.generate_audio_response(messages=messages)

    # Step 5: Cleanup
    logger.info("Cleaning up...")
    audio_server.shutdown()

    logger.info("Example completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())


# ============================================================================
# Integration with FastAPI (for production)
# ============================================================================

"""
from fastapi import FastAPI, HTTPException
from sglang.srt.entrypoints.openai.serving_audio import OpenAIServingAudio
from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest

app = FastAPI()

# Initialize components (do this at startup)
# sglang_engine = Engine(...)
# tts_engine = StepAudio2TTS(...)
# audio_server = AudioServer(sglang_engine, tts_engine)
# audio_handler = OpenAIServingAudio(tokenizer_manager, template_manager, audio_server)

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    '''
    OpenAI-compatible chat completions with optional audio support.

    If request.modalities includes 'audio', generates audio output.
    '''

    # Check if audio modality is requested
    has_audio_modality = (
        hasattr(request, 'modalities') and
        request.modalities and
        'audio' in request.modalities
    )

    if has_audio_modality:
        # Use audio handler
        return await audio_handler.create_chat_completion_with_audio(request)
    else:
        # Use regular text handler
        # return await regular_handler.create_chat_completion(request)
        pass

# TODO: Add dedicated audio endpoints
# @app.post("/v1/audio/speech")
# async def create_speech(...):
#     pass
"""
