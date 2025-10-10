"""
Audio Server - Manages SGLang Engine and TTS Engine integration

This server coordinates between the text generation engine (SGLang) and
the audio generation engine (TTS) to provide audio output capabilities.
"""

import base64
import logging
from typing import Any, Dict, List, Optional, Union

from sglang.srt.entrypoints.engine import Engine
from sglang.srt.parser.audio_token_parser import AudioTokenParser
from sglang.srt.tts.base_tts_engine import BaseTTSEngine

logger = logging.getLogger(__name__)


class AudioServer:
    """
    Audio Server that orchestrates text generation and audio synthesis.

    This class manages:
    1. SGLang Engine for text/token generation
    2. TTS Engine for audio synthesis from tokens
    3. Audio token parsing from generation output
    """

    def __init__(
        self,
        sglang_engine: Engine,
        tts_engine: BaseTTSEngine,
        default_prompt_wav: Optional[str] = None,
    ):
        """
        Initialize AudioServer.

        Args:
            sglang_engine: The SGLang Engine instance for text generation
            tts_engine: The TTS Engine instance for audio synthesis
            default_prompt_wav: Default prompt audio path for voice cloning
        """
        self.sglang_engine = sglang_engine
        self.tts_engine = tts_engine
        self.default_prompt_wav = default_prompt_wav
        self.audio_parser = AudioTokenParser()

        logger.info("AudioServer initialized")
        logger.info(f"SGLang Engine: {type(sglang_engine).__name__}")
        logger.info(f"TTS Engine: {type(tts_engine).__name__}")

    async def generate_audio_response(
        self,
        messages: Union[List[Dict], str],
        prompt_wav: Optional[str] = None,
        sampling_params: Optional[Dict] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate text and audio response from messages.

        Args:
            messages: Chat messages or text prompt
            prompt_wav: Optional prompt audio path for voice cloning
            sampling_params: Sampling parameters for text generation
            **kwargs: Additional parameters

        Returns:
            Dict containing:
                - text: Generated text (without audio tokens)
                - audio: Audio bytes in WAV format
                - audio_base64: Base64 encoded audio
                - audio_tokens: List of audio token IDs (for debugging)
                - raw_text: Original text with audio tokens
        """
        logger.info(f"Generating audio response for messages: {messages}")

        # Use default prompt if not provided
        if prompt_wav is None:
            prompt_wav = self.default_prompt_wav
            logger.debug(f"Using default prompt_wav: {prompt_wav}")

        # TODO: Handle different message formats (OpenAI chat format vs simple text)
        # For now, assume messages is a simple text or list of messages

        # Step 1: Generate tokens using SGLang Engine
        logger.debug("Step 1: Generating text with SGLang Engine")
        try:
            generation_result = await self.sglang_engine.async_generate(
                prompt=messages,
                sampling_params=sampling_params or {},
                **kwargs
            )
            logger.debug(f"Generation result: {generation_result}")
        except Exception as e:
            logger.error(f"Failed to generate text: {e}")
            raise RuntimeError(f"Text generation failed: {e}") from e

        # Extract text from result
        # TODO: Handle different result formats (dict vs string)
        if isinstance(generation_result, dict):
            raw_text = generation_result.get("text", "")
        else:
            raw_text = str(generation_result)

        logger.debug(f"Generated raw text: {raw_text[:100]}...")

        # Step 2: Extract audio tokens from generated text
        logger.debug("Step 2: Extracting audio tokens")
        clean_text, audio_tokens = self.audio_parser.parse_non_stream(raw_text)

        logger.info(f"Extracted {len(audio_tokens)} audio tokens")
        logger.debug(f"Clean text: {clean_text[:100]}...")

        # Step 3: Generate audio from tokens using TTS Engine
        audio_bytes = b""
        if audio_tokens:
            logger.debug("Step 3: Generating audio with TTS Engine")
            try:
                audio_bytes = self.tts_engine.generate(
                    audio_tokens=audio_tokens,
                    prompt_wav=prompt_wav
                )
                logger.info(f"Generated {len(audio_bytes)} bytes of audio")
            except Exception as e:
                logger.error(f"Failed to generate audio: {e}")
                # TODO: Decide whether to fail the request or return text-only response
                logger.warning("Continuing with text-only response")
        else:
            logger.warning("No audio tokens found in generation output")

        # Step 4: Encode audio to base64 for API response
        audio_base64 = ""
        if audio_bytes:
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
            logger.debug(f"Encoded audio to base64: {len(audio_base64)} chars")

        # TODO: Add caching for prompt_wav embeddings
        # TODO: Add metrics/monitoring
        # TODO: Add streaming support

        result = {
            "text": clean_text,
            "audio": audio_bytes,
            "audio_base64": audio_base64,
            "audio_tokens": audio_tokens,
            "raw_text": raw_text,
        }

        logger.info("Audio response generation completed")
        return result

    def generate_audio_response_sync(
        self,
        messages: Union[List[Dict], str],
        prompt_wav: Optional[str] = None,
        sampling_params: Optional[Dict] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Synchronous version of generate_audio_response.

        # TODO: Implement proper sync version
        For now, this is a placeholder that raises NotImplementedError.
        """
        raise NotImplementedError(
            "Synchronous generation not yet implemented. Use async version."
        )

    def shutdown(self):
        """
        Shutdown the AudioServer and cleanup resources.
        """
        logger.info("Shutting down AudioServer")

        # Unload TTS model
        if hasattr(self.tts_engine, 'unload_model'):
            logger.debug("Unloading TTS model")
            self.tts_engine.unload_model()

        # Shutdown SGLang engine
        if hasattr(self.sglang_engine, 'shutdown'):
            logger.debug("Shutting down SGLang engine")
            self.sglang_engine.shutdown()

        logger.info("AudioServer shutdown complete")

    # TODO: Add method for streaming generation
    # async def generate_audio_response_stream(self, ...):
    #     """Generate audio response in streaming mode"""
    #     pass

    # TODO: Add method for batch processing
    # async def generate_audio_batch(self, batch_messages: List[...]):
    #     """Generate audio for batch of messages"""
    #     pass
