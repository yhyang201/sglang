"""Audio Token Parser

This module provides functionality to parse and extract audio tokens from model output text.
Audio tokens are typically in the format <audio_XXX> where XXX is the token ID.
"""

import logging
import re
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


class AudioTokenParser:
    """
    Parser for extracting audio tokens from model output.

    Similar to FunctionCallParser, this class handles both streaming and
    non-streaming parsing of audio tokens from LLM outputs.

    Audio tokens are expected in the format: <audio_123>
    """

    # Regex pattern to match audio tokens like <audio_123>
    AUDIO_TOKEN_PATTERN = re.compile(r"<audio_(\d+)>")

    def __init__(self):
        """Initialize the AudioTokenParser."""
        self.stream_buffer = ""
        logger.debug("AudioTokenParser initialized")

    def has_audio_tokens(self, text: str) -> bool:
        """
        Check if the given text contains audio tokens.

        Args:
            text: The text to check for audio tokens

        Returns:
            True if the text contains audio tokens, False otherwise
        """
        return bool(self.AUDIO_TOKEN_PATTERN.search(text))

    def parse_non_stream(self, full_text: str) -> Tuple[str, List[int]]:
        """
        Parse the complete text to extract audio tokens.

        Args:
            full_text: The complete text to parse

        Returns:
            A tuple containing:
            - The text with audio tokens removed
            - A list of audio token IDs extracted from the text
        """
        # Find all audio tokens
        matches = self.AUDIO_TOKEN_PATTERN.findall(full_text)
        audio_tokens = [int(token_id) for token_id in matches]

        # Remove audio tokens from text
        clean_text = self.AUDIO_TOKEN_PATTERN.sub("", full_text)

        logger.debug(f"Parsed {len(audio_tokens)} audio tokens from text")
        return clean_text, audio_tokens

    def parse_stream_chunk(self, chunk_text: str) -> Tuple[str, List[int]]:
        """
        Streaming incremental parsing of text chunks as they arrive.

        Args:
            chunk_text: The new chunk of text to parse

        Returns:
            A tuple containing:
            - Normal text (with complete audio tokens removed)
            - List of audio token IDs extracted from this chunk

        # TODO: Implement proper streaming logic that handles partial tokens
        # For now, we use a simple buffer approach
        """
        # Add chunk to buffer
        self.stream_buffer += chunk_text

        # Extract complete audio tokens from buffer
        clean_text, audio_tokens = self.parse_non_stream(self.stream_buffer)

        # Update buffer to keep only unparsed text
        # TODO: Improve this to handle partial tokens at buffer end
        self.stream_buffer = clean_text

        # Return the chunk processed
        # For now, return empty for normal text (audio tokens are extracted)
        # TODO: Properly separate normal text from audio tokens in streaming
        return "", audio_tokens

    def reset_stream(self):
        """Reset the streaming buffer."""
        self.stream_buffer = ""
        logger.debug("Stream buffer reset")

    @staticmethod
    def extract_audio_tokens_simple(text: str) -> List[int]:
        """
        Simple static method to extract audio tokens without state.

        Useful for one-off parsing without creating a parser instance.

        Args:
            text: The text containing audio tokens

        Returns:
            List of audio token IDs
        """
        matches = AudioTokenParser.AUDIO_TOKEN_PATTERN.findall(text)
        return [int(token_id) for token_id in matches]
