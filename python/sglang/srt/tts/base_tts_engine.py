"""Base TTS Engine Abstract Class"""

from abc import ABC, abstractmethod
from typing import Any, List, Optional


class BaseTTSEngine(ABC):
    """
    Abstract base class for TTS (Text-to-Speech) engines.

    All TTS engine implementations should inherit from this class
    and implement the required abstract methods.
    """

    @abstractmethod
    def __init__(self, model_path: str, **kwargs):
        """
        Initialize the TTS engine.

        Args:
            model_path: Path to the TTS model
            **kwargs: Additional engine-specific parameters
        """
        pass

    @abstractmethod
    def generate(
        self,
        audio_tokens: List[int],
        prompt_wav: Optional[str] = None,
        **kwargs
    ) -> bytes:
        """
        Generate audio from audio tokens.

        Args:
            audio_tokens: List of audio token IDs
            prompt_wav: Optional path to prompt audio file for voice cloning
            **kwargs: Additional generation parameters

        Returns:
            bytes: Generated audio data in WAV format
        """
        pass

    @abstractmethod
    def load_model(self):
        """Load the TTS model into memory."""
        pass

    @abstractmethod
    def unload_model(self):
        """Unload the TTS model to free memory."""
        pass

    def __call__(self, audio_tokens: List[int], prompt_wav: Optional[str] = None, **kwargs) -> bytes:
        """Convenience method to call generate()."""
        return self.generate(audio_tokens, prompt_wav, **kwargs)
