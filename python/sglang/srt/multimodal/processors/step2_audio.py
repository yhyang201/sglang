import re
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from transformers import BatchFeature
from transformers.audio_utils import mel_filter_bank, spectrogram, window_function
from transformers.feature_extraction_sequence_utils import SequenceFeatureExtractor
from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.utils import TensorType, logging

from sglang.srt.managers.schedule_batch import Modality
from sglang.srt.models.step2_audio import StepAudio2ForCausalLM
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)


def _mel_filters(sr, n_mels: int, n_fft: int) -> torch.Tensor:
    """Load the mel filterbank matrix for projecting STFT into a Mel spectrogram,
    using mel_filter_bank"""
    assert n_mels in {80, 128}, f"Unsupported n_mels: {n_mels}"

    mel_filters = mel_filter_bank(
        num_frequency_bins=1 + n_fft // 2,  # n_fft//2 + 1 = 201
        num_mel_filters=n_mels,  # 80 or 128
        min_frequency=0.0,
        max_frequency=16000 // 2,  # sr/2 = 8000
        sampling_rate=16000,
        norm="slaney",
        mel_scale="slaney",
    )
    return torch.from_numpy(mel_filters.astype(np.float32)).T


class StepAudio2Processor:

    _mel_filters_cache = {}

    def __init__(
        self,
        config,
        tokenizer,
    ) -> None:
        super().__init__()

        self.config = config
        self.tokenizer = tokenizer
        self.audio_token = "<audio_patch>"
        self.n_mels = 128
        self.max_chunk_size = 29  # from audio encoder position embedding length equals 1500, means 29.98s audio # noqa: E501
        self.sampling_rate = 16000
        self._mel_filters = _mel_filters(
            sr=self.sampling_rate, n_mels=self.n_mels, n_fft=400
        )
        # self._mel_filters = torch.from_numpy(
        #     librosa.filters.mel(sr=self.sampling_rate,
        #                         n_fft=400,
        #                         n_mels=self.n_mels))

    @property
    def audio_token_id(self) -> int:
        return self.tokenizer.get_vocab()[self.audio_token]

    def _log_mel_spectrogram(
        self,
        audio: np.ndarray,
        padding: int = 0,
    ):
        audio = F.pad(torch.from_numpy(audio.astype(np.float32)), (0, padding))
        window = torch.hann_window(400).to(audio.device)
        stft = torch.stft(audio, 400, 160, window=window, return_complex=True)
        magnitudes = stft[..., :-1].abs() ** 2
        filters = self._mel_filters
        mel_spec = filters @ magnitudes

        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        return log_spec.t()

    def preprocess_audio(self, audio_tensor: np.ndarray) -> torch.Tensor:
        return self._log_mel_spectrogram(audio_tensor, padding=479)

    def get_num_audio_tokens(self, max_feature_len: int) -> int:
        encoder_output_dim = (
            (max_feature_len + 1) // 2 // 2
        )  # from hych: align with log-to-mel padding 479
        padding = 1
        kernel_size = 3
        stride = 2
        adapter_output_dim = (
            encoder_output_dim + 2 * padding - kernel_size
        ) // stride + 1
        return adapter_output_dim

    def _get_audio_repl(
        self,
        audio_feat_len: int,
    ) -> tuple[str, list[int]]:
        num_audio_tokens = self.get_num_audio_tokens(audio_feat_len)
        text = (
            "<audio_start>" + "<audio_patch>" * num_audio_tokens + "<audio_end>"
        )  # noqa: E501
        token_ids = (
            [self.tokenizer.convert_tokens_to_ids("<audio_start>")]
            + [self.audio_token_id] * num_audio_tokens
            + [self.tokenizer.convert_tokens_to_ids("<audio_end>")]
        )
        return text, token_ids

    def replace_placeholder(self, text: str, placeholder: str, repls: list[str]) -> str:
        parts = text.split(placeholder)
        if len(parts) - 1 != len(repls):
            raise ValueError(
                "The number of placeholders does not match the number of replacements."  # noqa: E501
            )

        result = [parts[0]]
        for i, repl in enumerate(repls):
            result.append(repl)
            result.append(parts[i + 1])

        return "".join(result)

    def __call__(
        self,
        text: Optional[Union[str, list[str]]] = None,
        audios: Union[np.ndarray, list[np.ndarray]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ) -> BatchFeature:
        if text is None:
            text = []
        if not isinstance(text, list):
            text = [text]
        if audios is None:
            audios = []
        if not isinstance(audios, list):
            audios = [audios]

        if len(audios) == 0:
            audio_inputs = {}
            text_inputs = self.tokenizer(text)
        else:
            audio_mels_lst = []
            audio_repl_str_lst = []
            audio_repl_ids_lst = []
            for audio in audios:
                audio_mels = self.preprocess_audio(audio)
                audio_mels_lst.append(audio_mels)
                audio_repl_str, audio_repl_ids = self._get_audio_repl(
                    audio_mels.shape[0]
                )
                audio_repl_str_lst.append(audio_repl_str)
                audio_repl_ids_lst.extend(audio_repl_ids)
            audio_inputs = {
                "input_features": torch.concat(audio_mels_lst),
                "audio_lens": [audio_mels.shape[0] for audio_mels in audio_mels_lst],
            }

            text = [
                self.replace_placeholder(t, self.audio_token, audio_repl_str_lst)
                for t in text
            ]
            text_inputs = self.tokenizer(text)
        return BatchFeature(
            {
                **text_inputs,
                **audio_inputs,
            },
            tensor_type=return_tensors,
        )


################################################ SGLang Processor ################################################


class StepAudio2MultimodalProcessor(BaseMultimodalProcessor):
    models = [StepAudio2ForCausalLM]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        tokenizer = _processor
        _processor = StepAudio2Processor(hf_config, tokenizer)
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)
        self.AUDIO_TOKEN = "<audio_start><audio_patch><audio_end>"
        self.AUDIO_TOKEN_REGEX = re.compile(
            r"<audio_start>(?:<audio_patch>)+<audio_end>"
        )
        # Collect special token ids
        self.audio_start_id = tokenizer.convert_tokens_to_ids("<audio_start>")
        self.audio_token_id = tokenizer.convert_tokens_to_ids("<audio_patch>")
        self.audio_end_id = tokenizer.convert_tokens_to_ids("<audio_end>")

        self.mm_tokens = MultimodalSpecialTokens(
            audio_token=self.AUDIO_TOKEN,
            audio_token_regex=self.AUDIO_TOKEN_REGEX,
            audio_token_id=self.audio_token_id,
        ).build(_processor)

        self.ATTR_NAME_TO_MODALITY.update({"feature_attention_mask": Modality.AUDIO})

    async def process_mm_data_async(
        self,
        audio_data,
        input_text,
        **kwargs,
    ):
        base_output = self.load_mm_data(
            prompt=input_text,
            audio_data=audio_data,
            multimodal_tokens=self.mm_tokens,
        )
        if base_output is None:
            return None

        mm_items, input_ids, ret = self.process_and_combine_mm_data(
            base_output, self.mm_tokens
        )

        mm_items[0].audio_feature_lens = ret["audio_lens"]

        return {
            "mm_items": mm_items,
            "input_ids": input_ids.tolist(),
            "audio_start_id": self.audio_start_id,
            "audio_token_id": self.audio_token_id,
            "audio_end_id": self.audio_end_id,
        }
