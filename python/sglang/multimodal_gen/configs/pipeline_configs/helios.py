# SPDX-License-Identifier: Apache-2.0
from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from sglang.multimodal_gen.configs.models import DiTConfig, EncoderConfig, VAEConfig
from sglang.multimodal_gen.configs.models.dits.helios import HeliosConfig
from sglang.multimodal_gen.configs.models.encoders import BaseEncoderOutput, T5Config
from sglang.multimodal_gen.configs.models.encoders.t5 import T5ArchConfig
from sglang.multimodal_gen.configs.models.vaes import WanVAEConfig
from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ModelTaskType,
    PipelineConfig,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


def umt5_postprocess_text(outputs: BaseEncoderOutput, _text_inputs) -> torch.Tensor:
    """Post-process UMT5 text encoder outputs, padding to 226 tokens."""
    mask: torch.Tensor = outputs.attention_mask
    hidden_state: torch.Tensor = outputs.last_hidden_state
    seq_lens = mask.gt(0).sum(dim=1).long()
    assert torch.isnan(hidden_state).sum() == 0
    prompt_embeds = [u[:v] for u, v in zip(hidden_state, seq_lens, strict=True)]
    prompt_embeds_tensor: torch.Tensor = torch.stack(
        [
            torch.cat([u, u.new_zeros(226 - u.size(0), u.size(1))])
            for u in prompt_embeds
        ],
        dim=0,
    )
    return prompt_embeds_tensor


@dataclass
class HeliosT2VConfig(PipelineConfig):
    """Configuration for the Helios T2V pipeline."""

    task_type: ModelTaskType = ModelTaskType.T2V

    # DiT
    dit_config: DiTConfig = field(default_factory=HeliosConfig)

    # VAE (same as Wan)
    vae_config: VAEConfig = field(default_factory=WanVAEConfig)
    vae_tiling: bool = False
    vae_sp: bool = False

    # Denoising stage
    flow_shift: float | None = 1.0

    # Text encoding stage (UMT5 is T5-compatible, override text_len to 226)
    text_encoder_configs: tuple[EncoderConfig, ...] = field(
        default_factory=lambda: (T5Config(arch_config=T5ArchConfig(text_len=226)),)
    )
    postprocess_text_funcs: tuple[Callable[[BaseEncoderOutput], torch.Tensor], ...] = (
        field(default_factory=lambda: (umt5_postprocess_text,))
    )

    # Precision for each component
    precision: str = "bf16"
    vae_precision: str = "fp32"
    text_encoder_precisions: tuple[str, ...] = field(default_factory=lambda: ("fp32",))

    # Helios-specific chunked denoising params
    num_latent_frames_per_chunk: int = 9
    history_sizes: list[int] = field(default_factory=lambda: [16, 2, 1])
    is_cfg_zero_star: bool = False
    zero_steps: int = 1
    keep_first_frame: bool = True

    def __post_init__(self):
        self.vae_config.load_encoder = False
        self.vae_config.load_decoder = True
