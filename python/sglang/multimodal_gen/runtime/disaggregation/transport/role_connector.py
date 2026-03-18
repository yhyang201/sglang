# SPDX-License-Identifier: Apache-2.0
"""
Role connectors for disaggregated diffusion pipelines (pool mode).

Field definitions and helpers for extracting/applying Req fields between
pipeline roles:
  Encoder -> Denoiser: text embeddings, latents, timesteps, metadata
  Denoiser -> Decoder: denoised latents, metadata
"""

import logging

import torch

logger = logging.getLogger(__name__)


# --- Field definitions for each role transition ---

# Tensor fields produced by Encoder stages, consumed by Denoiser
ENCODER_TO_DENOISER_TENSOR_FIELDS = [
    "prompt_embeds",
    "negative_prompt_embeds",
    "pooled_embeds",
    "neg_pooled_embeds",
    "prompt_attention_mask",
    "negative_attention_mask",
    "clip_embedding_pos",
    "clip_embedding_neg",
    "latents",
    "timesteps",
    "y",
    "image_latent",
    "latent_ids",
    # Audio (LTX-2)
    "audio_prompt_embeds",
    "negative_audio_prompt_embeds",
    "audio_latents",
    "audio_noise",
]

# Scalar fields from Encoder that Denoiser needs
ENCODER_TO_DENOISER_SCALAR_FIELDS = [
    "request_id",
    "do_classifier_free_guidance",
    "guidance_scale",
    "guidance_scale_2",
    "height",
    "width",
    "num_frames",
    "fps",
    "num_inference_steps",
    "eta",
    "sigmas",
    "n_tokens",
    "height_latents",
    "width_latents",
    "raw_latent_shape",
    "raw_audio_latent_shape",
    "seed",
    "seeds",
    "is_warmup",
    "is_prompt_processed",
    "generate_audio",
    "output_file_ext",
    # STA/VSA
    "STA_param",
    "is_cfg_negative",
    "mask_search_final_result_pos",
    "mask_search_final_result_neg",
    "VSA_sparsity",
]

# Tensor fields produced by Denoiser, consumed by Decoder
DENOISER_TO_DECODER_TENSOR_FIELDS = [
    "latents",
    "audio_latents",
    "noise_pred",
]

# Scalar fields from Denoiser that Decoder needs
DENOISER_TO_DECODER_SCALAR_FIELDS = [
    "request_id",
    "height",
    "width",
    "num_frames",
    "raw_latent_shape",
    "raw_audio_latent_shape",
    "is_warmup",
    "output_file_ext",
    "generate_audio",
    # Error propagation: set by denoiser when forward() fails
    "_disagg_error",
]


def _extract_tensor_fields(req, field_names: list[str]) -> dict:
    """Extract tensor fields from a Req object."""
    result = {}
    for name in field_names:
        value = getattr(req, name, None)
        if value is not None:
            result[name] = value
    return result


def _extract_scalar_fields(req, field_names: list[str]) -> dict:
    """Extract scalar fields from a Req, converting to JSON-serializable types."""
    result = {}
    for name in field_names:
        value = getattr(req, name, None)
        if value is None:
            continue
        # Convert torch-specific types to JSON-serializable
        if isinstance(value, torch.Tensor):
            # Some "scalar" fields may actually be small tensors (e.g., raw_latent_shape)
            result[name] = value.tolist()
        elif isinstance(value, (list, tuple)):
            # Convert any tensor items in lists
            converted = []
            for item in value:
                if isinstance(item, torch.Tensor):
                    converted.append(item.tolist())
                else:
                    converted.append(item)
            result[name] = converted
        else:
            result[name] = value
    return result


def _apply_scalar_fields(req, scalar_fields: dict, field_names: list[str]):
    """Apply scalar fields to a Req object."""
    for name in field_names:
        if name in scalar_fields:
            setattr(req, name, scalar_fields[name])


def _apply_tensor_fields(req, tensor_fields: dict) -> None:
    """Apply tensor fields to a Req object."""
    for name, value in tensor_fields.items():
        setattr(req, name, value)
