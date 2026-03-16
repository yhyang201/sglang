# SPDX-License-Identifier: Apache-2.0
"""
Role connectors for disaggregated diffusion pipelines (pool mode).

Handles serialization/deserialization of Req fields between pipeline roles:
  Encoder -> Denoiser: text embeddings, latents, timesteps, metadata
  Denoiser -> Decoder: denoised latents, metadata

Pool mode uses DiffusionServer as a relay: role instances pack/unpack
tensor data via pack_encoder_output / pack_denoiser_output / build_req_from_frames.
"""

import logging

import torch

from sglang.multimodal_gen.runtime.disaggregation.transport.relay.tensor_transport import (
    pack_tensors,
    unpack_tensors,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import (
    Req,
)

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


def _extract_tensor_fields(req: Req, field_names: list[str]) -> dict:
    """Extract tensor fields from a Req object."""
    result = {}
    for name in field_names:
        value = getattr(req, name, None)
        if value is not None:
            result[name] = value
    return result


def _extract_scalar_fields(req: Req, field_names: list[str]) -> dict:
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


def _apply_scalar_fields(req: Req, scalar_fields: dict, field_names: list[str]):
    """Apply scalar fields to a Req object."""
    for name in field_names:
        if name in scalar_fields:
            setattr(req, name, scalar_fields[name])


def _apply_tensor_fields(req: Req, tensor_fields: dict):
    """Apply tensor fields to a Req object."""
    for name, value in tensor_fields.items():
        setattr(req, name, value)


# --- Pool mode helpers (DiffusionServer-mediated transfers) ---


def pack_encoder_output(req: Req) -> tuple[bytes, list]:
    """Pack encoder output for relay via DiffusionServer.

    Returns (metadata_bytes, buffers) ready for send_multipart.
    """
    tensor_fields = _extract_tensor_fields(req, ENCODER_TO_DENOISER_TENSOR_FIELDS)
    scalar_fields = _extract_scalar_fields(req, ENCODER_TO_DENOISER_SCALAR_FIELDS)
    return pack_tensors(tensor_fields, scalar_fields)


def pack_denoiser_output(req: Req) -> tuple[bytes, list]:
    """Pack denoiser output for relay via DiffusionServer."""
    tensor_fields = _extract_tensor_fields(req, DENOISER_TO_DECODER_TENSOR_FIELDS)
    scalar_fields = _extract_scalar_fields(req, DENOISER_TO_DECODER_SCALAR_FIELDS)
    return pack_tensors(tensor_fields, scalar_fields)


def build_req_from_frames_async(
    parts: list,
    transition: str,
    device: str | torch.device = "cpu",
    stream: torch.cuda.Stream | None = None,
) -> tuple[Req, torch.cuda.Event | None]:
    """Build a Req from multipart ZMQ frames with async H2D on the given stream.

    Returns (req, h2d_event). Caller must wait on h2d_event before using
    tensor fields on the default/compute stream.
    """
    tensor_fields, scalar_fields = unpack_tensors(parts, device="cpu")

    if transition == "encoder_to_denoiser":
        scalar_field_names = ENCODER_TO_DENOISER_SCALAR_FIELDS
    elif transition == "denoiser_to_decoder":
        scalar_field_names = DENOISER_TO_DECODER_SCALAR_FIELDS
    else:
        raise ValueError(f"Unknown transition: {transition}")

    # H2D on transfer stream (non-blocking)
    h2d_event = None
    if stream is not None and device != "cpu" and device != torch.device("cpu"):
        with torch.cuda.stream(stream):
            for name in list(tensor_fields.keys()):
                val = tensor_fields[name]
                if isinstance(val, torch.Tensor):
                    tensor_fields[name] = val.to(device, non_blocking=True)
                elif isinstance(val, list):
                    tensor_fields[name] = [
                        (
                            t.to(device, non_blocking=True)
                            if isinstance(t, torch.Tensor)
                            else t
                        )
                        for t in val
                    ]
            h2d_event = torch.cuda.Event()
            h2d_event.record()
    elif device != "cpu" and device != torch.device("cpu"):
        for name in list(tensor_fields.keys()):
            val = tensor_fields[name]
            if isinstance(val, torch.Tensor):
                tensor_fields[name] = val.to(device)
            elif isinstance(val, list):
                tensor_fields[name] = [
                    t.to(device) if isinstance(t, torch.Tensor) else t for t in val
                ]

    # Build Req (CPU work, overlapped with H2D)
    init_kwargs = {}
    if "request_id" in scalar_fields:
        init_kwargs["request_id"] = scalar_fields["request_id"]
    if "guidance_scale" in scalar_fields:
        init_kwargs["guidance_scale"] = scalar_fields["guidance_scale"]

    req = Req(**init_kwargs)
    _apply_scalar_fields(req, scalar_fields, scalar_field_names)
    _apply_tensor_fields(req, tensor_fields)

    # Recreate torch.Generator from seed
    seed = scalar_fields.get("seed")
    if seed is not None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        req.generator = generator

    return req, h2d_event


def build_req_from_frames(
    parts: list,
    transition: str,
    device: str | torch.device = "cpu",
) -> Req:
    """Build a Req from multipart ZMQ frames (received via relay).

    Args:
        parts: ZMQ multipart frames (metadata JSON + tensor buffers)
        transition: "encoder_to_denoiser" or "denoiser_to_decoder"
        device: target device for tensors
    """
    tensor_fields, scalar_fields = unpack_tensors(parts, device=device)

    if transition == "encoder_to_denoiser":
        scalar_field_names = ENCODER_TO_DENOISER_SCALAR_FIELDS
    elif transition == "denoiser_to_decoder":
        scalar_field_names = DENOISER_TO_DECODER_SCALAR_FIELDS
    else:
        raise ValueError(f"Unknown transition: {transition}")

    # Build Req
    init_kwargs = {}
    if "request_id" in scalar_fields:
        init_kwargs["request_id"] = scalar_fields["request_id"]
    if "guidance_scale" in scalar_fields:
        init_kwargs["guidance_scale"] = scalar_fields["guidance_scale"]

    req = Req(**init_kwargs)
    _apply_scalar_fields(req, scalar_fields, scalar_field_names)
    _apply_tensor_fields(req, tensor_fields)

    # Recreate torch.Generator from seed
    seed = scalar_fields.get("seed")
    if seed is not None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        req.generator = generator

    return req
