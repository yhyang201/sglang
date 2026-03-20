# SPDX-License-Identifier: Apache-2.0
"""
Role connectors for disaggregated diffusion pipelines (pool mode).

Automatic field extraction for transferring Req state between pipeline roles.
Instead of maintaining per-transition field whitelists, we transfer all
non-None fields except a small stable exclude set of non-serializable or
internal-only fields. This ensures new model fields are automatically
covered without manual updates.
"""

import dataclasses
import json
import logging

import torch

logger = logging.getLogger(__name__)


# Fields that should never be transferred between roles.
# Reasons: non-serializable, internal-only, or receiver rebuilds them locally.
_EXCLUDE_FIELDS = frozenset(
    {
        # Non-serializable / receiver rebuilds locally
        "sampling_params",  # scalar fields extracted separately
        "generator",  # rebuilt from seed on receiver
        "modules",  # pipeline-internal references
        "metrics",  # receiver creates its own
        "extra_step_kwargs",  # scheduler-internal state
        "extra",  # mixed bag; serializable items extracted separately
        # Raw image inputs (already consumed by encoder, not needed downstream)
        "condition_image",  # PIL.Image
        "vae_image",  # PIL.Image
        "pixel_values",  # PIL.Image or preprocessed
        "preprocessed_image",  # encoder-internal intermediate
        "image_embeds",  # encoder-internal; downstream gets prompt_embeds
        "original_condition_image_size",
        "vae_image_sizes",
        # Final outputs (only produced by decoder, never transferred)
        "output",
        "audio",
        "audio_sample_rate",
        # Trajectory tracking (intermediate debug state)
        "trajectory_timesteps",
        "trajectory_latents",
        "trajectory_audio_latents",
        # Denoising loop internal state (not needed across roles)
        "timestep",  # current single timestep (loop variable)
        "step_index",  # current step index (loop variable)
        "prompt_template",
        "max_sequence_length",
    }
)

# SamplingParams fields to include in scalar extraction.
# These are commonly needed by downstream roles.
_SAMPLING_PARAMS_FIELDS = [
    "request_id",
    "guidance_scale",
    "guidance_scale_2",
    "height",
    "width",
    "num_frames",
    "fps",
    "num_inference_steps",
    "seed",
]


def _is_tensor_like(value) -> bool:
    """Check if a value is a tensor or a non-empty list of tensors."""
    if isinstance(value, torch.Tensor):
        return True
    if isinstance(value, list) and value and isinstance(value[0], torch.Tensor):
        return True
    return False


def _to_json_serializable(value):
    """Convert a value to a JSON-serializable form."""
    if isinstance(value, torch.Tensor):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        converted = []
        for item in value:
            if isinstance(item, torch.Tensor):
                converted.append(item.tolist())
            else:
                converted.append(item)
        return converted
    return value


def _is_default(value, field_info) -> bool:
    """Check if a value equals the dataclass field's default."""
    if field_info.default is not dataclasses.MISSING:
        return value == field_info.default
    if field_info.default_factory is not dataclasses.MISSING:
        # For mutable defaults (list, dict), check emptiness
        if isinstance(value, (list, dict)) and len(value) == 0:
            return True
    return False


def _extract_extra_fields(extra: dict, scalar_fields: dict) -> None:
    """Extract JSON-serializable entries from Req.extra into scalar_fields.

    The `extra` dict may contain a mix of serializable values (e.g., mu=float)
    and non-serializable objects (e.g., mesh, renderer). We try each entry
    individually and skip failures.
    """
    for key, value in extra.items():
        if key.startswith("_"):
            continue  # skip private/internal keys
        try:
            json.dumps(value)  # test serializability
            scalar_fields[f"_extra_{key}"] = value
        except (TypeError, ValueError, OverflowError):
            pass


def extract_transfer_fields(req) -> tuple[dict, dict]:
    """Extract all transferable fields from a Req, split into tensors and scalars.

    Automatically discovers fields by scanning the Req dataclass.
    Skips None values, default (empty) values, and excluded fields.

    Returns:
        (tensor_fields, scalar_fields) where tensor_fields contains
        torch.Tensor or list[torch.Tensor] values, and scalar_fields
        contains JSON-serializable values.
    """
    tensor_fields = {}
    scalar_fields = {}

    for f in dataclasses.fields(req):
        if f.name in _EXCLUDE_FIELDS:
            continue

        value = getattr(req, f.name, None)
        if value is None:
            continue
        if _is_default(value, f):
            continue

        if _is_tensor_like(value):
            tensor_fields[f.name] = value
        else:
            try:
                scalar_fields[f.name] = _to_json_serializable(value)
            except (TypeError, ValueError):
                # Skip non-serializable values silently
                pass

    # Extract serializable entries from Req.extra (e.g., mu for Z-Image)
    extra = getattr(req, "extra", None)
    if extra:
        _extract_extra_fields(extra, scalar_fields)

    # Also extract key fields from sampling_params
    sp = getattr(req, "sampling_params", None)
    if sp is not None:
        for name in _SAMPLING_PARAMS_FIELDS:
            if name in scalar_fields:
                continue  # already extracted from Req directly
            value = getattr(sp, name, None)
            if value is not None:
                scalar_fields[name] = _to_json_serializable(value)

    return tensor_fields, scalar_fields
