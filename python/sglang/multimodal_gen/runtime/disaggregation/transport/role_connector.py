# SPDX-License-Identifier: Apache-2.0
"""Role connectors: automatic field extraction for transferring Req state between roles.

Transfers all non-None fields except a small exclude set, so new model fields
are automatically covered without manual updates.
"""

import dataclasses
import json
import logging

import torch

logger = logging.getLogger(__name__)

# Fields that should never be transferred (non-serializable, internal, or receiver rebuilds)
_EXCLUDE_FIELDS = frozenset(
    {
        "sampling_params",
        "generator",
        "modules",
        "metrics",
        "extra_step_kwargs",
        "extra",
        "condition_image",
        "vae_image",
        "pixel_values",
        "preprocessed_image",
        "image_embeds",
        "original_condition_image_size",
        "vae_image_sizes",
        "output",
        "audio",
        "audio_sample_rate",
        "trajectory_timesteps",
        "trajectory_latents",
        "trajectory_audio_latents",
        "timestep",
        "step_index",
        "prompt_template",
        "max_sequence_length",
    }
)

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
    if isinstance(value, torch.Tensor):
        return True
    if isinstance(value, list) and value and isinstance(value[0], torch.Tensor):
        return True
    return False


def _to_json_serializable(value):
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
    if field_info.default is not dataclasses.MISSING:
        return value == field_info.default
    if field_info.default_factory is not dataclasses.MISSING:
        if isinstance(value, (list, dict)) and len(value) == 0:
            return True
    return False


def _extract_extra_fields(extra: dict, scalar_fields: dict) -> None:
    """Extract JSON-serializable entries from Req.extra into scalar_fields."""
    for key, value in extra.items():
        if key.startswith("_"):
            continue
        try:
            json.dumps(value)
            scalar_fields[f"_extra_{key}"] = value
        except (TypeError, ValueError, OverflowError):
            pass


def extract_transfer_fields(req) -> tuple[dict, dict]:
    """Extract all transferable fields from a Req, split into tensors and scalars."""
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
                pass

    extra = getattr(req, "extra", None)
    if extra:
        _extract_extra_fields(extra, scalar_fields)

    sp = getattr(req, "sampling_params", None)
    if sp is not None:
        for name in _SAMPLING_PARAMS_FIELDS:
            if name in scalar_fields:
                continue
            value = getattr(sp, name, None)
            if value is not None:
                scalar_fields[name] = _to_json_serializable(value)

    return tensor_fields, scalar_fields
