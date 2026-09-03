# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Rust-accelerated MiniMax M3 image preprocessing (sglang-mm `minimax_m3`).

Drop-in replacement for the HF ``MiniMaxM3VLImageProcessor`` (torchvision
path), bit-exact against it: smart_resize → uint8 antialias bicubic → fused
rescale+normalize → patchify (HF flatten order), all in Rust with the GIL
released and batch-level parallelism. Enabled per server by
``SGLANG_MINIMAX_M3_RS_MM_PREPROCESS=1`` (see
``processors/minimax_m3_vl.py``).

Images reach this processor already decoded (sglang loads them to PIL), so
both paths start from identical pixels and stay bit-identical regardless of
source format; the Rust crate's own bytes→decode entry
(``minimax_m3.preprocess``) is what the parity tests and the pure-Rust
pipeline drive.
"""

from __future__ import annotations

import json
from typing import List, Optional, Union

import numpy as np
import torch
from transformers.image_processing_utils import BaseImageProcessor, BatchFeature
from transformers.image_utils import ImageInput

from sglang.srt.rust_extensions import load_rust_extension

_rs = load_rust_extension("sglang.srt.rust_extensions._multimodal").minimax_m3

# The HF processor never exposes min_pixels as an attribute; it is the
# smart_resize function default (4 * 28 * 28), passed explicitly here.
_DEFAULT_MIN_PIXELS = 4 * 28 * 28


def _to_hwc_u8(image) -> np.ndarray:
    """One loaded image (PIL / numpy / torch) as a C-contiguous HWC u8 RGB array."""
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    if not isinstance(image, np.ndarray):
        # PIL Image; the HF processor's do_convert_rgb equivalent.
        if getattr(image, "mode", "RGB") != "RGB":
            image = image.convert("RGB")
        image = np.asarray(image)
    if image.ndim == 3 and image.shape[0] == 3 and image.shape[-1] != 3:
        image = image.transpose(1, 2, 0)  # CHW -> HWC
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    if image.dtype != np.uint8:
        raise ValueError(
            f"MiniMaxM3RustImageProcessor expects uint8 images, got {image.dtype}; "
            "the Rust pipeline reproduces the uint8 torchvision path bit-exactly "
            "and refuses to approximate a float input"
        )
    return np.ascontiguousarray(image)


class MiniMaxM3RustImageProcessor(BaseImageProcessor):
    model_input_names = ["pixel_values", "image_grid_thw"]

    # Mirrors of the HF class attributes that consumers read.
    do_resize = True
    do_rescale = True
    rescale_factor = 1 / 255
    do_normalize = True
    do_convert_rgb = True
    size = {"height": 672, "width": 672}

    def __init__(
        self,
        *,
        patch_size: int,
        merge_size: int,
        temporal_patch_size: int,
        min_pixels: int,
        max_pixels: int,
        image_mean: List[float],
        image_std: List[float],
        image_token_id: int,
        image_start_token_id: int,
        image_end_token_id: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.merge_size = merge_size
        self.temporal_patch_size = temporal_patch_size
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.image_mean = list(image_mean)
        self.image_std = list(image_std)
        self._spec = {
            "family": "minimax_m3",
            "image_token_id": image_token_id,
            "image_start_token_id": image_start_token_id,
            "image_end_token_id": image_end_token_id,
            "patch_size": patch_size,
            "merge_size": merge_size,
            "temporal_patch_size": temporal_patch_size,
            "min_pixels": min_pixels,
            "max_pixels": max_pixels,
            "image_mean": self.image_mean,
            "image_std": self.image_std,
        }

    @classmethod
    def from_hf_processor(
        cls,
        image_processor,
        *,
        image_token_id: int,
        image_start_token_id: int,
        image_end_token_id: int,
    ) -> MiniMaxM3RustImageProcessor:
        return cls(
            patch_size=image_processor.patch_size,
            merge_size=image_processor.merge_size,
            temporal_patch_size=image_processor.temporal_patch_size,
            min_pixels=getattr(image_processor, "min_pixels", _DEFAULT_MIN_PIXELS),
            max_pixels=image_processor.max_pixels,
            image_mean=list(image_processor.image_mean),
            image_std=list(image_processor.image_std),
            image_token_id=image_token_id,
            image_start_token_id=image_start_token_id,
            image_end_token_id=image_end_token_id,
        )

    def _spec_json(self, min_pixels: int, max_pixels: int) -> str:
        if (min_pixels, max_pixels) == (self.min_pixels, self.max_pixels):
            spec = self._spec
        else:
            spec = {
                **self._spec,
                "min_pixels": min_pixels,
                "max_pixels": max_pixels,
            }
        return json.dumps(spec)

    def preprocess(
        self,
        images: Union[ImageInput, List],
        return_tensors: Optional[str] = "pt",
        **kwargs,
    ) -> BatchFeature:
        # `device` arrives from the base processor's fast-path routing; the
        # Rust pipeline is CPU-only by design and `return_tensors` is honored
        # by construction below.
        kwargs.pop("device", None)
        del return_tensors
        min_pixels = kwargs.pop("min_pixels", self.min_pixels)
        max_pixels = kwargs.pop("max_pixels", self.max_pixels)
        # Any other per-call kwarg selects a different pipeline than the one
        # the Rust side reproduces bit-exactly; fail loudly rather than
        # approximate (e.g. an --mm-process-config image override).
        if kwargs:
            raise ValueError(
                f"MiniMaxM3RustImageProcessor cannot honor per-call kwarg(s) "
                f"{sorted(kwargs)}; it reproduces the HF processor's default "
                f"pipeline bit-exactly and nothing else"
            )
        if not isinstance(images, (list, tuple)):
            images = [images]

        arrays = [_to_hwc_u8(img) for img in images]
        spec_json = self._spec_json(min_pixels, max_pixels)
        results = _rs.preprocess_arrays(arrays, spec_json)

        feature_dim = 3 * self.temporal_patch_size * self.patch_size * self.patch_size
        pixel_values = torch.cat(
            [
                torch.from_numpy(pv).view(t * h * w, feature_dim)
                for pv, (t, h, w) in results
            ]
        )
        image_grid_thw = torch.tensor([grid for _, grid in results], dtype=torch.long)
        return BatchFeature(
            data={"pixel_values": pixel_values, "image_grid_thw": image_grid_thw},
            tensor_type=None,
        )

    def get_number_of_image_patches(
        self, height: int, width: int, images_kwargs=None
    ) -> int:
        """Same contract as the HF remote-code method of the same name."""
        images_kwargs = images_kwargs or {}
        patch_size = images_kwargs.get("patch_size", self.patch_size)
        merge_size = images_kwargs.get("merge_size", self.merge_size)
        min_pixels = images_kwargs.get("min_pixels", self.min_pixels)
        max_pixels = images_kwargs.get("max_pixels", self.max_pixels)
        resized_height, resized_width = _rs.smart_resize_py(
            height,
            width,
            patch_size * merge_size,
            min_pixels,
            max_pixels,
        )
        return (resized_height // patch_size) * (resized_width // patch_size)
