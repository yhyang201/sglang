# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
"""GPU image preprocessing for MiniMax M3, mirroring the Kimi K2.5/K3 pattern.

The M3 HF image processor (``MiniMaxM3VLImageProcessor``, remote code in the
checkpoint repo) runs smart_resize -> bicubic resize -> rescale+normalize ->
temporal-duplicate -> merge-block patchify.  This module reproduces that
pipeline with GPU ops: nvJPEG-decoded CHW uint8 tensors (or PIL images) stay
on device, resize uses the PIL-equivalent antialiased bicubic from
``kimi_k25``, and normalization is fused with patch extraction in the shared
``normalize_and_patchify`` kernel.  The token-expansion contract
(``]<]start of image[>[`` + N image tokens + ``]<]end of image[>[``) follows
``processing_minimax.MiniMaxVLProcessor``.
"""

import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image

from sglang.kernels.ops.mm.process import normalize_and_patchify
from sglang.srt.multimodal.kimi_k3_image_processing import normalization_tensors
from sglang.srt.multimodal.processors.kimi_k25 import (
    _resize_images_by_source_shape,
)

MAX_RATIO = 200


def m3_smart_resize(
    height: int,
    width: int,
    factor: int,
    min_pixels: Optional[int] = None,
    max_pixels: int = 451584,
) -> Tuple[int, int]:
    """Exact port of the M3 checkpoint image processor's ``smart_resize``.

    Qwen2-VL-style dynamic resolution: round to the nearest ``factor``
    multiple, then clamp the area into [min_pixels, max_pixels].  This is NOT
    ``get_hw_multiple_of`` (round-up with a per-side cap) from the sglang
    video path; image grids must match the HF processor exactly or the token
    expansion and pixel patches disagree with the CPU fallback.
    """
    if min_pixels is None:
        min_pixels = 4 * 28 * 28
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, "
            f"got {max(height, width) / min(height, width)}"
        )

    def round_by_factor(number: float) -> int:
        return round(number / factor) * factor

    def ceil_by_factor(number: float) -> int:
        return math.ceil(number / factor) * factor

    def floor_by_factor(number: float) -> int:
        return math.floor(number / factor) * factor

    h_bar = max(factor, round_by_factor(height))
    w_bar = max(factor, round_by_factor(width))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta)
        w_bar = floor_by_factor(width / beta)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta)
        w_bar = ceil_by_factor(width * beta)
    return h_bar, w_bar


def _get_image_dimensions(image: Union[torch.Tensor, Image.Image]) -> Tuple[int, int]:
    """Get (width, height) from a CHW tensor or PIL Image."""
    if isinstance(image, torch.Tensor):
        # nvJPEG returns (C, H, W) uint8
        return image.shape[2], image.shape[1]
    return image.size


def _to_cuda_chw(
    image: Union[torch.Tensor, Image.Image], device: Union[str, torch.device]
) -> torch.Tensor:
    """Coerce an image to a 3-channel (C, H, W) uint8 tensor on ``device``.

    PIL inputs are RGB-normalized by the caller, but pre-decoded tensor inputs
    (e.g. nvJPEG) keep their native channel count: grayscale JPEGs decode to 1
    channel, which would break the source-shape batching downstream.  Raw
    0-255 pixels are load-bearing: the resize rounds back to integers and the
    normalization folds in a 1/255 rescale, so a normalized float image would
    collapse to 0/1 and then be rescaled again.
    """
    if isinstance(image, Image.Image):
        array = np.array(image.convert("RGB"), copy=True)
        tensor = torch.from_numpy(array).permute(2, 0, 1)
    else:
        tensor = image
        if tensor.dtype != torch.uint8:
            raise ValueError(
                f"M3 GPU preprocessing expects raw uint8 pixels, got {tensor.dtype}"
            )
        if tensor.dim() == 2:  # (H, W) grayscale -> (1, H, W)
            tensor = tensor.unsqueeze(0)
        channels = tensor.shape[0]
        if channels == 1:
            tensor = tensor.repeat(3, 1, 1)
        elif channels != 3:
            # RGBA or other multi-channel layouts: keep the first 3 channels.
            tensor = tensor[:3]
    return tensor.to(device)


def expand_m3_image_token_ids(
    input_ids: Union[List[int], torch.Tensor],
    image_token_id: int,
    image_start_token_id: int,
    image_end_token_id: int,
    image_token_counts: List[int],
) -> torch.Tensor:
    """Expand each image placeholder into start + N image tokens + end.

    Same contract as ``MiniMaxVLProcessor.__call__``'s text-side expansion,
    but operating on the request's own token IDs so non-media tokens cannot
    drift through decode+retokenize.  Returns a (1, L) int64 tensor.
    """
    if isinstance(input_ids, torch.Tensor):
        input_ids = input_ids.detach().flatten().cpu().numpy()
    input_ids = np.asarray(input_ids, dtype=np.int64)

    placeholder_mask = input_ids == image_token_id
    placeholder_count = np.count_nonzero(placeholder_mask)
    if placeholder_count != len(image_token_counts):
        raise ValueError(
            f"Expected {len(image_token_counts)} image placeholder token(s), "
            f"found {placeholder_count}."
        )
    if placeholder_count == 0:
        return torch.from_numpy(input_ids.copy()).unsqueeze(0)

    counts = np.asarray(image_token_counts, dtype=np.int64)
    repeats = np.ones(input_ids.shape, dtype=np.int64)
    repeats[placeholder_mask] = counts + 2
    expanded = np.repeat(input_ids, repeats)
    # The k-th placeholder starts at its old index plus the growth of all
    # previous placeholders (each grew from length 1 to count + 2).
    placeholder_index = np.flatnonzero(placeholder_mask)
    group_starts = placeholder_index + np.concatenate(([0], np.cumsum(counts + 1)[:-1]))
    expanded[group_starts] = image_start_token_id
    expanded[group_starts + counts + 1] = image_end_token_id
    return torch.from_numpy(expanded).unsqueeze(0)


def m3_merge_patchify(
    patches: torch.Tensor,
    grid_height: int,
    grid_width: int,
    merge_size: int,
    temporal_patch_size: int,
) -> torch.Tensor:
    """Reorder ``normalize_and_patchify`` output into the M3 patch layout.

    Input: (B, gh * gw, C, ps, ps) with row-major (gy, gx) patch order.
    Output: (B, gh * gw, C * tps * ps * ps) where the patch sequence follows
    (gh // m, gw // m, m, m) merge-block order and the feature dim follows
    (C, tps, ps, ps) with the single frame duplicated across tps -- the exact
    output contract of ``MiniMaxM3VLImageProcessor._preprocess`` (grid_t == 1
    for images).
    """
    batch, _, channels, patch_size, _ = patches.shape
    x = patches.view(
        batch,
        grid_height // merge_size,
        merge_size,
        grid_width // merge_size,
        merge_size,
        channels,
        patch_size,
        patch_size,
    )
    x = x.permute(0, 1, 3, 2, 4, 5, 6, 7)
    x = x.reshape(batch, grid_height * grid_width, channels, 1, patch_size, patch_size)
    x = x.expand(-1, -1, -1, temporal_patch_size, -1, -1)
    return x.reshape(batch, grid_height * grid_width, -1)


def gpu_preprocess_m3_images(
    images: List[Union[torch.Tensor, Image.Image]],
    resized_hw: List[Tuple[int, int]],
    image_scale: torch.Tensor,
    image_bias: torch.Tensor,
    patch_size: int,
    merge_size: int,
    temporal_patch_size: int,
    device: Union[str, torch.device],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """GPU preprocessing pipeline for a batch of images.

    Groups images with an identical resize target for batched processing;
    within a group, the bicubic batches only source-compatible inputs (same
    trick as ``kimi_k25._gpu_preprocess_images``).  M3 targets are already
    multiples of patch_size * merge_size, so no zero padding is involved.

    Returns (pixel_values, image_grid_thw): fp32 CUDA tensor of shape
    [sum(gh*gw), C*tps*ps*ps] and an int64 [N, 3] tensor of [1, gh, gw] rows.
    """
    groups: Dict[Tuple[int, int], List[Tuple[int, object]]] = {}
    for index, (image, target_hw) in enumerate(zip(images, resized_hw)):
        groups.setdefault(target_hw, []).append((index, image))

    all_patches: List[Optional[torch.Tensor]] = [None] * len(images)
    all_grids: List[Optional[Tuple[int, int, int]]] = [None] * len(images)

    for (target_h, target_w), group in groups.items():
        grid_h, grid_w = target_h // patch_size, target_w // patch_size
        indexed_images = [
            (index, _to_cuda_chw(image, device)) for index, image in group
        ]
        resized = _resize_images_by_source_shape(indexed_images, target_h, target_w)
        batch = torch.cat(resized, dim=0)
        patches = normalize_and_patchify(
            batch, image_scale, image_bias, patch_size, target_h, target_w
        )
        patches = m3_merge_patchify(
            patches, grid_h, grid_w, merge_size, temporal_patch_size
        )
        for local_index, (index, _) in enumerate(group):
            all_patches[index] = patches[local_index]
            all_grids[index] = (1, grid_h, grid_w)

    pixel_values = torch.cat(all_patches, dim=0)
    grid_thw = torch.tensor(all_grids, dtype=torch.int64)
    return pixel_values, grid_thw


class MiniMaxM3GPUProcessorWrapper:
    """Wraps M3's HF processor to do GPU image preprocessing.

    GPU path (image-only requests, CUDA available): nvJPEG CUDA tensor / PIL
    -> ``gpu_preprocess_m3_images``.  CPU fallback (no CUDA, or any video in
    the request): the original ``MiniMaxVLProcessor.__call__``.

    Exposes the attributes ``BaseMultimodalProcessor.process_mm_data`` needs
    so it behaves like the wrapped HF processor from the outside:
    ``image_processor`` (checked via isinstance for the fast-processor device
    injection) and ``tokenizer`` / ``video_processor``.
    """

    def __init__(
        self,
        hf_processor,
        *,
        image_token: str,
        image_token_id: int,
        image_start_token: str,
        image_start_token_id: int,
        image_end_token: str,
        image_end_token_id: int,
        patch_size: int,
        temporal_patch_size: int,
        merge_size: int,
        max_pixels: int,
        image_mean: List[float],
        image_std: List[float],
        min_pixels: Optional[int] = None,
    ):
        self._hf_processor = hf_processor
        self._image_token = image_token
        self._image_token_id = image_token_id
        self._image_start_token = image_start_token
        self._image_start_token_id = image_start_token_id
        self._image_end_token = image_end_token
        self._image_end_token_id = image_end_token_id
        self._patch_size = patch_size
        self._temporal_patch_size = temporal_patch_size
        self._merge_size = merge_size
        self._max_pixels = max_pixels
        self._min_pixels = min_pixels
        self._image_mean = image_mean
        self._image_std = image_std
        self._gpu_norm_tensors: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

        self.image_processor = getattr(hf_processor, "image_processor", None)
        self.tokenizer = hf_processor.tokenizer
        self.video_processor = getattr(hf_processor, "video_processor", None)

    def __call__(self, text=None, images=None, videos=None, **kwargs):
        # process_mm_data passes media via kwargs
        images = images or kwargs.pop("images", None)
        videos = videos or kwargs.pop("videos", None)
        original_input_ids = kwargs.pop("sglang_original_input_ids", None)
        device = kwargs.pop("device", None)

        if images and not videos and torch.cuda.is_available():
            return self._gpu_call(text, images, original_input_ids, device, **kwargs)
        return self._cpu_call(text, images, videos, device=device, **kwargs)

    def _prepare_input_ids(
        self,
        input_text: str,
        image_token_counts: List[int],
        original_input_ids,
        **kwargs,
    ) -> torch.Tensor:
        if original_input_ids is not None:
            return expand_m3_image_token_ids(
                original_input_ids,
                self._image_token_id,
                self._image_start_token_id,
                self._image_end_token_id,
                image_token_counts,
            )

        parts = input_text.split(self._image_token)
        if len(parts) - 1 != len(image_token_counts):
            raise ValueError(
                f"Expected {len(image_token_counts)} image placeholder(s) in the "
                f"prompt, found {len(parts) - 1}."
            )
        result = [parts[0]]
        for num_tokens, part in zip(image_token_counts, parts[1:]):
            result.append(
                self._image_start_token
                + self._image_token * num_tokens
                + self._image_end_token
                + part
            )
        expanded_text = "".join(result)
        # Mirror the tokenizer call the HF processor would make; the base
        # class only ever customizes it with add_special_tokens=False.
        return self.tokenizer(
            expanded_text,
            return_tensors="pt",
            add_special_tokens=kwargs.get("add_special_tokens", True),
        )["input_ids"]

    def _gpu_call(
        self,
        text,
        images,
        original_input_ids=None,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ) -> Dict:
        """Bypass MiniMaxM3VLImageProcessor._preprocess entirely -- GPU ops."""
        input_text = text[0] if isinstance(text, list) else text
        device = device if device is not None else "cuda"

        # 1. Resize math (CPU scalars), identical to the HF image processor.
        factor = self._patch_size * self._merge_size
        resized_hw = []
        for image in images:
            width, height = _get_image_dimensions(image)
            resized_hw.append(
                m3_smart_resize(
                    height,
                    width,
                    factor,
                    min_pixels=self._min_pixels,
                    max_pixels=self._max_pixels,
                )
            )

        # 2. Token expansion: one placeholder -> start + prod(grid)/merge^2 + end
        image_token_counts = [
            (resized_h // self._patch_size)
            * (resized_w // self._patch_size)
            // (self._merge_size**2)
            for resized_h, resized_w in resized_hw
        ]
        input_ids = self._prepare_input_ids(
            input_text, image_token_counts, original_input_ids, **kwargs
        )

        # 3. GPU image preprocessing
        image_scale, image_bias = self._get_gpu_norm_tensors(device)
        pixel_values, grid_thw = gpu_preprocess_m3_images(
            images,
            resized_hw,
            image_scale,
            image_bias,
            self._patch_size,
            self._merge_size,
            self._temporal_patch_size,
            device,
        )

        return {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            # SGL-standard key so get_new_expanded_mm_items() can split
            # per-image for cache granularity (it looks up 'image_grid_thw').
            "image_grid_thw": grid_thw,
        }

    def _cpu_call(self, text, images, videos, device=None, **kwargs):
        """Fallback: the checkpoint's own MiniMaxVLProcessor, unchanged."""
        if device is not None:
            kwargs["device"] = device
        return self._hf_processor(text=text, images=images, videos=videos, **kwargs)

    def _get_gpu_norm_tensors(
        self, device: Union[str, torch.device]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        key = str(device)
        cached = self._gpu_norm_tensors.get(key)
        if cached is None:
            # Fused rescale+normalize: (v - mean * 255) / (std * 255) ==
            # v * scale + bias, matching the HF fast processor's folding.
            cached = normalization_tensors(self._image_mean, self._image_std, device)
            self._gpu_norm_tensors[key] = cached
        return cached
