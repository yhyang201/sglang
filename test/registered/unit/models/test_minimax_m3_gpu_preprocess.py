"""MiniMax M3 GPU image preprocessing tests.

The hermetic tests pin the pure math: smart_resize golden values captured from
the checkpoint's own ``MiniMaxM3VLImageProcessor`` (on a B300 devbox, see
docstrings), the start/end-wrapped token expansion, the merge-block patchify
layout against a verbatim port of the HF reference, and the CPU-fallback
passthrough of ``MiniMaxM3GPUProcessorWrapper``.

The ``test_live_*`` tests compare the wrapper end-to-end against the real HF
processor and need a local MiniMax-M3 checkpoint (only the processor/tokenizer
files are loaded). They run wherever MINIMAX_M3_MODEL_PATH (or
/root/models/MiniMax-M3-MXFP8) exists; with CUDA they exercise the Triton
path, otherwise the torch fallback of normalize_and_patchify.
"""

import os
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
from PIL import Image

from sglang.srt.multimodal.minimax_m3_image_processing import (
    MiniMaxM3GPUProcessorWrapper,
    expand_m3_image_token_ids,
    m3_merge_patchify,
    m3_smart_resize,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# (W, H) -> (resized_h, resized_w, grid_t, grid_h, grid_w, token_count),
# captured from MiniMaxM3VLImageProcessor of MiniMax-M3-MXFP8 (patch=14,
# merge=2, max_pixels=451584, min_pixels=3136).
M3_SMART_RESIZE_GOLDEN = {
    (1024, 1024): (672, 672, 1, 48, 48, 576),
    (500, 1000): (924, 448, 1, 66, 32, 528),
    (1000, 500): (448, 924, 1, 32, 66, 528),
    (672, 672): (672, 672, 1, 48, 48, 576),
    (28, 28): (56, 56, 1, 4, 4, 4),
    (27, 27): (56, 56, 1, 4, 4, 4),
    (100, 100): (112, 112, 1, 8, 8, 16),
    (4000, 224): (140, 2828, 1, 10, 202, 505),
    (224, 4000): (2828, 140, 1, 202, 10, 505),
    (2048, 1536): (560, 756, 1, 40, 54, 540),
    (1, 1): (56, 56, 1, 4, 4, 4),
    (29, 31): (84, 56, 1, 6, 4, 6),
    (336, 672): (672, 336, 1, 48, 24, 288),
}

M3_MODEL_PATH_CANDIDATES = (
    os.environ.get("MINIMAX_M3_MODEL_PATH"),
    "/root/models/MiniMax-M3-MXFP8",
)
M3_MODEL_PATH = next(
    (p for p in M3_MODEL_PATH_CANDIDATES if p and os.path.isdir(p)), None
)
needs_m3_model = pytest.mark.skipif(
    M3_MODEL_PATH is None, reason="MiniMax-M3 checkpoint not available"
)


def _make_test_image(width, height, seed=0):
    """Photo-like content, not pure noise: bicubic differences only show up
    against smooth gradients plus high-frequency checkerboard."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    plane = np.clip(
        128
        + 90 * np.sin(xx / 40) * np.cos(yy / 55)
        + 40 * ((xx // 37 + yy // 41) % 2)
        + rng.normal(0, 6, (height, width)),
        0,
        255,
    )
    array = np.stack([plane, np.roll(plane, 7, 0), np.roll(plane, 13, 1)], -1).astype(
        np.uint8
    )
    return Image.fromarray(array)


def test_m3_smart_resize_golden_values():
    for (width, height), (
        res_h,
        res_w,
        gt,
        gh,
        gw,
        tokens,
    ) in M3_SMART_RESIZE_GOLDEN.items():
        assert m3_smart_resize(height, width, 28) == (res_h, res_w), (width, height)
        assert (gh, gw) == (res_h // 14, res_w // 14)
        assert tokens == gt * gh * gw // 4


def test_m3_smart_resize_rejects_extreme_aspect_ratio():
    with pytest.raises(ValueError, match="aspect ratio"):
        m3_smart_resize(28, 28 * 201, 28)


def _reference_expand(input_ids, placeholder, start, end, counts):
    """The MiniMaxVLProcessor expansion as a loop: each placeholder becomes
    start + count image tokens + end."""
    rebuilt, next_image = [], 0
    for token_id in input_ids:
        if token_id == placeholder:
            rebuilt.append(start)
            rebuilt.extend([placeholder] * counts[next_image])
            rebuilt.append(end)
            next_image += 1
        else:
            rebuilt.append(token_id)
    return rebuilt


def test_expand_m3_image_token_ids_matches_reference():
    # 7 is the placeholder, 1/2 are start/end; images claim 3 and 2 tokens.
    input_ids = [10, 7, 20, 7, 30]
    expected = _reference_expand(input_ids, 7, 1, 2, [3, 2])

    expanded = expand_m3_image_token_ids(input_ids, 7, 1, 2, [3, 2])

    assert expanded.tolist() == [expected]


def test_expand_m3_image_token_ids_random_against_reference():
    rng = np.random.default_rng(0)
    for n_images in (1, 3, 8):
        ids = rng.integers(100, 5000, 400).tolist()
        for slot in range(n_images):
            ids.insert(slot * 37 + 5, 7)
        counts = rng.integers(1, 576, n_images).tolist()
        expected = _reference_expand(ids, 7, 1, 2, counts)
        expanded = expand_m3_image_token_ids(ids, 7, 1, 2, counts)
        assert expanded.flatten().tolist() == expected


def test_expand_m3_image_token_ids_rejects_placeholder_mismatch():
    with pytest.raises(ValueError, match="placeholder"):
        expand_m3_image_token_ids([1, 7, 2], 7, 1, 2, [3, 2])


def test_m3_merge_patchify_matches_hf_reference_layout():
    # Verbatim port of the patchify section of the checkpoint's
    # MiniMaxM3VLImageProcessor._preprocess (temporal duplicate + 10-dim
    # view/permute), minus the rescale/normalize which is applied to the same
    # input for both paths.
    def hf_reference_patchify(images, patch_size, temporal_patch_size, merge_size):
        patches = images
        if patches.ndim == 4:
            patches = patches.unsqueeze(1)
        if patches.shape[1] % temporal_patch_size != 0:
            repeats = patches[:, -1:].repeat(
                1,
                temporal_patch_size - (patches.shape[1] % temporal_patch_size),
                1,
                1,
                1,
            )
            patches = torch.cat([patches, repeats], dim=1)
        batch_size, grid_t, channel = patches.shape[:3]
        grid_t = grid_t // temporal_patch_size
        grid_h, grid_w = (
            images.shape[-2] // patch_size,
            images.shape[-1] // patch_size,
        )
        patches = patches.view(
            batch_size,
            grid_t,
            temporal_patch_size,
            channel,
            grid_h // merge_size,
            merge_size,
            patch_size,
            grid_w // merge_size,
            merge_size,
            patch_size,
        )
        patches = patches.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
        return patches.reshape(
            batch_size,
            grid_t * grid_h * grid_w,
            channel * temporal_patch_size * patch_size * patch_size,
        )

    torch.manual_seed(0)
    patch_size, merge_size, temporal_patch_size = 14, 2, 2
    for grid_h, grid_w in ((48, 48), (32, 66), (4, 4), (40, 54)):
        images = torch.randint(
            0, 255, (2, 3, grid_h * patch_size, grid_w * patch_size)
        ).float()
        expected = hf_reference_patchify(
            images, patch_size, temporal_patch_size, merge_size
        )
        scale = torch.full((1, 3, 1, 1), 1.0 / 255.0)
        bias = torch.zeros(1, 3, 1, 1)
        from sglang.kernels.ops.mm.process import normalize_and_patchify

        patches = normalize_and_patchify(
            images, scale, bias, patch_size, images.shape[-2], images.shape[-1]
        )
        actual = m3_merge_patchify(
            patches, grid_h, grid_w, merge_size, temporal_patch_size
        )
        assert actual.shape == expected.shape
        # Pure permutation + duplication of one affine map; values agree to fp noise.
        torch.testing.assert_close(
            actual, expected * (1.0 / 255.0), rtol=1e-5, atol=1e-5
        )


def _make_wrapper(hf_processor=None):
    if hf_processor is None:
        hf_processor = Mock()
    return MiniMaxM3GPUProcessorWrapper(
        hf_processor,
        image_token="]<]image[>[",
        image_token_id=7,
        image_start_token="]<]start of image[>[",
        image_start_token_id=1,
        image_end_token="]<]end of image[>[",
        image_end_token_id=2,
        patch_size=14,
        temporal_patch_size=2,
        merge_size=2,
        max_pixels=451584,
        image_mean=[0.48145466, 0.4578275, 0.40821073],
        image_std=[0.26862954, 0.26130258, 0.27577711],
    )


def test_wrapper_cpu_call_passthrough_is_bitwise_and_forwards_kwargs():
    sentinel = {"input_ids": torch.tensor([[5, 6]]), "pixel_values": torch.zeros(4, 4)}
    hf_processor = Mock(return_value=sentinel)
    wrapper = _make_wrapper(hf_processor)

    out = wrapper._cpu_call(
        ["a"],
        images=["img"],
        videos=None,
        device="cuda:3",
        padding=True,
        return_tensors="pt",
    )

    assert out is sentinel
    _, call_kwargs = hf_processor.call_args
    assert call_kwargs["device"] == "cuda:3"
    assert call_kwargs["images"] == ["img"]
    assert call_kwargs["videos"] is None
    assert call_kwargs["padding"] is True


def test_wrapper_routes_to_cpu_without_cuda_or_with_videos():
    hf_processor = Mock(return_value={"input_ids": torch.tensor([[1]])})
    wrapper = _make_wrapper(hf_processor)
    image = _make_test_image(64, 64)

    with patch("torch.cuda.is_available", return_value=False):
        wrapper(text=["x ]<]image[>[ y"], images=[image])
    assert hf_processor.called
    hf_processor.reset_mock()

    # Videos in the request always take the HF path, even with CUDA around.
    if torch.cuda.is_available():
        wrapper(text=["x"], images=[image], videos=["video"])
        assert hf_processor.called
        _, call_kwargs = hf_processor.call_args
        assert call_kwargs["videos"] == ["video"]


def test_wrapper_gpu_call_rejects_placeholder_mismatch():
    wrapper = _make_wrapper(Mock())
    image = _make_test_image(64, 64)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with pytest.raises(ValueError, match="placeholder"):
        wrapper._gpu_call("no placeholder here", [image], device=device)


@needs_m3_model
class TestLiveM3ProcessorParity:
    """GPU wrapper vs the checkpoint's HF processor (tolerance comparison)."""

    @classmethod
    def setup_class(cls):
        from transformers import AutoProcessor

        cls.hf_processor = AutoProcessor.from_pretrained(
            M3_MODEL_PATH, trust_remote_code=True
        )
        ip = cls.hf_processor.image_processor
        tok = cls.hf_processor.tokenizer
        im_tok = "]<]image[>["
        im_start = "]<]start of image[>["
        im_end = "]<]end of image[>["
        cls.wrapper = MiniMaxM3GPUProcessorWrapper(
            cls.hf_processor,
            image_token=im_tok,
            image_token_id=tok.convert_tokens_to_ids(im_tok),
            image_start_token=im_start,
            image_start_token_id=tok.convert_tokens_to_ids(im_start),
            image_end_token=im_end,
            image_end_token_id=tok.convert_tokens_to_ids(im_end),
            patch_size=ip.patch_size,
            temporal_patch_size=ip.temporal_patch_size,
            merge_size=ip.merge_size,
            max_pixels=ip.max_pixels,
            image_mean=list(ip.image_mean),
            image_std=list(ip.image_std),
        )
        cls.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _compare(self, images, prompt, original_input_ids=None):
        ref = self.wrapper._cpu_call(
            text=[prompt],
            images=images,
            videos=None,
            device=None,
            padding=True,
            return_tensors="pt",
        )
        got = self.wrapper._gpu_call(
            text=[prompt],
            images=images,
            original_input_ids=original_input_ids,
            device=self.device,
        )
        assert torch.equal(ref["input_ids"].cpu(), got["input_ids"].cpu())
        assert torch.equal(ref["image_grid_thw"].cpu(), got["image_grid_thw"].cpu())
        ref_pv = ref["pixel_values"].cpu().float()
        got_pv = got["pixel_values"].cpu().float()
        assert ref_pv.shape == got_pv.shape
        diff = (ref_pv - got_pv).abs()
        # GPU bicubic is within a couple of 8-bit levels of the PIL/torchvision
        # uint8 fixed-point path (one level ~= 0.0146 after normalization).
        assert diff.max().item() < 0.05, diff.max().item()
        return diff.max().item(), diff.mean().item()

    def test_live_single_image_parity(self):
        image = _make_test_image(1024, 1024, seed=0)
        max_d, mean_d = self._compare([image], "describe this ]<]image[>[ please")
        print(f"\nsingle-1024: max={max_d:.6f} mean={mean_d:.8f}")

    def test_live_multi_image_parity(self):
        images = [
            _make_test_image(1024, 1024, seed=0),
            _make_test_image(500, 1000, seed=1),
            _make_test_image(1024, 1024, seed=0),
        ]
        prompt = "a ]<]image[>[ b ]<]image[>[ c ]<]image[>[ d"
        max_d, mean_d = self._compare(images, prompt)
        print(f"\nmulti-mixed: max={max_d:.6f} mean={mean_d:.8f}")

    def test_live_tensor_input_parity(self):
        image = _make_test_image(1024, 1024, seed=0)
        tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1)
        max_d, mean_d = self._compare([tensor], "tensor ]<]image[>[ input")
        print(f"\ntensor-input: max={max_d:.6f} mean={mean_d:.8f}")

    def test_live_id_space_expansion_matches_string_expansion(self):
        image = _make_test_image(1024, 1024, seed=0)
        prompt = "describe this ]<]image[>[ please"
        original_ids = self.hf_processor.tokenizer(prompt, return_tensors="pt")[
            "input_ids"
        ]
        got_str = self.wrapper._gpu_call(
            text=[prompt], images=[image], device=self.device
        )
        got_ids = self.wrapper._gpu_call(
            text=[prompt],
            images=[image],
            original_input_ids=original_ids,
            device=self.device,
        )
        assert torch.equal(got_str["input_ids"], got_ids["input_ids"])

    def test_live_cpu_fallback_is_bitwise_identical_to_hf(self):
        image = _make_test_image(1024, 1024, seed=0)
        prompt = "describe this ]<]image[>[ please"
        direct = self.hf_processor(
            text=[prompt], images=[image], padding=True, return_tensors="pt"
        )
        fallback = self.wrapper._cpu_call(
            text=[prompt],
            images=[image],
            videos=None,
            padding=True,
            return_tensors="pt",
        )
        assert torch.equal(direct["input_ids"], fallback["input_ids"])
        assert torch.equal(direct["pixel_values"], fallback["pixel_values"])
        assert torch.equal(direct["image_grid_thw"], fallback["image_grid_thw"])
