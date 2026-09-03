"""MiniMax M3 native image preprocessing parity against Transformers.

Covers ``MiniMaxM3Processor::process_image`` and the shared ``smart_resize``
in ``rust/sglang-mm/src/minimax_m3/mod.rs`` (via the
``_core.minimax_m3.preprocess`` and ``smart_resize_py`` bindings), against the
HF ``MiniMaxM3VLImageProcessor`` — the torchvision (uint8 antialias bicubic)
path, the only image processor M3 ships. The reference processor comes from
the model checkpoint's remote code when ``MINIMAX_M3_MODEL_DIR`` (default
``/root/models/MiniMax-M3-MXFP8``) is present, else from the identical
in-tree transformers implementation.

Zero tolerance everywhere: any drift in any stage — smart_resize geometry,
the fixed-point kernel, fused rescale/normalize, HF patch order — shows up
here instead of being absorbed. Test images are PNG-encoded because PNG
decode is bit-exact between PIL and the Rust `image` crate (JPEG may differ
by ±1 LSB — a documented boundary).
"""

import json
import os
import sys
import unittest
from pathlib import Path

import numpy as np

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _mm_rust_utils import image_bytes, load_core, make_image  # noqa: E402

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

M3_CORE = getattr(load_core(), "minimax_m3", None)

MODEL_DIR = os.environ.get("MINIMAX_M3_MODEL_DIR", "/root/models/MiniMax-M3-MXFP8")

SIZES = ((640, 480), (1024, 683), (50, 40), (300, 301), (672, 672), (1365, 2048))


def load_hf_processor():
    """The reference HF image processor: the checkpoint's remote code with its
    real preprocessor_config.json when available, else the in-tree twin."""
    if os.path.isdir(MODEL_DIR):
        try:
            from transformers import AutoImageProcessor

            return AutoImageProcessor.from_pretrained(MODEL_DIR, trust_remote_code=True)
        except Exception:
            pass  # fall through to the in-tree implementation
    try:
        from transformers.models.minimax_m3_vl.image_processing_minimax_m3_vl import (
            MiniMaxM3VLImageProcessor,
        )
    except ImportError:
        return None
    return MiniMaxM3VLImageProcessor()


def spec_json(hf):
    """The Rust pipeline spec resolved from the HF processor's attributes —
    the same resolution ``MiniMaxM3RustImageProcessor.from_hf_processor``
    performs (token ids are layout-only; arbitrary here)."""
    return json.dumps(
        {
            "family": "minimax_m3",
            "image_token_id": 900,
            "image_start_token_id": 901,
            "image_end_token_id": 902,
            "patch_size": hf.patch_size,
            "merge_size": hf.merge_size,
            "temporal_patch_size": hf.temporal_patch_size,
            # The HF processor never exposes min_pixels as an attribute; it is
            # the smart_resize function default 4 * 28 * 28.
            "min_pixels": getattr(hf, "min_pixels", 4 * 28 * 28),
            "max_pixels": hf.max_pixels,
            "image_mean": list(hf.image_mean),
            "image_std": list(hf.image_std),
        }
    )


@unittest.skipUnless(M3_CORE, "sglang-mm MiniMax M3 binding not built")
class TestMiniMaxM3ImagePreprocess(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.hf = load_hf_processor()
        if cls.hf is None:
            raise unittest.SkipTest("no HF MiniMaxM3VLImageProcessor available")
        cls.spec = spec_json(cls.hf)

    def test_features_match_hf_processor_exactly(self):
        for index, size in enumerate(SIZES):
            with self.subTest(size=size):
                actual, grid = M3_CORE.preprocess(
                    image_bytes(*size, seed=index), self.spec
                )
                expected = self.hf(
                    images=[make_image(*size, seed=index)], return_tensors="pt"
                )
                self.assertEqual(grid, tuple(expected.image_grid_thw[0].tolist()))
                np.testing.assert_array_equal(
                    np.asarray(actual).reshape(expected.pixel_values.shape),
                    expected.pixel_values.numpy(),
                )

    def test_batch_matches_per_image(self):
        """A mixed-shape batch must equal per-image runs (HF groups by shape;
        Rust parallelizes per image) and keep input order."""
        images = [make_image(w, h, seed=i) for i, (w, h) in enumerate(SIZES[:4])]
        arrays = [np.asarray(img) for img in images]
        batched = M3_CORE.preprocess_arrays(arrays, self.spec)
        expected = self.hf(images=images, return_tensors="pt")
        row = 0
        for (pv, grid), (w, h) in zip(batched, [s[:2] for s in SIZES[:4]]):
            t, gh, gw = grid
            n = t * gh * gw
            np.testing.assert_array_equal(
                np.asarray(pv).reshape(n, -1),
                expected.pixel_values[row : row + n].numpy(),
            )
            row += n
        self.assertEqual(
            [list(g) for _, g in batched], expected.image_grid_thw.tolist()
        )

    def test_smart_resize_matches_hf(self):
        # The module that defined the reference processor's class — in-tree or
        # remote code — so both reference sources are covered.
        smart_resize = getattr(
            sys.modules[type(self.hf).__module__], "smart_resize", None
        )
        if smart_resize is None:
            from transformers.models.minimax_m3_vl.image_processing_minimax_m3_vl import (  # noqa: E501
                smart_resize,
            )

        cases = (
            (640, 480),
            (1024, 683),
            (50, 40),
            (300, 301),
            (1365, 2048),
            (4000, 3000),
            (20, 20),
            (42, 1024),  # round-half-even tie: 42/28 = 1.5 -> 2
            (70, 1024),  # 70/28 = 2.5 -> 2 (banker's), not 3
            (672, 672),
        )
        for h, w in cases:
            with self.subTest(case=(h, w)):
                self.assertEqual(
                    M3_CORE.smart_resize_py(h, w, 28, 4 * 28 * 28, 672 * 672),
                    smart_resize(h, w, factor=28, max_pixels=672 * 672),
                )

    def test_python_wrapper_matches_hf_processor(self):
        """The env-gated integration class, end to end from PIL images."""
        try:
            from sglang.srt.multimodal.minimax_m3.image_processing_rust import (
                MiniMaxM3RustImageProcessor,
            )
        except ImportError:
            self.skipTest("sglang python environment not fully available")
        rust_ip = MiniMaxM3RustImageProcessor.from_hf_processor(
            self.hf,
            image_token_id=900,
            image_start_token_id=901,
            image_end_token_id=902,
        )
        images = [make_image(w, h, seed=i) for i, (w, h) in enumerate(SIZES)]
        actual = rust_ip.preprocess(images)
        expected = self.hf(images=images, return_tensors="pt")
        self.assertEqual(
            actual.image_grid_thw.tolist(), expected.image_grid_thw.tolist()
        )
        np.testing.assert_array_equal(
            actual.pixel_values.numpy(), expected.pixel_values.numpy()
        )
        # The remote-code helper must agree too.
        for w, h in SIZES:
            self.assertEqual(
                rust_ip.get_number_of_image_patches(h, w),
                self.hf.get_number_of_image_patches(h, w),
            )


if __name__ == "__main__":
    unittest.main()
