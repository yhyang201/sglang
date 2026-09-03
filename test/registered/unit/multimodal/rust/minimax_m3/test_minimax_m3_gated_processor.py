"""End-to-end wiring of ``SGLANG_MINIMAX_M3_RS_MM_PREPROCESS``.

Builds the sglang ``MiniMaxM3VLProcessor`` over a tiny hand-built tokenizer
and the HF ``MiniMaxM3VLProcessor`` (in-tree twin of the checkpoint's remote
code), once with the env gate off (HF image processor) and once on (the
Rust-backed swap in ``processors/minimax_m3_vl.py``), then requires
token-for-token identical input_ids and bit-identical pixel_values /
image_grid_thw out of ``process_mm_data_async``.
"""

import asyncio
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.environ import envs  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _mm_rust_utils import load_core, make_image  # noqa: E402

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

M3_CORE = getattr(load_core(), "minimax_m3", None)

VOCAB = [
    "<unk>",
    "]<]image[>[",
    "]<]video[>[",
    "]<]start of image[>[",
    "]<]end of image[>[",
    "hello",
    "world",
    "<pad>",
]
SIZES = ((640, 480), (300, 301))


def make_hf_processor():
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    from transformers.models.minimax_m3_vl.image_processing_minimax_m3_vl import (
        MiniMaxM3VLImageProcessor,
    )
    from transformers.models.minimax_m3_vl.processing_minimax_m3_vl import (
        MiniMaxM3VLProcessor,
    )
    from transformers.models.minimax_m3_vl.video_processing_minimax_m3_vl import (
        MiniMaxM3VLVideoProcessor,
    )

    backend = Tokenizer(
        models.WordLevel({t: i for i, t in enumerate(VOCAB)}, unk_token=VOCAB[0])
    )
    backend.pre_tokenizer, backend.decoder = (
        pre_tokenizers.WhitespaceSplit(),
        decoders.Fuse(),
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token=VOCAB[0],
        pad_token=VOCAB[-1],
        additional_special_tokens=VOCAB[1:5],
    )
    return MiniMaxM3VLProcessor(
        image_processor=MiniMaxM3VLImageProcessor(),
        tokenizer=tokenizer,
        video_processor=MiniMaxM3VLVideoProcessor(),
    )


def make_sglang_processor(case):
    from sglang.srt.multimodal.processors.minimax_m3_vl import (
        MiniMaxM3VLProcessor as SglMiniMaxM3,
    )
    from sglang.srt.runtime_context import publish, reset_context
    from sglang.srt.server_args import ServerArgs

    hf_config = SimpleNamespace(
        model_type="minimax_m3_vl",
        architectures=["MiniMaxM3SparseForConditionalGeneration"],
        vision_config=SimpleNamespace(spatial_merge_size=2),
    )
    server_args = SimpleNamespace(
        # Non-auto: get_resolved_model_impl would choke on a SimpleNamespace.
        model_impl="sglang",
        keep_mm_feature_on_device=False,
        mm_feature_transport="cpu",
        image_processor_backend="auto",
        disable_fast_image_processor=True,  # keep features on CPU
        skip_tokenizer_init=False,
        mm_preprocess_cache_size_mb=0,
        trust_mm_content_hashes=False,
        tp_size=1,
        dist_init_addr=None,
        mm_process_config={},
        mm_io_worker_num=1,
        mm_processor_worker_num=1,
        tokenizer_worker_num=1,
        base_gpu_id=0,
        rl_on_policy_target=None,
        allowed_media_domains=[],
        media_url_max_file_size_mb=64,
    )
    publish(
        ServerArgs(
            model_path="dummy",
            mm_feature_transport=server_args.mm_feature_transport,
            mm_process_config=server_args.mm_process_config,
            allowed_media_domains=server_args.allowed_media_domains,
            disable_fast_image_processor=server_args.disable_fast_image_processor,
        ),
        role="tokenizer",
    )
    case.addCleanup(reset_context)
    return SglMiniMaxM3(
        hf_config, server_args, make_hf_processor(), None, skip_mm_pool=True
    )


def run_request(processor, images):
    request = SimpleNamespace(video_data=None)
    return asyncio.run(
        processor.process_mm_data_async(
            image_data=images,
            audio_data=None,
            input_text="hello ]<]image[>[ world ]<]image[>[",
            request_obj=request,
        )
    )


@unittest.skipUnless(M3_CORE, "sglang-mm MiniMax M3 binding not built")
class TestMiniMaxM3GatedProcessor(CustomTestCase):
    def test_gate_swaps_image_processor_and_matches(self):
        envs.SGLANG_MINIMAX_M3_RS_MM_PREPROCESS.clear()
        reference = make_sglang_processor(self)
        self.addCleanup(reference.io_executor.shutdown)
        self.addCleanup(reference.cpu_executor.shutdown)
        self.assertNotIn("Rust", type(reference._processor.image_processor).__name__)

        envs.SGLANG_MINIMAX_M3_RS_MM_PREPROCESS.set(True)
        self.addCleanup(envs.SGLANG_MINIMAX_M3_RS_MM_PREPROCESS.clear)
        gated = make_sglang_processor(self)
        self.addCleanup(gated.io_executor.shutdown)
        self.addCleanup(gated.cpu_executor.shutdown)
        self.assertEqual(
            type(gated._processor.image_processor).__name__,
            "MiniMaxM3RustImageProcessor",
        )

        images = [make_image(w, h, seed=i) for i, (w, h) in enumerate(SIZES)]
        expected = run_request(reference, images)
        actual = run_request(gated, images)

        self.assertEqual(actual.input_ids, expected.input_ids)
        self.assertEqual(len(actual.mm_items), len(expected.mm_items))
        for got, want in zip(actual.mm_items, expected.mm_items):
            np.testing.assert_array_equal(
                got.feature.detach().cpu().numpy(),
                want.feature.detach().cpu().numpy(),
            )
            np.testing.assert_array_equal(
                got.model_specific_data["image_grid_thw"].detach().cpu().numpy(),
                want.model_specific_data["image_grid_thw"].detach().cpu().numpy(),
            )


if __name__ == "__main__":
    unittest.main()
