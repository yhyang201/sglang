# SPDX-License-Identifier: Apache-2.0
"""End-to-end test for disaggregated diffusion server.

Launches a full disagg server via CLI (encoder + denoiser + decoder as separate
processes), sends a video generation request via the OpenAI-compatible API,
and validates the output.

Usage:
    # Requires 2+ GPUs (encoder+decoder share GPU 0, denoiser on GPU 1)
    python -m pytest python/sglang/multimodal_gen/test/server/test_disagg_server.py -v -s

    # With custom GPU assignment (3 GPUs)
    DISAGG_ENCODER_GPUS=0 DISAGG_DENOISER_GPUS=1 DISAGG_DECODER_GPUS=2 \
        python -m pytest python/sglang/multimodal_gen/test/server/test_disagg_server.py -v -s
"""

from __future__ import annotations

import os
import time

import pytest

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.test.server.test_server_utils import (
    ServerContext,
    ServerManager,
)
from sglang.multimodal_gen.test.test_utils import (
    get_dynamic_server_port,
)

logger = init_logger(__name__)

# Model for testing — small enough for CI
DEFAULT_MODEL = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"


def _build_disagg_extra_args() -> str:
    """Build CLI extra args for disagg server from env vars."""
    args = "--disagg-mode"

    encoder_gpus = os.environ.get("DISAGG_ENCODER_GPUS", "0")
    denoiser_gpus = os.environ.get("DISAGG_DENOISER_GPUS", "1")
    decoder_gpus = os.environ.get("DISAGG_DECODER_GPUS", "0")

    args += f" --encoder-gpus {encoder_gpus}"
    args += f" --denoiser-gpus {denoiser_gpus}"
    args += f" --decoder-gpus {decoder_gpus}"

    # Common args
    args += " --text-encoder-cpu-offload"
    args += " --pin-cpu-memory"
    args += " --warmup"

    return args


@pytest.fixture(scope="module")
def disagg_server() -> ServerContext:
    """Launch a disaggregated diffusion server and wait for it to be ready."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if torch.cuda.device_count() < 2:
        pytest.skip("Requires 2+ GPUs for disaggregated mode")

    model = os.environ.get("DISAGG_MODEL", DEFAULT_MODEL)
    port = int(os.environ.get("SGLANG_TEST_SERVER_PORT", get_dynamic_server_port()))
    extra_args = _build_disagg_extra_args()

    logger.info(
        "Starting disagg server: model=%s, port=%d, extra_args=%s",
        model,
        port,
        extra_args,
    )

    manager = ServerManager(
        model=model,
        port=port,
        wait_deadline=float(os.environ.get("SGLANG_TEST_WAIT_SECS", "600")),
        extra_args=extra_args,
    )
    ctx = manager.start()

    try:
        yield ctx
    finally:
        ctx.cleanup()


class TestDisaggServer:
    """End-to-end tests for disaggregated diffusion server."""

    def test_video_generation(self, disagg_server: ServerContext):
        """Test T2V generation through the full disagg pipeline."""
        from openai import OpenAI

        client = OpenAI(
            base_url=f"http://127.0.0.1:{disagg_server.port}/v1",
            api_key="unused",
        )

        prompt = "A curious raccoon exploring a garden"

        # Create video job
        job = client.videos.create(
            model=disagg_server.model,
            prompt=prompt,
            size="832x480",
            extra_body={
                "num_frames": 17,
                "num_inference_steps": 5,
                "guidance_scale": 1.0,
            },
        )
        video_id = job.id
        logger.info("Created video job: %s", video_id)

        # Poll for completion
        timeout = 300  # 5 minutes
        deadline = time.time() + timeout
        completed = False

        while time.time() < deadline:
            page = client.videos.list()
            item = next((v for v in page.data if v.id == video_id), None)
            if item and getattr(item, "status", None) == "completed":
                completed = True
                break
            time.sleep(2)

        assert completed, f"Video job {video_id} did not complete within {timeout}s"

        # Download and validate
        resp = client.videos.download_content(video_id=video_id)
        content = resp.read()

        assert len(content) > 0, "Empty video content"
        logger.info(
            "Video generation completed: job=%s, size=%d bytes",
            video_id,
            len(content),
        )

    def test_health_check(self, disagg_server: ServerContext):
        """Test that the health endpoint works."""
        import requests

        resp = requests.get(f"http://127.0.0.1:{disagg_server.port}/health")
        assert resp.status_code == 200
