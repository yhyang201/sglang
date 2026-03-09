# SPDX-License-Identifier: Apache-2.0
"""End-to-end test for disaggregated diffusion server (pool mode).

Launches a full pool-mode disagg server via launch_pool_disagg_server()
(encoder + denoiser + decoder as separate processes orchestrated by
DiffusionServer), sends a video generation request via the OpenAI-compatible
API, and validates the output.

Usage:
    # Requires 2+ GPUs (encoder+decoder share GPU 0, denoiser on GPU 1)
    python -m pytest python/sglang/multimodal_gen/test/server/test_disagg_server.py -v -s

    # With custom GPU assignment (3 GPUs)
    DISAGG_ENCODER_GPUS=0 DISAGG_DENOISER_GPUS=1 DISAGG_DECODER_GPUS=2 \
        python -m pytest python/sglang/multimodal_gen/test/server/test_disagg_server.py -v -s
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time

import pytest

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

# Model for testing — small enough for CI
DEFAULT_MODEL = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"


def _parse_gpu_list(env_var: str, default: str) -> list[list[int]]:
    """Parse GPU assignment from env var. E.g. '0' -> [[0]], '1,2' -> [[1,2]]."""
    raw = os.environ.get(env_var, default)
    return [[int(g) for g in raw.split(",")]]


@pytest.fixture(scope="module")
def disagg_server():
    """Launch a pool-mode disaggregated diffusion server."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if torch.cuda.device_count() < 2:
        pytest.skip("Requires 2+ GPUs for disaggregated mode")

    from sglang.multimodal_gen.runtime.launch_server import launch_pool_disagg_server
    from sglang.multimodal_gen.runtime.server_args import ServerArgs

    model = os.environ.get("DISAGG_MODEL", DEFAULT_MODEL)
    port = int(os.environ.get("SGLANG_TEST_SERVER_PORT", "30088"))

    encoder_gpus = _parse_gpu_list("DISAGG_ENCODER_GPUS", "0")
    denoiser_gpus = _parse_gpu_list("DISAGG_DENOISER_GPUS", "1")
    decoder_gpus = _parse_gpu_list("DISAGG_DECODER_GPUS", "0")

    server_args = ServerArgs.from_kwargs(
        model_path=model,
        port=port,
        host="127.0.0.1",
        warmup=True,
        text_encoder_cpu_offload=True,
        pin_cpu_memory=True,
    )

    logger.info(
        "Starting pool disagg server: model=%s, port=%d, "
        "encoder_gpus=%s, denoiser_gpus=%s, decoder_gpus=%s",
        model,
        port,
        encoder_gpus,
        denoiser_gpus,
        decoder_gpus,
    )

    # Launch in a subprocess so we can clean it up
    ctx = mp.get_context("spawn")
    server_process = ctx.Process(
        target=launch_pool_disagg_server,
        args=(server_args, encoder_gpus, denoiser_gpus, decoder_gpus, True),
        daemon=True,
    )
    server_process.start()

    # Wait for server to be ready
    import requests

    deadline = time.time() + float(os.environ.get("SGLANG_TEST_WAIT_SECS", "600"))
    ready = False
    while time.time() < deadline:
        try:
            resp = requests.get(f"http://127.0.0.1:{port}/health", timeout=2)
            if resp.status_code == 200:
                ready = True
                break
        except Exception:
            pass
        time.sleep(2)

    if not ready:
        server_process.kill()
        pytest.fail("Disagg server did not become ready in time")

    yield {"port": port, "model": model, "process": server_process}

    # Cleanup
    server_process.kill()
    server_process.join(timeout=10)


class TestDisaggServer:
    """End-to-end tests for pool-mode disaggregated diffusion server."""

    def test_video_generation(self, disagg_server):
        """Test T2V generation through the full disagg pipeline."""
        from openai import OpenAI

        port = disagg_server["port"]
        model = disagg_server["model"]

        client = OpenAI(
            base_url=f"http://127.0.0.1:{port}/v1",
            api_key="unused",
        )

        prompt = "A curious raccoon exploring a garden"

        job = client.videos.create(
            model=model,
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
        timeout = 300
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

        resp = client.videos.download_content(video_id=video_id)
        content = resp.read()

        assert len(content) > 0, "Empty video content"
        logger.info(
            "Video generation completed: job=%s, size=%d bytes",
            video_id,
            len(content),
        )

    def test_health_check(self, disagg_server):
        """Test that the health endpoint works."""
        import requests

        port = disagg_server["port"]
        resp = requests.get(f"http://127.0.0.1:{port}/health")
        assert resp.status_code == 200
