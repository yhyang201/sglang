# SPDX-License-Identifier: Apache-2.0
"""End-to-end tests for disaggregated diffusion server (relay & P2P modes).

Launches 4 CLI processes (encoder, denoiser, decoder, server head) via
``sglang serve --disagg-role ...``, sends a video generation request via
the OpenAI-compatible API, and validates the output.

Usage:
    # Relay mode (default) — requires 2+ GPUs
    python -m pytest python/sglang/multimodal_gen/test/server/test_disagg_server.py -v -s -k relay

    # P2P mode — requires 2+ GPUs + Mooncake TransferEngine
    python -m pytest python/sglang/multimodal_gen/test/server/test_disagg_server.py -v -s -k p2p

    # With custom GPU assignment
    DISAGG_ENCODER_GPU=0 DISAGG_DENOISER_GPU=1 DISAGG_DECODER_GPU=2 \
        python -m pytest python/sglang/multimodal_gen/test/server/test_disagg_server.py -v -s
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
import tempfile
import threading
import time
from collections import OrderedDict
from pathlib import Path

import pytest

from sglang.multimodal_gen.runtime.utils.common import kill_process_tree
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.test.test_utils import find_free_port

logger = init_logger(__name__)

# Model for testing — small enough for CI
DEFAULT_MODEL = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"

HOST = "127.0.0.1"


def _start_process(command: list[str], log_path: Path, env: dict) -> subprocess.Popen:
    """Start a subprocess and stream its output to a log file."""
    logger.info("Running: %s", shlex.join(command))
    logger.info("  Log: %s", log_path)

    fh = log_path.open("w", encoding="utf-8", buffering=1)
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )

    def _drain(pipe, file):
        try:
            with pipe:
                for line in iter(pipe.readline, ""):
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    file.write(line)
                    file.flush()
        except Exception:
            pass

    t = threading.Thread(target=_drain, args=(proc.stdout, fh), daemon=True)
    t.start()
    return proc


def _wait_for_log_message(
    log_path: Path,
    proc: subprocess.Popen,
    message: str,
    timeout: float,
    label: str = "",
) -> None:
    """Poll log file until a specific message appears."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"{label or 'Process'} exited early (code {proc.returncode}). "
                f"See log: {log_path}"
            )
        if log_path.exists():
            try:
                content = log_path.read_text(encoding="utf-8", errors="ignore")
                if message in content:
                    return
            except Exception:
                pass
        time.sleep(1)
    raise TimeoutError(
        f"{label or 'Process'} not ready within {timeout}s. See log: {log_path}"
    )


def _launch_disagg_cluster(p2p_mode: bool = False):
    """Launch a disaggregated diffusion cluster (4 CLI processes).

    Args:
        p2p_mode: If True, pass ``--disagg-p2p-mode`` to enable RDMA/P2P transfer.
    """
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if torch.cuda.device_count() < 2:
        pytest.skip("Requires 2+ GPUs for disaggregated mode")

    model = os.environ.get("DISAGG_MODEL", DEFAULT_MODEL)
    encoder_gpu = int(os.environ.get("DISAGG_ENCODER_GPU", "0"))
    denoiser_gpu = int(os.environ.get("DISAGG_DENOISER_GPU", "1"))
    decoder_gpu = int(os.environ.get("DISAGG_DECODER_GPU", "0"))

    # Allocate ports dynamically
    http_port = find_free_port(HOST)
    server_scheduler_port = find_free_port(HOST)
    encoder_port = find_free_port(HOST)
    denoiser_port = find_free_port(HOST)
    decoder_port = find_free_port(HOST)

    ds_addr = f"tcp://{HOST}:{server_scheduler_port}"
    mode_label = "p2p" if p2p_mode else "relay"
    log_dir = Path(tempfile.mkdtemp(prefix=f"sglang_disagg_{mode_label}_test_"))
    env = os.environ.copy()

    wait_timeout = float(os.environ.get("SGLANG_TEST_WAIT_SECS", "600"))

    processes: list[tuple[str, subprocess.Popen]] = []

    def _cleanup():
        for name, proc in processes:
            if proc.poll() is None:
                logger.info("Killing %s (pid=%d)...", name, proc.pid)
                kill_process_tree(proc.pid)
        for name, proc in processes:
            proc.wait(timeout=10)

    try:
        # 1-3. Launch role instances.
        # Roles sharing a GPU are launched sequentially to avoid port/init
        # contention; roles on different GPUs launch in declaration order.
        roles = [
            ("encoder", encoder_port, encoder_gpu),
            ("denoiser", denoiser_port, denoiser_gpu),
            ("decoder", decoder_port, decoder_gpu),
        ]

        gpu_groups: dict[int, list] = OrderedDict()
        for role_name, port, gpu_id in roles:
            gpu_groups.setdefault(gpu_id, []).append((role_name, port, gpu_id))

        def _launch_and_wait(role_name, port, gpu_id):
            cmd = [
                "sglang",
                "serve",
                "--model-path",
                model,
                "--disagg-role",
                role_name,
                "--disagg-server-addr",
                ds_addr,
                "--scheduler-port",
                str(port),
                "--num-gpus",
                "1",
                "--base-gpu-id",
                str(gpu_id),
                "--log-level",
                "debug",
            ]
            if p2p_mode:
                cmd += ["--disagg-p2p-mode"]
            log_path = log_dir / f"{role_name}.log"
            proc = _start_process(cmd, log_path, env)
            processes.append((role_name, proc))

            ready_msg = f"Role {role_name.upper()} ready"
            logger.info("Waiting for %s to be ready...", role_name)
            try:
                _wait_for_log_message(
                    log_dir / f"{role_name}.log",
                    proc,
                    ready_msg,
                    wait_timeout,
                    role_name,
                )
            except (RuntimeError, TimeoutError) as e:
                _cleanup()
                pytest.fail(str(e))
            logger.info("%s is ready.", role_name)

        for gpu_id, group in gpu_groups.items():
            for role_name, port, gid in group:
                _launch_and_wait(role_name, port, gid)

        # 4. Launch DiffusionServer head
        server_cmd = [
            "sglang",
            "serve",
            "--model-path",
            model,
            "--disagg-role",
            "server",
            "--encoder-urls",
            f"tcp://{HOST}:{encoder_port}",
            "--denoiser-urls",
            f"tcp://{HOST}:{denoiser_port}",
            "--decoder-urls",
            f"tcp://{HOST}:{decoder_port}",
            "--scheduler-port",
            str(server_scheduler_port),
            "--port",
            str(http_port),
            "--host",
            HOST,
            "--log-level",
            "debug",
        ]
        if p2p_mode:
            server_cmd += ["--disagg-p2p-mode"]
        server_log = log_dir / "server.log"
        server_proc = _start_process(server_cmd, server_log, env)
        processes.append(("server", server_proc))

        # Wait for HTTP server to be ready
        logger.info("Waiting for HTTP server at port %d...", http_port)
        _wait_for_log_message(
            server_log,
            server_proc,
            "Application startup complete.",
            wait_timeout,
            "server",
        )
        logger.info("All components ready! (mode=%s)", mode_label)

        yield {"port": http_port, "model": model, "log_dir": log_dir}

    finally:
        _cleanup()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def disagg_server_relay():
    """Disaggregated cluster using relay (ZMQ) transport."""
    yield from _launch_disagg_cluster(p2p_mode=False)


@pytest.fixture(scope="module")
def disagg_server_p2p():
    """Disaggregated cluster using P2P (RDMA/Mooncake) transport."""
    yield from _launch_disagg_cluster(p2p_mode=True)


# ---------------------------------------------------------------------------
# Shared test logic
# ---------------------------------------------------------------------------


def _test_health_check(server_info):
    import requests

    port = server_info["port"]
    resp = requests.get(f"http://{HOST}:{port}/health")
    assert resp.status_code == 200


def _test_video_generation(server_info):
    from openai import OpenAI

    port = server_info["port"]
    model = server_info["model"]

    client = OpenAI(
        base_url=f"http://{HOST}:{port}/v1",
        api_key="unused",
    )

    prompt = "A curious raccoon exploring a garden"

    job = client.videos.create(
        model=model,
        prompt=prompt,
        size="832x480",
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


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


class TestDisaggServerRelay:
    """End-to-end tests for disaggregated diffusion server (relay mode)."""

    def test_health_check(self, disagg_server_relay):
        _test_health_check(disagg_server_relay)

    def test_video_generation(self, disagg_server_relay):
        _test_video_generation(disagg_server_relay)


class TestDisaggServerP2P:
    """End-to-end tests for disaggregated diffusion server (P2P mode)."""

    def test_health_check(self, disagg_server_p2p):
        _test_health_check(disagg_server_p2p)

    def test_video_generation(self, disagg_server_p2p):
        _test_video_generation(disagg_server_p2p)
