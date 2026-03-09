# SPDX-License-Identifier: Apache-2.0
"""Integration tests for disaggregated pipeline IPC (pool mode).

Tests the tensor transport layer and pool-mode pack/unpack helpers
using ZMQ multipart with realistic tensor shapes.
"""

import threading
import time
import unittest

import torch
import zmq

from sglang.multimodal_gen.runtime.disaggregation.role_connector import (
    _extract_scalar_fields,
    _extract_tensor_fields,
    build_req_from_frames,
    pack_denoiser_output,
    pack_encoder_output,
)
from sglang.multimodal_gen.runtime.disaggregation.tensor_transport import (
    recv_tensors,
    send_tensors,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req


class TestPoolModePackUnpack(unittest.TestCase):
    """Test pool-mode pack/unpack helpers for DiffusionServer relay."""

    def test_pack_unpack_encoder_output(self):
        """Pack encoder output and reconstruct via build_req_from_frames."""
        req = Req(
            prompt="a curious raccoon",
            request_id="pool-e2d-001",
            guidance_scale=1.0,
        )
        req.prompt_embeds = [torch.randn(1, 512, 4096, dtype=torch.bfloat16)]
        req.latents = torch.randn(1, 16, 21, 30, 52, dtype=torch.bfloat16)
        req.timesteps = torch.linspace(999, 0, 50)
        req.do_classifier_free_guidance = False
        req.height = 480
        req.width = 832
        req.num_frames = 81
        req.num_inference_steps = 50
        req.is_prompt_processed = True

        # Pack
        metadata_bytes, buffers = pack_encoder_output(req)
        parts = [metadata_bytes] + buffers

        # Unpack
        recv_req = build_req_from_frames(parts, "encoder_to_denoiser")

        # Verify tensor fields
        self.assertEqual(recv_req.prompt_embeds[0].shape, torch.Size([1, 512, 4096]))
        self.assertEqual(recv_req.prompt_embeds[0].dtype, torch.bfloat16)
        self.assertEqual(recv_req.latents.shape, torch.Size([1, 16, 21, 30, 52]))
        self.assertEqual(len(recv_req.timesteps), 50)

        # Verify scalar fields
        self.assertEqual(recv_req.height, 480)
        self.assertEqual(recv_req.width, 832)
        self.assertEqual(recv_req.num_frames, 81)
        self.assertFalse(recv_req.do_classifier_free_guidance)

    def test_pack_unpack_denoiser_output(self):
        """Pack denoiser output and reconstruct via build_req_from_frames."""
        req = Req(
            prompt="",
            request_id="pool-d2d-001",
            guidance_scale=1.0,
        )
        req.latents = torch.randn(1, 16, 21, 30, 52, dtype=torch.bfloat16)
        req.height = 480
        req.width = 832
        req.num_frames = 81
        req.raw_latent_shape = [1, 16, 21, 30, 52]

        metadata_bytes, buffers = pack_denoiser_output(req)
        parts = [metadata_bytes] + buffers

        recv_req = build_req_from_frames(parts, "denoiser_to_decoder")

        self.assertEqual(recv_req.latents.shape, torch.Size([1, 16, 21, 30, 52]))
        self.assertEqual(recv_req.height, 480)

    def test_build_req_from_frames_invalid_transition(self):
        """build_req_from_frames should raise on unknown transition."""
        req = Req(prompt="", request_id="t", guidance_scale=1.0)
        req.latents = torch.randn(1, 4, 8, 16, 16)
        metadata_bytes, buffers = pack_encoder_output(req)
        parts = [metadata_bytes] + buffers

        with self.assertRaises(ValueError):
            build_req_from_frames(parts, "invalid_transition")


class TestFullDisaggFlow(unittest.TestCase):
    """Test the complete Encoder -> Denoiser -> Decoder flow using raw transport."""

    def test_three_role_pipeline(self):
        """Simulate a full disagg pipeline with 3 threads representing 3 roles."""
        context = zmq.Context()

        # Allocate sockets
        e2d_push = context.socket(zmq.PUSH)
        e2d_port = e2d_push.bind_to_random_port("tcp://127.0.0.1")
        e2d_pull = context.socket(zmq.PULL)
        e2d_pull.connect(f"tcp://127.0.0.1:{e2d_port}")

        d2d_push = context.socket(zmq.PUSH)
        d2d_port = d2d_push.bind_to_random_port("tcp://127.0.0.1")
        d2d_pull = context.socket(zmq.PULL)
        d2d_pull.connect(f"tcp://127.0.0.1:{d2d_port}")

        d2e_push = context.socket(zmq.PUSH)
        d2e_port = d2e_push.bind_to_random_port("tcp://127.0.0.1")
        d2e_pull = context.socket(zmq.PULL)
        d2e_pull.connect(f"tcp://127.0.0.1:{d2e_port}")

        time.sleep(0.1)

        errors = []
        final_output = [None]

        def encoder_role():
            try:
                prompt_embeds = [torch.randn(1, 512, 4096, dtype=torch.bfloat16)]
                latents = torch.randn(1, 16, 21, 30, 52, dtype=torch.bfloat16)
                timesteps = torch.linspace(999, 0, 50)

                send_tensors(
                    e2d_push,
                    {
                        "prompt_embeds": prompt_embeds,
                        "latents": latents,
                        "timesteps": timesteps,
                    },
                    {
                        "request_id": "flow-test-001",
                        "height": 480,
                        "width": 832,
                        "num_frames": 81,
                        "num_inference_steps": 50,
                    },
                )

                result_tf, result_sf = recv_tensors(d2e_pull)
                final_output[0] = result_tf.get("output")
            except Exception as e:
                errors.append(("encoder", e))

        def denoiser_role():
            try:
                tf, sf = recv_tensors(e2d_pull)
                latents = tf["latents"]
                denoised_latents = latents * 0.5

                send_tensors(
                    d2d_push,
                    {"latents": denoised_latents},
                    {
                        "request_id": sf["request_id"],
                        "height": sf["height"],
                        "width": sf["width"],
                        "num_frames": sf["num_frames"],
                    },
                )
            except Exception as e:
                errors.append(("denoiser", e))

        def decoder_role():
            try:
                tf, sf = recv_tensors(d2d_pull)
                output = torch.randn(
                    1,
                    3,
                    sf["num_frames"],
                    sf["height"],
                    sf["width"],
                    dtype=torch.float32,
                )

                send_tensors(
                    d2e_push,
                    {"output": output},
                    {"request_id": sf["request_id"]},
                )
            except Exception as e:
                errors.append(("decoder", e))

        threads = [
            threading.Thread(target=encoder_role),
            threading.Thread(target=denoiser_role),
            threading.Thread(target=decoder_role),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        self.assertEqual(len(errors), 0, f"Errors: {errors}")
        self.assertIsNotNone(final_output[0])
        self.assertEqual(final_output[0].shape, torch.Size([1, 3, 81, 480, 832]))

        for s in [e2d_push, e2d_pull, d2d_push, d2d_pull, d2e_push, d2e_pull]:
            s.close()
        context.term()

    def test_multiple_requests(self):
        """Test that the disagg flow handles multiple sequential requests."""
        context = zmq.Context()

        e2d_push = context.socket(zmq.PUSH)
        e2d_port = e2d_push.bind_to_random_port("tcp://127.0.0.1")
        e2d_pull = context.socket(zmq.PULL)
        e2d_pull.connect(f"tcp://127.0.0.1:{e2d_port}")

        d2e_push = context.socket(zmq.PUSH)
        d2e_port = d2e_push.bind_to_random_port("tcp://127.0.0.1")
        d2e_pull = context.socket(zmq.PULL)
        d2e_pull.connect(f"tcp://127.0.0.1:{d2e_port}")

        time.sleep(0.1)

        n_requests = 3
        results = []

        def worker():
            """Simulates denoiser+decoder as a single worker."""
            for _ in range(n_requests):
                tf, sf = recv_tensors(e2d_pull)
                output = torch.ones(1, 3, 8, 16, 16) * sf["idx"]
                send_tensors(d2e_push, {"output": output}, {"idx": sf["idx"]})

        worker_thread = threading.Thread(target=worker)
        worker_thread.start()

        for i in range(n_requests):
            send_tensors(
                e2d_push,
                {"latents": torch.randn(1, 4, 8, 16, 16)},
                {"idx": i},
            )
            tf, sf = recv_tensors(d2e_pull)
            results.append((sf["idx"], tf["output"]))

        worker_thread.join(timeout=5)

        self.assertEqual(len(results), n_requests)
        for i, (idx, output) in enumerate(results):
            self.assertEqual(idx, i)
            self.assertTrue(torch.all(output == i))

        for s in [e2d_push, e2d_pull, d2e_push, d2e_pull]:
            s.close()
        context.term()


class TestAsyncPipelining(unittest.TestCase):
    """Test async pipelining — multiple in-flight requests."""

    def test_multiple_concurrent_requests(self):
        """Encoder sends N requests without blocking, then collects results."""
        context = zmq.Context()

        e2w_push = context.socket(zmq.PUSH)
        e2w_port = e2w_push.bind_to_random_port("tcp://127.0.0.1")
        e2w_pull = context.socket(zmq.PULL)
        e2w_pull.connect(f"tcp://127.0.0.1:{e2w_port}")

        w2e_push = context.socket(zmq.PUSH)
        w2e_port = w2e_push.bind_to_random_port("tcp://127.0.0.1")
        w2e_pull = context.socket(zmq.PULL)
        w2e_pull.connect(f"tcp://127.0.0.1:{w2e_port}")

        time.sleep(0.1)

        n_requests = 5
        errors = []

        def worker():
            try:
                for _ in range(n_requests):
                    tf, sf = recv_tensors(e2w_pull)
                    time.sleep(0.05)
                    output = torch.ones(1, 3, 8, 16, 16) * sf["request_idx"]
                    send_tensors(
                        w2e_push,
                        {"output": output},
                        {
                            "request_id": sf["request_id"],
                            "request_idx": sf["request_idx"],
                        },
                    )
            except Exception as e:
                errors.append(("worker", e))

        worker_thread = threading.Thread(target=worker)
        worker_thread.start()

        pending = {}
        for i in range(n_requests):
            request_id = f"async-test-{i:03d}"
            send_tensors(
                e2w_push,
                {"latents": torch.randn(1, 4, 8, 16, 16)},
                {"request_id": request_id, "request_idx": i},
            )
            pending[request_id] = i

        results = {}
        while len(results) < n_requests:
            tf, sf = recv_tensors(w2e_pull)
            rid = sf["request_id"]
            results[rid] = (sf["request_idx"], tf["output"])

        worker_thread.join(timeout=10)

        self.assertEqual(len(errors), 0, f"Errors: {errors}")
        self.assertEqual(len(results), n_requests)
        for i in range(n_requests):
            rid = f"async-test-{i:03d}"
            idx, output = results[rid]
            self.assertEqual(idx, i)
            self.assertTrue(torch.all(output == i))

        for s in [e2w_push, e2w_pull, w2e_push, w2e_pull]:
            s.close()
        context.term()

    def test_noblock_recv_drains_multiple(self):
        """NOBLOCK recv should drain all available messages without blocking."""
        context = zmq.Context()

        push = context.socket(zmq.PUSH)
        port = push.bind_to_random_port("tcp://127.0.0.1")
        pull = context.socket(zmq.PULL)
        pull.connect(f"tcp://127.0.0.1:{port}")

        time.sleep(0.1)

        for i in range(3):
            send_tensors(push, {}, {"idx": i})

        time.sleep(0.1)

        collected = []
        while True:
            try:
                _, sf = recv_tensors(pull, flags=zmq.NOBLOCK)
                collected.append(sf["idx"])
            except zmq.Again:
                break

        self.assertEqual(collected, [0, 1, 2])

        push.close()
        pull.close()
        context.term()


class TestExtractFields(unittest.TestCase):
    """Test the field extraction helpers from role_connector."""

    def test_extract_tensor_fields_from_req(self):
        req = Req(prompt="test", guidance_scale=1.0, request_id="t1")
        req.prompt_embeds = [torch.randn(1, 77, 768)]
        req.latents = torch.randn(1, 4, 16, 32)
        req.timesteps = torch.tensor([999.0])

        fields = _extract_tensor_fields(
            req, ["prompt_embeds", "latents", "timesteps", "y"]
        )
        self.assertIn("prompt_embeds", fields)
        self.assertIn("latents", fields)
        self.assertIn("timesteps", fields)
        self.assertNotIn("y", fields)

    def test_extract_scalar_fields_from_req(self):
        req = Req(
            prompt="test",
            guidance_scale=7.5,
            request_id="t1",
            height=480,
            width=832,
        )
        req.do_classifier_free_guidance = True
        req.num_inference_steps = 50

        fields = _extract_scalar_fields(
            req,
            [
                "request_id",
                "guidance_scale",
                "height",
                "width",
                "do_classifier_free_guidance",
                "num_inference_steps",
                "sigmas",
            ],
        )
        self.assertEqual(fields["request_id"], "t1")
        self.assertEqual(fields["guidance_scale"], 7.5)
        self.assertEqual(fields["height"], 480)
        self.assertTrue(fields["do_classifier_free_guidance"])
        self.assertNotIn("sigmas", fields)


if __name__ == "__main__":
    unittest.main()
