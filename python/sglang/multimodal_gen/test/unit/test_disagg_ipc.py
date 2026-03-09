# SPDX-License-Identifier: Apache-2.0
"""Integration tests for disaggregated pipeline IPC (Phase 2).

Tests the full Encoder -> Denoiser -> Decoder communication flow
using ZMQ multipart with realistic tensor shapes.
"""

import threading
import time
import unittest

import torch
import zmq

from sglang.multimodal_gen.runtime.disaggregation.role_connector import (
    DENOISER_TO_DECODER_SCALAR_FIELDS,
    DENOISER_TO_DECODER_TENSOR_FIELDS,
    ENCODER_TO_DENOISER_SCALAR_FIELDS,
    ENCODER_TO_DENOISER_TENSOR_FIELDS,
    RoleConnectorReceiver,
    RoleConnectorSender,
    _extract_scalar_fields,
    _extract_tensor_fields,
)
from sglang.multimodal_gen.runtime.disaggregation.tensor_transport import (
    recv_tensors,
    send_tensors,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req


class TestRoleConnectorSendRecv(unittest.TestCase):
    """Test RoleConnectorSender/Receiver with Req objects."""

    def setUp(self):
        self.context = zmq.Context()

    def tearDown(self):
        self.context.term()

    def test_encoder_to_denoiser_with_req(self):
        """Test sending Req fields from encoder to denoiser."""
        # Bind sender (encoder side)
        sender = RoleConnectorSender(
            self.context,
            "tcp://127.0.0.1:*",
            ENCODER_TO_DENOISER_TENSOR_FIELDS,
            ENCODER_TO_DENOISER_SCALAR_FIELDS,
        )
        endpoint = sender.endpoint

        # Connect receiver (denoiser side)
        receiver = RoleConnectorReceiver(
            self.context,
            endpoint,
            ENCODER_TO_DENOISER_TENSOR_FIELDS,
            ENCODER_TO_DENOISER_SCALAR_FIELDS,
        )
        time.sleep(0.1)

        # Create a Req mimicking encoder output (Wan2.1 T2V)
        req = Req(
            prompt="a curious raccoon",
            request_id="test-e2d-001",
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

        # Send
        sender.send(req)

        # Receive
        recv_req = receiver.recv()

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

        sender.close()
        receiver.close()

    def test_denoiser_to_decoder_with_req(self):
        """Test sending Req fields from denoiser to decoder."""
        sender = RoleConnectorSender(
            self.context,
            "tcp://127.0.0.1:*",
            DENOISER_TO_DECODER_TENSOR_FIELDS,
            DENOISER_TO_DECODER_SCALAR_FIELDS,
        )
        endpoint = sender.endpoint

        receiver = RoleConnectorReceiver(
            self.context,
            endpoint,
            DENOISER_TO_DECODER_TENSOR_FIELDS,
            DENOISER_TO_DECODER_SCALAR_FIELDS,
        )
        time.sleep(0.1)

        # Create a Req mimicking denoiser output
        req = Req(
            prompt="",
            request_id="test-d2d-001",
            guidance_scale=1.0,
        )
        req.latents = torch.randn(1, 16, 21, 30, 52, dtype=torch.bfloat16)
        req.height = 480
        req.width = 832
        req.num_frames = 81
        req.raw_latent_shape = [1, 16, 21, 30, 52]

        sender.send(req)
        recv_req = receiver.recv()

        self.assertEqual(recv_req.latents.shape, torch.Size([1, 16, 21, 30, 52]))
        self.assertEqual(recv_req.height, 480)

        sender.close()
        receiver.close()

    def test_nonblocking_try_recv(self):
        """Test try_recv returns None when no message available."""
        sender = RoleConnectorSender(
            self.context,
            "tcp://127.0.0.1:*",
            ENCODER_TO_DENOISER_TENSOR_FIELDS,
            ENCODER_TO_DENOISER_SCALAR_FIELDS,
        )
        receiver = RoleConnectorReceiver(
            self.context,
            sender.endpoint,
            ENCODER_TO_DENOISER_TENSOR_FIELDS,
            ENCODER_TO_DENOISER_SCALAR_FIELDS,
        )
        time.sleep(0.1)

        result = receiver.try_recv()
        self.assertIsNone(result)

        sender.close()
        receiver.close()


class TestFullDisaggFlow(unittest.TestCase):
    """Test the complete Encoder -> Denoiser -> Decoder -> Encoder flow."""

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

        # Simulate encoder
        def encoder_role():
            try:
                # 1. "Run encoder stages" - produce embeddings + latents
                prompt_embeds = [torch.randn(1, 512, 4096, dtype=torch.bfloat16)]
                latents = torch.randn(1, 16, 21, 30, 52, dtype=torch.bfloat16)
                timesteps = torch.linspace(999, 0, 50)

                # 2. Send to denoiser
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

                # 3. Wait for decoder result
                result_tf, result_sf = recv_tensors(d2e_pull)
                final_output[0] = result_tf.get("output")
            except Exception as e:
                errors.append(("encoder", e))

        # Simulate denoiser
        def denoiser_role():
            try:
                # 1. Receive from encoder
                tf, sf = recv_tensors(e2d_pull)

                # 2. "Run denoising" - just modify latents
                latents = tf["latents"]
                denoised_latents = latents * 0.5  # simulate denoising

                # 3. Send to decoder
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

        # Simulate decoder
        def decoder_role():
            try:
                # 1. Receive from denoiser
                tf, sf = recv_tensors(d2d_pull)

                # 2. "Run VAE decode" - produce output
                latents = tf["latents"]
                output = torch.randn(
                    1,
                    3,
                    sf["num_frames"],
                    sf["height"],
                    sf["width"],
                    dtype=torch.float32,
                )

                # 3. Send result back to encoder
                send_tensors(
                    d2e_push,
                    {"output": output},
                    {
                        "request_id": sf["request_id"],
                    },
                )
            except Exception as e:
                errors.append(("decoder", e))

        # Run all three roles concurrently
        threads = [
            threading.Thread(target=encoder_role),
            threading.Thread(target=denoiser_role),
            threading.Thread(target=decoder_role),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        # Verify
        self.assertEqual(len(errors), 0, f"Errors: {errors}")
        self.assertIsNotNone(final_output[0])
        self.assertEqual(final_output[0].shape, torch.Size([1, 3, 81, 480, 832]))

        # Cleanup
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
                # "Process" and return
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
        # y is None, should not be included
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
        # sigmas is None, should not be included
        self.assertNotIn("sigmas", fields)


if __name__ == "__main__":
    unittest.main()
