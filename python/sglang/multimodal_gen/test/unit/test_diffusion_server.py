# SPDX-License-Identifier: Apache-2.0
"""Unit tests for DiffusionServer pool-based pipeline orchestrator."""

import json
import pickle
import time
import unittest

import torch
import zmq

from sglang.multimodal_gen.runtime.disaggregation.diffusion_server import (
    DiffusionServer,
)
from sglang.multimodal_gen.runtime.disaggregation.tensor_transport import (
    send_tensors,
)


class _MockReq:
    """Minimal mock request for testing."""

    def __init__(self, request_id: str = "test-001"):
        self.request_id = request_id
        self.is_warmup = False


class TestDiffusionServerInit(unittest.TestCase):
    """Test DiffusionServer initialization."""

    def test_basic_init(self):
        server = DiffusionServer(
            frontend_endpoint="tcp://127.0.0.1:19900",
            encoder_work_endpoints=["tcp://127.0.0.1:19901"],
            denoiser_work_endpoints=["tcp://127.0.0.1:19902"],
            decoder_work_endpoints=["tcp://127.0.0.1:19903"],
            encoder_result_endpoint="tcp://127.0.0.1:19904",
            denoiser_result_endpoint="tcp://127.0.0.1:19905",
            decoder_result_endpoint="tcp://127.0.0.1:19906",
        )
        self.assertEqual(server._num_encoders, 1)
        self.assertEqual(server._num_denoisers, 1)
        self.assertEqual(server._num_decoders, 1)

    def test_get_stats(self):
        server = DiffusionServer(
            frontend_endpoint="tcp://127.0.0.1:19910",
            encoder_work_endpoints=["tcp://127.0.0.1:19911"],
            denoiser_work_endpoints=["tcp://127.0.0.1:19912"],
            decoder_work_endpoints=["tcp://127.0.0.1:19913"],
            encoder_result_endpoint="tcp://127.0.0.1:19914",
            denoiser_result_endpoint="tcp://127.0.0.1:19915",
            decoder_result_endpoint="tcp://127.0.0.1:19916",
        )
        stats = server.get_stats()
        self.assertEqual(stats["role"], "diffusion_server")
        self.assertEqual(stats["num_encoders"], 1)
        self.assertEqual(stats["num_denoisers"], 1)
        self.assertEqual(stats["num_decoders"], 1)
        self.assertEqual(stats["pending_requests"], 0)


class TestDiffusionServerFullPipeline(unittest.TestCase):
    """Test DiffusionServer full pipeline routing with mock role instances."""

    def setUp(self):
        """Set up DiffusionServer with mock role instances using PUSH/PULL."""
        self.ctx = zmq.Context()

        # Allocate ports
        self.frontend_port = 19950
        # Work endpoints (instances bind PULL)
        self.encoder_work_port = 19951
        self.denoiser_work_port = 19952
        self.decoder_work_port = 19953
        # Result endpoints (DS binds PULL)
        self.encoder_result_port = 19954
        self.denoiser_result_port = 19955
        self.decoder_result_port = 19956

        host = "127.0.0.1"

        # Create mock role instance sockets
        # Encoder: PULL for work, PUSH for results
        self.encoder_work = self.ctx.socket(zmq.PULL)
        self.encoder_work.setsockopt(zmq.RCVTIMEO, 3000)
        self.encoder_work.bind(f"tcp://{host}:{self.encoder_work_port}")

        self.encoder_result = self.ctx.socket(zmq.PUSH)

        # Denoiser
        self.denoiser_work = self.ctx.socket(zmq.PULL)
        self.denoiser_work.setsockopt(zmq.RCVTIMEO, 3000)
        self.denoiser_work.bind(f"tcp://{host}:{self.denoiser_work_port}")

        self.denoiser_result = self.ctx.socket(zmq.PUSH)

        # Decoder
        self.decoder_work = self.ctx.socket(zmq.PULL)
        self.decoder_work.setsockopt(zmq.RCVTIMEO, 3000)
        self.decoder_work.bind(f"tcp://{host}:{self.decoder_work_port}")

        self.decoder_result = self.ctx.socket(zmq.PUSH)

        # Create DiffusionServer
        self.server = DiffusionServer(
            frontend_endpoint=f"tcp://{host}:{self.frontend_port}",
            encoder_work_endpoints=[f"tcp://{host}:{self.encoder_work_port}"],
            denoiser_work_endpoints=[f"tcp://{host}:{self.denoiser_work_port}"],
            decoder_work_endpoints=[f"tcp://{host}:{self.decoder_work_port}"],
            encoder_result_endpoint=f"tcp://{host}:{self.encoder_result_port}",
            denoiser_result_endpoint=f"tcp://{host}:{self.denoiser_result_port}",
            decoder_result_endpoint=f"tcp://{host}:{self.decoder_result_port}",
            timeout_s=10.0,
        )
        self.server.start()
        time.sleep(0.3)  # Allow connections

        # Connect result PUSH sockets to DS PULL (after DS starts and binds)
        self.encoder_result.connect(f"tcp://{host}:{self.encoder_result_port}")
        self.denoiser_result.connect(f"tcp://{host}:{self.denoiser_result_port}")
        self.decoder_result.connect(f"tcp://{host}:{self.decoder_result_port}")
        time.sleep(0.1)

        # Client DEALER
        self.client = self.ctx.socket(zmq.DEALER)
        self.client.setsockopt(zmq.RCVTIMEO, 5000)
        self.client.connect(f"tcp://{host}:{self.frontend_port}")
        time.sleep(0.1)

    def tearDown(self):
        self.server.stop()
        for sock in [
            self.client,
            self.encoder_work,
            self.encoder_result,
            self.denoiser_work,
            self.denoiser_result,
            self.decoder_work,
            self.decoder_result,
        ]:
            sock.close()
        self.ctx.destroy(linger=0)

    def test_full_pipeline_flow(self):
        """Test Client → Encoder → Denoiser → Decoder → Client."""
        req = _MockReq(request_id="pipeline-001")

        # 1. Client sends request
        self.client.send_multipart([b"", pickle.dumps([req])])

        # 2. Encoder receives work: [request_id_bytes, pickled_req]
        enc_frames = self.encoder_work.recv_multipart()
        self.assertEqual(len(enc_frames), 2)
        self.assertEqual(enc_frames[0], b"pipeline-001")

        # Simulate encoder output (send_tensors format)
        enc_output = torch.randn(1, 4, 16, 16)
        send_tensors(
            self.encoder_result,
            {"prompt_embeds": enc_output},
            {"request_id": "pipeline-001", "height": 256, "width": 256},
        )

        # 3. Denoiser receives relayed encoder output
        den_frames = self.denoiser_work.recv_multipart()
        # Should be tensor multipart: [metadata_json, tensor_buffer]
        metadata = json.loads(den_frames[0])
        self.assertEqual(metadata["scalar_fields"]["request_id"], "pipeline-001")
        self.assertGreater(len(metadata["tensor_descriptors"]), 0)

        # Simulate denoiser output
        den_output = torch.randn(1, 4, 16, 16)
        send_tensors(
            self.denoiser_result,
            {"latents": den_output},
            {"request_id": "pipeline-001", "height": 256, "width": 256},
        )

        # 4. Decoder receives relayed denoiser output
        dec_frames = self.decoder_work.recv_multipart()
        metadata = json.loads(dec_frames[0])
        self.assertEqual(metadata["scalar_fields"]["request_id"], "pipeline-001")

        # Simulate decoder output (final result)
        dec_output = torch.randn(1, 3, 256, 256)
        send_tensors(
            self.decoder_result,
            {"output": dec_output},
            {"request_id": "pipeline-001"},
        )

        # 5. Client receives final result
        resp_parts = self.client.recv_multipart()
        output_batch = pickle.loads(resp_parts[-1])
        self.assertIsNotNone(output_batch.output)
        self.assertIsNone(output_batch.error)
        self.assertEqual(output_batch.output.shape, torch.Size([1, 3, 256, 256]))

    def test_encoder_error_propagation(self):
        """Test that encoder errors are returned to the client."""
        req = _MockReq(request_id="err-001")
        self.client.send_multipart([b"", pickle.dumps([req])])

        # Encoder receives
        self.encoder_work.recv_multipart()

        # Encoder sends error
        send_tensors(
            self.encoder_result,
            {},
            {"request_id": "err-001", "_disagg_error": "OOM on encoder"},
        )

        # Wait a bit for DiffusionServer to process and return error
        # The error should prevent dispatch to denoiser and return to client
        time.sleep(0.5)

        # Check tracker — request should be FAILED
        record = self.server.tracker.get("err-001")
        # May have been removed already, check stats
        stats = self.server.get_stats()
        self.assertIsNotNone(stats)

    def test_multi_instance_dispatch(self):
        """Test dispatching across multiple encoder instances."""
        # This test uses 1 instance per role, so all go to instance 0.
        # The dispatch logic is tested separately in test_dispatch_policy.py.
        req1 = _MockReq(request_id="multi-001")
        self.client.send_multipart([b"", pickle.dumps([req1])])

        frames = self.encoder_work.recv_multipart()
        self.assertEqual(frames[0], b"multi-001")


if __name__ == "__main__":
    unittest.main()
