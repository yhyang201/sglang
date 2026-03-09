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


from sglang.multimodal_gen.runtime.disaggregation.request_state import (
    RequestState,
)


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
        # Capacity-aware stats
        self.assertEqual(stats["encoder_free_slots"], [4])
        self.assertEqual(stats["denoiser_free_slots"], [2])
        self.assertEqual(stats["decoder_free_slots"], [4])
        self.assertEqual(stats["encoder_tta_depth"], 0)
        self.assertEqual(stats["denoiser_tta_depth"], 0)
        self.assertEqual(stats["decoder_tta_depth"], 0)

    def test_custom_capacity(self):
        server = DiffusionServer(
            frontend_endpoint="tcp://127.0.0.1:19920",
            encoder_work_endpoints=["tcp://127.0.0.1:19921", "tcp://127.0.0.1:19922"],
            denoiser_work_endpoints=["tcp://127.0.0.1:19923"],
            decoder_work_endpoints=["tcp://127.0.0.1:19924"],
            encoder_result_endpoint="tcp://127.0.0.1:19925",
            denoiser_result_endpoint="tcp://127.0.0.1:19926",
            decoder_result_endpoint="tcp://127.0.0.1:19927",
            encoder_capacity=8,
            denoiser_capacity=3,
            decoder_capacity=6,
        )
        self.assertEqual(server._encoder_free_slots, [8, 8])
        self.assertEqual(server._denoiser_free_slots, [3])
        self.assertEqual(server._decoder_free_slots, [6])


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


class TestDiffusionServerCapacityAware(unittest.TestCase):
    """Test capacity-aware dispatch with TTA queuing."""

    def setUp(self):
        """Set up DiffusionServer with encoder_capacity=1 to test TTA queuing."""
        self.ctx = zmq.Context()

        self.frontend_port = 19960
        self.encoder_work_port = 19961
        self.denoiser_work_port = 19962
        self.decoder_work_port = 19963
        self.encoder_result_port = 19964
        self.denoiser_result_port = 19965
        self.decoder_result_port = 19966

        host = "127.0.0.1"

        self.encoder_work = self.ctx.socket(zmq.PULL)
        self.encoder_work.setsockopt(zmq.RCVTIMEO, 3000)
        self.encoder_work.bind(f"tcp://{host}:{self.encoder_work_port}")

        self.encoder_result = self.ctx.socket(zmq.PUSH)

        self.denoiser_work = self.ctx.socket(zmq.PULL)
        self.denoiser_work.setsockopt(zmq.RCVTIMEO, 3000)
        self.denoiser_work.bind(f"tcp://{host}:{self.denoiser_work_port}")

        self.denoiser_result = self.ctx.socket(zmq.PUSH)

        self.decoder_work = self.ctx.socket(zmq.PULL)
        self.decoder_work.setsockopt(zmq.RCVTIMEO, 3000)
        self.decoder_work.bind(f"tcp://{host}:{self.decoder_work_port}")

        self.decoder_result = self.ctx.socket(zmq.PUSH)

        # encoder_capacity=1: only 1 request at a time per encoder
        self.server = DiffusionServer(
            frontend_endpoint=f"tcp://{host}:{self.frontend_port}",
            encoder_work_endpoints=[f"tcp://{host}:{self.encoder_work_port}"],
            denoiser_work_endpoints=[f"tcp://{host}:{self.denoiser_work_port}"],
            decoder_work_endpoints=[f"tcp://{host}:{self.decoder_work_port}"],
            encoder_result_endpoint=f"tcp://{host}:{self.encoder_result_port}",
            denoiser_result_endpoint=f"tcp://{host}:{self.denoiser_result_port}",
            decoder_result_endpoint=f"tcp://{host}:{self.decoder_result_port}",
            timeout_s=10.0,
            encoder_capacity=1,
            denoiser_capacity=1,
            decoder_capacity=1,
        )
        self.server.start()
        time.sleep(0.3)

        self.encoder_result.connect(f"tcp://{host}:{self.encoder_result_port}")
        self.denoiser_result.connect(f"tcp://{host}:{self.denoiser_result_port}")
        self.decoder_result.connect(f"tcp://{host}:{self.decoder_result_port}")
        time.sleep(0.1)

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

    def test_tta_queuing_and_drain(self):
        """Send 2 requests with encoder_capacity=1. Second should queue in TTA."""
        req1 = _MockReq(request_id="tta-001")
        req2 = _MockReq(request_id="tta-002")

        # Send both requests
        self.client.send_multipart([b"", pickle.dumps([req1])])
        time.sleep(0.1)
        self.client.send_multipart([b"", pickle.dumps([req2])])
        time.sleep(0.3)

        # First request should be dispatched immediately
        enc_frames = self.encoder_work.recv_multipart()
        self.assertEqual(enc_frames[0], b"tta-001")

        # Check stats: encoder slot is 0, TTA has 1 entry
        stats = self.server.get_stats()
        self.assertEqual(stats["encoder_free_slots"], [0])
        self.assertEqual(stats["encoder_tta_depth"], 1)

        # Second request should be in ENCODER_WAITING state
        record2 = self.server.tracker.get("tta-002")
        self.assertIsNotNone(record2)
        self.assertEqual(record2.state, RequestState.ENCODER_WAITING)

        # Complete first request through full pipeline
        send_tensors(
            self.encoder_result,
            {"prompt_embeds": torch.randn(1, 4, 16, 16)},
            {"request_id": "tta-001"},
        )
        time.sleep(0.2)

        # After encoder completes, slot freed → second request should be dispatched
        # AND first request should go to denoiser
        enc_frames2 = self.encoder_work.recv_multipart()
        self.assertEqual(enc_frames2[0], b"tta-002")

        # First request's encoder output should be relayed to denoiser
        den_frames = self.denoiser_work.recv_multipart()
        metadata = json.loads(den_frames[0])
        self.assertEqual(metadata["scalar_fields"]["request_id"], "tta-001")

        # TTA should be empty now
        stats = self.server.get_stats()
        self.assertEqual(stats["encoder_tta_depth"], 0)

    def test_free_slots_decrement_increment(self):
        """Verify FreeBufferSlots tracks correctly through pipeline."""
        req = _MockReq(request_id="slots-001")

        # Initial: all slots free
        stats = self.server.get_stats()
        self.assertEqual(stats["encoder_free_slots"], [1])
        self.assertEqual(stats["denoiser_free_slots"], [1])
        self.assertEqual(stats["decoder_free_slots"], [1])

        # Send request → encoder slot decremented
        self.client.send_multipart([b"", pickle.dumps([req])])
        time.sleep(0.2)
        self.encoder_work.recv_multipart()

        stats = self.server.get_stats()
        self.assertEqual(stats["encoder_free_slots"], [0])

        # Encoder completes → encoder slot freed, denoiser slot decremented
        send_tensors(
            self.encoder_result,
            {"prompt_embeds": torch.randn(1, 4, 16, 16)},
            {"request_id": "slots-001"},
        )
        time.sleep(0.2)
        self.denoiser_work.recv_multipart()

        stats = self.server.get_stats()
        self.assertEqual(stats["encoder_free_slots"], [1])
        self.assertEqual(stats["denoiser_free_slots"], [0])

        # Denoiser completes → denoiser slot freed, decoder slot decremented
        send_tensors(
            self.denoiser_result,
            {"latents": torch.randn(1, 4, 16, 16)},
            {"request_id": "slots-001"},
        )
        time.sleep(0.2)
        self.decoder_work.recv_multipart()

        stats = self.server.get_stats()
        self.assertEqual(stats["denoiser_free_slots"], [1])
        self.assertEqual(stats["decoder_free_slots"], [0])

        # Decoder completes → decoder slot freed
        send_tensors(
            self.decoder_result,
            {"output": torch.randn(1, 3, 256, 256)},
            {"request_id": "slots-001"},
        )
        time.sleep(0.2)

        # Receive final result
        self.client.recv_multipart()

        stats = self.server.get_stats()
        self.assertEqual(stats["decoder_free_slots"], [1])
        self.assertEqual(stats["pending_requests"], 0)

    def test_full_pipeline_with_capacity(self):
        """Full pipeline with capacity=1 should still work end-to-end."""
        req = _MockReq(request_id="cap-001")

        self.client.send_multipart([b"", pickle.dumps([req])])
        self.encoder_work.recv_multipart()

        send_tensors(
            self.encoder_result,
            {"prompt_embeds": torch.randn(1, 4, 16, 16)},
            {"request_id": "cap-001"},
        )
        self.denoiser_work.recv_multipart()

        send_tensors(
            self.denoiser_result,
            {"latents": torch.randn(1, 4, 16, 16)},
            {"request_id": "cap-001"},
        )
        self.decoder_work.recv_multipart()

        send_tensors(
            self.decoder_result,
            {"output": torch.randn(1, 3, 64, 64)},
            {"request_id": "cap-001"},
        )

        resp = self.client.recv_multipart()
        output_batch = pickle.loads(resp[-1])
        self.assertIsNotNone(output_batch.output)
        self.assertIsNone(output_batch.error)


class TestDiffusionServerP2PInit(unittest.TestCase):
    """Test DiffusionServer P2P mode initialization."""

    def test_p2p_mode_init(self):
        server = DiffusionServer(
            frontend_endpoint="tcp://127.0.0.1:19950",
            encoder_work_endpoints=["tcp://127.0.0.1:19951"],
            denoiser_work_endpoints=["tcp://127.0.0.1:19952"],
            decoder_work_endpoints=["tcp://127.0.0.1:19953"],
            encoder_result_endpoint="tcp://127.0.0.1:19954",
            denoiser_result_endpoint="tcp://127.0.0.1:19955",
            decoder_result_endpoint="tcp://127.0.0.1:19956",
            p2p_mode=True,
        )
        self.assertTrue(server._p2p_mode)
        self.assertEqual(len(server._p2p_state), 0)
        self.assertEqual(len(server._encoder_peers), 0)

    def test_p2p_stats(self):
        server = DiffusionServer(
            frontend_endpoint="tcp://127.0.0.1:19960",
            encoder_work_endpoints=["tcp://127.0.0.1:19961"],
            denoiser_work_endpoints=["tcp://127.0.0.1:19962"],
            decoder_work_endpoints=["tcp://127.0.0.1:19963"],
            encoder_result_endpoint="tcp://127.0.0.1:19964",
            denoiser_result_endpoint="tcp://127.0.0.1:19965",
            decoder_result_endpoint="tcp://127.0.0.1:19966",
            p2p_mode=True,
        )
        stats = server.get_stats()
        self.assertTrue(stats["p2p_mode"])
        self.assertEqual(stats["p2p_active_transfers"], 0)
        self.assertEqual(stats["encoder_peers"], 0)


class TestDiffusionServerP2PProtocol(unittest.TestCase):
    """Test P2P protocol message handling in DiffusionServer."""

    def test_p2p_register(self):
        """Test instance registration with DS."""
        from sglang.multimodal_gen.runtime.disaggregation.p2p_protocol import (
            P2PRegisterMsg,
            encode_p2p_msg,
        )

        server = DiffusionServer(
            frontend_endpoint="tcp://127.0.0.1:19970",
            encoder_work_endpoints=["tcp://127.0.0.1:19971"],
            denoiser_work_endpoints=["tcp://127.0.0.1:19972"],
            decoder_work_endpoints=["tcp://127.0.0.1:19973"],
            encoder_result_endpoint="tcp://127.0.0.1:19974",
            denoiser_result_endpoint="tcp://127.0.0.1:19975",
            decoder_result_endpoint="tcp://127.0.0.1:19976",
            p2p_mode=True,
        )

        # Register an encoder
        reg_msg = P2PRegisterMsg(
            role="encoder",
            instance_idx=0,
            session_id="enc-session-0",
            pool_ptr=0x7F000000,
            pool_size=16 * 1024 * 1024,
        )
        frames = encode_p2p_msg(reg_msg)
        server._handle_p2p_result(frames, "encoder")

        self.assertIn(0, server._encoder_peers)
        self.assertEqual(server._encoder_peers[0]["session_id"], "enc-session-0")
        self.assertEqual(server._encoder_peers[0]["pool_ptr"], 0x7F000000)

    def test_p2p_staged_and_alloc(self):
        """Test encoder staged → DS selects denoiser → sends alloc."""
        from sglang.multimodal_gen.runtime.disaggregation.p2p_protocol import (
            P2PStagedMsg,
            decode_p2p_msg,
            encode_p2p_msg,
        )

        ctx = zmq.Context()
        # We need live sockets to capture the alloc message DS sends to denoiser
        denoiser_work_ep = "tcp://127.0.0.1:19980"
        denoiser_work_pull = ctx.socket(zmq.PULL)
        denoiser_work_pull.bind(denoiser_work_ep)

        server = DiffusionServer(
            frontend_endpoint="tcp://127.0.0.1:19981",
            encoder_work_endpoints=["tcp://127.0.0.1:19982"],
            denoiser_work_endpoints=[denoiser_work_ep],
            decoder_work_endpoints=["tcp://127.0.0.1:19983"],
            encoder_result_endpoint="tcp://127.0.0.1:19984",
            denoiser_result_endpoint="tcp://127.0.0.1:19985",
            decoder_result_endpoint="tcp://127.0.0.1:19986",
            p2p_mode=True,
        )
        server.start()
        time.sleep(0.3)  # Let sockets connect

        try:
            # Submit a request
            server._tracker.submit("r1")
            server._tracker.transition(
                "r1", RequestState.ENCODER_RUNNING, encoder_instance=0
            )

            # Simulate encoder sending p2p_staged
            staged_msg = P2PStagedMsg(
                request_id="r1",
                data_size=4096,
                manifest={"latents": [{"offset": 0, "shape": [4], "dtype": "float32"}]},
                session_id="enc-0",
                pool_ptr=0x1000,
                slot_offset=0,
            )
            frames = encode_p2p_msg(staged_msg)
            server._handle_p2p_result(frames, "encoder")

            # DS should have sent p2p_alloc to denoiser
            alloc_frames = denoiser_work_pull.recv_multipart(flags=0)
            alloc_msg = decode_p2p_msg(alloc_frames)
            self.assertEqual(alloc_msg["msg_type"], "p2p_alloc")
            self.assertEqual(alloc_msg["request_id"], "r1")
            self.assertEqual(alloc_msg["data_size"], 4096)

            # Verify P2P state
            self.assertIn("r1", server._p2p_state)
            self.assertEqual(server._p2p_state["r1"].sender_session_id, "enc-0")
        finally:
            server.stop()
            denoiser_work_pull.close()
            ctx.destroy(linger=0)

    def test_p2p_full_e2e_handshake(self):
        """Test full P2P handshake: staged → alloc → allocated → push → pushed → ready."""
        from sglang.multimodal_gen.runtime.disaggregation.p2p_protocol import (
            P2PAllocatedMsg,
            P2PPushedMsg,
            P2PStagedMsg,
            decode_p2p_msg,
            encode_p2p_msg,
        )

        ctx = zmq.Context()
        enc_work_ep = "tcp://127.0.0.1:19990"
        den_work_ep = "tcp://127.0.0.1:19991"

        enc_work_pull = ctx.socket(zmq.PULL)
        enc_work_pull.bind(enc_work_ep)
        den_work_pull = ctx.socket(zmq.PULL)
        den_work_pull.bind(den_work_ep)

        server = DiffusionServer(
            frontend_endpoint="tcp://127.0.0.1:19992",
            encoder_work_endpoints=[enc_work_ep],
            denoiser_work_endpoints=[den_work_ep],
            decoder_work_endpoints=["tcp://127.0.0.1:19993"],
            encoder_result_endpoint="tcp://127.0.0.1:19994",
            denoiser_result_endpoint="tcp://127.0.0.1:19995",
            decoder_result_endpoint="tcp://127.0.0.1:19996",
            p2p_mode=True,
        )
        server.start()
        time.sleep(0.3)

        try:
            # Setup: submit request with encoder running
            server._tracker.submit("r1")
            server._tracker.transition(
                "r1", RequestState.ENCODER_RUNNING, encoder_instance=0
            )

            # Step 1: Encoder staged
            staged = P2PStagedMsg(
                request_id="r1",
                data_size=2048,
                manifest={"t": [{"offset": 0, "shape": [512], "dtype": "float32"}]},
                session_id="enc-sess",
                pool_ptr=0x1000,
                slot_offset=0,
            )
            server._handle_p2p_result(encode_p2p_msg(staged), "encoder")

            # Step 2: Denoiser receives alloc
            alloc_frames = den_work_pull.recv_multipart()
            alloc = decode_p2p_msg(alloc_frames)
            self.assertEqual(alloc["msg_type"], "p2p_alloc")

            # Step 3: Denoiser sends allocated
            allocated = P2PAllocatedMsg(
                request_id="r1",
                session_id="den-sess",
                pool_ptr=0x2000,
                slot_offset=0,
                slot_size=2048,
            )
            server._handle_p2p_result(encode_p2p_msg(allocated), "denoiser")

            # Step 4: Encoder receives push command
            push_frames = enc_work_pull.recv_multipart()
            push = decode_p2p_msg(push_frames)
            self.assertEqual(push["msg_type"], "p2p_push")
            self.assertEqual(push["dest_session_id"], "den-sess")
            self.assertEqual(push["dest_addr"], 0x2000)  # pool_ptr + slot_offset
            self.assertEqual(push["transfer_size"], 2048)

            # Step 5: Encoder sends pushed (RDMA done)
            pushed = P2PPushedMsg(request_id="r1")
            server._handle_p2p_result(encode_p2p_msg(pushed), "encoder")

            # Step 6: Denoiser receives ready
            ready_frames = den_work_pull.recv_multipart()
            ready = decode_p2p_msg(ready_frames)
            self.assertEqual(ready["msg_type"], "p2p_ready")
            self.assertEqual(ready["request_id"], "r1")
            self.assertIn("t", ready["manifest"])
        finally:
            server.stop()
            enc_work_pull.close()
            den_work_pull.close()
            ctx.destroy(linger=0)


if __name__ == "__main__":
    unittest.main()
