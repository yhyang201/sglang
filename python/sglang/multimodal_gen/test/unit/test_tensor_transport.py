# SPDX-License-Identifier: Apache-2.0
"""Unit tests for tensor_transport and role_connector modules."""

import json
import threading
import time
import unittest

import torch
import zmq

from sglang.multimodal_gen.runtime.disaggregation.transport.relay.tensor_transport import (
    TensorDescriptor,
    TensorWrapper,
    dtype_to_str,
    pack_tensors,
    recv_tensors,
    send_tensors,
    str_to_dtype,
    unpack_tensors,
)


class TestDtypeConversion(unittest.TestCase):
    """Test dtype string conversion round-trips."""

    def test_all_supported_dtypes(self):
        dtypes = [
            torch.float16,
            torch.float32,
            torch.float64,
            torch.bfloat16,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
            torch.bool,
        ]
        for dt in dtypes:
            s = dtype_to_str(dt)
            recovered = str_to_dtype(s)
            self.assertEqual(dt, recovered, f"Round-trip failed for {dt}")

    def test_unsupported_dtype_raises(self):
        with self.assertRaises(ValueError):
            dtype_to_str(torch.complex64)

    def test_unknown_string_raises(self):
        with self.assertRaises(ValueError):
            str_to_dtype("not_a_dtype")


class TestTensorWrapper(unittest.TestCase):
    """Test TensorWrapper buffer protocol."""

    def test_cpu_tensor(self):
        t = torch.randn(3, 4)
        w = TensorWrapper(t)
        buf = memoryview(w)
        self.assertEqual(len(buf), t.numel() * t.element_size())

    def test_non_contiguous_tensor(self):
        t = torch.randn(4, 4).t()  # transpose makes it non-contiguous
        self.assertFalse(t.is_contiguous())
        w = TensorWrapper(t)
        # Should be contiguous after wrapping
        self.assertTrue(w.tensor.is_contiguous())


class TestTensorDescriptor(unittest.TestCase):
    def test_round_trip(self):
        desc = TensorDescriptor(
            field_name="latents", shape=[1, 4, 16, 32], dtype="float32", list_index=-1
        )
        d = desc.to_dict()
        recovered = TensorDescriptor.from_dict(d)
        self.assertEqual(desc.field_name, recovered.field_name)
        self.assertEqual(desc.shape, recovered.shape)
        self.assertEqual(desc.dtype, recovered.dtype)
        self.assertEqual(desc.list_index, recovered.list_index)

    def test_list_field(self):
        desc = TensorDescriptor(
            field_name="prompt_embeds",
            shape=[1, 77, 768],
            dtype="float16",
            list_index=0,
        )
        d = desc.to_dict()
        self.assertEqual(d["list_index"], 0)


class TestPackUnpack(unittest.TestCase):
    """Test pack_tensors / unpack_tensors without ZMQ."""

    def test_single_tensor(self):
        t = torch.randn(2, 3)
        metadata_bytes, buffers = pack_tensors({"latents": t})
        self.assertEqual(len(buffers), 1)

        metadata = json.loads(metadata_bytes)
        self.assertEqual(len(metadata["tensor_descriptors"]), 1)
        self.assertEqual(metadata["tensor_descriptors"][0]["field_name"], "latents")

    def test_list_of_tensors(self):
        tensors = [torch.randn(1, 77, 768), torch.randn(1, 77, 1024)]
        metadata_bytes, buffers = pack_tensors({"prompt_embeds": tensors})
        self.assertEqual(len(buffers), 2)

        metadata = json.loads(metadata_bytes)
        descs = metadata["tensor_descriptors"]
        self.assertEqual(descs[0]["list_index"], 0)
        self.assertEqual(descs[1]["list_index"], 1)

    def test_none_fields_skipped(self):
        metadata_bytes, buffers = pack_tensors(
            {"latents": torch.randn(2, 3), "empty": None}
        )
        self.assertEqual(len(buffers), 1)

    def test_scalar_fields(self):
        metadata_bytes, buffers = pack_tensors(
            {"latents": torch.randn(1, 1)},
            scalar_fields={"request_id": "abc123", "guidance_scale": 7.5},
        )
        metadata = json.loads(metadata_bytes)
        self.assertEqual(metadata["scalar_fields"]["request_id"], "abc123")
        self.assertEqual(metadata["scalar_fields"]["guidance_scale"], 7.5)

    def test_mixed_fields(self):
        """Test with a realistic set of encoder output fields."""
        tensor_fields = {
            "prompt_embeds": [torch.randn(1, 77, 768)],
            "negative_prompt_embeds": [torch.randn(1, 77, 768)],
            "latents": torch.randn(1, 4, 8, 16, 16),
            "timesteps": torch.tensor([999.0, 950.0, 900.0]),
            "pooled_embeds": [],  # empty list, no tensors
            "y": None,  # None, should be skipped
        }
        scalar_fields = {
            "request_id": "req-001",
            "do_classifier_free_guidance": True,
            "guidance_scale": 7.5,
            "num_inference_steps": 50,
        }
        metadata_bytes, buffers = pack_tensors(tensor_fields, scalar_fields)
        # prompt_embeds[0] + neg_prompt_embeds[0] + latents + timesteps = 4
        self.assertEqual(len(buffers), 4)

    def test_round_trip_data_integrity(self):
        """Verify tensor data survives pack -> frame simulation -> unpack."""
        original_latents = torch.randn(1, 4, 16, 32, 32)
        original_embeds = [torch.randn(1, 77, 768), torch.randn(1, 77, 1024)]

        metadata_bytes, buffers = pack_tensors(
            {"latents": original_latents, "prompt_embeds": original_embeds},
            scalar_fields={"request_id": "test-round-trip"},
        )

        # Simulate ZMQ frame: convert TensorWrapper -> bytes
        frames = [metadata_bytes]
        for buf in buffers:
            frames.append(bytes(memoryview(buf)))

        tensor_fields, scalar_fields = unpack_tensors(frames)

        # Verify latents
        self.assertTrue(torch.equal(tensor_fields["latents"], original_latents))

        # Verify list of tensors
        self.assertEqual(len(tensor_fields["prompt_embeds"]), 2)
        self.assertTrue(
            torch.equal(tensor_fields["prompt_embeds"][0], original_embeds[0])
        )
        self.assertTrue(
            torch.equal(tensor_fields["prompt_embeds"][1], original_embeds[1])
        )

        # Verify scalars
        self.assertEqual(scalar_fields["request_id"], "test-round-trip")

    def test_bfloat16_round_trip(self):
        """Verify bfloat16 tensors survive the round trip."""
        t = torch.randn(2, 3, dtype=torch.bfloat16)
        metadata_bytes, buffers = pack_tensors({"x": t})
        frames = [metadata_bytes, bytes(memoryview(buffers[0]))]
        tensor_fields, _ = unpack_tensors(frames)
        self.assertEqual(tensor_fields["x"].dtype, torch.bfloat16)
        self.assertTrue(torch.equal(tensor_fields["x"], t))


class TestZMQTransport(unittest.TestCase):
    """Test actual ZMQ send_tensors / recv_tensors."""

    def setUp(self):
        self.context = zmq.Context()
        self.push_socket = self.context.socket(zmq.PUSH)
        port = self.push_socket.bind_to_random_port("tcp://127.0.0.1")
        self.pull_socket = self.context.socket(zmq.PULL)
        self.pull_socket.connect(f"tcp://127.0.0.1:{port}")
        # Give sockets time to connect
        time.sleep(0.1)

    def tearDown(self):
        self.push_socket.close()
        self.pull_socket.close()
        self.context.term()

    def test_send_recv_single_tensor(self):
        original = torch.randn(4, 8)
        send_tensors(self.push_socket, {"data": original}, {"id": "test1"})
        tensor_fields, scalar_fields = recv_tensors(self.pull_socket)
        self.assertTrue(torch.equal(tensor_fields["data"], original))
        self.assertEqual(scalar_fields["id"], "test1")

    def test_send_recv_multiple_tensors(self):
        latents = torch.randn(1, 4, 16, 32, 32)
        embeds = [torch.randn(1, 77, 768), torch.randn(1, 77, 1024)]
        timesteps = torch.tensor([999.0, 950.0])

        send_tensors(
            self.push_socket,
            {"latents": latents, "prompt_embeds": embeds, "timesteps": timesteps},
            {"request_id": "multi-test"},
        )

        tensor_fields, scalar_fields = recv_tensors(self.pull_socket)
        self.assertTrue(torch.equal(tensor_fields["latents"], latents))
        self.assertEqual(len(tensor_fields["prompt_embeds"]), 2)
        self.assertTrue(torch.equal(tensor_fields["prompt_embeds"][0], embeds[0]))
        self.assertTrue(torch.equal(tensor_fields["prompt_embeds"][1], embeds[1]))
        self.assertTrue(torch.equal(tensor_fields["timesteps"], timesteps))
        self.assertEqual(scalar_fields["request_id"], "multi-test")

    def test_send_recv_empty_tensor_fields(self):
        """Send only scalar data, no tensors."""
        send_tensors(self.push_socket, {}, {"status": "done"})
        tensor_fields, scalar_fields = recv_tensors(self.pull_socket)
        self.assertEqual(len(tensor_fields), 0)
        self.assertEqual(scalar_fields["status"], "done")

    def test_nonblocking_recv(self):
        """NOBLOCK should raise zmq.Again when no message."""
        with self.assertRaises(zmq.Again):
            recv_tensors(self.pull_socket, flags=zmq.NOBLOCK)

    def test_concurrent_send_recv(self):
        """Test multiple messages in sequence."""
        n_messages = 5
        results = []

        def receiver():
            for _ in range(n_messages):
                tf, sf = recv_tensors(self.pull_socket)
                results.append((tf, sf))

        recv_thread = threading.Thread(target=receiver)
        recv_thread.start()

        for i in range(n_messages):
            t = torch.randn(2, 3) * i
            send_tensors(self.push_socket, {"data": t}, {"idx": i})

        recv_thread.join(timeout=5)
        self.assertEqual(len(results), n_messages)
        for i, (tf, sf) in enumerate(results):
            self.assertEqual(sf["idx"], i)


class TestRoleConnector(unittest.TestCase):
    """Test role_connector send/recv with realistic Req-like data."""

    def setUp(self):
        self.context = zmq.Context()
        self.push_socket = self.context.socket(zmq.PUSH)
        port = self.push_socket.bind_to_random_port("tcp://127.0.0.1")
        self.pull_socket = self.context.socket(zmq.PULL)
        self.pull_socket.connect(f"tcp://127.0.0.1:{port}")
        time.sleep(0.1)

    def tearDown(self):
        self.push_socket.close()
        self.pull_socket.close()
        self.context.term()

    def test_encoder_to_denoiser_payload(self):
        """Simulate an encoder-to-denoiser data transfer for Wan2.1."""
        # Typical Wan2.1 encoder output
        tensor_fields = {
            "prompt_embeds": [torch.randn(1, 512, 4096, dtype=torch.bfloat16)],
            "latents": torch.randn(1, 16, 21, 30, 52, dtype=torch.bfloat16),
            "timesteps": torch.tensor([999.0, 975.0, 950.0], dtype=torch.float32),
        }
        scalar_fields = {
            "request_id": "wan-test-001",
            "do_classifier_free_guidance": False,
            "guidance_scale": 1.0,
            "num_inference_steps": 50,
            "height": 480,
            "width": 832,
            "num_frames": 81,
        }

        send_tensors(self.push_socket, tensor_fields, scalar_fields)
        recv_tensors_result, recv_scalars = recv_tensors(self.pull_socket)

        # Verify shapes and dtypes preserved
        self.assertEqual(
            recv_tensors_result["prompt_embeds"][0].shape,
            torch.Size([1, 512, 4096]),
        )
        self.assertEqual(recv_tensors_result["prompt_embeds"][0].dtype, torch.bfloat16)
        self.assertEqual(
            recv_tensors_result["latents"].shape,
            torch.Size([1, 16, 21, 30, 52]),
        )
        self.assertEqual(recv_scalars["request_id"], "wan-test-001")
        self.assertEqual(recv_scalars["num_frames"], 81)

    def test_denoiser_to_decoder_payload(self):
        """Simulate a denoiser-to-decoder data transfer."""
        tensor_fields = {
            "latents": torch.randn(1, 16, 21, 30, 52, dtype=torch.bfloat16),
        }
        scalar_fields = {
            "request_id": "wan-test-001",
            "height": 480,
            "width": 832,
            "num_frames": 81,
            "raw_latent_shape": [1, 16, 21, 30, 52],
        }

        send_tensors(self.push_socket, tensor_fields, scalar_fields)
        recv_tf, recv_sf = recv_tensors(self.pull_socket)

        self.assertTrue(torch.equal(recv_tf["latents"], tensor_fields["latents"]))
        self.assertEqual(recv_sf["raw_latent_shape"], [1, 16, 21, 30, 52])


if __name__ == "__main__":
    unittest.main()
