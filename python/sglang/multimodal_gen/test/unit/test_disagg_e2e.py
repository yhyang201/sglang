# SPDX-License-Identifier: Apache-2.0
"""End-to-end test for disaggregated pipeline with real model.

Tests the full Encoder -> Denoiser -> Decoder flow with Wan2.1-T2V-1.3B
across separate GPU processes.

Usage:
    CUDA_VISIBLE_DEVICES=1,2,3 python -m pytest test/unit/test_disagg_e2e.py -v -s
"""

import multiprocessing as mp
import os
import time
import unittest

import torch


def _run_role(role, model_path, gpu_id, endpoints, result_dict, error_dict):
    """Run a single disagg role in a subprocess."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    try:
        import zmq

        from sglang.multimodal_gen.runtime.disaggregation.role_connector import (
            DENOISER_TO_DECODER_SCALAR_FIELDS,
            DENOISER_TO_DECODER_TENSOR_FIELDS,
            ENCODER_TO_DENOISER_SCALAR_FIELDS,
            ENCODER_TO_DENOISER_TENSOR_FIELDS,
            RoleConnectorReceiver,
            RoleConnectorSender,
        )
        from sglang.multimodal_gen.runtime.disaggregation.tensor_transport import (
            recv_tensors,
            send_tensors,
        )
        from sglang.multimodal_gen.runtime.distributed import (
            maybe_init_distributed_environment_and_model_parallel,
        )
        from sglang.multimodal_gen.runtime.pipelines_core import Req, build_pipeline
        from sglang.multimodal_gen.runtime.server_args import (
            ServerArgs,
            set_global_server_args,
        )

        e2d_ep, d2d_ep, d2e_ep = endpoints

        # Build server args with disagg role
        server_args = ServerArgs.from_kwargs(
            model_path=model_path,
            disagg_role=role,
            num_gpus=1,
            warmup=False,
        )
        set_global_server_args(server_args)

        # Init distributed env (single GPU, but required for pipeline)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(server_args.master_port)
        os.environ["LOCAL_RANK"] = "0"
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        maybe_init_distributed_environment_and_model_parallel(
            tp_size=1,
            enable_cfg_parallel=False,
            ulysses_degree=1,
            ring_degree=1,
            sp_size=1,
            dp_size=1,
            distributed_init_method=f"tcp://127.0.0.1:{server_args.master_port}",
        )

        # Build pipeline (only loads modules for this role)
        pipeline = build_pipeline(server_args)

        context = zmq.Context(io_threads=2)

        if role == "encoder":
            # Create sender to denoiser
            sender = RoleConnectorSender(
                context,
                e2d_ep,
                ENCODER_TO_DENOISER_TENSOR_FIELDS,
                ENCODER_TO_DENOISER_SCALAR_FIELDS,
            )
            # Create receiver for decoder result
            from sglang.multimodal_gen.runtime.utils.common import get_zmq_socket

            result_recv, _ = get_zmq_socket(context, zmq.PULL, d2e_ep, bind=True)

            # Create and run encoder request
            req = Req(
                prompt="A beautiful sunset over the ocean",
                negative_prompt="",
                height=480,
                width=832,
                num_frames=17,
                num_inference_steps=5,
                guidance_scale=1.0,
                seed=42,
                request_id="e2e-test-001",
            )

            start_time = time.monotonic()
            result = pipeline.forward(req, server_args)
            encoder_time = time.monotonic() - start_time

            # Send to denoiser
            sender.send(result)

            # Wait for decoder result
            tf, sf = recv_tensors(result_recv)
            total_time = time.monotonic() - start_time

            output = tf.get("output")
            result_dict["output_shape"] = (
                list(output.shape) if output is not None else None
            )
            result_dict["encoder_time"] = encoder_time
            result_dict["total_time"] = total_time
            result_dict["error"] = sf.get("error")

            sender.close()
            result_recv.close()

        elif role == "denoising":
            # Connect receiver from encoder (tensors go to GPU)
            receiver = RoleConnectorReceiver(
                context,
                e2d_ep,
                ENCODER_TO_DENOISER_TENSOR_FIELDS,
                ENCODER_TO_DENOISER_SCALAR_FIELDS,
                device="cuda",
            )
            # Create sender to decoder
            sender = RoleConnectorSender(
                context,
                d2d_ep,
                DENOISER_TO_DECODER_TENSOR_FIELDS,
                DENOISER_TO_DECODER_SCALAR_FIELDS,
            )

            # Wait for encoder data
            recv_req = receiver.recv()

            # Initialize the scheduler with timesteps (normally done by
            # TimestepPreparationStage which runs on encoder side)
            scheduler = pipeline.get_module("scheduler")
            if scheduler is not None and hasattr(recv_req, "num_inference_steps"):
                scheduler.set_timesteps(recv_req.num_inference_steps, device="cuda")

            # Run denoising
            start_time = time.monotonic()
            result = pipeline.forward(recv_req, server_args)
            denoising_time = time.monotonic() - start_time

            # Send to decoder
            sender.send(result)
            result_dict["denoising_time"] = denoising_time

            receiver.close()
            sender.close()

        elif role == "decoder":
            # Connect receiver from denoiser (tensors go to GPU)
            receiver = RoleConnectorReceiver(
                context,
                d2d_ep,
                DENOISER_TO_DECODER_TENSOR_FIELDS,
                DENOISER_TO_DECODER_SCALAR_FIELDS,
                device="cuda",
            )
            # Create sender for result back to encoder
            from sglang.multimodal_gen.runtime.utils.common import get_zmq_socket

            result_send, _ = get_zmq_socket(context, zmq.PUSH, d2e_ep, bind=False)

            # Wait for denoiser data
            recv_req = receiver.recv()

            # Run decoding
            start_time = time.monotonic()
            result = pipeline.forward(recv_req, server_args)
            decoding_time = time.monotonic() - start_time

            # Send result back to encoder
            tensor_fields = {}
            scalar_fields = {"request_id": getattr(recv_req, "request_id", "unknown")}
            if hasattr(result, "output") and result.output is not None:
                tensor_fields["output"] = result.output
            elif hasattr(result, "output") and result.output is not None:
                tensor_fields["output"] = result.output

            send_tensors(result_send, tensor_fields, scalar_fields)
            result_dict["decoding_time"] = decoding_time

            receiver.close()
            result_send.close()

        context.term()

    except Exception as e:
        import traceback

        error_dict["error"] = f"{role}: {e}\n{traceback.format_exc()}"


class TestDisaggEndToEnd(unittest.TestCase):
    """End-to-end test with real Wan2.1-T2V-1.3B model."""

    @unittest.skipUnless(
        torch.cuda.is_available() and torch.cuda.device_count() >= 3,
        "Requires 3+ GPUs",
    )
    def test_wan21_disagg_three_gpus(self):
        """Test full disagg pipeline with Wan2.1-1.3B across 3 GPUs."""
        model_path = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"

        # Endpoints - use ports that are likely free
        e2d_ep = "tcp://127.0.0.1:19201"
        d2d_ep = "tcp://127.0.0.1:19202"
        d2e_ep = "tcp://127.0.0.1:19203"
        endpoints = (e2d_ep, d2d_ep, d2e_ep)

        # Use available GPUs (skip GPU 0 which is often occupied)
        gpu_ids = {"encoder": 1, "denoising": 2, "decoder": 3}

        manager = mp.Manager()
        results = {
            "encoder": manager.dict(),
            "denoising": manager.dict(),
            "decoder": manager.dict(),
        }
        errors = {
            "encoder": manager.dict(),
            "denoising": manager.dict(),
            "decoder": manager.dict(),
        }

        processes = {}
        for role in ["encoder", "denoising", "decoder"]:
            p = mp.Process(
                target=_run_role,
                args=(
                    role,
                    model_path,
                    gpu_ids[role],
                    endpoints,
                    results[role],
                    errors[role],
                ),
            )
            processes[role] = p

        # Start all processes
        for p in processes.values():
            p.start()

        # Wait for completion (timeout 5 min)
        for role, p in processes.items():
            p.join(timeout=300)
            if p.is_alive():
                p.kill()
                self.fail(f"{role} process timed out")

        # Check for errors
        for role in ["encoder", "denoising", "decoder"]:
            if "error" in errors[role]:
                self.fail(f"Error in {role}: {errors[role]['error']}")

        # Verify encoder got the result
        encoder_result = dict(results["encoder"])
        self.assertIsNotNone(
            encoder_result.get("output_shape"),
            f"No output received. Results: {encoder_result}",
        )
        self.assertIsNone(
            encoder_result.get("error"),
            f"Error in result: {encoder_result.get('error')}",
        )

        # Output shape should be [batch, channels, frames, height, width]
        shape = encoder_result["output_shape"]
        print(f"\nDisagg E2E Results:")
        print(f"  Output shape: {shape}")
        print(f"  Encoder time: {encoder_result.get('encoder_time', 0):.2f}s")
        print(
            f"  Denoising time: {dict(results['denoising']).get('denoising_time', 0):.2f}s"
        )
        print(
            f"  Decoding time: {dict(results['decoder']).get('decoding_time', 0):.2f}s"
        )
        print(f"  Total time: {encoder_result.get('total_time', 0):.2f}s")


if __name__ == "__main__":
    unittest.main()
