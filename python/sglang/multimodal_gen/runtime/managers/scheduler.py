# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
import asyncio
import os
import pickle
import time
from collections import deque
from typing import Any, List

import torch
import zmq

from sglang.multimodal_gen.configs.pipeline_configs.base import ModelTaskType
from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.entrypoints.openai.utils import (
    _parse_size,
    save_image_to_path,
)
from sglang.multimodal_gen.runtime.entrypoints.post_training.io_struct import (
    GetWeightsChecksumReqInput,
    UpdateWeightFromDiskReqInput,
)
from sglang.multimodal_gen.runtime.entrypoints.utils import (
    ListLorasReq,
    MergeLoraWeightsReq,
    SetLoraReq,
    ShutdownReq,
    UnmergeLoraWeightsReq,
)
from sglang.multimodal_gen.runtime.managers.gpu_worker import GPUWorker
from sglang.multimodal_gen.runtime.pipelines_core import Req
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch
from sglang.multimodal_gen.runtime.server_args import (
    PortArgs,
    ServerArgs,
    set_global_server_args,
)
from sglang.multimodal_gen.runtime.utils.common import get_zmq_socket
from sglang.multimodal_gen.runtime.utils.distributed import broadcast_pyobj
from sglang.multimodal_gen.runtime.utils.logging_utils import GREEN, RESET, init_logger

logger = init_logger(__name__)

MINIMUM_PICTURE_BASE64_FOR_WARMUP = "data:image/jpg;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAACXBIWXMAAA7EAAAOxAGVKw4bAAAAbUlEQVRYhe3VsQ2AMAxE0Y/lIgNQULD/OqyCMgCihCKSG4yRuKuiNH6JLsoEbMACOGBcua9HOR7Y6w6swBwMy0qLTpkeI77qdEBpBFAHBBDAGH8WrwJKI4AAegUCfAKgEgpQDvh3CR3oQCuav58qlAw73kKCSgAAAABJRU5ErkJggg=="


class Scheduler:
    """
    Runs the main event loop for the rank 0 worker.
    It listens for external requests via ZMQ and coordinates with other workers.
    This class does NOT manage worker processes.
    """

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        port_args: PortArgs,
        task_pipes_to_slaves: list = None,
        result_pipes_from_slaves: list = None,
    ):
        self.server_args = server_args
        self.port_args = port_args

        set_global_server_args(server_args=server_args)

        # Inter-process Communication
        self.context = zmq.Context(io_threads=2)
        endpoint = server_args.scheduler_endpoint
        if gpu_id == 0:
            # router allocates identify (envelope) for each connection
            self.receiver, actual_endpoint = get_zmq_socket(
                self.context, zmq.ROUTER, endpoint, True
            )
            logger.info(f"Scheduler bind at endpoint: {actual_endpoint}")
        else:
            self.receiver = None

        worker = GPUWorker(
            local_rank=gpu_id,
            master_port=port_args.master_port,
            rank=gpu_id,
            server_args=server_args,
        )
        self.worker = worker
        self.task_pipes_to_slaves = task_pipes_to_slaves
        self.result_pipes_from_slaves = result_pipes_from_slaves
        self.gpu_id = gpu_id
        self._running = True

        self.request_handlers = {
            SetLoraReq: self._handle_set_lora,
            MergeLoraWeightsReq: self._handle_merge_lora,
            UnmergeLoraWeightsReq: self._handle_unmerge_lora,
            Req: self._handle_generation,
            List[Req]: self._handle_generation,
            ListLorasReq: self._handle_list_loras,
            ShutdownReq: self._handle_shutdown,
            UpdateWeightFromDiskReqInput: self._handle_update_weights_from_disk,
            GetWeightsChecksumReqInput: self._handle_get_weights_checksum,
        }

        # FIFO, new reqs are appended
        self.waiting_queue: deque[tuple[bytes, Req]] = deque()

        # whether we've send the necessary warmup reqs
        self.warmed_up = False
        # warmup progress tracking
        self._warmup_total = 0
        self._warmup_processed = 0

        self.prepare_server_warmup_reqs()

        # Maximum consecutive errors before terminating the event loop
        self._max_consecutive_errors = 3
        self._consecutive_error_count = 0

        # Disaggregation connectors
        self._disagg_role = server_args.disagg_role
        self._role_sender = None  # sends to next role
        self._role_receiver = None  # receives from previous role
        self._result_sender = None  # decoder -> encoder result return
        self._result_receiver = None  # encoder waits for decoder result
        if self._disagg_role != RoleType.MONOLITHIC:
            self._init_disagg_connectors()

    def _handle_set_lora(self, reqs: List[Any]) -> OutputBatch:
        # TODO: return set status
        # TODO: return with SetLoRAResponse or something more appropriate
        req = reqs[0]
        return self.worker.set_lora(
            req.lora_nickname, req.lora_path, req.target, req.strength
        )

    def _handle_merge_lora(self, reqs: List[Any]):
        req = reqs[0]
        return self.worker.merge_lora_weights(req.target, req.strength)

    def _handle_unmerge_lora(self, reqs: List[Any]) -> OutputBatch:
        req = reqs[0]
        return self.worker.unmerge_lora_weights(req.target)

    def _handle_list_loras(self, _reqs: List[Any]) -> OutputBatch:
        return self.worker.list_loras()

    def _handle_shutdown(self, _reqs: List[Any]) -> OutputBatch:
        self._running = False
        return OutputBatch()

    def _handle_update_weights_from_disk(self, reqs: List[Any]) -> OutputBatch:
        """Handle update_weights_from_disk request for RL workflows."""
        req = reqs[0]
        success, message = self.worker.update_weights_from_disk(
            model_path=req.model_path,
            flush_cache=req.flush_cache,
            target_modules=req.target_modules,
        )
        return OutputBatch(
            output={"success": success, "message": message},
            error=None if success else message,
        )

    def _handle_get_weights_checksum(self, reqs: List[Any]) -> OutputBatch:
        """Handle get_weights_checksum request."""
        req = reqs[0]
        checksums = self.worker.get_weights_checksum(module_names=req.module_names)
        return OutputBatch(output=checksums)

    def _handle_generation(self, reqs: List[Req]):
        warmup_reqs = [req for req in reqs if req.is_warmup]
        if warmup_reqs:
            self._warmup_processed += len(warmup_reqs)
            if self._warmup_total > 0:
                logger.info(
                    f"Processing warmup req... ({self._warmup_processed}/{self._warmup_total})"
                )
            else:
                logger.info("Processing warmup req...")

        # In disagg ENCODER mode, use the disagg handler that sends to denoiser
        # and waits for decoder result
        if self._disagg_role == RoleType.ENCODER:
            return self._handle_generation_disagg_encoder(reqs)

        return self.worker.execute_forward(reqs)

    def return_result(
        self,
        output_batch: OutputBatch,
        identity: bytes | None = None,
        is_warmup: bool = False,
    ):
        """
        replies to client, only on rank 0
        """
        if not is_warmup and self.receiver is not None and identity is not None:
            self.receiver.send_multipart([identity, b"", pickle.dumps(output_batch)])

    def get_next_batch_to_run(self) -> list[tuple[bytes, Req]] | None:
        """pull a req from waiting_queue"""
        if not self.waiting_queue:
            return None

        # pop the first (earliest)
        item = self.waiting_queue.popleft()

        return [item]

    def prepare_server_warmup_reqs(self):
        if (
            self.server_args.warmup
            and not self.warmed_up
            and self.server_args.warmup_resolutions is not None
        ):
            # insert warmup reqs constructed with each warmup-resolution
            self._warmup_total = len(self.server_args.warmup_resolutions)
            self._warmup_processed = 0

            for resolution in self.server_args.warmup_resolutions:
                width, height = _parse_size(resolution)
                task_type = self.server_args.pipeline_config.task_type

                if task_type in (
                    ModelTaskType.I2I,
                    ModelTaskType.TI2I,
                    ModelTaskType.I2V,
                    ModelTaskType.TI2V,
                ):
                    uploads_dir = os.path.join("outputs", "uploads")
                    os.makedirs(uploads_dir, exist_ok=True)
                    input_path = asyncio.run(
                        save_image_to_path(
                            MINIMUM_PICTURE_BASE64_FOR_WARMUP,
                            os.path.join(uploads_dir, "warmup_image.jpg"),
                        )
                    )
                    req = Req(
                        data_type=task_type.data_type(),
                        width=width,
                        height=height,
                        prompt="",
                        negative_prompt="",
                        image_path=[input_path],
                    )
                else:
                    req = Req(
                        data_type=task_type.data_type(),
                        width=width,
                        height=height,
                        prompt="",
                    )
                req.set_as_warmup(self.server_args.warmup_steps)
                self.waiting_queue.append((None, req))
            # if server is warmed-up, set this flag to avoid req-based warmup
            self.warmed_up = True

    def process_received_reqs_with_req_based_warmup(
        self, recv_reqs: List[tuple[bytes, Any]]
    ) -> List[tuple[bytes, Any]]:
        if (
            self.warmed_up
            or not self.server_args.warmup
            or not recv_reqs
            or self.server_args.warmup_resolutions is not None
        ):
            return recv_reqs

        # handle server req-based warmup by inserting an identical req to the beginning of the waiting queue
        # only the very first req through server's lifetime will be warmed up
        identity, req = recv_reqs[0]
        if isinstance(req, Req):
            warmup_req = req.copy_as_warmup(self.server_args.warmup_steps)
            recv_reqs.insert(0, (identity, warmup_req))
            self._warmup_total = 1
            self._warmup_processed = 0
            self.warmed_up = True
        return recv_reqs

    def recv_reqs(self) -> List[tuple[bytes, Any]]:
        """
        For non-main schedulers, reqs are broadcasted from main using broadcast_pyobj
        """
        if self.receiver is not None:
            try:
                try:
                    # Accept valid REQ envelopes only, ignore malformed/probe frames.
                    parts = self.receiver.recv_multipart(zmq.NOBLOCK)
                    identity, payload = parts[0], parts[-1]

                    # Ignore malformed probes or non-pickle data
                    recv_reqs = pickle.loads(payload) if len(parts) > 2 else []
                except (zmq.Again, pickle.UnpicklingError, IndexError, EOFError):
                    recv_reqs = []
            except zmq.ZMQError:
                # re-raise or handle appropriately to let the outer loop continue
                raise

            if recv_reqs:
                # Ensure recv_reqs is a list
                if not isinstance(recv_reqs, list):
                    recv_reqs = [recv_reqs]

                # Pack with identity for rank 0
                recv_reqs = [(identity, req) for req in recv_reqs]
        else:
            recv_reqs = None

        # TODO: fix this condition
        if self.server_args.sp_degree != 1:
            recv_reqs = broadcast_pyobj(
                recv_reqs,
                self.worker.sp_group.rank,
                self.worker.sp_cpu_group,
                src=self.worker.sp_group.ranks[0],
            )

        if self.server_args.enable_cfg_parallel:
            recv_reqs = broadcast_pyobj(
                recv_reqs,
                self.worker.cfg_group.rank,
                self.worker.cfg_cpu_group,
                src=self.worker.cfg_group.ranks[0],
            )

        if self.server_args.tp_size > 1:
            recv_reqs = broadcast_pyobj(
                recv_reqs,
                self.worker.tp_group.rank,
                self.worker.tp_cpu_group,
                src=self.worker.tp_group.ranks[0],
            )

        assert recv_reqs is not None

        return recv_reqs

    def _init_disagg_connectors(self):
        """Initialize ZMQ connectors for disaggregated role communication."""
        from sglang.multimodal_gen.runtime.disaggregation.role_connector import (
            create_denoiser_to_decoder_receiver,
            create_denoiser_to_decoder_sender,
            create_encoder_to_denoiser_receiver,
            create_encoder_to_denoiser_sender,
        )

        sa = self.server_args

        if self._disagg_role == RoleType.ENCODER:
            # Encoder sends to denoiser
            self._role_sender = create_encoder_to_denoiser_sender(
                self.context, sa.encoder_to_denoiser_endpoint
            )
            # Encoder receives final result from decoder
            result_socket, _ = get_zmq_socket(
                self.context, zmq.PULL, sa.decoder_to_encoder_endpoint, bind=True
            )
            self._result_receiver = result_socket
            logger.info(
                "Disagg ENCODER: sender=%s, result_receiver=%s",
                sa.encoder_to_denoiser_endpoint,
                sa.decoder_to_encoder_endpoint,
            )

        elif self._disagg_role == RoleType.DENOISING:
            # Denoiser receives from encoder
            self._role_receiver = create_encoder_to_denoiser_receiver(
                self.context, sa.encoder_to_denoiser_endpoint
            )
            # Denoiser sends to decoder
            self._role_sender = create_denoiser_to_decoder_sender(
                self.context, sa.denoiser_to_decoder_endpoint
            )
            logger.info(
                "Disagg DENOISING: receiver=%s, sender=%s",
                sa.encoder_to_denoiser_endpoint,
                sa.denoiser_to_decoder_endpoint,
            )

        elif self._disagg_role == RoleType.DECODER:
            # Decoder receives from denoiser
            self._role_receiver = create_denoiser_to_decoder_receiver(
                self.context, sa.denoiser_to_decoder_endpoint
            )
            # Decoder sends result back to encoder
            result_socket, _ = get_zmq_socket(
                self.context, zmq.PUSH, sa.decoder_to_encoder_endpoint, bind=False
            )
            self._result_sender = result_socket
            logger.info(
                "Disagg DECODER: receiver=%s, result_sender=%s",
                sa.denoiser_to_decoder_endpoint,
                sa.decoder_to_encoder_endpoint,
            )

    def _disagg_event_loop(self) -> None:
        """Event loop for DENOISING and DECODER roles.

        Instead of receiving requests from an HTTP client via ZMQ ROUTER,
        these roles receive intermediate tensors from the previous role
        via ZMQ PULL, process them, and send results to the next role.
        """
        from sglang.multimodal_gen.runtime.disaggregation.tensor_transport import (
            send_tensors,
        )

        role_name = self._disagg_role.value.upper()
        logger.info("Disagg %s event loop started, waiting for work...", role_name)

        while self._running:
            try:
                # 1. Receive from previous role (blocking)
                req = self._role_receiver.recv()
                logger.debug(
                    "Disagg %s: received request %s",
                    role_name,
                    getattr(req, "request_id", "unknown"),
                )

                # 2. Prepare for execution
                if self._disagg_role == RoleType.DENOISING:
                    # Initialize the scheduler timesteps (normally done by
                    # TimestepPreparationStage which runs on the encoder side)
                    scheduler_mod = self.worker.pipeline.get_module("scheduler")
                    num_steps = getattr(req, "num_inference_steps", None)
                    if scheduler_mod is not None and num_steps is not None:
                        device = torch.device(f"cuda:{self.gpu_id}")
                        scheduler_mod.set_timesteps(num_steps, device=device)

                # 3. Execute pipeline stages for this role
                start_time = time.monotonic()
                if self._disagg_role == RoleType.DENOISING:
                    # Run denoising stages, get back Req with updated latents
                    result = self.worker.execute_forward([req], return_req=True)
                    duration_s = time.monotonic() - start_time

                    if isinstance(result, Req):
                        # Send denoised latents to decoder
                        self._role_sender.send(result)
                        logger.info(
                            "Disagg DENOISING: processed request %s in %.2f s, sent to decoder",
                            getattr(result, "request_id", "unknown"),
                            duration_s,
                        )
                    else:
                        # Error case: result is OutputBatch with error
                        logger.error(
                            "Disagg DENOISING: error processing request: %s",
                            getattr(result, "error", "unknown"),
                        )

                elif self._disagg_role == RoleType.DECODER:
                    # Run decoding stages, get OutputBatch with final output
                    output_batch = self.worker.execute_forward([req])
                    duration_s = time.monotonic() - start_time

                    # Send result back to encoder via result channel
                    tensor_fields = {}
                    scalar_fields = {
                        "request_id": getattr(req, "request_id", "unknown"),
                    }
                    if output_batch.output is not None:
                        tensor_fields["output"] = output_batch.output
                    if output_batch.audio is not None:
                        tensor_fields["audio"] = output_batch.audio
                    if output_batch.audio_sample_rate is not None:
                        scalar_fields["audio_sample_rate"] = (
                            output_batch.audio_sample_rate
                        )
                    if output_batch.error is not None:
                        scalar_fields["error"] = output_batch.error

                    send_tensors(self._result_sender, tensor_fields, scalar_fields)
                    logger.info(
                        "Disagg DECODER: processed request %s in %.2f s, sent result to encoder",
                        getattr(req, "request_id", "unknown"),
                        duration_s,
                    )

            except Exception as e:
                self._consecutive_error_count += 1
                logger.error(
                    "Disagg %s: error processing request (attempt %d/%d): %s",
                    role_name,
                    self._consecutive_error_count,
                    self._max_consecutive_errors,
                    e,
                    exc_info=True,
                )
                if self._consecutive_error_count >= self._max_consecutive_errors:
                    raise RuntimeError(
                        f"Disagg {role_name} terminated after "
                        f"{self._max_consecutive_errors} consecutive errors: {e}"
                    ) from e
                continue

            # Reset error count on success
            self._consecutive_error_count = 0

        self._cleanup_disagg_connectors()

    def _handle_generation_disagg_encoder(self, reqs: List[Req]) -> OutputBatch:
        """Handle generation for ENCODER role in disagg mode.

        1. Run encoder pipeline stages → get Req with embeddings/latents
        2. Send intermediate Req to denoiser
        3. Wait for decoder to return the final result
        4. Return OutputBatch to the HTTP client
        """
        from sglang.multimodal_gen.runtime.disaggregation.tensor_transport import (
            recv_tensors,
        )

        # Run encoder stages (return raw Req to access intermediate tensors)
        req_result = self.worker.execute_forward(reqs, return_req=True)

        if not isinstance(req_result, Req):
            # Error case
            return req_result

        # Send encoder outputs to denoiser
        self._role_sender.send(req_result)
        logger.debug(
            "Disagg ENCODER: sent request %s to denoiser",
            getattr(req_result, "request_id", "unknown"),
        )

        # Block-wait for decoder to return the final result
        tensor_fields, scalar_fields = recv_tensors(self._result_receiver)

        # Build OutputBatch from decoder result
        output_batch = OutputBatch(
            output=tensor_fields.get("output"),
            audio=tensor_fields.get("audio"),
            audio_sample_rate=scalar_fields.get("audio_sample_rate"),
            error=scalar_fields.get("error"),
            metrics=req_result.metrics,
        )
        return output_batch

    def _cleanup_disagg_connectors(self):
        """Clean up disagg connector sockets."""
        if self._role_sender is not None:
            self._role_sender.close()
        if self._role_receiver is not None:
            self._role_receiver.close()
        if self._result_sender is not None:
            self._result_sender.close()
        if self._result_receiver is not None:
            self._result_receiver.close()

    def event_loop(self) -> None:
        """
        The main event loop that listens for ZMQ requests.
        Handles abortion
        """
        # For DENOISING/DECODER roles, use the disagg event loop
        if self._disagg_role in (RoleType.DENOISING, RoleType.DECODER):
            self._disagg_event_loop()
            return

        logger.debug(
            f"Rank 0 scheduler listening on tcp://*:{self.server_args.scheduler_port}"
        )

        while self._running:
            # 1: receive requests
            try:
                new_reqs = self.recv_reqs()
                new_reqs = self.process_received_reqs_with_req_based_warmup(new_reqs)
                self.waiting_queue.extend(new_reqs)
                # Reset error count on success
                self._consecutive_error_count = 0
            except Exception as e:
                self._consecutive_error_count += 1
                logger.error(
                    f"Error receiving requests in scheduler event loop "
                    f"(attempt {self._consecutive_error_count}/{self._max_consecutive_errors}): {e}",
                    exc_info=True,
                )
                if self._consecutive_error_count >= self._max_consecutive_errors:
                    logger.error(
                        f"Maximum consecutive errors ({self._max_consecutive_errors}) reached. "
                        "Terminating scheduler event loop."
                    )
                    raise RuntimeError(
                        f"Scheduler terminated after {self._max_consecutive_errors} "
                        f"consecutive errors. Last error: {e}"
                    ) from e
                continue

            # 2: execute, make sure a reply is always sent
            items = self.get_next_batch_to_run()
            if not items:
                continue

            identities = [item[0] for item in items]
            reqs = [item[1] for item in items]

            try:
                processed_req = reqs[0]
                handler = self.request_handlers.get(type(processed_req))
                if handler:
                    output_batch = handler(reqs)
                else:
                    output_batch = OutputBatch(
                        error=f"Unknown request type: {type(processed_req)}"
                    )
            except Exception as e:
                logger.error(
                    f"Error executing request in scheduler event loop: {e}",
                    exc_info=True,
                )
                # Determine appropriate error response format
                output_batch = (
                    OutputBatch(error=str(e))
                    if reqs and isinstance(reqs[0], Req)
                    else OutputBatch(error=str(e))
                )

            # 3. return results
            try:
                # log warmup info
                is_warmup = (
                    processed_req.is_warmup if isinstance(processed_req, Req) else False
                )
                if is_warmup:
                    if output_batch.error is None:
                        if self._warmup_total > 0:
                            logger.info(
                                f"Warmup req ({self._warmup_processed}/{self._warmup_total}) processed in {GREEN}%.2f{RESET} seconds",
                                output_batch.metrics.total_duration_s,
                            )
                        else:
                            logger.info(
                                f"Warmup req processed in {GREEN}%.2f{RESET} seconds",
                                output_batch.metrics.total_duration_s,
                            )
                    else:
                        if self._warmup_total > 0:
                            logger.info(
                                f"Warmup req ({self._warmup_processed}/{self._warmup_total}) processing failed"
                            )
                        else:
                            logger.info(f"Warmup req processing failed")

                # TODO: Support sending back to multiple identities if batched
                self.return_result(output_batch, identities[0], is_warmup=is_warmup)
            except zmq.ZMQError as e:
                # Reply failed; log and keep loop alive to accept future requests
                logger.error(f"ZMQ error sending reply: {e}")
                continue

        if self.receiver is not None:
            self.receiver.close()
        self._cleanup_disagg_connectors()
        self.context.destroy(linger=0)

    def _broadcast_task(self, payload: dict[str, Any]) -> None:
        """Broadcast a task to all slave worker processes."""
        method = payload["method"]
        kwargs = {k: v for k, v in payload.items() if k != "method"}
        task = {"method": method, "kwargs": kwargs}
        for pipe in self.task_pipes_to_slaves:
            pipe.send(task)

    def _collect_slave_results(self) -> List[dict[str, Any]]:
        """Collect results from all slave worker processes."""
        results = []
        for pipe in self.result_pipes_from_slaves:
            results.append(pipe.recv())
        return results
