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
    GetDisaggStatsReq,
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
            GetDisaggStatsReq: self._handle_get_disagg_stats,
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

        # Phase 4: Async pipelining state for ENCODER role.
        # Maps request_id -> (zmq_identity, RequestMetrics, submit_time)
        # so the encoder can fire-and-forget to the denoiser and reply
        # to the HTTP client later when the decoder result arrives.
        self._pending_disagg: dict[str, tuple[bytes | None, Any, float]] = {}
        self._disagg_timeout_s: float = float(
            getattr(server_args, "disagg_timeout", 600)
        )

        # Phase 4 P3: Per-role observability metrics
        self._disagg_metrics = None
        self._pool_mode = getattr(server_args, "disagg_pool_mode", False)
        # Pool mode sockets (set by _init_pool_mode_sockets)
        self._pool_work_pull = None
        self._pool_result_push = None

        if self._disagg_role != RoleType.MONOLITHIC:
            from sglang.multimodal_gen.runtime.disaggregation.metrics import (
                DisaggMetrics,
            )

            self._disagg_metrics = DisaggMetrics(role=self._disagg_role.value)
            if self._pool_mode:
                self._init_pool_mode_sockets()
            else:
                self._init_disagg_connectors()

    def get_disagg_metrics(self) -> dict | None:
        """Return disagg role metrics snapshot, or None if not in disagg mode."""
        if self._disagg_metrics is None:
            return None
        return self._disagg_metrics.snapshot().to_dict()

    def _handle_get_disagg_stats(self, _reqs: List[Any]) -> OutputBatch:
        """Handle stats request — return disagg metrics via OutputBatch.output."""
        stats = self.get_disagg_metrics()
        return OutputBatch(
            output=stats or {"role": "monolithic", "message": "not in disagg mode"}
        )

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

        # Note: disagg ENCODER is now handled directly in the event loop
        # via _handle_generation_disagg_encoder() with identity/is_warmup args.
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
            # Denoiser receives from encoder (tensors go directly to GPU)
            self._role_receiver = create_encoder_to_denoiser_receiver(
                self.context, sa.encoder_to_denoiser_endpoint, device="cuda"
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
            # Decoder receives from denoiser (tensors go directly to GPU)
            self._role_receiver = create_denoiser_to_decoder_receiver(
                self.context, sa.denoiser_to_decoder_endpoint, device="cuda"
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

        P2: After receiving the first request (blocking), drains any
        additional queued requests (non-blocking) to process them in a
        batch (sequentially — GPU batching requires pipeline changes).
        """
        from sglang.multimodal_gen.runtime.disaggregation.tensor_transport import (
            send_tensors,
        )

        role_name = self._disagg_role.value.upper()
        recv_timeout_ms = int(self._disagg_timeout_s * 1000)
        logger.info(
            "Disagg %s event loop started, recv_timeout=%ds, waiting for work...",
            role_name,
            self._disagg_timeout_s,
        )

        while self._running:
            try:
                # 1. Receive first request (blocking with timeout)
                try:
                    req = self._role_receiver.recv(timeout_ms=recv_timeout_ms)
                except TimeoutError:
                    logger.warning(
                        "Disagg %s: recv timed out after %ds, retrying...",
                        role_name,
                        self._disagg_timeout_s,
                    )
                    continue

                # P2: Drain additional queued requests (non-blocking)
                batch = [req]
                while True:
                    extra = self._role_receiver.try_recv()
                    if extra is None:
                        break
                    batch.append(extra)

                if len(batch) > 1:
                    logger.info(
                        "Disagg %s: drained %d requests from queue",
                        role_name,
                        len(batch),
                    )

                # 2. Process each request in the batch sequentially
                for req in batch:
                    request_id = getattr(req, "request_id", "unknown")
                    logger.debug(
                        "Disagg %s: processing request %s", role_name, request_id
                    )

                    # Record metrics
                    if self._disagg_metrics:
                        self._disagg_metrics.record_request_start(request_id)

                    try:
                        self._process_disagg_request(req, send_tensors)
                        if self._disagg_metrics:
                            self._disagg_metrics.record_request_complete(request_id)
                    except Exception as e:
                        if self._disagg_metrics:
                            self._disagg_metrics.record_request_failed(request_id)
                        raise

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

    def _process_disagg_request(self, req: Req, send_tensors) -> None:
        """Process a single request for DENOISING or DECODER role.

        Extracted from the event loop to enable batch iteration.
        """
        request_id = getattr(req, "request_id", "unknown")

        if self._disagg_role == RoleType.DENOISING:
            # Initialize the scheduler timesteps (normally done by
            # TimestepPreparationStage which runs on the encoder side)
            scheduler_mod = self.worker.pipeline.get_module("scheduler")
            num_steps = getattr(req, "num_inference_steps", None)
            if scheduler_mod is not None and num_steps is not None:
                device = torch.device(f"cuda:{self.gpu_id}")
                scheduler_mod.set_timesteps(num_steps, device=device)

            # Run denoising stages, get back Req with updated latents
            start_time = time.monotonic()
            result = self.worker.execute_forward([req], return_req=True)
            duration_s = time.monotonic() - start_time

            if isinstance(result, Req):
                self._role_sender.send(result)
                logger.info(
                    "Disagg DENOISING: processed request %s in %.2f s, sent to decoder",
                    getattr(result, "request_id", "unknown"),
                    duration_s,
                )
            else:
                # Error: forward error through decoder path to encoder
                error_msg = getattr(result, "error", "unknown denoiser error")
                logger.error(
                    "Disagg DENOISING: error processing request %s: %s, "
                    "forwarding error to decoder",
                    request_id,
                    error_msg,
                )
                req.latents = None
                req._disagg_error = error_msg
                self._role_sender.send(req)

        elif self._disagg_role == RoleType.DECODER:
            # Check for upstream error from denoiser
            disagg_error = getattr(req, "_disagg_error", None)
            if disagg_error:
                logger.warning(
                    "Disagg DECODER: received error from denoiser for %s: %s, "
                    "forwarding to encoder",
                    request_id,
                    disagg_error,
                )
                scalar_fields = {
                    "request_id": request_id,
                    "error": f"Denoiser error: {disagg_error}",
                }
                send_tensors(self._result_sender, {}, scalar_fields)
                return

            # In disagg mode, the decoder must return the raw output tensor
            req.save_output = False
            req.return_file_paths_only = False

            start_time = time.monotonic()
            output_batch = self.worker.execute_forward([req])
            duration_s = time.monotonic() - start_time

            # Send result back to encoder via result channel
            tensor_fields = {}
            scalar_fields = {"request_id": request_id}
            if output_batch.output is not None:
                tensor_fields["output"] = output_batch.output
            else:
                logger.warning(
                    "Disagg DECODER: output_batch.output is None for request %s, "
                    "error=%s, output_file_paths=%s",
                    request_id,
                    output_batch.error,
                    output_batch.output_file_paths,
                )
            if output_batch.audio is not None:
                tensor_fields["audio"] = output_batch.audio
            if output_batch.audio_sample_rate is not None:
                scalar_fields["audio_sample_rate"] = output_batch.audio_sample_rate
            if output_batch.error is not None:
                scalar_fields["error"] = output_batch.error

            send_tensors(self._result_sender, tensor_fields, scalar_fields)
            logger.info(
                "Disagg DECODER: processed request %s in %.2f s, sent result to encoder",
                request_id,
                duration_s,
            )

    def _handle_generation_disagg_encoder(
        self, reqs: List[Req], identity: bytes | None = None, is_warmup: bool = False
    ) -> OutputBatch | None:
        """Handle generation for ENCODER role in disagg mode (non-blocking).

        1. Run encoder pipeline stages → get Req with embeddings/latents
        2. Fire-and-forget to denoiser
        3. Stash identity in _pending_disagg for deferred reply
        4. Return None to signal the event loop that the reply is deferred

        For warmup requests, blocks synchronously (no HTTP client to defer).
        """
        from sglang.multimodal_gen.runtime.disaggregation.tensor_transport import (
            recv_tensors,
        )

        # Run encoder stages (return raw Req to access intermediate tensors)
        req_result = self.worker.execute_forward(reqs, return_req=True)

        if not isinstance(req_result, Req):
            # Error case — return immediately
            request_id = (
                getattr(reqs[0], "request_id", "unknown") if reqs else "unknown"
            )
            if self._disagg_metrics:
                self._disagg_metrics.record_request_failed(request_id)
            return req_result

        request_id = getattr(req_result, "request_id", None)

        # Track metrics
        if self._disagg_metrics and not is_warmup:
            self._disagg_metrics.record_request_start(request_id)

        # Send encoder outputs to denoiser
        self._role_sender.send(req_result)
        logger.debug(
            "Disagg ENCODER: sent request %s to denoiser (pending=%d)",
            request_id,
            len(self._pending_disagg),
        )

        if is_warmup:
            # Warmup: block-wait for result (no HTTP client to defer to)
            tensor_fields, scalar_fields = recv_tensors(self._result_receiver)
            return OutputBatch(
                output=tensor_fields.get("output"),
                audio=tensor_fields.get("audio"),
                audio_sample_rate=scalar_fields.get("audio_sample_rate"),
                error=scalar_fields.get("error"),
                metrics=req_result.metrics,
            )

        # Stash identity + metrics for deferred reply
        self._pending_disagg[request_id] = (
            identity,
            req_result.metrics,
            time.monotonic(),
        )
        return None  # Sentinel: reply deferred

    def _poll_disagg_results(self) -> None:
        """Non-blocking poll for completed decoder results.

        Drains all available results from _result_receiver and sends
        ZMQ replies back to the HTTP clients whose requests are complete.
        """
        from sglang.multimodal_gen.runtime.disaggregation.tensor_transport import (
            recv_tensors,
        )

        while True:
            try:
                tensor_fields, scalar_fields = recv_tensors(
                    self._result_receiver, flags=zmq.NOBLOCK
                )
            except zmq.Again:
                break  # No more results available

            request_id = scalar_fields.get("request_id")
            pending = self._pending_disagg.pop(request_id, None)
            if pending is None:
                logger.warning(
                    "Disagg ENCODER: received result for unknown request_id=%s "
                    "(may have timed out already)",
                    request_id,
                )
                continue

            identity, metrics, _submit_time = pending

            output_batch = OutputBatch(
                output=tensor_fields.get("output"),
                audio=tensor_fields.get("audio"),
                audio_sample_rate=scalar_fields.get("audio_sample_rate"),
                error=scalar_fields.get("error"),
                metrics=metrics,
            )

            # Record metrics
            if self._disagg_metrics:
                if output_batch.error:
                    self._disagg_metrics.record_request_failed(request_id)
                else:
                    self._disagg_metrics.record_request_complete(request_id)

            try:
                self.return_result(output_batch, identity)
            except zmq.ZMQError as e:
                logger.error(
                    "Disagg ENCODER: failed to send result for request %s: %s",
                    request_id,
                    e,
                )

            logger.debug(
                "Disagg ENCODER: returned result for request %s (pending=%d)",
                request_id,
                len(self._pending_disagg),
            )

    def _check_disagg_timeouts(self) -> None:
        """Check for timed-out pending disagg requests and send error replies."""
        now = time.monotonic()
        timed_out = [
            rid
            for rid, (_ident, _metrics, submit_time) in self._pending_disagg.items()
            if now - submit_time > self._disagg_timeout_s
        ]

        for request_id in timed_out:
            identity, metrics, submit_time = self._pending_disagg.pop(request_id)
            elapsed = now - submit_time
            error_msg = (
                f"Disagg pipeline timeout: request {request_id} "
                f"not completed within {elapsed:.1f}s "
                f"(limit={self._disagg_timeout_s}s)"
            )
            logger.error("Disagg ENCODER: %s", error_msg)
            if self._disagg_metrics:
                self._disagg_metrics.record_request_timeout(request_id)
            output_batch = OutputBatch(error=error_msg, metrics=metrics)
            try:
                self.return_result(output_batch, identity)
            except zmq.ZMQError as e:
                logger.error(
                    "Disagg ENCODER: failed to send timeout error for %s: %s",
                    request_id,
                    e,
                )

    def _init_pool_mode_sockets(self):
        """Initialize ZMQ sockets for pool mode (DiffusionServer-mediated)."""
        sa = self.server_args

        # PULL: receive work from DiffusionServer
        self._pool_work_pull, _ = get_zmq_socket(
            self.context, zmq.PULL, sa.pool_work_endpoint, bind=True
        )
        # PUSH: send results to DiffusionServer
        self._pool_result_push, _ = get_zmq_socket(
            self.context, zmq.PUSH, sa.pool_result_endpoint, bind=False
        )
        logger.info(
            "Pool mode %s: work_pull=%s, result_push=%s",
            self._disagg_role.value.upper(),
            sa.pool_work_endpoint,
            sa.pool_result_endpoint,
        )

    def _pool_mode_event_loop(self) -> None:
        """Event loop for all roles in pool mode (DiffusionServer-mediated).

        Each role:
        1. Receives work from DiffusionServer via PULL socket
        2. Processes the request
        3. Sends result to DiffusionServer via PUSH socket

        The data format depends on the role:
        - Encoder: receives [request_id, pickled_req], sends tensor multipart
        - Denoiser: receives tensor multipart, sends tensor multipart
        - Decoder: receives tensor multipart, sends tensor multipart
        """
        from sglang.multimodal_gen.runtime.disaggregation.role_connector import (
            DENOISER_TO_DECODER_SCALAR_FIELDS,
            DENOISER_TO_DECODER_TENSOR_FIELDS,
            ENCODER_TO_DENOISER_SCALAR_FIELDS,
            ENCODER_TO_DENOISER_TENSOR_FIELDS,
            _extract_scalar_fields,
            _extract_tensor_fields,
            build_req_from_frames,
        )
        from sglang.multimodal_gen.runtime.disaggregation.tensor_transport import (
            send_tensors,
        )

        role_name = self._disagg_role.value.upper()
        logger.info("Pool mode %s event loop started, waiting for work...", role_name)

        while self._running:
            try:
                if self._disagg_role == RoleType.ENCODER:
                    self._pool_mode_encoder_step(
                        send_tensors,
                        _extract_tensor_fields,
                        _extract_scalar_fields,
                        ENCODER_TO_DENOISER_TENSOR_FIELDS,
                        ENCODER_TO_DENOISER_SCALAR_FIELDS,
                    )
                elif self._disagg_role == RoleType.DENOISING:
                    self._pool_mode_denoiser_step(
                        send_tensors,
                        build_req_from_frames,
                        _extract_tensor_fields,
                        _extract_scalar_fields,
                        DENOISER_TO_DECODER_TENSOR_FIELDS,
                        DENOISER_TO_DECODER_SCALAR_FIELDS,
                    )
                elif self._disagg_role == RoleType.DECODER:
                    self._pool_mode_decoder_step(
                        send_tensors,
                        build_req_from_frames,
                    )

                self._consecutive_error_count = 0

            except Exception as e:
                self._consecutive_error_count += 1
                logger.error(
                    "Pool %s: error (attempt %d/%d): %s",
                    role_name,
                    self._consecutive_error_count,
                    self._max_consecutive_errors,
                    e,
                    exc_info=True,
                )
                if self._consecutive_error_count >= self._max_consecutive_errors:
                    raise RuntimeError(
                        f"Pool {role_name} terminated after "
                        f"{self._max_consecutive_errors} consecutive errors: {e}"
                    ) from e

        self._pool_work_pull.close()
        self._pool_result_push.close()

    def _pool_mode_encoder_step(
        self,
        send_tensors_fn,
        extract_tensor,
        extract_scalar,
        tensor_field_names,
        scalar_field_names,
    ):
        """Single encoder step in pool mode."""
        # Receive: [request_id_bytes, pickled_req_bytes]
        frames = self._pool_work_pull.recv_multipart()
        pickled_req = frames[-1]
        reqs = pickle.loads(pickled_req)
        if not isinstance(reqs, list):
            reqs = [reqs]

        req = reqs[0]
        request_id = getattr(req, "request_id", "unknown")

        if self._disagg_metrics:
            self._disagg_metrics.record_request_start(request_id)

        # Run encoder stages
        req_result = self.worker.execute_forward(reqs, return_req=True)

        if not isinstance(req_result, Req):
            # Error — send error via scalar fields
            error_msg = getattr(req_result, "error", "encoder error")
            send_tensors_fn(
                self._pool_result_push,
                {},
                {"request_id": request_id, "_disagg_error": str(error_msg)},
            )
            if self._disagg_metrics:
                self._disagg_metrics.record_request_failed(request_id)
            return

        # Pack and send encoder output
        tensor_fields = extract_tensor(req_result, tensor_field_names)
        scalar_fields = extract_scalar(req_result, scalar_field_names)
        send_tensors_fn(self._pool_result_push, tensor_fields, scalar_fields)

        if self._disagg_metrics:
            self._disagg_metrics.record_request_complete(request_id)

        logger.debug("Pool ENCODER: processed %s", request_id)

    def _pool_mode_denoiser_step(
        self,
        send_tensors_fn,
        build_req_fn,
        extract_tensor,
        extract_scalar,
        tensor_field_names,
        scalar_field_names,
    ):
        """Single denoiser step in pool mode."""
        # Receive tensor multipart from DiffusionServer relay
        frames = self._pool_work_pull.recv_multipart(copy=False)
        req = build_req_fn(frames, "encoder_to_denoiser", device="cuda")
        request_id = getattr(req, "request_id", "unknown")

        if self._disagg_metrics:
            self._disagg_metrics.record_request_start(request_id)

        # Initialize scheduler timesteps
        scheduler_mod = self.worker.pipeline.get_module("scheduler")
        num_steps = getattr(req, "num_inference_steps", None)
        if scheduler_mod is not None and num_steps is not None:
            device = torch.device(f"cuda:{self.gpu_id}")
            scheduler_mod.set_timesteps(num_steps, device=device)

        # Run denoising
        result = self.worker.execute_forward([req], return_req=True)

        if isinstance(result, Req):
            tensor_fields = extract_tensor(result, tensor_field_names)
            scalar_fields = extract_scalar(result, scalar_field_names)
            send_tensors_fn(self._pool_result_push, tensor_fields, scalar_fields)
            if self._disagg_metrics:
                self._disagg_metrics.record_request_complete(request_id)
        else:
            error_msg = getattr(result, "error", "denoiser error")
            send_tensors_fn(
                self._pool_result_push,
                {},
                {"request_id": request_id, "_disagg_error": str(error_msg)},
            )
            if self._disagg_metrics:
                self._disagg_metrics.record_request_failed(request_id)

        logger.debug("Pool DENOISER: processed %s", request_id)

    def _pool_mode_decoder_step(self, send_tensors_fn, build_req_fn):
        """Single decoder step in pool mode."""
        # Receive tensor multipart from DiffusionServer relay
        frames = self._pool_work_pull.recv_multipart(copy=False)
        req = build_req_fn(frames, "denoiser_to_decoder", device="cuda")
        request_id = getattr(req, "request_id", "unknown")

        # Check for upstream error
        disagg_error = getattr(req, "_disagg_error", None)
        if disagg_error:
            send_tensors_fn(
                self._pool_result_push,
                {},
                {"request_id": request_id, "error": f"Denoiser error: {disagg_error}"},
            )
            return

        if self._disagg_metrics:
            self._disagg_metrics.record_request_start(request_id)

        req.save_output = False
        req.return_file_paths_only = False

        output_batch = self.worker.execute_forward([req])

        # Pack result
        tensor_fields = {}
        scalar_fields = {"request_id": request_id}
        if output_batch.output is not None:
            tensor_fields["output"] = output_batch.output
        if output_batch.audio is not None:
            tensor_fields["audio"] = output_batch.audio
        if output_batch.audio_sample_rate is not None:
            scalar_fields["audio_sample_rate"] = output_batch.audio_sample_rate
        if output_batch.error is not None:
            scalar_fields["error"] = output_batch.error

        send_tensors_fn(self._pool_result_push, tensor_fields, scalar_fields)

        if self._disagg_metrics:
            if output_batch.error:
                self._disagg_metrics.record_request_failed(request_id)
            else:
                self._disagg_metrics.record_request_complete(request_id)

        logger.debug("Pool DECODER: processed %s", request_id)

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
        # Pool mode: all roles use the pool event loop
        if self._pool_mode and self._disagg_role != RoleType.MONOLITHIC:
            self._pool_mode_event_loop()
            return

        # For DENOISING/DECODER roles, use the disagg event loop (chain mode)
        if self._disagg_role in (RoleType.DENOISING, RoleType.DECODER):
            self._disagg_event_loop()
            return

        is_disagg_encoder = self._disagg_role == RoleType.ENCODER

        logger.debug(
            f"Rank 0 scheduler listening on tcp://*:{self.server_args.scheduler_port}"
        )

        while self._running:
            # Phase 4: Poll for completed disagg results (non-blocking)
            if is_disagg_encoder and self._pending_disagg:
                self._poll_disagg_results()
                self._check_disagg_timeouts()

            # P3: Update queue depth for metrics
            if self._disagg_metrics:
                self._disagg_metrics.update_queue_depth(len(self.waiting_queue))

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
                # Brief sleep to avoid busy-spin when there are pending disagg results
                if is_disagg_encoder and self._pending_disagg:
                    time.sleep(0.001)
                continue

            identities = [item[0] for item in items]
            reqs = [item[1] for item in items]

            try:
                processed_req = reqs[0]
                is_warmup = (
                    processed_req.is_warmup if isinstance(processed_req, Req) else False
                )

                handler = self.request_handlers.get(type(processed_req))
                if handler:
                    # Disagg encoder handler needs identity + is_warmup for deferred reply
                    if is_disagg_encoder and handler == self._handle_generation:
                        output_batch = self._handle_generation_disagg_encoder(
                            reqs,
                            identity=identities[0],
                            is_warmup=is_warmup,
                        )
                    else:
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
                output_batch = OutputBatch(error=str(e))

            # If output_batch is None, the reply is deferred (disagg encoder async)
            if output_batch is None:
                continue

            # 3. return results
            try:
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
