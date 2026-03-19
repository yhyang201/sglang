# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
import asyncio
import dataclasses
import json
import os
import pickle
import queue
import threading
import time
from collections import deque
from typing import Any, List

import torch
import zmq

from sglang.multimodal_gen.configs.pipeline_configs.base import ModelTaskType
from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.disaggregation.transport.p2p_protocol import (
    P2P_MAGIC,
    P2PAllocatedMsg,
    P2PDoneMsg,
    P2PMsgType,
    P2PPushedMsg,
    P2PRegisterMsg,
    decode_p2p_msg,
    encode_p2p_msg,
    is_p2p_message,
)
from sglang.multimodal_gen.runtime.disaggregation.transport.role_connector import (
    DENOISER_TO_DECODER_SCALAR_FIELDS,
    DENOISER_TO_DECODER_TENSOR_FIELDS,
    ENCODER_TO_DENOISER_SCALAR_FIELDS,
    ENCODER_TO_DENOISER_TENSOR_FIELDS,
    _extract_scalar_fields,
    _extract_tensor_fields,
)
from sglang.multimodal_gen.runtime.disaggregation.transport.tensor_codec import (
    send_tensors,
)
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
        local_rank: int | None = None,
    ):
        self.server_args = server_args
        self.port_args = port_args

        # local_rank is the physical GPU index for torch.cuda.set_device.
        # In non-disagg mode, it equals gpu_id. In disagg mode, it may differ
        # (e.g., denoiser rank 0 on physical GPU 1).
        if local_rank is None:
            local_rank = gpu_id

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
            local_rank=local_rank,
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

        # Disaggregation state
        self._disagg_role = server_args.disagg_role
        self._disagg_timeout_s: float = float(
            getattr(server_args, "disagg_timeout", 600)
        )

        # Per-role observability metrics
        self._disagg_metrics = None
        self._disagg_mode = getattr(server_args, "disagg_mode", False)
        # Disagg sockets (set by _init_disagg_sockets)
        self._pool_work_pull = None
        self._pool_result_push = None
        # P2P transfer manager (set by _init_disagg_transfer_manager)
        self._transfer_manager = None

        # Async transfer infrastructure (Phase 1 + Phase 8)
        self._transfer_stream = None
        # Phase 8a: Background RDMA push thread (sender roles)
        self._rdma_push_queue = None
        self._rdma_push_thread = None
        self._rdma_push_zmq = None
        # Phase 8b: Background recv+H2D prefetch thread (receiver roles)
        self._compute_ready_queue = None
        self._recv_prefetch_thread = None

        if self._disagg_role != RoleType.MONOLITHIC:
            from sglang.multimodal_gen.runtime.disaggregation.metrics import (
                DisaggMetrics,
            )

            self._disagg_metrics = DisaggMetrics(role=self._disagg_role.value)

            # Phase 1: Dedicated CUDA stream for D2H/H2D transfers
            device = torch.device(f"cuda:{local_rank}")
            self._transfer_stream = torch.cuda.Stream(device=device)

            self._init_disagg_sockets()
            self._init_disagg_transfer_manager()

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

    def _init_disagg_sockets(self):
        """Initialize ZMQ sockets for disaggregated mode (DiffusionServer-mediated).

        Only rank 0 creates ZMQ sockets. Non-rank-0 processes participate
        via NCCL broadcast from rank 0 (see _disagg_recv_work).
        """
        if self.gpu_id != 0:
            logger.info(
                "Pool mode %s rank %d: no ZMQ sockets (non-rank-0)",
                self._disagg_role.value.upper(),
                self.gpu_id,
            )
            return

        sa = self.server_args

        # PULL: receive work from DiffusionServer
        # Port must match what DiffusionServer connects to, so we retry the
        # same port (with sleep) rather than incrementing to a new one.
        import time as _time

        last_exc = None
        for _attempt in range(5):
            try:
                self._pool_work_pull, _ = get_zmq_socket(
                    self.context,
                    zmq.PULL,
                    sa.pool_work_endpoint,
                    bind=True,
                    max_bind_retries=1,
                )
                last_exc = None
                break
            except Exception as e:
                last_exc = e
                logger.warning(
                    "Pool work bind attempt %d failed (%s), retrying in 1s...",
                    _attempt + 1,
                    e,
                )
                _time.sleep(1)
        if last_exc is not None:
            raise last_exc
        # PUSH: send results to DiffusionServer
        self._pool_result_push, _ = get_zmq_socket(
            self.context, zmq.PUSH, sa.pool_result_endpoint, bind=False
        )
        logger.info(
            "Pool mode %s rank 0: work_pull=%s, result_push=%s",
            self._disagg_role.value.upper(),
            sa.pool_work_endpoint,
            sa.pool_result_endpoint,
        )

    def _init_disagg_transfer_manager(self):
        """Initialize TransferManager for P2P mode (rank 0 only).

        Creates a TransferTensorBuffer (pinned memory pool) and a
        BaseTransferEngine, then wraps them in a DiffusionTransferManager.
        Also sends a p2p_register message to DiffusionServer.
        """
        if self.gpu_id != 0:
            return
        from sglang.multimodal_gen.runtime.disaggregation.transport.rdma.transfer_buffer import (
            TransferTensorBuffer,
        )
        from sglang.multimodal_gen.runtime.disaggregation.transport.rdma.transfer_engine import (
            create_transfer_engine,
        )
        from sglang.multimodal_gen.runtime.disaggregation.transport.rdma.transfer_manager import (
            DiffusionTransferManager,
        )

        sa = self.server_args

        # Pool size: configurable, default 256 MiB
        pool_size = getattr(sa, "disagg_transfer_pool_size", 256 * 1024 * 1024)

        # Create transfer engine
        hostname = getattr(sa, "disagg_p2p_hostname", "127.0.0.1")
        ib_device = getattr(sa, "disagg_ib_device", None)
        engine = create_transfer_engine(
            hostname=hostname,
            gpu_id=self.gpu_id,
            ib_device=ib_device,
        )

        # Use GPU buffer when engine supports GPUDirect RDMA, CPU pinned otherwise
        device = f"cuda:{self.gpu_id}" if engine.supports_gpu_direct else "cpu"
        buffer = TransferTensorBuffer(
            pool_size=pool_size, device=device, role_name=self._disagg_role.value
        )

        # Create transfer manager
        self._transfer_manager = DiffusionTransferManager(engine=engine, buffer=buffer)

        # Phase 7b: Pre-allocate receive slots for receivers (denoiser/decoder)
        self._preallocated_slots: dict[int, object] = {}
        preallocated_slot_info = []
        if self._disagg_role in (RoleType.DENOISER, RoleType.DECODER):
            capacity = getattr(sa, "disagg_prealloc_slots", 2)
            typical_size = 64 * 1024 * 1024  # 64 MiB per slot
            for i in range(capacity):
                slot = buffer.allocate(typical_size, f"prealloc_{i}")
                if slot is not None:
                    self._preallocated_slots[i] = slot
                    preallocated_slot_info.append(
                        {
                            "offset": slot.offset,
                            "size": slot.size,
                            "slot_id": i,
                            "addr": self._transfer_manager.pool_data_ptr + slot.offset,
                        }
                    )
            if preallocated_slot_info:
                logger.info(
                    "P2P %s: pre-allocated %d receive slots",
                    self._disagg_role.value.upper(),
                    len(preallocated_slot_info),
                )

        # Register with DiffusionServer
        register_msg = P2PRegisterMsg(
            role=self._disagg_role.value,
            instance_idx=0,  # Set by launcher; single instance per process
            session_id=self._transfer_manager.session_id,
            pool_ptr=self._transfer_manager.pool_data_ptr,
            pool_size=self._transfer_manager.pool_size,
            preallocated_slots=preallocated_slot_info,
        )
        self._pool_result_push.send_multipart(encode_p2p_msg(register_msg))
        logger.info(
            "P2P %s: registered with DS (session=%s, pool=%d bytes, prealloc=%d)",
            self._disagg_role.value.upper(),
            self._transfer_manager.session_id,
            pool_size,
            len(preallocated_slot_info),
        )

        # Phase 8a: RDMA push thread for sender roles (encoder/denoiser)
        if self._disagg_role in (RoleType.ENCODER, RoleType.DENOISER):
            self._rdma_push_queue = queue.Queue(maxsize=4)
            self._rdma_push_zmq, _ = get_zmq_socket(
                self.context,
                zmq.PUSH,
                sa.pool_result_endpoint,
                bind=False,
            )
            self._rdma_push_thread = threading.Thread(
                target=self._rdma_push_loop,
                daemon=True,
                name=f"rdma-push-{self._disagg_role.value}",
            )
            self._rdma_push_thread.start()
            logger.info(
                "P2P %s: RDMA push thread started",
                self._disagg_role.value.upper(),
            )

        # Phase 8b: Recv prefetch thread for receiver roles (denoiser/decoder)
        # Only for single-rank P2P mode (multi-rank requires NCCL broadcast)
        is_single_rank = (
            sa.sp_degree == 1 and sa.tp_size <= 1 and not sa.enable_cfg_parallel
        )
        if (
            self._disagg_role in (RoleType.DENOISER, RoleType.DECODER)
            and is_single_rank
        ):
            self._compute_ready_queue = queue.Queue(maxsize=4)
            self._recv_prefetch_thread = threading.Thread(
                target=self._recv_prefetch_loop,
                daemon=True,
                name=f"recv-prefetch-{self._disagg_role.value}",
            )
            self._recv_prefetch_thread.start()
            logger.info(
                "P2P %s: recv prefetch thread started",
                self._disagg_role.value.upper(),
            )

    # ------------------------------------------------------------------
    # Phase 8a: Background RDMA push (sender side)
    # ------------------------------------------------------------------

    def _rdma_push_loop(self):
        """Background thread: execute RDMA push + notify DS.

        Runs push_to_peer (blocking RDMA) on a dedicated thread so the
        main event loop can immediately start processing the next request.
        """
        role_name = self._disagg_role.value.upper()
        while True:
            item = self._rdma_push_queue.get()
            if item is None:
                break  # Shutdown signal
            request_id, dest_session_id, dest_addr, transfer_size = item
            try:
                success = self._transfer_manager.push_to_peer(
                    request_id=request_id,
                    dest_session_id=dest_session_id,
                    dest_addr=dest_addr,
                    transfer_size=transfer_size,
                )
                if success:
                    self._transfer_manager.free_staged(request_id)

                pushed_msg = P2PPushedMsg(request_id=request_id)
                self._rdma_push_zmq.send_multipart(encode_p2p_msg(pushed_msg))

                if not success:
                    logger.error(
                        "P2P %s: RDMA push failed for %s", role_name, request_id
                    )
            except Exception:
                logger.exception(
                    "P2P %s: RDMA push thread error for %s", role_name, request_id
                )

    # ------------------------------------------------------------------
    # Phase 8b: Background recv + H2D prefetch (receiver side)
    # ------------------------------------------------------------------

    def _recv_prefetch_loop(self):
        """Background thread: recv P2P messages and prefetch H2D.

        For p2p_ready: does H2D + Req construction in this thread, then
        enqueues the ready-to-compute item. This allows H2D of request N+1
        to overlap with compute of request N on the main thread.

        For p2p_alloc/push: passes them through to the main thread for handling.
        """
        role_name = self._disagg_role.value.upper()
        while self._running:
            try:
                raw_frames = self._pool_work_pull.recv_multipart()
                frames = [bytes(f) for f in raw_frames]

                msg = decode_p2p_msg(frames)
                msg_type = msg.get("msg_type", "")

                if msg_type == P2PMsgType.READY:
                    # Prefetch: H2D + Req build in this thread
                    item = self._prefetch_p2p_ready(msg)
                    self._compute_ready_queue.put(("p2p_compute", item))
                else:
                    # alloc, push: pass to main thread
                    self._compute_ready_queue.put(("p2p_control", frames))

            except zmq.ZMQError as e:
                if not self._running:
                    break
                logger.error("P2P %s recv prefetch: ZMQ error: %s", role_name, e)
            except Exception:
                logger.exception("P2P %s recv prefetch: error", role_name)

    def _prefetch_p2p_ready(self, msg: dict) -> tuple:
        """Prefetch H2D and build Req for a p2p_ready message.

        Called from the recv prefetch thread. Does H2D on _transfer_stream
        and builds the Req, so the main thread can start compute immediately.

        Returns (req, h2d_event, request_id, role_name).
        """
        request_id = msg["request_id"]
        manifest = msg.get("manifest", {})
        scalar_fields = msg.get("scalar_fields", {})
        role_name = self._disagg_role.value.upper()

        if self._disagg_metrics:
            self._disagg_metrics.record_request_start(request_id)

        # Phase 7e: pre-allocated slot handling
        prealloc_slot_id = scalar_fields.pop("_prealloc_slot_id", None)
        if (
            prealloc_slot_id is not None
            and prealloc_slot_id in self._preallocated_slots
        ):
            slot = self._preallocated_slots[prealloc_slot_id]
            self._transfer_manager.register_prealloc_as_receive(request_id, slot)

        # H2D on transfer_stream (non-blocking GPU copy)
        local_device = f"cuda:{self.worker.local_rank}"
        tensors, h2d_event = self._transfer_manager.load_tensors_async(
            request_id,
            manifest,
            device=local_device,
            stream=self._transfer_stream,
        )

        # Free receive slot
        if prealloc_slot_id is not None:
            with self._transfer_manager._lock:
                self._transfer_manager._pending_receives.pop(request_id, None)
        else:
            self._transfer_manager.free_receive_slot(request_id)

        # Build Req (CPU work, overlapped with H2D)
        req = self._build_req_from_p2p(scalar_fields, tensors)

        # Init scheduler timesteps (CPU work, overlapped with H2D)
        if self._disagg_role == RoleType.DENOISER:
            scheduler_mod = self.worker.pipeline.get_module("scheduler")
            num_steps = getattr(req, "num_inference_steps", None)
            if scheduler_mod is not None and num_steps is not None:
                device = torch.device(local_device)
                scheduler_mod.set_timesteps(num_steps, device=device)

        return (req, h2d_event, request_id, role_name)

    def _disagg_recv_work(self) -> list[bytes] | None:
        """Receive work frames in pool mode, with multi-rank broadcast.

        Rank 0: recv from ZMQ PULL socket, broadcast to other ranks.
        Non-rank-0: receive via NCCL broadcast from rank 0.

        Returns list of bytes frames, or None on shutdown.
        """
        is_rank0 = self.gpu_id == 0
        sa = self.server_args

        if is_rank0:
            # Rank 0: receive from DiffusionServer
            raw_frames = self._pool_work_pull.recv_multipart()
            # Convert zmq.Frame to bytes for pickling
            frames = [bytes(f) for f in raw_frames]
        else:
            frames = None

        # Broadcast to all ranks if multi-GPU
        if sa.sp_degree != 1:
            frames = broadcast_pyobj(
                frames,
                self.worker.sp_group.rank,
                self.worker.sp_cpu_group,
                src=self.worker.sp_group.ranks[0],
            )

        if sa.enable_cfg_parallel:
            frames = broadcast_pyobj(
                frames,
                self.worker.cfg_group.rank,
                self.worker.cfg_cpu_group,
                src=self.worker.cfg_group.ranks[0],
            )

        if sa.tp_size > 1:
            frames = broadcast_pyobj(
                frames,
                self.worker.tp_group.rank,
                self.worker.tp_cpu_group,
                src=self.worker.tp_group.ranks[0],
            )

        return frames

    def _disagg_prefetch_event_loop(self, role_name: str) -> None:
        """Event loop for P2P receiver roles with recv prefetch thread.

        Phase 8b: The recv thread reads from ZMQ and does H2D prep.
        This loop reads from _compute_ready_queue:
          - "p2p_compute": H2D already done, just wait_event + compute
          - "p2p_control": alloc/push messages, handle on main thread
        """
        while self._running:
            try:
                try:
                    msg_type, data = self._compute_ready_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                if msg_type == "p2p_compute":
                    # H2D already done by recv thread
                    req, h2d_event, request_id, rn = data
                    # Wait for H2D to complete on compute stream
                    if h2d_event is not None:
                        torch.cuda.current_stream().wait_event(h2d_event)
                    # Run compute
                    if self._disagg_role == RoleType.DENOISER:
                        self._p2p_denoiser_compute(req, request_id, rn)
                    elif self._disagg_role == RoleType.DECODER:
                        self._p2p_decoder_compute(req, request_id, rn)

                elif msg_type == "p2p_control":
                    # alloc, push messages — handle on main thread
                    self._handle_p2p_message(data)

                self._consecutive_error_count = 0

            except Exception as e:
                self._consecutive_error_count += 1
                logger.error(
                    "Pool %s rank %d prefetch loop: error (attempt %d/%d): %s",
                    role_name,
                    self.gpu_id,
                    self._consecutive_error_count,
                    self._max_consecutive_errors,
                    e,
                    exc_info=True,
                )
                if self._consecutive_error_count >= self._max_consecutive_errors:
                    raise RuntimeError(
                        f"Pool {role_name} rank {self.gpu_id} terminated after "
                        f"{self._max_consecutive_errors} consecutive errors: {e}"
                    ) from e

        self._cleanup_disagg()

    def _disagg_event_loop(self) -> None:
        """Event loop for all roles in pool mode (DiffusionServer-mediated).

        Multi-rank support (Phase 7c):
        - Rank 0 receives from ZMQ, broadcasts to other ranks via NCCL
        - All ranks process work (execute_forward with SP/TP sharding)
        - Only rank 0 sends results back to DiffusionServer

        P2P mode (Phase 7b):
        - P2P control messages (p2p_alloc, p2p_push) are rank-0-only.
        - p2p_ready is broadcast to all ranks for compute.
        - Encoder receives pickled Req, runs compute, stages output for P2P.
        - Denoiser/decoder only receive P2P control messages.

        Phase 8b: When recv prefetch thread is active (single-rank P2P receiver),
        the main loop reads from _compute_ready_queue instead of ZMQ directly.
        """
        role_name = self._disagg_role.value.upper()
        is_rank0 = self.gpu_id == 0
        is_multi_rank = (
            self.server_args.sp_degree != 1
            or self.server_args.tp_size > 1
            or self.server_args.enable_cfg_parallel
        )
        use_prefetch = self._compute_ready_queue is not None
        logger.info(
            "Pool mode %s rank %d event loop started " "(multi_rank=%s, prefetch=%s)",
            role_name,
            self.gpu_id,
            is_multi_rank,
            use_prefetch,
        )

        # Phase 8b: prefetch event loop — recv thread handles ZMQ + H2D,
        # main thread only does compute + P2P control messages
        if use_prefetch:
            self._disagg_prefetch_event_loop(role_name)
            return

        while self._running:
            try:
                # All ranks receive work (rank 0 via ZMQ, others via broadcast)
                frames = self._disagg_recv_work()

                # P2P dispatch: check on ALL ranks (frames are broadcast)
                if self._is_p2p_frames(frames):
                    if is_rank0:
                        # Rank 0: handle all P2P messages
                        self._handle_p2p_message(frames)
                    else:
                        # Non-rank-0: only participate in p2p_ready compute
                        self._handle_p2p_non_rank0(frames)
                elif self._disagg_role == RoleType.ENCODER:
                    self._disagg_encoder_step(
                        send_tensors,
                        _extract_tensor_fields,
                        _extract_scalar_fields,
                        ENCODER_TO_DENOISER_TENSOR_FIELDS,
                        ENCODER_TO_DENOISER_SCALAR_FIELDS,
                        frames=frames,
                    )

                self._consecutive_error_count = 0

            except Exception as e:
                self._consecutive_error_count += 1
                logger.error(
                    "Pool %s rank %d: error (attempt %d/%d): %s",
                    role_name,
                    self.gpu_id,
                    self._consecutive_error_count,
                    self._max_consecutive_errors,
                    e,
                    exc_info=True,
                )
                if self._consecutive_error_count >= self._max_consecutive_errors:
                    raise RuntimeError(
                        f"Pool {role_name} rank {self.gpu_id} terminated after "
                        f"{self._max_consecutive_errors} consecutive errors: {e}"
                    ) from e

        self._cleanup_disagg()

    def _cleanup_disagg(self):
        """Clean up all pool mode resources (sockets, threads, transfer manager)."""
        # Phase 8: shutdown RDMA push thread
        if self._rdma_push_queue is not None:
            self._rdma_push_queue.put(None)
        if self._rdma_push_thread is not None:
            self._rdma_push_thread.join(timeout=5)
        if self._rdma_push_zmq is not None:
            self._rdma_push_zmq.close()
        # Phase 8b: recv prefetch thread stops when self._running = False
        if self._recv_prefetch_thread is not None:
            self._recv_prefetch_thread.join(timeout=5)
        if self._transfer_manager is not None:
            self._transfer_manager.cleanup()
        if self._pool_work_pull is not None:
            self._pool_work_pull.close()
        if self._pool_result_push is not None:
            self._pool_result_push.close()

    # ------------------------------------------------------------------
    # P2P message handling (Phase 7b)
    # ------------------------------------------------------------------

    @staticmethod
    def _is_p2p_frames(frames: list) -> bool:
        """Check if ZMQ multipart frames carry a P2P control message."""
        return is_p2p_message(frames)

    def _handle_p2p_message(self, frames: list) -> None:
        """Dispatch a P2P control message to the appropriate handler (rank 0)."""
        msg = decode_p2p_msg(frames)
        msg_type = msg.get("msg_type", "")
        request_id = msg.get("request_id", "")

        logger.debug(
            "P2P %s: received %s for %s",
            self._disagg_role.value.upper(),
            msg_type,
            request_id,
        )

        if msg_type == P2PMsgType.ALLOC:
            self._handle_p2p_alloc(msg)
        elif msg_type == P2PMsgType.PUSH:
            self._handle_p2p_push_cmd(msg)
        elif msg_type == P2PMsgType.READY:
            self._handle_p2p_ready(msg)
        else:
            logger.warning(
                "P2P %s: unknown message type %s",
                self._disagg_role.value.upper(),
                msg_type,
            )

    def _handle_p2p_non_rank0(self, frames: list) -> None:
        """Handle P2P messages on non-rank-0 workers.

        Only p2p_ready requires non-rank-0 participation (for compute).
        p2p_alloc and p2p_push are rank-0-only operations — skip them.
        """
        msg = decode_p2p_msg(frames)
        msg_type = msg.get("msg_type", "")

        if msg_type == P2PMsgType.READY:
            # Participate in compute — but non-rank-0 has no TransferManager.
            # Rank 0 loads tensors and broadcasts; non-rank-0 gets them
            # via the execute_forward's internal NCCL sync.
            # For now, non-rank-0 reconstructs the Req from scalar fields
            # (no tensor data — pipeline broadcasts internally).
            self._handle_p2p_ready_non_rank0(msg)
        # else: p2p_alloc, p2p_push — skip (rank-0-only operations)

    def _handle_p2p_ready_non_rank0(self, msg: dict) -> None:
        """Non-rank-0 handling of p2p_ready: participate in compute only.

        Rank 0 loads tensors from the transfer buffer and runs compute.
        The pipeline's forward() internally uses NCCL to broadcast/scatter
        tensors to all SP/TP ranks. Non-rank-0 needs to enter execute_forward
        with a minimal Req so the NCCL collectives match.
        """
        request_id = msg.get("request_id", "")
        scalar_fields = msg.get("scalar_fields", {})

        # Build a minimal Req with scalar fields only.
        # Tensor fields will be received via NCCL inside execute_forward.
        req = self._build_req_from_p2p(scalar_fields, {})

        if self._disagg_role == RoleType.DENOISER:
            # Initialize scheduler timesteps (same as rank 0)
            scheduler_mod = self.worker.pipeline.get_module("scheduler")
            num_steps = getattr(req, "num_inference_steps", None)
            if scheduler_mod is not None and num_steps is not None:
                device = torch.device(f"cuda:{self.worker.local_rank}")
                scheduler_mod.set_timesteps(num_steps, device=device)

            self.worker.execute_forward([req], return_req=True)

        elif self._disagg_role == RoleType.DECODER:
            req.save_output = False
            req.return_file_paths_only = False
            self.worker.execute_forward([req])

    def _handle_p2p_alloc(self, msg: dict) -> None:
        """Handle p2p_alloc: allocate a receive slot and reply with p2p_allocated."""
        request_id = msg["request_id"]
        data_size = msg.get("data_size", 0)

        pending = self._transfer_manager.allocate_receive_slot(request_id, data_size)
        if pending is None:
            logger.error(
                "P2P %s: failed to allocate receive slot for %s (%d bytes)",
                self._disagg_role.value.upper(),
                request_id,
                data_size,
            )
            return

        allocated_msg = P2PAllocatedMsg(
            request_id=request_id,
            session_id=self._transfer_manager.session_id,
            pool_ptr=self._transfer_manager.pool_data_ptr,
            slot_offset=pending.slot.offset,
            slot_size=pending.slot.size,
        )
        self._pool_result_push.send_multipart(encode_p2p_msg(allocated_msg))

        logger.debug(
            "P2P %s: allocated receive slot for %s (offset=%d, size=%d)",
            self._disagg_role.value.upper(),
            request_id,
            pending.slot.offset,
            pending.slot.size,
        )

    def _handle_p2p_push_cmd(self, msg: dict) -> None:
        """Handle p2p_push: RDMA push staged data to peer, reply with p2p_pushed.

        Phase 8a: If RDMA push thread is active, enqueue non-blocking.
        Otherwise fall back to blocking push (e.g., during shutdown).
        """
        request_id = msg["request_id"]
        dest_session_id = msg.get("dest_session_id", "")
        dest_addr = msg.get("dest_addr", 0)
        transfer_size = msg.get("transfer_size", 0)

        if self._rdma_push_queue is not None:
            # Non-blocking: enqueue to RDMA push thread
            self._rdma_push_queue.put(
                (
                    request_id,
                    dest_session_id,
                    dest_addr,
                    transfer_size,
                )
            )
            return

        # Fallback: blocking push on main thread
        success = self._transfer_manager.push_to_peer(
            request_id=request_id,
            dest_session_id=dest_session_id,
            dest_addr=dest_addr,
            transfer_size=transfer_size,
        )

        if success:
            self._transfer_manager.free_staged(request_id)

        pushed_msg = P2PPushedMsg(request_id=request_id)
        self._pool_result_push.send_multipart(encode_p2p_msg(pushed_msg))

        if not success:
            logger.error(
                "P2P %s: RDMA push failed for %s",
                self._disagg_role.value.upper(),
                request_id,
            )

    def _handle_p2p_ready(self, msg: dict) -> None:
        """Handle p2p_ready: load tensors from buffer, run compute, send result.

        Phase 3c: Overlap H2D with Req construction and scheduler init.
        After the RDMA data arrives:
        1. Start H2D on transfer_stream (non-blocking)
        2. Build Req from scalar fields + tensors (CPU, overlapped)
        3. Init scheduler timesteps if denoiser (CPU, overlapped)
        4. Wait for H2D before compute
        5. Run the role's compute
        """

        request_id = msg["request_id"]
        manifest = msg.get("manifest", {})
        scalar_fields = msg.get("scalar_fields", {})
        role_name = self._disagg_role.value.upper()

        if self._disagg_metrics:
            self._disagg_metrics.record_request_start(request_id)

        # Phase 7e: If using a pre-allocated slot, register it as pending receive
        prealloc_slot_id = scalar_fields.pop("_prealloc_slot_id", None)
        if (
            prealloc_slot_id is not None
            and prealloc_slot_id in self._preallocated_slots
        ):
            slot = self._preallocated_slots[prealloc_slot_id]
            self._transfer_manager.register_prealloc_as_receive(request_id, slot)

        # 1. Start H2D on transfer_stream (non-blocking)
        local_device = f"cuda:{self.worker.local_rank}"
        tensors, h2d_event = self._transfer_manager.load_tensors_async(
            request_id,
            manifest,
            device=local_device,
            stream=self._transfer_stream,
        )

        # Free receive slot after H2D is launched (data copied to GPU)
        if prealloc_slot_id is not None:
            # Pre-allocated slot: just remove from pending receives, don't free buffer
            # (DS recycles the slot via _recycle_prealloc_slot)
            with self._transfer_manager._lock:
                self._transfer_manager._pending_receives.pop(request_id, None)
        else:
            self._transfer_manager.free_receive_slot(request_id)

        # 2. Build Req from scalar fields + tensors (CPU work, overlapped)
        req = self._build_req_from_p2p(scalar_fields, tensors)

        # 3. Init scheduler timesteps if denoiser (CPU work, overlapped)
        if self._disagg_role == RoleType.DENOISER:
            scheduler_mod = self.worker.pipeline.get_module("scheduler")
            num_steps = getattr(req, "num_inference_steps", None)
            if scheduler_mod is not None and num_steps is not None:
                device = torch.device(local_device)
                scheduler_mod.set_timesteps(num_steps, device=device)

        # 4. Wait for H2D before compute (GPU must see the data)
        if h2d_event is not None:
            torch.cuda.current_stream().wait_event(h2d_event)

        # 5. Run compute
        if self._disagg_role == RoleType.DENOISER:
            self._p2p_denoiser_compute(req, request_id, role_name)
        elif self._disagg_role == RoleType.DECODER:
            self._p2p_decoder_compute(req, request_id, role_name)

    def _build_req_from_p2p(self, scalar_fields: dict, tensors: dict) -> "Req":
        """Reconstruct a Req from P2P scalar fields and loaded GPU tensors.

        Initializes all dataclass field defaults first, then overlays
        scalar and tensor fields from the P2P message.
        """
        req = object.__new__(Req)
        # Initialize all dataclass fields with their defaults
        for f in dataclasses.fields(Req):
            if f.default is not dataclasses.MISSING:
                object.__setattr__(req, f.name, f.default)
            elif f.default_factory is not dataclasses.MISSING:
                object.__setattr__(req, f.name, f.default_factory())
        # Ensure sampling_params is not None so __getattr__ delegation works
        object.__setattr__(req, "sampling_params", SamplingParams())
        # Overlay scalar fields from the P2P message
        req.__dict__.update(scalar_fields)
        # Set tensor fields
        for key, value in tensors.items():
            setattr(req, key, value)
        # Recreate torch.Generator from seed (not serializable over P2P)
        seed = scalar_fields.get("seed")
        if seed is not None:
            gen = torch.Generator(device="cpu")
            gen.manual_seed(int(seed))
            req.generator = gen
        return req

    def _p2p_denoiser_compute(
        self, req: "Req", request_id: str, role_name: str
    ) -> None:
        """Run denoiser compute in P2P mode, then stage output for decoder.

        Note: Scheduler timestep init is done in _handle_p2p_ready (Phase 3c)
        to overlap with H2D transfer.
        """
        # Run denoising
        start_time = time.monotonic()
        result = self.worker.execute_forward([req], return_req=True)
        duration_s = time.monotonic() - start_time

        if not isinstance(result, Req):
            error_msg = getattr(result, "error", "denoiser error")
            done_msg = P2PDoneMsg(request_id=request_id, error=str(error_msg))
            self._pool_result_push.send_multipart(encode_p2p_msg(done_msg))
            if self._disagg_metrics:
                self._disagg_metrics.record_request_failed(request_id)
            return

        # Stage denoiser output for decoder transfer (Phase 3b: async D2H)
        tensor_fields = _extract_tensor_fields(
            result, DENOISER_TO_DECODER_TENSOR_FIELDS
        )
        scalar_fields = _extract_scalar_fields(
            result, DENOISER_TO_DECODER_SCALAR_FIELDS
        )

        # 1. Start D2H on transfer_stream (non-blocking CPU return)
        staged, d2h_event = self._transfer_manager.stage_tensors_async(
            request_id=request_id,
            tensor_fields=tensor_fields,
            scalar_fields=scalar_fields,
            stream=self._transfer_stream,
        )

        if staged is None:
            done_msg = P2PDoneMsg(
                request_id=request_id,
                error="Failed to stage denoiser output for decoder",
            )
            self._pool_result_push.send_multipart(encode_p2p_msg(done_msg))
            if self._disagg_metrics:
                self._disagg_metrics.record_request_failed(request_id)
            return

        # 2. Build done_data dict while D2H runs (CPU work, overlapped)
        done_data = {
            "msg_type": "p2p_done",
            "request_id": request_id,
            "staged_for_decoder": True,
            "session_id": self._transfer_manager.session_id,
            "pool_ptr": self._transfer_manager.pool_data_ptr,
            "slot_offset": staged.slot.offset if staged.slot else 0,
            "data_size": staged.slot.size if staged.slot else 0,
            "manifest": staged.manifest,
            "scalar_fields": staged.scalar_fields,
        }
        msg_bytes = json.dumps(done_data, separators=(",", ":")).encode("utf-8")

        # 3. Wait for D2H to complete before sending
        if d2h_event is not None:
            d2h_event.synchronize()

        # 4. Send p2p_done with staged info
        self._pool_result_push.send_multipart([P2P_MAGIC, msg_bytes])

        if self._disagg_metrics:
            self._disagg_metrics.record_request_complete(request_id)

        logger.debug(
            "P2P DENOISER: processed %s in %.2f s, staged for decoder",
            request_id,
            duration_s,
        )

    def _p2p_decoder_compute(self, req: "Req", request_id: str, role_name: str) -> None:
        """Run decoder compute in P2P mode, send result to DS.

        Decoder result is sent as raw ZMQ multipart frames (same format as
        relay mode) so DiffusionServer handles it via _handle_decoder_result_frames
        without hex/JSON overhead.
        """

        # Check for upstream error
        disagg_error = getattr(req, "_disagg_error", None)
        if disagg_error:
            if self._pool_result_push is not None:
                send_tensors(
                    self._pool_result_push,
                    {},
                    {
                        "request_id": request_id,
                        "error": f"Upstream error: {disagg_error}",
                    },
                )
            return

        req.save_output = False
        req.return_file_paths_only = False

        start_time = time.monotonic()
        output_batch = self.worker.execute_forward([req])
        duration_s = time.monotonic() - start_time

        # Send result as raw ZMQ frames (no P2P_MAGIC prefix).
        # DiffusionServer will route it through _handle_decoder_result_frames,
        # the same path as relay mode.
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

        if self._pool_result_push is not None:
            send_tensors(self._pool_result_push, tensor_fields, scalar_fields)

        if self._disagg_metrics:
            if output_batch.error:
                self._disagg_metrics.record_request_failed(request_id)
            else:
                self._disagg_metrics.record_request_complete(request_id)

        logger.debug("P2P DECODER: processed %s in %.2f s", request_id, duration_s)

    def _disagg_encoder_step(
        self,
        send_tensors_fn,
        extract_tensor,
        extract_scalar,
        tensor_field_names,
        scalar_field_names,
        frames=None,
    ):
        """Single encoder step in pool mode."""
        # Receive: [request_id_bytes, pickled_req_bytes]
        if frames is None:
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
            # Error — send error via scalar fields (rank 0 only)
            if self._pool_result_push is not None:
                error_msg = getattr(req_result, "error", "encoder error")
                send_tensors_fn(
                    self._pool_result_push,
                    {},
                    {"request_id": request_id, "_disagg_error": str(error_msg)},
                )
            if self._disagg_metrics:
                self._disagg_metrics.record_request_failed(request_id)
            return

        # Pack and send encoder output (rank 0 only sends)
        tensor_fields = extract_tensor(req_result, tensor_field_names)
        scalar_fields = extract_scalar(req_result, scalar_field_names)

        if self._pool_result_push is not None:
            if self._transfer_manager is not None:
                # P2P mode: stage tensors to TransferBuffer, send p2p_staged
                self._disagg_encoder_p2p_stage(request_id, tensor_fields, scalar_fields)
            else:
                # Fallback: send error (P2P transfer manager not initialized)
                send_tensors_fn(
                    self._pool_result_push,
                    {},
                    {"request_id": request_id, "_disagg_error": "No transfer manager"},
                )

        if self._disagg_metrics:
            self._disagg_metrics.record_request_complete(request_id)

        logger.debug("Pool ENCODER: processed %s", request_id)

    def _disagg_encoder_p2p_stage(
        self, request_id: str, tensor_fields: dict, scalar_fields: dict
    ) -> None:
        """Stage encoder output and send p2p_staged to DS.

        Phase 3a: Overlap D2H with metadata JSON serialization.
        """
        # 1. Start D2H on transfer_stream (non-blocking CPU return)
        staged, d2h_event = self._transfer_manager.stage_tensors_async(
            request_id=request_id,
            tensor_fields=tensor_fields,
            scalar_fields=scalar_fields,
            stream=self._transfer_stream,
        )

        if staged is None:
            # Staging failed — send error via relay as fallback
            send_tensors(
                self._pool_result_push,
                {},
                {"request_id": request_id, "_disagg_error": "P2P staging failed"},
            )
            if self._disagg_metrics:
                self._disagg_metrics.record_request_failed(request_id)
            return

        # 2. Build P2P metadata dict while D2H runs (CPU work, overlapped)
        staged_data = {
            "msg_type": "p2p_staged",
            "request_id": request_id,
            "data_size": staged.slot.size if staged.slot else 0,
            "manifest": staged.manifest,
            "session_id": self._transfer_manager.session_id,
            "pool_ptr": self._transfer_manager.pool_data_ptr,
            "slot_offset": staged.slot.offset if staged.slot else 0,
            "scalar_fields": staged.scalar_fields,
        }
        msg_bytes = json.dumps(staged_data, separators=(",", ":")).encode("utf-8")

        # 3. Wait for D2H to complete before sending (buffer must be ready)
        if d2h_event is not None:
            d2h_event.synchronize()

        # 4. Send P2P staged message
        self._pool_result_push.send_multipart([P2P_MAGIC, msg_bytes])

    def event_loop(self) -> None:
        """
        The main event loop that listens for ZMQ requests.
        Handles abortion
        """
        # Pool mode: all roles use the pool event loop
        if self._disagg_role != RoleType.MONOLITHIC:
            self._disagg_event_loop()
            return

        logger.debug(
            f"Rank 0 scheduler listening on tcp://*:{self.server_args.scheduler_port}"
        )

        while self._running:
            # Update queue depth for metrics
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
