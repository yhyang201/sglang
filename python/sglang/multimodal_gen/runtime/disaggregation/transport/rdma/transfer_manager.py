# SPDX-License-Identifier: Apache-2.0
"""
Per-instance P2P transfer manager for disaggregated diffusion roles.

Each role instance (encoder/denoiser/decoder) creates a DiffusionTransferManager
that owns a TransferTensorBuffer (pinned memory pool) and a BaseTransferEngine
(RDMA or mock). The manager handles:

  - Buffer registration with the transfer engine (RDMA-accessible)
  - D2H staging: GPU tensors → pinned memory slot (with manifest)
  - P2P push: local slot → remote slot via RDMA
  - H2D loading: pinned memory slot → GPU tensors (from manifest)
  - Slot lifecycle management (allocate/free)

The DiffusionServer coordinates the P2P flow; this class executes the
local side of each transfer step.
"""

import logging
import threading
from dataclasses import dataclass, field

import torch

from sglang.multimodal_gen.runtime.disaggregation.transport.rdma.transfer_buffer import (
    SlotHandle,
    TransferTensorBuffer,
)
from sglang.multimodal_gen.runtime.disaggregation.transport.rdma.transfer_engine import (
    BaseTransferEngine,
)

logger = logging.getLogger(__name__)


@dataclass
class StagedTransfer:
    """Tracks a staged transfer (data in local buffer, awaiting RDMA push)."""

    request_id: str
    slot: SlotHandle
    manifest: dict  # {field_name: [{offset, shape, dtype}, ...]}
    scalar_fields: dict = field(default_factory=dict)


@dataclass
class PendingReceive:
    """Tracks an allocated slot awaiting incoming data."""

    request_id: str
    slot: SlotHandle


class DiffusionTransferManager:
    """Manages P2P tensor transfers for a single role instance.

    Lifecycle:
        1. __init__: create buffer + engine, register buffer
        2. stage_tensors(): D2H from GPU to local buffer slot
        3. push_to_peer(): RDMA push to remote peer's buffer
        4. allocate_receive_slot(): prepare to receive data
        5. load_tensors(): H2D from local buffer slot to GPU
        6. free_slot(): release buffer slot
    """

    def __init__(
        self,
        engine: BaseTransferEngine,
        buffer: TransferTensorBuffer,
    ):
        self._engine = engine
        self._buffer = buffer
        self._lock = threading.Lock()

        # Register the pinned memory pool with the transfer engine
        self._engine.register_buffer(self._buffer.pool_data_ptr, self._buffer.pool_size)

        # Track staged outgoing transfers and pending incoming
        self._staged: dict[str, StagedTransfer] = {}
        self._pending_receives: dict[str, PendingReceive] = {}

        logger.info(
            "DiffusionTransferManager initialized: session=%s, pool=%d bytes",
            self._engine.session_id,
            self._buffer.pool_size,
        )

    @property
    def session_id(self) -> str:
        return self._engine.session_id

    @property
    def pool_data_ptr(self) -> int:
        return self._buffer.pool_data_ptr

    @property
    def pool_size(self) -> int:
        return self._buffer.pool_size

    def stage_tensors(
        self,
        request_id: str,
        tensor_fields: dict[str, torch.Tensor | list[torch.Tensor] | None],
        scalar_fields: dict | None = None,
        stream: torch.cuda.Stream | None = None,
    ) -> StagedTransfer | None:
        """Stage GPU tensors into the local TransferBuffer (D2H).

        Returns a StagedTransfer with the manifest, or None if allocation fails.
        """
        # Calculate required size
        total_size = 0
        for name, t in tensor_fields.items():
            if t is None:
                continue
            if isinstance(t, list):
                for ti in t:
                    total_size += ti.nelement() * ti.element_size()
            else:
                total_size += t.nelement() * t.element_size()

        if total_size == 0:
            # No tensor data to transfer
            manifest = {}
            slot = None
            staged = StagedTransfer(
                request_id=request_id,
                slot=slot,
                manifest=manifest,
                scalar_fields=scalar_fields or {},
            )
            with self._lock:
                self._staged[request_id] = staged
            return staged

        # Allocate slot in TransferBuffer
        slot = self._buffer.allocate(total_size, request_id)
        if slot is None:
            logger.warning(
                "TransferManager: failed to allocate %d bytes for %s",
                total_size,
                request_id,
            )
            return None

        # D2H: write tensors to pinned memory
        manifest = self._buffer.write_tensors_from_gpu(slot, tensor_fields, stream)

        # Synchronize to ensure D2H is complete
        if stream is not None:
            stream.synchronize()
        elif torch.cuda.is_available():
            torch.cuda.synchronize()

        staged = StagedTransfer(
            request_id=request_id,
            slot=slot,
            manifest=manifest,
            scalar_fields=scalar_fields or {},
        )
        with self._lock:
            self._staged[request_id] = staged

        logger.debug(
            "TransferManager: staged %s (%d bytes, offset=%d)",
            request_id,
            total_size,
            slot.offset,
        )
        return staged

    def stage_tensors_async(
        self,
        request_id: str,
        tensor_fields: dict[str, torch.Tensor | list[torch.Tensor] | None],
        scalar_fields: dict | None = None,
        stream: torch.cuda.Stream | None = None,
    ) -> tuple[StagedTransfer | None, torch.cuda.Event | None]:
        """Stage GPU tensors into TransferBuffer (D2H), returning a CUDA event
        instead of blocking on synchronize.

        Returns (StagedTransfer, d2h_done_event) or (None, None) on failure.
        The caller MUST wait on d2h_done_event before reading the buffer data.
        """
        # Calculate required size
        total_size = 0
        for name, t in tensor_fields.items():
            if t is None:
                continue
            if isinstance(t, list):
                for ti in t:
                    total_size += ti.nelement() * ti.element_size()
            else:
                total_size += t.nelement() * t.element_size()

        if total_size == 0:
            manifest = {}
            slot = None
            staged = StagedTransfer(
                request_id=request_id,
                slot=slot,
                manifest=manifest,
                scalar_fields=scalar_fields or {},
            )
            with self._lock:
                self._staged[request_id] = staged
            return staged, None

        # Allocate slot in TransferBuffer
        slot = self._buffer.allocate(total_size, request_id)
        if slot is None:
            logger.warning(
                "TransferManager: failed to allocate %d bytes for %s",
                total_size,
                request_id,
            )
            return None, None

        # D2H: write tensors to pinned memory
        manifest = self._buffer.write_tensors_from_gpu(slot, tensor_fields, stream)

        # Record CUDA event instead of blocking synchronize
        d2h_event = None
        if stream is not None:
            d2h_event = torch.cuda.Event()
            d2h_event.record(stream)
        elif torch.cuda.is_available():
            d2h_event = torch.cuda.Event()
            d2h_event.record(torch.cuda.current_stream())

        staged = StagedTransfer(
            request_id=request_id,
            slot=slot,
            manifest=manifest,
            scalar_fields=scalar_fields or {},
        )
        with self._lock:
            self._staged[request_id] = staged

        logger.debug(
            "TransferManager: staged_async %s (%d bytes, offset=%d)",
            request_id,
            total_size,
            slot.offset,
        )
        return staged, d2h_event

    def load_tensors_async(
        self,
        request_id: str,
        manifest: dict,
        device: torch.device | str = "cuda",
        stream: torch.cuda.Stream | None = None,
    ) -> tuple[dict[str, torch.Tensor | list[torch.Tensor]], torch.cuda.Event | None]:
        """Load tensors from receive slot to GPU (H2D), returning a CUDA event.

        The caller MUST ensure the default/compute stream waits on h2d_done_event
        before using the returned tensors.
        """
        with self._lock:
            pending = self._pending_receives.get(request_id)

        if pending is None:
            raise ValueError(
                f"TransferManager: no pending receive slot for {request_id}"
            )

        tensors = self._buffer.read_tensors_from_manifest(
            pending.slot, manifest, device=device, stream=stream
        )

        # Record CUDA event instead of blocking synchronize
        load_event = None
        if stream is not None:
            load_event = torch.cuda.Event()
            load_event.record(stream)
        elif torch.cuda.is_available():
            load_event = torch.cuda.Event()
            load_event.record(torch.cuda.current_stream())

        logger.debug(
            "TransferManager: loaded_async %d tensor fields for %s to %s",
            len(tensors),
            request_id,
            device,
        )
        return tensors, load_event

    def push_to_peer(
        self,
        request_id: str,
        dest_session_id: str,
        dest_addr: int,
        transfer_size: int,
    ) -> bool:
        """Push staged data to a remote peer's buffer via RDMA.

        Args:
            request_id: Must match a previously staged transfer.
            dest_session_id: Remote engine's session_id.
            dest_addr: Absolute address in remote pool.
            transfer_size: Number of bytes to transfer.

        Returns True on success.
        """
        with self._lock:
            staged = self._staged.get(request_id)

        if staged is None:
            logger.error("TransferManager: no staged transfer for %s", request_id)
            return False

        if staged.slot is None:
            # No tensor data - nothing to transfer
            return True

        src_addr = self._buffer.pool_data_ptr + staged.slot.offset
        ret = self._engine.transfer_sync(
            dest_session_id, src_addr, dest_addr, transfer_size
        )

        if ret == 0:
            logger.debug(
                "TransferManager: pushed %s (%d bytes) to %s",
                request_id,
                transfer_size,
                dest_session_id,
            )
        else:
            logger.error(
                "TransferManager: RDMA push failed for %s (ret=%d)",
                request_id,
                ret,
            )

        return ret == 0

    def free_staged(self, request_id: str) -> None:
        """Free the local buffer slot for a completed outgoing transfer."""
        with self._lock:
            staged = self._staged.pop(request_id, None)

        if staged and staged.slot is not None:
            self._buffer.free(staged.slot)
            logger.debug("TransferManager: freed staged slot for %s", request_id)

    def allocate_receive_slot(
        self, request_id: str, size: int
    ) -> PendingReceive | None:
        """Allocate a local buffer slot to receive incoming P2P data.

        Returns PendingReceive with slot info, or None if allocation fails.
        """
        slot = self._buffer.allocate(size, request_id)
        if slot is None:
            logger.warning(
                "TransferManager: failed to allocate receive slot (%d bytes) for %s",
                size,
                request_id,
            )
            return None

        pending = PendingReceive(request_id=request_id, slot=slot)
        with self._lock:
            self._pending_receives[request_id] = pending

        logger.debug(
            "TransferManager: allocated receive slot for %s (offset=%d, size=%d)",
            request_id,
            slot.offset,
            slot.size,
        )
        return pending

    def load_tensors(
        self,
        request_id: str,
        manifest: dict,
        device: torch.device | str = "cuda",
        stream: torch.cuda.Stream | None = None,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        """Load tensors from a receive slot into GPU memory (H2D).

        Args:
            request_id: Must match a previously allocated receive slot.
            manifest: Tensor layout descriptor from the sender.
            device: Target device for tensors.
            stream: Optional CUDA stream for async H2D.

        Returns dict of tensors on the target device.
        """
        with self._lock:
            pending = self._pending_receives.get(request_id)

        if pending is None:
            raise ValueError(
                f"TransferManager: no pending receive slot for {request_id}"
            )

        tensors = self._buffer.read_tensors_from_manifest(
            pending.slot, manifest, device=device, stream=stream
        )

        # Synchronize to ensure H2D is complete
        if stream is not None:
            stream.synchronize()
        elif torch.cuda.is_available():
            torch.cuda.synchronize()

        logger.debug(
            "TransferManager: loaded %d tensor fields for %s to %s",
            len(tensors),
            request_id,
            device,
        )
        return tensors

    def register_prealloc_as_receive(
        self, request_id: str, slot: "SlotHandle"
    ) -> "PendingReceive":
        """Register a pre-allocated slot as a pending receive.

        Phase 7e: Used when RDMA data was written directly to a pre-allocated
        slot (fast path), bypassing allocate_receive_slot().
        """
        pending = PendingReceive(request_id=request_id, slot=slot)
        with self._lock:
            self._pending_receives[request_id] = pending
        return pending

    def free_receive_slot(self, request_id: str) -> None:
        """Free a receive slot after processing."""
        with self._lock:
            pending = self._pending_receives.pop(request_id, None)

        if pending:
            self._buffer.free(pending.slot)
            logger.debug("TransferManager: freed receive slot for %s", request_id)

    def get_receive_slot_addr(self, request_id: str) -> int | None:
        """Get the absolute address of a receive slot (for RDMA destination)."""
        with self._lock:
            pending = self._pending_receives.get(request_id)
        if pending is None:
            return None
        return self._buffer.pool_data_ptr + pending.slot.offset

    def get_receive_slot_offset(self, request_id: str) -> int | None:
        """Get the offset of a receive slot within the pool."""
        with self._lock:
            pending = self._pending_receives.get(request_id)
        if pending is None:
            return None
        return pending.slot.offset

    def get_staged_info(self, request_id: str) -> StagedTransfer | None:
        """Get info about a staged outgoing transfer."""
        with self._lock:
            return self._staged.get(request_id)

    def free_slots_count(self, typical_size: int = 64 * 1024 * 1024) -> int:
        """Estimate available capacity (number of slots of typical_size)."""
        return self._buffer.free_slots_count(typical_size)

    def cleanup(self) -> None:
        """Clean up: deregister buffer, free all slots."""
        self._engine.deregister_buffer(self._buffer.pool_data_ptr)
        logger.info("DiffusionTransferManager cleaned up")
