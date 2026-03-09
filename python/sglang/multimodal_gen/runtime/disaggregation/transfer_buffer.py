# SPDX-License-Identifier: Apache-2.0
"""
TransferBuffer: CPU-side pinned-memory staging area for disaggregated diffusion.

Design reference: RFC §2 — TransferBuffer (Base Class)

Components:
  - TransferMetaBuffer: Lightweight non-tensor request metadata store.
  - TransferTensorBuffer: Pinned-memory pool for heavy tensor payloads,
    managed by a BuddyAllocator for dynamic split/aggregate/coalesce.

Usage:
  1. Role computes output on GPU.
  2. D2H async copy into a TransferTensorBuffer slot (non-blocking).
  3. Metadata stored in TransferMetaBuffer.
  4. TransferManager reads from the slot for network transfer (Phase 7).
  5. On completion, slot is freed and coalesced.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

import torch

from sglang.multimodal_gen.runtime.disaggregation.transfer_allocator import (
    BuddyAllocator,
)

logger = logging.getLogger(__name__)


class TransferMetaBuffer:
    """Lightweight store for non-tensor request metadata.

    Thread-safe dict-based storage keyed by request_id.
    Stores scalar fields (prompt, config, timesteps, etc.) that are
    too small to warrant pinned-memory management.
    """

    def __init__(self):
        self._store: dict[str, dict] = {}
        self._lock = threading.Lock()

    def put(self, request_id: str, metadata: dict) -> None:
        """Store metadata for a request."""
        with self._lock:
            self._store[request_id] = metadata

    def get(self, request_id: str) -> dict | None:
        """Retrieve metadata for a request."""
        with self._lock:
            return self._store.get(request_id)

    def remove(self, request_id: str) -> dict | None:
        """Remove and return metadata for a request."""
        with self._lock:
            return self._store.pop(request_id, None)

    def count(self) -> int:
        with self._lock:
            return len(self._store)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()


@dataclass
class SlotHandle:
    """Handle returned by TransferTensorBuffer.allocate().

    Holds references needed to write tensors into the slot and later free it.
    """

    request_id: str
    offset: int  # byte offset in the pinned pool
    size: int  # allocated size in bytes
    # Tensor views into the pinned buffer (set by write_tensors)
    tensor_views: dict[str, torch.Tensor | list[torch.Tensor]] = field(
        default_factory=dict
    )


class TransferTensorBuffer:
    """Pinned-memory pool for staging tensor payloads between roles.

    Wraps a contiguous block of pinned CPU memory with a BuddyAllocator
    for dynamic slot management. Supports async D2H/H2D copies via
    CUDA streams.

    Args:
        pool_size: Total pool size in bytes.
        min_block_size: Minimum allocatable block in bytes (default 1 MiB).
        role_name: Name for logging (e.g., "encoder", "denoiser").
    """

    def __init__(
        self,
        pool_size: int,
        min_block_size: int = 1 << 20,
        role_name: str = "unknown",
    ):
        self._role_name = role_name
        self._allocator = BuddyAllocator(pool_size, min_block_size)
        actual_size = self._allocator.pool_size

        # Allocate contiguous pinned CPU memory
        self._pool = torch.empty(actual_size, dtype=torch.uint8, pin_memory=True)
        self._pool_ptr = self._pool.data_ptr()

        # Track active slots: offset -> SlotHandle
        self._slots: dict[int, SlotHandle] = {}
        self._lock = threading.Lock()

        logger.info(
            "TransferTensorBuffer[%s]: allocated %d MiB pinned memory "
            "(min_block=%d KiB)",
            role_name,
            actual_size >> 20,
            min_block_size >> 10,
        )

    @property
    def pool_size(self) -> int:
        return self._allocator.pool_size

    @property
    def pool_data_ptr(self) -> int:
        """Base address of the pinned pool. Used for RDMA registration (Phase 7)."""
        return self._pool_ptr

    def allocate(self, size: int, request_id: str) -> SlotHandle | None:
        """Allocate a slot for a request's tensor data.

        Args:
            size: Required size in bytes.
            request_id: Request ID for tracking.

        Returns:
            SlotHandle if successful, None if pool is full.
        """
        offset = self._allocator.allocate(size, request_id=request_id)
        if offset is None:
            logger.warning(
                "TransferTensorBuffer[%s]: allocation failed for %s (%d bytes). "
                "Pool stats: %s",
                self._role_name,
                request_id,
                size,
                self._allocator.get_stats(),
            )
            return None

        block = self._allocator.get_block_info(offset)
        handle = SlotHandle(
            request_id=request_id,
            offset=offset,
            size=block.size if block else size,
        )

        with self._lock:
            self._slots[offset] = handle

        return handle

    def free(self, handle: SlotHandle) -> bool:
        """Free a previously allocated slot.

        Args:
            handle: SlotHandle returned by allocate().

        Returns:
            True if freed successfully.
        """
        ok = self._allocator.free(handle.offset)
        if ok:
            with self._lock:
                self._slots.pop(handle.offset, None)
        return ok

    def get_slot_tensor(self, handle: SlotHandle) -> torch.Tensor:
        """Get a raw byte view into the pinned pool for this slot.

        The returned tensor shares memory with the pinned pool.
        Use for D2H/H2D async copies.
        """
        return self._pool[handle.offset : handle.offset + handle.size]

    def write_tensor(
        self,
        handle: SlotHandle,
        name: str,
        tensor: torch.Tensor,
        byte_offset: int = 0,
        stream: torch.cuda.Stream | None = None,
    ) -> int:
        """Copy a GPU tensor into the slot's pinned buffer (D2H).

        Args:
            handle: Target slot.
            name: Field name for this tensor.
            tensor: GPU tensor to copy.
            byte_offset: Offset within the slot to write at.
            stream: CUDA stream for async copy (None = default stream).

        Returns:
            Number of bytes written.
        """
        cpu_tensor = tensor.contiguous()
        nbytes = cpu_tensor.numel() * cpu_tensor.element_size()

        if byte_offset + nbytes > handle.size:
            raise ValueError(
                f"Write exceeds slot: offset={byte_offset}, nbytes={nbytes}, "
                f"slot_size={handle.size}"
            )

        # Create a typed view into the pinned pool at the correct offset
        dst = torch.frombuffer(
            self._pool[
                handle.offset + byte_offset : handle.offset + byte_offset + nbytes
            ].numpy(),
            dtype=cpu_tensor.dtype,
        ).reshape(cpu_tensor.shape)

        if stream is not None:
            with torch.cuda.stream(stream):
                dst.copy_(cpu_tensor, non_blocking=True)
        else:
            dst.copy_(cpu_tensor, non_blocking=True)

        return nbytes

    def read_tensor(
        self,
        handle: SlotHandle,
        shape: list[int],
        dtype: torch.dtype,
        byte_offset: int = 0,
        device: torch.device | str = "cpu",
        stream: torch.cuda.Stream | None = None,
    ) -> torch.Tensor:
        """Read a tensor from the slot's pinned buffer (H2D or CPU read).

        Args:
            handle: Source slot.
            shape: Tensor shape to read.
            dtype: Tensor dtype.
            byte_offset: Offset within the slot.
            device: Target device.
            stream: CUDA stream for async copy.

        Returns:
            Tensor on the target device.
        """
        nbytes = 1
        for s in shape:
            nbytes *= s
        nbytes *= torch.tensor([], dtype=dtype).element_size()

        src = torch.frombuffer(
            self._pool[
                handle.offset + byte_offset : handle.offset + byte_offset + nbytes
            ].numpy(),
            dtype=dtype,
        ).reshape(shape)

        if device == "cpu" or device == torch.device("cpu"):
            return src.clone()

        if stream is not None:
            with torch.cuda.stream(stream):
                return src.to(device, non_blocking=True)
        else:
            return src.to(device, non_blocking=True)

    def write_tensors_from_gpu(
        self,
        handle: SlotHandle,
        tensors: dict[str, torch.Tensor | list[torch.Tensor] | None],
        stream: torch.cuda.Stream | None = None,
    ) -> dict[str, list[dict]]:
        """Batch-write multiple GPU tensors into a slot (D2H).

        Writes tensors sequentially into the slot and returns a manifest
        describing the layout (offsets, shapes, dtypes) for later reads.

        Args:
            handle: Target slot.
            tensors: field_name -> Tensor | list[Tensor] | None.
            stream: CUDA stream for async copies.

        Returns:
            Manifest dict: field_name -> list of {offset, shape, dtype} entries.
        """
        manifest: dict[str, list[dict]] = {}
        byte_offset = 0

        for name, value in tensors.items():
            if value is None:
                continue

            entries = []
            if isinstance(value, torch.Tensor):
                nbytes = self.write_tensor(handle, name, value, byte_offset, stream)
                entries.append(
                    {
                        "offset": byte_offset,
                        "shape": list(value.shape),
                        "dtype": str(value.dtype).replace("torch.", ""),
                    }
                )
                byte_offset += nbytes
                # Align to 512 bytes for cache line efficiency
                byte_offset = (byte_offset + 511) & ~511

            elif isinstance(value, list):
                for i, t in enumerate(value):
                    if t is None:
                        continue
                    nbytes = self.write_tensor(
                        handle, f"{name}[{i}]", t, byte_offset, stream
                    )
                    entries.append(
                        {
                            "offset": byte_offset,
                            "shape": list(t.shape),
                            "dtype": str(t.dtype).replace("torch.", ""),
                            "list_index": i,
                        }
                    )
                    byte_offset += nbytes
                    byte_offset = (byte_offset + 511) & ~511

            manifest[name] = entries

        return manifest

    def read_tensors_from_manifest(
        self,
        handle: SlotHandle,
        manifest: dict[str, list[dict]],
        device: torch.device | str = "cpu",
        stream: torch.cuda.Stream | None = None,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        """Batch-read tensors from a slot using a manifest (H2D).

        Args:
            handle: Source slot.
            manifest: Layout descriptor from write_tensors_from_gpu().
            device: Target device.
            stream: CUDA stream for async copies.

        Returns:
            Reconstructed tensor fields.
        """
        from sglang.multimodal_gen.runtime.disaggregation.tensor_transport import (
            str_to_dtype,
        )

        result: dict[str, torch.Tensor | list[torch.Tensor]] = {}

        for name, entries in manifest.items():
            has_list_index = any("list_index" in e for e in entries)

            if has_list_index:
                max_idx = max(e.get("list_index", 0) for e in entries) + 1
                tensors = [None] * max_idx
                for entry in entries:
                    t = self.read_tensor(
                        handle,
                        entry["shape"],
                        str_to_dtype(entry["dtype"]),
                        entry["offset"],
                        device,
                        stream,
                    )
                    tensors[entry["list_index"]] = t
                result[name] = tensors
            else:
                entry = entries[0]
                result[name] = self.read_tensor(
                    handle,
                    entry["shape"],
                    str_to_dtype(entry["dtype"]),
                    entry["offset"],
                    device,
                    stream,
                )

        return result

    def free_slots_count(self, typical_request_size: int) -> int:
        """Estimate how many requests of typical size can still be buffered.

        Used to report FreeBufferSlots to DiffusionServer.
        """
        return self._allocator.count_free_slots(typical_request_size)

    def get_stats(self) -> dict:
        """Return buffer statistics."""
        alloc_stats = self._allocator.get_stats()
        with self._lock:
            alloc_stats["active_slots"] = len(self._slots)
            alloc_stats["role"] = self._role_name
        return alloc_stats
