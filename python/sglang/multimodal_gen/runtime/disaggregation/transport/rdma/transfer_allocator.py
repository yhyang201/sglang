# SPDX-License-Identifier: Apache-2.0
"""
Buddy-system memory allocator for TransferTensorBuffer.

Manages a contiguous pinned-memory region as power-of-2 blocks.
Supports split, allocate, free, and coalesce (defragmentation).

Design reference: RFC §2 — TransferTensorBufferAllocator
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Block:
    """A contiguous memory block within the pool."""

    offset: int  # byte offset from pool start
    size: int  # block size in bytes
    allocated: bool = False
    request_id: str | None = None  # which request owns this block


class BuddyAllocator:
    """Power-of-2 buddy-system allocator for pinned memory.

    The pool is divided into blocks of power-of-2 sizes.
    On allocation, the smallest sufficient block is found and split as needed.
    On free, the block is merged (coalesced) with its buddy if both are free.

    Args:
        pool_size: Total pool size in bytes. Rounded up to next power of 2.
        min_block_size: Minimum allocatable block size in bytes.
            Defaults to 1 MiB. Must be a power of 2.
    """

    def __init__(self, pool_size: int, min_block_size: int = 1 << 20):
        if min_block_size <= 0 or (min_block_size & (min_block_size - 1)) != 0:
            raise ValueError(
                f"min_block_size must be a power of 2, got {min_block_size}"
            )

        self._min_block_size = min_block_size
        self._pool_size = self._next_power_of_2(max(pool_size, min_block_size))
        self._lock = threading.Lock()

        # Free lists: order -> list of free block offsets
        # order 0 = min_block_size, order 1 = 2*min_block_size, ...
        self._max_order = self._size_to_order(self._pool_size)
        self._free_lists: list[list[int]] = [[] for _ in range(self._max_order + 1)]

        # Block map: offset -> Block
        self._blocks: dict[int, Block] = {}

        # Initialize: one big free block
        root = Block(offset=0, size=self._pool_size)
        self._blocks[0] = root
        self._free_lists[self._max_order].append(0)

        # Stats
        self._allocated_bytes = 0
        self._num_allocations = 0

    @property
    def pool_size(self) -> int:
        return self._pool_size

    @property
    def min_block_size(self) -> int:
        return self._min_block_size

    @property
    def allocated_bytes(self) -> int:
        with self._lock:
            return self._allocated_bytes

    @property
    def free_bytes(self) -> int:
        with self._lock:
            return self._pool_size - self._allocated_bytes

    @property
    def num_allocations(self) -> int:
        with self._lock:
            return self._num_allocations

    def allocate(self, size: int, request_id: str | None = None) -> int | None:
        """Allocate a block of at least `size` bytes.

        Args:
            size: Requested size in bytes.
            request_id: Optional request ID for tracking.

        Returns:
            Byte offset from pool start, or None if allocation fails.
        """
        if size <= 0:
            raise ValueError(f"Allocation size must be positive, got {size}")

        alloc_size = max(self._next_power_of_2(size), self._min_block_size)
        target_order = self._size_to_order(alloc_size)

        if target_order > self._max_order:
            logger.warning(
                "Requested size %d exceeds pool size %d", size, self._pool_size
            )
            return None

        with self._lock:
            return self._allocate_locked(target_order, request_id)

    def free(self, offset: int) -> bool:
        """Free the block at the given offset and coalesce with buddy if possible.

        Args:
            offset: Byte offset returned by allocate().

        Returns:
            True if freed successfully, False if offset not found or not allocated.
        """
        with self._lock:
            return self._free_locked(offset)

    def get_block_info(self, offset: int) -> Block | None:
        """Get block info at the given offset."""
        with self._lock:
            return self._blocks.get(offset)

    def get_stats(self) -> dict:
        """Return allocator statistics."""
        with self._lock:
            free_blocks_by_order = {}
            for order, offsets in enumerate(self._free_lists):
                if offsets:
                    block_size = self._min_block_size << order
                    free_blocks_by_order[block_size] = len(offsets)

            return {
                "pool_size": self._pool_size,
                "min_block_size": self._min_block_size,
                "allocated_bytes": self._allocated_bytes,
                "free_bytes": self._pool_size - self._allocated_bytes,
                "num_allocations": self._num_allocations,
                "num_blocks": len(self._blocks),
                "free_blocks_by_size": free_blocks_by_order,
            }

    def can_allocate(self, size: int) -> bool:
        """Check if a block of the given size can be allocated without doing it."""
        if size <= 0:
            return False
        alloc_size = max(self._next_power_of_2(size), self._min_block_size)
        target_order = self._size_to_order(alloc_size)
        if target_order > self._max_order:
            return False

        with self._lock:
            # Check if there's any free block at target_order or above
            for order in range(target_order, self._max_order + 1):
                if self._free_lists[order]:
                    return True
            return False

    def count_free_slots(self, slot_size: int) -> int:
        """Count how many allocations of the given size can fit.

        This is used to report FreeBufferSlots to DiffusionServer.
        """
        if slot_size <= 0:
            return 0
        alloc_size = max(self._next_power_of_2(slot_size), self._min_block_size)

        with self._lock:
            count = 0
            for order in range(self._size_to_order(alloc_size), self._max_order + 1):
                for _ in self._free_lists[order]:
                    # Each free block of this order can provide
                    # 2^(order - target_order) slots
                    block_size = self._min_block_size << order
                    count += block_size // alloc_size
            return count

    # --- Internal methods (must hold self._lock) ---

    def _allocate_locked(self, target_order: int, request_id: str | None) -> int | None:
        # Find smallest free block >= target_order
        found_order = -1
        for order in range(target_order, self._max_order + 1):
            if self._free_lists[order]:
                found_order = order
                break

        if found_order < 0:
            return None  # No space

        # Pop a free block at found_order
        offset = self._free_lists[found_order].pop(0)
        block = self._blocks[offset]

        # Split down to target_order
        while found_order > target_order:
            found_order -= 1
            buddy_size = self._min_block_size << found_order
            buddy_offset = offset + buddy_size

            # Create buddy block (free)
            buddy = Block(offset=buddy_offset, size=buddy_size)
            self._blocks[buddy_offset] = buddy
            self._free_lists[found_order].append(buddy_offset)

            # Shrink current block
            block.size = buddy_size

        # Mark allocated
        block.allocated = True
        block.request_id = request_id
        self._allocated_bytes += block.size
        self._num_allocations += 1

        return offset

    def _free_locked(self, offset: int) -> bool:
        block = self._blocks.get(offset)
        if block is None or not block.allocated:
            return False

        block.allocated = False
        block.request_id = None
        self._allocated_bytes -= block.size
        self._num_allocations -= 1

        # Coalesce with buddy
        self._coalesce(block)
        return True

    def _coalesce(self, block: Block) -> None:
        """Recursively merge with buddy if both are free."""
        while block.size < self._pool_size:
            buddy_offset = self._buddy_offset(block.offset, block.size)
            buddy = self._blocks.get(buddy_offset)

            if buddy is None or buddy.allocated or buddy.size != block.size:
                break

            # Both free and same size — merge
            order = self._size_to_order(buddy.size)
            self._free_lists[order].remove(buddy_offset)

            # Keep the lower offset block, remove the higher
            if buddy_offset < block.offset:
                # Buddy is lower — merge into buddy
                del self._blocks[block.offset]
                buddy.size *= 2
                block = buddy
            else:
                # Block is lower — merge into block
                del self._blocks[buddy_offset]
                block.size *= 2

        # Add merged block to free list
        order = self._size_to_order(block.size)
        self._free_lists[order].append(block.offset)

    def _buddy_offset(self, offset: int, size: int) -> int:
        """XOR the size bit to find the buddy's offset."""
        return offset ^ size

    def _size_to_order(self, size: int) -> int:
        """Convert block size to order (log2(size / min_block_size))."""
        order = 0
        s = self._min_block_size
        while s < size:
            s <<= 1
            order += 1
        return order

    @staticmethod
    def _next_power_of_2(n: int) -> int:
        if n <= 0:
            return 1
        n -= 1
        n |= n >> 1
        n |= n >> 2
        n |= n >> 4
        n |= n >> 8
        n |= n >> 16
        n |= n >> 32
        return n + 1
