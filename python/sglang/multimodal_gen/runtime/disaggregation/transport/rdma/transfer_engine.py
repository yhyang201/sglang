# SPDX-License-Identifier: Apache-2.0
"""
Transfer engine abstraction for P2P tensor transfer between role instances.

Provides a unified interface over MooncakeTransferEngine (RDMA) with a
mock fallback for unit testing. Each role instance creates one engine,
registers its TransferBuffer's pinned memory pool, and uses it to push
data directly to remote instances.

Usage:
    engine = create_transfer_engine(hostname="10.0.0.1", gpu_id=0)
    engine.register_buffer(pool_ptr, pool_size)
    engine.transfer_sync(dest_session_id, src_addr, dst_addr, length)
"""

import logging
import threading
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

# Sentinel for detecting whether mooncake is available
_MOONCAKE_AVAILABLE = None


def _check_mooncake() -> bool:
    global _MOONCAKE_AVAILABLE
    if _MOONCAKE_AVAILABLE is None:
        try:
            from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import (  # noqa: F401
                MooncakeTransferEngine as _MTE,
            )

            _MOONCAKE_AVAILABLE = True
        except ImportError:
            _MOONCAKE_AVAILABLE = False
    return _MOONCAKE_AVAILABLE


class BaseTransferEngine(ABC):
    """Abstract transfer engine for P2P data movement."""

    @property
    def supports_gpu_direct(self) -> bool:
        """Whether this engine supports GPUDirect RDMA (GPU memory addresses)."""
        return False

    @property
    @abstractmethod
    def session_id(self) -> str:
        """Unique session identifier for this engine instance."""

    @abstractmethod
    def register_buffer(self, ptr: int, length: int) -> None:
        """Register a memory region for RDMA access."""

    @abstractmethod
    def deregister_buffer(self, ptr: int) -> None:
        """Deregister a previously registered memory region."""

    @abstractmethod
    def transfer_sync(
        self, dst_session_id: str, src_addr: int, dst_addr: int, length: int
    ) -> int:
        """Synchronously transfer data from local src_addr to remote dst_addr.

        Returns 0 on success, negative on failure.
        """

    @abstractmethod
    def batch_transfer_sync(
        self,
        dst_session_id: str,
        src_addrs: list[int],
        dst_addrs: list[int],
        lengths: list[int],
    ) -> int:
        """Batch synchronous transfer. Returns 0 on success."""


class MooncakeDiffusionEngine(BaseTransferEngine):
    """Production transfer engine backed by MooncakeTransferEngine (RDMA/RPC)."""

    @property
    def supports_gpu_direct(self) -> bool:
        return True

    def __init__(
        self,
        hostname: str,
        gpu_id: int = 0,
        ib_device: str | None = None,
    ):
        from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import (
            MooncakeTransferEngine,
        )

        self._engine = MooncakeTransferEngine(
            hostname=hostname,
            gpu_id=gpu_id,
            ib_device=ib_device,
        )
        logger.info(
            "MooncakeDiffusionEngine initialized: session_id=%s",
            self._engine.session_id,
        )

    @property
    def session_id(self) -> str:
        return self._engine.session_id

    def register_buffer(self, ptr: int, length: int) -> None:
        self._engine.register(ptr, length)

    def deregister_buffer(self, ptr: int) -> None:
        self._engine.deregister(ptr)

    def transfer_sync(
        self, dst_session_id: str, src_addr: int, dst_addr: int, length: int
    ) -> int:
        return self._engine.transfer_sync(dst_session_id, src_addr, dst_addr, length)

    def batch_transfer_sync(
        self,
        dst_session_id: str,
        src_addrs: list[int],
        dst_addrs: list[int],
        lengths: list[int],
    ) -> int:
        return self._engine.batch_transfer_sync(
            dst_session_id, src_addrs, dst_addrs, lengths
        )


class MockTransferEngine(BaseTransferEngine):
    """In-process mock engine for unit testing P2P transfer logic.

    Simulates RDMA by copying data between registered memory regions
    within the same process (using ctypes memmove). Multiple instances
    share a class-level registry so they can "see" each other's buffers.
    """

    # Class-level registry: session_id -> {ptr -> (ctypes_buffer, length)}
    _registry: dict[str, dict[int, tuple[object, int]]] = {}
    _lock = threading.Lock()
    _counter = 0

    def __init__(self, session_id: str | None = None):
        with MockTransferEngine._lock:
            if session_id is None:
                MockTransferEngine._counter += 1
                session_id = f"mock-session-{MockTransferEngine._counter}"
            self._session_id = session_id
            MockTransferEngine._registry[session_id] = {}

    @property
    def session_id(self) -> str:
        return self._session_id

    def register_buffer(self, ptr: int, length: int) -> None:
        with MockTransferEngine._lock:
            self._registry[self._session_id][ptr] = (None, length)

    def deregister_buffer(self, ptr: int) -> None:
        with MockTransferEngine._lock:
            self._registry[self._session_id].pop(ptr, None)

    def transfer_sync(
        self, dst_session_id: str, src_addr: int, dst_addr: int, length: int
    ) -> int:
        """Simulate RDMA by direct memory copy (ctypes memmove)."""
        import ctypes

        try:
            ctypes.memmove(dst_addr, src_addr, length)
            return 0
        except Exception as e:
            logger.error("MockTransferEngine transfer failed: %s", e)
            return -1

    def batch_transfer_sync(
        self,
        dst_session_id: str,
        src_addrs: list[int],
        dst_addrs: list[int],
        lengths: list[int],
    ) -> int:
        for src, dst, length in zip(src_addrs, dst_addrs, lengths):
            ret = self.transfer_sync(dst_session_id, src, dst, length)
            if ret != 0:
                return ret
        return 0

    @classmethod
    def reset(cls):
        """Reset global registry (for test cleanup)."""
        with cls._lock:
            cls._registry.clear()
            cls._counter = 0


def create_transfer_engine(
    hostname: str = "127.0.0.1",
    gpu_id: int = 0,
    ib_device: str | None = None,
    force_mock: bool = False,
) -> BaseTransferEngine:
    """Factory: create the best available transfer engine.

    Uses MooncakeTransferEngine if available, falls back to MockTransferEngine.
    Set force_mock=True for testing.
    """
    if not force_mock and _check_mooncake():
        return MooncakeDiffusionEngine(
            hostname=hostname, gpu_id=gpu_id, ib_device=ib_device
        )
    logger.info(
        "Using MockTransferEngine (mooncake not available or force_mock=%s)",
        force_mock,
    )
    return MockTransferEngine()
