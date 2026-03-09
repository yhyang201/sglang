# SPDX-License-Identifier: Apache-2.0
"""
Request state machine for disaggregated diffusion pipelines.

Tracks the lifecycle of each request as it flows through
Encoder -> Denoiser -> Decoder roles in a multi-instance deployment.

Usage:
    tracker = RequestTracker()
    tracker.submit("req-001")
    tracker.transition("req-001", RequestState.ENCODER_RUNNING)
    ...
    tracker.transition("req-001", RequestState.DONE)
"""

import enum
import logging
import threading
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class RequestState(enum.Enum):
    """Lifecycle states for a disagg pipeline request.

    *_WAITING states: request is in the TryToAdd (TTA) queue, awaiting a
    free buffer slot on some instance.
    *_RUNNING states: request has been dispatched to a specific instance.
    """

    PENDING = "pending"
    # Encoder
    ENCODER_WAITING = "encoder_waiting"
    ENCODER_RUNNING = "encoder_running"
    ENCODER_DONE = "encoder_done"
    # Denoiser
    DENOISING_WAITING = "denoising_waiting"
    DENOISING_RUNNING = "denoising_running"
    DENOISING_DONE = "denoising_done"
    # Decoder
    DECODER_WAITING = "decoder_waiting"
    DECODER_RUNNING = "decoder_running"
    # Terminal
    DONE = "done"
    FAILED = "failed"
    TIMED_OUT = "timed_out"


# Valid state transitions
_VALID_TRANSITIONS: dict[RequestState, set[RequestState]] = {
    RequestState.PENDING: {
        RequestState.ENCODER_WAITING,
        RequestState.ENCODER_RUNNING,
        RequestState.FAILED,
    },
    RequestState.ENCODER_WAITING: {
        RequestState.ENCODER_RUNNING,
        RequestState.FAILED,
    },
    RequestState.ENCODER_RUNNING: {
        RequestState.ENCODER_DONE,
        RequestState.FAILED,
    },
    RequestState.ENCODER_DONE: {
        RequestState.DENOISING_WAITING,
        RequestState.DENOISING_RUNNING,
        RequestState.FAILED,
    },
    RequestState.DENOISING_WAITING: {
        RequestState.DENOISING_RUNNING,
        RequestState.FAILED,
    },
    RequestState.DENOISING_RUNNING: {
        RequestState.DENOISING_DONE,
        RequestState.FAILED,
    },
    RequestState.DENOISING_DONE: {
        RequestState.DECODER_WAITING,
        RequestState.DECODER_RUNNING,
        RequestState.FAILED,
    },
    RequestState.DECODER_WAITING: {
        RequestState.DECODER_RUNNING,
        RequestState.FAILED,
    },
    RequestState.DECODER_RUNNING: {
        RequestState.DONE,
        RequestState.FAILED,
    },
    # Terminal states — can transition to TIMED_OUT from any non-terminal
    RequestState.DONE: set(),
    RequestState.FAILED: set(),
    RequestState.TIMED_OUT: set(),
}

# Non-terminal states that can time out
_ACTIVE_STATES = {
    RequestState.PENDING,
    RequestState.ENCODER_WAITING,
    RequestState.ENCODER_RUNNING,
    RequestState.ENCODER_DONE,
    RequestState.DENOISING_WAITING,
    RequestState.DENOISING_RUNNING,
    RequestState.DENOISING_DONE,
    RequestState.DECODER_WAITING,
    RequestState.DECODER_RUNNING,
}


@dataclass
class RequestRecord:
    """Tracking record for a single request."""

    request_id: str
    state: RequestState = RequestState.PENDING
    submit_time: float = field(default_factory=time.monotonic)
    last_transition_time: float = field(default_factory=time.monotonic)
    encoder_instance: int | None = None
    denoiser_instance: int | None = None
    decoder_instance: int | None = None
    error: str | None = None

    def elapsed_s(self) -> float:
        return time.monotonic() - self.submit_time

    def is_terminal(self) -> bool:
        return self.state in (
            RequestState.DONE,
            RequestState.FAILED,
            RequestState.TIMED_OUT,
        )


class RequestTracker:
    """Thread-safe tracker for request state machines.

    Maintains the state of all in-flight requests and provides
    query methods for dispatch policies (e.g., count by state,
    find timed-out requests).
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._requests: dict[str, RequestRecord] = {}

    def submit(self, request_id: str) -> RequestRecord:
        """Register a new request in PENDING state."""
        with self._lock:
            if request_id in self._requests:
                raise ValueError(f"Duplicate request_id: {request_id}")
            record = RequestRecord(request_id=request_id)
            self._requests[request_id] = record
            return record

    def transition(
        self,
        request_id: str,
        new_state: RequestState,
        *,
        error: str | None = None,
        encoder_instance: int | None = None,
        denoiser_instance: int | None = None,
        decoder_instance: int | None = None,
    ) -> RequestRecord:
        """Transition a request to a new state.

        Raises ValueError for invalid transitions or unknown request_id.
        """
        with self._lock:
            record = self._requests.get(request_id)
            if record is None:
                raise ValueError(f"Unknown request_id: {request_id}")

            old_state = record.state

            # Allow timeout from any active state
            if new_state == RequestState.TIMED_OUT:
                if old_state not in _ACTIVE_STATES:
                    raise ValueError(
                        f"Cannot time out request {request_id} in state {old_state.value}"
                    )
            elif new_state not in _VALID_TRANSITIONS.get(old_state, set()):
                raise ValueError(
                    f"Invalid transition for {request_id}: "
                    f"{old_state.value} -> {new_state.value}"
                )

            record.state = new_state
            record.last_transition_time = time.monotonic()
            if error is not None:
                record.error = error
            if encoder_instance is not None:
                record.encoder_instance = encoder_instance
            if denoiser_instance is not None:
                record.denoiser_instance = denoiser_instance
            if decoder_instance is not None:
                record.decoder_instance = decoder_instance

            logger.debug(
                "Request %s: %s -> %s", request_id, old_state.value, new_state.value
            )
            return record

    def get(self, request_id: str) -> RequestRecord | None:
        """Get a request record by ID."""
        with self._lock:
            return self._requests.get(request_id)

    def remove(self, request_id: str) -> RequestRecord | None:
        """Remove and return a request record (for completed/failed requests)."""
        with self._lock:
            return self._requests.pop(request_id, None)

    def count_by_state(self, state: RequestState) -> int:
        """Count requests in a given state."""
        with self._lock:
            return sum(1 for r in self._requests.values() if r.state == state)

    def count_active(self) -> int:
        """Count all non-terminal requests."""
        with self._lock:
            return sum(1 for r in self._requests.values() if not r.is_terminal())

    def find_timed_out(self, timeout_s: float) -> list[str]:
        """Find active requests that have exceeded the timeout."""
        now = time.monotonic()
        with self._lock:
            return [
                r.request_id
                for r in self._requests.values()
                if r.state in _ACTIVE_STATES and (now - r.submit_time) > timeout_s
            ]

    def count_at_instance(self, role: str, instance_id: int) -> int:
        """Count active requests assigned to a specific role instance.

        Args:
            role: "encoder", "denoiser", or "decoder"
            instance_id: The instance index
        """
        attr = f"{role}_instance"
        with self._lock:
            return sum(
                1
                for r in self._requests.values()
                if not r.is_terminal() and getattr(r, attr, None) == instance_id
            )

    def snapshot(self) -> dict:
        """Return a summary snapshot for observability."""
        with self._lock:
            state_counts = {}
            for r in self._requests.values():
                state_counts[r.state.value] = state_counts.get(r.state.value, 0) + 1
            return {
                "total": len(self._requests),
                "active": sum(
                    1 for r in self._requests.values() if not r.is_terminal()
                ),
                "by_state": state_counts,
            }
