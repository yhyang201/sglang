# SPDX-License-Identifier: Apache-2.0
"""
Central request router for disaggregated diffusion pipelines.

DiffusionServer is the global pipeline orchestrator with capacity-aware
dispatch. It manages independent pools of N encoders, M denoisers, and
K decoders, dispatching at every role transition only when the target
instance has free buffer slots:

  1. Client request → TTA[encoder] queue → when slot free → Encoder[i]
  2. Encoder[i] completes → slot freed → TTA[denoiser] queue → Denoiser[j]
  3. Denoiser[j] completes → slot freed → TTA[decoder] queue → Decoder[k]
  4. Decoder[k] completes → slot freed → result to Client

Admission control (RFC §3):
  - FreeBufferSlots: per-instance counter tracking available buffer capacity
  - TryToAdd (TTA) queues: per-role-type FIFO for requests awaiting slots
  - Completion callbacks: role results increment FreeBufferSlots, drain TTA

Socket topology:
  - Frontend: ROUTER (bind) — HTTP server DEALER connects here
  - Per role instance: PUSH (connect) → instance PULL (bind) for work dispatch
  - Per role type: PULL (bind) ← instance PUSH (connect) for result return
"""

import json
import logging
import pickle
import threading
import time
from collections import deque
from dataclasses import dataclass

import zmq

from sglang.multimodal_gen.runtime.disaggregation.dispatch_policy import (
    PoolDispatcher,
)
from sglang.multimodal_gen.runtime.disaggregation.request_state import (
    RequestState,
    RequestTracker,
)
from sglang.multimodal_gen.runtime.utils.common import get_zmq_socket

logger = logging.getLogger(__name__)


@dataclass
class _EncoderTTAEntry:
    """TTA queue entry for encoder dispatch."""

    request_id: str
    client_identity: bytes
    payload: bytes  # pickled request


@dataclass
class _RoleTTAEntry:
    """TTA queue entry for denoiser/decoder dispatch."""

    request_id: str
    frames: list  # ZMQ multipart frames to relay


class DiffusionServer:
    """Global pipeline orchestrator for N:M:K disaggregated diffusion.

    Manages independent encoder, denoiser, and decoder pools with
    capacity-aware dispatch. Each role instance has a FreeBufferSlots
    counter. Requests are queued in TTA when all instances are full,
    and dispatched when slots become available.
    """

    def __init__(
        self,
        frontend_endpoint: str,
        encoder_work_endpoints: list[str],
        denoiser_work_endpoints: list[str],
        decoder_work_endpoints: list[str],
        encoder_result_endpoint: str,
        denoiser_result_endpoint: str,
        decoder_result_endpoint: str,
        dispatch_policy_name: str = "round_robin",
        timeout_s: float = 600.0,
        encoder_capacity: int = 4,
        denoiser_capacity: int = 2,
        decoder_capacity: int = 4,
    ):
        """
        Args:
            frontend_endpoint: ROUTER socket endpoint (HTTP server connects here).
            encoder_work_endpoints: PULL endpoints for each encoder instance.
            denoiser_work_endpoints: PULL endpoints for each denoiser instance.
            decoder_work_endpoints: PULL endpoints for each decoder instance.
            encoder_result_endpoint: PULL endpoint for encoder results (DS binds).
            denoiser_result_endpoint: PULL endpoint for denoiser results (DS binds).
            decoder_result_endpoint: PULL endpoint for decoder results (DS binds).
            dispatch_policy_name: "round_robin" or "max_free_slots".
            timeout_s: Request timeout in seconds.
            encoder_capacity: Initial FreeBufferSlots per encoder instance.
            denoiser_capacity: Initial FreeBufferSlots per denoiser instance.
            decoder_capacity: Initial FreeBufferSlots per decoder instance.
        """
        self._frontend_endpoint = frontend_endpoint
        self._encoder_work_endpoints = encoder_work_endpoints
        self._denoiser_work_endpoints = denoiser_work_endpoints
        self._decoder_work_endpoints = decoder_work_endpoints
        self._encoder_result_endpoint = encoder_result_endpoint
        self._denoiser_result_endpoint = denoiser_result_endpoint
        self._decoder_result_endpoint = decoder_result_endpoint

        self._num_encoders = len(encoder_work_endpoints)
        self._num_denoisers = len(denoiser_work_endpoints)
        self._num_decoders = len(decoder_work_endpoints)
        self._timeout_s = timeout_s

        self._tracker = RequestTracker()
        self._dispatcher = PoolDispatcher(
            num_encoders=self._num_encoders,
            num_denoisers=self._num_denoisers,
            num_decoders=self._num_decoders,
            policy_name=dispatch_policy_name,
        )

        self._context = zmq.Context(io_threads=2)
        self._running = False
        self._thread: threading.Thread | None = None

        # Maps request_id -> client ZMQ identity (for final response)
        self._pending: dict[str, bytes] = {}
        self._lock = threading.Lock()

        # --- Capacity-aware dispatch (RFC §3) ---

        # FreeBufferSlots per instance
        self._encoder_free_slots = [encoder_capacity] * self._num_encoders
        self._denoiser_free_slots = [denoiser_capacity] * self._num_denoisers
        self._decoder_free_slots = [decoder_capacity] * self._num_decoders

        # TryToAdd (TTA) queues per role type
        self._encoder_tta: deque[_EncoderTTAEntry] = deque()
        self._denoiser_tta: deque[_RoleTTAEntry] = deque()
        self._decoder_tta: deque[_RoleTTAEntry] = deque()

    @property
    def tracker(self) -> RequestTracker:
        return self._tracker

    @property
    def dispatcher(self) -> PoolDispatcher:
        return self._dispatcher

    def start(self) -> None:
        """Start the orchestrator event loop in a background thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._event_loop,
            name="DiffusionServer",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "DiffusionServer started: frontend=%s, "
            "%d encoder(s), %d denoiser(s), %d decoder(s), policy=%s, "
            "capacity=(%d/%d/%d)",
            self._frontend_endpoint,
            self._num_encoders,
            self._num_denoisers,
            self._num_decoders,
            type(self._dispatcher.encoder_policy).__name__,
            self._encoder_free_slots[0] if self._encoder_free_slots else 0,
            self._denoiser_free_slots[0] if self._denoiser_free_slots else 0,
            self._decoder_free_slots[0] if self._decoder_free_slots else 0,
        )

    def stop(self) -> None:
        """Stop the orchestrator."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def _event_loop(self) -> None:
        """Main event loop: poll frontend and all role result sockets."""
        # Frontend ROUTER: receives from HTTP server
        frontend, _ = get_zmq_socket(
            self._context, zmq.ROUTER, self._frontend_endpoint, bind=True
        )

        # Per-instance PUSH sockets for sending work
        encoder_pushes: list[zmq.Socket] = []
        for i, ep in enumerate(self._encoder_work_endpoints):
            sock, _ = get_zmq_socket(self._context, zmq.PUSH, ep, bind=False)
            encoder_pushes.append(sock)

        denoiser_pushes: list[zmq.Socket] = []
        for i, ep in enumerate(self._denoiser_work_endpoints):
            sock, _ = get_zmq_socket(self._context, zmq.PUSH, ep, bind=False)
            denoiser_pushes.append(sock)

        decoder_pushes: list[zmq.Socket] = []
        for i, ep in enumerate(self._decoder_work_endpoints):
            sock, _ = get_zmq_socket(self._context, zmq.PUSH, ep, bind=False)
            decoder_pushes.append(sock)

        # Per-role-type PULL sockets for receiving results
        encoder_result_pull, _ = get_zmq_socket(
            self._context, zmq.PULL, self._encoder_result_endpoint, bind=True
        )
        denoiser_result_pull, _ = get_zmq_socket(
            self._context, zmq.PULL, self._denoiser_result_endpoint, bind=True
        )
        decoder_result_pull, _ = get_zmq_socket(
            self._context, zmq.PULL, self._decoder_result_endpoint, bind=True
        )

        poller = zmq.Poller()
        poller.register(frontend, zmq.POLLIN)
        poller.register(encoder_result_pull, zmq.POLLIN)
        poller.register(denoiser_result_pull, zmq.POLLIN)
        poller.register(decoder_result_pull, zmq.POLLIN)

        # Store push sockets for use by handlers
        self._encoder_pushes = encoder_pushes
        self._denoiser_pushes = denoiser_pushes
        self._decoder_pushes = decoder_pushes
        self._frontend = frontend

        all_sockets = (
            [frontend, encoder_result_pull, denoiser_result_pull, decoder_result_pull]
            + encoder_pushes
            + denoiser_pushes
            + decoder_pushes
        )

        try:
            while self._running:
                events = dict(poller.poll(timeout=100))

                self._handle_timeouts()

                if frontend in events:
                    self._handle_client_request(frontend)

                if encoder_result_pull in events:
                    self._handle_encoder_result(encoder_result_pull)

                if denoiser_result_pull in events:
                    self._handle_denoiser_result(denoiser_result_pull)

                if decoder_result_pull in events:
                    self._handle_decoder_result(decoder_result_pull)

        except Exception:
            logger.exception("DiffusionServer event loop error")
        finally:
            for sock in all_sockets:
                sock.close()
            self._context.destroy(linger=0)

    # --- Event handlers ---

    def _handle_client_request(self, frontend: zmq.Socket) -> None:
        """Receive client request, dispatch to encoder or enqueue in TTA."""
        try:
            parts = frontend.recv_multipart(zmq.NOBLOCK)
        except zmq.Again:
            return

        if len(parts) < 3:
            return

        client_identity = parts[0]
        payload = parts[-1]

        try:
            reqs = pickle.loads(payload)
        except (pickle.UnpicklingError, EOFError):
            logger.warning("DiffusionServer: failed to deserialize request")
            return

        if not isinstance(reqs, list):
            reqs = [reqs]

        req = reqs[0]
        request_id = getattr(req, "request_id", None)
        if request_id is None:
            request_id = f"ds-{time.monotonic()}"

        # Track request
        try:
            self._tracker.submit(request_id)
        except ValueError:
            logger.warning("DiffusionServer: duplicate request_id %s", request_id)
            return

        with self._lock:
            self._pending[request_id] = client_identity

        # Try to dispatch immediately
        encoder_idx = self._dispatcher.select_encoder_with_capacity(
            self._encoder_free_slots
        )

        if encoder_idx is not None:
            self._dispatch_to_encoder(request_id, payload, encoder_idx)
        else:
            # No encoder has free slots — enqueue in TTA
            try:
                self._tracker.transition(request_id, RequestState.ENCODER_WAITING)
            except ValueError:
                pass
            self._encoder_tta.append(
                _EncoderTTAEntry(
                    request_id=request_id,
                    client_identity=client_identity,
                    payload=payload,
                )
            )
            logger.debug(
                "DiffusionServer: %s queued in encoder TTA (depth=%d)",
                request_id,
                len(self._encoder_tta),
            )

    def _handle_encoder_result(self, result_pull: zmq.Socket) -> None:
        """Receive encoder output, free encoder slot, dispatch to denoiser or enqueue."""
        try:
            frames = result_pull.recv_multipart(zmq.NOBLOCK, copy=True)
        except zmq.Again:
            return

        request_id = self._extract_request_id(frames)
        if request_id is None:
            logger.warning("DiffusionServer: encoder result missing request_id")
            return

        # Free encoder slot
        record = self._tracker.get(request_id)
        if record and record.encoder_instance is not None:
            self._encoder_free_slots[record.encoder_instance] += 1

        # Check for error
        error = self._extract_error(frames)
        if error:
            self._complete_with_error(request_id, f"Encoder error: {error}")
            self._drain_encoder_tta()
            return

        # Transition state
        try:
            self._tracker.transition(request_id, RequestState.ENCODER_DONE)
        except ValueError:
            pass

        # Try to dispatch to denoiser
        denoiser_idx = self._dispatcher.select_denoiser_with_capacity(
            self._denoiser_free_slots
        )

        if denoiser_idx is not None:
            self._dispatch_to_denoiser(request_id, frames, denoiser_idx)
        else:
            try:
                self._tracker.transition(request_id, RequestState.DENOISING_WAITING)
            except ValueError:
                pass
            self._denoiser_tta.append(
                _RoleTTAEntry(request_id=request_id, frames=frames)
            )
            logger.debug(
                "DiffusionServer: %s queued in denoiser TTA (depth=%d)",
                request_id,
                len(self._denoiser_tta),
            )

        # Drain encoder TTA — a slot just freed
        self._drain_encoder_tta()

    def _handle_denoiser_result(self, result_pull: zmq.Socket) -> None:
        """Receive denoiser output, free denoiser slot, dispatch to decoder or enqueue."""
        try:
            frames = result_pull.recv_multipart(zmq.NOBLOCK, copy=True)
        except zmq.Again:
            return

        request_id = self._extract_request_id(frames)
        if request_id is None:
            logger.warning("DiffusionServer: denoiser result missing request_id")
            return

        # Free denoiser slot
        record = self._tracker.get(request_id)
        if record and record.denoiser_instance is not None:
            self._denoiser_free_slots[record.denoiser_instance] += 1

        error = self._extract_error(frames)
        if error:
            self._complete_with_error(request_id, f"Denoiser error: {error}")
            self._drain_denoiser_tta()
            return

        try:
            self._tracker.transition(request_id, RequestState.DENOISING_DONE)
        except ValueError:
            pass

        # Try to dispatch to decoder
        decoder_idx = self._dispatcher.select_decoder_with_capacity(
            self._decoder_free_slots
        )

        if decoder_idx is not None:
            self._dispatch_to_decoder(request_id, frames, decoder_idx)
        else:
            try:
                self._tracker.transition(request_id, RequestState.DECODER_WAITING)
            except ValueError:
                pass
            self._decoder_tta.append(
                _RoleTTAEntry(request_id=request_id, frames=frames)
            )
            logger.debug(
                "DiffusionServer: %s queued in decoder TTA (depth=%d)",
                request_id,
                len(self._decoder_tta),
            )

        # Drain denoiser TTA — a slot just freed
        self._drain_denoiser_tta()

    def _handle_decoder_result(self, result_pull: zmq.Socket) -> None:
        """Receive decoder output, free decoder slot, return result to client."""
        from sglang.multimodal_gen.runtime.disaggregation.tensor_transport import (
            unpack_tensors,
        )
        from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import (
            OutputBatch,
        )

        try:
            frames = result_pull.recv_multipart(zmq.NOBLOCK, copy=True)
        except zmq.Again:
            return

        request_id = self._extract_request_id(frames)
        if request_id is None:
            logger.warning("DiffusionServer: decoder result missing request_id")
            return

        # Free decoder slot
        record = self._tracker.get(request_id)
        if record and record.decoder_instance is not None:
            self._decoder_free_slots[record.decoder_instance] += 1

        # Unpack decoder output
        tensor_fields, scalar_fields = unpack_tensors(frames, device="cpu")

        output_batch = OutputBatch(
            output=tensor_fields.get("output"),
            audio=tensor_fields.get("audio"),
            audio_sample_rate=scalar_fields.get("audio_sample_rate"),
            error=scalar_fields.get("error"),
        )

        # Transition to terminal state
        try:
            if output_batch.error:
                self._tracker.transition(
                    request_id, RequestState.FAILED, error=output_batch.error
                )
            else:
                self._tracker.transition(request_id, RequestState.DONE)
        except ValueError:
            pass

        # Send result to HTTP client
        with self._lock:
            client_identity = self._pending.pop(request_id, None)

        if client_identity is None:
            logger.warning(
                "DiffusionServer: no pending client for decoder result %s",
                request_id,
            )
            self._tracker.remove(request_id)
            self._drain_decoder_tta()
            return

        try:
            self._frontend.send_multipart(
                [client_identity, b"", pickle.dumps(output_batch)]
            )
        except zmq.ZMQError as e:
            logger.error(
                "DiffusionServer: failed to send result for %s: %s",
                request_id,
                e,
            )

        logger.debug("DiffusionServer: returned result for %s", request_id)
        self._tracker.remove(request_id)

        # Drain decoder TTA — a slot just freed
        self._drain_decoder_tta()

    # --- Dispatch helpers ---

    def _dispatch_to_encoder(
        self, request_id: str, payload: bytes, encoder_idx: int
    ) -> None:
        """Dispatch request to encoder instance, decrement FreeBufferSlots."""
        self._encoder_free_slots[encoder_idx] -= 1

        try:
            self._tracker.transition(
                request_id,
                RequestState.ENCODER_RUNNING,
                encoder_instance=encoder_idx,
            )
        except ValueError:
            pass

        self._encoder_pushes[encoder_idx].send_multipart(
            [request_id.encode("utf-8"), payload]
        )
        logger.debug(
            "DiffusionServer: dispatched %s to encoder[%d] (free=%d)",
            request_id,
            encoder_idx,
            self._encoder_free_slots[encoder_idx],
        )

    def _dispatch_to_denoiser(
        self, request_id: str, frames: list, denoiser_idx: int
    ) -> None:
        """Dispatch encoder output to denoiser instance."""
        self._denoiser_free_slots[denoiser_idx] -= 1

        try:
            self._tracker.transition(
                request_id,
                RequestState.DENOISING_RUNNING,
                denoiser_instance=denoiser_idx,
            )
        except ValueError:
            pass

        self._denoiser_pushes[denoiser_idx].send_multipart(frames)
        logger.debug(
            "DiffusionServer: relayed %s to denoiser[%d] (free=%d)",
            request_id,
            denoiser_idx,
            self._denoiser_free_slots[denoiser_idx],
        )

    def _dispatch_to_decoder(
        self, request_id: str, frames: list, decoder_idx: int
    ) -> None:
        """Dispatch denoiser output to decoder instance."""
        self._decoder_free_slots[decoder_idx] -= 1

        try:
            self._tracker.transition(
                request_id,
                RequestState.DECODER_RUNNING,
                decoder_instance=decoder_idx,
            )
        except ValueError:
            pass

        self._decoder_pushes[decoder_idx].send_multipart(frames)
        logger.debug(
            "DiffusionServer: relayed %s to decoder[%d] (free=%d)",
            request_id,
            decoder_idx,
            self._decoder_free_slots[decoder_idx],
        )

    # --- TTA queue draining ---

    def _drain_encoder_tta(self) -> None:
        """Try to dispatch queued encoder requests when slots become available."""
        while self._encoder_tta:
            idx = self._dispatcher.select_encoder_with_capacity(
                self._encoder_free_slots
            )
            if idx is None:
                break
            entry = self._encoder_tta.popleft()
            self._dispatch_to_encoder(entry.request_id, entry.payload, idx)

    def _drain_denoiser_tta(self) -> None:
        """Try to dispatch queued denoiser requests when slots become available."""
        while self._denoiser_tta:
            idx = self._dispatcher.select_denoiser_with_capacity(
                self._denoiser_free_slots
            )
            if idx is None:
                break
            entry = self._denoiser_tta.popleft()
            self._dispatch_to_denoiser(entry.request_id, entry.frames, idx)

    def _drain_decoder_tta(self) -> None:
        """Try to dispatch queued decoder requests when slots become available."""
        while self._decoder_tta:
            idx = self._dispatcher.select_decoder_with_capacity(
                self._decoder_free_slots
            )
            if idx is None:
                break
            entry = self._decoder_tta.popleft()
            self._dispatch_to_decoder(entry.request_id, entry.frames, idx)

    # --- Helpers ---

    def _extract_request_id(self, frames: list) -> str | None:
        """Extract request_id from the JSON metadata frame (frame 0)."""
        try:
            metadata = json.loads(frames[0])
            return metadata.get("scalar_fields", {}).get("request_id")
        except (json.JSONDecodeError, IndexError, TypeError):
            return None

    def _extract_error(self, frames: list) -> str | None:
        """Extract error field from the JSON metadata frame."""
        try:
            metadata = json.loads(frames[0])
            return metadata.get("scalar_fields", {}).get("_disagg_error")
        except (json.JSONDecodeError, IndexError, TypeError):
            return None

    def _complete_with_error(self, request_id: str, error_msg: str) -> None:
        """Send error response to the HTTP client for a failed request."""
        from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import (
            OutputBatch,
        )

        logger.error("DiffusionServer: %s — %s", request_id, error_msg)

        try:
            self._tracker.transition(request_id, RequestState.FAILED, error=error_msg)
        except ValueError:
            pass

        with self._lock:
            client_identity = self._pending.pop(request_id, None)

        if client_identity is None:
            self._tracker.remove(request_id)
            return

        error_batch = OutputBatch(error=error_msg)
        try:
            self._frontend.send_multipart(
                [client_identity, b"", pickle.dumps(error_batch)]
            )
        except zmq.ZMQError as e:
            logger.error(
                "DiffusionServer: failed to send error for %s: %s",
                request_id,
                e,
            )

        self._tracker.remove(request_id)

    def _handle_timeouts(self) -> None:
        """Check for and handle timed-out requests."""
        timed_out = self._tracker.find_timed_out(self._timeout_s)
        for request_id in timed_out:
            # Free the slot for the timed-out request
            record = self._tracker.get(request_id)
            if record:
                self._free_slot_for_record(record)

            self._complete_with_error(
                request_id,
                f"DiffusionServer timeout: request {request_id} "
                f"not completed within {self._timeout_s}s",
            )

        # Also remove timed-out entries from TTA queues
        if timed_out:
            timed_set = set(timed_out)
            self._encoder_tta = deque(
                e for e in self._encoder_tta if e.request_id not in timed_set
            )
            self._denoiser_tta = deque(
                e for e in self._denoiser_tta if e.request_id not in timed_set
            )
            self._decoder_tta = deque(
                e for e in self._decoder_tta if e.request_id not in timed_set
            )

    def _free_slot_for_record(self, record) -> None:
        """Increment FreeBufferSlots for the role instance this request occupied."""
        if (
            record.state in (RequestState.ENCODER_RUNNING, RequestState.ENCODER_DONE)
            and record.encoder_instance is not None
        ):
            self._encoder_free_slots[record.encoder_instance] += 1
        if (
            record.state
            in (RequestState.DENOISING_RUNNING, RequestState.DENOISING_DONE)
            and record.denoiser_instance is not None
        ):
            self._denoiser_free_slots[record.denoiser_instance] += 1
        if (
            record.state == RequestState.DECODER_RUNNING
            and record.decoder_instance is not None
        ):
            self._decoder_free_slots[record.decoder_instance] += 1

    def get_stats(self) -> dict:
        """Return router-level statistics for observability."""
        with self._lock:
            pending_count = len(self._pending)
        return {
            "role": "diffusion_server",
            "num_encoders": self._num_encoders,
            "num_denoisers": self._num_denoisers,
            "num_decoders": self._num_decoders,
            "pending_requests": pending_count,
            "dispatch_policy": type(self._dispatcher.encoder_policy).__name__,
            "encoder_free_slots": list(self._encoder_free_slots),
            "denoiser_free_slots": list(self._denoiser_free_slots),
            "decoder_free_slots": list(self._decoder_free_slots),
            "encoder_tta_depth": len(self._encoder_tta),
            "denoiser_tta_depth": len(self._denoiser_tta),
            "decoder_tta_depth": len(self._decoder_tta),
            "tracker": self._tracker.snapshot(),
        }
