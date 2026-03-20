# SPDX-License-Identifier: Apache-2.0
"""
Central request router for disaggregated diffusion pipelines.

DiffusionServer is the global pipeline orchestrator with capacity-aware
dispatch. It manages independent pools of N encoders, M denoisers, and
K decoders, dispatching at every role transition only when the target
instance has free buffer slots.

Tensor transport is via RDMA/TransferEngine. DiffusionServer routes
only metadata and transfer control messages. The only exception is the decoder
result path: decoders send final output as regular ZMQ frames.

Transfer message flow (encoder → denoiser):
  1. Encoder → DS: transfer_staged (data in local TransferBuffer)
  2. DS → Denoiser: transfer_alloc (allocate receive buffer)
  3. Denoiser → DS: transfer_allocated (slot ready)
  4. DS → Encoder: transfer_push (RDMA push to denoiser's buffer)
  5. Encoder → DS: transfer_pushed (transfer complete)
  6. DS → Denoiser: transfer_ready (data arrived, process it)
  7. Denoiser → DS: transfer_done (compute finished)

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
from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.disaggregation.transport.tensor_codec import (
    unpack_tensors,
)
from sglang.multimodal_gen.runtime.disaggregation.transport.transfer_protocol import (
    TransferAllocMsg,
    TransferMsgType,
    TransferPushMsg,
    TransferReadyMsg,
    decode_transfer_msg,
    encode_transfer_msg,
    is_transfer_message,
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
class _TransferRequestState:
    """Per-request transfer state tracked by DiffusionServer."""

    # Sender (encoder/denoiser) info
    sender_session_id: str = ""
    sender_pool_ptr: int = 0
    sender_slot_offset: int = 0
    data_size: int = 0
    manifest: dict = None
    scalar_fields: dict = None

    # Receiver (denoiser/decoder) info (set after allocation)
    receiver_session_id: str = ""
    receiver_pool_ptr: int = 0
    receiver_slot_offset: int = 0

    # Which instance indices are involved
    sender_instance: int = -1
    receiver_instance: int = -1

    # Phase 7: pre-allocated slot id (if used), for recycling after compute
    prealloc_slot_id: int | None = None

    def __post_init__(self):
        if self.manifest is None:
            self.manifest = {}
        if self.scalar_fields is None:
            self.scalar_fields = {}


@dataclass
class _RoleTTAEntry:
    """TTA queue entry for denoiser/decoder dispatch (transfer mode)."""

    request_id: str
    transfer_state: _TransferRequestState | None = None  # transfer state


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
        p2p_mode: bool = True,
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
            p2p_mode: Kept for API compatibility (always True).
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
        # TODO: derive capacity dynamically from actual GPU memory / TransferBuffer
        # pool size instead of using hardcoded defaults.
        self._encoder_free_slots = [encoder_capacity] * self._num_encoders
        self._denoiser_free_slots = [denoiser_capacity] * self._num_denoisers
        self._decoder_free_slots = [decoder_capacity] * self._num_decoders

        # TryToAdd (TTA) queues per role type
        self._encoder_tta: deque[_EncoderTTAEntry] = deque()
        self._denoiser_tta: deque[_RoleTTAEntry] = deque()
        self._decoder_tta: deque[_RoleTTAEntry] = deque()

        # --- Transfer mode (RFC §4) ---
        self._transfer_mode = p2p_mode

        # Per-request transfer state
        self._transfer_state: dict[str, _TransferRequestState] = {}

        # Per-instance registration: instance_idx -> {session_id, pool_ptr, pool_size}
        self._encoder_peers: dict[int, dict] = {}
        self._denoiser_peers: dict[int, dict] = {}
        self._decoder_peers: dict[int, dict] = {}

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
                events = dict(poller.poll(timeout=10))

                self._handle_timeouts()

                if frontend in events:
                    self._handle_client_request(frontend)

                if encoder_result_pull in events:
                    self._handle_role_result(encoder_result_pull, RoleType.ENCODER)

                if denoiser_result_pull in events:
                    self._handle_role_result(denoiser_result_pull, RoleType.DENOISER)

                if decoder_result_pull in events:
                    self._handle_role_result(decoder_result_pull, RoleType.DECODER)

                # Drain all queues after processing events
                self._drain_all_queues()

        except Exception:
            logger.exception("DiffusionServer event loop error")
        finally:
            for sock in all_sockets:
                sock.close()
            self._context.destroy(linger=0)

    # --- Event handlers ---

    def _handle_role_result(self, result_pull: zmq.Socket, role: RoleType) -> None:
        """Dispatch result message to transfer handler or decoder result handler.

        Encoder and denoiser results are always transfer control messages.
        Decoder results come as regular ZMQ frames (the final output).
        """

        try:
            frames = result_pull.recv_multipart(zmq.NOBLOCK, copy=True)
        except zmq.Again:
            return

        if is_transfer_message(frames):
            self._handle_transfer_result(frames, role)
            return

        # Only decoder sends non-transfer frames (final output back to client)
        if role == RoleType.DECODER:
            self._handle_decoder_result_frames(frames)
        else:
            logger.warning(
                "DiffusionServer: unexpected non-transfer frames from %s", role.value
            )

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

        # Filter out non-Req messages (e.g., ping dicts, stats requests)
        if isinstance(req, dict) or not hasattr(req, "request_id"):
            logger.debug(
                "DiffusionServer: ignoring non-Req message of type %s",
                type(req).__name__,
            )
            # Send empty reply so REQ socket doesn't hang
            try:
                frontend.send_multipart(
                    [client_identity, b"", pickle.dumps({"status": "ignored"})],
                    zmq.NOBLOCK,
                )
            except zmq.Again:
                pass
            return

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

        # Enqueue, then drain
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

    def _handle_decoder_result_frames(self, frames: list) -> None:
        """Handle decoder result frames (relay mode)."""
        from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import (
            OutputBatch,
        )

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
        # Clean up transfer state if applicable (transfer decoder results also come
        # through this path after the optimization to send raw frames)
        self._transfer_state.pop(request_id, None)
        self._tracker.remove(request_id)

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

    # --- TTA queue draining ---

    def _drain_all_queues(self) -> None:
        """Drain all TTA queues. Called once per event loop iteration."""
        self._drain_encoder_tta()
        self._drain_denoiser_tta()
        self._drain_decoder_tta()

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
            self._transfer_dispatch_to_denoiser(
                entry.request_id, entry.transfer_state, idx
            )

    def _drain_decoder_tta(self) -> None:
        """Try to dispatch queued decoder requests when slots become available."""
        while self._decoder_tta:
            idx = self._dispatcher.select_decoder_with_capacity(
                self._decoder_free_slots
            )
            if idx is None:
                break
            entry = self._decoder_tta.popleft()
            self._transfer_dispatch_to_decoder(
                entry.request_id, entry.transfer_state, idx
            )

    # --- Helpers ---

    def _extract_request_id(self, frames: list) -> str | None:
        """Extract request_id from the JSON metadata frame (frame 0)."""
        try:
            metadata = json.loads(frames[0])
            return metadata.get("scalar_fields", {}).get("request_id")
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

    # --- Transfer message handlers ---

    def _handle_transfer_result(self, frames: list, role: RoleType) -> None:
        """Handle a transfer control message from a role instance."""

        try:
            msg = decode_transfer_msg(frames)
        except (ValueError, Exception) as e:
            logger.error("DiffusionServer: failed to decode transfer message: %s", e)
            return

        msg_type = msg.get("msg_type")

        if msg_type == TransferMsgType.REGISTER:
            self._handle_transfer_register(msg)
        elif msg_type == TransferMsgType.STAGED:
            self._handle_transfer_staged(msg)
        elif msg_type == TransferMsgType.ALLOCATED:
            self._handle_transfer_allocated(msg)
        elif msg_type == TransferMsgType.PUSHED:
            self._handle_transfer_pushed(msg)
        elif msg_type == TransferMsgType.DONE:
            self._handle_transfer_done(msg, role)
        else:
            logger.warning("DiffusionServer: unknown transfer msg_type=%s", msg_type)

    def _handle_transfer_register(self, msg: dict) -> None:
        """Instance registers its TransferEngine session + buffer info.

        Phase 7a: Also stores pre-allocated receive slots from receivers.
        """
        try:
            role = RoleType.from_string(msg.get("role", ""))
        except ValueError:
            logger.warning(
                "DiffusionServer transfer: unknown role in register: %s",
                msg.get("role"),
            )
            return
        info = {
            "session_id": msg.get("session_id", ""),
            "pool_ptr": msg.get("pool_ptr", 0),
            "pool_size": msg.get("pool_size", 0),
        }
        # Phase 7: store pre-allocated slots as a free list per instance
        prealloc = msg.get("preallocated_slots", [])
        info["free_preallocated_slots"] = list(prealloc) if prealloc else []

        # Assign instance index by registration order (server-managed, not self-reported)
        if role == RoleType.ENCODER:
            idx = len(self._encoder_peers)
            self._encoder_peers[idx] = info
        elif role == RoleType.DENOISER:
            idx = len(self._denoiser_peers)
            self._denoiser_peers[idx] = info
        elif role == RoleType.DECODER:
            idx = len(self._decoder_peers)
            self._decoder_peers[idx] = info
        else:
            idx = 0
        logger.info(
            "DiffusionServer transfer: registered %s[%d] session=%s pool_ptr=%#x prealloc=%d",
            role,
            idx,
            info["session_id"],
            info["pool_ptr"],
            len(prealloc),
        )

    def _handle_transfer_staged(self, msg: dict) -> None:
        """Handle TRANSFER_STAGED: upstream role finished, data staged in local TransferBuffer.

        Currently only encoder sends STAGED; denoiser uses TRANSFER_DONE with
        staged_for_decoder=True instead (see _handle_transfer_done).
        """

        request_id = msg["request_id"]

        # Store transfer state
        record = self._tracker.get(request_id)
        encoder_idx = record.encoder_instance if record else 0

        p2p = _TransferRequestState(
            sender_session_id=msg.get("session_id", ""),
            sender_pool_ptr=msg.get("pool_ptr", 0),
            sender_slot_offset=msg.get("slot_offset", 0),
            data_size=msg.get("data_size", 0),
            manifest=msg.get("manifest", {}),
            scalar_fields=msg.get("scalar_fields", {}),
            sender_instance=encoder_idx,
        )
        self._transfer_state[request_id] = p2p

        # Note: encoder slot is NOT freed here — the encoder's TransferBuffer
        # still holds the data. Slot is freed in _handle_transfer_pushed after
        # RDMA transfer completes and the encoder releases its local buffer.
        try:
            self._tracker.transition(request_id, RequestState.ENCODER_DONE)
        except ValueError:
            pass

        # Enqueue for denoiser transfer, then drain both
        try:
            self._tracker.transition(request_id, RequestState.DENOISING_WAITING)
        except ValueError:
            pass
        self._denoiser_tta.append(
            _RoleTTAEntry(request_id=request_id, transfer_state=p2p)
        )

    def _transfer_dispatch_to_denoiser(
        self, request_id: str, p2p: _TransferRequestState, denoiser_idx: int
    ) -> None:
        """Dispatch transfer to denoiser.

        Phase 7c: If the denoiser has pre-allocated slots, skip ALLOC roundtrip
        and send PUSH directly to the encoder.
        """

        self._denoiser_free_slots[denoiser_idx] -= 1
        p2p.receiver_instance = denoiser_idx

        try:
            self._tracker.transition(
                request_id,
                RequestState.DENOISING_RUNNING,
                denoiser_instance=denoiser_idx,
            )
        except ValueError:
            pass

        # Phase 7c: Try to use a pre-allocated slot (skip ALLOC→ALLOCATED roundtrip)
        peer_info = self._denoiser_peers.get(denoiser_idx, {})
        free_slots = peer_info.get("free_preallocated_slots", [])
        # Only use prealloc slot if it's large enough for the data
        if free_slots and free_slots[0].get("size", 0) >= p2p.data_size:
            slot_info = free_slots.pop(0)
            # Store receiver info directly
            p2p.receiver_session_id = peer_info.get("session_id", "")
            p2p.receiver_pool_ptr = peer_info.get("pool_ptr", 0)
            p2p.receiver_slot_offset = slot_info["offset"]
            p2p.prealloc_slot_id = slot_info.get("slot_id")

            # Send PUSH directly to encoder (skip ALLOC roundtrip)
            dest_addr = slot_info["addr"]
            push_msg = TransferPushMsg(
                request_id=request_id,
                dest_session_id=p2p.receiver_session_id,
                dest_addr=dest_addr,
                transfer_size=p2p.data_size,
            )
            sender_idx = p2p.sender_instance
            self._encoder_pushes[sender_idx].send_multipart(
                encode_transfer_msg(push_msg)
            )
            logger.debug(
                "DiffusionServer transfer: fast-path push to denoiser[%d] for %s "
                "(prealloc slot %s, %d bytes)",
                denoiser_idx,
                request_id,
                slot_info.get("slot_id"),
                p2p.data_size,
            )
        else:
            # Fallback: original ALLOC roundtrip
            alloc_msg = TransferAllocMsg(
                request_id=request_id,
                data_size=p2p.data_size,
                source_role="encoder",
            )
            self._denoiser_pushes[denoiser_idx].send_multipart(
                encode_transfer_msg(alloc_msg)
            )
            logger.debug(
                "DiffusionServer transfer: sent alloc to denoiser[%d] for %s (%d bytes)",
                denoiser_idx,
                request_id,
                p2p.data_size,
            )

    def _handle_transfer_allocated(self, msg: dict) -> None:
        """Receiver allocated a slot. Send transfer_push to sender."""

        request_id = msg["request_id"]
        p2p = self._transfer_state.get(request_id)
        if p2p is None:
            logger.warning(
                "DiffusionServer transfer: no state for allocated %s", request_id
            )
            return

        p2p.receiver_session_id = msg.get("session_id", "")
        p2p.receiver_pool_ptr = msg.get("pool_ptr", 0)
        p2p.receiver_slot_offset = msg.get("slot_offset", 0)

        # Calculate absolute destination address
        dest_addr = p2p.receiver_pool_ptr + p2p.receiver_slot_offset

        # Tell sender to push via RDMA
        push_msg = TransferPushMsg(
            request_id=request_id,
            dest_session_id=p2p.receiver_session_id,
            dest_addr=dest_addr,
            transfer_size=p2p.data_size,
        )

        # Send push to the actual sender (encoder or denoiser depending on stage)
        sender_idx = p2p.sender_instance
        record = self._tracker.get(request_id)
        if record and record.state in (
            RequestState.DECODER_RUNNING,
            RequestState.DECODER_WAITING,
        ):
            # Denoiser→Decoder transfer: sender is denoiser
            self._denoiser_pushes[sender_idx].send_multipart(
                encode_transfer_msg(push_msg)
            )
            logger.debug(
                "DiffusionServer transfer: sent push command to denoiser[%d] for %s",
                sender_idx,
                request_id,
            )
        else:
            # Encoder→Denoiser transfer: sender is encoder
            self._encoder_pushes[sender_idx].send_multipart(
                encode_transfer_msg(push_msg)
            )
            logger.debug(
                "DiffusionServer transfer: sent push command to encoder[%d] for %s",
                sender_idx,
                request_id,
            )

    def _handle_transfer_pushed(self, msg: dict) -> None:
        """Sender completed RDMA push. Free sender slot and tell receiver data is ready.

        Handles both encoder→denoiser and denoiser→decoder transfers.
        """

        request_id = msg["request_id"]
        p2p = self._transfer_state.get(request_id)
        if p2p is None:
            logger.warning(
                "DiffusionServer transfer: no state for pushed %s", request_id
            )
            return

        # Free sender slot — RDMA transfer is done, sender's local buffer is released.
        # Use the record's current state to determine which role was the sender,
        # because sender_idx alone is ambiguous when encoder and denoiser share
        # the same instance index (e.g., both are instance 0).
        record = self._tracker.get(request_id)
        if record and record.state in (
            RequestState.DENOISING_RUNNING,
            RequestState.DENOISING_WAITING,
            RequestState.DENOISING_DONE,
        ):
            # Encoder→Denoiser push completed: free encoder slot
            if record.encoder_instance is not None:
                self._encoder_free_slots[record.encoder_instance] += 1
        elif record and record.state in (
            RequestState.DECODER_RUNNING,
            RequestState.DECODER_WAITING,
        ):
            # Denoiser→Decoder push completed: free denoiser slot
            if record.denoiser_instance is not None:
                self._denoiser_free_slots[record.denoiser_instance] += 1

        # Tell receiver that data has arrived
        # Phase 7: include prealloc_slot_id so receiver can use pre-allocated slot
        scalar_fields = dict(p2p.scalar_fields) if p2p.scalar_fields else {}
        if p2p.prealloc_slot_id is not None:
            scalar_fields["_prealloc_slot_id"] = p2p.prealloc_slot_id
        ready_msg = TransferReadyMsg(
            request_id=request_id,
            manifest=p2p.manifest,
            slot_offset=p2p.receiver_slot_offset,
            scalar_fields=scalar_fields,
        )

        receiver_idx = p2p.receiver_instance
        # Determine which pushes to use based on what role is receiving
        record = self._tracker.get(request_id)
        if record and record.state in (
            RequestState.DENOISING_RUNNING,
            RequestState.DENOISING_WAITING,
        ):
            self._denoiser_pushes[receiver_idx].send_multipart(
                encode_transfer_msg(ready_msg)
            )
        elif record and record.state in (
            RequestState.DECODER_RUNNING,
            RequestState.DECODER_WAITING,
        ):
            self._decoder_pushes[receiver_idx].send_multipart(
                encode_transfer_msg(ready_msg)
            )

        logger.debug(
            "DiffusionServer transfer: notified receiver for %s (data ready)",
            request_id,
        )

    def _recycle_prealloc_slot(
        self, p2p: _TransferRequestState, role: RoleType
    ) -> None:
        """Phase 7e: Recycle a pre-allocated slot back to the receiver's free list."""
        if p2p is None or p2p.prealloc_slot_id is None:
            return
        receiver_idx = p2p.receiver_instance
        if role == RoleType.DENOISER:
            peer_info = self._denoiser_peers.get(receiver_idx, {})
        elif role == RoleType.DECODER:
            peer_info = self._decoder_peers.get(receiver_idx, {})
        else:
            return
        free_list = peer_info.get("free_preallocated_slots", [])
        # Re-add the slot info for reuse
        free_list.append(
            {
                "offset": p2p.receiver_slot_offset,
                "size": p2p.data_size,
                "slot_id": p2p.prealloc_slot_id,
                "addr": p2p.receiver_pool_ptr + p2p.receiver_slot_offset,
            }
        )
        p2p.prealloc_slot_id = None  # Mark as recycled

    def _handle_transfer_done(self, msg: dict, role: RoleType) -> None:
        """Role instance finished compute in transfer mode."""
        request_id = msg.get("request_id", "")
        error = msg.get("error")
        p2p = self._transfer_state.get(request_id)

        if role == RoleType.DENOISER:
            record = self._tracker.get(request_id)

            # Phase 7e: Recycle denoiser's pre-allocated slot
            if p2p is not None:
                self._recycle_prealloc_slot(p2p, RoleType.DENOISER)

            if error:
                # On error, free denoiser slot immediately (no pending transfer)
                if record and record.denoiser_instance is not None:
                    self._denoiser_free_slots[record.denoiser_instance] += 1
                self._complete_with_error(request_id, f"Denoiser error: {error}")
                return

            try:
                self._tracker.transition(request_id, RequestState.DENOISING_DONE)
            except ValueError:
                pass

            # The denoiser's done message includes new staged info for decoder
            if p2p is not None and msg.get("staged_for_decoder"):
                # Denoiser slot stays occupied until RDMA push completes
                # (freed in _handle_transfer_pushed)

                # Update transfer state with denoiser→decoder transfer info
                p2p.sender_session_id = msg.get("session_id", "")
                p2p.sender_pool_ptr = msg.get("pool_ptr", 0)
                p2p.sender_slot_offset = msg.get("slot_offset", 0)
                p2p.data_size = msg.get("data_size", 0)
                p2p.manifest = msg.get("manifest", {})
                p2p.scalar_fields = msg.get("scalar_fields", {})
                p2p.sender_instance = record.denoiser_instance if record else 0

                # Enqueue for decoder transfer
                try:
                    self._tracker.transition(request_id, RequestState.DECODER_WAITING)
                except ValueError:
                    pass
                self._decoder_tta.append(
                    _RoleTTAEntry(request_id=request_id, transfer_state=p2p)
                )
            else:
                # No pending transfer — free denoiser slot immediately
                if record and record.denoiser_instance is not None:
                    self._denoiser_free_slots[record.denoiser_instance] += 1

        elif role == RoleType.DECODER:
            # Phase 7e: Recycle decoder's pre-allocated slot
            if p2p is not None:
                self._recycle_prealloc_slot(p2p, RoleType.DECODER)

            # Free decoder slot
            record = self._tracker.get(request_id)
            if record and record.decoder_instance is not None:
                self._decoder_free_slots[record.decoder_instance] += 1

            if error:
                self._complete_with_error(request_id, f"Decoder error: {error}")
            else:
                # Decoder sends final output back to client
                try:
                    self._tracker.transition(request_id, RequestState.DONE)
                except ValueError:
                    pass

                # Forward result to HTTP client
                result_frames = msg.get("result_frames")
                if result_frames:
                    self._transfer_return_to_client(request_id, result_frames)
                else:
                    self._transfer_return_to_client_from_msg(request_id, msg)

            # Cleanup transfer state
            self._transfer_state.pop(request_id, None)

    def _transfer_dispatch_to_decoder(
        self, request_id: str, p2p: _TransferRequestState, decoder_idx: int
    ) -> None:
        """Dispatch transfer to decoder.

        Phase 7c: Same pre-allocated slot fast-path as denoiser dispatch.
        """

        self._decoder_free_slots[decoder_idx] -= 1
        p2p.receiver_instance = decoder_idx

        try:
            self._tracker.transition(
                request_id,
                RequestState.DECODER_RUNNING,
                decoder_instance=decoder_idx,
            )
        except ValueError:
            pass

        # Phase 7c: Try to use a pre-allocated slot
        peer_info = self._decoder_peers.get(decoder_idx, {})
        free_slots = peer_info.get("free_preallocated_slots", [])
        # Only use prealloc slot if it's large enough for the data
        if free_slots and free_slots[0].get("size", 0) >= p2p.data_size:
            slot_info = free_slots.pop(0)
            p2p.receiver_session_id = peer_info.get("session_id", "")
            p2p.receiver_pool_ptr = peer_info.get("pool_ptr", 0)
            p2p.receiver_slot_offset = slot_info["offset"]
            p2p.prealloc_slot_id = slot_info.get("slot_id")

            dest_addr = slot_info["addr"]
            push_msg = TransferPushMsg(
                request_id=request_id,
                dest_session_id=p2p.receiver_session_id,
                dest_addr=dest_addr,
                transfer_size=p2p.data_size,
            )
            sender_idx = p2p.sender_instance
            self._denoiser_pushes[sender_idx].send_multipart(
                encode_transfer_msg(push_msg)
            )
            logger.debug(
                "DiffusionServer transfer: fast-path push to decoder[%d] for %s "
                "(prealloc slot %s)",
                decoder_idx,
                request_id,
                slot_info.get("slot_id"),
            )
        else:
            alloc_msg = TransferAllocMsg(
                request_id=request_id,
                data_size=p2p.data_size,
                source_role="denoiser",
            )
            self._decoder_pushes[decoder_idx].send_multipart(
                encode_transfer_msg(alloc_msg)
            )

    def _transfer_return_to_client(self, request_id: str, result_frames: list) -> None:
        """Return transfer result to HTTP client.

        Decodes hex-encoded frames from the transfer decoder, unpacks them into
        an OutputBatch, and sends the pickled result to the frontend.
        """
        from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import (
            OutputBatch,
        )

        with self._lock:
            client_identity = self._pending.pop(request_id, None)

        if client_identity is None:
            self._tracker.remove(request_id)
            return

        try:
            # Decode hex-encoded frames back to bytes
            raw_frames = []
            for f in result_frames:
                if isinstance(f, str):
                    raw_frames.append(bytes.fromhex(f))
                else:
                    raw_frames.append(f)

            # Unpack frames into tensor/scalar fields
            tensor_fields, scalar_fields = unpack_tensors(raw_frames)

            # Build OutputBatch
            output_batch = OutputBatch(
                output=tensor_fields.get("output"),
                audio=tensor_fields.get("audio"),
                audio_sample_rate=scalar_fields.get("audio_sample_rate"),
                error=scalar_fields.get("error"),
            )

            self._frontend.send_multipart(
                [client_identity, b"", pickle.dumps(output_batch)]
            )
        except Exception as e:
            logger.error(
                "DiffusionServer transfer: failed to send result for %s: %s",
                request_id,
                e,
            )

        self._tracker.remove(request_id)

    def _transfer_return_to_client_from_msg(self, request_id: str, msg: dict) -> None:
        """Return transfer result to HTTP client from message fields."""
        from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import (
            OutputBatch,
        )

        with self._lock:
            client_identity = self._pending.pop(request_id, None)

        if client_identity is None:
            self._tracker.remove(request_id)
            return

        output_batch = OutputBatch(error=msg.get("error"))

        try:
            self._frontend.send_multipart(
                [client_identity, b"", pickle.dumps(output_batch)]
            )
        except zmq.ZMQError as e:
            logger.error(
                "DiffusionServer transfer: failed to send result for %s: %s",
                request_id,
                e,
            )
        self._tracker.remove(request_id)

    def get_stats(self) -> dict:
        """Return router-level statistics for observability."""
        with self._lock:
            pending_count = len(self._pending)
        return {
            "role": "diffusion_server",
            "transfer_mode": self._transfer_mode,
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
            "transfer_active_transfers": len(self._transfer_state),
            "encoder_peers": len(self._encoder_peers),
            "denoiser_peers": len(self._denoiser_peers),
            "decoder_peers": len(self._decoder_peers),
            "tracker": self._tracker.snapshot(),
        }
