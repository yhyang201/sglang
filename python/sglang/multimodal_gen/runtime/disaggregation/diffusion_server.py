# SPDX-License-Identifier: Apache-2.0
"""
Central request router for disaggregated diffusion pipelines.

DiffusionServer is the global pipeline orchestrator. It manages independent
pools of N encoders, M denoisers, and K decoders, dispatching at every
role transition:

  1. Client request → DiffusionServer picks Encoder[i]
  2. Encoder[i] completes → DiffusionServer picks Denoiser[j], relays tensors
  3. Denoiser[j] completes → DiffusionServer picks Decoder[k], relays tensors
  4. Decoder[k] completes → DiffusionServer returns result to Client

Socket topology:
  - Frontend: ROUTER (bind) — HTTP server DEALER connects here
  - Per role instance: PUSH (connect) → instance PULL (bind) for work dispatch
  - Per role type: PULL (bind) ← instance PUSH (connect) for result return

Data format:
  - HTTP ↔ DiffusionServer: pickle (existing protocol)
  - DiffusionServer → Encoder: [request_id_bytes, pickled_req_bytes]
  - Encoder → DiffusionServer: send_tensors multipart (ENCODER_TO_DENOISER fields)
  - DiffusionServer → Denoiser: relayed multipart frames (zero-copy)
  - Denoiser → DiffusionServer: send_tensors multipart (DENOISER_TO_DECODER fields)
  - DiffusionServer → Decoder: relayed multipart frames (zero-copy)
  - Decoder → DiffusionServer: send_tensors multipart (output + metadata)
  - DiffusionServer → HTTP: pickle(OutputBatch)
"""

import json
import logging
import pickle
import threading
import time

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


class DiffusionServer:
    """Global pipeline orchestrator for N:M:K disaggregated diffusion.

    Manages independent encoder, denoiser, and decoder pools. Dispatches
    at every role transition using per-role dispatch policies. Relays
    tensor data between roles as zero-copy ZMQ multipart frames.
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
            "%d encoder(s), %d denoiser(s), %d decoder(s), policy=%s",
            self._frontend_endpoint,
            self._num_encoders,
            self._num_denoisers,
            self._num_decoders,
            dispatch_policy_name := type(self._dispatcher.encoder_policy).__name__,
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
        logger.info("DiffusionServer frontend bound at %s", self._frontend_endpoint)

        # Per-instance PUSH sockets for sending work
        encoder_pushes: list[zmq.Socket] = []
        for i, ep in enumerate(self._encoder_work_endpoints):
            sock, _ = get_zmq_socket(self._context, zmq.PUSH, ep, bind=False)
            encoder_pushes.append(sock)
            logger.info("DiffusionServer → encoder[%d] work at %s", i, ep)

        denoiser_pushes: list[zmq.Socket] = []
        for i, ep in enumerate(self._denoiser_work_endpoints):
            sock, _ = get_zmq_socket(self._context, zmq.PUSH, ep, bind=False)
            denoiser_pushes.append(sock)
            logger.info("DiffusionServer → denoiser[%d] work at %s", i, ep)

        decoder_pushes: list[zmq.Socket] = []
        for i, ep in enumerate(self._decoder_work_endpoints):
            sock, _ = get_zmq_socket(self._context, zmq.PUSH, ep, bind=False)
            decoder_pushes.append(sock)
            logger.info("DiffusionServer → decoder[%d] work at %s", i, ep)

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
        logger.info(
            "DiffusionServer result sockets bound: encoder=%s, denoiser=%s, decoder=%s",
            self._encoder_result_endpoint,
            self._denoiser_result_endpoint,
            self._decoder_result_endpoint,
        )

        poller = zmq.Poller()
        poller.register(frontend, zmq.POLLIN)
        poller.register(encoder_result_pull, zmq.POLLIN)
        poller.register(denoiser_result_pull, zmq.POLLIN)
        poller.register(decoder_result_pull, zmq.POLLIN)

        all_sockets = (
            [frontend, encoder_result_pull, denoiser_result_pull, decoder_result_pull]
            + encoder_pushes
            + denoiser_pushes
            + decoder_pushes
        )

        try:
            while self._running:
                events = dict(poller.poll(timeout=100))

                self._handle_timeouts(frontend)

                if frontend in events:
                    self._handle_client_request(frontend, encoder_pushes)

                if encoder_result_pull in events:
                    self._handle_encoder_result(
                        encoder_result_pull, denoiser_pushes, frontend
                    )

                if denoiser_result_pull in events:
                    self._handle_denoiser_result(
                        denoiser_result_pull, decoder_pushes, frontend
                    )

                if decoder_result_pull in events:
                    self._handle_decoder_result(decoder_result_pull, frontend)

        except Exception:
            logger.exception("DiffusionServer event loop error")
        finally:
            for sock in all_sockets:
                sock.close()
            self._context.destroy(linger=0)

    # --- Event handlers ---

    def _handle_client_request(
        self, frontend: zmq.Socket, encoder_pushes: list[zmq.Socket]
    ) -> None:
        """Receive client request from HTTP server, dispatch to encoder."""
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

        # Select encoder instance
        active_counts = [
            self._tracker.count_at_instance("encoder", i)
            for i in range(self._num_encoders)
        ]
        encoder_idx = self._dispatcher.select_encoder(active_counts)

        # Track request
        try:
            self._tracker.submit(request_id)
            self._tracker.transition(
                request_id,
                RequestState.ENCODER_RUNNING,
                encoder_instance=encoder_idx,
            )
        except ValueError:
            logger.warning("DiffusionServer: duplicate request_id %s", request_id)

        with self._lock:
            self._pending[request_id] = client_identity

        # Send to encoder: [request_id_bytes, pickled_req_bytes]
        encoder_pushes[encoder_idx].send_multipart(
            [request_id.encode("utf-8"), payload]
        )
        logger.debug(
            "DiffusionServer: dispatched %s to encoder[%d]",
            request_id,
            encoder_idx,
        )

    def _handle_encoder_result(
        self,
        result_pull: zmq.Socket,
        denoiser_pushes: list[zmq.Socket],
        frontend: zmq.Socket,
    ) -> None:
        """Receive encoder output, dispatch to denoiser."""
        try:
            frames = result_pull.recv_multipart(zmq.NOBLOCK, copy=True)
        except zmq.Again:
            return

        request_id = self._extract_request_id(frames)
        if request_id is None:
            logger.warning("DiffusionServer: encoder result missing request_id")
            return

        # Check for error
        error = self._extract_error(frames)
        if error:
            self._complete_with_error(request_id, f"Encoder error: {error}", frontend)
            return

        # Transition state
        try:
            self._tracker.transition(request_id, RequestState.ENCODER_DONE)
        except ValueError:
            pass

        # Select denoiser
        active_counts = [
            self._tracker.count_at_instance("denoiser", i)
            for i in range(self._num_denoisers)
        ]
        denoiser_idx = self._dispatcher.select_denoiser(active_counts)

        try:
            self._tracker.transition(
                request_id,
                RequestState.DENOISING_RUNNING,
                denoiser_instance=denoiser_idx,
            )
        except ValueError:
            pass

        # Relay multipart frames to denoiser (zero-copy of bytes)
        denoiser_pushes[denoiser_idx].send_multipart(frames)
        logger.debug(
            "DiffusionServer: relayed %s to denoiser[%d]",
            request_id,
            denoiser_idx,
        )

    def _handle_denoiser_result(
        self,
        result_pull: zmq.Socket,
        decoder_pushes: list[zmq.Socket],
        frontend: zmq.Socket,
    ) -> None:
        """Receive denoiser output, dispatch to decoder."""
        try:
            frames = result_pull.recv_multipart(zmq.NOBLOCK, copy=True)
        except zmq.Again:
            return

        request_id = self._extract_request_id(frames)
        if request_id is None:
            logger.warning("DiffusionServer: denoiser result missing request_id")
            return

        error = self._extract_error(frames)
        if error:
            self._complete_with_error(request_id, f"Denoiser error: {error}", frontend)
            return

        try:
            self._tracker.transition(request_id, RequestState.DENOISING_DONE)
        except ValueError:
            pass

        # Select decoder
        active_counts = [
            self._tracker.count_at_instance("decoder", i)
            for i in range(self._num_decoders)
        ]
        decoder_idx = self._dispatcher.select_decoder(active_counts)

        try:
            self._tracker.transition(
                request_id,
                RequestState.DECODER_RUNNING,
                decoder_instance=decoder_idx,
            )
        except ValueError:
            pass

        decoder_pushes[decoder_idx].send_multipart(frames)
        logger.debug(
            "DiffusionServer: relayed %s to decoder[%d]",
            request_id,
            decoder_idx,
        )

    def _handle_decoder_result(
        self, result_pull: zmq.Socket, frontend: zmq.Socket
    ) -> None:
        """Receive decoder output, return result to HTTP client."""
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

        # Unpack decoder output
        tensor_fields, scalar_fields = unpack_tensors(frames, device="cpu")

        output_batch = OutputBatch(
            output=tensor_fields.get("output"),
            audio=tensor_fields.get("audio"),
            audio_sample_rate=scalar_fields.get("audio_sample_rate"),
            error=scalar_fields.get("error"),
        )

        # Transition to DONE
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
            frontend.send_multipart([client_identity, b"", pickle.dumps(output_batch)])
        except zmq.ZMQError as e:
            logger.error(
                "DiffusionServer: failed to send result for %s: %s",
                request_id,
                e,
            )

        logger.debug("DiffusionServer: returned result for %s", request_id)
        self._tracker.remove(request_id)

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

    def _complete_with_error(
        self, request_id: str, error_msg: str, frontend: zmq.Socket
    ) -> None:
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
            frontend.send_multipart([client_identity, b"", pickle.dumps(error_batch)])
        except zmq.ZMQError as e:
            logger.error(
                "DiffusionServer: failed to send error for %s: %s",
                request_id,
                e,
            )

        self._tracker.remove(request_id)

    def _handle_timeouts(self, frontend: zmq.Socket) -> None:
        """Check for and handle timed-out requests."""
        from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import (
            OutputBatch,
        )

        timed_out = self._tracker.find_timed_out(self._timeout_s)
        for request_id in timed_out:
            try:
                self._tracker.transition(request_id, RequestState.TIMED_OUT)
            except ValueError:
                continue

            with self._lock:
                client_identity = self._pending.pop(request_id, None)

            if client_identity is None:
                self._tracker.remove(request_id)
                continue

            error_msg = (
                f"DiffusionServer timeout: request {request_id} "
                f"not completed within {self._timeout_s}s"
            )
            logger.error(error_msg)

            error_batch = OutputBatch(error=error_msg)
            try:
                frontend.send_multipart(
                    [client_identity, b"", pickle.dumps(error_batch)]
                )
            except zmq.ZMQError as e:
                logger.error(
                    "DiffusionServer: failed to send timeout error for %s: %s",
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
            "num_encoders": self._num_encoders,
            "num_denoisers": self._num_denoisers,
            "num_decoders": self._num_decoders,
            "pending_requests": pending_count,
            "dispatch_policy": type(self._dispatcher.encoder_policy).__name__,
            "tracker": self._tracker.snapshot(),
        }
