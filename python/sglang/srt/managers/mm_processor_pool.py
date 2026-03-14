"""
Multiprocessing pool for multimodal data processing.

Runs N worker processes, each with their own HF processor instance,
communicating via ZMQ PUSH/PULL with zero-copy tensor transfer.
"""

import asyncio
import logging
import multiprocessing as mp
import os
import pickle
import time
import uuid
from multiprocessing import shared_memory
from typing import Any, Dict, List, Optional, Union

import zmq
import zmq.asyncio

from sglang.srt.utils.zmq_multipart_utils import (
    decode_multipart,
    encode_multipart,
    extract_mm_request_blobs,
    restore_mm_result_blobs,
)

logger = logging.getLogger(__name__)

# Sentinel value to tell workers to shut down
_POISON_PILL = b"__SHUTDOWN__"

# Attributes from request_obj that processors actually use.
# We only serialize these lightweight fields instead of the entire request object.
_REQUEST_OBJ_FIELDS = (
    "rid",
    "video_data",
    "audio_data",
    "modalities",
    "image_data",
    "image_max_dynamic_patch",
    "video_max_dynamic_patch",
    "max_dynamic_patch",
    "sampling_params",
)


class _RequestObjProxy:
    """Lightweight proxy that carries only the fields processors need."""

    def __init__(self, fields: dict):
        self.__dict__.update(fields)

    def __getattr__(self, name):
        # Return None for any field not present, matching getattr(request_obj, x, None) pattern
        return None


def _extract_request_obj_fields(request_obj: Any) -> dict:
    """Extract only the needed fields from request_obj for serialization."""
    fields = {}
    for field in _REQUEST_OBJ_FIELDS:
        val = getattr(request_obj, field, None)
        if val is not None:
            fields[field] = val
    return fields


class MMProcessorPool:
    """Pool of multimodal processor worker processes with ZMQ IPC."""

    def __init__(self, num_workers: int, server_args: Any, hf_config: Any):
        self.num_workers = num_workers
        self.server_args = server_args
        self.hf_config = hf_config
        self.workers: List[mp.Process] = []
        self.pending_requests: Dict[str, asyncio.Future] = {}
        self._receiver_task: Optional[asyncio.Task] = None
        self._shm: Optional[shared_memory.SharedMemory] = None
        self._started = False

    def start(self):
        """Spawn worker processes and set up ZMQ sockets."""
        if self._started:
            return
        self._started = True

        # Write config to shared memory for workers to read
        self._shm_name = f"mm_pool_config_{os.getpid()}_{id(self)}"
        config_data = pickle.dumps((self.server_args, self.hf_config))
        self._shm = shared_memory.SharedMemory(
            create=True, size=len(config_data), name=self._shm_name
        )
        self._shm.buf[: len(config_data)] = config_data

        # Create ZMQ context for async operations
        self._ctx = zmq.asyncio.Context()

        # PUSH socket for dispatching requests to workers (sync context for send)
        self._dispatch_ctx = zmq.Context()
        self._dispatch_socket = self._dispatch_ctx.socket(zmq.PUSH)
        dispatch_port = self._dispatch_socket.bind_to_random_port("tcp://127.0.0.1")
        self._dispatch_addr = f"tcp://127.0.0.1:{dispatch_port}"

        # PULL socket for receiving results from workers (async)
        self._result_socket = self._ctx.socket(zmq.PULL)
        result_port = self._result_socket.bind_to_random_port("tcp://127.0.0.1")
        self._result_addr = f"tcp://127.0.0.1:{result_port}"

        # Spawn worker processes (use 'spawn' to avoid CUDA re-init issues in forked subprocesses)
        spawn_ctx = mp.get_context("spawn")
        for i in range(self.num_workers):
            p = spawn_ctx.Process(
                target=mm_processor_worker_main,
                args=(i, self._dispatch_addr, self._result_addr, self._shm_name),
                daemon=True,
            )
            p.start()
            self.workers.append(p)
            logger.info(f"Started MM processor worker {i} (pid={p.pid})")

    async def process(
        self,
        *,
        image_data: Optional[List[Union[str, bytes]]] = None,
        audio_data: Optional[List[Union[str, bytes]]] = None,
        input_text_or_ids: Union[str, List[int], None] = None,
        request_obj: Any,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Send a multimodal processing request to the worker pool."""
        # Lazily start the receiver task on first call (when event loop exists)
        if self._receiver_task is None:
            loop = asyncio.get_running_loop()
            self._receiver_task = loop.create_task(self._result_receiver_loop())

        request_id = str(uuid.uuid4())

        # Create future for this request
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self.pending_requests[request_id] = future

        # Build request dict — only serialize lightweight fields from request_obj
        t0 = time.perf_counter()
        request_dict = {
            "request_id": request_id,
            "input_text_or_ids": input_text_or_ids,
            "image_data": list(image_data) if image_data else None,
            "audio_data": list(audio_data) if audio_data else None,
            "request_obj_fields": _extract_request_obj_fields(request_obj),
            "kwargs": kwargs,
        }

        # Encode and send via multipart
        frames = encode_multipart(request_dict, extract_mm_request_blobs)
        t1 = time.perf_counter()
        self._dispatch_socket.send_multipart(frames, copy=False)
        t2 = time.perf_counter()

        try:
            result = await future
            t3 = time.perf_counter()
            logger.info(
                f"[MMPool Perf] {request_id[:8]}: "
                f"encode={((t1-t0)*1000):.2f}ms, "
                f"send={((t2-t1)*1000):.2f}ms, "
                f"wait_result={((t3-t2)*1000):.2f}ms, "
                f"total={((t3-t0)*1000):.2f}ms"
            )
            return result
        except Exception:
            self.pending_requests.pop(request_id, None)
            raise

    async def _result_receiver_loop(self):
        """Background task that receives results from workers and resolves futures.

        Uses copy=False to avoid double-copying tensor data, and decodes inline
        to avoid GIL contention from thread pool scheduling.
        """
        while True:
            try:
                parts = await self._result_socket.recv_multipart(copy=False)
                self._decode_and_resolve(parts)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error in MM processor pool result receiver")

    def _decode_and_resolve(self, parts: list):
        """Decode a result inline and resolve the corresponding future."""
        try:
            t_recv = time.perf_counter()
            result = decode_multipart(parts, restore_mm_result_blobs)
            t_decode = time.perf_counter()
            logger.info(
                f"[MMPool Recv] decode={((t_decode-t_recv)*1000):.2f}ms, "
                f"num_frames={len(parts)}"
            )

            request_id = result.pop("request_id", None)
            if request_id is None:
                logger.warning("Received result without request_id, discarding")
                return

            future = self.pending_requests.pop(request_id, None)
            if future is None:
                logger.warning(
                    f"Received result for unknown/timed-out request {request_id}, discarding"
                )
                return

            error = result.pop("error", None)
            if error is not None:
                future.set_exception(RuntimeError(f"MM worker error: {error}"))
            else:
                future.set_result(result)
        except Exception:
            logger.exception("Error decoding MM processor pool result")

    def shutdown(self):
        """Shut down workers and clean up resources."""
        # Cancel receiver task
        if self._receiver_task is not None:
            self._receiver_task.cancel()

        # Send poison pills to workers
        for _ in self.workers:
            try:
                self._dispatch_socket.send(_POISON_PILL)
            except Exception:
                pass

        # Wait for workers to exit
        for w in self.workers:
            w.join(timeout=5)
            if w.is_alive():
                logger.warning(f"MM worker {w.pid} did not exit, terminating")
                w.terminate()
                w.join(timeout=2)

        self.workers.clear()

        # Close sockets
        try:
            self._dispatch_socket.close(linger=0)
            self._result_socket.close(linger=0)
            self._dispatch_ctx.term()
            self._ctx.term()
        except Exception:
            pass

        # Clean up shared memory
        if self._shm is not None:
            try:
                self._shm.close()
                self._shm.unlink()
            except Exception:
                pass
            self._shm = None

        # Fail any pending requests
        for request_id, future in self.pending_requests.items():
            if not future.done():
                future.set_exception(RuntimeError("MM processor pool shut down"))
        self.pending_requests.clear()

        self._started = False


def mm_processor_worker_main(
    worker_id: int,
    recv_addr: str,
    send_addr: str,
    shm_config_name: str,
):
    """Entry point for a multimodal processor worker process."""
    try:
        _set_worker_title(worker_id)

        # Read config from shared memory
        shm = shared_memory.SharedMemory(name=shm_config_name)
        server_args, hf_config = pickle.loads(bytes(shm.buf))
        shm.close()

        # Set global server args (required by processor internals)
        from sglang.srt.server_args import set_global_server_args_for_tokenizer

        set_global_server_args_for_tokenizer(server_args)

        # Initialize HF processor
        from sglang.srt.managers.multimodal_processor import (
            get_mm_processor,
            import_processors,
        )
        from sglang.srt.managers.tokenizer_manager import (
            _determine_tensor_transport_mode,
            _get_processor_wrapper,
        )

        import_processors("sglang.srt.multimodal.processors")

        from sglang.srt.environ import envs

        if mm_process_pkg := envs.SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE.get():
            import_processors(mm_process_pkg, overwrite=True)

        _processor = _get_processor_wrapper(server_args)
        transport_mode = _determine_tensor_transport_mode(server_args)
        mm_processor = get_mm_processor(
            hf_config, server_args, _processor, transport_mode
        )

        # Disable the internal ProcessPoolExecutor inside the processor —
        # each worker IS already a separate process, so forking more processes
        # inside is wasteful. Replace with a single-threaded executor.
        import concurrent.futures

        if hasattr(mm_processor, "cpu_executor"):
            mm_processor.cpu_executor.shutdown(wait=False)
            mm_processor.cpu_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=2
            )

        logger.info(f"MM processor worker {worker_id}: initialized processor")

        # Set up ZMQ sockets
        ctx = zmq.Context()
        recv_socket = ctx.socket(zmq.PULL)
        recv_socket.connect(recv_addr)

        send_socket = ctx.socket(zmq.PUSH)
        send_socket.connect(send_addr)

        # Run the async processing loop
        asyncio.run(_worker_loop(worker_id, mm_processor, recv_socket, send_socket))

    except Exception:
        logger.exception(f"MM processor worker {worker_id} crashed")
    finally:
        logger.info(f"MM processor worker {worker_id} exiting")


async def _worker_loop(
    worker_id: int,
    mm_processor: Any,
    recv_socket: zmq.Socket,
    send_socket: zmq.Socket,
):
    """Async loop that processes requests in the worker."""
    from sglang.srt.utils.zmq_multipart_utils import (
        decode_multipart,
        encode_multipart,
        extract_mm_result_blobs,
        restore_mm_request_blobs,
    )

    while True:
        # Receive request (blocking in sync zmq, but we're in our own process)
        try:
            raw = recv_socket.recv(copy=False)
        except zmq.ZMQError:
            break

        # Check for poison pill
        if bytes(raw) == _POISON_PILL:
            logger.info(f"MM processor worker {worker_id}: received shutdown signal")
            break

        # It's a multipart message - receive the rest
        parts = [raw]
        while recv_socket.getsockopt(zmq.RCVMORE):
            parts.append(recv_socket.recv(copy=False))

        request_id = None
        frames = None
        try:
            t_w0 = time.perf_counter()
            request = decode_multipart(parts, restore_mm_request_blobs)
            t_w1 = time.perf_counter()
            request_id = request["request_id"]

            # Reconstruct a lightweight request_obj proxy from serialized fields
            request_obj = _RequestObjProxy(request.get("request_obj_fields", {}))

            # Run processing
            proc_async = getattr(mm_processor, "process_mm_data_async", None)
            if proc_async and asyncio.iscoroutinefunction(proc_async):
                result = await proc_async(
                    image_data=request.get("image_data"),
                    audio_data=request.get("audio_data"),
                    input_text=request.get("input_text_or_ids"),
                    request_obj=request_obj,
                    **request.get("kwargs", {}),
                )
            else:
                sync_fn = getattr(mm_processor, "process_mm_data", None)
                if sync_fn is None:
                    raise RuntimeError(
                        "mm_processor has neither process_mm_data_async nor process_mm_data"
                    )
                result = sync_fn(
                    image_data=request.get("image_data"),
                    audio_data=request.get("audio_data"),
                    input_text=request.get("input_text_or_ids"),
                    request_obj=request_obj,
                    **request.get("kwargs", {}),
                )
            t_w2 = time.perf_counter()

            if result is None:
                result = {}

            result["request_id"] = request_id
            result["error"] = None

            frames = encode_multipart(result, extract_mm_result_blobs)
            t_w3 = time.perf_counter()
            logger.info(
                f"[MMWorker {worker_id} Perf] {request_id[:8]}: "
                f"decode_req={((t_w1-t_w0)*1000):.2f}ms, "
                f"process={((t_w2-t_w1)*1000):.2f}ms, "
                f"encode_res={((t_w3-t_w2)*1000):.2f}ms, "
                f"total={((t_w3-t_w0)*1000):.2f}ms"
            )

        except Exception as e:
            logger.exception(
                f"MM processor worker {worker_id}: error processing request {request_id}"
            )
            result = {
                "request_id": request_id,
                "error": str(e),
            }
            frames = encode_multipart(result, extract_mm_result_blobs)

        # Send result back
        try:
            send_socket.send_multipart(frames, copy=False)
        except Exception:
            logger.exception(
                f"MM processor worker {worker_id}: error sending result for {request_id}"
            )

    # Cleanup
    recv_socket.close(linger=0)
    send_socket.close(linger=0)


def _set_worker_title(worker_id: int):
    """Set process title for easy identification."""
    try:
        from setproctitle import setproctitle

        setproctitle(f"sglang::mm_processor_worker:{worker_id}")
    except ImportError:
        pass
