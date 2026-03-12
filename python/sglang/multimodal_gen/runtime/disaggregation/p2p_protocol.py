# SPDX-License-Identifier: Apache-2.0
"""
P2P transfer protocol messages for disaggregated diffusion.

Defines the message types exchanged between DiffusionServer and role
instances for P2P tensor transfer coordination. All messages are sent
as ZMQ multipart with a b"__p2p__" discriminator in frame[0] and
JSON payload in frame[1].

Message flow (encoder → denoiser via DS):
  1. Encoder → DS: P2P_STAGED   (encoder done, data in local buffer)
  2. DS → Denoiser: P2P_ALLOC   (allocate buffer slot for incoming data)
  3. Denoiser → DS: P2P_ALLOCATED (slot allocated, here's the offset)
  4. DS → Encoder: P2P_PUSH     (push data to denoiser's buffer via RDMA)
  5. Encoder → DS: P2P_PUSHED   (RDMA transfer complete)
  6. DS → Denoiser: P2P_READY   (data arrived, process it)
  7. Denoiser → DS: P2P_DONE    (compute finished, here's the result)

Message flow (denoiser → decoder via DS):
  Same 7-step protocol. Denoiser's P2P_DONE message doubles as the
  trigger by including staged_for_decoder=True with its buffer info.
  1. Denoiser → DS: P2P_DONE + staged_for_decoder (data staged in local buffer)
  2. DS → Decoder: P2P_ALLOC
  3. Decoder → DS: P2P_ALLOCATED
  4. DS → Denoiser: P2P_PUSH
  5. Denoiser → DS: P2P_PUSHED
  6. DS → Decoder: P2P_READY
  7. Decoder → DS: P2P_DONE    (final result returned to client)
"""

import json
import logging
from dataclasses import asdict, dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Discriminator prefix for P2P control messages on existing ZMQ sockets
P2P_MAGIC = b"__p2p__"


# --- Message types ---


class P2PMsgType:
    """P2P control message types."""

    # Instance → DS
    STAGED = "p2p_staged"  # Encoder finished, data staged in local buffer
    ALLOCATED = "p2p_allocated"  # Receiver allocated a slot
    PUSHED = "p2p_pushed"  # RDMA push complete
    DONE = "p2p_done"  # Compute done, result ready

    # DS → Instance
    ALLOC = "p2p_alloc"  # DS asks instance to allocate a slot
    PUSH = "p2p_push"  # DS tells sender to do RDMA push
    READY = "p2p_ready"  # DS tells receiver data has arrived

    # Registration (at startup)
    REGISTER = "p2p_register"  # Instance registers with DS
    REGISTER_ACK = "p2p_register_ack"  # DS acknowledges registration


# --- Message payloads ---


@dataclass
class P2PStagedMsg:
    """Encoder → DS: data staged in local TransferBuffer."""

    msg_type: str = P2PMsgType.STAGED
    request_id: str = ""
    data_size: int = 0  # bytes
    manifest: dict = None  # {field_name: [{offset, shape, dtype}, ...]}
    session_id: str = ""  # sender's engine session_id
    pool_ptr: int = 0  # sender's pool base address
    slot_offset: int = 0  # offset within pool

    def __post_init__(self):
        if self.manifest is None:
            self.manifest = {}


@dataclass
class P2PAllocMsg:
    """DS → Receiver: allocate a buffer slot for incoming data."""

    msg_type: str = P2PMsgType.ALLOC
    request_id: str = ""
    data_size: int = 0
    source_role: str = ""  # "encoder" or "denoiser"


@dataclass
class P2PAllocatedMsg:
    """Receiver → DS: slot allocated."""

    msg_type: str = P2PMsgType.ALLOCATED
    request_id: str = ""
    session_id: str = ""  # receiver's engine session_id
    pool_ptr: int = 0  # receiver's pool base address
    slot_offset: int = 0  # offset within pool
    slot_size: int = 0


@dataclass
class P2PPushMsg:
    """DS → Sender: push data to receiver's buffer via RDMA."""

    msg_type: str = P2PMsgType.PUSH
    request_id: str = ""
    dest_session_id: str = ""
    dest_addr: int = 0  # absolute address in receiver's pool
    transfer_size: int = 0


@dataclass
class P2PPushedMsg:
    """Sender → DS: RDMA transfer complete."""

    msg_type: str = P2PMsgType.PUSHED
    request_id: str = ""


@dataclass
class P2PReadyMsg:
    """DS → Receiver: data has arrived, process it."""

    msg_type: str = P2PMsgType.READY
    request_id: str = ""
    manifest: dict = None  # tensor layout in the slot
    slot_offset: int = 0
    # Include scalar fields for reconstructing the Req
    scalar_fields: dict = None

    def __post_init__(self):
        if self.manifest is None:
            self.manifest = {}
        if self.scalar_fields is None:
            self.scalar_fields = {}


@dataclass
class P2PDoneMsg:
    """Receiver → DS: compute finished."""

    msg_type: str = P2PMsgType.DONE
    request_id: str = ""
    # For decoder: include result payload
    result_frames: list[bytes] | None = None
    error: str | None = None


@dataclass
class P2PRegisterMsg:
    """Instance → DS: register at startup."""

    msg_type: str = P2PMsgType.REGISTER
    role: str = ""  # "encoder", "denoiser", "decoder"
    instance_idx: int = 0
    session_id: str = ""
    pool_ptr: int = 0
    pool_size: int = 0


# --- Serialization helpers ---


def encode_p2p_msg(msg: Any) -> list[bytes]:
    """Encode a P2P message as ZMQ multipart frames.

    Returns [P2P_MAGIC, json_payload_bytes].
    """
    if hasattr(msg, "__dataclass_fields__"):
        d = asdict(msg)
    elif isinstance(msg, dict):
        d = msg
    else:
        raise TypeError(f"Cannot encode P2P message: {type(msg)}")

    # Remove non-serializable fields
    d.pop("result_frames", None)

    return [P2P_MAGIC, json.dumps(d, separators=(",", ":")).encode("utf-8")]


def decode_p2p_msg(frames: list[bytes]) -> dict:
    """Decode a P2P message from ZMQ multipart frames.

    Expects frames[0] == P2P_MAGIC, frames[1] == JSON payload.
    Returns the parsed dict.
    """
    if len(frames) < 2 or frames[0] != P2P_MAGIC:
        raise ValueError(f"Not a P2P message: frame[0]={frames[0]!r}")
    return json.loads(frames[1])


def is_p2p_message(frames: list) -> bool:
    """Check if a ZMQ multipart message is a P2P control message."""
    return len(frames) >= 2 and (
        frames[0] == P2P_MAGIC
        or (isinstance(frames[0], memoryview) and bytes(frames[0]) == P2P_MAGIC)
        or (hasattr(frames[0], "bytes") and frames[0].bytes == P2P_MAGIC)
    )
