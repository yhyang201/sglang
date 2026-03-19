# SPDX-License-Identifier: Apache-2.0
"""
Transfer protocol messages for disaggregated diffusion.

Defines the message types exchanged between DiffusionServer and role
instances for RDMA tensor transfer coordination. All messages are sent
as ZMQ multipart with a b"__transfer__" discriminator in frame[0] and
JSON payload in frame[1].

Message flow (encoder → denoiser via DS):
  1. Encoder → DS: TRANSFER_STAGED   (encoder done, data in local buffer)
  2. DS → Denoiser: TRANSFER_ALLOC   (allocate buffer slot for incoming data)
  3. Denoiser → DS: TRANSFER_ALLOCATED (slot allocated, here's the offset)
  4. DS → Encoder: TRANSFER_PUSH     (push data to denoiser's buffer via RDMA)
  5. Encoder → DS: TRANSFER_PUSHED   (RDMA transfer complete)
  6. DS → Denoiser: TRANSFER_READY   (data arrived, process it)
  7. Denoiser → DS: TRANSFER_DONE    (compute finished, here's the result)

Message flow (denoiser → decoder via DS):
  Same 7-step protocol. Denoiser's TRANSFER_DONE message doubles as the
  trigger by including staged_for_decoder=True with its buffer info.
  1. Denoiser → DS: TRANSFER_DONE + staged_for_decoder (data staged in local buffer)
  2. DS → Decoder: TRANSFER_ALLOC
  3. Decoder → DS: TRANSFER_ALLOCATED
  4. DS → Denoiser: TRANSFER_PUSH
  5. Denoiser → DS: TRANSFER_PUSHED
  6. DS → Decoder: TRANSFER_READY
  7. Decoder → DS: TRANSFER_DONE    (final result returned to client)
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Discriminator prefix for transfer control messages on existing ZMQ sockets
TRANSFER_MAGIC = b"__transfer__"


# --- Message types ---


class TransferMsgType:
    """Transfer control message types."""

    # Instance → DS
    STAGED = "transfer_staged"  # Encoder finished, data staged in local buffer
    ALLOCATED = "transfer_allocated"  # Receiver allocated a slot
    PUSHED = "transfer_pushed"  # RDMA push complete
    DONE = "transfer_done"  # Compute done, result ready

    # DS → Instance
    ALLOC = "transfer_alloc"  # DS asks instance to allocate a slot
    PUSH = "transfer_push"  # DS tells sender to do RDMA push
    READY = "transfer_ready"  # DS tells receiver data has arrived

    # Registration (at startup)
    REGISTER = "transfer_register"  # Instance registers with DS
    REGISTER_ACK = "transfer_register_ack"  # DS acknowledges registration


# --- Message payloads ---


@dataclass
class TransferStagedMsg:
    """Encoder → DS: data staged in local TransferBuffer."""

    msg_type: str = TransferMsgType.STAGED
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
class TransferAllocMsg:
    """DS → Receiver: allocate a buffer slot for incoming data."""

    msg_type: str = TransferMsgType.ALLOC
    request_id: str = ""
    data_size: int = 0
    source_role: str = ""  # "encoder" or "denoiser"


@dataclass
class TransferAllocatedMsg:
    """Receiver → DS: slot allocated."""

    msg_type: str = TransferMsgType.ALLOCATED
    request_id: str = ""
    session_id: str = ""  # receiver's engine session_id
    pool_ptr: int = 0  # receiver's pool base address
    slot_offset: int = 0  # offset within pool
    slot_size: int = 0


@dataclass
class TransferPushMsg:
    """DS → Sender: push data to receiver's buffer via RDMA."""

    msg_type: str = TransferMsgType.PUSH
    request_id: str = ""
    dest_session_id: str = ""
    dest_addr: int = 0  # absolute address in receiver's pool
    transfer_size: int = 0


@dataclass
class TransferPushedMsg:
    """Sender → DS: RDMA transfer complete."""

    msg_type: str = TransferMsgType.PUSHED
    request_id: str = ""


@dataclass
class TransferReadyMsg:
    """DS → Receiver: data has arrived, process it."""

    msg_type: str = TransferMsgType.READY
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
class TransferDoneMsg:
    """Receiver → DS: compute finished."""

    msg_type: str = TransferMsgType.DONE
    request_id: str = ""
    # For decoder: include result payload
    result_frames: list[bytes] | None = None
    error: str | None = None


@dataclass
class TransferRegisterMsg:
    """Instance → DS: register at startup.

    Phase 7a: includes pre-allocated receive slots so DS can bypass
    the ALLOC→ALLOCATED roundtrip.
    """

    msg_type: str = TransferMsgType.REGISTER
    role: str = ""  # "encoder", "denoiser", "decoder"
    session_id: str = ""
    pool_ptr: int = 0
    pool_size: int = 0
    # Phase 7: pre-allocated receive slots
    # Each dict: {"offset": int, "size": int, "slot_id": int, "addr": int}
    preallocated_slots: list = field(default_factory=list)


# --- Serialization helpers ---


def encode_transfer_msg(msg: Any) -> list[bytes]:
    """Encode a transfer message as ZMQ multipart frames.

    Returns [TRANSFER_MAGIC, json_payload_bytes].
    """
    if hasattr(msg, "__dataclass_fields__"):
        d = asdict(msg)
    elif isinstance(msg, dict):
        d = msg
    else:
        raise TypeError(f"Cannot encode transfer message: {type(msg)}")

    # Remove non-serializable fields
    d.pop("result_frames", None)

    return [TRANSFER_MAGIC, json.dumps(d, separators=(",", ":")).encode("utf-8")]


def decode_transfer_msg(frames: list[bytes]) -> dict:
    """Decode a transfer message from ZMQ multipart frames.

    Expects frames[0] == TRANSFER_MAGIC, frames[1] == JSON payload.
    Returns the parsed dict.
    """
    if len(frames) < 2 or frames[0] != TRANSFER_MAGIC:
        raise ValueError(f"Not a transfer message: frame[0]={frames[0]!r}")
    return json.loads(frames[1])


def is_transfer_message(frames: list) -> bool:
    """Check if a ZMQ multipart message is a transfer control message."""
    return len(frames) >= 2 and (
        frames[0] == TRANSFER_MAGIC
        or (isinstance(frames[0], memoryview) and bytes(frames[0]) == TRANSFER_MAGIC)
        or (hasattr(frames[0], "bytes") and frames[0].bytes == TRANSFER_MAGIC)
    )
