# SPDX-License-Identifier: Apache-2.0
"""
Role connectors for disaggregated diffusion pipelines.

Handles serialization/deserialization of Req fields between pipeline roles:
  Encoder -> Denoiser: text embeddings, latents, timesteps, metadata
  Denoiser -> Decoder: denoised latents, metadata

Uses ZMQ PUSH/PULL with zero-copy tensor transport (send_multipart).
"""

import logging

import torch
import zmq

from sglang.multimodal_gen.runtime.disaggregation.tensor_transport import (
    recv_tensors,
    send_tensors,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import (
    Req,
)
from sglang.multimodal_gen.runtime.utils.common import get_zmq_socket

logger = logging.getLogger(__name__)


# --- Field definitions for each role transition ---

# Tensor fields produced by Encoder stages, consumed by Denoiser
ENCODER_TO_DENOISER_TENSOR_FIELDS = [
    "prompt_embeds",
    "negative_prompt_embeds",
    "pooled_embeds",
    "neg_pooled_embeds",
    "prompt_attention_mask",
    "negative_attention_mask",
    "clip_embedding_pos",
    "clip_embedding_neg",
    "latents",
    "timesteps",
    "y",
    "image_latent",
    "latent_ids",
    # Audio (LTX-2)
    "audio_prompt_embeds",
    "negative_audio_prompt_embeds",
    "audio_latents",
    "audio_noise",
]

# Scalar fields from Encoder that Denoiser needs
ENCODER_TO_DENOISER_SCALAR_FIELDS = [
    "request_id",
    "do_classifier_free_guidance",
    "guidance_scale",
    "guidance_scale_2",
    "height",
    "width",
    "num_frames",
    "fps",
    "num_inference_steps",
    "eta",
    "sigmas",
    "n_tokens",
    "height_latents",
    "width_latents",
    "raw_latent_shape",
    "raw_audio_latent_shape",
    "seed",
    "seeds",
    "is_warmup",
    "is_prompt_processed",
    "generate_audio",
    "output_file_ext",
    # STA/VSA
    "STA_param",
    "is_cfg_negative",
    "mask_search_final_result_pos",
    "mask_search_final_result_neg",
    "VSA_sparsity",
]

# Tensor fields produced by Denoiser, consumed by Decoder
DENOISER_TO_DECODER_TENSOR_FIELDS = [
    "latents",
    "audio_latents",
    "noise_pred",
]

# Scalar fields from Denoiser that Decoder needs
DENOISER_TO_DECODER_SCALAR_FIELDS = [
    "request_id",
    "height",
    "width",
    "num_frames",
    "raw_latent_shape",
    "raw_audio_latent_shape",
    "is_warmup",
    "output_file_ext",
    "generate_audio",
    # Error propagation: set by denoiser when forward() fails
    "_disagg_error",
]


def _extract_tensor_fields(req: Req, field_names: list[str]) -> dict:
    """Extract tensor fields from a Req object."""
    result = {}
    for name in field_names:
        value = getattr(req, name, None)
        if value is not None:
            result[name] = value
    return result


def _extract_scalar_fields(req: Req, field_names: list[str]) -> dict:
    """Extract scalar fields from a Req, converting to JSON-serializable types."""
    result = {}
    for name in field_names:
        value = getattr(req, name, None)
        if value is None:
            continue
        # Convert torch-specific types to JSON-serializable
        if isinstance(value, torch.Tensor):
            # Some "scalar" fields may actually be small tensors (e.g., raw_latent_shape)
            result[name] = value.tolist()
        elif isinstance(value, (list, tuple)):
            # Convert any tensor items in lists
            converted = []
            for item in value:
                if isinstance(item, torch.Tensor):
                    converted.append(item.tolist())
                else:
                    converted.append(item)
            result[name] = converted
        else:
            result[name] = value
    return result


def _apply_scalar_fields(req: Req, scalar_fields: dict, field_names: list[str]):
    """Apply scalar fields to a Req object."""
    for name in field_names:
        if name in scalar_fields:
            setattr(req, name, scalar_fields[name])


def _apply_tensor_fields(req: Req, tensor_fields: dict):
    """Apply tensor fields to a Req object."""
    for name, value in tensor_fields.items():
        setattr(req, name, value)


class RoleConnectorSender:
    """Sends Req data from one role to the next via ZMQ PUSH.

    Usage:
        sender = RoleConnectorSender(context, endpoint, tensor_field_names, scalar_field_names)
        sender.send(req)
        sender.close()
    """

    def __init__(
        self,
        context: zmq.Context,
        endpoint: str,
        tensor_field_names: list[str],
        scalar_field_names: list[str],
    ):
        self._tensor_field_names = tensor_field_names
        self._scalar_field_names = scalar_field_names
        self._socket, self._endpoint = get_zmq_socket(
            context, zmq.PUSH, endpoint, bind=True
        )
        # Resolve actual endpoint (needed when binding to port 0 or wildcard)
        actual = self._socket.getsockopt(zmq.LAST_ENDPOINT)
        if isinstance(actual, bytes):
            actual = actual.decode("utf-8")
        if actual:
            self._endpoint = actual
        logger.info("RoleConnectorSender bound at %s", self._endpoint)

    @property
    def endpoint(self) -> str:
        return self._endpoint

    def send(self, req: Req) -> None:
        """Extract and send relevant fields from Req."""
        tensor_fields = _extract_tensor_fields(req, self._tensor_field_names)
        scalar_fields = _extract_scalar_fields(req, self._scalar_field_names)
        send_tensors(self._socket, tensor_fields, scalar_fields)
        logger.debug(
            "Sent %d tensor fields, %d scalar fields for request %s",
            len(tensor_fields),
            len(scalar_fields),
            getattr(req, "request_id", "unknown"),
        )

    def close(self):
        self._socket.close()


class RoleConnectorReceiver:
    """Receives Req data from a previous role via ZMQ PULL.

    Usage:
        receiver = RoleConnectorReceiver(context, endpoint, tensor_field_names, scalar_field_names)
        req = receiver.recv()  # blocking
        receiver.close()
    """

    def __init__(
        self,
        context: zmq.Context,
        endpoint: str,
        tensor_field_names: list[str],
        scalar_field_names: list[str],
        device: str | torch.device = "cpu",
    ):
        self._tensor_field_names = tensor_field_names
        self._scalar_field_names = scalar_field_names
        self._device = device
        self._socket, self._endpoint = get_zmq_socket(
            context, zmq.PULL, endpoint, bind=False
        )
        logger.info("RoleConnectorReceiver connected to %s", endpoint)

    def recv(self, flags: int = 0, timeout_ms: int | None = None) -> Req:
        """Receive and reconstruct a Req with the transferred fields.

        Args:
            flags: ZMQ flags (e.g., zmq.NOBLOCK)
            timeout_ms: Optional receive timeout in milliseconds.
                If set, raises TimeoutError when no message arrives in time.

        Returns:
            A new Req populated with the received tensor and scalar fields.

        Raises:
            zmq.Again: if flags includes NOBLOCK and no message is ready.
            TimeoutError: if timeout_ms is set and expires before a message.
        """
        # Apply timeout via socket option (restored after recv)
        old_rcvtimeo = None
        if timeout_ms is not None:
            old_rcvtimeo = self._socket.getsockopt(zmq.RCVTIMEO)
            self._socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        try:
            tensor_fields, scalar_fields = recv_tensors(
                self._socket, flags=flags, device=self._device
            )
        except zmq.Again:
            if timeout_ms is not None:
                raise TimeoutError(
                    f"RoleConnectorReceiver.recv timed out after {timeout_ms}ms"
                ) from None
            raise
        finally:
            if old_rcvtimeo is not None:
                self._socket.setsockopt(zmq.RCVTIMEO, old_rcvtimeo)

        # Build a minimal Req with received data
        # Use scalar_fields to initialize Req (request_id, prompt params, etc.)
        init_kwargs = {}
        # request_id and prompt are needed for Req initialization
        if "request_id" in scalar_fields:
            init_kwargs["request_id"] = scalar_fields["request_id"]
        # Req.validate() needs guidance_scale and negative_prompt
        if "guidance_scale" in scalar_fields:
            init_kwargs["guidance_scale"] = scalar_fields["guidance_scale"]

        req = Req(**init_kwargs)

        _apply_scalar_fields(req, scalar_fields, self._scalar_field_names)
        _apply_tensor_fields(req, tensor_fields)

        # Recreate torch.Generator from seed (generators can't be serialized)
        seed = scalar_fields.get("seed")
        if seed is not None:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(seed))
            req.generator = generator

        logger.debug(
            "Received %d tensor fields, %d scalar fields for request %s",
            len(tensor_fields),
            len(scalar_fields),
            getattr(req, "request_id", "unknown"),
        )
        return req

    def try_recv(self) -> Req | None:
        """Non-blocking receive. Returns None if no message available."""
        try:
            return self.recv(flags=zmq.NOBLOCK)
        except zmq.Again:
            return None

    def close(self):
        self._socket.close()


def create_encoder_to_denoiser_sender(
    context: zmq.Context, endpoint: str
) -> RoleConnectorSender:
    """Create a sender for the Encoder -> Denoiser transition."""
    return RoleConnectorSender(
        context,
        endpoint,
        ENCODER_TO_DENOISER_TENSOR_FIELDS,
        ENCODER_TO_DENOISER_SCALAR_FIELDS,
    )


def create_encoder_to_denoiser_receiver(
    context: zmq.Context, endpoint: str, device: str | torch.device = "cpu"
) -> RoleConnectorReceiver:
    """Create a receiver for the Encoder -> Denoiser transition."""
    return RoleConnectorReceiver(
        context,
        endpoint,
        ENCODER_TO_DENOISER_TENSOR_FIELDS,
        ENCODER_TO_DENOISER_SCALAR_FIELDS,
        device=device,
    )


def create_denoiser_to_decoder_sender(
    context: zmq.Context, endpoint: str
) -> RoleConnectorSender:
    """Create a sender for the Denoiser -> Decoder transition."""
    return RoleConnectorSender(
        context,
        endpoint,
        DENOISER_TO_DECODER_TENSOR_FIELDS,
        DENOISER_TO_DECODER_SCALAR_FIELDS,
    )


def create_denoiser_to_decoder_receiver(
    context: zmq.Context, endpoint: str, device: str | torch.device = "cpu"
) -> RoleConnectorReceiver:
    """Create a receiver for the Denoiser -> Decoder transition."""
    return RoleConnectorReceiver(
        context,
        endpoint,
        DENOISER_TO_DECODER_TENSOR_FIELDS,
        DENOISER_TO_DECODER_SCALAR_FIELDS,
        device=device,
    )


# --- Pool mode helpers (DiffusionServer-mediated transfers) ---


def pack_encoder_output(req: Req) -> tuple[bytes, list]:
    """Pack encoder output for relay via DiffusionServer.

    Returns (metadata_bytes, buffers) ready for send_multipart.
    """
    from sglang.multimodal_gen.runtime.disaggregation.tensor_transport import (
        pack_tensors,
    )

    tensor_fields = _extract_tensor_fields(req, ENCODER_TO_DENOISER_TENSOR_FIELDS)
    scalar_fields = _extract_scalar_fields(req, ENCODER_TO_DENOISER_SCALAR_FIELDS)
    return pack_tensors(tensor_fields, scalar_fields)


def pack_denoiser_output(req: Req) -> tuple[bytes, list]:
    """Pack denoiser output for relay via DiffusionServer."""
    from sglang.multimodal_gen.runtime.disaggregation.tensor_transport import (
        pack_tensors,
    )

    tensor_fields = _extract_tensor_fields(req, DENOISER_TO_DECODER_TENSOR_FIELDS)
    scalar_fields = _extract_scalar_fields(req, DENOISER_TO_DECODER_SCALAR_FIELDS)
    return pack_tensors(tensor_fields, scalar_fields)


def build_req_from_frames(
    parts: list,
    transition: str,
    device: str | torch.device = "cpu",
) -> Req:
    """Build a Req from multipart ZMQ frames (received via relay).

    Args:
        parts: ZMQ multipart frames (metadata JSON + tensor buffers)
        transition: "encoder_to_denoiser" or "denoiser_to_decoder"
        device: target device for tensors
    """
    from sglang.multimodal_gen.runtime.disaggregation.tensor_transport import (
        unpack_tensors,
    )

    tensor_fields, scalar_fields = unpack_tensors(parts, device=device)

    if transition == "encoder_to_denoiser":
        scalar_field_names = ENCODER_TO_DENOISER_SCALAR_FIELDS
    elif transition == "denoiser_to_decoder":
        scalar_field_names = DENOISER_TO_DECODER_SCALAR_FIELDS
    else:
        raise ValueError(f"Unknown transition: {transition}")

    # Build Req
    init_kwargs = {}
    if "request_id" in scalar_fields:
        init_kwargs["request_id"] = scalar_fields["request_id"]
    if "guidance_scale" in scalar_fields:
        init_kwargs["guidance_scale"] = scalar_fields["guidance_scale"]

    req = Req(**init_kwargs)
    _apply_scalar_fields(req, scalar_fields, scalar_field_names)
    _apply_tensor_fields(req, tensor_fields)

    # Recreate torch.Generator from seed
    seed = scalar_fields.get("seed")
    if seed is not None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        req.generator = generator

    return req
