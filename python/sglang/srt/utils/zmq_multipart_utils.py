"""
Centralized multipart message protocol for zero-copy ZMQ tensor/bytes transfer.

Used by MMProcessorPool for efficient IPC between the tokenizer manager and
multimodal processor worker processes.

Frame layout (both directions):
  Frame 0: pickled metadata dict (with BlobDescriptors replacing large blobs)
  Frame 1..N: raw bytes of extracted tensors or binary data
"""

import ctypes
import pickle
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch


class TensorWrapper:
    """Wrapper to keep tensor alive while exposing buffer for zero-copy ZMQ send."""

    def __init__(self, tensor: torch.Tensor):
        if tensor.is_cuda:
            tensor = tensor.cpu()
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()

        self.tensor = tensor
        self.shape = list(tensor.shape)
        self.dtype = tensor.dtype

    def __buffer__(self, flags=0):
        data_ptr = self.tensor.data_ptr()
        total_bytes = self.tensor.numel() * self.tensor.element_size()
        c_obj = (ctypes.c_char * total_bytes).from_address(data_ptr)
        c_obj._keep_alive_ref = self
        return memoryview(c_obj)


@dataclass
class BlobDescriptor:
    """Placeholder inserted into pickled metadata for large binary/tensor data."""

    frame_index: int
    dtype: Optional[str] = None  # torch dtype string, e.g. "torch.float32"
    shape: Optional[List[int]] = None
    field_path: str = ""  # e.g. "mm_items.0.feature"


def encode_multipart(
    obj: Any,
    extract_fn: Callable[[Any], Tuple[Any, List[bytes]]],
) -> List:
    """Encode an object into multipart ZMQ frames.

    Args:
        obj: The object to encode.
        extract_fn: Callback that takes obj, replaces large blobs with
            BlobDescriptors, and returns (modified_obj, list_of_blob_frames).

    Returns:
        List of frames: [pickled_metadata, blob_0, blob_1, ...]
    """
    modified_obj, blob_frames = extract_fn(obj)
    frames = [pickle.dumps(modified_obj)]
    frames.extend(blob_frames)
    return frames


def decode_multipart(
    parts: List,
    restore_fn: Callable[[Any, List], Any],
) -> Any:
    """Decode multipart ZMQ frames back into the original object.

    Args:
        parts: List of ZMQ frames [pickled_metadata, blob_0, blob_1, ...].
        restore_fn: Callback that takes (unpickled_obj, parts) and restores
            BlobDescriptors back to tensors/bytes.

    Returns:
        The restored object.
    """
    obj = pickle.loads(bytes(parts[0]))
    return restore_fn(obj, parts)


# ---------------------------------------------------------------------------
# Response: worker → pool  (mm processing results with tensor features)
# ---------------------------------------------------------------------------


def extract_mm_result_blobs(result_dict: Dict) -> Tuple[Dict, List]:
    """Extract tensor blobs from mm processing result for zero-copy send.

    Replaces mm_items[i].feature tensors with BlobDescriptors.
    """
    blobs = []
    frame_index = 1  # frame 0 is the pickled metadata

    mm_items = result_dict.get("mm_items")
    if mm_items:
        for i, item in enumerate(mm_items):
            feature = getattr(item, "feature", None)
            if feature is not None and isinstance(feature, torch.Tensor):
                wrapper = TensorWrapper(feature)
                blobs.append(wrapper)
                item.feature = BlobDescriptor(
                    frame_index=frame_index,
                    dtype=str(wrapper.dtype),
                    shape=wrapper.shape,
                    field_path=f"mm_items.{i}.feature",
                )
                frame_index += 1

            precomputed = getattr(item, "precomputed_embeddings", None)
            if precomputed is not None and isinstance(precomputed, torch.Tensor):
                wrapper = TensorWrapper(precomputed)
                blobs.append(wrapper)
                item.precomputed_embeddings = BlobDescriptor(
                    frame_index=frame_index,
                    dtype=str(wrapper.dtype),
                    shape=wrapper.shape,
                    field_path=f"mm_items.{i}.precomputed_embeddings",
                )
                frame_index += 1

    return result_dict, blobs


def restore_mm_result_blobs(result_dict: Dict, parts: List) -> Dict:
    """Restore tensor blobs in mm processing result from ZMQ frames."""
    mm_items = result_dict.get("mm_items")
    if mm_items:
        for item in mm_items:
            feature = getattr(item, "feature", None)
            if isinstance(feature, BlobDescriptor):
                item.feature = _restore_tensor(feature, parts)

            precomputed = getattr(item, "precomputed_embeddings", None)
            if isinstance(precomputed, BlobDescriptor):
                item.precomputed_embeddings = _restore_tensor(precomputed, parts)

    return result_dict


def _restore_tensor(desc: BlobDescriptor, parts: List) -> torch.Tensor:
    """Reconstruct a tensor from a BlobDescriptor and raw frame bytes."""
    raw = bytes(parts[desc.frame_index])
    dtype = _parse_dtype(desc.dtype)
    tensor = torch.frombuffer(bytearray(raw), dtype=dtype).reshape(desc.shape)
    return tensor


def _parse_dtype(dtype_str: str) -> torch.dtype:
    """Convert dtype string like 'torch.float32' to torch.dtype."""
    return getattr(torch, dtype_str.replace("torch.", ""))


# ---------------------------------------------------------------------------
# Request: pool → worker  (mm processing requests with image/audio bytes)
# ---------------------------------------------------------------------------


def extract_mm_request_blobs(request_dict: Dict) -> Tuple[Dict, List]:
    """Extract large binary data from mm processing request for zero-copy send.

    Replaces image_data/audio_data entries that are raw bytes with BlobDescriptors.
    """
    blobs = []
    frame_index = 1

    for field_name in ("image_data", "audio_data"):
        data_list = request_dict.get(field_name)
        if data_list is None:
            continue
        for i, item in enumerate(data_list):
            if isinstance(item, (bytes, bytearray)):
                blobs.append(bytes(item))
                data_list[i] = BlobDescriptor(
                    frame_index=frame_index,
                    field_path=f"{field_name}.{i}",
                )
                frame_index += 1

    return request_dict, blobs


def restore_mm_request_blobs(request_dict: Dict, parts: List) -> Dict:
    """Restore binary data in mm processing request from ZMQ frames."""
    for field_name in ("image_data", "audio_data"):
        data_list = request_dict.get(field_name)
        if data_list is None:
            continue
        for i, item in enumerate(data_list):
            if isinstance(item, BlobDescriptor):
                data_list[i] = bytes(parts[item.frame_index])

    return request_dict
