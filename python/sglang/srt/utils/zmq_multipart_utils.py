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
from dataclasses import fields as dataclass_fields
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
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
    is_numpy: bool = False  # True if original was numpy array


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


def _is_tensor_like(v: Any) -> bool:
    return isinstance(v, (torch.Tensor, np.ndarray))


def _wrap_tensor(
    v: Any, frame_index: int, field_path: str, blobs: list
) -> BlobDescriptor:
    """Convert a tensor/ndarray to TensorWrapper + BlobDescriptor."""
    is_numpy = isinstance(v, np.ndarray)
    if is_numpy:
        v = torch.from_numpy(np.ascontiguousarray(v))
    wrapper = TensorWrapper(v)
    blobs.append(wrapper)
    return BlobDescriptor(
        frame_index=frame_index,
        dtype=str(wrapper.dtype),
        shape=wrapper.shape,
        field_path=field_path,
        is_numpy=is_numpy,
    )


def _restore_tensor(desc: BlobDescriptor, parts: List):
    """Reconstruct a tensor/ndarray from a BlobDescriptor and raw frame bytes."""
    frame = parts[desc.frame_index]
    # Single copy: frame → writable bytearray (torch.frombuffer needs writable buffer)
    buf = bytearray(frame) if not isinstance(frame, bytearray) else frame
    dtype = _parse_dtype(desc.dtype)
    tensor = torch.frombuffer(buf, dtype=dtype).reshape(desc.shape)
    if desc.is_numpy:
        return tensor.numpy()
    return tensor


def _parse_dtype(dtype_str: str) -> torch.dtype:
    """Convert dtype string like 'torch.float32' to torch.dtype."""
    return getattr(torch, dtype_str.replace("torch.", ""))


# ---------------------------------------------------------------------------
# Response: worker → pool  (mm processing results with tensor features)
# ---------------------------------------------------------------------------


def extract_mm_result_blobs(result_dict: Dict) -> Tuple[Dict, List]:
    """Extract ALL tensor blobs from mm processing result for zero-copy send.

    Handles:
    - Top-level tensor fields (e.g. mrope_positions, mrope_position_delta)
    - mm_items[i].feature, mm_items[i].precomputed_embeddings
    - mm_items[i].model_specific_data tensors
    - mm_items[i] other tensor attributes
    """
    blobs = []
    frame_index = 1  # frame 0 is the pickled metadata

    # 1) Top-level tensors in result dict
    for key, val in result_dict.items():
        if key == "mm_items":
            continue
        if _is_tensor_like(val):
            result_dict[key] = _wrap_tensor(val, frame_index, key, blobs)
            frame_index += 1

    # 2) Tensors inside mm_items
    mm_items = result_dict.get("mm_items")
    if mm_items:
        for i, item in enumerate(mm_items):
            # Scan all attributes of the dataclass item
            for field in dataclass_fields(item):
                attr_name = field.name
                val = getattr(item, attr_name, None)
                if val is None:
                    continue

                if _is_tensor_like(val):
                    desc = _wrap_tensor(
                        val, frame_index, f"mm_items.{i}.{attr_name}", blobs
                    )
                    setattr(item, attr_name, desc)
                    frame_index += 1

            # Scan model_specific_data dict
            msd = getattr(item, "model_specific_data", None)
            if msd and isinstance(msd, dict):
                for k, v in msd.items():
                    if _is_tensor_like(v):
                        msd[k] = _wrap_tensor(
                            v, frame_index, f"mm_items.{i}.msd.{k}", blobs
                        )
                        frame_index += 1

    return result_dict, blobs


def restore_mm_result_blobs(result_dict: Dict, parts: List) -> Dict:
    """Restore ALL tensor blobs in mm processing result from ZMQ frames."""
    # 1) Top-level tensors
    for key, val in result_dict.items():
        if key == "mm_items":
            continue
        if isinstance(val, BlobDescriptor):
            result_dict[key] = _restore_tensor(val, parts)

    # 2) Tensors inside mm_items
    mm_items = result_dict.get("mm_items")
    if mm_items:
        for item in mm_items:
            for field in dataclass_fields(item):
                attr_name = field.name
                val = getattr(item, attr_name, None)
                if isinstance(val, BlobDescriptor):
                    setattr(item, attr_name, _restore_tensor(val, parts))

            msd = getattr(item, "model_specific_data", None)
            if msd and isinstance(msd, dict):
                for k, v in msd.items():
                    if isinstance(v, BlobDescriptor):
                        msd[k] = _restore_tensor(v, parts)

    return result_dict


# ---------------------------------------------------------------------------
# Request: pool → worker  (mm processing requests with image/audio/video bytes)
# ---------------------------------------------------------------------------

# All list-of-media fields that may contain large binary items.
# Covers both top-level params and request_obj_fields.
_MEDIA_LIST_FIELDS = ("image_data", "audio_data", "video_data")


def _extract_bytes_from_list(
    data_list: list,
    field_path: str,
    frame_index: int,
    blobs: list,
) -> int:
    """Replace raw bytes items in a list with BlobDescriptors, append to blobs.

    Returns the updated frame_index.
    """
    for i, item in enumerate(data_list):
        if isinstance(item, (bytes, bytearray)):
            blobs.append(bytes(item))
            data_list[i] = BlobDescriptor(
                frame_index=frame_index,
                field_path=f"{field_path}.{i}",
            )
            frame_index += 1
    return frame_index


def extract_mm_request_blobs(request_dict: Dict) -> Tuple[Dict, List]:
    """Extract large binary data from mm processing request for zero-copy send.

    Scans both top-level fields (image_data, audio_data) and nested
    request_obj_fields (video_data, audio_data, image_data) for raw bytes,
    replacing them with BlobDescriptors.
    """
    blobs: List[bytes] = []
    frame_index = 1  # frame 0 is the pickled metadata

    # Top-level media lists
    for field_name in _MEDIA_LIST_FIELDS:
        data_list = request_dict.get(field_name)
        if data_list is None:
            continue
        frame_index = _extract_bytes_from_list(
            data_list, field_name, frame_index, blobs
        )

    # Nested media lists inside request_obj_fields
    obj_fields = request_dict.get("request_obj_fields")
    if obj_fields and isinstance(obj_fields, dict):
        for field_name in _MEDIA_LIST_FIELDS:
            data_list = obj_fields.get(field_name)
            if data_list is None or not isinstance(data_list, list):
                continue
            frame_index = _extract_bytes_from_list(
                data_list, f"request_obj_fields.{field_name}", frame_index, blobs
            )

    return request_dict, blobs


def _restore_bytes_in_list(data_list: list, parts: List) -> None:
    """Restore BlobDescriptors back to bytes in-place."""
    for i, item in enumerate(data_list):
        if isinstance(item, BlobDescriptor):
            data_list[i] = bytes(parts[item.frame_index])


def restore_mm_request_blobs(request_dict: Dict, parts: List) -> Dict:
    """Restore binary data in mm processing request from ZMQ frames."""
    # Top-level media lists
    for field_name in _MEDIA_LIST_FIELDS:
        data_list = request_dict.get(field_name)
        if data_list is None:
            continue
        _restore_bytes_in_list(data_list, parts)

    # Nested media lists inside request_obj_fields
    obj_fields = request_dict.get("request_obj_fields")
    if obj_fields and isinstance(obj_fields, dict):
        for field_name in _MEDIA_LIST_FIELDS:
            data_list = obj_fields.get(field_name)
            if data_list is None or not isinstance(data_list, list):
                continue
            _restore_bytes_in_list(data_list, parts)

    return request_dict
