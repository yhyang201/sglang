import abc
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator


class MultimodalCache(abc.ABC):
    @abc.abstractmethod
    def __init__(
        self,
    ): ...

    @staticmethod
    def combine_hashes(mm_hashes: List[int]) -> Optional[int]:
        """
        Get a combined hash from individual mm item hashes
        """
        if not mm_hashes:
            return None
        return hash(tuple(mm_hashes))

    @abc.abstractmethod
    def get(
        self, mm_hashes: List[int], combined_hash: Optional[int] = None
    ) -> Optional[torch.Tensor]:
        """
        Extract the embedding with the hash-ids of the queried items. Try combined hash first, if missed, fallback to individual hashes
        The returned tensor may not be contiguous
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def set(
        self,
        mm_hash: int,
        embedding: torch.Tensor,
        mm_embedding_allocator: BaseTokenToKVPoolAllocator,
    ) -> bool:
        """
        Set the embedding to the pre-allocated locations with a hash id
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def has(self, mm_hash: int) -> bool:
        raise NotImplementedError()

    @abc.abstractmethod
    def free(
        self, mm_hash: int, mm_embedding_allocator: BaseTokenToKVPoolAllocator
    ) -> bool:
        raise NotImplementedError()

    @abc.abstractmethod
    def clear(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def available_size(self):
        raise NotImplementedError()


def _get_tensor_size(embedding: torch.Tensor):
    return embedding.element_size() * embedding.numel()


@dataclass(kw_only=True)
class EmbeddingResult:
    embedding: torch.Tensor


class MultiModalStaticCache(MultimodalCache):
    """
    A server-level cache for multimodal embedding.
    Embeddings are computed prior, and this cache does not really pre-alloc.

    Supports pinning entries so they are not evicted by LRU. Pinned entries
    have a refcount tracked in `pinned_refcount` and their size is accounted
    in `pinned_size` (not `current_size`).
    """

    def __init__(
        self,
        max_size: int,
    ):
        super().__init__()
        self.max_size = max_size
        self.mm_cache: OrderedDict[int, EmbeddingResult] = OrderedDict()
        self.current_size = 0
        self.pinned_refcount: Dict[int, int] = {}
        self.pinned_size: int = 0

    def get(
        self, mm_hashes: List[int], combined_hash: Optional[int] = None
    ) -> Optional[EmbeddingResult]:
        combined_hash = self.combine_hashes(mm_hashes)
        # MultiModalStaticCache does not fallback to individual item lookup

        embedding = self.mm_cache.get(combined_hash)
        if embedding is not None:
            self.mm_cache.move_to_end(combined_hash)
        return embedding

    def set(
        self,
        mm_hash: int,
        embedding: EmbeddingResult,
        loc: Optional[torch.Tensor] = None,
    ) -> bool:
        assert isinstance(embedding, EmbeddingResult), embedding
        if mm_hash in self.mm_cache:
            self.mm_cache.move_to_end(mm_hash)
            return True
        data_size = _get_tensor_size(embedding.embedding)
        # Evict unpinned entries to make room
        while self.current_size + data_size > self.max_size:
            evicted = self._evict_one()
            if not evicted:
                return False

        self.mm_cache[mm_hash] = embedding
        self.current_size += data_size
        return True

    def set_pinned(self, mm_hash: int, embedding: EmbeddingResult) -> bool:
        """Store an embedding as a pinned entry, bypassing max_size constraints.

        If the key already exists and is pinned, just move to end (no refcount change).
        If the key already exists but is not pinned, transfer its size to pinned_size.
        """
        assert isinstance(embedding, EmbeddingResult), embedding
        if mm_hash in self.mm_cache:
            self.mm_cache.move_to_end(mm_hash)
            if mm_hash not in self.pinned_refcount:
                # Transfer from unpinned to pinned
                data_size = _get_tensor_size(self.mm_cache[mm_hash].embedding)
                self.current_size -= data_size
                self.pinned_size += data_size
                self.pinned_refcount[mm_hash] = 0
            return True

        data_size = _get_tensor_size(embedding.embedding)
        self.mm_cache[mm_hash] = embedding
        self.pinned_size += data_size
        self.pinned_refcount[mm_hash] = 0
        return True

    def pin(self, mm_hash: int) -> bool:
        """Pin an existing cache entry. Increments refcount.

        Returns False if the hash is not in the cache.
        """
        if mm_hash not in self.mm_cache:
            return False
        if mm_hash in self.pinned_refcount:
            self.pinned_refcount[mm_hash] += 1
        else:
            # Transfer from unpinned to pinned
            data_size = _get_tensor_size(self.mm_cache[mm_hash].embedding)
            self.current_size -= data_size
            self.pinned_size += data_size
            self.pinned_refcount[mm_hash] = 1
        return True

    def unpin(self, mm_hash: int) -> bool:
        """Unpin a cache entry. Decrements refcount.

        When refcount reaches 0, the entry becomes evictable again.
        Returns False if the hash is not pinned.
        """
        if mm_hash not in self.pinned_refcount:
            return False
        self.pinned_refcount[mm_hash] -= 1
        if self.pinned_refcount[mm_hash] <= 0:
            del self.pinned_refcount[mm_hash]
            if mm_hash in self.mm_cache:
                data_size = _get_tensor_size(self.mm_cache[mm_hash].embedding)
                self.pinned_size -= data_size
                self.current_size += data_size
            self._evict_if_needed()
        return True

    def _evict_one(self) -> bool:
        """Evict the least-recently-used unpinned entry. Returns True if one was evicted."""
        for key in self.mm_cache:
            if key not in self.pinned_refcount:
                entry = self.mm_cache.pop(key)
                self.current_size -= _get_tensor_size(entry.embedding)
                return True
        return False

    def _evict_if_needed(self):
        """Evict unpinned entries until current_size <= max_size."""
        while self.current_size > self.max_size:
            if not self._evict_one():
                break

    def has(self, mm_hash: int) -> bool:
        return mm_hash in self.mm_cache

    def free(
        self, mm_hash: int, mm_embedding_allocator: BaseTokenToKVPoolAllocator
    ) -> bool:
        if mm_hash not in self.mm_cache:
            return False
        old_embedding = self.mm_cache.pop(mm_hash)
        data_size = _get_tensor_size(old_embedding.embedding)
        if mm_hash in self.pinned_refcount:
            self.pinned_size -= data_size
            del self.pinned_refcount[mm_hash]
        else:
            self.current_size -= data_size
        return True

    def clear(self):
        self.mm_cache.clear()
        self.current_size = 0
        self.pinned_refcount.clear()
        self.pinned_size = 0

    def __len__(self):
        return len(self.mm_cache)

    def available_size(self):
        return self.__len__()
