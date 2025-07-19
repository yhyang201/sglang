import unittest

import torch

from sglang.srt.mem_cache.multimodal_cache import PagedMultiModalCache
from sglang.test.test_utils import CustomTestCase


class TestPagedMMCache(CustomTestCase):

    def test_correctness(self):
        """
        A short code snippet to verify the functionality of PagedMultiModalCache.
        """
        # 1. Setup parameters
        device = "cuda" if torch.cuda.is_available() else "cpu"
        num_token_slots = 1024
        hidden_size = 128
        page_size = 16
        dtype = torch.float16

        print("--- 1. Initializing PagedMultiModalCache ---")
        cache = PagedMultiModalCache(
            size=num_token_slots,
            hidden_size=hidden_size,
            page_size=page_size,
            dtype=dtype,
            device=device,
        )
        print(f"Cache created on device: '{device}'")
        print(f"Buffer shape: {cache.mm_buffer.shape}\n")

        # 2. Simulate allocation for a new embedding
        print("--- 2. Simulating allocation from an allocator ---")
        embedding_len = 5
        # Let's assume the allocator gives us these non-contiguous token indices
        allocated_locs = torch.tensor([42, 15, 1, 888, 233], device=device)
        print(f"A new embedding (hash=12345) needs {embedding_len} tokens.")
        print(
            f"Allocator assigned non-contiguous locations: {allocated_locs.cpu().tolist()}\n"
        )

        # 3. Get pointers for the allocated locations
        print("--- 3. Verifying get_pointers_from_locs ---")
        pointers = cache.get_pointers_from_locs(allocated_locs)
        print(f"Retrieved {len(pointers)} pointers from the cache.")
        print(f"First pointer: {pointers[0]}")
        print(f"Pointers tensor (first 5): {pointers[:5].cpu().tolist()}\n")
        # A simple check to see if pointers are calculated correctly
        expected_first_ptr = (
            cache.mm_buffer.data_ptr()
            + allocated_locs[0] * hidden_size * cache.mm_buffer.element_size()
        )
        assert pointers[0] == expected_first_ptr, "Pointer calculation seems incorrect"
        print("Pointer calculation sanity check passed.\n")

        # 4. Create a dummy embedding and write to cache
        print("--- 4. Writing embedding to the cache ---")
        mm_hash = 12345
        dummy_embedding = torch.randn(
            embedding_len, hidden_size, dtype=dtype, device=device
        )
        is_success = cache.set_mm_embedding(
            mm_hash=mm_hash, embedding=dummy_embedding, loc=allocated_locs
        )
        assert is_success, "set_mm_embedding failed"
        print(f"Successfully called set_mm_embedding for hash {mm_hash}.\n")

        # 5. Read the embedding back
        print("--- 5. Reading embedding back from the cache ---")
        retrieved_embedding = cache.get_mm_embedding(mm_hash)
        print(f"Retrieved embedding of shape: {retrieved_embedding.shape}\n")

        # 6. Verification
        print("--- 6. Verifying correctness ---")
        are_equal = torch.allclose(dummy_embedding, retrieved_embedding)
        print(f"Is the retrieved embedding identical to the original? -> {are_equal}")
        assert (
            are_equal
        ), "Verification failed: retrieved embedding does not match the original."
        print("\nVerification successful!")


if __name__ == "__main__":
    unittest.main()
