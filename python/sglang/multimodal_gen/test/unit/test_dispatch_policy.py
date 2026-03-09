# SPDX-License-Identifier: Apache-2.0
"""Unit tests for disagg dispatch policies."""

import unittest

from sglang.multimodal_gen.runtime.disaggregation.dispatch_policy import (
    MaxFreeSlotsFirst,
    RoundRobin,
    create_dispatch_policy,
)


class TestRoundRobin(unittest.TestCase):
    """Test RoundRobin dispatch policy."""

    def test_cycles_through_instances(self):
        policy = RoundRobin(num_instances=3)
        results = [policy.select() for _ in range(6)]
        self.assertEqual(results, [0, 1, 2, 0, 1, 2])

    def test_single_instance(self):
        policy = RoundRobin(num_instances=1)
        results = [policy.select() for _ in range(3)]
        self.assertEqual(results, [0, 0, 0])

    def test_ignores_active_counts(self):
        policy = RoundRobin(num_instances=2)
        # active_counts should be ignored
        self.assertEqual(policy.select(active_counts=[100, 0]), 0)
        self.assertEqual(policy.select(active_counts=[0, 100]), 1)

    def test_invalid_num_instances(self):
        with self.assertRaises(ValueError):
            RoundRobin(num_instances=0)


class TestMaxFreeSlotsFirst(unittest.TestCase):
    """Test MaxFreeSlotsFirst dispatch policy."""

    def test_selects_least_loaded(self):
        policy = MaxFreeSlotsFirst(num_instances=3, max_slots_per_instance=4)
        # Instance 2 has the most free slots (4-0=4)
        result = policy.select(active_counts=[2, 3, 0])
        self.assertEqual(result, 2)

    def test_selects_least_loaded_all_busy(self):
        policy = MaxFreeSlotsFirst(num_instances=3, max_slots_per_instance=2)
        # Instance 0 has 1 free slot, others have 0
        result = policy.select(active_counts=[1, 2, 2])
        self.assertEqual(result, 0)

    def test_fallback_without_counts(self):
        policy = MaxFreeSlotsFirst(num_instances=3, max_slots_per_instance=2)
        # Without active_counts, falls back to round-robin
        results = [policy.select() for _ in range(3)]
        self.assertEqual(results, [0, 1, 2])

    def test_all_at_capacity(self):
        policy = MaxFreeSlotsFirst(num_instances=2, max_slots_per_instance=1)
        # Both at capacity — should still pick one
        result = policy.select(active_counts=[1, 1])
        self.assertIn(result, [0, 1])

    def test_tie_breaking(self):
        policy = MaxFreeSlotsFirst(num_instances=3, max_slots_per_instance=4)
        # All equal — should not always pick the same
        results = set()
        for _ in range(10):
            results.add(policy.select(active_counts=[0, 0, 0]))
        # With tie-breaking, should see at least 2 different instances
        self.assertGreater(len(results), 1)


class TestCreateDispatchPolicy(unittest.TestCase):
    """Test factory function."""

    def test_round_robin(self):
        policy = create_dispatch_policy("round_robin", num_instances=2)
        self.assertIsInstance(policy, RoundRobin)

    def test_max_free_slots(self):
        policy = create_dispatch_policy(
            "max_free_slots", num_instances=2, max_slots_per_instance=3
        )
        self.assertIsInstance(policy, MaxFreeSlotsFirst)

    def test_unknown_policy_raises(self):
        with self.assertRaises(ValueError):
            create_dispatch_policy("unknown", num_instances=2)


if __name__ == "__main__":
    unittest.main()
