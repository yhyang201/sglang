# SPDX-License-Identifier: Apache-2.0
"""
Dispatch policies for multi-instance disaggregated diffusion pipelines.

Selects which role instance should handle the next request based on
load balancing criteria.

Usage:
    policy = MaxFreeSlotsFirst(num_instances=3, max_slots_per_instance=2)
    instance_id = policy.select(tracker)

    # Or simpler:
    policy = RoundRobin(num_instances=3)
    instance_id = policy.select()
"""

import abc
import logging
import threading

logger = logging.getLogger(__name__)


class DispatchPolicy(abc.ABC):
    """Base class for dispatch policies."""

    def __init__(self, num_instances: int):
        if num_instances < 1:
            raise ValueError(f"num_instances must be >= 1, got {num_instances}")
        self._num_instances = num_instances

    @property
    def num_instances(self) -> int:
        return self._num_instances

    @abc.abstractmethod
    def select(self, active_counts: list[int] | None = None) -> int:
        """Select the next instance to dispatch to.

        Args:
            active_counts: Optional list of active request counts per instance.
                Length must equal num_instances. Used by load-aware policies.

        Returns:
            Instance index (0-based).
        """
        ...

    def record_completion(self, instance_id: int) -> None:
        """Notify the policy that an instance completed a request.

        Override in subclasses that track per-instance state.
        """
        pass


class RoundRobin(DispatchPolicy):
    """Simple round-robin dispatch across instances.

    Thread-safe. Ignores load information.
    """

    def __init__(self, num_instances: int):
        super().__init__(num_instances)
        self._lock = threading.Lock()
        self._next = 0

    def select(self, active_counts: list[int] | None = None) -> int:
        with self._lock:
            chosen = self._next
            self._next = (self._next + 1) % self._num_instances
        return chosen


class MaxFreeSlotsFirst(DispatchPolicy):
    """Dispatch to the instance with the most free slots.

    Requires active_counts to be provided. Falls back to the instance
    with the fewest active requests if max_slots is not set.

    Thread-safe.
    """

    def __init__(self, num_instances: int, max_slots_per_instance: int = 1):
        super().__init__(num_instances)
        self._max_slots = max_slots_per_instance
        self._lock = threading.Lock()
        # Fallback counter for tie-breaking
        self._tiebreak = 0

    def select(self, active_counts: list[int] | None = None) -> int:
        with self._lock:
            if active_counts is None or len(active_counts) != self._num_instances:
                # Fallback to round-robin if no load info
                chosen = self._tiebreak % self._num_instances
                self._tiebreak += 1
                return chosen

            # Find instance with most free slots
            best_id = 0
            best_free = self._max_slots - active_counts[0]
            for i in range(1, self._num_instances):
                free = self._max_slots - active_counts[i]
                if free > best_free:
                    best_free = free
                    best_id = i
                elif free == best_free:
                    # Tie-break: prefer the one after tiebreak counter
                    if i == (self._tiebreak % self._num_instances):
                        best_id = i

            self._tiebreak += 1

            if best_free <= 0:
                logger.warning(
                    "All %d instances are at capacity (%d slots each), "
                    "dispatching to instance %d anyway",
                    self._num_instances,
                    self._max_slots,
                    best_id,
                )

            return best_id


class PoolDispatcher:
    """Wraps three independent dispatch policies for encoder/denoiser/decoder pools."""

    def __init__(
        self,
        num_encoders: int,
        num_denoisers: int,
        num_decoders: int,
        policy_name: str = "round_robin",
        **kwargs,
    ):
        self.encoder_policy = create_dispatch_policy(
            policy_name, num_encoders, **kwargs
        )
        self.denoiser_policy = create_dispatch_policy(
            policy_name, num_denoisers, **kwargs
        )
        self.decoder_policy = create_dispatch_policy(
            policy_name, num_decoders, **kwargs
        )

    def select_encoder(self, active_counts: list[int] | None = None) -> int:
        return self.encoder_policy.select(active_counts)

    def select_denoiser(self, active_counts: list[int] | None = None) -> int:
        return self.denoiser_policy.select(active_counts)

    def select_decoder(self, active_counts: list[int] | None = None) -> int:
        return self.decoder_policy.select(active_counts)


def create_dispatch_policy(name: str, num_instances: int, **kwargs) -> DispatchPolicy:
    """Factory function for dispatch policies.

    Args:
        name: Policy name — "round_robin" or "max_free_slots"
        num_instances: Number of instances to dispatch across
        **kwargs: Additional policy-specific arguments
    """
    policies = {
        "round_robin": RoundRobin,
        "max_free_slots": MaxFreeSlotsFirst,
    }
    cls = policies.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown dispatch policy '{name}'. Available: {list(policies.keys())}"
        )
    return cls(num_instances=num_instances, **kwargs)
