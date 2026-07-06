from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, FrozenSet, Hashable, Iterable, Mapping, Optional, Sequence


RobotID = Hashable
CommGroup = FrozenSet[RobotID]
CommPartition = FrozenSet[CommGroup]


def normalize_partition(partition: Iterable[Iterable[RobotID]]) -> CommPartition:
    """Return a canonical immutable communication partition."""
    groups = []
    seen = set()
    for group in partition:
        frozen = frozenset(group)
        if not frozen:
            raise ValueError("Communication groups cannot be empty.")
        overlap = seen.intersection(frozen)
        if overlap:
            raise ValueError(f"Robot(s) appear in multiple communication groups: {overlap}")
        seen.update(frozen)
        groups.append(frozen)
    if not groups:
        raise ValueError("A communication partition must contain at least one group.")
    return frozenset(groups)


def full_partition(robot_ids: Sequence[RobotID]) -> CommPartition:
    return normalize_partition([robot_ids])


def isolated_partition(robot_ids: Sequence[RobotID]) -> CommPartition:
    return normalize_partition((rid,) for rid in robot_ids)


def group_of(partition: CommPartition, robot_id: RobotID) -> CommGroup:
    for group in partition:
        if robot_id in group:
            return group
    raise KeyError(f"Robot {robot_id!r} is not present in partition {partition!r}.")


def can_communicate(partition: CommPartition, left: RobotID, right: RobotID) -> bool:
    return right in group_of(partition, left)


@dataclass(frozen=True)
class CommState:
    """
    Markov communication state for one planning/execution step.

    `partition` tells which robots can communicate now. `sojourn_remaining`
    counts how many steps, including the current one, this partition persists.
    """

    partition: CommPartition
    sojourn_remaining: int

    def __post_init__(self) -> None:
        if self.sojourn_remaining < 1:
            raise ValueError("sojourn_remaining must be at least 1.")

    def group_of(self, robot_id: RobotID) -> CommGroup:
        return group_of(self.partition, robot_id)

    def can_communicate(self, left: RobotID, right: RobotID) -> bool:
        return can_communicate(self.partition, left, right)

    def is_fully_connected(self, robot_ids: Sequence[RobotID]) -> bool:
        return self.partition == full_partition(robot_ids)


class SemiMarkovCommModel:
    """
    Semi-Markov model over communication partitions.

    The model samples a partition, samples how long it persists, and then
    decrements the sojourn timer on each `step`. When the timer expires, a new
    partition and duration are sampled.
    """

    def __init__(
        self,
        robot_ids: Sequence[RobotID],
        partition_probs: Mapping[Iterable[Iterable[RobotID]], float],
        sojourn_probs: Mapping[int, float],
        *,
        initial_partition: Optional[Iterable[Iterable[RobotID]]] = None,
        initial_sojourn: Optional[int] = None,
        rng: Optional[random.Random] = None,
    ):
        self.robot_ids = tuple(robot_ids)
        if not self.robot_ids:
            raise ValueError("robot_ids cannot be empty.")

        self.partition_probs = self._normalize_partition_probs(partition_probs)
        self.sojourn_probs = self._normalize_numeric_probs(sojourn_probs, "sojourn")
        if any(duration < 1 for duration in self.sojourn_probs):
            raise ValueError("Sojourn durations must be positive integers.")

        self.initial_partition = (
            normalize_partition(initial_partition)
            if initial_partition is not None
            else None
        )
        if self.initial_partition is not None:
            self._validate_partition_membership(self.initial_partition)

        if initial_sojourn is not None and initial_sojourn < 1:
            raise ValueError("initial_sojourn must be at least 1.")
        self.initial_sojourn = initial_sojourn
        self.rng = rng or random.Random()

    def initial_state(self) -> CommState:
        partition = self.initial_partition or self.sample_partition()
        sojourn = self.initial_sojourn or self.sample_sojourn()
        return CommState(partition=partition, sojourn_remaining=sojourn)

    def step(self, state: CommState) -> CommState:
        self._validate_partition_membership(state.partition)
        if state.sojourn_remaining > 1:
            return CommState(
                partition=state.partition,
                sojourn_remaining=state.sojourn_remaining - 1,
            )
        return CommState(
            partition=self.sample_partition(),
            sojourn_remaining=self.sample_sojourn(),
        )

    def sample_partition(self) -> CommPartition:
        return self._sample_from_dist(self.partition_probs)

    def sample_sojourn(self) -> int:
        return self._sample_from_dist(self.sojourn_probs)

    def _normalize_partition_probs(
        self,
        probs: Mapping[Iterable[Iterable[RobotID]], float],
    ) -> Dict[CommPartition, float]:
        normalized = {
            normalize_partition(partition): float(prob)
            for partition, prob in probs.items()
        }
        for partition in normalized:
            self._validate_partition_membership(partition)
        return self._normalize_numeric_probs(normalized, "partition")

    def _validate_partition_membership(self, partition: CommPartition) -> None:
        members = frozenset().union(*partition)
        expected = frozenset(self.robot_ids)
        if members != expected:
            raise ValueError(
                f"Partition members {members!r} do not match robot_ids {expected!r}."
            )

    @staticmethod
    def _normalize_numeric_probs(
        probs: Mapping[object, float],
        label: str,
    ) -> Dict[object, float]:
        if not probs:
            raise ValueError(f"{label} probability distribution cannot be empty.")
        if any(float(prob) < 0.0 for prob in probs.values()):
            raise ValueError(f"{label} probabilities cannot be negative.")
        total = sum(float(prob) for prob in probs.values())
        if total <= 0.0:
            raise ValueError(f"{label} probability distribution must have positive mass.")
        return {key: float(prob) / total for key, prob in probs.items()}

    def _sample_from_dist(self, dist: Mapping[object, float]):
        r = self.rng.random()
        cumulative = 0.0
        last_key = None
        for key, prob in dist.items():
            last_key = key
            cumulative += prob
            if r <= cumulative:
                return key
        return last_key
