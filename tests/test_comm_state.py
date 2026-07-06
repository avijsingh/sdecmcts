import random

import pytest

from semidec.comm_state import (
    CommState,
    SemiMarkovCommModel,
    can_communicate,
    full_partition,
    group_of,
    isolated_partition,
    normalize_partition,
)


def test_partition_helpers_are_canonical_and_queryable():
    partition = normalize_partition(((1, 0), (2,)))

    assert partition == frozenset({frozenset({0, 1}), frozenset({2})})
    assert full_partition([0, 1, 2]) == frozenset({frozenset({0, 1, 2})})
    assert isolated_partition([0, 1]) == frozenset({frozenset({0}), frozenset({1})})
    assert group_of(partition, 0) == frozenset({0, 1})
    assert can_communicate(partition, 0, 1)
    assert not can_communicate(partition, 0, 2)


def test_partition_validation_rejects_overlap_and_missing_members():
    with pytest.raises(ValueError):
        normalize_partition(((0, 1), (1, 2)))

    with pytest.raises(ValueError):
        SemiMarkovCommModel(
            robot_ids=[0, 1, 2],
            partition_probs={((0, 1),): 1.0},
            sojourn_probs={1: 1.0},
        )


def test_sojourn_counts_down_before_sampling_new_partition():
    model = SemiMarkovCommModel(
        robot_ids=[0, 1],
        partition_probs={((0,), (1,)): 1.0},
        sojourn_probs={1: 1.0},
        initial_partition=((0, 1),),
        initial_sojourn=3,
        rng=random.Random(7),
    )

    state = model.initial_state()
    assert state == CommState(
        partition=frozenset({frozenset({0, 1})}),
        sojourn_remaining=3,
    )

    state = model.step(state)
    assert state.partition == full_partition([0, 1])
    assert state.sojourn_remaining == 2

    state = model.step(state)
    assert state.partition == full_partition([0, 1])
    assert state.sojourn_remaining == 1

    state = model.step(state)
    assert state.partition == isolated_partition([0, 1])
    assert state.sojourn_remaining == 1


def test_sampling_is_reproducible_with_seeded_rng():
    def rollout():
        model = SemiMarkovCommModel(
            robot_ids=[0, 1],
            partition_probs={
                ((0, 1),): 0.25,
                ((0,), (1,)): 0.75,
            },
            sojourn_probs={1: 0.6, 2: 0.4},
            rng=random.Random(11),
        )
        state = model.initial_state()
        out = [state]
        for _ in range(8):
            state = model.step(state)
            out.append(state)
        return out

    assert rollout() == rollout()


def test_comm_state_requires_positive_sojourn():
    with pytest.raises(ValueError):
        CommState(partition=full_partition([0, 1]), sojourn_remaining=0)
