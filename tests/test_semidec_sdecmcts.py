import random
from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import pytest

from semidec.comm_state import SemiMarkovCommModel, full_partition, isolated_partition
from semidec.sdecmcts import ActionStats, BeliefNode, SDecMCTS, StepResult


@dataclass(frozen=True)
class RevealingBitModel:
    n_actions_per_agent: int = 2

    def sample_state_from_belief(self, belief: Sequence[float], rng: random.Random) -> int:
        return 0 if rng.random() < belief[0] else 1

    def legal_actions(self, _belief, _robot_id, _depth):
        return [0, 1]

    def joint_action_from_dict(self, actions: Dict[int, int]) -> Tuple[int, int]:
        return actions[0], actions[1]

    def split_obs(self, joint_obs: Tuple[int, int]) -> Tuple[int, int]:
        return joint_obs

    def step(self, state: int, joint_action: Tuple[int, int], _rng: random.Random) -> StepResult:
        reward = 1.0 if joint_action == (state, state) else 0.0
        return StepResult(
            next_state=state,
            joint_obs=(state, state),
            reward=reward,
        )

    def update_joint_belief(
        self,
        _belief,
        _joint_action: Tuple[int, int],
        joint_obs: Tuple[int, int],
    ):
        out = [0.0, 0.0]
        out[joint_obs[0]] = 1.0
        return out


def make_comm_model(initial_sojourn=2):
    return SemiMarkovCommModel(
        robot_ids=[0, 1],
        partition_probs={((0,), (1,)): 1.0},
        sojourn_probs={1: 1.0},
        initial_partition=((0, 1),),
        initial_sojourn=initial_sojourn,
        rng=random.Random(5),
    )


def make_planner(horizon=2, initial_sojourn=2):
    return SDecMCTS(
        robot_ids=[0, 1],
        root_belief=[0.5, 0.5],
        model=RevealingBitModel(),
        comm_model=make_comm_model(initial_sojourn=initial_sojourn),
        horizon=horizon,
        seed=13,
    )


def test_centralized_tree_grows_and_carries_comm_state():
    planner = make_planner(horizon=2, initial_sojourn=2)
    planner.run(30)

    assert planner.root.visits == 30
    assert planner.root.actions
    assert planner.root.comm_state.partition == full_partition([0, 1])

    children = [
        child
        for edge in planner.root.actions.values()
        for child in edge.obs_children.values()
    ]
    assert children
    assert all(child.depth == 1 for child in children)
    assert all(child.comm_state.sojourn_remaining == 1 for child in children)
    assert all(child.comm_state.partition == full_partition([0, 1]) for child in children)


def test_comm_state_resamples_when_sojourn_expires():
    planner = make_planner(horizon=2, initial_sojourn=1)
    planner.run(10)

    children = [
        child
        for edge in planner.root.actions.values()
        for child in edge.obs_children.values()
    ]
    assert children
    assert all(child.comm_state.partition == isolated_partition([0, 1]) for child in children)


def test_extract_policy_projects_centralized_tree_to_local_histories():
    planner = make_planner(horizon=2, initial_sojourn=2)

    root_edge = ActionStats((0, 0))
    root_edge.visits = 20
    root_edge.value_sum = 10.0
    planner.root.actions[(0, 0)] = root_edge

    child_zero = BeliefNode(
        belief=[1.0, 0.0],
        depth=1,
        histories={0: ((0, 0),), 1: ((0, 0),)},
        comm_state=planner.comm_model.step(planner.root.comm_state),
        legal_joint_actions=[(0, 0), (1, 1)],
    )
    child_one = BeliefNode(
        belief=[0.0, 1.0],
        depth=1,
        histories={0: ((0, 1),), 1: ((0, 1),)},
        comm_state=planner.comm_model.step(planner.root.comm_state),
        legal_joint_actions=[(0, 0), (1, 1)],
    )
    root_edge.obs_children[(0, 0)] = child_zero
    root_edge.obs_children[(1, 1)] = child_one

    zero_good = ActionStats((0, 0))
    zero_good.visits = 10
    zero_good.value_sum = 10.0
    zero_bad = ActionStats((1, 1))
    zero_bad.visits = 10
    zero_bad.value_sum = 0.0
    child_zero.actions = {(0, 0): zero_good, (1, 1): zero_bad}
    child_zero.visits = 20

    one_good = ActionStats((1, 1))
    one_good.visits = 10
    one_good.value_sum = 10.0
    one_bad = ActionStats((0, 0))
    one_bad.visits = 10
    one_bad.value_sum = 0.0
    child_one.actions = {(1, 1): one_good, (0, 0): one_bad}
    child_one.visits = 20

    planner._all_nodes = [planner.root, child_zero, child_one]

    policy = planner.extract_policy(default_actions={0: 0, 1: 0})

    assert policy.action(0, 1, ((0, 0),)) == 0
    assert policy.action(1, 1, ((0, 0),)) == 0
    assert policy.action(0, 1, ((0, 1),)) == 1
    assert policy.action(1, 1, ((0, 1),)) == 1


def test_extracted_policy_executes_without_replanning_between_syncs():
    planner = make_planner(horizon=2, initial_sojourn=2)
    planner.run(80)
    policy = planner.extract_policy(default_actions={0: 0, 1: 0})

    model = RevealingBitModel()
    histories = {0: tuple(), 1: tuple()}
    true_state = 1

    root_actions = policy.joint_action_from_histories([0, 1], 0, histories)
    joint_action = model.joint_action_from_dict(root_actions)
    step = model.step(true_state, joint_action, random.Random(99))
    local_obs = model.split_obs(step.joint_obs)

    for rid in [0, 1]:
        histories[rid] = histories[rid] + ((root_actions[rid], local_obs[rid]),)

    followup_actions = policy.joint_action_from_histories([0, 1], 1, histories)
    assert followup_actions == {0: 1, 1: 1}


def test_best_joint_action_prefers_root_value_before_visits():
    planner = make_planner(horizon=1)
    edge_a = ActionStats((0, 0))
    edge_a.visits = 2
    edge_a.value_sum = 2.0
    edge_b = ActionStats((1, 1))
    edge_b.visits = 5
    edge_b.value_sum = 1.0
    planner.root.actions = {(0, 0): edge_a, (1, 1): edge_b}

    assert planner.best_joint_action() == {0: 0, 1: 0}


def test_extraction_forces_best_joint_action_at_synchronized_root():
    planner = make_planner(horizon=2)
    good_root = ActionStats((1, 1))
    good_root.visits = 10
    good_root.value_sum = 100.0
    bad_root = ActionStats((0, 1))
    bad_root.visits = 3
    bad_root.value_sum = 0.0
    planner.root.actions = {
        (1, 1): good_root,
        (0, 1): bad_root,
    }

    child = BeliefNode(
        belief=[1.0, 0.0],
        depth=1,
        histories={0: ((1, 0),), 1: ((1, 0),)},
        comm_state=planner.comm_model.step(planner.root.comm_state),
        legal_joint_actions=[(0, 1)],
    )
    deep_edge = ActionStats((0, 1))
    deep_edge.visits = 100
    deep_edge.value_sum = 1000.0
    child.actions = {(0, 1): deep_edge}
    child.visits = 100
    planner._all_nodes = [planner.root, child]

    policy = planner.extract_policy(default_actions={0: 0, 1: 0})

    assert policy.action(0, 0, tuple()) == 1
    assert policy.action(1, 0, tuple()) == 1
