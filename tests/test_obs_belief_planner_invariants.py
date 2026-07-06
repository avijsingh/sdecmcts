"""
Core invariants for ObsDecMCTS and BeliefObsDecMCTS on tiny deterministic models.

These tests do not establish optimality. They verify structural properties that
must hold for technically sound planner execution: legal actions, normalized
policy distributions, deterministic seeding, depth bounds, normalized beliefs,
and communication of policy distributions.
"""

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import pytest

from obs_decmcts import ObsDecMCTS, ObsDecMCTSTeam, StepResult
from belief_obs_decmcts import BeliefObsDecMCTS, BeliefObsDecMCTSTeam


@dataclass(frozen=True)
class TinyModel:
    n_states: int = 2
    n_actions_per_agent: int = 2
    n_obs_per_agent: int = 2

    def sample_state_from_belief(self, belief: Sequence[float], rng) -> int:
        return 0 if rng.random() < belief[0] else 1

    def joint_action_from_dict(self, actions: Dict[int, int]) -> int:
        return actions[0] + self.n_actions_per_agent * actions[1]

    def split_action(self, joint_action: int) -> Tuple[int, int]:
        return (
            joint_action % self.n_actions_per_agent,
            joint_action // self.n_actions_per_agent,
        )

    def split_obs(self, joint_obs: int) -> Tuple[int, int]:
        return (
            joint_obs % self.n_obs_per_agent,
            joint_obs // self.n_obs_per_agent,
        )

    def step(self, state: int, joint_action: int, rng) -> StepResult:
        a0, a1 = self.split_action(joint_action)
        next_state = state ^ (1 if a0 != a1 else 0)
        joint_obs = next_state + self.n_obs_per_agent * next_state
        reward = 1.0 if a0 == state and a1 == state else 0.0
        return StepResult(next_state=next_state, joint_obs=joint_obs, reward=reward)

    def update_belief(self, belief, joint_action: int, local_obs: int, robot_id: int):
        del belief, joint_action, robot_id
        out = [0.0, 0.0]
        out[int(local_obs)] = 1.0
        return out


def _obs_legal_actions(_history, _depth):
    return [0, 1]


def _belief_legal_actions(_belief, _depth):
    return [0, 1]


def _obs_default(_history):
    return 0


def _belief_default(_belief):
    return 0


def _collect_obs_nodes(node, out):
    if id(node) in {id(n) for n in out}:
        return
    out.append(node)
    for edge in node.actions.values():
        for child in edge.obs_children.values():
            _collect_obs_nodes(child, out)


def _collect_belief_nodes(node, out, seen=None):
    seen = seen or set()
    if id(node) in seen:
        return
    seen.add(id(node))
    out.append(node)
    for edge in node.actions.values():
        for child in edge.obs_children.values():
            _collect_belief_nodes(child, out, seen)


def test_obs_decmcts_distribution_actions_and_depths_are_valid():
    model = TinyModel()
    planners = {
        rid: ObsDecMCTS(
            robot_id=rid,
            robot_ids=[0, 1],
            root_belief=[0.5, 0.5],
            model=model,
            legal_actions_fn=_obs_legal_actions,
            default_action_fn=_obs_default,
            default_action_fns_by_robot={0: _obs_default, 1: _obs_default},
            horizon=3,
            tau=4,
            num_policies=4,
            num_samples=3,
            seed=11 + rid,
        )
        for rid in [0, 1]
    }

    team = ObsDecMCTSTeam(planners)
    team.iterate_and_communicate(n_outer=3, comm_period=1)

    for rid, planner in planners.items():
        assert planner.X_hat
        assert planner.q
        assert sum(planner.q.values()) == pytest.approx(1.0)
        assert all(p >= 0.0 for p in planner.q.values())
        assert planner.best_action(()) in [0, 1]

        nodes = []
        _collect_obs_nodes(planner.root, nodes)
        assert all(0 <= node.depth <= planner.horizon for node in nodes)
        assert all(set(node.legal_actions).issubset({0, 1}) for node in nodes)

        other_id = 1 - rid
        assert planner.received_dists[other_id]


def test_obs_decmcts_distribution_update_moves_mass_to_better_policy():
    planner = ObsDecMCTS(
        robot_id=0,
        robot_ids=[0, 1],
        root_belief=[0.5, 0.5],
        model=TinyModel(),
        legal_actions_fn=_obs_legal_actions,
        default_action_fn=_obs_default,
        default_action_fns_by_robot={0: _obs_default, 1: _obs_default},
        horizon=1,
        tau=1,
        num_policies=2,
        num_samples=1,
        alpha=0.1,
        beta_init=2.0,
        seed=7,
    )
    good_key = ((((), 1)),)
    bad_key = ((((), 0)),)
    planner.X_hat = [good_key, bad_key]
    planner.q = {good_key: 0.5, bad_key: 0.5}
    planner.min_reward = 0.0
    planner.max_reward = 10.0

    def fixed_expectation(fixed_policy_key):
        if fixed_policy_key is None:
            return 5.0
        return 10.0 if fixed_policy_key == good_key else 0.0

    planner._estimate_expectation = fixed_expectation
    planner._update_distribution()

    assert planner.q[good_key] > 0.5
    assert planner.q[bad_key] < 0.5
    assert sum(planner.q.values()) == pytest.approx(1.0)


def test_obs_decmcts_value_initialized_q_prefers_higher_score():
    planner = ObsDecMCTS(
        robot_id=0,
        robot_ids=[0, 1],
        root_belief=[0.5, 0.5],
        model=TinyModel(),
        legal_actions_fn=_obs_legal_actions,
        default_action_fn=_obs_default,
        default_action_fns_by_robot={0: _obs_default, 1: _obs_default},
        horizon=1,
        tau=1,
        num_policies=2,
        num_samples=1,
        seed=11,
    )
    good_key = ((((), 1)),)
    bad_key = ((((), 0)),)

    q = planner._value_initialized_q(
        [good_key, bad_key],
        {good_key: 10.0, bad_key: 0.0},
    )

    assert q[good_key] > q[bad_key]
    assert sum(q.values()) == pytest.approx(1.0)


def test_belief_obs_decmcts_distribution_beliefs_and_depths_are_valid():
    model = TinyModel()
    planners = {
        rid: BeliefObsDecMCTS(
            robot_id=rid,
            robot_ids=[0, 1],
            root_belief=[0.5, 0.5],
            root_beliefs_by_robot={0: [0.5, 0.5], 1: [0.5, 0.5]},
            model=model,
            legal_actions_fn=_belief_legal_actions,
            default_action_fn=_belief_default,
            default_action_fns_by_robot={0: _belief_default, 1: _belief_default},
            horizon=3,
            tau=4,
            num_policies=4,
            num_samples=3,
            seed=23 + rid,
        )
        for rid in [0, 1]
    }

    team = BeliefObsDecMCTSTeam(planners)
    team.iterate_and_communicate(n_outer=3, comm_period=1)

    for rid, planner in planners.items():
        assert planner.X_hat
        assert planner.q
        assert sum(planner.q.values()) == pytest.approx(1.0)
        assert all(p >= 0.0 for p in planner.q.values())
        assert planner.best_action([0.5, 0.5]) in [0, 1]

        nodes = []
        _collect_belief_nodes(planner.root, nodes)
        assert all(0 <= node.depth <= planner.horizon for node in nodes)
        assert all(set(node.legal_actions).issubset({0, 1}) for node in nodes)
        for node in nodes:
            assert sum(node.belief) == pytest.approx(1.0)
            assert all(p >= 0.0 for p in node.belief)

        other_id = 1 - rid
        assert planner.received_dists[other_id]


def test_belief_obs_decmcts_repeated_seed_is_deterministic():
    def run_once():
        model = TinyModel()
        planner = BeliefObsDecMCTS(
            robot_id=0,
            robot_ids=[0, 1],
            root_belief=[0.5, 0.5],
            root_beliefs_by_robot={0: [0.5, 0.5], 1: [0.5, 0.5]},
            model=model,
            legal_actions_fn=_belief_legal_actions,
            default_action_fn=_belief_default,
            default_action_fns_by_robot={0: _belief_default, 1: _belief_default},
            horizon=3,
            tau=5,
            num_policies=4,
            num_samples=3,
            seed=99,
        )
        planner.iterate(3)
        return planner.best_action([0.5, 0.5]), planner.X_hat, planner.q

    a1, x1, q1 = run_once()
    a2, x2, q2 = run_once()

    assert a1 == a2
    assert x1 == x2
    assert q1 == q2
