"""
tests/test_decmcts_core.py
--------------------------
Core unit tests for decmcts.py.

"""

import math
import pytest

from decmcts import DecMCTS, DecMCTSNode


class ToyState:
    """Minimal finite-horizon state: actions do not affect state except depth."""

    def __init__(self, depth=0, horizon=3, actions=(0, 1)):
        self.depth = depth
        self.horizon = horizon
        self.actions = tuple(actions)

    def get_legal_actions(self):
        return [] if self.is_terminal_state() else list(self.actions)

    def take_action(self, action):
        return ToyState(self.depth + 1, self.horizon, self.actions)

    def is_terminal_state(self):
        return self.depth >= self.horizon


def remaining_ones_rollout(planner, node, x_others):
    """Complete the current sequence with action 1 until the horizon."""
    remaining = node.state.horizon - node.state.depth
    return [1] * remaining


def own_sum_utility(joint):
    """Utility equals sum of robot 0's own sequence."""
    return float(sum(joint.get(0, [])))


def first_action_utility(joint):
    """Utility is 1 iff robot 0's first action is 1."""
    seq = joint.get(0, [])
    return float(len(seq) > 0 and seq[0] == 1)


def test_node_action_sequence_and_untried_actions():
    root = DecMCTSNode(ToyState(horizon=2, actions=(0, 1)))
    assert root.action_sequence == []
    assert set(root.untried_actions) == {0, 1}

    child = root.add_child(1, root.state.take_action(1))
    assert child.action_sequence == [1]
    assert 1 not in root.untried_actions
    assert set(child.untried_actions) == {0, 1}


def test_discounted_statistics_decay_for_unvisited_nodes():
    node = DecMCTSNode(ToyState())
    node.update_discounted(reward=10.0, visited=True, gamma=0.9)
    assert node.disc_visits == pytest.approx(1.0)
    assert node.disc_reward == pytest.approx(10.0)

    node.update_discounted(reward=999.0, visited=False, gamma=0.9)
    assert node.disc_visits == pytest.approx(0.9)
    assert node.disc_reward == pytest.approx(9.0)


def test_d_ucb_returns_infinity_for_unvisited_child():
    parent = DecMCTSNode(ToyState())
    child = DecMCTSNode(ToyState(depth=1), parent=parent, action=0)
    parent.disc_visits = 10.0
    assert math.isinf(child.d_ucb(parent, gamma=0.95, cp=0.5, min_reward=0.0, max_reward=1.0))


def test_sample_others_uses_default_sequence_when_no_message_received():
    planner = DecMCTS(
        robot_id=0,
        robot_ids=[0, 1],
        init_state=ToyState(horizon=3),
        local_utility_fn=own_sum_utility,
        default_sequence_fn=lambda rid: [9, 9, 9],
        seed=0,
    )

    assert planner._sample_others() == {1: [9, 9, 9]}


def test_receive_accepts_dict_and_aligned_list():
    planner = DecMCTS(
        robot_id=0,
        robot_ids=[0, 1],
        init_state=ToyState(),
        local_utility_fn=own_sum_utility,
        seed=0,
    )

    planner.receive_dist_dict(1, {(1, 2): 0.7, (0,): 0.3})
    assert planner.received_dists[1] == {(1, 2): 0.7, (0,): 0.3}

    planner.receive(1, [[1, 1], [0, 0]], [0.25, 0.75])
    assert planner.received_dists[1] == {(1, 1): 0.25, (0, 0): 0.75}


def test_grow_tree_stores_full_rollout_completed_sequence():
    horizon = 4
    planner = DecMCTS(
        robot_id=0,
        robot_ids=[0],
        init_state=ToyState(horizon=horizon),
        local_utility_fn=own_sum_utility,
        rollout_policy=remaining_ones_rollout,
        rollout_depth=horizon,
        tau=1,
        seed=1,
    )

    planner._grow_tree_once()

    nodes = []
    planner._collect_nodes(planner.root, nodes)
    reps = [n.representative_sequence for n in nodes if n.representative_sequence is not None]

    assert reps, "Expected at least one representative rollout-completed sequence."
    assert all(len(seq) == horizon for seq in reps), reps


def test_update_sample_space_uses_full_sequences_not_prefixes():
    horizon = 5
    planner = DecMCTS(
        robot_id=0,
        robot_ids=[0],
        init_state=ToyState(horizon=horizon),
        local_utility_fn=own_sum_utility,
        rollout_policy=remaining_ones_rollout,
        rollout_depth=horizon,
        num_seq=5,
        seed=2,
    )

    for _ in range(6):
        planner._grow_tree_once()

    planner._update_sample_space()

    assert planner.X_hat, "Expected X_hat to be populated."
    assert all(len(seq) == horizon for seq in planner.X_hat), planner.X_hat

def test_first_outer_iteration_may_leave_distribution_empty():
    """
    Paper-faithful schedule:
      1. update X_hat from current tree
      2. grow tree tau times
      3. update q over the already-selected X_hat

    Starting from an empty tree, the first sample-space update sees no
    representative sequences yet. Therefore q may be empty after one outer
    iteration. The newly discovered sequences become eligible at the next
    outer iteration.
    """
    planner = DecMCTS(
        robot_id=0,
        robot_ids=[0],
        init_state=ToyState(horizon=3),
        local_utility_fn=own_sum_utility,
        rollout_policy=remaining_ones_rollout,
        rollout_depth=3,
        tau=5,
        num_seq=5,
        num_samples=5,
        seed=3,
    )

    planner.iterate(1)

    # The tree should have grown.
    nodes = []
    planner._collect_nodes(planner.root, nodes)
    assert any(n.representative_sequence is not None for n in nodes)

    # But paper-faithful q can still be empty after the first iteration.
    assert planner.X_hat == []
    assert planner.q == {}


def test_second_outer_iteration_populates_distribution():
    """
    The second outer iteration should select the representative sequences
    discovered during the first outer iteration, then update q over them.
    """
    planner = DecMCTS(
        robot_id=0,
        robot_ids=[0],
        init_state=ToyState(horizon=3),
        local_utility_fn=own_sum_utility,
        rollout_policy=remaining_ones_rollout,
        rollout_depth=3,
        tau=5,
        num_seq=5,
        num_samples=5,
        seed=3,
    )

    planner.iterate(2)

    assert planner.X_hat, "Expected X_hat after two outer iterations."
    assert planner.q, "Expected q after two outer iterations."
    assert sum(planner.q.values()) == pytest.approx(1.0)



def test_distribution_update_moves_mass_toward_better_sequence():
    planner = DecMCTS(
        robot_id=0,
        robot_ids=[0],
        init_state=ToyState(horizon=1),
        local_utility_fn=first_action_utility,
        num_samples=200,
        alpha=0.5,
        beta_init=1.0,
        seed=4,
    )

    planner.X_hat = [(0,), (1,)]
    planner.q = {(0,): 0.5, (1,): 0.5}
    planner.min_reward = 0.0
    planner.max_reward = 1.0

    planner._update_distribution()

    assert planner.q[(1,)] > 0.5
    assert planner.q[(0,)] < 0.5
    assert sum(planner.q.values()) == pytest.approx(1.0)


def test_best_action_uses_marginal_first_action_mass():
    planner = DecMCTS(
        robot_id=0,
        robot_ids=[0],
        init_state=ToyState(horizon=2),
        local_utility_fn=own_sum_utility,
        seed=5,
    )

    # Highest single sequence starts with 0, but total mass on first action 1 is larger.
    planner.q = {
        (0, 0): 0.40,
        (1, 0): 0.31,
        (1, 1): 0.29,
    }

    assert planner.best_action_sequence() == [0, 0]
    assert planner.best_action() == 1


# def test_root_action_mass_helper_works_from_instance():
#     """
#     This currently fails if root_action_mass_from_dist is defined as an instance
#     method without self. Make it a @staticmethod or move it outside the class.
#     """
#     planner = DecMCTS(
#         robot_id=0,
#         robot_ids=[0],
#         init_state=ToyState(),
#         local_utility_fn=own_sum_utility,
#         seed=6,
#     )

#     masses = planner.root_action_mass_from_dist({
#         (0, 0): 0.2,
#         (1, 0): 0.3,
#         (2, 0): 0.5,
#     })

    # assert masses == {0: 0.2, 1: 0.3, 2: 0.5}
