"""
Validation tests for benchmarks/labyrinth_online.py.

These tests are deliberately model-level and deterministic. They are meant to
catch encoding, normalization, sync-propagation, reward/sink, and sparse-belief
regressions before running expensive MCTS sweeps.
"""

import random

import pytest

from benchmarks import labyrinth_online as lab


LABYRINTH_ALIASES = [
    "extcross9",
    "lopsidedy10",
    "ladder10",
    "maze12",
    "hiddentail11",
    "mesh10",
]


def _assert_normalized(dist, tol=1e-9):
    total = sum(p for _x, p in dist)
    assert total == pytest.approx(1.0, abs=tol)
    assert all(p >= 0.0 for _x, p in dist)


@pytest.mark.parametrize("name", LABYRINTH_ALIASES)
def test_labyrinth_model_basic_dimensions_and_initial_belief(name):
    model = lab.LabyrinthModel(name, mode="semi")

    assert model.num_nodes >= 2
    assert model.num_targets == model.num_nodes - 1
    assert model.n_states == model.num_file_states + 1
    assert model.sink_state == model.n_states - 1
    assert model.n_actions == model.act_per_agent ** 2
    assert model.n_obs == model.obs_per_agent ** 2

    assert sum(model.init_belief) == pytest.approx(1.0)
    assert model.init_belief[model.sink_state] == pytest.approx(0.0)
    assert all(p >= 0.0 for p in model.init_belief)

    init_support = [s for s, p in enumerate(model.init_belief) if p > 0.0]
    assert len(init_support) == model.num_targets
    for s in init_support:
        u1, u2, target_idx, found1, found2 = model.state_to_tuple(s)
        assert (u1, u2) == (model.start_node, model.start_node)
        assert target_idx in range(model.num_targets)
        assert (found1, found2) == (0, 0)


@pytest.mark.parametrize("name", LABYRINTH_ALIASES)
def test_labyrinth_transition_and_observation_rows_normalized(name):
    model = lab.LabyrinthModel(name, mode="semi")
    states_to_check = {model.sink_state}
    states_to_check.update(s for s, p in enumerate(model.init_belief) if p > 0.0)

    # Include some states reached in one step from initial support.
    for s in list(states_to_check):
        for a in range(model.n_actions):
            states_to_check.update(sp for sp, _p in model.transition_dist(s, a))

    for s in sorted(states_to_check):
        for a in range(model.n_actions):
            tdist = model.transition_dist(s, a)
            assert tdist, (name, s, a)
            _assert_normalized(tdist)
            for sp, p in tdist:
                assert 0 <= sp < model.n_states
                assert p >= 0.0
                odist = model.obs_dist(sp, a)
                assert odist, (name, sp, a)
                _assert_normalized(odist)
                for o, op in odist:
                    assert 0 <= o < model.n_obs
                    assert op >= 0.0


@pytest.mark.parametrize("name", LABYRINTH_ALIASES)
def test_labyrinth_sink_is_absorbing_and_zero_reward(name):
    model = lab.LabyrinthModel(name, mode="semi")

    for a in range(model.n_actions):
        assert model.reward(model.sink_state, a) == pytest.approx(0.0)
        assert model.transition_dist(model.sink_state, a) == [(model.sink_state, 1.0)]
        assert model.obs_dist(model.sink_state, a) == [(0, 1.0)]


@pytest.mark.parametrize("name", LABYRINTH_ALIASES)
def test_labyrinth_success_transitions_go_to_sink_with_success_reward(name):
    model = lab.LabyrinthModel(name, mode="semi")
    saw_success = False

    for s in range(model.n_states - 1):
        for a in range(model.n_actions):
            if model.reward(s, a) >= lab.SUCCESS_REWARD - 1e-9:
                saw_success = True
                assert any(sp == model.sink_state and p > 0.0 for sp, p in model.transition_dist(s, a))

    assert saw_success, f"Expected at least one success transition in {name}."


@pytest.mark.parametrize("name", LABYRINTH_ALIASES)
def test_labyrinth_sync_propagation_removes_one_sided_found_at_sync_destinations(name):
    model = lab.LabyrinthModel(name, mode="semi")
    trigger_set = set(model.state_triggers)
    assert trigger_set, f"Expected semi-decentralized triggers for {name}."

    for s in range(model.n_states):
        for a in range(model.n_actions):
            for sp, _p in model.transition_dist(s, a):
                if sp == model.sink_state:
                    continue
                if model._is_sync_position(sp, trigger_set):
                    _u1, _u2, _target_idx, found1, found2 = model.state_to_tuple(sp)
                    assert (found1, found2) not in {(1, 0), (0, 1)}


def _old_dense_predict(belief, action, model):
    out = [0.0] * model.n_states
    for s, b_s in enumerate(belief):
        if b_s <= 0.0:
            continue
        for sp, p in model.transition_dist(s, action):
            out[sp] += b_s * p
    return lab.normalize_belief(out)


def _old_dense_local_update(belief, action, local_obs, rid, model):
    pred = _old_dense_predict(belief, action, model)
    post = [
        pred[sp] * model.local_obs_prob(rid, sp, action, local_obs)
        for sp in range(model.n_states)
    ]
    return lab.normalize_belief(post, [s for s, p in enumerate(pred) if p > 0.0])


def _old_dense_joint_update(belief, action, obs, model):
    pred = _old_dense_predict(belief, action, model)
    post = [
        pred[sp] * model.obs_prob(sp, action, obs)
        for sp in range(model.n_states)
    ]
    return lab.normalize_belief(post, [s for s, p in enumerate(pred) if p > 0.0])


def _max_abs_diff(a, b):
    return max(abs(x - y) for x, y in zip(a, b))


@pytest.mark.parametrize("name", ["extcross9", "maze12", "mesh10"])
def test_sparse_belief_updates_match_dense_equations(name):
    model = lab.LabyrinthModel(name, mode="semi")
    rng = random.Random(123)

    for _case in range(150):
        belief = [0.0] * model.n_states
        support_size = rng.randint(1, min(25, model.n_states - 1))
        support = rng.sample(range(model.n_states - 1), support_size)
        weights = [rng.random() for _ in support]
        total = sum(weights)
        for s, w in zip(support, weights):
            belief[s] = w / total

        action = rng.randrange(model.n_actions)
        probe_state = rng.choice(support)
        next_state = model.sample_next_state(probe_state, action, rng)
        obs = model.sample_joint_obs(next_state, action, rng)
        o0, o1 = lab.split_obs(obs, model.obs_per_agent)

        assert _max_abs_diff(
            lab.predict_belief_open_loop(belief, action, model),
            _old_dense_predict(belief, action, model),
        ) <= 1e-12
        assert _max_abs_diff(
            lab.update_local_belief(belief, action, o0, 0, model),
            _old_dense_local_update(belief, action, o0, 0, model),
        ) <= 1e-12
        assert _max_abs_diff(
            lab.update_local_belief(belief, action, o1, 1, model),
            _old_dense_local_update(belief, action, o1, 1, model),
        ) <= 1e-12
        assert _max_abs_diff(
            lab.update_joint_belief(belief, action, obs, model),
            _old_dense_joint_update(belief, action, obs, model),
        ) <= 1e-12
