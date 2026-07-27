"""
tests/test_mars_online_benchmark.py
-----------------------------------
Benchmark-level tests for benchmarks/mars_online.py.
"""

from types import SimpleNamespace
import importlib

import pytest


mars = importlib.import_module("benchmarks.mars_online")


def assert_normalized(belief):
    assert sum(belief) == pytest.approx(1.0)
    assert all(p >= 0.0 for p in belief)


def test_mars_model_loads_rssda_dimensions_and_initial_belief():
    model = mars.MarsModel()

    assert len(model.T) == mars.N_ACTS * mars.N_STATES * mars.N_STATES
    assert len(model.O) == mars.N_ACTS * mars.N_STATES * mars.N_OBS
    assert len(model.R) == mars.N_ACTS * mars.N_STATES
    assert model.init_belief[0] == pytest.approx(1.0)
    assert sum(model.init_belief) == pytest.approx(1.0)


def test_mars_transition_rows_are_normalized():
    model = mars.MarsModel()

    for joint_a in range(mars.N_ACTS):
        base = joint_a * mars.N_STATES * mars.N_STATES
        for s in [0, 1, 16, 80, 255]:
            row_sum = sum(
                model.T[base + s * mars.N_STATES + sp]
                for sp in range(mars.N_STATES)
            )
            assert row_sum == pytest.approx(1.0)


def test_mars_observation_rows_are_normalized():
    model = mars.MarsModel()

    for joint_a in range(mars.N_ACTS):
        base = joint_a * mars.N_STATES * mars.N_OBS
        for sp in [0, 1, 16, 80, 255]:
            row_sum = sum(
                model.O[base + sp * mars.N_OBS + o]
                for o in range(mars.N_OBS)
            )
            assert row_sum == pytest.approx(1.0)


def test_local_and_joint_belief_updates_normalize():
    model = mars.MarsModel()
    prior = list(model.init_belief)
    joint_a = mars.joint_action(0, 1)
    joint_o = 0
    o0, o1 = mars.split_obs(joint_o)

    local0 = mars.update_local_belief(prior, joint_a, o0, 0, model)
    local1 = mars.update_local_belief(prior, joint_a, o1, 1, model)
    joint = mars.update_joint_belief(prior, joint_a, joint_o, model)

    assert_normalized(local0)
    assert_normalized(local1)
    assert_normalized(joint)


def test_mars_trigger_sets_match_script_assumptions():
    assert len(mars.mars_right_band_triggers()) == 64
    assert len(mars.mars_chebyshev1_triggers()) == 100

    args = SimpleNamespace(env_comm_mode="right-band", env_comm_period=0)
    assert mars.should_env_communicate(args, 0, mars.mars_right_band_triggers()[0]) is True
    assert mars.should_env_communicate(args, 0, 0) is False


def test_tiny_obs_decmcts_episode_smoke():
    model = mars.MarsModel()
    args = SimpleNamespace(
        horizon=1,
        outer_iters=1,
        tau=2,
        num_seq=3,
        num_samples=2,
        comm_period=1,
        env_comm_mode="none",
        env_comm_period=0,
        gamma=1.0,
        cp=0.5,
        beta_init=2.0,
        beta_decay=0.995,
        alpha=0.2,
        default_action=0,
        default_policy="qmdp",
        guard_actions=False,
        full_action_space=True,
        qmdp_action_limit=0,
        qmdp_legal_belief="average",
        qmdp_average_weight=0.0,
        heuristic_expansion=True,
        # ObsDecMCTS only implements policy_marginal and visits; "tree" used to
        # fall through to policy_marginal silently.
        action_source="policy_marginal",
        seed=7,
        debug_obs=False,
    )

    ret = mars.simulate_online_episode(model, args, episode_seed=7, verbose=False)

    assert isinstance(ret, float)
