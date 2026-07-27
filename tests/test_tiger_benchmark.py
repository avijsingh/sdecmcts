"""
tests/test_tiger_benchmark.py
-----------------------------
Benchmark-level unit tests for benchmarks/tiger_online.py.

"""

from types import SimpleNamespace
import importlib

import pytest


tiger = importlib.import_module("benchmarks.tiger_online")


def obs_id(name_part: str) -> int:
    """Find observation id from OBS_NAME, e.g. 'left' -> HEAR_LEFT."""
    for k, v in tiger.OBS_NAME.items():
        if name_part.lower() in str(v).lower():
            return k
    raise AssertionError(f"Could not find observation containing {name_part!r} in OBS_NAME={tiger.OBS_NAME}")


def joint_obs(model, o0, o1):
    if hasattr(model, "joint_obs"):
        return model.joint_obs(o0, o1)
    return o0 + 2 * o1


def assert_belief_close(actual, expected, tol=1e-9):
    assert len(actual) == len(expected)
    for a, e in zip(actual, expected):
        assert a == pytest.approx(e, abs=tol)


def make_model():
    return tiger.TigerModel()


def test_tiger_reward_table_left_state():
    model = make_model()
    s = tiger.TIGER_LEFT

    assert model.reward(s, model.joint_action(tiger.LISTEN, tiger.LISTEN)) == pytest.approx(-2.0)

    assert model.reward(s, model.joint_action(tiger.OPEN_RIGHT, tiger.OPEN_RIGHT)) == pytest.approx(20.0)
    assert model.reward(s, model.joint_action(tiger.OPEN_RIGHT, tiger.LISTEN)) == pytest.approx(9.0)
    assert model.reward(s, model.joint_action(tiger.LISTEN, tiger.OPEN_RIGHT)) == pytest.approx(9.0)

    assert model.reward(s, model.joint_action(tiger.OPEN_LEFT, tiger.OPEN_LEFT)) == pytest.approx(-50.0)
    assert model.reward(s, model.joint_action(tiger.OPEN_LEFT, tiger.LISTEN)) == pytest.approx(-101.0)
    assert model.reward(s, model.joint_action(tiger.LISTEN, tiger.OPEN_LEFT)) == pytest.approx(-101.0)

    assert model.reward(s, model.joint_action(tiger.OPEN_LEFT, tiger.OPEN_RIGHT)) == pytest.approx(-100.0)
    assert model.reward(s, model.joint_action(tiger.OPEN_RIGHT, tiger.OPEN_LEFT)) == pytest.approx(-100.0)


def test_tiger_reward_table_right_state():
    model = make_model()
    s = tiger.TIGER_RIGHT

    assert model.reward(s, model.joint_action(tiger.LISTEN, tiger.LISTEN)) == pytest.approx(-2.0)

    assert model.reward(s, model.joint_action(tiger.OPEN_LEFT, tiger.OPEN_LEFT)) == pytest.approx(20.0)
    assert model.reward(s, model.joint_action(tiger.OPEN_LEFT, tiger.LISTEN)) == pytest.approx(9.0)
    assert model.reward(s, model.joint_action(tiger.LISTEN, tiger.OPEN_LEFT)) == pytest.approx(9.0)

    assert model.reward(s, model.joint_action(tiger.OPEN_RIGHT, tiger.OPEN_RIGHT)) == pytest.approx(-50.0)
    assert model.reward(s, model.joint_action(tiger.OPEN_RIGHT, tiger.LISTEN)) == pytest.approx(-101.0)
    assert model.reward(s, model.joint_action(tiger.LISTEN, tiger.OPEN_RIGHT)) == pytest.approx(-101.0)

    assert model.reward(s, model.joint_action(tiger.OPEN_LEFT, tiger.OPEN_RIGHT)) == pytest.approx(-100.0)
    assert model.reward(s, model.joint_action(tiger.OPEN_RIGHT, tiger.OPEN_LEFT)) == pytest.approx(-100.0)


def test_listen_does_not_change_hidden_state_belief():
    model = make_model()
    b_left = [1.0, 0.0]
    a = model.joint_action(tiger.LISTEN, tiger.LISTEN)

    pred = tiger.predict_belief_open_loop(b_left, a, model)

    assert_belief_close(pred, [1.0, 0.0])


@pytest.mark.parametrize(
    "a0,a1",
    [
        ("OPEN_LEFT", "LISTEN"),
        ("LISTEN", "OPEN_LEFT"),
        ("OPEN_RIGHT", "LISTEN"),
        ("LISTEN", "OPEN_RIGHT"),
        ("OPEN_LEFT", "OPEN_LEFT"),
        ("OPEN_RIGHT", "OPEN_RIGHT"),
        ("OPEN_LEFT", "OPEN_RIGHT"),
        ("OPEN_RIGHT", "OPEN_LEFT"),
    ],
)
def test_any_open_resets_hidden_state_to_uniform(a0, a1):
    model = make_model()
    b_left = [1.0, 0.0]
    joint_a = model.joint_action(getattr(tiger, a0), getattr(tiger, a1))

    pred = tiger.predict_belief_open_loop(b_left, joint_a, model)

    assert_belief_close(pred, [0.5, 0.5])


def test_local_belief_update_single_hear_left_from_uniform():
    model = make_model()
    prior = [0.5, 0.5]
    hear_left = obs_id("left")
    joint_a = model.joint_action(tiger.LISTEN, tiger.LISTEN)

    post0 = tiger.update_local_belief(prior, joint_a, hear_left, 0, model)
    post1 = tiger.update_local_belief(prior, joint_a, hear_left, 1, model)

    # Listening accuracy is 0.85: 0.5*0.85 / (0.5*0.85 + 0.5*0.15) = 0.85.
    assert_belief_close(post0, [0.85, 0.15])
    assert_belief_close(post1, [0.85, 0.15])


def test_joint_belief_update_two_matching_observations():
    model = make_model()
    prior = [0.5, 0.5]
    hear_left = obs_id("left")
    joint_a = model.joint_action(tiger.LISTEN, tiger.LISTEN)
    joint_o = joint_obs(model, hear_left, hear_left)

    post = tiger.update_joint_belief(prior, joint_a, joint_o, model)

    # Two independent 0.85-accurate observations: 0.85^2 / (0.85^2 + 0.15^2).
    assert_belief_close(post, [0.85 ** 2 / (0.85 ** 2 + 0.15 ** 2),
                               0.15 ** 2 / (0.85 ** 2 + 0.15 ** 2)])


def test_joint_belief_update_conflicting_observations_stays_uniform():
    model = make_model()
    prior = [0.5, 0.5]
    hear_left = obs_id("left")
    hear_right = obs_id("right")
    joint_a = model.joint_action(tiger.LISTEN, tiger.LISTEN)
    joint_o = joint_obs(model, hear_left, hear_right)

    post = tiger.update_joint_belief(prior, joint_a, joint_o, model)

    assert_belief_close(post, [0.5, 0.5])


def test_belief_after_open_and_observation_resets_to_uniform():
    model = make_model()
    prior = [0.9, 0.1]
    hear_left = obs_id("left")
    hear_right = obs_id("right")
    joint_a = model.joint_action(tiger.OPEN_RIGHT, tiger.OPEN_RIGHT)
    joint_o = joint_obs(model, hear_left, hear_right)

    post = tiger.update_joint_belief(prior, joint_a, joint_o, model)

    assert_belief_close(post, [0.5, 0.5])


def fixed_first_action_value(model, belief, horizon, a0, a1):
    seqs = {
        0: [a0] + [tiger.LISTEN] * max(0, horizon - 1),
        1: [a1] + [tiger.LISTEN] * max(0, horizon - 1),
    }
    return tiger.expected_open_loop_return(
        belief=belief,
        joint_sequences=seqs,
        model=model,
        horizon=horizon,
    )


def test_expected_open_loop_values_from_uniform_prior_h8():
    model = make_model()
    belief = [0.5, 0.5]
    H = 8

    vals = {
        "LL": fixed_first_action_value(model, belief, H, tiger.LISTEN, tiger.LISTEN),
        "OL_OL": fixed_first_action_value(model, belief, H, tiger.OPEN_LEFT, tiger.OPEN_LEFT),
        "OR_OR": fixed_first_action_value(model, belief, H, tiger.OPEN_RIGHT, tiger.OPEN_RIGHT),
        "OL_L": fixed_first_action_value(model, belief, H, tiger.OPEN_LEFT, tiger.LISTEN),
        "L_OL": fixed_first_action_value(model, belief, H, tiger.LISTEN, tiger.OPEN_LEFT),
        "OL_OR": fixed_first_action_value(model, belief, H, tiger.OPEN_LEFT, tiger.OPEN_RIGHT),
    }

    assert vals["LL"] == pytest.approx(-16.0)
    assert vals["OL_OL"] == pytest.approx(-29.0)
    assert vals["OR_OR"] == pytest.approx(-29.0)
    assert vals["OL_L"] == pytest.approx(-60.0)
    assert vals["L_OL"] == pytest.approx(-60.0)
    assert vals["OL_OR"] == pytest.approx(-114.0)

    assert vals["LL"] > vals["OL_OL"]
    assert vals["LL"] > vals["OL_L"]
    assert vals["LL"] > vals["OL_OR"]


def test_expected_open_loop_values_from_confident_right_belief_h1():
    model = make_model()
    belief = [0.1, 0.9]  # tiger likely right; safe door is left
    H = 1

    ll = fixed_first_action_value(model, belief, H, tiger.LISTEN, tiger.LISTEN)
    olol = fixed_first_action_value(model, belief, H, tiger.OPEN_LEFT, tiger.OPEN_LEFT)
    oror = fixed_first_action_value(model, belief, H, tiger.OPEN_RIGHT, tiger.OPEN_RIGHT)

    assert ll == pytest.approx(-2.0)
    assert olol == pytest.approx(13.0)
    assert oror == pytest.approx(-43.0)
    assert olol > ll > oror


def test_should_env_communicate_periodic_mode():
    if not hasattr(tiger, "should_env_communicate"):
        pytest.skip("should_env_communicate helper not present")

    args = SimpleNamespace(env_comm_mode="periodic", env_comm_period=2)
    assert tiger.should_env_communicate(args, 0, tiger.LISTEN, tiger.LISTEN) is False
    assert tiger.should_env_communicate(args, 1, tiger.OPEN_LEFT, tiger.LISTEN) is True


def test_should_env_communicate_both_listen_mode():
    if not hasattr(tiger, "should_env_communicate"):
        pytest.skip("should_env_communicate helper not present")

    args = SimpleNamespace(env_comm_mode="both-listen", env_comm_period=1)
    assert tiger.should_env_communicate(args, 0, tiger.LISTEN, tiger.LISTEN) is True
    assert tiger.should_env_communicate(args, 0, tiger.OPEN_LEFT, tiger.LISTEN) is False
    assert tiger.should_env_communicate(args, 0, tiger.LISTEN, tiger.OPEN_LEFT) is False
    assert tiger.should_env_communicate(args, 0, tiger.OPEN_LEFT, tiger.OPEN_LEFT) is False


def test_guard_vetoes_open_when_uncertain_and_corrects_open_when_confident():
    if not hasattr(tiger, "guard_tiger_action"):
        pytest.skip("guard_tiger_action helper not present")

    assert tiger.guard_tiger_action(
        belief=[0.5, 0.5],
        proposed_action=tiger.OPEN_LEFT,
        open_threshold=0.85,
    ) == tiger.LISTEN

    assert tiger.guard_tiger_action(
        belief=[0.9, 0.1],
        proposed_action=tiger.OPEN_LEFT,
        open_threshold=0.85,
    ) == tiger.OPEN_RIGHT

    assert tiger.guard_tiger_action(
        belief=[0.1, 0.9],
        proposed_action=tiger.OPEN_RIGHT,
        open_threshold=0.85,
    ) == tiger.OPEN_LEFT
