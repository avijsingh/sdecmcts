from types import SimpleNamespace

from benchmarks import tiger_online as tiger
from benchmarks import semidec_tiger
from semidec.comm_state import CommState, full_partition, isolated_partition


def test_semidec_tiger_batch_runs_and_returns_numbers():
    args = SimpleNamespace(
        horizon=3,
        episodes=2,
        iterations=20,
        sync_period=2,
        sync_mode="both-listen",
        seed=17,
        cp=1.0,
        gamma=1.0,
        qmdp_leaf=False,
        guide="exact",
    )

    returns = semidec_tiger.run_batch(args)

    assert len(returns) == 2
    assert all(isinstance(ret, float) for ret in returns)


def test_tiger_sync_gate_requires_both_agents_to_listen():
    assert semidec_tiger.should_sync(
        "both-listen",
        sync_period=99,
        t=0,
        a0=tiger.LISTEN,
        a1=tiger.LISTEN,
    )
    assert not semidec_tiger.should_sync(
        "both-listen",
        sync_period=1,
        t=0,
        a0=tiger.OPEN_LEFT,
        a1=tiger.LISTEN,
    )
    assert semidec_tiger.should_sync(
        "periodic",
        sync_period=2,
        t=1,
        a0=tiger.OPEN_LEFT,
        a1=tiger.OPEN_RIGHT,
    )


def test_tiger_gated_comm_transition_sets_partition_from_joint_action():
    prev = CommState(partition=isolated_partition([0, 1]), sojourn_remaining=1)

    connected = semidec_tiger.tiger_gated_comm_transition(
        prev,
        {0: tiger.LISTEN, 1: tiger.LISTEN},
        tiger.TigerModel.joint_action(tiger.LISTEN, tiger.LISTEN),
        tiger.TigerModel.joint_obs(tiger.TIGER_LEFT, tiger.TIGER_LEFT),
        [0.9, 0.1],
        1,
    )
    isolated = semidec_tiger.tiger_gated_comm_transition(
        prev,
        {0: tiger.OPEN_RIGHT, 1: tiger.LISTEN},
        tiger.TigerModel.joint_action(tiger.OPEN_RIGHT, tiger.LISTEN),
        tiger.TigerModel.joint_obs(tiger.TIGER_LEFT, tiger.TIGER_LEFT),
        [0.5, 0.5],
        1,
    )

    assert connected.partition == full_partition([0, 1])
    assert isolated.partition == isolated_partition([0, 1])
