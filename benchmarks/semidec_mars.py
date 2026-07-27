from __future__ import annotations

import argparse
import functools
import random
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.pomdp_mars import (
    MarsModel,
    N_STATES,
    ACT_PER_AGENT,
    OBS_PER_AGENT,
    N_ACTS,
    N_OBS,
)
from semidec.comm_state import CommState, SemiMarkovCommModel, full_partition, isolated_partition
from semidec.sdecmcts import SDecMCTS, StepResult


# ── QMDP GUIDE ────────────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=None)
def _qmdp_tables(horizon: int) -> Tuple[tuple, tuple]:
    """Return (v_by_step, q_by_step) where index 0 = 1 step remaining."""
    model = MarsModel()
    v_tables: list = []
    q_tables: list = []
    values = [0.0] * N_STATES
    for _ in range(horizon):
        q_step = [0.0] * (N_ACTS * N_STATES)
        next_v = [0.0] * N_STATES
        for s in range(N_STATES):
            best = float("-inf")
            for a in range(N_ACTS):
                q = model.reward(a, s)
                base = a * N_STATES * N_STATES + s * N_STATES
                for sp in range(N_STATES):
                    q += model.T[base + sp] * values[sp]
                q_step[a * N_STATES + s] = q
                if q > best:
                    best = q
            next_v[s] = best
        values = next_v
        v_tables.append(tuple(values))
        q_tables.append(tuple(q_step))
    return tuple(v_tables), tuple(q_tables)


def qmdp_value(belief: Sequence[float], remaining: int) -> float:
    if remaining <= 0:
        return 0.0
    v_tables, _ = _qmdp_tables(remaining)
    values = v_tables[-1]
    return sum(belief[s] * values[s] for s in range(N_STATES))


def qmdp_best_joint_action(belief: Sequence[float], remaining: int) -> int:
    if remaining <= 0:
        return 0
    _, q_tables = _qmdp_tables(remaining)
    q_step = q_tables[-1]
    best_a = 0
    best_q = float("-inf")
    for a in range(N_ACTS):
        q = sum(belief[s] * q_step[a * N_STATES + s] for s in range(N_STATES))
        if q > best_q:
            best_q = q
            best_a = a
    return best_a


def make_qmdp_leaf_value(planning_horizon: int):
    def leaf_value(belief: Sequence[float], depth: int) -> float:
        return qmdp_value(belief, max(0, planning_horizon - depth))
    return leaf_value


def make_qmdp_rollout_policy(planning_horizon: int):
    def rollout_policy(belief: Sequence[float], depth: int, _rng: random.Random) -> Dict[int, int]:
        remaining = max(0, planning_horizon - depth)
        ja = qmdp_best_joint_action(belief, remaining)
        return {0: ja % ACT_PER_AGENT, 1: ja // ACT_PER_AGENT}
    return rollout_policy


def _local_belief_update(
    belief: List[float],
    model: MarsModel,
    sparse_T: list,
    obs_from: list,
    my_action: int,
    my_obs: int,
    my_robot_id: int,
    planning_horizon: int,
    depth: int,
) -> List[float]:
    """Update belief using agent's private (action, obs), inferring partner action via QMDP."""
    remaining = max(1, planning_horizon - depth)
    ja_joint = qmdp_best_joint_action(belief, remaining)
    a_other = ja_joint // ACT_PER_AGENT if my_robot_id == 0 else ja_joint % ACT_PER_AGENT
    if my_robot_id == 0:
        joint_action = my_action + ACT_PER_AGENT * a_other
    else:
        joint_action = a_other + ACT_PER_AGENT * my_action

    posterior = [0.0] * N_STATES
    for s in range(N_STATES):
        bs = belief[s]
        if bs <= 0.0:
            continue
        for sp, t_prob in sparse_T[joint_action * N_STATES + s]:
            joint_obs = obs_from[joint_action * N_STATES + sp]
            agent_obs = joint_obs % OBS_PER_AGENT if my_robot_id == 0 else joint_obs // OBS_PER_AGENT
            if agent_obs == my_obs:
                posterior[sp] += bs * t_prob
    total = sum(posterior)
    if total <= 1e-12:
        return list(belief)
    return [p / total for p in posterior]


def make_qmdp_default_action(robot_id: int, planning_horizon: int, root_belief: Sequence[float], adapter):
    root_belief = list(root_belief)

    def default_action(depth: int, history) -> int:
        belief = list(root_belief)
        for d, (a_local, o_local) in enumerate(history):
            belief = _local_belief_update(
                belief, adapter.model, adapter._sparse_T, adapter._obs_from,
                a_local, o_local, robot_id, planning_horizon, d,
            )
        remaining = max(1, planning_horizon - depth)
        ja = qmdp_best_joint_action(belief, remaining)
        if robot_id == 0:
            return ja % ACT_PER_AGENT
        return ja // ACT_PER_AGENT
    return default_action


def _positions(s: int):
    """Decode joint state to (rover0_grid_pos, rover1_grid_pos)."""
    return s // 16, s % 16


def _collocated(s: int) -> bool:
    p0, p1 = _positions(s)
    return p0 == p1


class SDecMarsAdapter:
    def __init__(self, model: MarsModel):
        self.model = model
        # Precompute sparse transition and deterministic obs maps for fast belief ops
        self._sparse_T: List[List[Tuple[int, float]]] = []
        self._obs_from: List[int] = []  # index: a * N_STATES + sp → deterministic joint obs
        for a in range(N_ACTS):
            for s in range(N_STATES):
                base_t = a * N_STATES * N_STATES + s * N_STATES
                self._sparse_T.append(
                    [(sp, model.T[base_t + sp]) for sp in range(N_STATES)
                     if model.T[base_t + sp] > 1e-9]
                )
        for a in range(N_ACTS):
            for sp in range(N_STATES):
                base_o = a * N_STATES * N_OBS + sp * N_OBS
                o = next(o for o in range(N_OBS) if model.O[base_o + o] > 0.5)
                self._obs_from.append(o)

    def _sample_next(self, s: int, joint_action: int, rng: random.Random) -> int:
        trans = self._sparse_T[joint_action * N_STATES + s]
        r = rng.random()
        cum = 0.0
        for sp, p in trans:
            cum += p
            if r <= cum:
                return sp
        return trans[-1][0]

    def _get_obs(self, joint_action: int, next_state: int) -> int:
        return self._obs_from[joint_action * N_STATES + next_state]

    def _update_belief_sparse(
        self, belief: Sequence[float], joint_action: int, joint_obs: int
    ) -> List[float]:
        posterior = [0.0] * N_STATES
        for s in range(N_STATES):
            bs = belief[s]
            if bs <= 0.0:
                continue
            for sp, t_prob in self._sparse_T[joint_action * N_STATES + s]:
                if self._obs_from[joint_action * N_STATES + sp] == joint_obs:
                    posterior[sp] += bs * t_prob
        total = sum(posterior)
        if total <= 1e-12:
            return list(belief)
        return [p / total for p in posterior]

    def legal_actions(self, _belief: Sequence[float], _robot_id: int, _depth: int) -> List[int]:
        return list(range(ACT_PER_AGENT))

    def joint_action_from_dict(self, actions: Dict[int, int]) -> int:
        return actions[0] + ACT_PER_AGENT * actions[1]

    def split_obs(self, joint_obs: int) -> List[int]:
        return [joint_obs % OBS_PER_AGENT, joint_obs // OBS_PER_AGENT]

    def sample_belief_step(
        self,
        belief: Sequence[float],
        joint_action: int,
        rng: random.Random,
    ):
        expected_reward = sum(
            belief[s] * self.model.reward(joint_action, s) for s in range(N_STATES)
        )
        r = rng.random()
        cum = 0.0
        s = N_STATES - 1
        for i, p in enumerate(belief):
            cum += p
            if r <= cum:
                s = i
                break
        next_state = self._sample_next(s, joint_action, rng)
        joint_obs = self._get_obs(joint_action, next_state)
        next_belief = self._update_belief_sparse(belief, joint_action, joint_obs)
        return next_belief, joint_obs, expected_reward

    def update_joint_belief(
        self,
        belief: Sequence[float],
        joint_action: int,
        joint_obs: int,
    ) -> List[float]:
        return self._update_belief_sparse(belief, joint_action, joint_obs)


def mars_collocated_comm_transition(
    _comm_state: CommState,
    _action_dict: Dict[int, int],
    _joint_action: int,
    _joint_obs: int,
    next_belief: Sequence[float],
    _next_depth: int,
) -> CommState:
    p_collocated = sum(next_belief[s] for s in range(N_STATES) if _collocated(s))
    if p_collocated >= 0.5:
        return CommState(partition=full_partition([0, 1]), sojourn_remaining=1)
    return CommState(partition=isolated_partition([0, 1]), sojourn_remaining=1)


def make_comm_model(seed: int, initial_collocated: bool) -> SemiMarkovCommModel:
    if initial_collocated:
        return SemiMarkovCommModel(
            robot_ids=[0, 1],
            partition_probs={((0, 1),): 1.0},
            sojourn_probs={1: 1.0},
            initial_partition=((0, 1),),
            initial_sojourn=1,
            rng=random.Random(seed),
        )
    return SemiMarkovCommModel(
        robot_ids=[0, 1],
        partition_probs={((0,), (1,)): 1.0},
        sojourn_probs={1: 1.0},
        initial_partition=((0,), (1,)),
        initial_sojourn=1,
        rng=random.Random(seed),
    )


def simulate_episode(
    *,
    horizon: int,
    iterations: int,
    seed: int,
    cp: float,
    gamma: float,
    guide: str,
    min_edge_visits: int = 1,
    max_tree_depth: int = None,
    shared_belief: bool = False,
) -> float:
    env_rng = random.Random(seed)
    model = MarsModel()
    adapter = SDecMarsAdapter(model)

    # Both rovers start at grid position 0 → collocated
    true_state = 0
    common_belief = list(model.init_belief)
    histories = {0: tuple(), 1: tuple()}
    pending_history = []
    active_policy = None
    total_reward = 0.0
    # For shared_belief mode: track running joint belief between syncs
    running_belief: List[float] = list(model.init_belief)
    belief_ref: List[List[float]] = [list(model.init_belief)]

    for t in range(horizon):
        if active_policy is None:
            remaining = horizon - t
            init_collocated = _collocated(true_state)
            if guide == "qmdp":
                rollout_policy = make_qmdp_rollout_policy(remaining)
                leaf_value_fn = make_qmdp_leaf_value(remaining)
                if shared_belief:
                    # Default actions use the mutable belief_ref (updated each step)
                    belief_ref[0] = list(common_belief)

                    def _make_shared_default(robot_id: int, r: int, bref: List[List[float]]):
                        def _default(depth: int, history) -> int:
                            rem = max(1, r - depth)
                            ja = qmdp_best_joint_action(bref[0], rem)
                            return ja % ACT_PER_AGENT if robot_id == 0 else ja // ACT_PER_AGENT
                        return _default

                    default_actions = {
                        0: _make_shared_default(0, remaining, belief_ref),
                        1: _make_shared_default(1, remaining, belief_ref),
                    }
                else:
                    default_actions = {
                        0: make_qmdp_default_action(0, remaining, common_belief, adapter),
                        1: make_qmdp_default_action(1, remaining, common_belief, adapter),
                    }
            elif guide == "heuristic":
                rollout_policy = None
                leaf_value_fn = None
                default_actions = {0: 0, 1: 0}
            else:
                raise ValueError(f"Unknown guide: {guide!r}")
            planner = SDecMCTS(
                robot_ids=[0, 1],
                root_belief=common_belief,
                model=adapter,
                comm_model=make_comm_model(seed + 1009 * t, init_collocated),
                horizon=remaining,
                gamma=gamma,
                cp=cp,
                rollout_policy=rollout_policy,
                leaf_value_fn=leaf_value_fn,
                comm_transition_fn=mars_collocated_comm_transition,
                seed=seed + 7919 * t,
                max_tree_depth=max_tree_depth,
            )
            planner.run(iterations)
            active_policy = planner.extract_policy(default_actions=default_actions, min_edge_visits=min_edge_visits)
            histories = {0: tuple(), 1: tuple()}
            running_belief = list(common_belief)

        # Update belief_ref to current running_belief before action selection
        if shared_belief:
            belief_ref[0] = list(running_belief)

        depth_since_sync = len(histories[0])
        action_dict = active_policy.joint_action_from_histories(
            [0, 1],
            depth_since_sync,
            histories,
        )
        a0, a1 = int(action_dict[0]), int(action_dict[1])
        joint_a = a0 + ACT_PER_AGENT * a1

        reward = model.reward(joint_a, true_state)
        total_reward += reward

        base_t = joint_a * N_STATES * N_STATES + true_state * N_STATES
        r = env_rng.random()
        cum = 0.0
        next_state = N_STATES - 1
        for sp in range(N_STATES):
            cum += model.T[base_t + sp]
            if r <= cum:
                next_state = sp
                break

        base_o = joint_a * N_STATES * N_OBS + next_state * N_OBS
        r2 = env_rng.random()
        cum = 0.0
        joint_o = N_OBS - 1
        for o in range(N_OBS):
            cum += model.O[base_o + o]
            if r2 <= cum:
                joint_o = o
                break

        o0 = joint_o % OBS_PER_AGENT
        o1 = joint_o // OBS_PER_AGENT
        true_state = next_state

        histories[0] = histories[0] + ((a0, o0),)
        histories[1] = histories[1] + ((a1, o1),)
        pending_history.append((joint_a, joint_o))

        if shared_belief:
            # Update running joint belief with full joint obs at every step
            running_belief = adapter.update_joint_belief(running_belief, joint_a, joint_o)

        if _collocated(true_state):
            for hist_a, hist_o in pending_history:
                common_belief = adapter.update_joint_belief(common_belief, hist_a, hist_o)
            pending_history = []
            active_policy = None

    return total_reward


def run_batch(args) -> List[float]:
    return [
        simulate_episode(
            horizon=args.horizon,
            iterations=args.iterations,
            seed=args.seed + episode,
            cp=args.cp,
            gamma=args.gamma,
            guide=args.guide,
            min_edge_visits=args.min_edge_visits,
            max_tree_depth=args.max_tree_depth if args.max_tree_depth > 0 else None,
            shared_belief=getattr(args, "shared_belief", False),
        )
        for episode in range(args.episodes)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=7)
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cp", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--min-edge-visits", type=int, default=5,
                        help="Minimum edge visits to include in policy extraction.")
    parser.add_argument("--max-tree-depth", type=int, default=0,
                        help="Max tree depth before always using rollout/leaf. 0 = unlimited.")
    parser.add_argument(
        "--guide",
        choices=["qmdp", "heuristic"],
        default="qmdp",
        help="Guide used for rollouts, leaf values, and fallback defaults.",
    )
    parser.add_argument(
        "--shared-belief",
        action="store_true",
        default=False,
        help="Share joint observations between agents at every step (oracle belief tracking).",
    )
    args = parser.parse_args()

    returns = run_batch(args)
    mean = statistics.fmean(returns)
    stderr = (
        statistics.stdev(returns) / (len(returns) ** 0.5)
        if len(returns) > 1
        else 0.0
    )
    print("SDecMCTS Mars prototype")
    print(
        f"horizon={args.horizon} episodes={args.episodes} "
        f"iterations={args.iterations} guide={args.guide} seed={args.seed}"
    )
    print(f"mean_return={mean:.3f} stderr={stderr:.3f}")
    print(f"returns={returns}")


if __name__ == "__main__":
    main()
