from __future__ import annotations

import argparse
import functools
import random
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks import tiger_online as tiger
from semidec.comm_state import CommState, SemiMarkovCommModel, full_partition, isolated_partition
from semidec.sdecmcts import SDecMCTS, StepResult


class SDecTigerAdapter:
    def __init__(self, model: tiger.TigerModel):
        self.model = model

    def sample_state_from_belief(self, belief: Sequence[float], rng: random.Random) -> int:
        return tiger.TIGER_LEFT if rng.random() < belief[tiger.TIGER_LEFT] else tiger.TIGER_RIGHT

    def legal_actions(self, _belief: Sequence[float], _robot_id: int, _depth: int) -> List[int]:
        return [tiger.OPEN_LEFT, tiger.OPEN_RIGHT, tiger.LISTEN]

    def joint_action_from_dict(self, actions: Dict[int, int]) -> int:
        return self.model.joint_action(actions[0], actions[1])

    def step(self, state: int, joint_action: int, rng: random.Random) -> StepResult:
        reward = self.model.reward(state, joint_action)
        next_state = self.model.sample_next_state(state, joint_action, rng)
        joint_obs = self.model.sample_joint_obs(next_state, joint_action, rng)
        return StepResult(next_state=next_state, joint_obs=joint_obs, reward=reward)

    def split_obs(self, joint_obs: int):
        return self.model.split_obs(joint_obs)

    def update_joint_belief(
        self,
        belief: Sequence[float],
        joint_action: int,
        joint_obs: int,
    ) -> List[float]:
        return tiger.update_joint_belief(belief, joint_action, joint_obs, self.model)

    def sample_belief_step(
        self,
        belief: Sequence[float],
        joint_action: int,
        rng: random.Random,
    ):
        expected_reward = sum(
            belief[state] * self.model.reward(state, joint_action)
            for state in range(tiger.N_STATES)
        )
        obs_probs = []
        for joint_obs in range(tiger.N_OBS):
            p_obs = 0.0
            for next_state in range(tiger.N_STATES):
                pred = sum(
                    belief[state] * self.model.transition_prob(state, joint_action, next_state)
                    for state in range(tiger.N_STATES)
                )
                p_obs += pred * self.model.obs_prob(next_state, joint_action, joint_obs)
            obs_probs.append(p_obs)

        r = rng.random()
        cumulative = 0.0
        sampled_obs = tiger.N_OBS - 1
        for joint_obs, p_obs in enumerate(obs_probs):
            cumulative += p_obs
            if r <= cumulative:
                sampled_obs = joint_obs
                break

        next_belief = self.update_joint_belief(belief, joint_action, sampled_obs)
        return next_belief, sampled_obs, expected_reward


def make_comm_model(seed: int, sync_period: int) -> SemiMarkovCommModel:
    if sync_period <= 0:
        return SemiMarkovCommModel(
            robot_ids=[0, 1],
            partition_probs={((0,), (1,)): 1.0},
            sojourn_probs={1_000_000: 1.0},
            initial_partition=((0,), (1,)),
            initial_sojourn=1_000_000,
            rng=random.Random(seed),
        )
    return SemiMarkovCommModel(
        robot_ids=[0, 1],
        partition_probs={((0,), (1,)): 1.0},
        sojourn_probs={sync_period: 1.0},
        initial_partition=((0, 1),),
        initial_sojourn=sync_period,
        rng=random.Random(seed),
    )


def tiger_gated_comm_transition(
    _comm_state: CommState,
    action_dict: Dict[int, int],
    _joint_action: int,
    _joint_obs: int,
    _next_belief: Sequence[float],
    _next_depth: int,
) -> CommState:
    if action_dict[0] == tiger.LISTEN and action_dict[1] == tiger.LISTEN:
        return CommState(partition=full_partition([0, 1]), sojourn_remaining=1)
    return CommState(partition=isolated_partition([0, 1]), sojourn_remaining=1)


def should_sync(sync_mode: str, sync_period: int, t: int, a0: int, a1: int) -> bool:
    if sync_mode == "none":
        return False
    if sync_mode == "both-listen":
        return a0 == tiger.LISTEN and a1 == tiger.LISTEN
    if sync_mode == "periodic":
        return sync_period > 0 and ((t + 1) % sync_period == 0)
    raise ValueError(f"Unknown sync_mode: {sync_mode}")


def all_joint_actions():
    return [
        model_action
        for model_action in range(tiger.N_ACTS)
    ]


@functools.lru_cache(maxsize=None)
def qmdp_state_values(remaining_horizon: int):
    model = tiger.TigerModel()
    values = [0.0] * tiger.N_STATES
    for _ in range(remaining_horizon):
        next_values = []
        for state in range(tiger.N_STATES):
            best = float("-inf")
            for joint_action in all_joint_actions():
                q = model.reward(state, joint_action)
                for next_state in range(tiger.N_STATES):
                    q += (
                        model.transition_prob(state, joint_action, next_state)
                        * values[next_state]
                    )
                best = max(best, q)
            next_values.append(best)
        values = next_values
    return tuple(values)


@functools.lru_cache(maxsize=None)
def qmdp_action_values(remaining_horizon: int):
    model = tiger.TigerModel()
    future_values = qmdp_state_values(max(0, remaining_horizon - 1))
    table = {}
    for joint_action in all_joint_actions():
        values = []
        for state in range(tiger.N_STATES):
            q = model.reward(state, joint_action)
            for next_state in range(tiger.N_STATES):
                q += (
                    model.transition_prob(state, joint_action, next_state)
                    * future_values[next_state]
                )
            values.append(q)
        table[joint_action] = tuple(values)
    return table


def qmdp_joint_action(belief: Sequence[float], remaining_horizon: int) -> int:
    if remaining_horizon <= 0:
        return tiger.TigerModel().joint_action(tiger.LISTEN, tiger.LISTEN)
    action_values = qmdp_action_values(remaining_horizon)
    return max(
        action_values,
        key=lambda joint_action: sum(
            belief[state] * action_values[joint_action][state]
            for state in range(tiger.N_STATES)
        ),
    )


def qmdp_belief_value(belief: Sequence[float], remaining_horizon: int) -> float:
    if remaining_horizon <= 0:
        return 0.0
    values = qmdp_state_values(remaining_horizon)
    return sum(belief[state] * values[state] for state in range(tiger.N_STATES))


def _belief_key(belief: Sequence[float]):
    return tuple(round(float(x), 12) for x in belief)


def _normalize(weights: Sequence[float]) -> List[float]:
    total = float(sum(weights))
    if total <= 0.0:
        return [1.0 / len(weights)] * len(weights)
    return [float(x) / total for x in weights]


@functools.lru_cache(maxsize=None)
def exact_pomdp_value(belief_key, remaining_horizon: int) -> float:
    model = tiger.TigerModel()
    belief = tuple(float(x) for x in belief_key)
    if remaining_horizon <= 0:
        return 0.0

    best = float("-inf")
    for joint_action in all_joint_actions():
        immediate = sum(
            belief[state] * model.reward(state, joint_action)
            for state in range(tiger.N_STATES)
        )
        future = 0.0
        for joint_obs in range(tiger.N_OBS):
            obs_prob = 0.0
            posterior = [0.0] * tiger.N_STATES
            for next_state in range(tiger.N_STATES):
                pred = sum(
                    belief[state] * model.transition_prob(state, joint_action, next_state)
                    for state in range(tiger.N_STATES)
                )
                p = pred * model.obs_prob(next_state, joint_action, joint_obs)
                posterior[next_state] = p
                obs_prob += p
            if obs_prob > 1e-12:
                future += obs_prob * exact_pomdp_value(
                    _belief_key(_normalize(posterior)),
                    remaining_horizon - 1,
                )
        best = max(best, immediate + future)
    return best


@functools.lru_cache(maxsize=None)
def exact_pomdp_action(belief_key, remaining_horizon: int) -> int:
    model = tiger.TigerModel()
    belief = tuple(float(x) for x in belief_key)
    if remaining_horizon <= 0:
        return model.joint_action(tiger.LISTEN, tiger.LISTEN)

    best_value = float("-inf")
    best_action = model.joint_action(tiger.LISTEN, tiger.LISTEN)
    for joint_action in all_joint_actions():
        immediate = sum(
            belief[state] * model.reward(state, joint_action)
            for state in range(tiger.N_STATES)
        )
        future = 0.0
        for joint_obs in range(tiger.N_OBS):
            obs_prob = 0.0
            posterior = [0.0] * tiger.N_STATES
            for next_state in range(tiger.N_STATES):
                pred = sum(
                    belief[state] * model.transition_prob(state, joint_action, next_state)
                    for state in range(tiger.N_STATES)
                )
                p = pred * model.obs_prob(next_state, joint_action, joint_obs)
                posterior[next_state] = p
                obs_prob += p
            if obs_prob > 1e-12:
                future += obs_prob * exact_pomdp_value(
                    _belief_key(_normalize(posterior)),
                    remaining_horizon - 1,
                )
        value = immediate + future
        if value > best_value:
            best_value = value
            best_action = joint_action
    return best_action


def make_exact_rollout_policy(planning_horizon: int):
    def rollout_policy(belief: Sequence[float], depth: int, _rng: random.Random) -> Dict[int, int]:
        remaining = max(0, planning_horizon - depth)
        joint_action = exact_pomdp_action(_belief_key(belief), remaining)
        a0, a1 = tiger.TigerModel.split_action(joint_action)
        return {0: a0, 1: a1}

    return rollout_policy


def make_exact_leaf_value(planning_horizon: int):
    def leaf_value(belief: Sequence[float], depth: int) -> float:
        remaining = max(0, planning_horizon - depth)
        return exact_pomdp_value(_belief_key(belief), remaining)

    return leaf_value


def make_exact_local_default(robot_id: int, planning_horizon: int):
    def default_action(depth: int, history) -> int:
        local_belief = tiger.tiger_belief_from_local_history(history)
        remaining = max(1, planning_horizon - depth)
        joint_action = exact_pomdp_action(_belief_key(local_belief), remaining)
        actions = tiger.TigerModel.split_action(joint_action)
        return actions[robot_id]

    return default_action


def make_qmdp_rollout_policy(planning_horizon: int):
    def rollout_policy(belief: Sequence[float], depth: int, _rng: random.Random) -> Dict[int, int]:
        remaining = max(0, planning_horizon - depth)
        joint_action = qmdp_joint_action(belief, remaining)
        a0, a1 = tiger.TigerModel.split_action(joint_action)
        return {0: a0, 1: a1}

    return rollout_policy


def make_qmdp_leaf_value(planning_horizon: int):
    def leaf_value(belief: Sequence[float], depth: int) -> float:
        remaining = max(0, planning_horizon - depth)
        return qmdp_belief_value(belief, remaining)

    return leaf_value


def make_qmdp_local_default(robot_id: int, planning_horizon: int):
    def default_action(depth: int, history) -> int:
        local_belief = tiger.tiger_belief_from_local_history(history)
        remaining = max(1, planning_horizon - depth)
        joint_action = qmdp_joint_action(local_belief, remaining)
        actions = tiger.TigerModel.split_action(joint_action)
        return actions[robot_id]

    return default_action


def simulate_episode(
    *,
    horizon: int,
    iterations: int,
    sync_period: int,
    sync_mode: str,
    seed: int,
    cp: float,
    gamma: float,
    qmdp_leaf: bool,
    guide: str,
    min_edge_visits: int = 1,
    max_tree_depth: int = None,
) -> float:
    env_rng = random.Random(seed)
    model = tiger.TigerModel()
    adapter = SDecTigerAdapter(model)
    true_state = model.sample_initial_state(env_rng)
    common_belief = list(model.init_belief)
    histories = {0: tuple(), 1: tuple()}
    pending_history = []
    active_policy = None
    total_reward = 0.0

    for t in range(horizon):
        if active_policy is None:
            remaining = horizon - t
            if guide == "exact":
                rollout_policy = make_exact_rollout_policy(remaining)
                leaf_value_fn = make_exact_leaf_value(remaining)
                default_actions = {
                    0: make_exact_local_default(0, remaining),
                    1: make_exact_local_default(1, remaining),
                }
            elif guide == "qmdp":
                rollout_policy = make_qmdp_rollout_policy(remaining)
                leaf_value_fn = make_qmdp_leaf_value(remaining) if qmdp_leaf else None
                default_actions = {
                    0: make_qmdp_local_default(0, remaining),
                    1: make_qmdp_local_default(1, remaining),
                }
            elif guide == "heuristic":
                rollout_policy = None
                leaf_value_fn = None
                default_actions = {0: tiger.LISTEN, 1: tiger.LISTEN}
            else:
                raise ValueError(f"Unknown guide: {guide}")
            planner = SDecMCTS(
                robot_ids=[0, 1],
                root_belief=common_belief,
                model=adapter,
                comm_model=make_comm_model(seed + 1009 * t, sync_period),
                horizon=remaining,
                gamma=gamma,
                cp=cp,
                rollout_policy=rollout_policy,
                leaf_value_fn=leaf_value_fn,
                comm_transition_fn=(
                    tiger_gated_comm_transition
                    if sync_mode == "both-listen"
                    else None
                ),
                seed=seed + 7919 * t,
                max_tree_depth=max_tree_depth,
            )
            planner.run(iterations)
            active_policy = planner.extract_policy(default_actions=default_actions, min_edge_visits=min_edge_visits)
            histories = {0: tuple(), 1: tuple()}

        depth_since_sync = len(histories[0])
        action_dict = active_policy.joint_action_from_histories(
            [0, 1],
            depth_since_sync,
            histories,
        )
        a0, a1 = int(action_dict[0]), int(action_dict[1])
        joint_a = model.joint_action(a0, a1)

        reward = model.reward(true_state, joint_a)
        total_reward += reward
        next_state = model.sample_next_state(true_state, joint_a, env_rng)
        joint_o = model.sample_joint_obs(next_state, joint_a, env_rng)
        o0, o1 = model.split_obs(joint_o)
        true_state = next_state

        histories[0] = histories[0] + ((a0, o0),)
        histories[1] = histories[1] + ((a1, o1),)
        pending_history.append((joint_a, joint_o))

        if should_sync(sync_mode, sync_period, t, a0, a1):
            for hist_joint_a, hist_joint_o in pending_history:
                common_belief = tiger.update_joint_belief(
                    common_belief,
                    hist_joint_a,
                    hist_joint_o,
                    model,
                )
            pending_history = []
            active_policy = None

    return total_reward


def run_batch(args) -> List[float]:
    return [
        simulate_episode(
            horizon=args.horizon,
            iterations=args.iterations,
            sync_period=args.sync_period,
            sync_mode=args.sync_mode,
            seed=args.seed + episode,
            cp=args.cp,
            gamma=args.gamma,
            qmdp_leaf=args.qmdp_leaf,
            guide=args.guide,
            min_edge_visits=args.min_edge_visits,
            max_tree_depth=args.max_tree_depth if args.max_tree_depth > 0 else None,
        )
        for episode in range(args.episodes)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--sync-period", type=int, default=2)
    parser.add_argument(
        "--sync-mode",
        choices=["both-listen", "periodic", "none"],
        default="both-listen",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cp", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--min-edge-visits", type=int, default=1,
                        help="Minimum edge visits to include in policy extraction.")
    parser.add_argument("--max-tree-depth", type=int, default=0,
                        help="Max depth the MCTS tree grows before always using rollout/leaf. 0 = unlimited.")
    parser.add_argument(
        "--qmdp-leaf",
        action="store_true",
        help="Use optimistic QMDP value as the leaf evaluator instead of sampled QMDP-guided rollouts.",
    )
    parser.add_argument(
        "--guide",
        choices=["exact", "qmdp", "heuristic"],
        default="exact",
        help="Tiger-specific guide used for rollouts, leaf values, and fallback defaults.",
    )
    args = parser.parse_args()

    returns = run_batch(args)
    mean = statistics.fmean(returns)
    stderr = (
        statistics.stdev(returns) / (len(returns) ** 0.5)
        if len(returns) > 1
        else 0.0
    )
    print("SDecMCTS Tiger prototype")
    print(
        f"horizon={args.horizon} episodes={args.episodes} "
        f"iterations={args.iterations} sync_mode={args.sync_mode} "
        f"sync_period={args.sync_period} guide={args.guide} seed={args.seed}"
    )
    print(f"mean_return={mean:.3f} stderr={stderr:.3f}")
    print(f"returns={returns}")


if __name__ == "__main__":
    main()
