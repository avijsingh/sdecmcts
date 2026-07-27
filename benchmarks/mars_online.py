"""
mars_online.py
--------------
Online/receding-horizon Mars benchmark for ObsDecMCTS.

The domain matrices are loaded from ``mars.data`` using the same action,
observation, transition, reward, and initial-belief conventions as the RSSDA
Mars driver:

    state        = rover0_pos * 16 + rover1_pos
    joint action = a0 + 6 * a1
    joint obs    = o0 + 8 * o1
    init belief  = state 0 with probability 1

Execution follows the Tiger/Medevac online benchmark pattern:

    plan an observation-conditioned policy tree for the remaining horizon
    execute only the root action
    sample transition and observation from mars.data
    update each rover belief using its local observation
    optionally synchronize beliefs at explicit communication events
    replan

This file benchmarks ObsDecMCTS as a sealed implementation. It only supplies a
generative adapter, exact Bayes filters, action priors, and the episode driver.
"""

from __future__ import annotations

import argparse
import bisect
from functools import lru_cache
from itertools import accumulate
import math
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from obs_decmcts import ObsDecMCTS, ObsDecMCTSTeam, StepResult
from belief_obs_decmcts import BeliefObsDecMCTS, BeliefObsDecMCTSTeam


N_AGENTS = 2
GRID = 4
N_STATES = 256
ACT_PER_AGENT = 6
OBS_PER_AGENT = 8
N_ACTS = ACT_PER_AGENT ** N_AGENTS
N_OBS = OBS_PER_AGENT ** N_AGENTS

DEFAULT_HORIZON = 6
DEFAULT_DATA_FILE = Path(__file__).resolve().with_name("mars.data")

ACTION_NAME = {
    0: "A0",
    1: "A1",
    2: "A2",
    3: "A3",
    4: "A4",
    5: "A5",
}

OBS_NAME = {i: f"O{i}" for i in range(OBS_PER_AGENT)}


def pos_to_xy(pos: int) -> Tuple[int, int]:
    return pos % GRID, pos // GRID


def state_to_positions(s: int) -> Tuple[int, int]:
    return s // 16, s % 16


def state_to_xy(s: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    p0, p1 = state_to_positions(s)
    return pos_to_xy(p0), pos_to_xy(p1)


def joint_action(a0: int, a1: int) -> int:
    return a0 + ACT_PER_AGENT * a1


def split_action(a: int) -> Tuple[int, int]:
    return a % ACT_PER_AGENT, a // ACT_PER_AGENT


def joint_obs(o0: int, o1: int) -> int:
    return o0 + OBS_PER_AGENT * o1


def split_obs(o: int) -> Tuple[int, int]:
    return o % OBS_PER_AGENT, o // OBS_PER_AGENT


def mars_right_band_triggers() -> List[int]:
    states: List[int] = []
    for y0 in range(GRID):
        for x0 in (2, 3):
            for y1 in range(GRID):
                for x1 in (2, 3):
                    sid = ((y0 * GRID + x0) * 16) + (y1 * GRID + x1)
                    states.append(sid)
    return sorted(states)


def mars_chebyshev1_triggers() -> List[int]:
    states: List[int] = []
    for y0 in range(GRID):
        for x0 in range(GRID):
            for y1 in range(GRID):
                for x1 in range(GRID):
                    if max(abs(x0 - x1), abs(y0 - y1)) <= 1:
                        sid = ((y0 * GRID + x0) * 16) + (y1 * GRID + x1)
                        states.append(sid)
    return sorted(set(states))


RIGHT_BAND_TRIGGERS = set(mars_right_band_triggers())
CHEBYSHEV1_TRIGGERS = set(mars_chebyshev1_triggers())


class MarsModel:
    """
    Dense T/O/R Mars model loaded from RSSDA-format mars.data.
    """

    def __init__(self, data_file: Path = DEFAULT_DATA_FILE):
        nsq = N_STATES * N_STATES
        nso = N_STATES * N_OBS

        self.T: List[float] = [0.0] * (N_ACTS * nsq)
        self.O: List[float] = [0.0] * (N_ACTS * nso)
        self.R: List[float] = [0.0] * (N_ACTS * N_STATES)

        self.init_belief: List[float] = [0.0] * N_STATES
        self.init_belief[0] = 1.0
        self.init_state = 0

        self.n_states = N_STATES
        self.n_obs = N_OBS
        self.act_per_agent = ACT_PER_AGENT
        self.obs_per_agent = OBS_PER_AGENT

        self._load(data_file)

    def _load(self, path: Path) -> None:
        with open(path, "r") as f:
            for line in f:
                d = line.split()
                if not d:
                    continue

                kind = d[0][0]
                if kind == "T":
                    a0, a1 = int(d[1]), int(d[2])
                    s = int(d[4])
                    sp = int(d[6])
                    p = float(d[8])
                    a = joint_action(a0, a1)
                    self.T[a * N_STATES * N_STATES + s * N_STATES + sp] = p

                elif kind == "O":
                    a0, a1 = int(d[1]), int(d[2])
                    sp = int(d[4])
                    o0, o1 = int(d[6]), int(d[7])
                    p = float(d[9])
                    a = joint_action(a0, a1)
                    o = joint_obs(o0, o1)
                    self.O[a * N_STATES * N_OBS + sp * N_OBS + o] = p

                elif kind == "R":
                    a0, a1 = int(d[1]), int(d[2])
                    s = int(d[4])
                    r = float(d[-1])
                    a = joint_action(a0, a1)
                    self.R[a * N_STATES + s] = r

    def transition_prob(self, s: int, a: int, sp: int) -> float:
        return self.T[a * N_STATES * N_STATES + s * N_STATES + sp]

    def obs_prob(self, sp: int, a: int, o: int) -> float:
        return self.O[a * N_STATES * N_OBS + sp * N_OBS + o]

    def reward(self, s: int, a: int) -> float:
        return self.R[a * N_STATES + s]

    def sample_initial_state(self, rng: random.Random) -> int:
        return self.init_state

    def sample_next_state(self, s: int, a: int, rng: random.Random) -> int:
        r = rng.random()
        cum = 0.0
        base = a * N_STATES * N_STATES + s * N_STATES

        for sp in range(N_STATES):
            cum += self.T[base + sp]
            if r <= cum:
                return sp

        return s

    def sample_joint_obs(self, sp: int, a: int, rng: random.Random) -> int:
        r = rng.random()
        cum = 0.0
        base = a * N_STATES * N_OBS + sp * N_OBS

        for o in range(N_OBS):
            cum += self.O[base + o]
            if r <= cum:
                return o

        return 0

    def local_obs_prob(self, rid: int, sp: int, a: int, local_o: int) -> float:
        total = 0.0
        for o in range(N_OBS):
            o0, o1 = split_obs(o)
            if (rid == 0 and o0 == local_o) or (rid == 1 and o1 == local_o):
                total += self.obs_prob(sp, a, o)
        return total


class MarsObsModelAdapter:
    def __init__(self, model: MarsModel):
        self.model = model
        # id(belief) -> (belief ref, cumulative sums); the ref keeps the list
        # alive so ids are never recycled while cached.
        self._belief_cums: Dict[int, Tuple[Sequence[float], List[float]]] = {}

    def sample_state_from_belief(self, belief: Sequence[float], rng: random.Random) -> int:
        cached = self._belief_cums.get(id(belief))
        if cached is not None and cached[0] is belief:
            cum = cached[1]
        else:
            cum = list(accumulate(belief))
            if len(self._belief_cums) > 256:
                self._belief_cums.clear()
            self._belief_cums[id(belief)] = (belief, cum)

        r = rng.random()
        if not cum:
            return N_STATES - 1
        idx = bisect.bisect_left(cum, r)
        return idx if idx < len(cum) else N_STATES - 1

    def step(self, state: int, joint_a: int, rng: random.Random) -> StepResult:
        reward = self.model.reward(state, joint_a)
        next_state = self.model.sample_next_state(state, joint_a, rng)
        obs = self.model.sample_joint_obs(next_state, joint_a, rng)
        return StepResult(next_state=next_state, joint_obs=obs, reward=reward)

    def split_obs(self, obs: int) -> Tuple[int, int]:
        return split_obs(obs)

    def joint_action_from_dict(self, actions: Dict[int, int]) -> int:
        return joint_action(actions[0], actions[1])

    def update_belief(
        self,
        belief: Sequence[float],
        joint_a: int,
        local_o: int,
        robot_id: int,
    ) -> List[float]:
        return update_local_belief(belief, joint_a, local_o, robot_id, self.model)


def normalize_belief(b: Sequence[float]) -> List[float]:
    total = float(sum(b))
    if total <= 1e-15:
        return [1.0 / N_STATES] * N_STATES
    return [float(x) / total for x in b]


def predict_belief_open_loop(
    belief: Sequence[float],
    a: int,
    model: MarsModel,
) -> List[float]:
    out = [0.0] * N_STATES

    for s, b_s in enumerate(belief):
        if b_s <= 0.0:
            continue

        base = a * N_STATES * N_STATES + s * N_STATES
        for sp in range(N_STATES):
            p = model.T[base + sp]
            if p > 0.0:
                out[sp] += b_s * p

    return normalize_belief(out)


def update_local_belief(
    belief: Sequence[float],
    a: int,
    local_o: int,
    rid: int,
    model: MarsModel,
) -> List[float]:
    pred = predict_belief_open_loop(belief, a, model)
    post = [
        pred[sp] * model.local_obs_prob(rid, sp, a, local_o)
        for sp in range(N_STATES)
    ]
    return normalize_belief(post)


def update_joint_belief(
    belief: Sequence[float],
    a: int,
    obs: int,
    model: MarsModel,
) -> List[float]:
    pred = predict_belief_open_loop(belief, a, model)
    post = [
        pred[sp] * model.obs_prob(sp, a, obs)
        for sp in range(N_STATES)
    ]
    return normalize_belief(post)


def expected_one_step_reward(
    belief: Sequence[float],
    a0: int,
    a1: int,
    model: MarsModel,
) -> float:
    a = joint_action(a0, a1)
    return sum(belief[s] * model.reward(s, a) for s in range(N_STATES))


@lru_cache(maxsize=None)
def _qmdp_values_for_path(data_file: str, horizon: int) -> Tuple[float, ...]:
    model = MarsModel(Path(data_file))
    values = [0.0] * N_STATES

    for _ in range(horizon):
        next_values: List[float] = []

        for s in range(N_STATES):
            best = -math.inf

            for a in range(N_ACTS):
                base = a * N_STATES * N_STATES + s * N_STATES
                val = model.reward(s, a)
                val += sum(
                    model.T[base + sp] * values[sp]
                    for sp in range(N_STATES)
                )

                if val > best:
                    best = val

            next_values.append(best)

        values = next_values

    return tuple(values)


@lru_cache(maxsize=None)
def _qmdp_action_values_for_path(
    data_file: str,
    remaining_horizon: int,
) -> Tuple[Tuple[float, ...], ...]:
    model = MarsModel(Path(data_file))
    future_values = (
        _qmdp_values_for_path(data_file, remaining_horizon - 1)
        if remaining_horizon > 1
        else (0.0,) * N_STATES
    )

    rows: List[Tuple[float, ...]] = []
    for a in range(N_ACTS):
        vals: List[float] = []
        for s in range(N_STATES):
            base = a * N_STATES * N_STATES + s * N_STATES
            q_s = model.reward(s, a)
            q_s += sum(
                model.T[base + sp] * future_values[sp]
                for sp in range(N_STATES)
            )
            vals.append(q_s)
        rows.append(tuple(vals))

    return tuple(rows)


def best_qmdp_joint_action(
    belief: Sequence[float],
    remaining_horizon: int,
    model: MarsModel,
) -> int:
    action_values = _qmdp_action_values_for_path(
        str(DEFAULT_DATA_FILE),
        remaining_horizon,
    )

    best_a = 0
    best_v = -math.inf

    for a in range(N_ACTS):
        # TESTING: previous implementation recomputed the transition backup
        # for every belief/default-policy query. The cached action_values table
        # is the same QMDP backup, reused across histories.
        #
        # total = 0.0
        # for s, b_s in enumerate(belief):
        #     if b_s <= 0.0:
        #         continue
        #     base = a * N_STATES * N_STATES + s * N_STATES
        #     q_s = model.reward(s, a)
        #     q_s += sum(
        #         model.T[base + sp] * future_values[sp]
        #         for sp in range(N_STATES)
        #     )
        #     total += b_s * q_s
        total = sum(
            b_s * action_values[a][s]
            for s, b_s in enumerate(belief)
            if b_s > 0.0
        )

        if total > best_v:
            best_v = total
            best_a = a

    return best_a


def ranked_qmdp_local_actions(
    belief: Sequence[float],
    remaining_horizon: int,
    model: MarsModel,
    rid: int,
    limit: Optional[int] = None,
) -> List[int]:
    action_values = _qmdp_action_values_for_path(
        str(DEFAULT_DATA_FILE),
        remaining_horizon,
    )
    scores: Dict[int, float] = {a: -math.inf for a in range(ACT_PER_AGENT)}

    for joint_a in range(N_ACTS):
        # TESTING: previous implementation recomputed the same QMDP backup
        # inside every ranked-action query.
        #
        # total = 0.0
        # for s, b_s in enumerate(belief):
        #     if b_s <= 0.0:
        #         continue
        #     base = joint_a * N_STATES * N_STATES + s * N_STATES
        #     q_s = model.reward(s, joint_a)
        #     q_s += sum(
        #         model.T[base + sp] * future_values[sp]
        #         for sp in range(N_STATES)
        #     )
        #     total += b_s * q_s
        total = sum(
            b_s * action_values[joint_a][s]
            for s, b_s in enumerate(belief)
            if b_s > 0.0
        )

        a0, a1 = split_action(joint_a)
        local_a = a0 if rid == 0 else a1
        scores[local_a] = max(scores[local_a], total)

    ranked = [
        a for a, _score in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    ]
    if limit is not None and limit > 0:
        ranked = ranked[:limit]
    return ranked


def average_belief(beliefs: Dict[int, Sequence[float]]) -> List[float]:
    out = [0.0] * N_STATES
    n = max(len(beliefs), 1)

    for belief in beliefs.values():
        for s, p in enumerate(belief):
            out[s] += p / n

    return normalize_belief(out)


def mix_beliefs(
    first: Sequence[float],
    second: Sequence[float],
    second_weight: float,
) -> List[float]:
    w = min(1.0, max(0.0, float(second_weight)))
    return normalize_belief([
        (1.0 - w) * first[s] + w * second[s]
        for s in range(N_STATES)
    ])


def heuristic_root_action(
    root_belief: Sequence[float],
    rid: int,
    model: MarsModel,
    teammate_default: int,
    remaining_horizon: int = 1,
    mode: str = "qmdp",
) -> int:
    if mode == "qmdp":
        a0, a1 = split_action(
            best_qmdp_joint_action(root_belief, remaining_horizon, model)
        )
        return a0 if rid == 0 else a1

    best_a = 0
    best_v = -math.inf

    for my_a in range(ACT_PER_AGENT):
        if rid == 0:
            v = expected_one_step_reward(root_belief, my_a, teammate_default, model)
        else:
            v = expected_one_step_reward(root_belief, teammate_default, my_a, model)

        if v > best_v:
            best_v = v
            best_a = my_a

    return best_a


def make_mars_legal_actions_fn(full_action_space: bool = True):
    def legal_actions(_history, _depth: int) -> Sequence[int]:
        if full_action_space:
            return list(range(ACT_PER_AGENT))
        return [0, 1, 2, 3, 5]

    return legal_actions


def mars_belief_from_local_history(
    root_belief: Sequence[float],
    history,
    rid: int,
    model: MarsModel,
    teammate_default: int,
) -> List[float]:
    belief = list(root_belief)

    for own_action, local_obs in history:
        if rid == 0:
            joint_a = joint_action(own_action, teammate_default)
        else:
            joint_a = joint_action(teammate_default, own_action)
        belief = update_local_belief(belief, joint_a, local_obs, rid, model)

    return belief


def make_mars_belief_legal_actions_fn(
    model: MarsModel,
    remaining_horizon: int,
    rid: int,
    qmdp_reference_belief: Optional[Sequence[float]] = None,
    full_action_space: bool = True,
    qmdp_action_limit: int = 0,
    qmdp_legal_belief: str = "local",
):
    def legal_actions(belief: Sequence[float], depth: int) -> Sequence[int]:
        rem = max(1, remaining_horizon - depth)
        qmdp_belief = belief
        if (
            qmdp_legal_belief in {"average", "common", "mixed"}
            and depth == 0
            and qmdp_reference_belief is not None
        ):
            qmdp_belief = qmdp_reference_belief

        if qmdp_action_limit > 0:
            if qmdp_action_limit == 1:
                q_a = best_qmdp_joint_action(qmdp_belief, rem, model)
                return [split_action(q_a)[rid]]

            return ranked_qmdp_local_actions(
                belief=qmdp_belief,
                remaining_horizon=rem,
                model=model,
                rid=rid,
                limit=qmdp_action_limit,
            )

        if full_action_space:
            return list(range(ACT_PER_AGENT))

        q_a = best_qmdp_joint_action(qmdp_belief, rem, model)
        return [split_action(q_a)[rid]]

    return legal_actions


def make_mars_default_action_fn(
    root_belief: Sequence[float],
    rid: int,
    model: MarsModel,
    teammate_default: int,
    remaining_horizon: int,
    mode: str,
):
    action_cache: Dict[Any, int] = {}

    def default_action(history) -> int:
        # TESTING: previous obs default used QMDP only at the root and then
        # fell back to teammate_default for every deeper observation history.
        # That ignored the observations stored in ObsDecMCTS nodes.
        #
        # if history == ():
        #     return root_default
        # return teammate_default
        if history in action_cache:
            return action_cache[history]

        if len(history) > 1:
            # TESTING: full history-conditioned defaults at every rollout
            # depth were too slow for Mars because sampled observation
            # histories are nearly all unique. Keep the observation-node fix
            # at the first post-root branch and use the cheap prior deeper.
            action_cache[history] = teammate_default
            return teammate_default

        local_belief = mars_belief_from_local_history(
            root_belief=root_belief,
            history=history,
            rid=rid,
            model=model,
            teammate_default=teammate_default,
        )
        rem = max(1, remaining_horizon - len(history))
        # TESTING: using full QMDP at every deep sampled observation history
        # made ObsDecMCTS rollouts explode in runtime. Keep the requested root
        # default mode, but use the cheap local one-step heuristic after the
        # root; this is still history-conditioned and only affects unexplored
        # default-policy branches.
        default_mode = mode if history == () else "one-step"
        action = heuristic_root_action(
            root_belief=local_belief,
            rid=rid,
            model=model,
            teammate_default=teammate_default,
            remaining_horizon=rem,
            mode=default_mode,
        )
        action_cache[history] = action
        return action

    return default_action


def make_mars_belief_default_action_fn(
    rid: int,
    model: MarsModel,
    remaining_horizon: int,
    mode: str,
    teammate_default: int,
):
    def default_action(belief: Sequence[float]) -> int:
        return heuristic_root_action(
            root_belief=belief,
            rid=rid,
            model=model,
            teammate_default=teammate_default,
            remaining_horizon=remaining_horizon,
            mode=mode,
        )

    return default_action


class HeuristicExpansionBeliefObsDecMCTS(BeliefObsDecMCTS):
    """
    Mars-local heuristic expansion wrapper.

    This leaves the full legal action set intact, but when a node is first
    expanded it tries the benchmark's default action before random unexplored
    actions. For Mars, the default action can be QMDP-derived from the current
    belief, which is analogous to using QMDP as search guidance rather than as
    a legal-action filter or execution guard.
    """

    def _select_or_expand_action(self, node):
        if node.untried_actions:
            default_action = self.default_action_fn(node.belief)
            if default_action in node.untried_actions:
                action = default_action
            else:
                action = self.rng.choice(node.untried_actions)
            node.add_action_edge(action)
            return action

        return max(node.actions.values(), key=lambda edge: self._ucb(edge, node)).action


def obs_root_action_masses(planner: ObsDecMCTS) -> Dict[str, float]:
    masses = {a: 0.0 for a in range(ACT_PER_AGENT)}

    for key, p in planner.q.items():
        for hist, action in key:
            if hist == ():
                masses[int(action)] = masses.get(int(action), 0.0) + p
                break

    return {ACTION_NAME[a]: round(masses[a], 4) for a in range(ACT_PER_AGENT)}


def belief_root_action_masses(planner: BeliefObsDecMCTS) -> Dict[str, float]:
    masses = {a: 0.0 for a in range(ACT_PER_AGENT)}
    root_key = planner.root.belief_key

    for policy_key, p in planner.q.items():
        root_action = None
        for belief_key, action in policy_key:
            if belief_key == root_key:
                root_action = int(action)
                break

        if root_action is not None:
            masses[root_action] += p

    return {ACTION_NAME[a]: round(masses[a], 4) for a in range(ACT_PER_AGENT)}


def obs_root_edge_stats(planner: ObsDecMCTS) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    for a, edge in planner.root.actions.items():
        out[ACTION_NAME[int(a)]] = {
            "visits": edge.visits,
            "q": round(edge.q(), 3),
            "disc_visits": round(edge.disc_visits, 3),
            "disc_q": round(edge.disc_q(), 3),
            "obs_children": [OBS_NAME[int(o)] for o in edge.obs_children.keys()],
        }

    return out


def obs_policy_support_debug(planner: ObsDecMCTS) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    for key, p in sorted(planner.q.items(), key=lambda kv: kv[1], reverse=True):
        root_action: Optional[int] = None
        for hist, action in key:
            if hist == ():
                root_action = int(action)
                break

        out.append({
            "p": round(p, 4),
            "root": ACTION_NAME[root_action] if root_action is not None else None,
            "num_entries": len(key),
        })

    return out


def belief_support_summary(belief: Sequence[float], k: int = 5) -> str:
    top = sorted(
        [(p, s) for s, p in enumerate(belief) if p > 1e-9],
        reverse=True,
    )[:k]
    parts = [f"{state_to_xy(s)}:{p:.3f}" for p, s in top]
    support = sum(1 for p in belief if p > 1e-9)
    return f"support={support}, top={parts}"


def should_env_communicate(args, t: int, next_state: int) -> bool:
    if args.env_comm_mode == "none":
        return False

    if args.env_comm_mode == "periodic":
        return args.env_comm_period > 0 and (t + 1) % args.env_comm_period == 0

    if args.env_comm_mode == "right-band":
        return next_state in RIGHT_BAND_TRIGGERS

    if args.env_comm_mode == "chebyshev1":
        return next_state in CHEBYSHEV1_TRIGGERS

    if args.env_comm_mode == "collocated":
        p0, p1 = state_to_positions(next_state)
        return p0 == p1

    raise ValueError(f"Unknown env_comm_mode: {args.env_comm_mode}")


def run_obs_planning_step(
    beliefs: Dict[int, List[float]],
    remaining_horizon: int,
    model: MarsModel,
    args,
    seed: int,
):
    planners: Dict[int, ObsDecMCTS] = {}
    adapter = MarsObsModelAdapter(model)
    default_action_fns_by_robot = {
        rid: make_mars_default_action_fn(
            root_belief=beliefs[rid],
            rid=rid,
            model=model,
            teammate_default=args.default_action,
            remaining_horizon=remaining_horizon,
            mode=args.default_policy,
        )
        for rid in range(N_AGENTS)
    }

    for rid in range(N_AGENTS):
        planners[rid] = ObsDecMCTS(
            robot_id=rid,
            robot_ids=[0, 1],
            root_belief=beliefs[rid],
            model=adapter,
            legal_actions_fn=make_mars_legal_actions_fn(
                full_action_space=args.full_action_space,
            ),
            default_action_fn=default_action_fns_by_robot[rid],
            default_action_fns_by_robot=default_action_fns_by_robot,
            gamma=args.gamma,
            cp=args.cp,
            horizon=remaining_horizon,
            tau=args.tau,
            num_policies=args.num_seq,
            num_samples=args.num_samples,
            beta_init=args.beta_init,
            beta_decay=args.beta_decay,
            alpha=args.alpha,
            prefer_default_expansion=True,
            seed=seed + 1009 * rid,
        )

    team = ObsDecMCTSTeam(planners)
    team.iterate_and_communicate(
        n_outer=args.outer_iters,
        comm_period=args.comm_period,
    )

    if getattr(args, "debug_obs", False):
        for rid, planner in planners.items():
            print(
                f"DEBUG obs planner {rid}: "
                f"root_actions={[ACTION_NAME[int(a)] for a in planner.root.actions.keys()]} "
                f"root_visits={planner.root.visits} "
                f"num_X_hat={len(planner.X_hat)} "
                f"num_q={len(planner.q)} "
                f"entropy={team.entropies().get(rid, 0.0):.3f} "
                f"min_reward={planner.min_reward:.3f} "
                f"max_reward={planner.max_reward:.3f}",
                flush=True,
            )
            print(f"DEBUG obs root edges {rid}: {obs_root_edge_stats(planner)}", flush=True)
            print(f"DEBUG obs policy support {rid}: {obs_policy_support_debug(planner)}", flush=True)

    policies = team.best_policies()
    raw_actions = team.best_actions(source=args.action_source)
    actions = {rid: int(raw_actions[rid]) for rid in range(N_AGENTS)}

    if getattr(args, "guard_actions", False):
        guarded_joint = best_qmdp_joint_action(
            average_belief(beliefs),
            remaining_horizon,
            model,
        )
        g0, g1 = split_action(guarded_joint)
        actions = {0: g0, 1: g1}

    root_masses = {rid: obs_root_action_masses(planner) for rid, planner in planners.items()}

    return actions, raw_actions, policies, team.entropies(), root_masses


def run_belief_obs_planning_step(
    beliefs: Dict[int, List[float]],
    common_belief: Sequence[float],
    remaining_horizon: int,
    model: MarsModel,
    args,
    seed: int,
):
    planners: Dict[int, BeliefObsDecMCTS] = {}
    adapter = MarsObsModelAdapter(model)
    avg_belief = average_belief(beliefs)
    if args.qmdp_legal_belief == "average":
        qmdp_reference_belief = avg_belief
    elif args.qmdp_legal_belief == "common":
        qmdp_reference_belief = list(common_belief)
    elif args.qmdp_legal_belief == "mixed":
        qmdp_reference_belief = mix_beliefs(
            common_belief,
            avg_belief,
            args.qmdp_average_weight,
        )
    else:
        qmdp_reference_belief = None
    default_action_fns_by_robot = {
        rid: make_mars_belief_default_action_fn(
            rid=rid,
            model=model,
            remaining_horizon=remaining_horizon,
            mode=args.default_policy,
            teammate_default=args.default_action,
        )
        for rid in range(N_AGENTS)
    }

    for rid in range(N_AGENTS):
        planner_cls = (
            HeuristicExpansionBeliefObsDecMCTS
            if args.heuristic_expansion
            else BeliefObsDecMCTS
        )
        planners[rid] = planner_cls(
            robot_id=rid,
            robot_ids=[0, 1],
            root_belief=beliefs[rid],
            root_beliefs_by_robot=beliefs,
            model=adapter,
            legal_actions_fn=make_mars_belief_legal_actions_fn(
                model=model,
                remaining_horizon=remaining_horizon,
                rid=rid,
                qmdp_reference_belief=qmdp_reference_belief,
                full_action_space=args.full_action_space,
                qmdp_action_limit=args.qmdp_action_limit,
                qmdp_legal_belief=args.qmdp_legal_belief,
            ),
            default_action_fn=default_action_fns_by_robot[rid],
            default_action_fns_by_robot=default_action_fns_by_robot,
            share_belief_nodes=args.belief_share_nodes,
            gamma=args.gamma,
            cp=args.cp,
            horizon=remaining_horizon,
            tau=args.tau,
            num_policies=args.num_seq,
            num_samples=args.num_samples,
            beta_init=args.beta_init,
            beta_decay=args.beta_decay,
            alpha=args.alpha,
            seed=seed + 1009 * rid,
        )

    team = BeliefObsDecMCTSTeam(planners)
    team.iterate_and_communicate(
        n_outer=args.outer_iters,
        comm_period=args.comm_period,
    )

    policies = team.best_policies()
    raw_actions = team.best_actions(beliefs=beliefs, source=args.action_source)
    actions = {rid: int(raw_actions[rid]) for rid in range(N_AGENTS)}

    if getattr(args, "guard_actions", False):
        guarded_joint = best_qmdp_joint_action(
            average_belief(beliefs),
            remaining_horizon,
            model,
        )
        g0, g1 = split_action(guarded_joint)
        actions = {0: g0, 1: g1}

    root_masses = {
        rid: belief_root_action_masses(planner)
        for rid, planner in planners.items()
    }

    return actions, raw_actions, policies, team.entropies(), root_masses


def simulate_online_episode(
    model: MarsModel,
    args,
    episode_seed: int,
    verbose: bool = False,
) -> float:
    rng = random.Random(episode_seed)
    true_state = model.sample_initial_state(rng)

    common_belief = list(model.init_belief)
    beliefs = {
        0: list(model.init_belief),
        1: list(model.init_belief),
    }
    pending_history: List[Tuple[int, int]] = []

    total_reward = 0.0
    discount = 1.0

    def flush_pending_common_belief() -> None:
        nonlocal common_belief, pending_history

        for hist_a, hist_o in pending_history:
            common_belief = update_joint_belief(common_belief, hist_a, hist_o, model)

        beliefs[0] = list(common_belief)
        beliefs[1] = list(common_belief)
        pending_history = []

    if verbose:
        print(f"initial true_state={state_to_xy(true_state)}")

    planner_kind = getattr(args, "planner", "obs")

    for t in range(args.horizon):
        remaining = args.horizon - t

        if planner_kind == "belief-obs":
            actions, raw_actions, policies, entropies, root_masses = run_belief_obs_planning_step(
                beliefs=beliefs,
                common_belief=common_belief,
                remaining_horizon=remaining,
                model=model,
                args=args,
                seed=episode_seed + 7919 * t,
            )
        else:
            actions, raw_actions, policies, entropies, root_masses = run_obs_planning_step(
                beliefs=beliefs,
                remaining_horizon=remaining,
                model=model,
                args=args,
                seed=episode_seed + 7919 * t,
            )

        a0, a1 = actions[0], actions[1]
        a = joint_action(a0, a1)

        reward = model.reward(true_state, a)
        total_reward += discount * reward
        discount *= args.gamma

        next_state = model.sample_next_state(true_state, a, rng)
        obs = model.sample_joint_obs(next_state, a, rng)
        o0, o1 = split_obs(obs)

        beliefs[0] = update_local_belief(beliefs[0], a, o0, 0, model)
        beliefs[1] = update_local_belief(beliefs[1], a, o1, 1, model)
        pending_history.append((a, obs))

        if should_env_communicate(args, t, next_state):
            flush_pending_common_belief()

        if verbose:
            print(
                f"t={t:02d} rem={remaining:02d} "
                f"a=({ACTION_NAME[a0]},{ACTION_NAME[a1]}) "
                f"raw=({ACTION_NAME[int(raw_actions[0])]},{ACTION_NAME[int(raw_actions[1])]}) "
                f"r={reward:7.3f} "
                f"obs=({OBS_NAME[o0]},{OBS_NAME[o1]}) "
                f"next={state_to_xy(next_state)} "
                f"H={{{0}: {entropies.get(0, 0.0):.3f}, {1}: {entropies.get(1, 0.0):.3f}}}"
            )
            print(f"    belief0={belief_support_summary(beliefs[0])}")
            print(f"    belief1={belief_support_summary(beliefs[1])}")
            print(f"    root_mass={root_masses}")
            if planner_kind == "belief-obs":
                key0 = (0, tuple(round(x, 12) for x in beliefs[0]))
                key1 = (0, tuple(round(x, 12) for x in beliefs[1]))
                print(f"    policy0_root={ACTION_NAME[int(policies[0].action(beliefs[0], key0))]}")
                print(f"    policy1_root={ACTION_NAME[int(policies[1].action(beliefs[1], key1))]}")
            else:
                print(f"    policy0_root={ACTION_NAME[int(policies[0].action(()))]}")
                print(f"    policy1_root={ACTION_NAME[int(policies[1].action(()))]}")

        true_state = next_state

    return total_reward


def confidence_interval_95(values: Sequence[float]) -> Tuple[float, float]:
    n = len(values)
    mean = sum(values) / n

    if n <= 1:
        return mean, 0.0

    var = sum((x - mean) ** 2 for x in values) / (n - 1)
    se = math.sqrt(var / n)
    return mean, 1.96 * se


def run_episode_worker(payload):
    ep, args_dict = payload
    args = argparse.Namespace(**args_dict)
    model = MarsModel()

    ret = simulate_online_episode(
        model=model,
        args=args,
        episode_seed=args.seed + 104729 * ep,
        verbose=False,
    )

    return ep, ret


def run_initial_planning_timing(model: MarsModel, args) -> None:
    beliefs = {
        0: list(model.init_belief),
        1: list(model.init_belief),
    }
    t0 = time.time()

    if args.planner == "belief-obs":
        actions, raw_actions, _policies, entropies, root_masses = run_belief_obs_planning_step(
            beliefs=beliefs,
            common_belief=list(model.init_belief),
            remaining_horizon=args.horizon,
            model=model,
            args=args,
            seed=args.seed,
        )
    else:
        actions, raw_actions, _policies, entropies, root_masses = run_obs_planning_step(
            beliefs=beliefs,
            remaining_horizon=args.horizon,
            model=model,
            args=args,
            seed=args.seed,
        )

    elapsed = time.time() - t0
    a0, a1 = int(actions[0]), int(actions[1])
    raw0, raw1 = int(raw_actions[0]), int(raw_actions[1])

    print("Initial Mars planning timing")
    print(f"horizon={args.horizon}")
    print(f"planning_time={elapsed:.6f}s")
    print(f"actions=({ACTION_NAME[a0]},{ACTION_NAME[a1]})")
    print(f"raw_actions=({ACTION_NAME[raw0]},{ACTION_NAME[raw1]})")
    print(f"entropies={{{0}: {entropies.get(0, 0.0):.6f}, {1}: {entropies.get(1, 0.0):.6f}}}")
    print(f"root_mass={root_masses}")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--episodes", type=int, default=50)

    parser.add_argument("--outer-iters", type=int, default=30)
    parser.add_argument("--tau", type=int, default=40)
    parser.add_argument("--num-seq", type=int, default=30)
    parser.add_argument("--num-samples", type=int, default=30)
    parser.add_argument("--comm-period", type=int, default=1)

    parser.add_argument(
        "--env-comm-mode",
        choices=["none", "periodic", "right-band", "chebyshev1", "collocated"],
        default="none",
        help="Execution-time observation sharing rule.",
    )
    parser.add_argument(
        "--env-comm-period",
        type=int,
        default=0,
        help="Period for --env-comm-mode periodic. Use 0 for no periodic sync.",
    )

    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--cp", type=float, default=0.5)
    parser.add_argument("--beta-init", type=float, default=2.0)
    parser.add_argument("--beta-decay", type=float, default=0.995)
    parser.add_argument("--alpha", type=float, default=0.2)

    parser.add_argument("--default-action", type=int, default=4, choices=range(ACT_PER_AGENT))
    parser.add_argument(
        "--default-policy",
        choices=["qmdp", "one-step"],
        default="qmdp",
        help="Domain prior used for unexplored Mars policy-tree histories.",
    )
    parser.add_argument(
        "--guard-actions",
        action="store_true",
        help="Execute the QMDP joint action from the current average belief.",
    )
    parser.add_argument(
        "--no-guard-actions",
        dest="guard_actions",
        action="store_false",
        help="Disable the Mars QMDP execution guard.",
    )
    parser.add_argument(
        "--full-action-space",
        action="store_true",
        default=True,
        help="Allow all six Mars actions at every history.",
    )
    parser.add_argument(
        "--restricted-action-space",
        dest="full_action_space",
        action="store_false",
        help="Use only the local component of the QMDP joint action at each belief.",
    )
    parser.add_argument(
        "--qmdp-action-limit",
        type=int,
        default=0,
        help="If >0, expose only the top-k QMDP-ranked local actions to the belief planner.",
    )
    parser.add_argument(
        "--qmdp-legal-belief",
        choices=["local", "average", "common", "mixed"],
        default="average",
        help="Belief used for root QMDP legal-action restriction.",
    )
    parser.add_argument(
        "--qmdp-average-weight",
        type=float,
        default=0.0,
        help="For --qmdp-legal-belief mixed, weight on average private belief vs common belief.",
    )
    parser.add_argument(
        "--action-source",
        choices=["tree", "disc_tree", "policy", "visits", "policy_value", "policy_marginal"],
        default="policy_marginal",
    )
    parser.add_argument(
        "--planner",
        choices=["obs", "belief-obs"],
        default="belief-obs",
        help="Planner backend: history-keyed ObsDecMCTS or belief-indexed ObsDecMCTS.",
    )
    parser.add_argument(
        "--belief-share-nodes",
        action="store_true",
        help="For --planner belief-obs, reuse nodes with the same depth and belief key.",
    )
    parser.add_argument(
        "--heuristic-expansion",
        action="store_true",
        default=True,
        help="For --planner belief-obs, expand the QMDP/default action before random untried actions.",
    )
    parser.add_argument(
        "--random-expansion",
        dest="heuristic_expansion",
        action="store_false",
        help="Use the unmodified belief planner's random untried-action expansion.",
    )

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose-first", action="store_true")
    parser.add_argument("--debug-obs", action="store_true")
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of parallel worker processes. Use -1 for all CPU cores.",
    )
    parser.add_argument(
        "--timing-only",
        action="store_true",
        help="Run one initial-belief planning call and report planning wall time.",
    )

    args = parser.parse_args()
    model = MarsModel()

    print("Online ObsDecMCTS Mars benchmark")
    print(f"horizon={args.horizon}, episodes={args.episodes}")
    print(
        f"planning: outer_iters={args.outer_iters}, tau={args.tau}, "
        f"num_seq={args.num_seq}, num_samples={args.num_samples}"
    )
    print(
        f"env_comm_mode={args.env_comm_mode}, env_comm_period={args.env_comm_period}, "
        f"planner={args.planner}, "
        f"default_action={ACTION_NAME[args.default_action]}, "
        f"default_policy={args.default_policy}, "
        f"full_action_space={args.full_action_space}, "
        f"qmdp_action_limit={args.qmdp_action_limit}, "
        f"qmdp_legal_belief={args.qmdp_legal_belief}, "
        f"qmdp_average_weight={args.qmdp_average_weight}, "
        f"heuristic_expansion={args.heuristic_expansion}, "
        f"action_source={args.action_source}, guard_actions={args.guard_actions}, "
        f"seed={args.seed}"
    )
    if args.env_comm_mode != "none":
        print(
            "WARNING: execution-time communication shares joint observations; "
            "use --env-comm-mode none for a decentralized baseline."
        )
    if abs(args.gamma - 1.0) > 1e-12:
        print("WARNING: gamma != 1.0; RSSDA finite-horizon values are undiscounted.")
    print()

    if args.timing_only:
        run_initial_planning_timing(model, args)
        return

    returns: List[float] = []

    if args.jobs == 1 or args.verbose_first or args.debug_obs:
        for ep in range(args.episodes):
            ret = simulate_online_episode(
                model=model,
                args=args,
                episode_seed=args.seed + 104729 * ep,
                verbose=(args.verbose_first and ep == 0),
            )

            returns.append(ret)

            if (ep + 1) % max(1, args.episodes // 10) == 0 or ep == 0:
                mean, ci = confidence_interval_95(returns)
                print(
                    f"episode {ep + 1:4d}/{args.episodes}: "
                    f"last={ret:8.3f}, mean={mean:8.3f} ± {ci:.3f}",
                    flush=True,
                )

    else:
        n_jobs = os.cpu_count() if args.jobs == -1 else args.jobs
        n_jobs = max(1, int(n_jobs))
        args_dict = vars(args).copy()

        print(f"Running episodes in parallel with {n_jobs} workers", flush=True)

        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            futures = [
                ex.submit(run_episode_worker, (ep, args_dict))
                for ep in range(args.episodes)
            ]

            completed = 0
            for fut in as_completed(futures):
                ep, ret = fut.result()
                returns.append(ret)
                completed += 1

                if completed % max(1, args.episodes // 10) == 0 or completed == 1:
                    mean, ci = confidence_interval_95(returns)
                    print(
                        f"completed {completed:4d}/{args.episodes}: "
                        f"last_ep={ep + 1:4d}, last={ret:8.3f}, "
                        f"mean={mean:8.3f} ± {ci:.3f}",
                        flush=True,
                    )

    mean, ci = confidence_interval_95(returns)
    print()
    print(f"Final mean return: {mean:.5f} ± {ci:.5f} (95% CI)")


if __name__ == "__main__":
    main()
