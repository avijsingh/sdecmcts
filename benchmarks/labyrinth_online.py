"""
Online/receding-horizon Labyrinth benchmark for ObsDecMCTS.

The domain is loaded from Mahdi Al-Husseini's RSSDA labyrinth ``.data`` files.
Two agents start at node 0, the target is hidden uniformly among all non-start
nodes, observations reveal each agent's position and whether it has found the
target, and reward is collected when an informed agent returns to the start.

This file intentionally adapts the benchmark around the existing ObsDecMCTS and
BeliefObsDecMCTS implementations; it does not modify planner internals.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import os
import random
import re
import sys
import time
from itertools import accumulate
from contextlib import nullcontext as _nullcontext
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import obs_decmcts as obs_core
import belief_obs_decmcts as belief_core
from obs_decmcts import ObsDecMCTS, ObsDecMCTSTeam, StepResult
from belief_obs_decmcts import BeliefObsDecMCTS, BeliefObsDecMCTSTeam


N_AGENTS = 2
START_NODE = 0
SUCCESS_REWARD = 100.0
STEP_REWARD = -1.0

BENCHMARK_ALIASES = {
    "extcross9": "1",
    "lopsidedy10": "2",
    "ladder10": "3",
    "maze12": "4",
    "hiddentail11": "5",
    "mesh10": "7",
}

DEFAULT_TABLE_HORIZONS = {
    "extcross9": [6, 7, 8],
    "lopsidedy10": [5, 6, 7],
    "ladder10": [5, 6, 7],
    "maze12": [5, 6, 7],
    "hiddentail11": [4, 5, 6],
    "mesh10": [5, 6, 7],
}

ACTION_NAME: Dict[int, str] = {}
OBS_NAME: Dict[int, str] = {}


class ProfileStats:
    def __init__(self):
        self.times: Dict[str, float] = {}
        self.counts: Dict[str, int] = {}

    def add(self, name: str, elapsed: float, count: int = 1) -> None:
        self.times[name] = self.times.get(name, 0.0) + elapsed
        self.counts[name] = self.counts.get(name, 0) + count

    def scoped(self, name: str):
        return _ProfileScope(self, name)

    def merge(self, other: "ProfileStats") -> None:
        for name, elapsed in other.times.items():
            self.times[name] = self.times.get(name, 0.0) + elapsed
        for name, count in other.counts.items():
            self.counts[name] = self.counts.get(name, 0) + count

    def print_summary(self, total_time: Optional[float] = None, limit: int = 40) -> None:
        denom = total_time if total_time and total_time > 0 else sum(self.times.values())
        print("\nProfile summary")
        print(f"{'section':<34} {'time(s)':>10} {'%total':>8} {'count':>10} {'avg(ms)':>10}")
        print("-" * 78)
        for name, elapsed in sorted(self.times.items(), key=lambda kv: kv[1], reverse=True)[:limit]:
            count = self.counts.get(name, 0)
            pct = 100.0 * elapsed / denom if denom > 0 else 0.0
            avg_ms = 1000.0 * elapsed / count if count > 0 else 0.0
            print(f"{name:<34} {elapsed:10.3f} {pct:8.1f} {count:10d} {avg_ms:10.3f}")


class _ProfileScope:
    def __init__(self, stats: ProfileStats, name: str):
        self.stats = stats
        self.name = name
        self.t0 = 0.0

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stats.add(self.name, time.perf_counter() - self.t0)
        return False


@dataclass(frozen=True)
class LabyrinthSpec:
    benchmark: str
    bid: str
    data_file: Path
    trigger_file: Path


def resolve_benchmark(name: str) -> str:
    key = str(name).strip().lower().replace("_", "").replace("-", "")
    return BENCHMARK_ALIASES.get(key, str(name))


def benchmark_spec(name: str) -> LabyrinthSpec:
    bid = resolve_benchmark(name)
    data_file = REPO_ROOT / "labyrinth_benchmarks" / f"labyrinth_{bid}.data"
    trigger_file = REPO_ROOT / "labyrinth_benchmarks" / "trigger_config.json"
    if not data_file.exists():
        raise FileNotFoundError(
            f"Missing {data_file}. Download Mahdi's labyrinth_benchmarks assets first."
        )
    if not trigger_file.exists():
        raise FileNotFoundError(f"Missing {trigger_file}.")
    return LabyrinthSpec(name, bid, data_file, trigger_file)


def joint_action(a0: int, a1: int, act_per_agent: int) -> int:
    return a0 + act_per_agent * a1


def split_action(a: int, act_per_agent: int) -> Tuple[int, int]:
    return a % act_per_agent, a // act_per_agent


def joint_obs(o0: int, o1: int, obs_per_agent: int) -> int:
    return o0 + obs_per_agent * o1


def split_obs(o: int, obs_per_agent: int) -> Tuple[int, int]:
    return o % obs_per_agent, o // obs_per_agent


class LabyrinthModel:
    def __init__(self, benchmark: str, mode: str = "semi"):
        self.spec = benchmark_spec(benchmark)
        self.benchmark = benchmark
        self.bid = self.spec.bid
        self.mode = mode
        self.start_node = START_NODE

        self.num_nodes = 0
        self.num_targets = 0
        self.num_file_states = 0
        self.act_per_agent = 0
        self.obs_per_agent = 0
        self.n_actions = 0
        self.n_obs = 0
        self.n_states = 0
        self.sink_state = 0
        self.targets: List[int] = []
        self.state_triggers: List[int] = []
        self.edges: List[Tuple[int, int]] = []

        self.transitions: List[List[Tuple[int, float]]] = []
        self.observations: List[List[Tuple[int, float]]] = []
        self.rewards: List[float] = []
        self.init_belief: List[float] = []
        self.valid_actions_per_position: List[Dict[int, List[int]]] = [{}, {}]
        self._qmdp_cache: Dict[int, Tuple[float, ...]] = {}

        self._load()

    def _load_metadata(self, header_lines: Sequence[str]) -> None:
        text = "\n".join(header_lines)
        m = re.search(r"Nodes:\s*(\d+),\s*Targets:\s*(\d+),\s*States:\s*(\d+)", text)
        if not m:
            raise ValueError(f"Could not parse metadata from {self.spec.data_file}")
        self.num_nodes = int(m.group(1))
        self.num_targets = int(m.group(2))
        self.num_file_states = int(m.group(3))

        m = re.search(r"Actions per agent:\s*(\d+)", text)
        if not m:
            raise ValueError("Could not parse action count.")
        self.act_per_agent = int(m.group(1))

        m = re.search(r"Observations per agent:\s*(\d+)", text)
        if not m:
            raise ValueError("Could not parse observation count.")
        self.obs_per_agent = int(m.group(1))

        self.n_actions = self.act_per_agent ** N_AGENTS
        self.n_obs = self.obs_per_agent ** N_AGENTS
        self.n_states = self.num_file_states + 1
        self.sink_state = self.n_states - 1
        self.targets = [i for i in range(self.num_nodes) if i != self.start_node]

    def state_to_tuple(self, s: int) -> Tuple[int, int, int, int, int]:
        if s == self.sink_state:
            return -1, -1, -1, -1, -1
        found2 = s % 2
        temp = s // 2
        found1 = temp % 2
        temp //= 2
        target_idx = temp % self.num_targets
        temp //= self.num_targets
        u2 = temp % self.num_nodes
        u1 = temp // self.num_nodes
        return u1, u2, target_idx, found1, found2

    def tuple_to_state(self, u1: int, u2: int, target_idx: int, found1: int, found2: int) -> int:
        return (
            u1 * (self.num_nodes * self.num_targets * 4)
            + u2 * (self.num_targets * 4)
            + target_idx * 4
            + found1 * 2
            + found2
        )

    def _load_triggers(self) -> List[int]:
        if self.mode == "decentralized":
            return []
        if self.mode == "centralized":
            return list(range(self.n_states - 1))

        with self.spec.trigger_file.open("r") as f:
            data = json.load(f)
        item = data.get(str(self.bid), [])
        if isinstance(item, list):
            return [int(x) for x in item]
        if isinstance(item, dict):
            return [int(x) for x in item.get("states", [])]
        return []

    def _is_goal_state(self, s: int) -> bool:
        if s == self.sink_state:
            return False
        u1, u2, _target_idx, found1, found2 = self.state_to_tuple(s)
        return (u1 == self.start_node and found1 == 1) or (
            u2 == self.start_node and found2 == 1
        )

    def _is_sync_position(self, s: int, trigger_set: set[int]) -> bool:
        if s == self.sink_state or not trigger_set:
            return False
        u1, u2, target_idx, _found1, _found2 = self.state_to_tuple(s)
        for f1 in range(2):
            for f2 in range(2):
                if self.tuple_to_state(u1, u2, target_idx, f1, f2) in trigger_set:
                    return True
        return False

    def _redirect_goal_transitions(
        self,
        flat_t: Dict[Tuple[int, int, int], float],
        rewards: List[float],
    ) -> Dict[Tuple[int, int, int], float]:
        out: Dict[Tuple[int, int, int], float] = {}
        for (a, s, sp), p in flat_t.items():
            if p <= 0.0:
                continue
            if sp != self.sink_state and self._is_goal_state(sp):
                rewards[a * self.n_states + s] = SUCCESS_REWARD
                out[(a, s, self.sink_state)] = out.get((a, s, self.sink_state), 0.0) + p
            else:
                out[(a, s, sp)] = out.get((a, s, sp), 0.0) + p
        return out

    def _apply_sync_knowledge_propagation(
        self,
        flat_t: Dict[Tuple[int, int, int], float],
        rewards: List[float],
    ) -> Dict[Tuple[int, int, int], float]:
        trigger_set = set(self.state_triggers)
        if not trigger_set:
            return flat_t

        out: Dict[Tuple[int, int, int], float] = {}
        goal_prob: Dict[Tuple[int, int], float] = {}

        for (a, s, sp), p in flat_t.items():
            new_sp = sp
            is_new_goal = False
            if sp != self.sink_state and self._is_sync_position(sp, trigger_set):
                u1, u2, target_idx, found1, found2 = self.state_to_tuple(sp)
                if found1 == 1 or found2 == 1:
                    propagated = self.tuple_to_state(u1, u2, target_idx, 1, 1)
                    if propagated != sp:
                        if self._is_goal_state(propagated) and not self._is_goal_state(sp):
                            new_sp = self.sink_state
                            is_new_goal = True
                        else:
                            new_sp = propagated

            out[(a, s, new_sp)] = out.get((a, s, new_sp), 0.0) + p
            if is_new_goal:
                goal_prob[(a, s)] = goal_prob.get((a, s), 0.0) + p

        for (a, s), p_goal in goal_prob.items():
            idx = a * self.n_states + s
            rewards[idx] = p_goal * SUCCESS_REWARD + (1.0 - p_goal) * rewards[idx]

        return out

    def _build_tables(
        self,
        flat_t: Dict[Tuple[int, int, int], float],
        flat_o: Dict[Tuple[int, int, int], float],
    ) -> None:
        size = self.n_actions * self.n_states
        self.transitions = [[] for _ in range(size)]
        self.observations = [[] for _ in range(size)]

        for (a, s, sp), p in flat_t.items():
            self.transitions[a * self.n_states + s].append((sp, p))
        for (a, sp, o), p in flat_o.items():
            self.observations[a * self.n_states + sp].append((o, p))

        for a in range(self.n_actions):
            self.transitions[a * self.n_states + self.sink_state] = [(self.sink_state, 1.0)]
            self.observations[a * self.n_states + self.sink_state] = [(0, 1.0)]

    def _build_action_masks(self) -> None:
        valid_single = [{node: {0} for node in range(self.num_nodes)} for _ in range(N_AGENTS)]
        edges = set()

        for node in range(self.num_nodes):
            ref_a0 = self.tuple_to_state(node, 0, 0, 0, 0)
            for a0 in range(1, self.act_per_agent):
                ja = joint_action(a0, 0, self.act_per_agent)
                for sp, p in self.transitions[ja * self.n_states + ref_a0]:
                    if p > 0.0:
                        nu1, _nu2, _t, _f1, _f2 = self.state_to_tuple(sp)
                        if nu1 != node and nu1 != -1:
                            valid_single[0][node].add(a0)
                            edges.add(tuple(sorted((node, nu1))))
                        break

            ref_a1 = self.tuple_to_state(0, node, 0, 0, 0)
            for a1 in range(1, self.act_per_agent):
                ja = joint_action(0, a1, self.act_per_agent)
                for sp, p in self.transitions[ja * self.n_states + ref_a1]:
                    if p > 0.0:
                        _nu1, nu2, _t, _f1, _f2 = self.state_to_tuple(sp)
                        if nu2 != node and nu2 != -1:
                            valid_single[1][node].add(a1)
                            edges.add(tuple(sorted((node, nu2))))
                        break

        self.valid_actions_per_position = [
            {node: sorted(actions) for node, actions in valid_single[rid].items()}
            for rid in range(N_AGENTS)
        ]
        self.edges = sorted(edges)

    def _load(self) -> None:
        with self.spec.data_file.open("r") as f:
            header = [next(f).strip() for _ in range(8)]
        self._load_metadata(header)
        self.state_triggers = self._load_triggers()

        flat_t: Dict[Tuple[int, int, int], float] = {}
        flat_o: Dict[Tuple[int, int, int], float] = {}
        self.rewards = [STEP_REWARD] * (self.n_actions * self.n_states)

        with self.spec.data_file.open("r") as f:
            for line in f:
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if not parts:
                    continue
                if parts[0] == "T":
                    a0, a1 = int(parts[1]), int(parts[2])
                    s, sp = int(parts[3]), int(parts[4])
                    p = float(parts[5])
                    a = joint_action(a0, a1, self.act_per_agent)
                    flat_t[(a, s, sp)] = flat_t.get((a, s, sp), 0.0) + p
                elif parts[0] == "O":
                    a0, a1 = int(parts[1]), int(parts[2])
                    sp = int(parts[3])
                    o0, o1 = int(parts[4]), int(parts[5])
                    p = float(parts[6])
                    a = joint_action(a0, a1, self.act_per_agent)
                    o = joint_obs(o0, o1, self.obs_per_agent)
                    key = (a, sp, o)
                    old_p = flat_o.get(key)
                    if old_p is not None and abs(old_p - p) > 1e-12:
                        raise ValueError(
                            f"Inconsistent duplicate observation entry {key}: {old_p} vs {p}"
                        )
                    flat_o[key] = p

        flat_t = self._redirect_goal_transitions(flat_t, self.rewards)
        flat_t = self._apply_sync_knowledge_propagation(flat_t, self.rewards)

        for a in range(self.n_actions):
            self.rewards[a * self.n_states + self.sink_state] = 0.0

        self.init_belief = [0.0] * self.n_states
        p_target = 1.0 / self.num_targets
        for target_idx in range(self.num_targets):
            s = self.tuple_to_state(self.start_node, self.start_node, target_idx, 0, 0)
            self.init_belief[s] = p_target

        self._build_tables(flat_t, flat_o)
        self._build_action_masks()

        global ACTION_NAME, OBS_NAME
        ACTION_NAME = {0: "WAIT"}
        ACTION_NAME.update({a: f"MOVE{a}" for a in range(1, self.act_per_agent)})
        OBS_NAME = {
            o: f"N{o // 2}:F{o % 2}"
            for o in range(self.obs_per_agent)
        }

    def reward(self, s: int, a: int) -> float:
        return self.rewards[a * self.n_states + s]

    def transition_dist(self, s: int, a: int) -> List[Tuple[int, float]]:
        return self.transitions[a * self.n_states + s]

    def obs_dist(self, sp: int, a: int) -> List[Tuple[int, float]]:
        return self.observations[a * self.n_states + sp]

    def obs_prob(self, sp: int, a: int, o: int) -> float:
        for obs, p in self.obs_dist(sp, a):
            if obs == o:
                return p
        return 0.0

    def local_obs_prob(self, rid: int, sp: int, a: int, local_o: int) -> float:
        total = 0.0
        for obs, p in self.obs_dist(sp, a):
            o0, o1 = split_obs(obs, self.obs_per_agent)
            if (rid == 0 and o0 == local_o) or (rid == 1 and o1 == local_o):
                total += p
        return total

    def sample_initial_state(self, rng: random.Random, fixed_target_idx: Optional[int] = None) -> int:
        if fixed_target_idx is not None:
            return self.tuple_to_state(self.start_node, self.start_node, fixed_target_idx, 0, 0)
        return sample_from_dist(enumerate(self.init_belief), rng, self.sink_state)

    def sample_next_state(self, s: int, a: int, rng: random.Random) -> int:
        return sample_from_dist(self.transition_dist(s, a), rng, s)

    def sample_joint_obs(self, sp: int, a: int, rng: random.Random) -> int:
        return sample_from_dist(self.obs_dist(sp, a), rng, 0)

    def qmdp_values(self, horizon: int) -> Tuple[float, ...]:
        if horizon in self._qmdp_cache:
            return self._qmdp_cache[horizon]

        values = [0.0] * self.n_states
        for h in range(1, horizon + 1):
            next_values: List[float] = []
            for s in range(self.n_states):
                best = -math.inf
                for a in range(self.n_actions):
                    val = self.reward(s, a)
                    val += sum(p * values[sp] for sp, p in self.transition_dist(s, a))
                    if val > best:
                        best = val
                next_values.append(best)
            values = next_values
            self._qmdp_cache[h] = tuple(values)
        return self._qmdp_cache[horizon]


class LabyrinthObsModelAdapter:
    def __init__(self, model: LabyrinthModel, profiler: Optional[ProfileStats] = None):
        self.model = model
        self.profiler = profiler
        # id(belief) -> (belief ref, cumulative sums); the ref keeps the list
        # alive so ids are never recycled while cached.
        self._belief_cums: Dict[int, Tuple[Sequence[float], List[float]]] = {}

    def sample_state_from_belief(self, belief: Sequence[float], rng: random.Random) -> int:
        t0 = time.perf_counter()
        cached = self._belief_cums.get(id(belief))
        if cached is not None and cached[0] is belief:
            cum = cached[1]
        else:
            cum = list(accumulate(float(p) for p in belief))
            if len(self._belief_cums) > 256:
                self._belief_cums.clear()
            self._belief_cums[id(belief)] = (belief, cum)

        # Bit-exact match of the old linear scan: same rng draw, same running
        # sums, first index with r <= cum, last index if the mass falls short.
        r = rng.random()
        if not cum:
            out = self.model.sink_state
        else:
            idx = bisect.bisect_left(cum, r)
            out = idx if idx < len(cum) else len(cum) - 1
        if self.profiler is not None:
            self.profiler.add("model.sample_state_from_belief", time.perf_counter() - t0)
        return out

    def step(self, state: int, joint_a: int, rng: random.Random) -> StepResult:
        t0 = time.perf_counter()
        reward = self.model.reward(state, joint_a)
        next_state = self.model.sample_next_state(state, joint_a, rng)
        obs = self.model.sample_joint_obs(next_state, joint_a, rng)
        if self.profiler is not None:
            self.profiler.add("model.step", time.perf_counter() - t0)
        return StepResult(next_state=next_state, joint_obs=obs, reward=reward)

    def split_obs(self, obs: int) -> Tuple[int, int]:
        return split_obs(obs, self.model.obs_per_agent)

    def joint_action_from_dict(self, actions: Dict[int, int]) -> int:
        return joint_action(actions[0], actions[1], self.model.act_per_agent)

    def update_belief(
        self,
        belief: Sequence[float],
        joint_a: int,
        local_o: int,
        robot_id: int,
    ) -> List[float]:
        t0 = time.perf_counter()
        out = update_local_belief(belief, joint_a, local_o, robot_id, self.model)
        if self.profiler is not None:
            self.profiler.add("model.update_belief", time.perf_counter() - t0)
        return out


def sample_from_dist(items: Any, rng: random.Random, fallback: int) -> int:
    r = rng.random()
    cum = 0.0
    last = fallback
    for item, p in items:
        last = int(item)
        cum += float(p)
        if r <= cum:
            return int(item)
    return last


def normalize_belief(belief: Sequence[float], fallback_support: Optional[Sequence[int]] = None) -> List[float]:
    total = float(sum(belief))
    if total > 1e-15:
        return [float(x) / total for x in belief]
    out = [0.0] * len(belief)
    support = list(fallback_support or range(len(belief)))
    p = 1.0 / max(1, len(support))
    for s in support:
        out[s] = p
    return out


def predict_belief_open_loop(
    belief: Sequence[float],
    action: int,
    model: LabyrinthModel,
) -> List[float]:
    accum: Dict[int, float] = {}
    for s, b_s in belief_support(belief):
        for sp, p in model.transition_dist(s, action):
            accum[sp] = accum.get(sp, 0.0) + b_s * p

    total = sum(accum.values())
    if total <= 1e-15:
        return normalize_belief([0.0] * model.n_states)

    out = [0.0] * model.n_states
    inv_total = 1.0 / total
    for sp, p in accum.items():
        out[sp] = p * inv_total
    return out


def update_local_belief(
    belief: Sequence[float],
    action: int,
    local_obs: int,
    rid: int,
    model: LabyrinthModel,
) -> List[float]:
    pred_accum: Dict[int, float] = {}
    for s, b_s in belief_support(belief):
        for sp, p in model.transition_dist(s, action):
            pred_accum[sp] = pred_accum.get(sp, 0.0) + b_s * p

    post_accum: Dict[int, float] = {}
    for sp, pred_p in pred_accum.items():
        obs_p = model.local_obs_prob(rid, sp, action, local_obs)
        if obs_p > 0.0:
            post_accum[sp] = pred_p * obs_p

    out = [0.0] * model.n_states
    total = sum(post_accum.values())
    if total > 1e-15:
        inv_total = 1.0 / total
        for sp, p in post_accum.items():
            out[sp] = p * inv_total
        return out

    support = list(pred_accum.keys())
    p = 1.0 / max(1, len(support))
    for sp in support:
        out[sp] = p
    return out


def update_joint_belief(
    belief: Sequence[float],
    action: int,
    obs: int,
    model: LabyrinthModel,
    support: Optional[List[Tuple[int, float]]] = None,
) -> List[float]:
    if support is None:
        support = belief_support(belief)
    pred_accum: Dict[int, float] = {}
    for s, b_s in support:
        for sp, p in model.transition_dist(s, action):
            pred_accum[sp] = pred_accum.get(sp, 0.0) + b_s * p

    post_accum: Dict[int, float] = {}
    for sp, pred_p in pred_accum.items():
        obs_p = model.obs_prob(sp, action, obs)
        if obs_p > 0.0:
            post_accum[sp] = pred_p * obs_p

    out = [0.0] * model.n_states
    total = sum(post_accum.values())
    if total > 1e-15:
        inv_total = 1.0 / total
        for sp, p in post_accum.items():
            out[sp] = p * inv_total
        return out

    support = list(pred_accum.keys())
    p = 1.0 / max(1, len(support))
    for sp in support:
        out[sp] = p
    return out


def average_belief(beliefs: Dict[int, Sequence[float]], n_states: int) -> List[float]:
    out = [0.0] * n_states
    for belief in beliefs.values():
        for s, p in enumerate(belief):
            out[s] += p / max(1, len(beliefs))
    return normalize_belief(out)


def belief_support(belief: Sequence[float]) -> List[Tuple[int, float]]:
    return [(s, float(p)) for s, p in enumerate(belief) if p > 1e-12]


def sparse_rounded_belief_key(
    belief: Sequence[float],
    ndigits: int = 12,
    threshold: float = 1e-12,
) -> Tuple[Tuple[int, float], ...]:
    return tuple(
        (idx, round(float(prob), ndigits))
        for idx, prob in enumerate(belief)
        if prob > threshold
    )


def best_qmdp_joint_action(
    belief: Sequence[float],
    remaining_horizon: int,
    model: LabyrinthModel,
) -> int:
    future = (
        model.qmdp_values(remaining_horizon - 1)
        if remaining_horizon > 1
        else (0.0,) * model.n_states
    )
    best_a = 0
    best_v = -math.inf
    support = belief_support(belief)
    for a in range(model.n_actions):
        total = 0.0
        for s, b_s in support:
            q_s = model.reward(s, a)
            q_s += sum(p * future[sp] for sp, p in model.transition_dist(s, a))
            total += b_s * q_s
        if total > best_v:
            best_v = total
            best_a = a
    return best_a


def ranked_qmdp_local_actions(
    belief: Sequence[float],
    remaining_horizon: int,
    model: LabyrinthModel,
    rid: int,
    limit: int,
    symmetry_break: bool = True,
) -> List[int]:
    future = (
        model.qmdp_values(remaining_horizon - 1)
        if remaining_horizon > 1
        else (0.0,) * model.n_states
    )
    scores = {a: -math.inf for a in range(model.act_per_agent)}
    support = belief_support(belief)
    for joint_a in range(model.n_actions):
        total = 0.0
        for s, b_s in support:
            q_s = model.reward(s, joint_a)
            q_s += sum(p * future[sp] for sp, p in model.transition_dist(s, joint_a))
            total += b_s * q_s
        a0, a1 = split_action(joint_a, model.act_per_agent)
        local_a = a0 if rid == 0 else a1
        scores[local_a] = max(scores[local_a], total)
    if symmetry_break:
        # Deterministically break score ties differently by agent. This is a
        # generic role prior for symmetric decentralized search; it does not
        # remove actions or encode scenario routes.
        if rid % 2 == 0:
            sort_key = lambda kv: (-kv[1], kv[0])
        else:
            sort_key = lambda kv: (-kv[1], -kv[0])
    else:
        sort_key = lambda kv: (-kv[1], kv[0])
    actions = [a for a, _ in sorted(scores.items(), key=sort_key)]
    return actions[:limit] if limit > 0 else actions


def likely_positions_from_belief(belief: Sequence[float], rid: int, model: LabyrinthModel) -> List[int]:
    positions = set()
    for s, p in enumerate(belief):
        if p <= 1e-9 or s == model.sink_state:
            continue
        u1, u2, _target_idx, _f1, _f2 = model.state_to_tuple(s)
        positions.add(u1 if rid == 0 else u2)
    return sorted(positions) or [model.start_node]


def make_belief_legal_actions_fn(model: LabyrinthModel, rid: int, args):
    def legal_actions(belief: Sequence[float], depth: int) -> Sequence[int]:
        rem = max(1, args.current_remaining_horizon - depth)
        if args.qmdp_action_limit > 0:
            return ranked_qmdp_local_actions(belief, rem, model, rid, args.qmdp_action_limit)

        if args.position_action_mask:
            allowed = set()
            for pos in likely_positions_from_belief(belief, rid, model):
                allowed.update(model.valid_actions_per_position[rid].get(pos, [0]))
            return sorted(allowed)

        return list(range(model.act_per_agent))

    return legal_actions


def make_belief_action_rank_fn(model: LabyrinthModel, rid: int, args):
    def rank_actions(belief: Sequence[float], depth: int) -> Sequence[int]:
        rem = max(1, args.current_remaining_horizon - depth)
        return ranked_qmdp_local_actions(
            belief=belief,
            remaining_horizon=rem,
            model=model,
            rid=rid,
            limit=model.act_per_agent,
            symmetry_break=args.symmetry_break_rank,
        )

    return rank_actions


def make_obs_legal_actions_fn(model: LabyrinthModel):
    def legal_actions(_history, _depth: int) -> Sequence[int]:
        return list(range(model.act_per_agent))

    return legal_actions


def qmdp_local_action(
    belief: Sequence[float],
    rid: int,
    model: LabyrinthModel,
    remaining_horizon: int,
) -> int:
    a = best_qmdp_joint_action(belief, remaining_horizon, model)
    a0, a1 = split_action(a, model.act_per_agent)
    return a0 if rid == 0 else a1


def make_belief_default_action_fn(
    rid: int,
    model: LabyrinthModel,
    remaining_horizon: int,
    fallback: int,
    mode: str = "qmdp-joint",
):
    cache: Dict[Tuple[Tuple[int, float], ...], int] = {}
    obj_cache: Dict[int, Tuple[Sequence[float], int]] = {}

    def default_action(belief: Sequence[float]) -> int:
        if remaining_horizon <= 0:
            return fallback

        obj_id = id(belief)
        cached_obj = obj_cache.get(obj_id)
        if cached_obj is not None and cached_obj[0] is belief:
            return cached_obj[1]

        key = sparse_rounded_belief_key(belief)
        if key not in cache:
            if mode == "qmdp-local-rank":
                ranked = ranked_qmdp_local_actions(
                    belief=belief,
                    remaining_horizon=remaining_horizon,
                    model=model,
                    rid=rid,
                    limit=1,
                    symmetry_break=True,
                )
                cache[key] = ranked[0] if ranked else fallback
            else:
                cache[key] = qmdp_local_action(belief, rid, model, remaining_horizon)
        action = cache[key]
        obj_cache[obj_id] = (belief, action)
        return action

    return default_action


def labyrinth_belief_from_local_history(
    root_belief: Sequence[float],
    history,
    rid: int,
    model: LabyrinthModel,
    teammate_default: int,
) -> List[float]:
    belief = list(root_belief)

    for own_action, local_obs in history:
        if rid == 0:
            action = joint_action(own_action, teammate_default, model.act_per_agent)
        else:
            action = joint_action(teammate_default, own_action, model.act_per_agent)
        belief = update_local_belief(belief, action, local_obs, rid, model)

    return belief


def make_obs_default_action_fn(
    root_belief: Sequence[float],
    rid: int,
    model: LabyrinthModel,
    remaining_horizon: int,
    fallback: int,
):
    # TESTING: previous obs default used the QMDP root action only at history
    # () and then returned fallback for every deeper observation history.
    #
    # root_default = qmdp_local_action(root_belief, rid, model, remaining_horizon)
    action_cache: Dict[Any, int] = {}

    def default_action(history) -> int:
        # TESTING:
        # return root_default if history == () else fallback
        if history in action_cache:
            return action_cache[history]

        if len(history) > 1:
            # TESTING: full history-conditioned QMDP defaults at every rollout
            # depth are expensive because sampled observation histories rarely
            # repeat. Preserve the first observation-conditioned branch and use
            # the configured cheap fallback deeper in rollouts.
            action_cache[history] = fallback
            return fallback

        local_belief = labyrinth_belief_from_local_history(
            root_belief=root_belief,
            history=history,
            rid=rid,
            model=model,
            teammate_default=fallback,
        )
        rem = max(1, remaining_horizon - len(history))
        action = qmdp_local_action(local_belief, rid, model, rem)
        action_cache[history] = action
        return action

    return default_action


class HeuristicExpansionBeliefObsDecMCTS(BeliefObsDecMCTS):
    def __init__(self, *args, action_rank_fn=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_rank_fn = action_rank_fn

    def _select_or_expand_action(self, node):
        if node.untried_actions:
            ranked_actions = (
                list(self.action_rank_fn(node.belief, node.depth))
                if self.action_rank_fn is not None
                else []
            )
            action = None
            for candidate in ranked_actions:
                if candidate in node.untried_actions:
                    action = candidate
                    break
            if action is None:
                default_action = self.default_action_fn(node.belief)
                action = (
                    default_action
                    if default_action in node.untried_actions
                    else self.rng.choice(node.untried_actions)
                )
            node.add_action_edge(action)
            return action
        return max(node.actions.values(), key=lambda edge: self._ucb(edge, node)).action


class ProfiledHeuristicExpansionBeliefObsDecMCTS(HeuristicExpansionBeliefObsDecMCTS):
    def __init__(self, *args, profiler: Optional[ProfileStats] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.profiler = profiler

    def iterate(self, n_outer: int = 1) -> None:
        if self.profiler is None:
            return super().iterate(n_outer)

        for _ in range(n_outer):
            with self.profiler.scoped("planner.update_sample_space"):
                self._update_sample_space()
            with self.profiler.scoped("planner.grow_tree_batch"):
                for _ in range(self.tau):
                    self._grow_tree_once()
            if self.X_hat:
                with self.profiler.scoped("planner.update_distribution"):
                    self._update_distribution()
            self.beta *= self.beta_decay

    def _grow_tree_once(self) -> None:
        if self.profiler is None:
            return super()._grow_tree_once()
        with self.profiler.scoped("planner.grow_tree_once"):
            return super()._grow_tree_once()

    def _update_sample_space(self) -> None:
        if self.profiler is None:
            return super()._update_sample_space()
        with self.profiler.scoped("planner.update_sample_space.inner"):
            return super()._update_sample_space()

    def _score_policy_key(self, key, n_eval: int = 5) -> float:
        if self.profiler is None:
            return super()._score_policy_key(key, n_eval)
        with self.profiler.scoped("planner.score_policy_key"):
            return super()._score_policy_key(key, n_eval)

    def _estimate_expectation(self, fixed_policy_key) -> float:
        if self.profiler is None:
            return super()._estimate_expectation(fixed_policy_key)
        with self.profiler.scoped("planner.estimate_expectation"):
            return super()._estimate_expectation(fixed_policy_key)

    def _simulate_from_node(self, *args, **kwargs) -> float:
        if self.profiler is None:
            return super()._simulate_from_node(*args, **kwargs)
        with self.profiler.scoped("planner.simulate_from_node"):
            return super()._simulate_from_node(*args, **kwargs)

    def _rollout(self, *args, **kwargs) -> float:
        if self.profiler is None:
            return super()._rollout(*args, **kwargs)
        with self.profiler.scoped("planner.rollout"):
            return super()._rollout(*args, **kwargs)


def belief_root_action_masses(planner: BeliefObsDecMCTS, model: LabyrinthModel) -> Dict[str, float]:
    masses = {a: 0.0 for a in range(model.act_per_agent)}
    root_key = planner.root.belief_key
    for policy_key, p in planner.q.items():
        for belief_key, action in policy_key:
            if belief_key == root_key:
                masses[int(action)] += p
                break
    return {ACTION_NAME[a]: round(masses[a], 4) for a in range(model.act_per_agent)}


def obs_root_action_masses(planner: ObsDecMCTS, model: LabyrinthModel) -> Dict[str, float]:
    masses = {a: 0.0 for a in range(model.act_per_agent)}
    for key, p in planner.q.items():
        for hist, action in key:
            if hist == ():
                masses[int(action)] += p
                break
    return {ACTION_NAME[a]: round(masses[a], 4) for a in range(model.act_per_agent)}


def run_belief_obs_planning_step(
    beliefs: Dict[int, List[float]],
    common_belief: Sequence[float],
    remaining_horizon: int,
    model: LabyrinthModel,
    args,
    seed: int,
    profiler: Optional[ProfileStats] = None,
):
    del common_belief
    planners: Dict[int, BeliefObsDecMCTS] = {}
    adapter = LabyrinthObsModelAdapter(model, profiler)
    args.current_remaining_horizon = remaining_horizon

    with profiler.scoped("planning.build_defaults") if profiler else _nullcontext():
        defaults = {
            rid: make_belief_default_action_fn(
                rid,
                model,
                remaining_horizon,
                fallback=args.default_action,
                mode=args.default_policy,
            )
            for rid in range(N_AGENTS)
        }

    with profiler.scoped("planning.construct_planners") if profiler else _nullcontext():
        for rid in range(N_AGENTS):
            if args.heuristic_expansion:
                cls = (
                    ProfiledHeuristicExpansionBeliefObsDecMCTS
                    if profiler is not None
                    else HeuristicExpansionBeliefObsDecMCTS
                )
            else:
                cls = BeliefObsDecMCTS
            planner_kwargs = {}
            if issubclass(cls, HeuristicExpansionBeliefObsDecMCTS):
                planner_kwargs["action_rank_fn"] = (
                    make_belief_action_rank_fn(model, rid, args)
                    if args.qmdp_expansion_order
                    else None
                )
            if cls is ProfiledHeuristicExpansionBeliefObsDecMCTS:
                planner_kwargs["profiler"] = profiler

            planners[rid] = cls(
                robot_id=rid,
                robot_ids=[0, 1],
                root_belief=beliefs[rid],
                root_beliefs_by_robot=beliefs,
                model=adapter,
                legal_actions_fn=make_belief_legal_actions_fn(model, rid, args),
                default_action_fn=defaults[rid],
                default_action_fns_by_robot=defaults,
                share_belief_nodes=args.belief_share_nodes,
                belief_key_fn=(
                    sparse_rounded_belief_key
                    if args.belief_key == "sparse"
                    else belief_core.rounded_belief_key
                ),
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
                **planner_kwargs,
            )

    team = BeliefObsDecMCTSTeam(planners)
    with profiler.scoped("planning.iterate_and_communicate") if profiler else _nullcontext():
        team.iterate_and_communicate(args.outer_iters, args.comm_period)
    with profiler.scoped("planning.best_actions") if profiler else _nullcontext():
        raw_actions = team.best_actions(beliefs=beliefs, source=args.action_source)
    actions = {rid: int(raw_actions[rid]) for rid in range(N_AGENTS)}

    if args.guard_actions:
        with profiler.scoped("planning.guard_action") if profiler else _nullcontext():
            guarded = best_qmdp_joint_action(
                average_belief(beliefs, model.n_states),
                remaining_horizon,
                model,
            )
        a0, a1 = split_action(guarded, model.act_per_agent)
        actions = {0: a0, 1: a1}

    with profiler.scoped("planning.summaries") if profiler else _nullcontext():
        policies = team.best_policies()
        entropies = team.entropies()
        masses = {
            rid: belief_root_action_masses(planners[rid], model) for rid in range(N_AGENTS)
        }
    return actions, raw_actions, policies, entropies, masses


def run_obs_planning_step(
    beliefs: Dict[int, List[float]],
    remaining_horizon: int,
    model: LabyrinthModel,
    args,
    seed: int,
    profiler: Optional[ProfileStats] = None,
):
    planners: Dict[int, ObsDecMCTS] = {}
    adapter = LabyrinthObsModelAdapter(model, profiler)
    defaults = {
        rid: make_obs_default_action_fn(
            beliefs[rid], rid, model, remaining_horizon, fallback=args.default_action
        )
        for rid in range(N_AGENTS)
    }

    for rid in range(N_AGENTS):
        planners[rid] = ObsDecMCTS(
            robot_id=rid,
            robot_ids=[0, 1],
            root_belief=beliefs[rid],
            model=adapter,
            legal_actions_fn=make_obs_legal_actions_fn(model),
            default_action_fn=defaults[rid],
            default_action_fns_by_robot=defaults,
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
    with profiler.scoped("planning.iterate_and_communicate") if profiler else _nullcontext():
        team.iterate_and_communicate(args.outer_iters, args.comm_period)
    with profiler.scoped("planning.best_actions") if profiler else _nullcontext():
        raw_actions = team.best_actions(source=args.action_source)
    actions = {rid: int(raw_actions[rid]) for rid in range(N_AGENTS)}

    if args.guard_actions:
        with profiler.scoped("planning.guard_action") if profiler else _nullcontext():
            guarded = best_qmdp_joint_action(
                average_belief(beliefs, model.n_states),
                remaining_horizon,
                model,
            )
        a0, a1 = split_action(guarded, model.act_per_agent)
        actions = {0: a0, 1: a1}

    with profiler.scoped("planning.summaries") if profiler else _nullcontext():
        policies = team.best_policies()
        entropies = team.entropies()
        masses = {
            rid: obs_root_action_masses(planners[rid], model) for rid in range(N_AGENTS)
        }
    return actions, raw_actions, policies, entropies, masses


def should_env_communicate(model: LabyrinthModel, args, next_state: int) -> bool:
    if args.env_comm_mode == "none":
        return False
    if args.env_comm_mode == "trigger":
        return next_state in set(model.state_triggers)
    if args.env_comm_mode == "centralized":
        return next_state != model.sink_state
    raise ValueError(f"Unknown env_comm_mode: {args.env_comm_mode}")


def state_summary(model: LabyrinthModel, s: int) -> str:
    if s == model.sink_state:
        return "SINK"
    u1, u2, target_idx, f1, f2 = model.state_to_tuple(s)
    target = model.targets[target_idx]
    return f"u=({u1},{u2}) target={target} found=({f1},{f2})"


def belief_support_summary(model: LabyrinthModel, belief: Sequence[float], limit: int = 5) -> str:
    top = sorted(((p, s) for s, p in enumerate(belief) if p > 1e-9), reverse=True)[:limit]
    parts = [f"{state_summary(model, s)}:{p:.3f}" for p, s in top]
    return f"support={sum(1 for p in belief if p > 1e-9)}, top={parts}"


def simulate_online_episode(
    model: LabyrinthModel,
    args,
    episode_seed: int,
    fixed_target_idx: Optional[int] = None,
    verbose: bool = False,
    profiler: Optional[ProfileStats] = None,
) -> float:
    rng = random.Random(episode_seed)
    with profiler.scoped("episode.sample_initial_state") if profiler else _nullcontext():
        true_state = model.sample_initial_state(rng, fixed_target_idx)

    common_belief = list(model.init_belief)
    beliefs = {0: list(model.init_belief), 1: list(model.init_belief)}
    pending_history: List[Tuple[int, int]] = []
    total_reward = 0.0
    discount = 1.0

    def flush_pending_common_belief() -> None:
        nonlocal common_belief, pending_history
        with profiler.scoped("episode.flush_common_belief") if profiler else _nullcontext():
            for hist_a, hist_o in pending_history:
                common_belief = update_joint_belief(common_belief, hist_a, hist_o, model)
            beliefs[0] = list(common_belief)
            beliefs[1] = list(common_belief)
            pending_history = []

    if verbose:
        print(f"initial {state_summary(model, true_state)}")

    for t in range(args.horizon):
        if true_state == model.sink_state:
            break
        remaining = args.horizon - t

        if args.planner == "belief-obs":
            with profiler.scoped("episode.plan_step") if profiler else _nullcontext():
                actions, raw_actions, _policies, entropies, root_masses = run_belief_obs_planning_step(
                    beliefs,
                    common_belief,
                    remaining,
                    model,
                    args,
                    episode_seed + 7919 * t,
                    profiler=profiler,
                )
        else:
            with profiler.scoped("episode.plan_step") if profiler else _nullcontext():
                actions, raw_actions, _policies, entropies, root_masses = run_obs_planning_step(
                    beliefs,
                    remaining,
                    model,
                    args,
                    episode_seed + 7919 * t,
                    profiler=profiler,
                )

        a0, a1 = actions[0], actions[1]
        a = joint_action(a0, a1, model.act_per_agent)
        with profiler.scoped("episode.env_step") if profiler else _nullcontext():
            reward = model.reward(true_state, a)
            total_reward += discount * reward
            discount *= args.gamma

            next_state = model.sample_next_state(true_state, a, rng)
            obs = model.sample_joint_obs(next_state, a, rng)
            o0, o1 = split_obs(obs, model.obs_per_agent)

        with profiler.scoped("episode.local_belief_updates") if profiler else _nullcontext():
            beliefs[0] = update_local_belief(beliefs[0], a, o0, 0, model)
            beliefs[1] = update_local_belief(beliefs[1], a, o1, 1, model)
            pending_history.append((a, obs))

        if should_env_communicate(model, args, next_state):
            flush_pending_common_belief()

        if verbose:
            print(
                f"t={t:02d} rem={remaining:02d} "
                f"a=({ACTION_NAME[a0]},{ACTION_NAME[a1]}) "
                f"raw=({ACTION_NAME[int(raw_actions[0])]},{ACTION_NAME[int(raw_actions[1])]}) "
                f"r={reward:7.3f} obs=({OBS_NAME[o0]},{OBS_NAME[o1]}) "
                f"next={state_summary(model, next_state)} "
                f"H=({entropies.get(0, 0.0):.3f},{entropies.get(1, 0.0):.3f})"
            )
            print(f"    belief0={belief_support_summary(model, beliefs[0])}")
            print(f"    belief1={belief_support_summary(model, beliefs[1])}")
            print(f"    root_mass={root_masses}")

        true_state = next_state

    return total_reward


def confidence_interval_95(values: Sequence[float]) -> Tuple[float, float]:
    mean = sum(values) / max(1, len(values))
    if len(values) <= 1:
        return mean, 0.0
    var = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return mean, 1.96 * math.sqrt(var / len(values))


def run_episode_worker(payload):
    ep, args_dict = payload
    args = argparse.Namespace(**args_dict)
    model = LabyrinthModel(args.benchmark, args.mode)
    ret = simulate_online_episode(
        model,
        args,
        episode_seed=args.seed + 104729 * ep,
        verbose=False,
    )
    return ep, ret


def run_initial_planning_timing(model: LabyrinthModel, args) -> None:
    beliefs = {0: list(model.init_belief), 1: list(model.init_belief)}
    common = list(model.init_belief)
    t0 = time.time()
    if args.planner == "belief-obs":
        actions, raw_actions, _policies, entropies, masses = run_belief_obs_planning_step(
            beliefs, common, args.horizon, model, args, args.seed
        )
    else:
        actions, raw_actions, _policies, entropies, masses = run_obs_planning_step(
            beliefs, args.horizon, model, args, args.seed
        )
    elapsed = time.time() - t0
    print("Initial Labyrinth planning timing")
    print(f"benchmark={args.benchmark} bid={model.bid} horizon={args.horizon}")
    print(f"planning_time={elapsed:.6f}s")
    print(f"actions=({ACTION_NAME[int(actions[0])]}, {ACTION_NAME[int(actions[1])]})")
    print(f"raw_actions=({ACTION_NAME[int(raw_actions[0])]}, {ACTION_NAME[int(raw_actions[1])]})")
    print(f"entropies={entropies}")
    print(f"root_mass={masses}")


def run_table(args) -> None:
    print("Labyrinth table sweep")
    names = ["extcross9", "lopsidedy10", "ladder10", "maze12", "hiddentail11", "mesh10"]
    for name in names:
        horizons = DEFAULT_TABLE_HORIZONS[name]
        print()
        print(f"{name} ({horizons})")
        for h in horizons:
            local_args = argparse.Namespace(**vars(args))
            local_args.benchmark = name
            local_args.horizon = h
            model = LabyrinthModel(name, local_args.mode)
            returns = []
            t0 = time.time()
            if local_args.fullsim:
                for target_idx, _target_node in enumerate(model.targets):
                    for trial in range(local_args.trials_per_target):
                        returns.append(
                            simulate_online_episode(
                                model,
                                local_args,
                                episode_seed=(
                                    local_args.seed
                                    + 104729 * target_idx
                                    + 7919 * trial
                                ),
                                fixed_target_idx=target_idx,
                                verbose=False,
                            )
                        )
            else:
                for ep in range(local_args.episodes):
                    returns.append(
                        simulate_online_episode(
                            model,
                            local_args,
                            episode_seed=local_args.seed + 104729 * ep,
                            verbose=False,
                        )
                    )
            mean, ci = confidence_interval_95(returns)
            print(f"  H={h:2d}: {mean:8.3f} ± {ci:.3f} time={time.time() - t0:.2f}s")


def run_fullsim(args) -> None:
    model = LabyrinthModel(args.benchmark, args.mode)
    profiler = ProfileStats() if args.profile else None
    obs_core.set_profiler(profiler)
    belief_core.set_profiler(profiler)
    total_t0 = time.perf_counter()
    try:
        print("Labyrinth stratified target evaluation")
        print(
            f"benchmark={args.benchmark} bid={model.bid}, horizon={args.horizon}, "
            f"targets={model.targets}, trials_per_target={args.trials_per_target}"
        )

        target_means: List[float] = []
        target_times: List[float] = []
        for target_idx, target_node in enumerate(model.targets):
            rewards = []
            t0 = time.time()
            for trial in range(args.trials_per_target):
                rewards.append(
                    simulate_online_episode(
                        model,
                        args,
                        episode_seed=args.seed + 104729 * target_idx + 7919 * trial,
                        fixed_target_idx=target_idx,
                        verbose=args.verbose_first and target_idx == 0 and trial == 0,
                        profiler=profiler,
                    )
                )
            elapsed = time.time() - t0
            mean, ci = confidence_interval_95(rewards)
            target_means.append(mean)
            target_times.append(elapsed)
            print(
                f"target={target_node:2d}: mean={mean:8.3f} ± {ci:.3f} "
                f"time={elapsed:.2f}s",
                flush=True,
            )

        mean, ci = confidence_interval_95(target_means)
        print()
        print(
            f"Stratified mean return: {mean:.5f} ± {ci:.5f} "
            f"(between-target 95% CI), total_time={sum(target_times):.2f}s"
        )
        if profiler is not None:
            profiler.print_summary(total_time=time.perf_counter() - total_t0)
    finally:
        obs_core.set_profiler(None)
        belief_core.set_profiler(None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="extcross9")
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--mode", choices=["decentralized", "semi", "centralized"], default="semi")
    parser.add_argument("--planner", choices=["obs", "belief-obs"], default="belief-obs")

    parser.add_argument("--outer-iters", type=int, default=30)
    parser.add_argument("--tau", type=int, default=40)
    parser.add_argument("--num-seq", type=int, default=30)
    parser.add_argument("--num-samples", type=int, default=30)
    parser.add_argument("--comm-period", type=int, default=1)

    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--cp", type=float, default=0.5)
    parser.add_argument("--beta-init", type=float, default=2.0)
    parser.add_argument("--beta-decay", type=float, default=0.995)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--default-action", type=int, default=0)
    parser.add_argument(
        "--default-policy",
        choices=["qmdp-joint", "qmdp-local-rank"],
        default="qmdp-joint",
        help="Loose rollout/default heuristic; neither option restricts legal actions.",
    )
    parser.add_argument(
        "--action-source",
        choices=["tree", "disc_tree", "policy", "visits", "policy_value", "policy_marginal"],
        default="policy_marginal",
    )
    parser.add_argument(
        "--env-comm-mode",
        choices=["none", "trigger", "centralized"],
        default="trigger",
        help="Execution-time belief sharing. Use trigger for semi, centralized for upper-bound style runs.",
    )
    parser.add_argument("--guard-actions", action="store_true")
    parser.add_argument("--qmdp-action-limit", type=int, default=0)
    parser.add_argument("--position-action-mask", action="store_true", default=False)
    parser.add_argument("--no-position-action-mask", dest="position_action_mask", action="store_false")
    parser.add_argument("--belief-share-nodes", action="store_true")
    parser.add_argument(
        "--belief-key",
        choices=["sparse", "dense"],
        default="sparse",
        help="Belief key representation for belief-obs planner.",
    )
    parser.add_argument("--heuristic-expansion", action="store_true", default=True)
    parser.add_argument("--random-expansion", dest="heuristic_expansion", action="store_false")
    parser.add_argument(
        "--qmdp-expansion-order",
        action="store_true",
        default=True,
        help="Order untried actions by QMDP score without restricting the action space.",
    )
    parser.add_argument(
        "--random-expansion-order",
        dest="qmdp_expansion_order",
        action="store_false",
        help="Use random order among untried actions after the default action.",
    )
    parser.add_argument(
        "--symmetry-break-rank",
        action="store_true",
        default=True,
        help="Break equal QMDP local-rank scores differently by agent.",
    )
    parser.add_argument(
        "--no-symmetry-break-rank",
        dest="symmetry_break_rank",
        action="store_false",
    )

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--verbose-first", action="store_true")
    parser.add_argument("--timing-only", action="store_true")
    parser.add_argument("--table", action="store_true")
    parser.add_argument("--fullsim", action="store_true")
    parser.add_argument("--trials-per-target", type=int, default=1)
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Print timing breakdowns for benchmark adapter and planner phases.",
    )

    args = parser.parse_args()

    if args.mode == "centralized" and args.env_comm_mode == "trigger":
        args.env_comm_mode = "centralized"
    if args.mode == "decentralized" and args.env_comm_mode == "trigger":
        args.env_comm_mode = "none"

    if args.table:
        run_table(args)
        return

    if args.fullsim:
        run_fullsim(args)
        return

    model = LabyrinthModel(args.benchmark, args.mode)
    print("Online ObsDecMCTS Labyrinth benchmark")
    print(
        f"benchmark={args.benchmark} bid={model.bid}, nodes={model.num_nodes}, "
        f"edges={len(model.edges)}, horizon={args.horizon}, episodes={args.episodes}"
    )
    print(
        f"mode={args.mode}, env_comm_mode={args.env_comm_mode}, planner={args.planner}, "
        f"outer_iters={args.outer_iters}, tau={args.tau}, num_seq={args.num_seq}, "
        f"num_samples={args.num_samples}, qmdp_action_limit={args.qmdp_action_limit}, "
        f"position_action_mask={args.position_action_mask}, action_source={args.action_source}, "
        f"seed={args.seed}"
    )
    print()

    if args.timing_only:
        run_initial_planning_timing(model, args)
        return

    returns: List[float] = []
    if args.jobs == 1 or args.verbose_first:
        for ep in range(args.episodes):
            ret = simulate_online_episode(
                model,
                args,
                episode_seed=args.seed + 104729 * ep,
                verbose=args.verbose_first and ep == 0,
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
        n_jobs = os.cpu_count() if args.jobs == -1 else max(1, args.jobs)
        args_dict = vars(args).copy()
        print(f"Running episodes in parallel with {n_jobs} workers", flush=True)
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            futures = [ex.submit(run_episode_worker, (ep, args_dict)) for ep in range(args.episodes)]
            completed = 0
            for fut in as_completed(futures):
                ep, ret = fut.result()
                returns.append(ret)
                completed += 1
                if completed % max(1, args.episodes // 10) == 0 or completed == 1:
                    mean, ci = confidence_interval_95(returns)
                    print(
                        f"completed {completed:4d}/{args.episodes}: "
                        f"last_ep={ep + 1:4d}, last={ret:8.3f}, mean={mean:8.3f} ± {ci:.3f}",
                        flush=True,
                    )

    mean, ci = confidence_interval_95(returns)
    print()
    print(f"Final mean return: {mean:.5f} ± {ci:.5f} (95% CI)")


if __name__ == "__main__":
    main()
