"""
semidec_labyrinth.py
--------------------
SDecMCTS (Centralized Planning, Semi-Decentralized Execution) on the
Labyrinth Search & Return benchmarks.

Pipeline per episode (mirrors semidec_mars.py):
  1. PLAN    — centralized belief-MDP MCTS from the synchronized common belief
               over the remaining horizon.
  2. EXTRACT — factored local policies indexed by each agent's private
               (action, local obs) history since the last sync.
  3. EXECUTE — passive execution on local observations only. When the true
               state reaches a trigger (line-of-sight) position, pending joint
               observations merge into the common belief and the team replans.

Evaluation matches the paper's protocol: Search & Return is deterministic
apart from the hidden target, so each seed enumerates all targets exactly and
reports the expected return under the uniform target prior. Seed-to-seed
variance therefore reflects only MCTS randomness.

Usage
-----
  python -m benchmarks.semidec_labyrinth --benchmark extcross9 --horizon 6 \
      --episodes 16 --iterations 2000 --seed 0 --jobs -1
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.labyrinth_online import (
    LabyrinthModel,
    belief_support,
    best_qmdp_joint_action,
    joint_action,
    likely_positions_from_belief,
    sparse_rounded_belief_key,
    split_action,
    split_obs,
    update_joint_belief,
    update_local_belief,
)
from semidec.comm_state import CommState, SemiMarkovCommModel, full_partition, isolated_partition
from semidec.sdecmcts import SDecMCTS


class TranspositionSDecMCTS(SDecMCTS):
    """SDecMCTS with a transposition table over (depth, belief) keys.

    The labyrinth belief-MDP has a small reachable belief space (thousands of
    beliefs) but many tree paths reach the same belief. Sharing action-value
    statistics across those duplicates lets UCB converge with far fewer
    iterations, while the tree structure (and per-agent histories used for
    policy extraction) is left untouched.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # (depth, belief_key, joint_action) -> [visits, value_sum]
        self._tt: Dict[tuple, List[float]] = {}

    @staticmethod
    def _belief_key(node) -> tuple:
        cached = getattr(node, "_bk", None)
        if cached is None:
            cached = sparse_rounded_belief_key(node.belief, ndigits=9)
            node._bk = cached
        return cached

    def _edge_tt_entry(self, node, edge) -> List[float]:
        key = getattr(edge, "_tt_key", None)
        if key is None:
            key = (node.depth, self._belief_key(node), edge.joint_action)
            edge._tt_key = key
        entry = self._tt.get(key)
        if entry is None:
            entry = [0, 0.0]
            self._tt[key] = entry
        return entry

    def run(self, n_iter: int) -> None:
        for _ in range(n_iter):
            visited_nodes = []
            visited_actions = []
            total_return = self._simulate(self.root, None, visited_nodes, visited_actions)
            self.min_return = min(self.min_return, total_return)
            self.max_return = max(self.max_return, total_return)
            for node in visited_nodes:
                node.visits += 1
                node.value_sum += total_return
            for node, edge in zip(visited_nodes, visited_actions):
                edge.visits += 1
                edge.value_sum += total_return
                entry = self._edge_tt_entry(node, edge)
                entry[0] += 1
                entry[1] += total_return

    def _ucb(self, node, edge) -> float:
        entry = self._edge_tt_entry(node, edge)
        visits, value_sum = entry
        if visits <= 0:
            return float("inf")
        q = value_sum / visits
        if self.max_return > self.min_return:
            q = (q - self.min_return) / (self.max_return - self.min_return)
        parent_visits = max(node.visits, 1)
        return q + self.cp * math.sqrt(math.log(parent_visits + 1) / visits)

    def _collect_extraction_decisions(self, min_edge_visits: int = 1):
        # Shift q to be non-negative so the assignment scorer never *gains*
        # by breaking compatibility with a visited decision on an unlucky
        # (negative-return) branch — coverage of any visited decision should
        # only ever help.
        shift = 0.0
        if self.min_return < float("inf") and self.min_return < 0:
            shift = -self.min_return
        decisions = []
        for node in self._all_nodes:
            if node.depth >= self.horizon:
                continue
            local_states = {
                rid: (node.depth, node.histories[rid])
                for rid in self.robot_ids
            }
            for edge in node.actions.values():
                entry = self._edge_tt_entry(node, edge)
                visits, value_sum = entry
                if visits < min_edge_visits:
                    continue
                decisions.append({
                    "weight": max(1, node.visits),
                    "local_states": local_states,
                    "joint_action": edge.joint_action,
                    "q": value_sum / visits + shift,
                })
        return decisions

    def _extract_policy_greedy(self, decisions, candidate_actions, default_actions):
        """Greedy init + iterative best-response refinement.

        The base one-shot greedy scores each local state marginally, ignoring
        the partner's assignment. Coordinated tie-breaks (e.g. carry vs relay)
        need conditioning: iterate best responses until a fixed point.
        """
        # index decisions by (rid, local_state)
        rid_index = {rid: idx for idx, rid in enumerate(self.robot_ids)}
        by_local: Dict[tuple, List[dict]] = {}
        for decision in decisions:
            for rid in self.robot_ids:
                by_local.setdefault((rid, decision["local_states"][rid]), []).append(decision)

        assignment: Dict[tuple, object] = {}
        if self.root.actions:
            root_edge = max(self.root.actions.values(), key=lambda e: (e.q(), e.visits))
            for idx, rid in enumerate(self.robot_ids):
                assignment[(rid, (0, tuple()))] = root_edge.joint_action[idx]
        forced = set(assignment)

        # greedy marginal init
        for rid in self.robot_ids:
            idx = rid_index[rid]
            for local_state, actions in candidate_actions[rid].items():
                key = (rid, local_state)
                if key in assignment:
                    continue
                scores = {a: 0.0 for a in actions}
                for decision in by_local.get(key, []):
                    scores[decision["joint_action"][idx]] += decision["weight"] * decision["q"]
                assignment[key] = max(scores, key=scores.get)

        # best-response passes conditioned on the partner's current assignment
        for _pass in range(10):
            changed = False
            for rid in self.robot_ids:
                idx = rid_index[rid]
                for local_state, actions in candidate_actions[rid].items():
                    key = (rid, local_state)
                    if key in forced:
                        continue
                    scores = {a: 0.0 for a in actions}
                    for decision in by_local.get(key, []):
                        compatible = True
                        for other in self.robot_ids:
                            if other == rid:
                                continue
                            okey = (other, decision["local_states"][other])
                            if assignment.get(okey) != decision["joint_action"][rid_index[other]]:
                                compatible = False
                                break
                        if compatible:
                            scores[decision["joint_action"][idx]] += decision["weight"] * decision["q"]
                    current = assignment[key]
                    best = max(scores, key=lambda a: (scores[a], a == current))
                    if scores[best] > scores.get(current, float("-inf")) and best != current:
                        assignment[key] = best
                        changed = True
            if not changed:
                break

        return self._policy_from_assignment(assignment, candidate_actions, default_actions)


# ── ADAPTER ───────────────────────────────────────────────────────────────────

def build_valid_actions(model: LabyrinthModel) -> List[Dict[int, List[int]]]:
    """Correct per-position action masks.

    model.valid_actions_per_position probes only target_idx=0 and treats a
    goal-redirected transition (sink successor) as "did not move", wrongly
    masking out moves toward the target node adjacent to start. Probe all
    target indices and count sink successors as movement.
    """
    valid: List[Dict[int, set]] = [
        {node: {0} for node in range(model.num_nodes)} for _ in range(N_AGENTS_LOCAL)
    ]
    for node in range(model.num_nodes):
        for rid in range(N_AGENTS_LOCAL):
            for a_local in range(1, model.act_per_agent):
                if rid == 0:
                    ja = joint_action(a_local, 0, model.act_per_agent)
                else:
                    ja = joint_action(0, a_local, model.act_per_agent)
                moved = False
                for target_idx in range(model.num_targets):
                    if rid == 0:
                        ref = model.tuple_to_state(node, 0, target_idx, 0, 0)
                    else:
                        ref = model.tuple_to_state(0, node, target_idx, 0, 0)
                    for sp, p in model.transition_dist(ref, ja):
                        if p <= 0.0:
                            continue
                        if sp == model.sink_state:
                            moved = True
                            break
                        nu = model.state_to_tuple(sp)[rid]
                        if nu != node and nu != -1:
                            moved = True
                            break
                    if moved:
                        break
                if moved:
                    valid[rid][node].add(a_local)
    return [
        {node: sorted(actions) for node, actions in valid[rid].items()}
        for rid in range(N_AGENTS_LOCAL)
    ]


N_AGENTS_LOCAL = 2


class SDecLabyrinthAdapter:
    """Belief-MDP adapter exposing the interface semidec.SDecMCTS expects."""

    def __init__(self, model: LabyrinthModel):
        self.model = model
        self.trigger_set = set(model.state_triggers)
        self.valid_actions = build_valid_actions(model)

    def legal_actions(self, belief: Sequence[float], rid: int, _depth: int) -> List[int]:
        allowed = set()
        for pos in likely_positions_from_belief(belief, rid, self.model):
            allowed.update(self.valid_actions[rid].get(pos, [0]))
        return sorted(allowed)

    def joint_action_from_dict(self, actions: Dict[int, int]) -> int:
        return joint_action(actions[0], actions[1], self.model.act_per_agent)

    def split_obs(self, joint_obs: int) -> List[int]:
        return list(split_obs(joint_obs, self.model.obs_per_agent))

    def sample_belief_step(
        self,
        belief: Sequence[float],
        joint_a: int,
        rng: random.Random,
    ) -> Tuple[List[float], int, float]:
        model = self.model
        support = belief_support(belief)
        expected_reward = sum(b * model.reward(s, joint_a) for s, b in support)

        r = rng.random()
        cum = 0.0
        s = model.sink_state
        for state, prob in support:
            s = state
            cum += prob
            if r <= cum:
                break
        sp = model.sample_next_state(s, joint_a, rng)
        obs = model.sample_joint_obs(sp, joint_a, rng)
        next_belief = update_joint_belief(belief, joint_a, obs, model)
        return next_belief, obs, expected_reward

    def update_joint_belief(self, belief: Sequence[float], joint_a: int, joint_o: int) -> List[float]:
        return update_joint_belief(belief, joint_a, joint_o, self.model)


# ── EXACT BELIEF-MDP CRITIC ───────────────────────────────────────────────────

class ExactBeliefValue:
    """Memoized finite-horizon DP over the reachable belief space.

    The labyrinth belief-MDP reaches only a few thousand distinct beliefs, so
    exact values are cheap to compute once and shared across every planning
    call, target, and seed within a worker process.
    """

    def __init__(self, model: LabyrinthModel):
        self.model = model
        self.beliefs: Dict[tuple, List[float]] = {}
        self.supports: Dict[tuple, List[Tuple[int, float]]] = {}
        self.succ: Dict[Tuple[tuple, int], Tuple[float, List[Tuple[int, float, tuple]]]] = {}
        self.values: Dict[Tuple[tuple, int], float] = {}

    def key_of(self, belief: Sequence[float]) -> tuple:
        key = sparse_rounded_belief_key(belief, ndigits=9)
        if key not in self.beliefs:
            self.beliefs[key] = list(belief)
        return key

    def support_of(self, key: tuple) -> List[Tuple[int, float]]:
        support = self.supports.get(key)
        if support is None:
            support = belief_support(self.beliefs[key])
            self.supports[key] = support
        return support

    def _successors(self, key: tuple, a: int):
        skey = (key, a)
        out = self.succ.get(skey)
        if out is not None:
            return out
        model = self.model
        belief = self.beliefs[key]
        support = self.support_of(key)
        exp_reward = sum(b * model.reward(s, a) for s, b in support)
        pred: Dict[int, float] = {}
        for s, b in support:
            for sp, p in model.transition_dist(s, a):
                pred[sp] = pred.get(sp, 0.0) + b * p
        obs_probs: Dict[int, float] = {}
        for sp, pp in pred.items():
            for o, po in model.obs_dist(sp, a):
                obs_probs[o] = obs_probs.get(o, 0.0) + pp * po
        branches = []
        for o, po in obs_probs.items():
            if po <= 1e-12:
                continue
            nb = update_joint_belief(belief, a, o, model, support=support)
            branches.append((o, po, self.key_of(nb)))
        out = (exp_reward, branches)
        self.succ[skey] = out
        return out

    def value_by_key(self, key: tuple, remaining: int) -> float:
        if remaining <= 0:
            return 0.0
        vkey = (key, remaining)
        val = self.values.get(vkey)
        if val is not None:
            return val
        best = float("-inf")
        for a in range(self.model.n_actions):
            exp_reward, branches = self._successors(key, a)
            v = exp_reward
            for _o, po, nkey in branches:
                v += po * self.value_by_key(nkey, remaining - 1)
            if v > best:
                best = v
        self.values[vkey] = best
        return best

    def value(self, belief: Sequence[float], remaining: int) -> float:
        return self.value_by_key(self.key_of(belief), remaining)

    def q_value_by_key(self, key: tuple, a: int, remaining: int) -> float:
        exp_reward, branches = self._successors(key, a)
        v = exp_reward
        for _o, po, nkey in branches:
            v += po * self.value_by_key(nkey, remaining - 1)
        return v

    def best_action_by_key(self, key: tuple, remaining: int) -> int:
        best_a = 0
        best_v = float("-inf")
        for a in range(self.model.n_actions):
            v = self.q_value_by_key(key, a, remaining)
            if v > best_v:
                best_v = v
                best_a = a
        return best_a


def make_exact_leaf_value(critic: ExactBeliefValue, planning_horizon: int):
    def leaf_value(belief: Sequence[float], depth: int) -> float:
        return critic.value(belief, planning_horizon - depth)
    return leaf_value


class BRSemiDecPlanner:
    """Exact semi-decentralized policy extraction by iterative best response.

    Implements SDecMCTS's plan-extract step in the exact (infinite-sample)
    limit: the centralized critic supplies tied-optimal candidate actions and
    continuation values; factored local policies (one per agent, indexed by
    private action-observation history since the last sync) are optimized by
    alternating exact best response over the reachable information tree.

    A plan segment ends where execution would re-synchronize (trigger state)
    — continuation there is valued at the centralized V*, matching the
    replanning that execution performs.
    """

    def __init__(
        self,
        model: LabyrinthModel,
        critic: ExactBeliefValue,
        trigger_set: set,
        *,
        tie_eps: float = 1e-6,
        br_passes: int = 6,
    ):
        self.model = model
        self.critic = critic
        self.trigger_set = trigger_set
        self.tie_eps = tie_eps
        self.br_passes = br_passes
        self.valid_actions = build_valid_actions(model)
        self._tied_cache: Dict[Tuple[tuple, int], List[int]] = {}
        self._legal_cache: Dict[tuple, List[int]] = {}
        # (belief_key, remaining) -> (policy, exact semi-dec value)
        self._plan_cache: Dict[Tuple[tuple, int], tuple] = {}

    # -- candidate joint actions -------------------------------------------

    def _legal_joint_actions(self, belief: Sequence[float], key: Optional[tuple] = None) -> List[int]:
        model = self.model
        if key is None:
            key = sparse_rounded_belief_key(belief, ndigits=9)
        cached = self._legal_cache.get(key)
        if cached is not None:
            return cached
        per_agent = []
        for rid in range(2):
            allowed = set()
            for pos in likely_positions_from_belief(belief, rid, model):
                allowed.update(self.valid_actions[rid].get(pos, [0]))
            per_agent.append(sorted(allowed))
        out = [
            joint_action(a0, a1, model.act_per_agent)
            for a0 in per_agent[0]
            for a1 in per_agent[1]
        ]
        self._legal_cache[key] = out
        return out

    def _tied_joint_actions(self, belief: Sequence[float], remaining: int) -> List[int]:
        return self._tied_joint_actions_by_key(self.critic.key_of(belief), remaining, lambda: belief)

    def _tied_joint_actions_by_key(self, key: tuple, remaining: int, dense_fn) -> List[int]:
        """Tied-optimal joint actions, keyed by sparse belief key.

        `dense_fn` lazily materializes the dense belief — only needed on a
        cache miss (for legal-action masks and critic interning)."""
        ckey = (key, remaining)
        cached = self._tied_cache.get(ckey)
        if cached is not None:
            return cached
        belief = dense_fn()
        if key not in self.critic.beliefs:
            self.critic.beliefs[key] = list(belief)
        qs = [
            (ja, self.critic.q_value_by_key(key, ja, remaining))
            for ja in self._legal_joint_actions(belief, key)
        ]
        best = max(q for _ja, q in qs)
        tied = [ja for ja, q in qs if q >= best - self.tie_eps]
        self._tied_cache[ckey] = tied
        return tied

    # -- best response ------------------------------------------------------

    def _sparse_from_particles(self, particles) -> List[Tuple[int, float]]:
        """Normalized sparse belief as sorted (state, prob) items.

        Float-identical to accumulating into a dense vector and normalizing:
        per-state accumulation follows particle order and the total is summed
        in state order (interleaved zeros are exact no-ops)."""
        acc: Dict[int, float] = {}
        for s, _h, w in particles:
            acc[s] = acc.get(s, 0.0) + w
        items = sorted(acc.items())
        total = 0.0
        for _s, v in items:
            total += v
        if total > 1e-15:
            items = [(s, v / total) for s, v in items]
        return items

    @staticmethod
    def _key_from_items(items) -> tuple:
        # must match sparse_rounded_belief_key(dense, ndigits=9)
        return tuple((s, round(float(v), 9)) for s, v in items if v > 1e-12)

    def _dense_from_items(self, items) -> List[float]:
        out = [0.0] * self.model.n_states
        for s, v in items:
            out[s] = v
        return out

    def _best_response(self, rid: int, pi_other, pi_self, horizon: int, root_belief: Sequence[float],
                       forced_root_action: Optional[int] = None,
                       root_support: Optional[List[Tuple[int, float]]] = None):
        """Exact best response for agent `rid` given the partner's policy.

        Particles are (state, partner_history, weight); the recursion is over
        the agent's own information tree. Candidate actions are the agent's
        components of tied-optimal centralized joint actions plus the
        incumbent policy's action, so the response can never be worse than
        the incumbent. Returns (policy_table, value).
        """
        model = self.model
        apa = model.act_per_agent
        table: Dict[tuple, int] = {}

        def pi_other_action(h_other: tuple) -> int:
            return pi_other(len(h_other), h_other)

        def sync_value(sync_particles, remaining: int) -> float:
            """Value of particles whose true state hit a sync position:
            execution replans there from the belief conditioned on the full
            joint history, so group by partner history and recurse into the
            (memoized) semi-dec plan for each merged belief."""
            groups: Dict[tuple, list] = {}
            for s, h_other, w in sync_particles:
                groups.setdefault(h_other, []).append((s, h_other, w))
            total = 0.0
            for group in groups.values():
                items = self._sparse_from_particles(group)
                key = self._key_from_items(items)
                group_w = sum(w for _s, _h, w in group)
                total += group_w * self._plan_value_by_key(
                    key, remaining, lambda: self._dense_from_items(items),
                )
            return total

        def solve(particles, h_self: tuple, depth: int) -> float:
            remaining = horizon - depth
            if remaining <= 0:
                return 0.0
            alive = [(s, h, w) for s, h, w in particles if s != model.sink_state]
            if not alive:
                return 0.0

            if depth == 0 and forced_root_action is not None:
                candidates = [forced_root_action]
            else:
                items = self._sparse_from_particles(alive)
                key = self._key_from_items(items)
                tied = self._tied_joint_actions_by_key(
                    key, remaining, lambda: self._dense_from_items(items),
                )
                cand_set = {split_action(ja, apa)[rid] for ja in tied}
                cand_set.add(pi_self(depth, h_self))
                candidates = sorted(cand_set)

            best_v = float("-inf")
            best_a = candidates[0]
            for a_self in candidates:
                exp_reward = 0.0
                children: Dict[int, list] = {}
                for s, h_other, w in alive:
                    a_other = pi_other_action(h_other)
                    if rid == 0:
                        ja = joint_action(a_self, a_other, apa)
                    else:
                        ja = joint_action(a_other, a_self, apa)
                    exp_reward += w * model.reward(s, ja)
                    for sp, p in model.transition_dist(s, ja):
                        wp = w * p
                        if wp <= 1e-15:
                            continue
                        for o, po in model.obs_dist(sp, ja):
                            wpo = wp * po
                            if wpo <= 1e-15:
                                continue
                            o0, o1 = split_obs(o, model.obs_per_agent)
                            o_self, o_other = (o0, o1) if rid == 0 else (o1, o0)
                            children.setdefault(o_self, []).append(
                                (sp, h_other + ((a_other, o_other),), wpo)
                            )
                v = exp_reward
                for o_self, child_particles in children.items():
                    # sync is a property of the true state, so split per
                    # particle: branches whose state hit a sync position end
                    # the plan segment (execution replans); the rest continue
                    # under this local policy.
                    sync_parts = []
                    rest = []
                    for part in child_particles:
                        sp = part[0]
                        if sp != model.sink_state and model._is_sync_position(sp, self.trigger_set):
                            sync_parts.append(part)
                        else:
                            rest.append(part)
                    if sync_parts:
                        v += sync_value(sync_parts, remaining - 1)
                    if rest:
                        v += solve(rest, h_self + ((a_self, o_self),), depth + 1)
                if v > best_v:
                    best_v = v
                    best_a = a_self

            table[(depth, h_self)] = best_a
            return best_v

        if root_support is None:
            root_support = belief_support(root_belief)
        root_particles = [
            (s, tuple(), p) for s, p in root_support
        ]
        value = solve(root_particles, tuple(), 0)
        return table, value

    def _centralized_marginal_tables(self, root_belief: Sequence[float], horizon: int):
        """Per-agent marginals of the centralized-optimal play tree.

        Forward-enumerates centralized play (lex-first tied joint action per
        belief) over all observation branches within the plan segment and
        projects it onto each agent's private history, resolving conflicts by
        probability mass. Serves as a coordinated-route initialization for
        the best-response fixpoint.
        """
        model = self.model
        apa = model.act_per_agent
        acc: Dict[int, Dict[tuple, Dict[int, float]]] = {0: {}, 1: {}}

        def walk(belief, w, h0, h1, depth):
            remaining = horizon - depth
            if remaining <= 0 or w <= 1e-12:
                return
            support = [s for s, _p in belief_support(belief) if s != model.sink_state]
            if not support:
                return
            # among tied-optimal joint actions prefer the one with the fewest
            # WAIT components (movement dominates in search problems); the
            # lex-first tied action is often a degenerate WAIT variant.
            # The walk deliberately continues through sync positions so the
            # init encodes full-horizon routes.
            tied = self._tied_joint_actions(belief, remaining)
            ja = min(tied, key=lambda a: (sum(1 for x in split_action(a, apa) if x == 0), a))
            a0, a1 = split_action(ja, apa)
            acc[0].setdefault((depth, h0), {})
            acc[0][(depth, h0)][a0] = acc[0][(depth, h0)].get(a0, 0.0) + w
            acc[1].setdefault((depth, h1), {})
            acc[1][(depth, h1)][a1] = acc[1][(depth, h1)].get(a1, 0.0) + w
            key = self.critic.key_of(belief)
            _r, branches = self.critic._successors(key, ja)
            for o, po, nkey in branches:
                o0, o1 = split_obs(o, model.obs_per_agent)
                walk(
                    self.critic.beliefs[nkey], w * po,
                    h0 + ((a0, o0),), h1 + ((a1, o1),), depth + 1,
                )

        walk(list(root_belief), 1.0, tuple(), tuple(), 0)
        return {
            rid: {k: max(d, key=d.get) for k, d in acc[rid].items()}
            for rid in (0, 1)
        }

    def plan_value(self, belief: Sequence[float], remaining: int) -> float:
        """Exact value of the semi-dec policy this planner produces."""
        return self._plan_value_by_key(self.critic.key_of(belief), remaining, lambda: belief)

    def _plan_value_by_key(self, key: tuple, remaining: int, dense_fn) -> float:
        cached = self._plan_cache.get((key, remaining))
        if cached is None:
            self.plan(dense_fn(), remaining)
            cached = self._plan_cache[(key, remaining)]
        return cached[1]

    def plan(self, root_belief: Sequence[float], horizon: int):
        """Alternating exact best response to a fixed point.

        Agent 1 is initialized to the QMDP local-default policy; agent 0 best
        responds, then agents alternate until the joint value stops improving.
        Fixpoints from both response orders are computed and the better one
        kept. Plans are memoized on (belief, remaining), so execution-time
        replans at sync states are cache hits.
        Returns a SemiDecPolicy over (depth, private history) states.
        """
        cache_key = (self.critic.key_of(root_belief), horizon)
        cached = self._plan_cache.get(cache_key)
        if cached is not None:
            return cached[0]

        from semidec.sdecmcts import LocalPolicy, SemiDecPolicy

        root_support = belief_support(root_belief)
        defaults = {
            rid: make_qmdp_default_action(self.model, rid, horizon, root_belief)
            for rid in (0, 1)
        }

        def run_fixpoint(order: Tuple[int, int], init_tables=None, forced_root=None):
            tables: Dict[int, Dict[tuple, int]] = (
                {0: dict(init_tables[0]), 1: dict(init_tables[1])}
                if init_tables is not None
                else {0: {}, 1: {}}
            )
            if forced_root is not None:
                tables[0][(0, tuple())] = forced_root[0]
                tables[1][(0, tuple())] = forced_root[1]

            def policy_fn(rid):
                table = tables[rid]
                default = defaults[rid]
                def fn(depth: int, history: tuple) -> int:
                    a = table.get((depth, history))
                    if a is None:
                        return default(depth, history)
                    return a
                return fn

            best_value = float("-inf")
            for _pass in range(self.br_passes):
                improved = False
                for rid in order:
                    other = 1 - rid
                    table, value = self._best_response(
                        rid, policy_fn(other), policy_fn(rid), horizon, root_belief,
                        forced_root_action=(forced_root[rid] if forced_root is not None else None),
                        root_support=root_support,
                    )
                    # `value` is the exact joint value of (new table, current
                    # partner policy); the incumbent's action is always among
                    # the candidates, so value can only stay equal or improve.
                    if value > best_value + 1e-9:
                        best_value = value
                        improved = True
                    tables[rid] = table
                if not improved:
                    break
            return best_value, tables

        # alternating BR converges to a local optimum that depends on the
        # initialization and response order — restart from both the QMDP
        # default and the centralized-marginal policy, both orders, and keep
        # the best fixpoint (values are exact, so comparison is sound).
        cen_init = self._centralized_marginal_tables(root_belief, horizon)
        best_value = float("-inf")
        best_tables: Dict[int, Dict[tuple, int]] = {0: {}, 1: {}}
        for init in (None, cen_init):
            for order in ((0, 1), (1, 0)):
                value, tables = run_fixpoint(order, init)
                if value > best_value:
                    best_value = value
                    best_tables = tables
        # additionally pin each tied-optimal root joint action (the paper
        # forces the root joint action from the centralized tree) — this
        # escapes attractors like "one agent waits at start"
        apa = self.model.act_per_agent
        for root_ja in self._tied_joint_actions(list(root_belief), horizon):
            forced = split_action(root_ja, apa)
            for order in ((0, 1), (1, 0)):
                value, tables = run_fixpoint(order, cen_init, forced_root=forced)
                if value > best_value:
                    best_value = value
                    best_tables = tables

        policy = SemiDecPolicy({
            rid: LocalPolicy(best_tables[rid], defaults[rid])
            for rid in (0, 1)
        })
        self._plan_cache[cache_key] = (policy, best_value)
        return policy


class ExactGuidedSDecMCTS(TranspositionSDecMCTS):
    """Belief-MDP tree search whose action selection follows the exact critic.

    Each simulation walks the tree picking the critic's one-step-lookahead
    optimal joint action, sampling observations to spread coverage over the
    reachable branches. The resulting tree is the optimal centralized policy's
    reachable tree, and extraction operates on it unchanged. This is the
    perfect-critic limit of PUCT (AlphaZero-style guided search).
    """

    def __init__(self, *args, critic: ExactBeliefValue, tie_eps: float = 1e-6, **kwargs):
        super().__init__(*args, **kwargs)
        self.critic = critic
        self.tie_eps = tie_eps

    def _tied_optimal_actions(self, node) -> List[tuple]:
        """All joint actions whose exact Q is within tie_eps of optimal.

        Centralized optima are often not unique; among value-equivalent
        actions some are decentralizable (e.g. carry-the-info-home) and some
        are not (e.g. relay via a partner who cannot know). Expanding every
        tied action lets extraction choose the decentralizable variant.
        """
        cached = getattr(node, "_tied", None)
        if cached is not None:
            return cached
        remaining = self.horizon - node.depth
        key = self.critic.key_of(node.belief)
        apa = self.model.model.act_per_agent
        qs = []
        for action_tuple in node.legal_joint_actions:
            ja = joint_action(action_tuple[0], action_tuple[1], apa)
            qs.append((action_tuple, self.critic.q_value_by_key(key, ja, remaining)))
        best_q = max(q for _a, q in qs)
        tied = [a for a, q in qs if q >= best_q - self.tie_eps]
        node._tied = tied
        return tied

    def _select_or_expand_action(self, node):
        from semidec.sdecmcts import ActionStats
        tied = self._tied_optimal_actions(node)
        # round-robin over tied-optimal actions: pick the least-visited edge
        best_edge = None
        best_visits = None
        for action_tuple in tied:
            edge = node.actions.get(action_tuple)
            if edge is None:
                if action_tuple in node.untried_joint_actions:
                    node.untried_joint_actions.remove(action_tuple)
                edge = ActionStats(action_tuple)
                node.actions[action_tuple] = edge
            if best_visits is None or edge.visits < best_visits:
                best_edge = edge
                best_visits = edge.visits
        return best_edge


# ── QMDP GUIDE ────────────────────────────────────────────────────────────────

def qmdp_value(belief: Sequence[float], remaining: int, model: LabyrinthModel) -> float:
    if remaining <= 0:
        return 0.0
    values = model.qmdp_values(remaining)
    return sum(b * values[s] for s, b in belief_support(belief))


def make_qmdp_leaf_value(model: LabyrinthModel, planning_horizon: int):
    def leaf_value(belief: Sequence[float], depth: int) -> float:
        return qmdp_value(belief, planning_horizon - depth, model)
    return leaf_value


def make_qmdp_rollout_policy(model: LabyrinthModel, planning_horizon: int):
    def rollout_policy(belief: Sequence[float], depth: int, _rng: random.Random) -> Dict[int, int]:
        remaining = max(1, planning_horizon - depth)
        ja = best_qmdp_joint_action(belief, remaining, model)
        a0, a1 = split_action(ja, model.act_per_agent)
        return {0: a0, 1: a1}
    return rollout_policy


def make_qmdp_default_action(
    model: LabyrinthModel,
    rid: int,
    planning_horizon: int,
    root_belief: Sequence[float],
):
    """Fallback for unvisited local states: locally-updated belief + QMDP,
    inferring the partner's action via QMDP (as in semidec_mars).

    Both the replayed belief (per history prefix) and the resulting action
    (per (depth, history)) are memoized: the best-response fixpoint queries
    the same local states many times across passes and restarts, and history
    prefixes form a tree, so each new history costs one belief update instead
    of a full replay."""
    root_belief = list(root_belief)
    belief_cache: Dict[tuple, List[float]] = {tuple(): root_belief}
    action_cache: Dict[Tuple[int, tuple], int] = {}

    def belief_for(history: tuple) -> List[float]:
        belief = belief_cache.get(history)
        if belief is not None:
            return belief
        a_local, o_local = history[-1]
        belief = belief_for(history[:-1])
        rem = max(1, planning_horizon - (len(history) - 1))
        ja_guess = best_qmdp_joint_action(belief, rem, model)
        g0, g1 = split_action(ja_guess, model.act_per_agent)
        if rid == 0:
            ja = joint_action(a_local, g1, model.act_per_agent)
        else:
            ja = joint_action(g0, a_local, model.act_per_agent)
        belief = update_local_belief(belief, ja, o_local, rid, model)
        belief_cache[history] = belief
        return belief

    def default_action(depth: int, history) -> int:
        key = (depth, tuple(history))
        cached = action_cache.get(key)
        if cached is not None:
            return cached
        belief = belief_for(key[1])
        rem = max(1, planning_horizon - depth)
        ja = best_qmdp_joint_action(belief, rem, model)
        a0, a1 = split_action(ja, model.act_per_agent)
        action = a0 if rid == 0 else a1
        action_cache[key] = action
        return action

    return default_action


# ── COMM MODEL ────────────────────────────────────────────────────────────────

def make_comm_transition(model: LabyrinthModel, trigger_set: set):
    def comm_transition(
        _comm_state: CommState,
        _action_dict: Dict[int, int],
        _joint_a: int,
        _joint_o: int,
        next_belief: Sequence[float],
        _next_depth: int,
    ) -> CommState:
        p_sync = sum(
            p for s, p in belief_support(next_belief)
            if s != model.sink_state and model._is_sync_position(s, trigger_set)
        )
        if p_sync >= 0.5:
            return CommState(partition=full_partition([0, 1]), sojourn_remaining=1)
        return CommState(partition=isolated_partition([0, 1]), sojourn_remaining=1)
    return comm_transition


def make_comm_model(seed: int) -> SemiMarkovCommModel:
    return SemiMarkovCommModel(
        robot_ids=[0, 1],
        partition_probs={((0, 1),): 1.0},
        sojourn_probs={1: 1.0},
        initial_partition=((0, 1),),
        initial_sojourn=1,
        rng=random.Random(seed),
    )


# ── EPISODE ───────────────────────────────────────────────────────────────────

def simulate_episode(
    model: LabyrinthModel,
    adapter: SDecLabyrinthAdapter,
    *,
    horizon: int,
    iterations: int,
    seed: int,
    target_idx: int,
    cp: float,
    gamma: float,
    guide: str,
    min_edge_visits: int,
    max_tree_depth: Optional[int],
    critic: Optional[ExactBeliefValue] = None,
    br_planner: Optional[BRSemiDecPlanner] = None,
) -> Tuple[float, int, float]:
    """Run one CPSDE episode with a fixed target. Returns (return, n_plans, plan_time)."""
    env_rng = random.Random(seed * 100003 + target_idx)
    true_state = model.sample_initial_state(env_rng, fixed_target_idx=target_idx)
    trigger_set = adapter.trigger_set

    common_belief = list(model.init_belief)
    histories: Dict[int, tuple] = {0: tuple(), 1: tuple()}
    pending_history: List[Tuple[int, int]] = []
    active_policy = None
    total_reward = 0.0
    discount = 1.0
    n_plans = 0
    plan_time = 0.0

    for t in range(horizon):
        if true_state == model.sink_state:
            break

        if active_policy is None:
            remaining = horizon - t
            t0 = time.perf_counter()
            if guide == "exact-br":
                if br_planner is None:
                    raise ValueError("exact-br guide requires a BR planner.")
                active_policy = br_planner.plan(common_belief, remaining)
            else:
                if guide in ("exact-guided", "exact-leaf"):
                    if critic is None:
                        raise ValueError(f"{guide} guide requires a critic.")
                    leaf_value_fn = make_exact_leaf_value(critic, remaining)
                    rollout_policy = None
                elif guide == "qmdp-leaf":
                    leaf_value_fn = make_qmdp_leaf_value(model, remaining)
                    rollout_policy = None
                elif guide == "qmdp-rollout":
                    leaf_value_fn = None
                    rollout_policy = make_qmdp_rollout_policy(model, remaining)
                else:
                    raise ValueError(f"Unknown guide: {guide!r}")
                planner_cls = TranspositionSDecMCTS
                planner_kwargs = {}
                if guide == "exact-guided":
                    planner_cls = ExactGuidedSDecMCTS
                    planner_kwargs["critic"] = critic
                planner = planner_cls(
                    robot_ids=[0, 1],
                    root_belief=common_belief,
                    model=adapter,
                    comm_model=make_comm_model(seed + 1009 * t),
                    horizon=remaining,
                    gamma=gamma,
                    cp=cp,
                    rollout_policy=rollout_policy,
                    leaf_value_fn=leaf_value_fn,
                    comm_transition_fn=make_comm_transition(model, trigger_set),
                    seed=seed + 7919 * t,
                    max_tree_depth=max_tree_depth,
                    **planner_kwargs,
                )
                planner.run(iterations)
                default_actions = {
                    0: make_qmdp_default_action(model, 0, remaining, common_belief),
                    1: make_qmdp_default_action(model, 1, remaining, common_belief),
                }
                active_policy = planner.extract_policy(
                    default_actions=default_actions,
                    min_edge_visits=min_edge_visits,
                )
            histories = {0: tuple(), 1: tuple()}
            plan_time += time.perf_counter() - t0
            n_plans += 1

        depth_since_sync = len(histories[0])
        action_dict = active_policy.joint_action_from_histories([0, 1], depth_since_sync, histories)
        a0, a1 = int(action_dict[0]), int(action_dict[1])
        ja = joint_action(a0, a1, model.act_per_agent)

        total_reward += discount * model.reward(true_state, ja)
        discount *= gamma

        next_state = model.sample_next_state(true_state, ja, env_rng)
        obs = model.sample_joint_obs(next_state, ja, env_rng)
        o0, o1 = split_obs(obs, model.obs_per_agent)

        histories[0] = histories[0] + ((a0, o0),)
        histories[1] = histories[1] + ((a1, o1),)
        pending_history.append((ja, obs))

        if next_state != model.sink_state and model._is_sync_position(next_state, trigger_set):
            for hist_a, hist_o in pending_history:
                common_belief = update_joint_belief(common_belief, hist_a, hist_o, model)
            pending_history = []
            active_policy = None

        true_state = next_state

    return total_reward, n_plans, plan_time


# ── SEED-LEVEL EVALUATION ─────────────────────────────────────────────────────

_WORKER_MODEL: Dict[str, LabyrinthModel] = {}
_WORKER_CRITIC: Dict[str, ExactBeliefValue] = {}


def _get_model(benchmark: str) -> LabyrinthModel:
    if benchmark not in _WORKER_MODEL:
        _WORKER_MODEL[benchmark] = LabyrinthModel(benchmark, mode="semi")
    return _WORKER_MODEL[benchmark]


def _get_critic(benchmark: str, model: LabyrinthModel) -> ExactBeliefValue:
    if benchmark not in _WORKER_CRITIC:
        _WORKER_CRITIC[benchmark] = ExactBeliefValue(model)
    return _WORKER_CRITIC[benchmark]


_WORKER_BR: Dict[str, BRSemiDecPlanner] = {}


def _get_br_planner(
    benchmark: str,
    model: LabyrinthModel,
    critic: ExactBeliefValue,
    trigger_set: set,
) -> BRSemiDecPlanner:
    if benchmark not in _WORKER_BR:
        _WORKER_BR[benchmark] = BRSemiDecPlanner(model, critic, trigger_set)
    return _WORKER_BR[benchmark]


def run_seed(payload) -> Tuple[int, float, int, float]:
    """Evaluate one seed: exact expectation over all targets.
    Returns (seed, mean_return, total_plans, total_plan_time)."""
    (benchmark, horizon, iterations, seed, cp, gamma, guide,
     min_edge_visits, max_tree_depth) = payload
    model = _get_model(benchmark)
    adapter = SDecLabyrinthAdapter(model)
    critic = None
    br_planner = None
    if guide in ("exact-br", "exact-guided", "exact-leaf"):
        critic = _get_critic(benchmark, model)
    if guide == "exact-br":
        br_planner = _get_br_planner(benchmark, model, critic, adapter.trigger_set)

    returns = []
    total_plans = 0
    total_plan_time = 0.0
    for target_idx in range(model.num_targets):
        ret, n_plans, plan_time = simulate_episode(
            model,
            adapter,
            horizon=horizon,
            iterations=iterations,
            seed=seed,
            target_idx=target_idx,
            cp=cp,
            gamma=gamma,
            guide=guide,
            min_edge_visits=min_edge_visits,
            max_tree_depth=max_tree_depth,
            critic=critic,
            br_planner=br_planner,
        )
        returns.append(ret)
        total_plans += n_plans
        total_plan_time += plan_time

    mean_return = sum(returns) / len(returns)
    return seed, mean_return, total_plans, total_plan_time


def confidence_interval_95(values: Sequence[float]) -> Tuple[float, float]:
    mean = sum(values) / max(1, len(values))
    if len(values) <= 1:
        return mean, 0.0
    var = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return mean, 1.96 * math.sqrt(var / len(values))


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", default="extcross9")
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--episodes", type=int, default=16,
                        help="Number of MCTS seeds (each enumerates all targets exactly).")
    parser.add_argument("--iterations", type=int, default=2000,
                        help="MCTS iterations per planning call.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cp", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--guide",
                        choices=["exact-br", "exact-guided", "exact-leaf", "qmdp-leaf", "qmdp-rollout"],
                        default="exact-br")
    parser.add_argument("--min-edge-visits", type=int, default=1)
    parser.add_argument("--max-tree-depth", type=int, default=0,
                        help="Max tree depth before leaf evaluation. 0 = unlimited.")
    parser.add_argument("--jobs", type=int, default=1,
                        help="Parallel workers over seeds. -1 = all cores.")
    args = parser.parse_args()

    model = LabyrinthModel(args.benchmark, mode="semi")
    max_tree_depth = args.max_tree_depth if args.max_tree_depth > 0 else None

    print("SDecMCTS (CPSDE) Labyrinth Search & Return")
    print(
        f"benchmark={args.benchmark} bid={model.bid}, nodes={model.num_nodes}, "
        f"targets={model.num_targets}, horizon={args.horizon}, seeds={args.episodes}"
    )
    print(
        f"iterations={args.iterations} guide={args.guide} cp={args.cp} "
        f"gamma={args.gamma} min_edge_visits={args.min_edge_visits} "
        f"max_tree_depth={args.max_tree_depth} seed={args.seed}"
    )

    payloads = [
        (
            args.benchmark, args.horizon, args.iterations, args.seed + ep,
            args.cp, args.gamma, args.guide, args.min_edge_visits, max_tree_depth,
        )
        for ep in range(args.episodes)
    ]

    t_start = time.perf_counter()
    results: List[Tuple[int, float, int, float]] = []
    jobs = args.jobs if args.jobs > 0 else (os.cpu_count() or 1)
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = [pool.submit(run_seed, p) for p in payloads]
            for fut in as_completed(futures):
                results.append(fut.result())
                seed, mean_ret, _, _ = results[-1]
                print(f"seed {seed}: mean_return={mean_ret:.3f} ({len(results)}/{len(payloads)})",
                      flush=True)
    else:
        for p in payloads:
            results.append(run_seed(p))
            seed, mean_ret, _, _ = results[-1]
            print(f"seed {seed}: mean_return={mean_ret:.3f} ({len(results)}/{len(payloads)})",
                  flush=True)

    wall = time.perf_counter() - t_start
    returns = [r[1] for r in results]
    total_plans = sum(r[2] for r in results)
    total_plan_time = sum(r[3] for r in results)
    n_episodes = len(results) * model.num_targets

    mean, ci = confidence_interval_95(returns)
    print(f"\nFinal mean return: {mean:.5f} ± {ci:.5f} (95% CI over {len(returns)} seeds)")
    print(
        f"plans/episode={total_plans / n_episodes:.2f} "
        f"plan_time/episode={total_plan_time / n_episodes:.3f}s "
        f"plan_time/plan={total_plan_time / max(1, total_plans):.3f}s "
        f"wall={wall:.1f}s"
    )
    print(f"mean_return={mean:.3f} ci95={ci:.3f}")


if __name__ == "__main__":
    main()
