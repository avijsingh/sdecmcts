"""
medevac_obs_online.py
---------------------
Online/receding-horizon Maritime MEDEVAC benchmark for ObsDecMCTS.

This version uses the observation-conditioned Dec-MCTS planner:

    local action-observation history -> action

Execution remains online/receding-horizon:

    plan policy tree for remaining horizon
    execute root action only
    observe
    update belief
    replan

For decentralized RS-MAA* comparisons, agents keep private beliefs during
execution. Set --env-comm-period > 0 only for explicit communication ablations.

Domain:
  - 2 agents:
        agent 0 = helicopter
        agent 1 = ship
  - 4x4 grid
  - state = (helo_x, helo_y, ship_x, ship_y, carry)
  - carry = 0 means patient not picked up
  - carry = 1 means patient onboard / needs dropoff
  - target is patient if carry=0, hospital if carry=1
  - actions:
        WAIT     = 0
        ADVANCE  = 1
        EXCHANGE = 2
  - observations:
        NOT_AT_TARGET = 0
        AT_TARGET     = 1
"""

from __future__ import annotations

import argparse
import bisect
import math
from itertools import accumulate
import os
import random
import sys
import time
from array import array
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from obs_decmcts import ObsDecMCTS, ObsDecMCTSTeam, StepResult, PolicyTree
from belief_obs_decmcts import BeliefObsDecMCTS, BeliefObsDecMCTSTeam, BeliefPolicyTree, rounded_belief_key


# =============================================================================
# Constants
# =============================================================================

G = 4
N_AGENTS = 2

WAIT = 0
ADVANCE = 1
EXCHANGE = 2

ACTION_NAME = {
    WAIT: "WAIT",
    ADVANCE: "ADVANCE",
    EXCHANGE: "EXCHANGE",
}

NOT_AT_TARGET = 0
AT_TARGET = 1

OBS_NAME = {
    NOT_AT_TARGET: "not-at-target",
    AT_TARGET: "at-target",
}

ACT_PER_AGENT = 3
OBS_PER_AGENT = 2
N_ACTIONS = ACT_PER_AGENT ** N_AGENTS
N_OBS = OBS_PER_AGENT ** N_AGENTS
N_STATES = (G * G) * (G * G) * 2

PATIENT = (1, 1)
HOSPITAL = (3, 3)
HELO_START = (0, 1)
SHIP_START = (1, 0)

P_MOVE_HELO = 0.95
P_MOVE_SHIP = 0.85
P_PICKUP = 0.95
P_DROPOFF = 0.95

STEP_COST = -0.3
WRONG_EXCHANGE_COST = -1.0
PICKUP_REWARD = 5.0
DROPOFF_REWARD = 12.0
PICKUP_MISMATCH_PENALTY = -6.0
DROPOFF_MISMATCH_PENALTY = -6.0

# Paper reference values for Maritime MEDEVAC. These are expected finite-horizon
# values, so comparison runs should use --gamma 1.0 and many episodes.
REFERENCE_DECENTRALIZED_RSMAA = {
    5: 3.46,
    6: 3.18348,
    7: 3.26710,
    8: 8.03228,
    9: 10.79,
    10: 12.15,
}

REFERENCE_CENTRALIZED_RSSDA = {
    5: 3.50,
    6: 3.19945,
    7: 6.61819,
    8: 10.88244,
    9: 13.01,
    10: 13.61,
}


# =============================================================================
# State utilities
# =============================================================================

def pos_to_idx(x: int, y: int) -> int:
    return y * G + x


def idx_to_pos(idx: int) -> Tuple[int, int]:
    return idx % G, idx // G


def state_id(hx: int, hy: int, sx: int, sy: int, carry: int) -> int:
    return (
        carry * (G * G * G * G)
        + pos_to_idx(hx, hy) * (G * G)
        + pos_to_idx(sx, sy)
    )


def id_to_state(s: int) -> Tuple[int, int, int, int, int]:
    carry = 1 if s >= (G * G * G * G) else 0
    rem = s - carry * (G * G * G * G)
    hpos = rem // (G * G)
    spos = rem % (G * G)
    hx, hy = idx_to_pos(hpos)
    sx, sy = idx_to_pos(spos)
    return hx, hy, sx, sy, carry


def target_for_carry(carry: int) -> Tuple[int, int]:
    return PATIENT if carry == 0 else HOSPITAL


def at_patient(x: int, y: int) -> bool:
    return (x, y) == PATIENT


def at_hospital(x: int, y: int) -> bool:
    return (x, y) == HOSPITAL


def at_target_for_agent(x: int, y: int, carry: int) -> int:
    return AT_TARGET if (x, y) == target_for_carry(carry) else NOT_AT_TARGET


# def next_step_toward(x: int, y: int, tx: int, ty: int, prefer: str) -> Tuple[int, int]:
#     if (x, y) == (tx, ty):
#         return x, y

#     if prefer == "H":
#         if x < tx:
#             return x + 1, y
#         if x > tx:
#             return x - 1, y
#         if y < ty:
#             return x, y + 1
#         if y > ty:
#             return x, y - 1
#     else:
#         if y < ty:
#             return x, y + 1
#         if y > ty:
#             return x, y - 1
#         if x < tx:
#             return x + 1, y
#         if x > tx:
#             return x - 1, y

#     return x, y

# REMOVE CHECK
def next_step_toward(x: int, y: int, tx: int, ty: int, prefer: str) -> Tuple[int, int]:
    if (x, y) == (tx, ty):
        return x, y

    if prefer == "H":
        if x < tx:
            return x + 1, y
        if y < ty:
            return x, y + 1
    else:
        if y < ty:
            return x, y + 1
        if x < tx:
            return x + 1, y

    return x, y

def advance_helo(x: int, y: int, carry: int) -> Tuple[int, int, float]:
    tx, ty = target_for_carry(carry)
    nx, ny = next_step_toward(x, y, tx, ty, prefer="H")
    p_succ = P_MOVE_HELO if (nx, ny) != (x, y) else 1.0
    return nx, ny, p_succ


def advance_ship(x: int, y: int, carry: int) -> Tuple[int, int, float]:
    tx, ty = target_for_carry(carry)
    nx, ny = next_step_toward(x, y, tx, ty, prefer="V")
    p_succ = P_MOVE_SHIP if (nx, ny) != (x, y) else 1.0
    return nx, ny, p_succ


# =============================================================================
# MEDEVAC sparse generative model
# =============================================================================

class MedevacModel:
    def __init__(self):
        self.rewards: List[float] = [0.0] * (N_ACTIONS * N_STATES)
        self.transitions: List[Tuple[array, array]] = []
        self.obs_of_state_action: List[int] = [0] * (N_ACTIONS * N_STATES)

        self.init_state = state_id(
            HELO_START[0],
            HELO_START[1],
            SHIP_START[0],
            SHIP_START[1],
            0,
        )

        self.init_belief = [0.0] * N_STATES
        self.init_belief[self.init_state] = 1.0

        self._build()

    @staticmethod
    def joint_action(a0: int, a1: int) -> int:
        return a0 + ACT_PER_AGENT * a1

    @staticmethod
    def split_action(joint_a: int) -> Tuple[int, int]:
        return joint_a % ACT_PER_AGENT, joint_a // ACT_PER_AGENT

    @staticmethod
    def joint_obs(o0: int, o1: int) -> int:
        return o0 + OBS_PER_AGENT * o1

    @staticmethod
    def split_obs(joint_o: int) -> Tuple[int, int]:
        return joint_o % OBS_PER_AGENT, joint_o // OBS_PER_AGENT

    def reward(self, s: int, joint_a: int) -> float:
        return self.rewards[joint_a * N_STATES + s]

    def transition_sparse(self, s: int, joint_a: int) -> Tuple[array, array]:
        return self.transitions[joint_a * N_STATES + s]

    def obs_from_next_state(self, sp: int, joint_a: int) -> int:
        return self.obs_of_state_action[joint_a * N_STATES + sp]

    def sample_initial_state(self, rng: random.Random) -> int:
        return self.init_state

    def sample_next_state(self, s: int, joint_a: int, rng: random.Random) -> int:
        indices, probs = self.transition_sparse(s, joint_a)
        r = rng.random()
        cum = 0.0

        for sp, p in zip(indices, probs):
            cum += p
            if r <= cum:
                return int(sp)

        return int(indices[-1])

    def _build(self) -> None:
        self.transitions = [None] * (N_ACTIONS * N_STATES)  # type: ignore

        for a0 in range(ACT_PER_AGENT):
            for a1 in range(ACT_PER_AGENT):
                joint_a = self.joint_action(a0, a1)

                for s in range(N_STATES):
                    hx, hy, sx, sy, carry = id_to_state(s)

                    reward = STEP_COST

                    helo_at_pat = at_patient(hx, hy)
                    ship_at_pat = at_patient(sx, sy)
                    helo_at_hosp = at_hospital(hx, hy)
                    ship_at_hosp = at_hospital(sx, sy)

                    # Penalty for exchange away from relevant sites.
                    if a0 == EXCHANGE and not (helo_at_pat or helo_at_hosp):
                        reward += WRONG_EXCHANGE_COST
                    if a1 == EXCHANGE and not (ship_at_pat or ship_at_hosp):
                        reward += WRONG_EXCHANGE_COST

                    # Expected immediate rewards/penalties for pickup/dropoff.
                    if carry == 0:
                        if helo_at_pat and ship_at_pat and a0 == EXCHANGE and a1 == EXCHANGE:
                            reward += P_PICKUP * PICKUP_REWARD

                        if (
                            (a0 == EXCHANGE and helo_at_pat and not (a1 == EXCHANGE and ship_at_pat))
                            or (a1 == EXCHANGE and ship_at_pat and not (a0 == EXCHANGE and helo_at_pat))
                        ):
                            reward += PICKUP_MISMATCH_PENALTY

                    else:
                        if helo_at_hosp and ship_at_hosp and a0 == EXCHANGE and a1 == EXCHANGE:
                            reward += P_DROPOFF * DROPOFF_REWARD

                        if (
                            (a0 == EXCHANGE and helo_at_hosp and not (a1 == EXCHANGE and ship_at_hosp))
                            or (a1 == EXCHANGE and ship_at_hosp and not (a0 == EXCHANGE and helo_at_hosp))
                        ):
                            reward += DROPOFF_MISMATCH_PENALTY

                    self.rewards[joint_a * N_STATES + s] = reward

                    # Movement attempts.
                    if a0 == ADVANCE:
                        nx_h, ny_h, p_h = advance_helo(hx, hy, carry)
                    else:
                        nx_h, ny_h, p_h = hx, hy, 1.0

                    if a1 == ADVANCE:
                        nx_s, ny_s, p_s = advance_ship(sx, sy, carry)
                    else:
                        nx_s, ny_s, p_s = sx, sy, 1.0

                    movement_cases = [
                        (nx_h, ny_h, nx_s, ny_s, p_h * p_s),
                        (nx_h, ny_h, sx, sy, p_h * (1.0 - p_s)),
                        (hx, hy, nx_s, ny_s, (1.0 - p_h) * p_s),
                        (hx, hy, sx, sy, (1.0 - p_h) * (1.0 - p_s)),
                    ]

                    # EXCHANGE means no movement for that agent.
                    fixed_cases = []
                    for hx2, hy2, sx2, sy2, p_case in movement_cases:
                        if p_case <= 0.0:
                            continue

                        if a0 == EXCHANGE:
                            hx2, hy2 = hx, hy
                        if a1 == EXCHANGE:
                            sx2, sy2 = sx, sy

                        fixed_cases.append((hx2, hy2, sx2, sy2, p_case))

                    accum: Dict[int, float] = {}

                    for hx2, hy2, sx2, sy2, p_case in fixed_cases:
                        if p_case <= 0.0:
                            continue

                        carry_outcomes: List[Tuple[int, float]] = [(carry, 1.0)]

                        if carry == 0:
                            if helo_at_pat and ship_at_pat and a0 == EXCHANGE and a1 == EXCHANGE:
                                carry_outcomes = [
                                    (1, P_PICKUP),
                                    (0, 1.0 - P_PICKUP),
                                ]
                        else:
                            if helo_at_hosp and ship_at_hosp and a0 == EXCHANGE and a1 == EXCHANGE:
                                carry_outcomes = [
                                    (0, P_DROPOFF),
                                    (1, 1.0 - P_DROPOFF),
                                ]

                        for carry2, p_carry in carry_outcomes:
                            sp = state_id(hx2, hy2, sx2, sy2, carry2)
                            accum[sp] = accum.get(sp, 0.0) + p_case * p_carry

                    # Normalize sparse transition.
                    total_p = sum(accum.values())
                    if total_p <= 0.0:
                        accum[s] = 1.0
                        total_p = 1.0

                    indices = array("I")
                    probs = array("d")

                    for sp, p in sorted(accum.items()):
                        if p > 0.0:
                            indices.append(sp)
                            probs.append(p / total_p)

                    self.transitions[joint_a * N_STATES + s] = (indices, probs)

        # Deterministic observations from resulting state.
        for joint_a in range(N_ACTIONS):
            for sp in range(N_STATES):
                hx, hy, sx, sy, carry = id_to_state(sp)
                o0 = at_target_for_agent(hx, hy, carry)
                o1 = at_target_for_agent(sx, sy, carry)
                self.obs_of_state_action[joint_a * N_STATES + sp] = self.joint_obs(o0, o1)


# =============================================================================
# ObsDecMCTS adapter
# =============================================================================

class MedevacObsModelAdapter:
    def __init__(self, model: MedevacModel):
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
            return 0
        idx = bisect.bisect_left(cum, r)
        return idx if idx < len(cum) else len(cum) - 1

    def step(self, state: int, joint_action: int, rng: random.Random) -> StepResult:
        reward = self.model.reward(state, joint_action)
        next_state = self.model.sample_next_state(state, joint_action, rng)
        joint_obs = self.model.obs_from_next_state(next_state, joint_action)
        return StepResult(
            next_state=next_state,
            joint_obs=joint_obs,
            reward=reward,
        )

    def split_obs(self, joint_obs: int) -> Tuple[int, int]:
        return self.model.split_obs(joint_obs)

    def joint_action_from_dict(self, actions: Dict[int, int]) -> int:
        return self.model.joint_action(actions[0], actions[1])

    def update_belief(
        self,
        belief: Sequence[float],
        joint_action: int,
        local_obs: int,
        robot_id: int,
    ) -> List[float]:
        return update_local_belief(
            belief=belief,
            joint_a=joint_action,
            local_obs=local_obs,
            rid=robot_id,
            model=self.model,
        )


# =============================================================================
# Belief updates
# =============================================================================

def normalize_belief(b: List[float]) -> List[float]:
    total = sum(b)
    if total <= 1e-15:
        return [1.0 / len(b)] * len(b)
    return [x / total for x in b]


def update_local_belief(
    belief: Sequence[float],
    joint_a: int,
    local_obs: int,
    rid: int,
    model: MedevacModel,
) -> List[float]:
    out = [0.0] * N_STATES

    for s, b_s in enumerate(belief):
        if b_s <= 0.0:
            continue

        indices, probs = model.transition_sparse(s, joint_a)

        for sp, p in zip(indices, probs):
            joint_o = model.obs_from_next_state(int(sp), joint_a)
            o0, o1 = model.split_obs(joint_o)
            obs_i = o0 if rid == 0 else o1

            if obs_i == local_obs:
                out[int(sp)] += b_s * p

    return normalize_belief(out)


def update_joint_belief(
    belief: Sequence[float],
    joint_a: int,
    joint_obs: int,
    model: MedevacModel,
) -> List[float]:
    out = [0.0] * N_STATES

    for s, b_s in enumerate(belief):
        if b_s <= 0.0:
            continue

        indices, probs = model.transition_sparse(s, joint_a)

        for sp, p in zip(indices, probs):
            if model.obs_from_next_state(int(sp), joint_a) == joint_obs:
                out[int(sp)] += b_s * p

    return normalize_belief(out)


def belief_support_summary(belief: Sequence[float], k: int = 5) -> str:
    top = sorted(
        [(p, s) for s, p in enumerate(belief) if p > 1e-9],
        reverse=True,
    )[:k]

    parts = [
        f"{id_to_state(s)}:{p:.3f}"
        for p, s in top
    ]

    support = sum(1 for p in belief if p > 1e-9)
    return f"support={support}, top={parts}"


def belief_prob_agent_at_target(belief: Sequence[float], rid: int) -> float:
    total = 0.0

    for s, p in enumerate(belief):
        if p <= 0.0:
            continue

        hx, hy, sx, sy, carry = id_to_state(s)

        if rid == 0:
            obs = at_target_for_agent(hx, hy, carry)
        else:
            obs = at_target_for_agent(sx, sy, carry)

        if obs == AT_TARGET:
            total += p

    return total


# =============================================================================
# ObsDecMCTS policy helpers
# =============================================================================

def medevac_legal_actions_from_history(history, depth: int) -> Sequence[int]:
    return [WAIT, ADVANCE, EXCHANGE]


def medevac_default_action_from_history(history) -> int:
    # Good rollout prior:
    #   no history or not at target -> advance
    #   at target -> exchange
    if not history:
        return ADVANCE

    _last_action, last_obs = history[-1]
    if last_obs == AT_TARGET:
        return EXCHANGE

    return ADVANCE


def obs_root_action_masses(planner: ObsDecMCTS) -> Dict[str, float]:
    masses = {
        WAIT: 0.0,
        ADVANCE: 0.0,
        EXCHANGE: 0.0,
    }

    for key, p in planner.q.items():
        policy = PolicyTree.from_key(key, medevac_default_action_from_history)
        try:
            root_a = policy.action(())
            masses[root_a] += p
        except Exception:
            pass

    return {
        ACTION_NAME[a]: round(masses[a], 4)
        for a in [WAIT, ADVANCE, EXCHANGE]
    }


def obs_root_edge_stats(planner: ObsDecMCTS) -> Dict[str, Any]:
    out = {}

    for a, edge in planner.root.actions.items():
        out[ACTION_NAME[a]] = {
            "visits": edge.visits,
            "q": round(edge.q(), 3),
            "disc_visits": round(edge.disc_visits, 3),
            "disc_q": round(edge.disc_q(), 3),
            "obs_children": [OBS_NAME[o] for o in edge.obs_children.keys()],
        }

    return out


def obs_policy_support_debug(planner: ObsDecMCTS) -> List[Dict[str, Any]]:
    out = []

    for policy_key, p in sorted(planner.q.items(), key=lambda kv: kv[1], reverse=True):
        root_action = None
        for hist, action in policy_key:
            if hist == ():
                root_action = action
                break

        out.append({
            "p": round(p, 4),
            "root": ACTION_NAME[root_action] if root_action is not None else None,
            "num_entries": len(policy_key),
        })

    return out


def maybe_guard_exchange(
    belief: Sequence[float],
    rid: int,
    proposed_action: int,
    exchange_threshold: float,
) -> int:
    if proposed_action != EXCHANGE:
        return proposed_action

    p_ready = belief_prob_agent_at_target(belief, rid)

    if p_ready >= exchange_threshold:
        return EXCHANGE

    return ADVANCE

def make_medevac_legal_actions_fn(
    root_belief,
    rid,
    exchange_threshold,
    full_action_space: bool = False,
):
    def legal_actions(history, depth):
        if full_action_space:
            return [WAIT, ADVANCE, EXCHANGE]

        # Root of online replanning: use current belief.
        if history == ():
            p_ready = belief_prob_agent_at_target(root_belief, rid)

            if p_ready >= exchange_threshold:
                return [EXCHANGE, WAIT]

            return [ADVANCE]

        # Non-root policy-tree histories: use last local observation.
        _last_action, last_obs = history[-1]

        if last_obs == AT_TARGET:
            return [EXCHANGE, WAIT]

        return [ADVANCE]

    return legal_actions

def make_medevac_default_action_fn(root_belief, rid, exchange_threshold):
    def default_action(history):
        if history == ():
            p_ready = belief_prob_agent_at_target(root_belief, rid)
            if p_ready >= exchange_threshold:
                return EXCHANGE
            return ADVANCE

        _last_action, last_obs = history[-1]
        if last_obs == AT_TARGET and _last_action != EXCHANGE:
            return EXCHANGE
        return ADVANCE

    return default_action


def make_medevac_belief_legal_actions_fn(
    rid,
    exchange_threshold,
    full_action_space: bool = False,
):
    def legal_actions(belief, depth):
        if full_action_space:
            return [WAIT, ADVANCE, EXCHANGE]

        p_ready = belief_prob_agent_at_target(belief, rid)
        if p_ready >= exchange_threshold:
            return [EXCHANGE, WAIT]
        return [ADVANCE]

    return legal_actions


def make_medevac_belief_default_action_fn(rid, exchange_threshold):
    def default_action(belief):
        p_ready = belief_prob_agent_at_target(belief, rid)
        if p_ready >= exchange_threshold:
            return EXCHANGE
        return ADVANCE

    return default_action


def belief_obs_root_action_masses(planner: BeliefObsDecMCTS) -> Dict[str, float]:
    masses = {
        WAIT: 0.0,
        ADVANCE: 0.0,
        EXCHANGE: 0.0,
    }
    root_key = (0, rounded_belief_key(planner.root_belief))

    for policy_key, p in planner.q.items():
        policy = BeliefPolicyTree.from_key(
            policy_key,
            planner.default_action_fn,
            planner.belief_lookup,
        )
        try:
            root_a = policy.action(planner.root_belief, root_key)
            masses[root_a] += p
        except Exception:
            pass

    return {
        ACTION_NAME[a]: round(masses[a], 4)
        for a in [WAIT, ADVANCE, EXCHANGE]
    }

## ABLATION
# def make_medevac_default_action_fn(legal_actions_fn, rng, root):
#     def default_action(history):
#         actions = list(legal_actions_fn(history, len(history)))
#         return rng.choice(actions)
#     return default_action

# def make_random_default_action_fn(legal_actions_fn, rng: random.Random):
#     cache = {}

#     def default_action(history):
#         if history not in cache:
#             actions = list(legal_actions_fn(history, len(history)))
#             cache[history] = rng.choice(actions)
#         return cache[history]

#     return default_action

    ##

def run_obs_planning_step(
    beliefs: Dict[int, List[float]],
    remaining_horizon: int,
    model: MedevacModel,
    args,
    seed: int,
):
    planners: Dict[int, ObsDecMCTS] = {}
    adapter = MedevacObsModelAdapter(model)
    default_action_fns_by_robot = {
        rid: make_medevac_default_action_fn(
            root_belief=beliefs[rid],
            rid=rid,
            exchange_threshold=args.exchange_threshold,
        )
        for rid in range(N_AGENTS)
    }

    for rid in range(N_AGENTS):
        planners[rid] = ObsDecMCTS(
            robot_id=rid,
            robot_ids=[0, 1],
            root_belief=beliefs[rid],
            model=adapter,
            legal_actions_fn=make_medevac_legal_actions_fn(
                root_belief=beliefs[rid],
                rid=rid,
                exchange_threshold=args.exchange_threshold,
                full_action_space=args.full_action_space,
            ),
            # default_action_fn=medevac_default_action_from_history,
            # legal_actions_fn=make_medevac_legal_actions_fn(
            #     root_belief=beliefs[rid],
            #     rid=rid,
            #     exchange_threshold=args.exchange_threshold,
            # ),
            # default_action_fn=make_medevac_default_action_fn(
            #     root_belief=beliefs[rid],
            #     rid=rid,
            #     exchange_threshold=args.exchange_threshold,
            # ),

            ## ABLATION
            # default_action_fn=make_random_default_action_fn(
            #     medevac_legal_actions_from_history,
            #     default_rng,
            # ),      
            ##

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
                f"root_actions={[ACTION_NAME[a] for a in planner.root.actions.keys()]} "
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

    # Only computed under --debug-obs: each extra source re-extracts actions for
    # every planner. ObsDecMCTS implements a subset of the belief planner's
    # extraction rules, so ask it only for the ones it supports.
    debug_actions = (
        {
            source: team.best_actions(source=source)
            for source in ObsDecMCTS.SUPPORTED_ACTION_SOURCES
        }
        if getattr(args, "debug_obs", False)
        else {}
    )

    actions = {}
    for rid in range(N_AGENTS):
        raw_a = int(raw_actions[rid])

        if getattr(args, "guard_actions", False):
            actions[rid] = maybe_guard_exchange(
                belief=beliefs[rid],
                rid=rid,
                proposed_action=raw_a,
                exchange_threshold=args.exchange_threshold,
            )
        else:
            actions[rid] = raw_a

    root_masses = {
        rid: obs_root_action_masses(planner)
        for rid, planner in planners.items()
    }

    # return actions, raw_actions, policies, team.entropies(), root_masses
    return actions, raw_actions, policies, team.entropies(), root_masses, debug_actions


def run_belief_obs_planning_step(
    beliefs: Dict[int, List[float]],
    remaining_horizon: int,
    model: MedevacModel,
    args,
    seed: int,
):
    planners: Dict[int, BeliefObsDecMCTS] = {}
    adapter = MedevacObsModelAdapter(model)
    default_action_fns_by_robot = {
        rid: make_medevac_belief_default_action_fn(
            rid=rid,
            exchange_threshold=args.exchange_threshold,
        )
        for rid in range(N_AGENTS)
    }

    for rid in range(N_AGENTS):
        planners[rid] = BeliefObsDecMCTS(
            robot_id=rid,
            robot_ids=[0, 1],
            root_belief=beliefs[rid],
            root_beliefs_by_robot=beliefs,
            model=adapter,
            legal_actions_fn=make_medevac_belief_legal_actions_fn(
                rid=rid,
                exchange_threshold=args.exchange_threshold,
                full_action_space=args.full_action_space,
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

    # Only computed under --debug-obs: each extra source re-extracts actions for
    # every planner.
    debug_actions = (
        {
            source: team.best_actions(beliefs=beliefs, source=source)
            for source in ("tree", "disc_tree", "policy", "visits")
        }
        if getattr(args, "debug_obs", False)
        else {}
    )

    actions = {}
    for rid in range(N_AGENTS):
        raw_a = int(raw_actions[rid])
        if getattr(args, "guard_actions", False):
            actions[rid] = maybe_guard_exchange(
                belief=beliefs[rid],
                rid=rid,
                proposed_action=raw_a,
                exchange_threshold=args.exchange_threshold,
            )
        else:
            actions[rid] = raw_a

    root_masses = {
        rid: belief_obs_root_action_masses(planner)
        for rid, planner in planners.items()
    }

    return actions, raw_actions, policies, team.entropies(), root_masses, debug_actions


# =============================================================================
# Episode simulation
# =============================================================================

def simulate_online_episode(
    model: MedevacModel,
    args,
    episode_seed: int,
    verbose: bool = False,
) -> float:
    rng = random.Random(episode_seed)

    true_state = model.sample_initial_state(rng)

    beliefs = {
        0: list(model.init_belief),
        1: list(model.init_belief),
    }
    common_belief = list(model.init_belief)
    pending_history: List[Tuple[int, int]] = []

    def flush_pending_common_belief() -> None:
        nonlocal common_belief, pending_history

        for hist_joint_a, hist_joint_o in pending_history:
            common_belief = update_joint_belief(
                common_belief,
                hist_joint_a,
                hist_joint_o,
                model,
            )

        beliefs[0] = list(common_belief)
        beliefs[1] = list(common_belief)
        pending_history = []

    total_reward = 0.0
    discount = 1.0

    episode_counts = {
        "pickup_exchange": 0,
        "dropoff_exchange": 0,
        "mismatch_exchange": 0,
    }

    if verbose:
        print(f"initial true_state={id_to_state(true_state)}")
        # print(f"initial beliefs: {belief_support_summary(beliefs[0])}")
        
    for t in range(args.horizon):
        remaining = args.horizon - t

        if verbose:
            print(f"DEBUG entering t={t}, remaining={remaining}, planner={args.planner}")

        if args.planner == "belief-obs":
            actions, raw_actions, policies, entropies, root_masses, debug_actions = run_belief_obs_planning_step(
                beliefs=beliefs,
                remaining_horizon=remaining,
                model=model,
                args=args,
                seed=episode_seed + 7919 * t,
            )
        else:
            actions, raw_actions, policies, entropies, root_masses, debug_actions = run_obs_planning_step(
                beliefs=beliefs,
                remaining_horizon=remaining,
                model=model,
                args=args,
                seed=episode_seed + 7919 * t,
            )

        a0, a1 = int(actions[0]), int(actions[1])
        joint_a = model.joint_action(a0, a1)

        ## REMOVE

        hx, hy, sx, sy, carry = id_to_state(true_state)
        # if a0 == EXCHANGE or a1 == EXCHANGE:
        #     if carry == 0 and at_patient(hx, hy) and at_patient(sx, sy):
        #         episode_counts["pickup_exchange"] += 1
        #     elif carry == 1 and at_hospital(hx, hy) and at_hospital(sx, sy):
        #         episode_counts["dropoff_exchange"] += 1
        #     else:
        #         episode_counts["mismatch_exchange"] += 1

        if a0 == EXCHANGE or a1 == EXCHANGE:
            if (
                carry == 0
                and at_patient(hx, hy)
                and at_patient(sx, sy)
                and a0 == EXCHANGE
                and a1 == EXCHANGE
            ):
                episode_counts["pickup_exchange"] += 1
            elif (
                carry == 1
                and at_hospital(hx, hy)
                and at_hospital(sx, sy)
                and a0 == EXCHANGE
                and a1 == EXCHANGE
            ):
                episode_counts["dropoff_exchange"] += 1
            else:
                episode_counts["mismatch_exchange"] += 1

        reward = model.reward(true_state, joint_a)
        total_reward += discount * reward
        discount *= args.gamma

        next_state = model.sample_next_state(true_state, joint_a, rng)
        joint_o = model.obs_from_next_state(next_state, joint_a)
        o0, o1 = model.split_obs(joint_o)

        beliefs[0] = update_local_belief(beliefs[0], joint_a, o0, 0, model)
        beliefs[1] = update_local_belief(beliefs[1], joint_a, o1, 1, model)

        pending_history.append((joint_a, joint_o))

        if args.env_comm_mode == "periodic":
            if args.env_comm_period > 0 and (t + 1) % args.env_comm_period == 0:
                flush_pending_common_belief()
        elif args.env_comm_mode == "collocated":
            next_hx, next_hy, next_sx, next_sy, _ = id_to_state(next_state)
            if next_hx == next_sx and next_hy == next_sy:
                flush_pending_common_belief()
        else:
            raise ValueError(f"Unknown env_comm_mode: {args.env_comm_mode}")

        if verbose:
            print(
                f"t={t:02d} rem={remaining:02d} "
                f"a=({ACTION_NAME[a0]},{ACTION_NAME[a1]}) "
                f"raw=({ACTION_NAME[int(raw_actions[0])]},{ACTION_NAME[int(raw_actions[1])]}) "
                f"r={reward:7.3f} "
                f"obs=({OBS_NAME[o0]},{OBS_NAME[o1]}) "
                f"next={id_to_state(next_state)} "
                f"H={{{0}: {entropies.get(0, 0.0):.3f}, {1}: {entropies.get(1, 0.0):.3f}}}"
            )
            if t >= 4:
                hx, hy, sx, sy, carry = id_to_state(true_state)
                print(
                    "    DEBUG late state",
                    {
                        "t": t,
                        "state": id_to_state(true_state),
                        "carry": carry,
                        "actions": (ACTION_NAME[a0], ACTION_NAME[a1]),
                        "p_ready0": round(belief_prob_agent_at_target(beliefs[0], 0), 3),
                        "p_ready1": round(belief_prob_agent_at_target(beliefs[1], 1), 3),
                    },
                )            
            print(f"    belief0={belief_support_summary(beliefs[0])}")
            print(f"    belief1={belief_support_summary(beliefs[1])}")
            print(f"    root_mass={root_masses}")
            if args.planner == "belief-obs":
                key0 = (0, rounded_belief_key(beliefs[0]))
                key1 = (0, rounded_belief_key(beliefs[1]))
                print(f"    policy0_root_from_q={ACTION_NAME[policies[0].action(beliefs[0], key0)]}")
                print(f"    policy1_root_from_q={ACTION_NAME[policies[1].action(beliefs[1], key1)]}")
            else:
                print(f"    policy0_root_from_q={ACTION_NAME[policies[0].action(())]}")
                print(f"    policy1_root_from_q={ACTION_NAME[policies[1].action(())]}")
            print(f"    executed0_from_{args.action_source}={ACTION_NAME[int(raw_actions[0])]}")
            print(f"    executed1_from_{args.action_source}={ACTION_NAME[int(raw_actions[1])]}")

            if debug_actions:
                print(
                    "    action_sources="
                    + str({
                        name: (
                            ACTION_NAME[int(acts[0])],
                            ACTION_NAME[int(acts[1])],
                        )
                        for name, acts in debug_actions.items()
                    })
                )

        true_state = next_state

    if verbose:
        print("episode_counts:", episode_counts)

    return total_reward


# =============================================================================
# Stats / workers
# =============================================================================

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
    model = MedevacModel()

    ret = simulate_online_episode(
        model=model,
        args=args,
        episode_seed=args.seed + 104729 * ep,
        verbose=False,
    )

    return ep, ret


# =============================================================================
# Main
# =============================================================================

def run_initial_planning_timing(model: MedevacModel, args) -> None:
    """Time a single full-horizon ObsDecMCTS planning step (paper 'time' metric)."""
    beliefs = {
        0: list(model.init_belief),
        1: list(model.init_belief),
    }
    t0 = time.time()
    run_obs_planning_step(
        beliefs=beliefs,
        remaining_horizon=args.horizon,
        model=model,
        args=args,
        seed=args.seed,
    )
    elapsed = time.time() - t0
    print("Initial MEDEVAC planning timing")
    print(f"horizon={args.horizon}")
    print(f"planning_time={elapsed:.6f}s")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--horizon", type=int, default=7)
    parser.add_argument("--episodes", type=int, default=50)

    parser.add_argument("--outer-iters", type=int, default=30)
    parser.add_argument("--tau", type=int, default=40)
    parser.add_argument("--num-seq", type=int, default=30)
    parser.add_argument("--num-samples", type=int, default=30)
    parser.add_argument("--comm-period", type=int, default=1)
    parser.add_argument(
        "--env-comm-mode",
        choices=["periodic", "collocated"],
        default="periodic",
        help="Execution-time belief sync rule: fixed period or only when agents are collocated.",
    )
    parser.add_argument(
        "--env-comm-period",
        type=int,
        default=0,
        help="Execution-time observation sharing period for --env-comm-mode periodic. Use 0 for no execution sync.",
    )

    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--cp", type=float, default=0.5)
    parser.add_argument("--beta-init", type=float, default=2.0)
    parser.add_argument("--beta-decay", type=float, default=0.995)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--progressive-widening-visits", type=int, default=0)

    parser.add_argument("--exchange-threshold", type=float, default=0.80)
    parser.add_argument("--guard-actions", action="store_true")
    parser.add_argument(
        "--full-action-space",
        action="store_true",
        help="Allow WAIT, ADVANCE, and EXCHANGE at every MEDEVAC history.",
    )
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--verbose-first", action="store_true")
    parser.add_argument("--debug-obs", action="store_true")
    parser.add_argument(
        "--planner",
        choices=["obs", "belief-obs"],
        default="obs",
        help="Planner backend: history-conditioned ObsDecMCTS or belief-node ObsDecMCTS.",
    )
    parser.add_argument(
        "--belief-share-nodes",
        action="store_true",
        help="Reuse belief nodes at the same depth when --planner belief-obs.",
    )
    # REMOVE
    parser.add_argument(
        "--action-source",
        choices=["tree", "disc_tree", "policy", "visits", "policy_value", "policy_marginal"],
        default="policy_marginal",
    )

    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of parallel worker processes. Use -1 for all CPU cores.",
    )

    parser.add_argument("--timing-only", action="store_true",
                        help="Time a single full-horizon planning step and exit.")

    args = parser.parse_args()
    model = MedevacModel()

    if args.timing_only:
        run_initial_planning_timing(model, args)
        return

    print("Online ObsDecMCTS MEDEVAC benchmark")
    print(f"horizon={args.horizon}, episodes={args.episodes}")
    print(
        f"planning: outer_iters={args.outer_iters}, tau={args.tau}, "
        f"num_seq={args.num_seq}, num_samples={args.num_samples}"
    )
    print(
        f"exchange_threshold={args.exchange_threshold}, "
        f"env_comm_mode={args.env_comm_mode}, "
        f"env_comm_period={args.env_comm_period}, "
        f"full_action_space={args.full_action_space}, "
        f"guard_actions={args.guard_actions}, "
        f"planner={args.planner}, "
        f"seed={args.seed}"
    )
    if args.env_comm_mode == "collocated" or args.env_comm_period != 0:
        print(
            "WARNING: execution-time communication shares joint observations; "
            "this is not a pure decentralized RS-MAA* comparison."
        )
    if abs(args.gamma - 1.0) > 1e-12:
        print(
            "WARNING: gamma != 1.0; paper reference values are undiscounted "
            "finite-horizon expected values."
        )
    print()

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

    lower_ref = REFERENCE_DECENTRALIZED_RSMAA.get(args.horizon)
    upper_ref = REFERENCE_CENTRALIZED_RSSDA.get(args.horizon)
    if lower_ref is not None or upper_ref is not None:
        print("Reference values:")
        if lower_ref is not None:
            print(f"  Decentralized RS-MAA*: {lower_ref:.5f}")
        if upper_ref is not None:
            print(f"  Centralized RSSDA* upper bound: {upper_ref:.5f}")

            if mean > upper_ref + 1e-9:
                print(
                    "WARNING: sample mean exceeds the centralized RSSDA* upper bound. "
                    "Check communication settings, gamma, reward timing, and Monte Carlo variance."
                )
            if mean - ci > upper_ref + 1e-9:
                print(
                    "WARNING: the 95% CI lower edge exceeds the upper bound; "
                    "this strongly suggests a benchmark mismatch."
                )


if __name__ == "__main__":
    main()
