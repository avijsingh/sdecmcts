"""
tiger_online_benchmark.py
-------------------------

for t in 0..H-1:
    build/run Dec-MCTS from each agent's current belief
    execute only the first action from each selected sequence
    sample transition + observation
    update each agent's belief using its local observation
    replan

Thus each individual Dec-MCTS call is open-loop, but the executed controller is
observation-conditioned through belief updates and replanning.

Action encoding matches the RSSDA Tiger script:
    OPEN_LEFT  = 0
    OPEN_RIGHT = 1
    LISTEN     = 2

MOD='C' observations:
    - both listen: each local observation is 0.75 accurate
    - exactly one listens: listener is 0.75 accurate, non-listener is uniform
    - opening: tiger resets uniformly; observations are otherwise uniform except
      the single-listener correction above
"""

from __future__ import annotations

import os 
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import math
import random
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from decmcts import DecMCTS, DecMCTSTeam
import decmcts

from obs_decmcts import ObsDecMCTS, ObsDecMCTSTeam, StepResult, History
import obs_decmcts
from belief_obs_decmcts import BeliefObsDecMCTS, BeliefObsDecMCTSTeam


# =============================================================================
# Constants
# =============================================================================

TIGER_LEFT = 0
TIGER_RIGHT = 1

OPEN_LEFT = 0
OPEN_RIGHT = 1
LISTEN = 2

N_AGENTS = 2
N_STATES = 2
ACT_PER_AGENT = 3
OBS_PER_AGENT = 2
N_ACTS = ACT_PER_AGENT ** N_AGENTS
N_OBS = OBS_PER_AGENT ** N_AGENTS

ACTION_NAME = {
    OPEN_LEFT: "OL",
    OPEN_RIGHT: "OR",
    LISTEN: "L",
}

OBS_NAME = {
    TIGER_LEFT: "hear-left",
    TIGER_RIGHT: "hear-right",
}

# =============================================================================
# Tiger Model
# =============================================================================

class TigerModel:
    def __init__(self):
        self.T = [0.0] * (N_ACTS * N_STATES * N_STATES)
        self.O = [0.0] * (N_ACTS * N_STATES * N_OBS)
        self.R = [0.0] * (N_ACTS * N_STATES)
        self.init_belief = [0.5, 0.5]
        self._build()

    @staticmethod
    def joint_action(a0: int, a1: int) -> int:
        return a0 + ACT_PER_AGENT * a1

    @staticmethod
    def split_action(a: int) -> Tuple[int, int]:
        return a % ACT_PER_AGENT, a // ACT_PER_AGENT

    @staticmethod
    def joint_obs(o0: int, o1: int) -> int:
        return o0 + OBS_PER_AGENT * o1

    @staticmethod
    def split_obs(o: int) -> Tuple[int, int]:
        return o % OBS_PER_AGENT, o // OBS_PER_AGENT

    def _t_idx(self, a: int, s: int, sp: int) -> int:
        return a * N_STATES * N_STATES + s * N_STATES + sp

    def _o_idx(self, a: int, sp: int, o: int) -> int:
        return a * N_STATES * N_OBS + sp * N_OBS + o

    def _r_idx(self, a: int, s: int) -> int:
        return a * N_STATES + s

    def _set_t(self, a: int, s: int, sp: int, p: float) -> None:
        self.T[self._t_idx(a, s, sp)] = p

    def _set_o(self, a: int, sp: int, o: int, p: float) -> None:
        self.O[self._o_idx(a, sp, o)] = p

    def _set_r(self, a: int, s: int, r: float) -> None:
        self.R[self._r_idx(a, s)] = r

    def _build(self) -> None:
        both_listen = self.joint_action(LISTEN, LISTEN)

        for a in range(N_ACTS):
            a0, a1 = self.split_action(a)

            if a == both_listen:
                for s in range(N_STATES):
                    self._set_r(a, s, -2.0)
                    for sp in range(N_STATES):
                        self._set_t(a, s, sp, 1.0 if sp == s else 0.0)

                # MOD='C': both-listen local observations are 0.85/0.15.
                for sp in range(N_STATES):
                    for o in range(N_OBS):
                        o0, o1 = self.split_obs(o)
                        p0 = 0.85 if o0 == sp else 0.15
                        p1 = 0.85 if o1 == sp else 0.15
                        self._set_o(a, sp, o, p0 * p1)
                continue

            # Any non-both-listen action: tiger resets uniformly.
            for s in range(N_STATES):
                r = (
                    -101 * ((a0 == s) * (a1 == LISTEN) +
                            (a1 == s) * (a0 == LISTEN))
                    -50 * ((a0 == s) * (a1 == s))
                    -100 * ((a0 != LISTEN) * (a0 != s) * (a1 == s) +
                            (a1 != LISTEN) * (a1 != s) * (a0 == s))
                    +9 * ((a0 != LISTEN) * (a0 != s) * (a1 == LISTEN) +
                          (a1 != LISTEN) * (a1 != s) * (a0 == LISTEN))
                    +20 * ((a0 != LISTEN) * (a0 != s) *
                           (a1 != LISTEN) * (a1 != s))
                )
                self._set_r(a, s, float(r))
                for sp in range(N_STATES):
                    self._set_t(a, s, sp, 0.5)

            # Default opening obs model: uniform joint observations.
            for sp in range(N_STATES):
                for o in range(N_OBS):
                    self._set_o(a, sp, o, 1.0 / N_OBS)

        # MOD='C': exactly one listener gets 0.85/0.15; non-listener gets uniform.
        for a in range(N_ACTS):
            a0, a1 = self.split_action(a)
            if not ((a0 == LISTEN) ^ (a1 == LISTEN)):
                continue

            for sp in range(N_STATES):
                for o in range(N_OBS):
                    o0, o1 = self.split_obs(o)
                    if a0 == LISTEN:
                        p_listen = 0.85 if o0 == sp else 0.15
                        self._set_o(a, sp, o, p_listen * 0.5)
                    else:
                        p_listen = 0.85 if o1 == sp else 0.15
                        self._set_o(a, sp, o, 0.5 * p_listen)

    def transition_prob(self, s: int, a: int, sp: int) -> float:
        return self.T[self._t_idx(a, s, sp)]

    def obs_prob(self, sp: int, a: int, o: int) -> float:
        return self.O[self._o_idx(a, sp, o)]

    def reward(self, s: int, a: int) -> float:
        return self.R[self._r_idx(a, s)]

    def sample_initial_state(self, rng: random.Random) -> int:
        return 0 if rng.random() < self.init_belief[0] else 1

    def sample_next_state(self, s: int, a: int, rng: random.Random) -> int:
        r = rng.random()
        cum = 0.0
        for sp in range(N_STATES):
            cum += self.transition_prob(s, a, sp)
            if r <= cum:
                return sp
        return N_STATES - 1

    def sample_joint_obs(self, sp: int, a: int, rng: random.Random) -> int:
        r = rng.random()
        cum = 0.0
        for o in range(N_OBS):
            cum += self.obs_prob(sp, a, o)
            if r <= cum:
                return o
        return N_OBS - 1

    def local_obs_prob(self, agent_id: int, sp: int, joint_a: int, local_o: int) -> float:
        """
        Marginal P(o_i | s', joint_a), summing over the other agent's obs.
        """
        total = 0.0
        for joint_o in range(N_OBS):
            o0, o1 = self.split_obs(joint_o)
            if (agent_id == 0 and o0 == local_o) or (agent_id == 1 and o1 == local_o):
                total += self.obs_prob(sp, joint_a, joint_o)
        return total

# =============================================================================
# Wrapper adapting tiger benchmark to observation-conditioned decmcts
# =============================================================================

class TigerObsModelAdapter:
    """
    Thin generative adapter expected by ObsDecMCTS.
    Wraps the existing TigerModel.
    """

    def __init__(self, model: TigerModel):
        self.model = model

    def sample_state_from_belief(self, belief, rng: random.Random) -> int:
        r = rng.random()
        return TIGER_LEFT if r < belief[TIGER_LEFT] else TIGER_RIGHT

    def joint_action_from_dict(self, actions: Dict[int, int]) -> int:
        return self.model.joint_action(actions[0], actions[1])

    def step(self, state: int, joint_action: int, rng: random.Random) -> StepResult:
        reward = self.model.reward(state, joint_action)
        next_state = self.model.sample_next_state(state, joint_action, rng)
        joint_obs = self.model.sample_joint_obs(next_state, joint_action, rng)
        return StepResult(
            next_state=next_state,
            joint_obs=joint_obs,
            reward=reward,
        )

    def split_obs(self, joint_obs: int) -> Tuple[int, int]:
        return self.model.split_obs(joint_obs)

    def update_belief(
        self,
        belief: Sequence[float],
        joint_action: int,
        local_obs: int,
        robot_id: int,
    ) -> List[float]:
        return update_local_belief(
            belief,
            joint_action,
            local_obs,
            robot_id,
            self.model,
        )

# =============================================================================
# Planning state and exact belief calculations
# =============================================================================

@dataclass(frozen=True)
class TigerPlanningState:
    depth: int
    max_horizon: int

    def get_legal_actions(self) -> List[int]:
        if self.is_terminal_state():
            return []
        return [OPEN_LEFT, OPEN_RIGHT, LISTEN]

    def take_action(self, action: int) -> "TigerPlanningState":
        return TigerPlanningState(self.depth + 1, self.max_horizon)

    def is_terminal_state(self) -> bool:
        return self.depth >= self.max_horizon


def normalize_belief(b: Sequence[float]) -> List[float]:
    total = float(sum(b))
    if total <= 0:
        return [0.5, 0.5]
    return [float(x) / total for x in b]


def predict_belief_open_loop(
    belief: Sequence[float],
    joint_a: int,
    model: TigerModel,
) -> List[float]:
    """
    Predict next belief without conditioning on observation:
        b'(s') = sum_s b(s) T(s' | s, a)
    Used inside open-loop sequence evaluation and heuristic rollouts.
    """
    out = [0.0, 0.0]
    for s in range(N_STATES):
        for sp in range(N_STATES):
            out[sp] += belief[s] * model.transition_prob(s, joint_a, sp)
    return normalize_belief(out)


def update_local_belief(
    belief: Sequence[float],
    joint_a: int,
    local_obs: int,
    agent_id: int,
    model: TigerModel,
) -> List[float]:
    """
    Bayes update using each agent's local observation:
        b'(s') proportional to P(o_i | s', a) sum_s T(s' | s,a)b(s)
    """
    pred = predict_belief_open_loop(belief, joint_a, model)
    post = [
        pred[sp] * model.local_obs_prob(agent_id, sp, joint_a, local_obs)
        for sp in range(N_STATES)
    ]
    return normalize_belief(post)


def expected_open_loop_return(
    belief: Sequence[float],
    joint_sequences: Dict[int, Sequence[int]],
    model: TigerModel,
    horizon: int,
    null_action: int = LISTEN,
) -> float:
    """
    Exact expected return for fixed open-loop joint action sequences from belief.

    Future observations are not used because the sequence is fixed. Belief evolves
    only through the transition model.
    """
    b = list(belief)
    total = 0.0

    seq0 = list(joint_sequences.get(0, []))
    seq1 = list(joint_sequences.get(1, []))

    for t in range(horizon):
        a0 = seq0[t] if t < len(seq0) else null_action
        a1 = seq1[t] if t < len(seq1) else null_action
        joint_a = model.joint_action(a0, a1)

        total += sum(b[s] * model.reward(s, joint_a) for s in range(N_STATES))
        b = predict_belief_open_loop(b, joint_a, model)

    return total


def make_local_utility_fn(
    agent_id: int,
    belief: Sequence[float],
    model: TigerModel,
    horizon: int,
    mode: str = "difference",
) -> Callable[[Dict[int, List[int]]], float]:
    """
    Local utility for Dec-MCTS.

    mode='difference':
        f^r(x) = g(x^r, x^{-r}) - g(null^r, x^{-r})
        This matches the Dec-MCTS paper's local utility idea.

    mode='global':
        f^r(x) = g(x)
        Useful for debugging if difference rewards create poor local equilibria.
    """
    if mode not in {"difference", "global"}:
        raise ValueError("mode must be 'difference' or 'global'.")

    b0 = list(belief)
    value_cache: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], float] = {}

    def g(joint: Dict[int, List[int]]) -> float:
        key = (
            tuple(joint.get(0, ())),
            tuple(joint.get(1, ())),
        )
        cached = value_cache.get(key)
        if cached is not None:
            return cached
        value = expected_open_loop_return(b0, joint, model, horizon)
        value_cache[key] = value
        return value

    def local_utility(joint: Dict[int, List[int]]) -> float:
        if mode == "global":
            return g(joint)

        null_joint = {rid: list(seq) for rid, seq in joint.items()}
        null_joint[agent_id] = [LISTEN] * horizon
        return g(joint) - g(null_joint)

    return local_utility

    def update_joint_belief(belief, joint_a, joint_obs, model):
        pred = predict_belief_open_loop(belief, joint_a, model)
        post = [
            pred[sp] * model.obs_prob(sp, joint_a, joint_obs)
            for sp in range(N_STATES)
        ]
        return normalize_belief(post)
    
   

# =============================================================================
# Heuristic
# =============================================================================

def heuristic_action_from_belief(
    belief: Sequence[float],
    open_threshold: float = 0.85,
) -> int:
    """
    Non-clairvoyant belief heuristic.

    If tiger is likely left, open right.
    If tiger is likely right, open left.
    Otherwise listen.
    """
    p_left = belief[TIGER_LEFT]
    p_right = belief[TIGER_RIGHT]

    if p_left >= open_threshold:
        return OPEN_RIGHT
    if p_right >= open_threshold:
        return OPEN_LEFT
    return LISTEN


def make_tiger_rollout_policy(
    belief: Sequence[float],
    model: TigerModel,
    horizon: int,
    open_threshold: float,
) -> Callable:
    """
    Create rollout_policy(planner, node, x_others) -> list[action].

    This heuristic never samples or uses the hidden true state. It only uses the
    planner's current belief and predicted open-loop transitions.
    """
    root_belief = list(belief)

    def rollout_policy(planner: DecMCTS, node, x_others: Dict[int, List[int]]) -> List[int]:
        b = list(root_belief)
        own_prefix = list(node.action_sequence)
        other_id = next(r for r in planner.robot_ids if r != planner.robot_id)
        other_seq = list(x_others.get(other_id, []))

        # Roll belief forward through the already chosen prefix.
        for t, my_a in enumerate(own_prefix):
            other_a = other_seq[t] if t < len(other_seq) else LISTEN
            if planner.robot_id == 0:
                joint_a = model.joint_action(my_a, other_a)
            else:
                joint_a = model.joint_action(other_a, my_a)
            b = predict_belief_open_loop(b, joint_a, model)

        # Complete the sequence with the belief heuristic.
        actions: List[int] = []
        for t in range(len(own_prefix), horizon):
            my_a = heuristic_action_from_belief(b, open_threshold=open_threshold)
            actions.append(my_a)

            other_a = other_seq[t] if t < len(other_seq) else LISTEN
            if planner.robot_id == 0:
                joint_a = model.joint_action(my_a, other_a)
            else:
                joint_a = model.joint_action(other_a, my_a)
            b = predict_belief_open_loop(b, joint_a, model)

        return actions

    return rollout_policy

def tiger_legal_actions_from_history(history: History, depth: int) -> List[int]:
    return [OPEN_LEFT, OPEN_RIGHT, LISTEN]

def tiger_default_action_from_history(history: History) -> int:
    # Safe default for unexplored policy-tree histories.
    return LISTEN

# # REMOVE/CHECK
def tiger_legal_actions_from_history(history: History, depth: int) -> List[int]:
    # Put LISTEN first for Tiger. This is safer for low-budget policy support.
    return [LISTEN, OPEN_LEFT, OPEN_RIGHT]


def tiger_belief_from_local_history(history: History) -> List[float]:
    """
    Approximate local belief induced by an agent's own listen observations.

    This is only for rollout/default policy behavior inside ObsDecMCTS.
    The real online episode still uses exact Bayes updates on the actual belief.
    """
    b = [0.5, 0.5]

    for action, obs in history:
        if action == LISTEN:
            b = normalize_belief([
                b[TIGER_LEFT] * (0.85 if obs == TIGER_LEFT else 0.15),
                b[TIGER_RIGHT] * (0.85 if obs == TIGER_RIGHT else 0.15),
            ])
        elif action in (OPEN_LEFT, OPEN_RIGHT):
            # Opening resets the tiger uniformly.
            b = [0.5, 0.5]

    return b


def make_tiger_legal_actions_fn(
    root_belief: Sequence[float],
    open_threshold: float,
):
    def legal_actions(history: History, depth: int) -> List[int]:
        if history == ():
            root_action = heuristic_action_from_belief(
                root_belief,
                open_threshold=open_threshold,
            )

            if root_action == LISTEN:
                return [LISTEN, OPEN_LEFT, OPEN_RIGHT]

            # If root belief is confident, still allow LISTEN as fallback,
            # but do not include the obviously wrong open.
            return [root_action, LISTEN]

        return [LISTEN, OPEN_LEFT, OPEN_RIGHT]

    return legal_actions


def make_tiger_default_action_fn(
    root_belief: Sequence[float],
    open_threshold: float,
):
    def default_action(history: History) -> int:
        if history == ():
            return heuristic_action_from_belief(
                root_belief,
                open_threshold=open_threshold,
            )

        local_belief = tiger_belief_from_local_history(history)
        return heuristic_action_from_belief(
            local_belief,
            open_threshold=open_threshold,
        )

    return default_action


def tiger_belief_legal_actions(belief: Sequence[float], depth: int) -> List[int]:
    return [LISTEN, OPEN_LEFT, OPEN_RIGHT]


def tiger_belief_default_action(belief: Sequence[float]) -> int:
    return LISTEN


# =============================================================================
# Online Dec-MCTS episode simulation
# =============================================================================

def run_planning_step(
    beliefs: Dict[int, List[float]],
    remaining_horizon: int,
    model: TigerModel,
    args,
    seed: int,
) -> Tuple[Dict[int, int], Dict[int, List[int]], Dict[int, float]]:
    """
    Run one online planning step from the agents' current beliefs and return
    first actions.
    """
    planners = {}

    for rid in range(N_AGENTS):
        init_state = TigerPlanningState(depth=0, max_horizon=remaining_horizon)
        local_fn = make_local_utility_fn(
            agent_id=rid,
            belief=beliefs[rid],
            model=model,
            horizon=remaining_horizon,
            mode=args.local_utility,
        )

        rollout_policy = make_tiger_rollout_policy(
            belief=beliefs[rid],
            model=model,
            horizon=remaining_horizon,
            open_threshold=args.open_threshold,
        )

        planners[rid] = DecMCTS(
            robot_id=rid,
            robot_ids=[0, 1],
            init_state=init_state,
            local_utility_fn=local_fn,
            rollout_policy=rollout_policy,
            default_sequence_fn=lambda _rid, H=remaining_horizon: [LISTEN] * H,
            gamma=args.gamma,
            cp=args.cp,
            rollout_depth=remaining_horizon,
            tau=args.tau,
            num_seq=args.num_seq,
            num_samples=args.num_samples,
            beta_init=args.beta_init,
            beta_decay=args.beta_decay,
            alpha=args.alpha,
            seed=seed + 1009 * rid,
        )

    team = DecMCTSTeam(planners)
    team.iterate_and_communicate(n_outer=args.outer_iters, comm_period=args.comm_period)

    raw_actions = {}
    actions = {}
    sequences = team.best_sequences()

    for rid in range(N_AGENTS):
        a = planners[rid].best_action()
        raw_a = LISTEN if a is None else int(a)
        raw_actions[rid] = raw_a

        if getattr(args, "guard_actions", False):
            actions[rid] = guard_tiger_action(
                belief=beliefs[rid],
                proposed_action=raw_a,
                open_threshold=args.open_threshold,
                force_open_when_confident=getattr(args, "force_open_when_confident", False),
            )
        else:
            actions[rid] = raw_a

    root_masses = {
        rid: root_action_masses(planner)
        for rid, planner in planners.items()
    }

    return actions, raw_actions, sequences, team.entropies(), root_masses

# def run_obs_planning_step(
#     beliefs: Dict[int, List[float]],
#     remaining_horizon: int,
#     model: TigerModel,
#     args,
#     seed: int,
# ):
#     """
#     Observation-conditioned Dec-MCTS planning step.

#     Unlike run_planning_step(), this planner searches over partial policy trees:
#         local action-observation history -> action

#     The returned action is the action at root local history ().
#     """
#     planners = {}
#     adapter = TigerObsModelAdapter(model)

#     for rid in range(N_AGENTS):
#         planners[rid] = ObsDecMCTS(
#             robot_id=rid,
#             robot_ids=[0, 1],
#             root_belief=beliefs[rid],
#             model=adapter,
#             legal_actions_fn=tiger_legal_actions_from_history,
#             default_action_fn=tiger_default_action_from_history,
#             gamma=args.gamma,
#             cp=args.cp,
#             horizon=remaining_horizon,
#             tau=args.tau,
#             num_policies=args.num_seq,
#             num_samples=args.num_samples,
#             beta_init=args.beta_init,
#             beta_decay=args.beta_decay,
#             alpha=args.alpha,
#             seed=seed + 1009 * rid,
#         )
    
#     # REMOVE
#     for rid, planner in planners.items():
#         print("DEBUG obs root edges", rid, ObsDecMCTS.obs_root_edge_stats(planner))

#     team = ObsDecMCTSTeam(planners)
#     team.iterate_and_communicate(
#         n_outer=args.outer_iters,
#         comm_period=args.comm_period,
#     )

#     raw_actions = {}
#     actions = {}
#     policies = team.best_policies()

#     for rid in range(N_AGENTS):
#         raw_a = int(planners[rid].best_action(history=()))
#         raw_actions[rid] = raw_a

#         if getattr(args, "guard_actions", False):
#             actions[rid] = guard_tiger_action(
#                 belief=beliefs[rid],
#                 proposed_action=raw_a,
#                 open_threshold=args.open_threshold,
#                 force_open_when_confident=getattr(args, "force_open_when_confident", False),
#             )
#         else:
#             actions[rid] = raw_a

#     root_masses = {
#         rid: obs_root_action_masses(planner)
#         for rid, planner in planners.items()
#     }

#     return actions, raw_actions, policies, team.entropies(), root_masses
def obs_root_edge_stats(planner):
    out = {}
    for a, edge in planner.root.actions.items():
        out[ACTION_NAME[a]] = {
            "visits": edge.visits,
            "q": round(edge.q(), 3),
            "disc_visits": round(edge.disc_visits, 3),
            "disc_q": round(edge.disc_q(), 3),
            "num_obs_children": len(edge.obs_children),
        }
    return out

def obs_root_teammate_debug(planner):
    teammate = {
        ACTION_NAME[a]: c
        for a, c in planner.debug_root_teammate_actions.items()
    }

    joint = {
        f"{ACTION_NAME[a0]},{ACTION_NAME[a1]}": c
        for (a0, a1), c in planner.debug_root_joint_actions.items()
    }

    return {
        "teammate_actions": teammate,
        "joint_actions_from_planner_perspective": joint,
    }

def run_obs_planning_step(
    beliefs: Dict[int, List[float]],
    remaining_horizon: int,
    model: TigerModel,
    args,
    seed: int,
):
    """
    Observation-conditioned Dec-MCTS planning step.

    This planner searches over partial policy trees:
        local action-observation history -> action

    The returned action is the action at root local history ().
    """
    planners = {}
    adapter = TigerObsModelAdapter(model)

    for rid in range(N_AGENTS):
        planners[rid] = ObsDecMCTS(
            robot_id=rid,
            robot_ids=[0, 1],
            root_belief=beliefs[rid],
            model=adapter,
            legal_actions_fn=make_tiger_legal_actions_fn(
                root_belief=beliefs[rid],
                open_threshold=args.open_threshold,
            ),
            default_action_fn=make_tiger_default_action_fn(
                root_belief=beliefs[rid],
                open_threshold=args.open_threshold,
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
            # TESTING: shared seeds can make both decentralized planners sample
            # highly correlated supports. Restore per-robot offsets while
            # keeping episode-level determinism.
            #
            # seed=seed,
            seed=seed + 1009 * rid,
        )

    team = ObsDecMCTSTeam(planners)

    team.iterate_and_communicate(
        n_outer=args.outer_iters,
        comm_period=args.comm_period,
    )

    # DEBUG: must be after iterate_and_communicate.
    if getattr(args, "debug_obs", False):
        for rid, planner in planners.items():
            print(
                f"DEBUG obs planner {rid}: "
                f"root_actions={list(ACTION_NAME[a] for a in planner.root.actions.keys())} "
                f"root_visits={planner.root.visits} "
                f"num_X_hat={len(planner.X_hat)} "
                f"num_q={len(planner.q)} "
                f"entropy={team.entropies().get(rid, 0.0):.3f} "
                f"min_reward={planner.min_reward:.3f} "
                f"max_reward={planner.max_reward:.3f}",
                flush=True,
            )

            print(
                f"DEBUG obs root edges {rid}: "
                f"{obs_root_edge_stats(planner)}",
                flush=True,
            )

            print(
                f"DEBUG obs teammate model {rid}: "
                f"{obs_root_teammate_debug(planner)}",
                flush=True,
            )

            print(
                f"DEBUG obs policy support {rid}: "
                f"{obs_policy_support_debug(planner)}",
                flush=True,
            )

    raw_actions = {}
    actions = {}
    policies = team.best_policies()

    for rid in range(N_AGENTS):
        raw_a = int(planners[rid].best_action(history=(), source=args.action_source))
        raw_actions[rid] = raw_a

        if getattr(args, "guard_actions", False):
            actions[rid] = guard_tiger_action(
                belief=beliefs[rid],
                proposed_action=raw_a,
                open_threshold=args.open_threshold,
                force_open_when_confident=getattr(args, "force_open_when_confident", False),
            )
        else:
            actions[rid] = raw_a

    root_masses = {
        rid: obs_root_action_masses(planner)
        for rid, planner in planners.items()
    }

    return actions, raw_actions, policies, team.entropies(), root_masses


def run_belief_obs_planning_step(
    beliefs: Dict[int, List[float]],
    remaining_horizon: int,
    model: TigerModel,
    args,
    seed: int,
):
    planners = {}
    adapter = TigerObsModelAdapter(model)

    default_action_fns_by_robot = {
        rid: tiger_belief_default_action
        for rid in range(N_AGENTS)
    }

    for rid in range(N_AGENTS):
        planners[rid] = BeliefObsDecMCTS(
            robot_id=rid,
            robot_ids=[0, 1],
            root_belief=beliefs[rid],
            root_beliefs_by_robot=beliefs,
            model=adapter,
            legal_actions_fn=tiger_belief_legal_actions,
            default_action_fn=tiger_belief_default_action,
            default_action_fns_by_robot=default_action_fns_by_robot,
            share_belief_nodes=args.belief_share_nodes,
            gamma=args.gamma,
            cp=args.cp,
            horizon=remaining_horizon,
            tau=args.tau,
            num_policies=args.num_seq,
            num_samples=args.num_samples,
            score_samples=None if args.score_samples <= 0 else args.score_samples,
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

    raw_actions = team.best_actions(beliefs=beliefs, source=args.action_source)
    actions = {}
    for rid in range(N_AGENTS):
        raw_a = int(raw_actions[rid])
        if getattr(args, "guard_actions", False):
            actions[rid] = guard_tiger_action(
                belief=beliefs[rid],
                proposed_action=raw_a,
                open_threshold=args.open_threshold,
                force_open_when_confident=getattr(args, "force_open_when_confident", False),
            )
        else:
            actions[rid] = raw_a

    root_masses = {
        rid: belief_root_action_masses(planner)
        for rid, planner in planners.items()
    }

    return actions, raw_actions, team.best_policies(), team.entropies(), root_masses

def guard_tiger_action(
    belief,
    proposed_action,
    open_threshold,
    force_open_when_confident=False,
    ):
        """
        Safety guard for online Tiger execution.

        If belief is not confident, veto OPEN actions and LISTEN instead.
        If belief is confident and an open is proposed, force the safe-looking door.
        Optionally force opening whenever belief is confident.
        """
        p_left = belief[TIGER_LEFT]
        p_right = belief[TIGER_RIGHT]
        confidence = max(p_left, p_right)

        safe_open = OPEN_RIGHT if p_left >= p_right else OPEN_LEFT

        if confidence < open_threshold:
            if proposed_action in (OPEN_LEFT, OPEN_RIGHT):
                return LISTEN
            return proposed_action

        # Confident belief: if opening, make sure it is the safe-looking door.
        if proposed_action in (OPEN_LEFT, OPEN_RIGHT):
            return safe_open

        if force_open_when_confident:
            return safe_open

        return proposed_action



def update_joint_belief(belief, joint_a, joint_obs, model):
    """
    Bayes update using the full joint observation:
        b'(s') proportional to P(o_joint | s', a) * sum_s T(s' | s,a)b(s)
    """
    pred = predict_belief_open_loop(belief, joint_a, model)
    post = [
        pred[sp] * model.obs_prob(sp, joint_a, joint_obs)
        for sp in range(N_STATES)
    ]
    return normalize_belief(post)


def simulate_online_episode(
    model: TigerModel,
    args,
    episode_seed: int,
    verbose: bool = False,
) -> float:
    rng = random.Random(episode_seed)
    true_state = model.sample_initial_state(rng)

    # Belief at the most recent communication synchronization point.
    common_belief = list(model.init_belief)

    # Joint action/observation history since the last communication event.
    pending_history = []

    # Private beliefs used for planning between communication events.
    beliefs = {
        0: list(model.init_belief),
        1: list(model.init_belief),
    }

    total_reward = 0.0

    if verbose:
        print(f"initial true_state={'LEFT' if true_state == TIGER_LEFT else 'RIGHT'}")
        print(f"initial beliefs={beliefs}")

    for t in range(args.horizon):
        remaining = args.horizon - t

        if verbose:
            print(
                f"DEBUG entering t={t}, remaining={remaining}, "
                f"planner={getattr(args, 'planner', 'open-loop')}"
            )

        if getattr(args, "planner", "open-loop") == "belief-obs":
            actions, raw_actions, plans, entropies, root_masses = run_belief_obs_planning_step(
                beliefs=beliefs,
                remaining_horizon=remaining,
                model=model,
                args=args,
                seed=episode_seed + 7919 * t,
            )
        elif getattr(args, "planner", "open-loop") == "obs":
            actions, raw_actions, plans, entropies, root_masses = run_obs_planning_step(
                beliefs=beliefs,
                remaining_horizon=remaining,
                model=model,
                args=args,
                seed=episode_seed + 7919 * t,
            )
        else:
            actions, raw_actions, plans, entropies, root_masses = run_planning_step(
                beliefs=beliefs,
                remaining_horizon=remaining,
                model=model,
                args=args,
                seed=episode_seed + 7919 * t,
            )
            ## REMOVE
        if verbose or getattr(args, "debug_obs", False):
            print(
                f"DEBUG expected one-step rewards t={t}: "
                f"belief0={[round(x, 3) for x in beliefs[0]]} "
                f"belief1={[round(x, 3) for x in beliefs[1]]} "
                f"values_b0={all_one_step_expected_rewards(beliefs[0], model)} "
                f"values_b1={all_one_step_expected_rewards(beliefs[1], model)}",
                flush=True,
            )

        if verbose:
            print(
                f"DEBUG planned t={t}: "
                f"raw=({ACTION_NAME[raw_actions[0]]},{ACTION_NAME[raw_actions[1]]}), "
                f"exec=({ACTION_NAME[actions[0]]},{ACTION_NAME[actions[1]]}), "
                f"root_mass={root_masses}"
            )

        pre_beliefs = {
            0: list(beliefs[0]),
            1: list(beliefs[1]),
        }

        a0, a1 = actions[0], actions[1]
        joint_a = model.joint_action(a0, a1)

        reward = model.reward(true_state, joint_a)
        total_reward += reward

        next_state = model.sample_next_state(true_state, joint_a, rng)
        joint_o = model.sample_joint_obs(next_state, joint_a, rng)
        o0, o1 = model.split_obs(joint_o)

        beliefs[0] = update_local_belief(beliefs[0], joint_a, o0, 0, model)
        beliefs[1] = update_local_belief(beliefs[1], joint_a, o1, 1, model)

        pending_history.append((joint_a, joint_o))

        if should_env_communicate(args, t, a0, a1):
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

        if verbose:
            print(
                f"t={t:02d} rem={remaining:02d} "
                f"a=({ACTION_NAME[a0]},{ACTION_NAME[a1]}) "
                f"r={reward:6.1f} "
                f"obs=({OBS_NAME[o0]},{OBS_NAME[o1]}) "
                f"next={'LEFT' if next_state == TIGER_LEFT else 'RIGHT'} "
                f"belief0={[round(x,3) for x in beliefs[0]]} "
                f"belief1={[round(x,3) for x in beliefs[1]]} "
                f"H={{{0}: {entropies.get(0,0):.3f}, {1}: {entropies.get(1,0):.3f}}} "
                f"raw_actions=({ACTION_NAME[raw_actions[0]]},{ACTION_NAME[raw_actions[1]]}) "
                f"exec_actions=({ACTION_NAME[a0]},{ACTION_NAME[a1]})"
            )
            print(f"    pre_belief0={[round(x, 3) for x in pre_beliefs[0]]}")
            print(f"    pre_belief1={[round(x, 3) for x in pre_beliefs[1]]}")
            print(f"    root_mass={root_masses}")

            if getattr(args, "planner", "open-loop") == "belief-obs":
                key0 = (0, tuple(round(x, 12) for x in beliefs[0]))
                key1 = (0, tuple(round(x, 12) for x in beliefs[1]))
                print(f"    policy0_root={ACTION_NAME[plans[0].action(beliefs[0], key0)]}")
                print(f"    policy1_root={ACTION_NAME[plans[1].action(beliefs[1], key1)]}")
            elif getattr(args, "planner", "open-loop") == "obs":
                print(f"    policy0_root={ACTION_NAME[plans[0].action(())]}")
                print(f"    policy1_root={ACTION_NAME[plans[1].action(())]}")
            else:
                print(f"    seq0={[ACTION_NAME[x] for x in plans[0]]}")
                print(f"    seq1={[ACTION_NAME[x] for x in plans[1]]}")

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

# HELPERS
def root_action_masses(planner):
    masses = {OPEN_LEFT: 0.0, OPEN_RIGHT: 0.0, LISTEN: 0.0}

    for seq, p in planner.q.items():
        if len(seq) > 0:
            masses[seq[0]] += p

    return {ACTION_NAME[a]: round(v, 4) for a, v in masses.items()}

def expected_one_step_reward(belief, a0, a1, model):
    joint_a = model.joint_action(a0, a1)
    return sum(
        belief[s] * model.reward(s, joint_a)
        for s in range(N_STATES)
    )

def obs_root_action_masses(planner):
    masses = {OPEN_LEFT: 0.0, OPEN_RIGHT: 0.0, LISTEN: 0.0}

    for policy_key, p in planner.q.items():
        root_action = None

        for hist, action in policy_key:
            if hist == ():
                root_action = action
                break

        if root_action is not None:
            masses[root_action] += p

    return {ACTION_NAME[a]: round(v, 4) for a, v in masses.items()}


def belief_root_action_masses(planner):
    masses = {OPEN_LEFT: 0.0, OPEN_RIGHT: 0.0, LISTEN: 0.0}
    root_key = planner.root.belief_key

    for policy_key, p in planner.q.items():
        root_action = None
        for belief_key, action in policy_key:
            if belief_key == root_key:
                root_action = action
                break
        if root_action is not None:
            masses[root_action] += p

    return {ACTION_NAME[a]: round(v, 4) for a, v in masses.items()}


def all_one_step_expected_rewards(belief, model):
    pairs = [
        (OPEN_LEFT, OPEN_LEFT),
        (OPEN_LEFT, OPEN_RIGHT),
        (OPEN_LEFT, LISTEN),
        (OPEN_RIGHT, OPEN_LEFT),
        (OPEN_RIGHT, OPEN_RIGHT),
        (OPEN_RIGHT, LISTEN),
        (LISTEN, OPEN_LEFT),
        (LISTEN, OPEN_RIGHT),
        (LISTEN, LISTEN),
    ]

    return {
        f"{ACTION_NAME[a0]},{ACTION_NAME[a1]}": round(
            expected_one_step_reward(belief, a0, a1, model),
            3,
        )
        for a0, a1 in pairs
    }


def obs_policy_support_debug(planner):
    out = []

    for policy_key, p in sorted(planner.q.items(), key=lambda kv: kv[1], reverse=True):
        root_action = None
        num_entries = len(policy_key)

        for hist, action in policy_key:
            if hist == ():
                root_action = action
                break

        out.append({
            "p": round(p, 4),
            "root": ACTION_NAME[root_action] if root_action is not None else None,
            "num_entries": num_entries,
        })

    return out

        
def run_episode_worker(payload):
    ep, args_dict = payload
    args = argparse.Namespace(**args_dict)
    if ep == 0:
        print("WORKER using decmcts from:", decmcts.__file__, flush=True)

    model = TigerModel()
    ret = simulate_online_episode(
        model=model,
        args=args,
        episode_seed=args.seed + 104729 * ep,
        verbose=False,
    )
    return ep, ret

def should_env_communicate(args, t, a0, a1):
    if args.env_comm_mode == "none":
        return False

    if args.env_comm_mode == "both-listen":
        return a0 == LISTEN and a1 == LISTEN

    # periodic mode
    return args.env_comm_period > 0 and (t + 1) % args.env_comm_period == 0

def run_initial_planning_timing(model: TigerModel, args) -> None:
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
    print("Initial Tiger planning timing")
    print(f"horizon={args.horizon}")
    print(f"planning_time={elapsed:.6f}s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--episodes", type=int, default=50)

    # Dec-MCTS planning budget.
    parser.add_argument("--outer-iters", type=int, default=40)
    parser.add_argument("--tau", type=int, default=100)
    parser.add_argument("--num-seq", type=int, default=10)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument(
        "--score-samples",
        type=int,
        default=0,
        help="MC rollouts per candidate during obs/belief-obs sample-space scoring. 0 reuses --num-samples.",
    )
    parser.add_argument("--comm-period", type=int, default=1)
    parser.add_argument("--env-comm-period", type=int, default=1)

    # Dec-MCTS hyperparameters.
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--cp", type=float, default=0.5)
    parser.add_argument("--beta-init", type=float, default=2.0)
    parser.add_argument("--beta-decay", type=float, default=0.995)
    parser.add_argument("--alpha", type=float, default=0.02)
    parser.add_argument("--progressive-widening-visits", type=int, default=0)

    # Tiger heuristic/evaluation.
    parser.add_argument("--open-threshold", type=float, default=0.85)
    parser.add_argument("--local-utility", choices=["difference", "global"], default="difference")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose-first", action="store_true")
    parser.add_argument("--guard-actions", action="store_true")
    parser.add_argument("--env-comm-mode", choices=["periodic", "both-listen", "none"], default="periodic", help="Execution-time observation sharing mode.")
    parser.add_argument("--diverse-sample-space", action="store_true")
    parser.add_argument(
        "--planner",
        choices=["open-loop", "obs", "belief-obs"],
        default="open-loop",
        help="Planner backend: vanilla open-loop DecMCTS, history ObsDecMCTS, or belief-indexed ObsDecMCTS.",
    )
    parser.add_argument(
        "--action-source",
        choices=["tree", "disc_tree", "policy", "visits", "policy_value", "policy_marginal"],
        default="policy_marginal",
    )
    parser.add_argument("--debug-obs", action="store_true", help="Print ObsDecMCTS tree, edge, policy-support, and one-step reward diagnostics.")
    parser.add_argument(
        "--belief-share-nodes",
        action="store_true",
        help="For --planner belief-obs, reuse nodes with the same depth and belief key.",
    )

    # Adding parallelism across all cores
    parser.add_argument(
        "--jobs",
        type=int,
        default=-1,
        help="Number of parallel worker processes. Use 1 for serial or -1 for all CPU cores.",
    )

    parser.add_argument("--timing-only", action="store_true",
                        help="Time a single full-horizon planning step and exit.")

    args = parser.parse_args()

    model = TigerModel()

    if args.timing_only:
        run_initial_planning_timing(model, args)
        return

    print("Online Dec-MCTS Tiger benchmark")
    print(f"horizon={args.horizon}, episodes={args.episodes}")
    print(
        f"planning: outer_iters={args.outer_iters}, tau={args.tau}, "
        f"num_seq={args.num_seq}, num_samples={args.num_samples}, "
        f"score_samples={args.score_samples}")
    print(
        f"utility={args.local_utility}, open_threshold={args.open_threshold}, "
        f"env_comm_period={args.env_comm_period}, seed={args.seed}")
    print()

    print(
        f"utility={args.local_utility}, open_threshold={args.open_threshold}, "
        f"env_comm_mode={args.env_comm_mode}, env_comm_period={args.env_comm_period}, "
        f"guard_actions={args.guard_actions}, "
        f"planner={getattr(args, 'planner', 'open-loop')}, "
        f"seed={args.seed}"
    )

    returns = []

    # Keep verbose runs serial so the debug trace is readable.
    if args.jobs == 1 or args.verbose_first:
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
        n_jobs = min(n_jobs, args.episodes)

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
                        f"last_ep={ep:4d}, last={ret:8.3f}, "
                        f"mean={mean:8.3f} ± {ci:.3f}",
                        flush=True,
                    )

    mean, ci = confidence_interval_95(returns)
    print()
    print(f"Final mean return: {mean:.5f} ± {ci:.5f} (95% CI)")



if __name__ == "__main__":
    main()
