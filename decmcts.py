"""
decmcts.py
----------

This file implements the *open-loop action-sequence* Dec-MCTS planner described
by Best et al. Each robot maintains a local MCTS tree over its own action
sequences and a sparse probability distribution q^r over promising complete
action sequences. 

Each edge is an action, each node is an action sequence <-- tree architecture

Only implements a planner, receding-horizon control scheme implemented in benchmarks

"""

from __future__ import annotations

import copy
import math
import random
from typing import Any, Callable, Dict, Hashable, Iterable, List, Optional, Sequence, Tuple


Action = Any
RobotID = Hashable
ActionSeq = Tuple[Action, ...]
JointSequences = Dict[RobotID, List[Action]]


class DecMCTSNode:
    """
    Node in robot r's local search tree T^r.

    Each edge is one action by robot r.
    A root-to-node path is an open-loop action prefix x^r.

    The node stores:
      - action_sequence: prefix from root to this node
      - untried_actions: legal actions not yet expanded at this node
      - standard stats: visits, cum_reward
      - discounted D-UCT stats: disc_visits, disc_reward
      - representative_sequence: best full rollout-completed sequence observed
        through this node
    """

    def __init__(self, state: Any, parent: Optional["DecMCTSNode"] = None, action: Any = None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children: List[DecMCTSNode] = []

        self.visits = 0
        self.cum_reward = 0.0

        self.disc_visits = 0.0
        self.disc_reward = 0.0

        self.action_sequence: List[Action] = (
            parent.action_sequence + [action] if parent is not None else []
        )

        self.untried_actions: List[Action] = (
            list(state.get_legal_actions())
            if state is not None and not state.is_terminal_state()
            else []
        )

        # Full sequence completed by rollout; used for X_hat.
        self.representative_sequence: Optional[ActionSeq] = None
        self.representative_reward: float = float("-inf")

    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0

    def is_terminal(self) -> bool:
        return self.state is None or self.state.is_terminal_state()

    def add_child(self, action: Action, next_state: Any) -> "DecMCTSNode":
        child = DecMCTSNode(next_state, parent=self, action=action)
        self.children.append(child)
        if action in self.untried_actions:
            self.untried_actions.remove(action)
        return child

    def q(self) -> float:
        return self.cum_reward / self.visits if self.visits > 0 else 0.0

    def disc_q(self) -> float:
        return self.disc_reward / self.disc_visits if self.disc_visits > 0 else 0.0

    def d_ucb(
        self,
        parent: "DecMCTSNode",
        gamma: float,
        cp: float,
        min_reward: float,
        max_reward: float,
    ) -> float:
        """
        D-UCT score:
            discounted empirical mean + discounted exploration bonus

        The empirical mean is normalized to [0,1] using observed reward bounds
        so that cp is reasonably stable across reward scales.
        """
        if self.disc_visits <= 0.0:
            return float("inf")
        if parent.disc_visits <= 1.0:
            return float("inf")

        q = self.disc_q()
        if max_reward > min_reward:
            q_norm = (q - min_reward) / (max_reward - min_reward)
        else:
            q_norm = 0.5

        # Guard against log(<1) from early discounted counts.
        parent_count = max(parent.disc_visits, 1.0000001)
        bonus = 2.0 * cp * math.sqrt(math.log(parent_count) / self.disc_visits)
        return q_norm + bonus

    def update_discounted(self, reward: float, visited: bool, gamma: float) -> None:
        self.disc_visits = gamma * self.disc_visits + (1.0 if visited else 0.0)
        self.disc_reward = gamma * self.disc_reward + (reward if visited else 0.0)


class DecMCTS:
    """
    Dec-MCTS planner for one robot r.

    Required state interface:
        state.get_legal_actions() -> list
        state.take_action(action) -> new state
        state.is_terminal_state() -> bool

    Parameters
    ----------
    robot_id:
        This planner's robot id.
    robot_ids:
        All robot ids.
    init_state:
        Root state for this planning call.
    local_utility_fn:
        f^r(joint_sequences) -> float, where
        joint_sequences = {robot_id: [a0, a1, ...], ...}.
        Usually the difference reward:
            g(x^r, x^{-r}) - g(x^r_null, x^{-r})
    rollout_policy:
        Optional callable: rollout_policy(planner, node, x_others) -> list[action].
        Use this to inject a domain heuristic. It must not use hidden true state
        unavailable to the agent.
    default_sequence_fn:
        Optional callable: default_sequence_fn(robot_id) -> list[action].
        Used when no distribution has been received for another robot.
    """

    def __init__(
        self,
        robot_id: RobotID,
        robot_ids: Sequence[RobotID],
        init_state: Any,
        local_utility_fn: Callable[[JointSequences], float],
        *,
        rollout_policy: Optional[Callable[["DecMCTS", DecMCTSNode, JointSequences], List[Action]]] = None,
        default_sequence_fn: Optional[Callable[[RobotID], List[Action]]] = None,
        gamma: float = 0.95,
        cp: float = 0.5,
        rollout_depth: int = 20,
        tau: int = 10,
        num_seq: int = 10,
        num_samples: int = 30,
        beta_init: float = 1.0,
        beta_decay: float = 0.995,
        alpha: float = 0.01,
        diverse_sample_space: bool = False,
        seed: Optional[int] = None,
    ):
        if not (0.5 < gamma <= 1.0):
            raise ValueError("gamma should be in (0.5, 1].")
        if tau <= 0:
            raise ValueError("tau must be positive.")
        if num_seq <= 0:
            raise ValueError("num_seq must be positive.")
        if num_samples <= 0:
            raise ValueError("num_samples must be positive.")

        self.robot_id = robot_id
        self.robot_ids = list(robot_ids)
        self.local_utility_fn = local_utility_fn
        self.rollout_policy = rollout_policy
        self.default_sequence_fn = default_sequence_fn

        self.gamma = gamma
        self.cp = cp
        self.rollout_depth = rollout_depth
        self.tau = tau
        self.num_seq = num_seq
        self.num_samples = num_samples
        self.beta = beta_init
        self.beta_init = beta_init
        self.beta_decay = beta_decay
        self.alpha = alpha
        self.diverse_sample_space = diverse_sample_space

        self.rng = random.Random(seed)

        self.min_reward = float("inf")
        self.max_reward = float("-inf")

        self.root = DecMCTSNode(init_state)
        self._all_nodes: List[DecMCTSNode] = [self.root]
        self.t = 0

        # Other robots' distributions: {other_robot_id: {action_seq_tuple: prob}}
        self.received_dists: Dict[RobotID, Dict[ActionSeq, float]] = {
            r: {} for r in self.robot_ids if r != self.robot_id
        }

        # Own sparse sample space and distribution.
        self.X_hat: List[ActionSeq] = []
        self.q: Dict[ActionSeq, float] = {}

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def iterate(self, n_outer: int = 1) -> None:
        """
        Run n_outer Dec-MCTS outer iterations.

        Correct schedule:
            update sample space
            grow tree tau times
            update sample space again to include newly found rollouts
            update q once
            cool beta once
        """
        for _ in range(n_outer):
            self._update_sample_space()

            for _ in range(self.tau):
                self._grow_tree_once()

            if self.X_hat:
                self._update_distribution()

            self.beta *= self.beta_decay

    def receive(self, robot_id: RobotID, X_hat: Sequence[Sequence[Action]], q: Any) -> None:
        """
        Receive another robot's distribution.

        q may be:
          - dict {sequence_tuple: probability}
          - list of probabilities aligned with X_hat
        """
        if isinstance(q, dict):
            self.received_dists[robot_id] = {
                tuple(seq): float(prob) for seq, prob in q.items()
            }
        else:
            self.received_dists[robot_id] = {
                tuple(seq): float(prob) for seq, prob in zip(X_hat, q)
            }

    def receive_dist_dict(self, robot_id: RobotID, dist_dict: Dict[ActionSeq, float]) -> None:
        self.received_dists[robot_id] = {
            tuple(seq): float(prob) for seq, prob in dist_dict.items()
        }

    def get_distribution(self) -> Tuple[List[ActionSeq], Dict[ActionSeq, float]]:
        return list(self.X_hat), copy.copy(self.q)

    def best_action_sequence(self) -> List[Action]:
        """
        Return argmax_x q^r(x). If q is empty, fall back to greedy tree descent.
        """
        if self.q:
            return list(max(self.q, key=self.q.get))
        return self._greedy_tree_sequence()

    def best_action(self):
        """
        Return the most likely first action under q^r.

        This is more stable for online/receding-horizon execution than taking the
        first action of the single argmax sequence, especially when q is high-entropy.
        """
        if self.q:
            action_mass = {}
            for seq, prob in self.q.items():
                if len(seq) == 0:
                    continue
                a0 = seq[0]
                action_mass[a0] = action_mass.get(a0, 0.0) + prob

            if action_mass:
                return max(action_mass, key=action_mass.get)

        seq = self._greedy_tree_sequence()
        return seq[0] if seq else None

    @staticmethod
    def root_action_mass_from_dist(dist: Dict[ActionSeq, float]) -> Dict[Action, float]:
        masses: Dict[Action, float] = {}

        for seq, p in dist.items():
            if len(seq) == 0:
                continue
            a0 = seq[0]
            masses[a0] = masses.get(a0, 0.0) + p

        return masses

    # -------------------------------------------------------------------------
    # Tree growth
    # -------------------------------------------------------------------------

    def _grow_tree_once(self) -> None:
        self.t += 1

        # Selection.
        node = self.root
        path = [node]
        while not node.is_terminal() and node.is_fully_expanded() and node.children:
            node = max(
                node.children,
                key=lambda c: c.d_ucb(
                    parent=node,
                    gamma=self.gamma,
                    cp=self.cp,
                    min_reward=self.min_reward,
                    max_reward=self.max_reward,
                ),
            )
            path.append(node)

        # Expansion.
        if not node.is_terminal() and not node.is_fully_expanded():
            action = self.rng.choice(node.untried_actions)
            next_state = node.state.take_action(action)
            node = node.add_child(action, next_state)
            self._all_nodes.append(node)
            path.append(node)

        # Sample other robots before rollout, as in Algorithm 2.
        x_others = self._sample_others()

        # Rollout completion for this robot.
        rollout_actions = self._rollout(node, x_others)
        x_r = list(node.action_sequence) + list(rollout_actions)

        # Evaluate local utility.
        joint = {**x_others, self.robot_id: x_r}
        F_t = float(self.local_utility_fn(joint))

        self.min_reward = min(self.min_reward, F_t)
        self.max_reward = max(self.max_reward, F_t)

        # Store the best full rollout-completed sequence seen through this node.
        if F_t > node.representative_reward:
            node.representative_reward = F_t
            node.representative_sequence = tuple(x_r)

        self._backprop(path, F_t)

    def _rollout(self, node: DecMCTSNode, x_others: JointSequences) -> List[Action]:
        if self.rollout_policy is not None:
            return list(self.rollout_policy(self, node, x_others))

        # Default random rollout.
        state = node.state
        actions: List[Action] = []
        depth = 0

        while state is not None and not state.is_terminal_state() and depth < self.rollout_depth:
            legal = list(state.get_legal_actions())
            if not legal:
                break
            a = self.rng.choice(legal)
            actions.append(a)
            state = state.take_action(a)
            depth += 1

        return actions

    def _backprop(self, path: List[DecMCTSNode], F_t: float) -> None:
        path_ids = {id(n) for n in path}

        for node in self._all_nodes:
            on_path = id(node) in path_ids
            if on_path:
                node.visits += 1
                node.cum_reward += F_t
            node.update_discounted(F_t, visited=on_path, gamma=self.gamma)

    def _collect_nodes(self, node: DecMCTSNode, out: List[DecMCTSNode]) -> None:
        out.append(node)
        for child in node.children:
            self._collect_nodes(child, out)

    # -------------------------------------------------------------------------
    # Product-distribution update
    # -------------------------------------------------------------------------

    def _update_sample_space(self) -> None:
        """
        Select X_hat as the num_seq best full rollout-completed sequences.

        If diverse_sample_space is True, guarantees at least one representative
        per root action before filling the rest by global disc_q ranking.
        """
        candidates = []
        self._collect_nodes(self.root, candidates)

        candidates = [n for n in candidates if n.representative_sequence is not None]
        if not candidates:
            return

        candidates.sort(key=lambda n: n.disc_q(), reverse=True)

        new_X_hat = []
        seen = set()

        if self.diverse_sample_space:
            # Include the best sequence for each root action first.
            by_root_action = {}
            for n in candidates:
                seq = tuple(n.representative_sequence)
                if not seq:
                    continue
                a0 = seq[0]
                if a0 not in by_root_action:
                    by_root_action[a0] = seq

            for seq in by_root_action.values():
                if seq not in seen:
                    new_X_hat.append(seq)
                    seen.add(seq)

        # Fill with globally best candidates.
        for n in candidates:
            if len(new_X_hat) >= self.num_seq:
                break
            seq = tuple(n.representative_sequence)
            if not seq or seq in seen:
                continue
            new_X_hat.append(seq)
            seen.add(seq)

        if not new_X_hat:
            return

        if set(new_X_hat) != set(self.X_hat):
            self.X_hat = new_X_hat
            self.q = {seq: 1.0 / len(new_X_hat) for seq in new_X_hat}
            self.beta = self.beta_init

    def _update_distribution(self) -> None:
        if not self.X_hat:
            return

        E_f = self._estimate_expectation(fixed_xr=None)
        H = -sum(p * math.log(p) for p in self.q.values() if p > 0)

        new_q: Dict[ActionSeq, float] = {}
        denom = self.max_reward - self.min_reward
        beta = max(self.beta, 1e-9)

        for xr_tup in self.X_hat:
            E_f_given_xr = self._estimate_expectation(fixed_xr=xr_tup)
            q_val = self.q.get(xr_tup, 1.0 / len(self.X_hat))
            ln_q = math.log(max(q_val, 1e-12))

            if denom > 0:
                norm_E_f = (E_f - self.min_reward) / denom
                norm_E_f_given = (E_f_given_xr - self.min_reward) / denom
            else:
                norm_E_f = 0.5
                norm_E_f_given = 0.5

            # Algorithm 3 additive first-order update.
            delta = self.alpha * q_val * (
                (norm_E_f - norm_E_f_given) / beta + H + ln_q
            )
            new_q[xr_tup] = max(1e-12, q_val - delta)

        self.q = self._normalize(new_q)

    def _estimate_expectation(self, fixed_xr: Optional[ActionSeq] = None) -> float:
        total = 0.0
        for _ in range(self.num_samples):
            xr = list(fixed_xr) if fixed_xr is not None else self._sample_own()
            x_others = self._sample_others()
            joint = {**x_others, self.robot_id: xr}
            total += float(self.local_utility_fn(joint))
        return total / self.num_samples

    def _sample_own(self) -> List[Action]:
        return list(self._sample_from_dist(self.q, self.robot_id))

    def _sample_others(self) -> JointSequences:
        return {
            r: list(self._sample_from_dist(self.received_dists.get(r, {}), r))
            for r in self.robot_ids
            if r != self.robot_id
        }

    def _sample_from_dist(self, dist: Dict[ActionSeq, float], robot_id: RobotID) -> ActionSeq:
        if not dist:
            if self.default_sequence_fn is not None:
                return tuple(self.default_sequence_fn(robot_id))
            return tuple()

        keys = list(dist.keys())
        probs = list(dist.values())
        total = sum(probs)
        if total <= 0:
            return self.rng.choice(keys)

        r = self.rng.random() * total
        cum = 0.0
        for key, p in zip(keys, probs):
            cum += p
            if r <= cum:
                return key
        return keys[-1]

    @staticmethod
    def _normalize(dist: Dict[ActionSeq, float]) -> Dict[ActionSeq, float]:
        total = sum(max(0.0, v) for v in dist.values())
        if total <= 0:
            n = max(len(dist), 1)
            return {k: 1.0 / n for k in dist}
        return {k: max(0.0, v) / total for k, v in dist.items()}

    def _greedy_tree_sequence(self) -> List[Action]:
        node = self.root
        seq: List[Action] = []
        while node.children:
            visited = [c for c in node.children if c.visits > 0]
            if not visited:
                break
            best = max(visited, key=lambda c: c.q())
            seq.append(best.action)
            node = best
        return seq


class DecMCTSTeam:
    """
    Convenience wrapper for synchronous simulation of a Dec-MCTS team.

    The paper's algorithm is decentralized/asynchronous; this wrapper is just for
    reproducible experiments in one process.
    """

    def __init__(self, planners: Dict[RobotID, DecMCTS]):
        self.planners = planners

    def iterate_and_communicate(self, n_outer: int = 1, comm_period: int = 1) -> None:
        for i in range(1, n_outer + 1):
            for planner in self.planners.values():
                planner.iterate(1)

            if comm_period > 0 and i % comm_period == 0:
                for rid, planner in self.planners.items():
                    _X, q = planner.get_distribution()
                    for other_id, other_planner in self.planners.items():
                        if other_id != rid:
                            other_planner.receive_dist_dict(rid, q)

    def best_sequences(self) -> Dict[RobotID, List[Action]]:
        return {rid: p.best_action_sequence() for rid, p in self.planners.items()}

    def best_actions(self) -> Dict[RobotID, Optional[Action]]:
        return {rid: p.best_action() for rid, p in self.planners.items()}

    def entropies(self) -> Dict[RobotID, float]:
        out = {}
        for rid, planner in self.planners.items():
            out[rid] = -sum(p * math.log(p) for p in planner.q.values() if p > 0)
        return out
