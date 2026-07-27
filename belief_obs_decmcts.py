from __future__ import annotations

import copy
import math
import random
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Hashable, List, Optional, Sequence, Tuple


Action = Any
Belief = Sequence[float]
BeliefKey = Hashable
NodeKey = Tuple[int, BeliefKey]
Observation = Any
RobotID = Hashable
State = Any

PolicyKey = Tuple[Tuple[NodeKey, Action], ...]

_PROFILER: Any = None


def set_profiler(profiler: Any) -> None:
    """Register an optional profiler with add(name, elapsed, count=1)."""
    global _PROFILER
    _PROFILER = profiler


def _profiled_method(label: str, fn: Callable) -> Callable:
    def wrapped(self, *args, **kwargs):
        profiler = _PROFILER
        if profiler is None:
            return fn(self, *args, **kwargs)
        t0 = time.perf_counter()
        try:
            return fn(self, *args, **kwargs)
        finally:
            profiler.add(label, time.perf_counter() - t0)

    wrapped.__name__ = getattr(fn, "__name__", "wrapped")
    wrapped.__doc__ = getattr(fn, "__doc__", None)
    return wrapped


@dataclass(frozen=True)
class StepResult:
    next_state: State
    joint_obs: Any
    reward: float


def rounded_belief_key(belief: Belief, ndigits: int = 12) -> Tuple[float, ...]:
    return tuple(round(float(x), ndigits) for x in belief)


class BeliefPolicyTree:
    """
    Sparse belief-conditioned policy.

    table:
        local belief key -> action

    Missing belief keys fall back to default_action_fn(belief).
    """

    def __init__(
        self,
        table: Optional[Dict[NodeKey, Action]] = None,
        default_action_fn: Optional[Callable[[Belief], Action]] = None,
        belief_lookup: Optional[Dict[BeliefKey, List[float]]] = None,
    ):
        self.table: Dict[NodeKey, Action] = dict(table or {})
        self.default_action_fn = default_action_fn
        self.belief_lookup = dict(belief_lookup or {})

    def action(self, belief: Belief, node_key: NodeKey) -> Action:
        if node_key in self.table:
            return self.table[node_key]
        if self.default_action_fn is not None:
            return self.default_action_fn(belief)
        raise KeyError(f"No action for node key {node_key} and no default_action_fn.")

    def set_action(
        self,
        node_key: NodeKey,
        belief_key: BeliefKey,
        belief: Belief,
        action: Action,
    ) -> None:
        self.table[node_key] = action
        self.belief_lookup[belief_key] = list(belief)

    def key(self) -> PolicyKey:
        return tuple(sorted(self.table.items(), key=lambda x: repr(x[0])))

    @staticmethod
    def from_key(
        key: PolicyKey,
        default_action_fn: Optional[Callable[[Belief], Action]] = None,
        belief_lookup: Optional[Dict[BeliefKey, List[float]]] = None,
    ) -> "BeliefPolicyTree":
        return BeliefPolicyTree(dict(key), default_action_fn, belief_lookup)


class BeliefActionEdge:
    def __init__(self, action: Action):
        self.action = action
        self.visits = 0
        self.value_sum = 0.0
        self.disc_visits = 0.0
        self.disc_reward = 0.0
        self.obs_children: Dict[Observation, BeliefNode] = {}

    def q(self) -> float:
        return self.value_sum / self.visits if self.visits > 0 else 0.0

    def disc_q(self) -> float:
        return self.disc_reward / self.disc_visits if self.disc_visits > 0 else 0.0

    def update_discounted(self, reward: float, visited: bool, gamma: float) -> None:
        self.disc_visits = gamma * self.disc_visits + (1.0 if visited else 0.0)
        self.disc_reward = gamma * self.disc_reward + (reward if visited else 0.0)


class BeliefNode:
    def __init__(
        self,
        belief: Belief,
        belief_key: BeliefKey,
        depth: int,
        legal_actions: Sequence[Action],
    ):
        self.belief = list(belief)
        self.belief_key = belief_key
        self.node_key: NodeKey = (depth, belief_key)
        self.depth = depth
        self.legal_actions = list(legal_actions)
        self.actions: Dict[Action, BeliefActionEdge] = {}
        self.untried_actions = list(legal_actions)
        self.visits = 0
        self.value_sum = 0.0
        self.representative_policy: Optional[PolicyKey] = None
        self.representative_reward = float("-inf")
        self.disc_visits = 0.0
        self.disc_reward = 0.0

    def add_action_edge(self, action: Action) -> BeliefActionEdge:
        edge = BeliefActionEdge(action)
        self.actions[action] = edge
        if action in self.untried_actions:
            self.untried_actions.remove(action)
        return edge

    def q(self) -> float:
        return self.value_sum / self.visits if self.visits > 0 else 0.0

    def disc_q(self) -> float:
        return self.disc_reward / self.disc_visits if self.disc_visits > 0 else 0.0

    def update_discounted(self, reward: float, visited: bool, gamma: float) -> None:
        self.disc_visits = gamma * self.disc_visits + (1.0 if visited else 0.0)
        self.disc_reward = gamma * self.disc_reward + (reward if visited else 0.0)


class BeliefObsDecMCTS:
    """
    Belief-indexed ObsDecMCTS variant.

    Nodes are local belief states, action edges are actions, and observations
    trigger Bayes updates into child belief nodes.

    Required model interface:
    - sample_state_from_belief(belief, rng)
    - step(state, joint_action, rng) -> StepResult
    - split_obs(joint_obs)
    - joint_action_from_dict({rid: action})
    - update_belief(belief, joint_action, local_obs, robot_id) -> belief
    """

    def __init__(
        self,
        robot_id: RobotID,
        robot_ids: Sequence[RobotID],
        root_belief: Belief,
        model: Any,
        legal_actions_fn: Callable[[Belief, int], Sequence[Action]],
        default_action_fn: Callable[[Belief], Action],
        default_action_fns_by_robot: Optional[
            Dict[RobotID, Callable[[Belief], Action]]
        ] = None,
        root_beliefs_by_robot: Optional[Dict[RobotID, Belief]] = None,
        belief_key_fn: Callable[[Belief], BeliefKey] = rounded_belief_key,
        share_belief_nodes: bool = False,
        *,
        gamma: float = 1.0,
        cp: float = 1.0,
        horizon: int = 8,
        tau: int = 100,
        num_policies: int = 10,
        num_samples: int = 30,
        score_samples: Optional[int] = None,
        beta_init: float = 2.0,
        beta_decay: float = 0.995,
        alpha: float = 0.01,
        seed: Optional[int] = None,
    ):
        self.robot_id = robot_id
        self.robot_ids = list(robot_ids)
        self.root_belief = list(root_belief)
        self.root_beliefs_by_robot = {
            rid: list(root_belief)
            for rid in self.robot_ids
        }
        if root_beliefs_by_robot:
            self.root_beliefs_by_robot.update({
                rid: list(b)
                for rid, b in root_beliefs_by_robot.items()
            })
        self.model = model
        self.legal_actions_fn = legal_actions_fn
        self.default_action_fn = default_action_fn
        self.default_action_fns_by_robot = dict(default_action_fns_by_robot or {})
        self.default_action_fns_by_robot.setdefault(self.robot_id, self.default_action_fn)
        self.belief_key_fn = belief_key_fn
        self.share_belief_nodes = share_belief_nodes

        self.gamma = gamma
        self.cp = cp
        self.horizon = horizon
        self.tau = tau
        self.num_policies = num_policies
        self.num_samples = num_samples
        self.score_samples = score_samples
        self.beta_init = beta_init
        self.beta = beta_init
        self.beta_decay = beta_decay
        self.alpha = alpha
        self.rng = random.Random(seed)

        self.belief_lookup: Dict[BeliefKey, List[float]] = {}
        self.belief_update_cache: Dict[
            Tuple[RobotID, BeliefKey, Any, Observation],
            Tuple[List[float], BeliefKey],
        ] = {}
        self.belief_obj_key_cache: Dict[int, Tuple[Belief, BeliefKey]] = {}
        self.belief_value_key_cache: Dict[Tuple[float, ...], BeliefKey] = {}
        root_key = self._belief_key(self.root_belief)
        self.root = BeliefNode(
            belief=self.root_belief,
            belief_key=root_key,
            depth=0,
            legal_actions=self.legal_actions_fn(self.root_belief, 0),
        )
        self._all_nodes: List[BeliefNode] = [self.root]
        self._all_edges: List[BeliefActionEdge] = []
        self.node_table: Dict[Tuple[int, BeliefKey], BeliefNode] = {
            (0, root_key): self.root
        }

        self.received_dists: Dict[RobotID, Dict[PolicyKey, float]] = {
            rid: {} for rid in self.robot_ids if rid != self.robot_id
        }
        self.X_hat: List[PolicyKey] = []
        self.q: Dict[PolicyKey, float] = {}
        self.min_reward = float("inf")
        self.max_reward = float("-inf")

    def _belief_key(self, belief: Belief) -> BeliefKey:
        obj_id = id(belief)
        cached = self.belief_obj_key_cache.get(obj_id)
        if cached is not None and cached[0] is belief:
            return cached[1]
        value_key = tuple(float(x) for x in belief)
        key = self.belief_value_key_cache.get(value_key)
        if key is None:
            key = self.belief_key_fn(belief)
            self.belief_value_key_cache[value_key] = key
        self.belief_lookup[key] = list(belief)
        self.belief_obj_key_cache[obj_id] = (belief, key)
        return key

    def _default_action_fn_for_robot(
        self,
        robot_id: RobotID,
    ) -> Callable[[Belief], Action]:
        return self.default_action_fns_by_robot.get(robot_id, self.default_action_fn)

    def _get_or_create_node(
        self,
        belief: Belief,
        belief_key: BeliefKey,
        depth: int,
    ) -> BeliefNode:
        table_key = (depth, belief_key)
        if not self.share_belief_nodes:
            node = BeliefNode(
                belief=belief,
                belief_key=belief_key,
                depth=depth,
                legal_actions=self.legal_actions_fn(belief, depth),
            )
            self._all_nodes.append(node)
            return node

        node = self.node_table.get(table_key)
        if node is not None:
            return node

        node = BeliefNode(
            belief=belief,
            belief_key=belief_key,
            depth=depth,
            legal_actions=self.legal_actions_fn(belief, depth),
        )
        self.node_table[table_key] = node
        self._all_nodes.append(node)
        return node

    def _update_belief_cached(
        self,
        robot_id: RobotID,
        belief: Belief,
        joint_action: Any,
        local_obs: Observation,
    ) -> Tuple[List[float], BeliefKey]:
        belief_key = self._belief_key(belief)
        cache_key = (robot_id, belief_key, joint_action, local_obs)
        cached = self.belief_update_cache.get(cache_key)
        if cached is not None:
            next_belief, next_key = cached
            return list(next_belief), next_key

        next_belief = list(
            self.model.update_belief(
                belief,
                joint_action,
                local_obs,
                robot_id,
            )
        )
        next_key = self._belief_key(next_belief)
        self.belief_update_cache[cache_key] = (list(next_belief), next_key)
        return next_belief, next_key

    def iterate(self, n_outer: int = 1) -> None:
        for _ in range(n_outer):
            self._update_sample_space()
            for _ in range(self.tau):
                self._grow_tree_once()
            if self.X_hat:
                self._update_distribution()
            self.beta *= self.beta_decay

    def receive_dist_dict(self, robot_id: RobotID, dist: Dict[PolicyKey, float]) -> None:
        self.received_dists[robot_id] = {
            tuple(policy_key): float(prob)
            for policy_key, prob in dist.items()
        }

    def get_distribution(self) -> Tuple[List[PolicyKey], Dict[PolicyKey, float]]:
        return list(self.X_hat), copy.copy(self.q)

    def best_policy(self) -> BeliefPolicyTree:
        if self.q:
            best_key = max(self.q, key=self.q.get)
            return BeliefPolicyTree.from_key(
                best_key,
                self.default_action_fn,
                self.belief_lookup,
            )
        return self._greedy_policy_from_tree()

    def best_action(self, belief: Optional[Belief] = None, source: str = "policy") -> Action:
        belief = self.root_belief if belief is None else list(belief)
        belief_key = self._belief_key(belief)
        node_key = (0, belief_key)

        if source == "policy":
            return self.best_policy().action(belief, node_key)

        if self.root.actions:
            if source == "tree":
                return max(self.root.actions.values(), key=lambda e: e.q()).action
            if source == "disc_tree":
                return max(self.root.actions.values(), key=lambda e: e.disc_q()).action
            if source == "visits":
                return max(self.root.actions.values(), key=lambda e: e.visits).action
            if source == "policy_marginal":
                masses: Dict[Action, float] = {}
                for policy_key, p in self.q.items():
                    policy = BeliefPolicyTree.from_key(
                    policy_key,
                    self.default_action_fn,
                    self.belief_lookup,
                )
                    a = policy.action(belief, node_key)
                    masses[a] = masses.get(a, 0.0) + p
                if masses:
                    return max(masses, key=masses.get)

        return self.best_policy().action(belief, node_key)

    def _policy_from_forced_root_action(self, root_action: Action) -> BeliefPolicyTree:
        policy = BeliefPolicyTree(default_action_fn=self.default_action_fn)
        policy.set_action(
            self.root.node_key,
            self.root.belief_key,
            self.root.belief,
            root_action,
        )
        edge = self.root.actions.get(root_action)
        if edge is None:
            return policy
        for child in edge.obs_children.values():
            self._fill_greedy_policy(child, policy)
        return policy

    def _grow_tree_once(self) -> None:
        state = self.model.sample_state_from_belief(self.root_belief, self.rng)
        own_policy = BeliefPolicyTree(default_action_fn=self.default_action_fn)
        other_policies = self._sample_other_policies()

        visited_edges: List[BeliefActionEdge] = []
        visited_nodes: List[BeliefNode] = []

        total_return = self._simulate_from_node(
            node=self.root,
            state=state,
            beliefs={rid: list(self.root_beliefs_by_robot[rid]) for rid in self.robot_ids},
            own_policy=own_policy,
            other_policies=other_policies,
            visited_nodes=visited_nodes,
            visited_edges=visited_edges,
            depth=0,
        )

        self.min_reward = min(self.min_reward, total_return)
        self.max_reward = max(self.max_reward, total_return)

        known_edge_ids = {id(e) for e in self._all_edges}
        for edge in visited_edges:
            if id(edge) not in known_edge_ids:
                self._all_edges.append(edge)
                known_edge_ids.add(id(edge))

        visited_edge_ids = {id(e) for e in visited_edges}
        for edge in self._all_edges:
            on_path = id(edge) in visited_edge_ids
            if on_path:
                edge.visits += 1
                edge.value_sum += total_return
            edge.update_discounted(total_return, on_path, self.gamma)

        visited_node_ids = {id(n) for n in visited_nodes}
        nodes: List[BeliefNode] = []
        self._collect_nodes(self.root, nodes)

        for node in nodes:
            on_path = id(node) in visited_node_ids
            if on_path:
                node.visits += 1
                node.value_sum += total_return
                if total_return > node.representative_reward:
                    node.representative_reward = total_return
                    node.representative_policy = own_policy.key()
            node.update_discounted(total_return, on_path, self.gamma)

    def _simulate_from_node(
        self,
        node: BeliefNode,
        state: State,
        beliefs: Dict[RobotID, List[float]],
        own_policy: BeliefPolicyTree,
        other_policies: Dict[RobotID, BeliefPolicyTree],
        visited_nodes: List[BeliefNode],
        visited_edges: List[BeliefActionEdge],
        depth: int,
    ) -> float:
        if depth >= self.horizon:
            return 0.0

        visited_nodes.append(node)
        action = self._select_or_expand_action(node)
        edge = node.actions[action]
        visited_edges.append(edge)
        own_policy.set_action(node.node_key, node.belief_key, node.belief, action)

        actions: Dict[RobotID, Action] = {}
        node_keys = {
            rid: (depth, self._belief_key(beliefs[rid]))
            for rid in self.robot_ids
        }
        for rid in self.robot_ids:
            if rid == self.robot_id:
                actions[rid] = action
            else:
                actions[rid] = other_policies[rid].action(beliefs[rid], node_keys[rid])

        joint_action = self.model.joint_action_from_dict(actions)
        step = self.model.step(state, joint_action, self.rng)
        local_obs_all = self.model.split_obs(step.joint_obs)

        next_beliefs = dict(beliefs)
        for idx, rid in enumerate(self.robot_ids):
            next_belief, _next_key = self._update_belief_cached(
                rid,
                beliefs[rid],
                joint_action,
                local_obs_all[idx],
            )
            next_beliefs[rid] = next_belief

        own_obs = local_obs_all[self.robot_ids.index(self.robot_id)]
        if own_obs not in edge.obs_children:
            next_belief = next_beliefs[self.robot_id]
            next_key = self._belief_key(next_belief)
            edge.obs_children[own_obs] = self._get_or_create_node(
                belief=next_belief,
                belief_key=next_key,
                depth=depth + 1,
            )
            future = self._rollout(
                state=step.next_state,
                beliefs=next_beliefs,
                own_policy=own_policy,
                other_policies=other_policies,
                depth=depth + 1,
            )
        else:
            child = edge.obs_children[own_obs]
            future = self._simulate_from_node(
                node=child,
                state=step.next_state,
                beliefs=next_beliefs,
                own_policy=own_policy,
                other_policies=other_policies,
                visited_nodes=visited_nodes,
                visited_edges=visited_edges,
                depth=depth + 1,
            )

        return step.reward + self.gamma * future

    def _select_or_expand_action(self, node: BeliefNode) -> Action:
        if node.untried_actions:
            action = self.rng.choice(node.untried_actions)
            edge = node.add_action_edge(action)
            self._all_edges.append(edge)
            return action
        return max(node.actions.values(), key=lambda edge: self._ucb(edge, node)).action

    def _ucb(self, edge: BeliefActionEdge, node: BeliefNode) -> float:
        if edge.disc_visits <= 0:
            return float("inf")
        q = edge.disc_q()
        if self.max_reward > self.min_reward:
            q = (q - self.min_reward) / (self.max_reward - self.min_reward)
        else:
            q = 0.5
        parent_count = max(node.disc_visits, 1.0000001)
        return q + self.cp * math.sqrt(math.log(parent_count) / edge.disc_visits)

    def _rollout(
        self,
        state: State,
        beliefs: Dict[RobotID, List[float]],
        own_policy: BeliefPolicyTree,
        other_policies: Dict[RobotID, BeliefPolicyTree],
        depth: int,
    ) -> float:
        total = 0.0
        discount = 1.0
        for t in range(depth, self.horizon):
            actions: Dict[RobotID, Action] = {}
            node_keys = {
                rid: (t, self._belief_key(beliefs[rid]))
                for rid in self.robot_ids
            }
            for rid in self.robot_ids:
                if rid == self.robot_id:
                    a = self.default_action_fn(beliefs[rid])
                    own_policy.set_action(
                        node_keys[rid],
                        node_keys[rid][1],
                        beliefs[rid],
                        a,
                    )
                    actions[rid] = a
                else:
                    actions[rid] = other_policies[rid].action(beliefs[rid], node_keys[rid])

            joint_action = self.model.joint_action_from_dict(actions)
            step = self.model.step(state, joint_action, self.rng)
            local_obs_all = self.model.split_obs(step.joint_obs)
            total += discount * step.reward
            discount *= self.gamma
            next_beliefs = dict(beliefs)
            for idx, rid in enumerate(self.robot_ids):
                next_belief, _next_key = self._update_belief_cached(
                    rid,
                    beliefs[rid],
                    joint_action,
                    local_obs_all[idx],
                )
                next_beliefs[rid] = next_belief
            beliefs = next_beliefs
            state = step.next_state
        return total

    def _replace_sample_space_preserve_q(self, new_X_hat: List[PolicyKey]) -> None:
        if not new_X_hat:
            return
        old_q = self.q
        if set(new_X_hat) == set(self.X_hat):
            self.X_hat = new_X_hat
            self.q = self._normalize({key: old_q.get(key, 0.0) for key in new_X_hat})
            return

        eps = 1e-3
        self.X_hat = new_X_hat
        self.q = self._normalize({
            key: old_q[key] if key in old_q else eps
            for key in new_X_hat
        })

    def _score_policy_key(self, key: PolicyKey, n_eval: int = 5) -> float:
        total = 0.0
        for _ in range(n_eval):
            own_policy = BeliefPolicyTree.from_key(
                key,
                self.default_action_fn,
                self.belief_lookup,
            )
            other_policies = self._sample_other_policies()
            state = self.model.sample_state_from_belief(self.root_belief, self.rng)
            total += self._eval_joint_policies(state, own_policy, other_policies)
        return total / n_eval

    def _update_sample_space(self) -> None:
        candidate_scores: Dict[PolicyKey, float] = {}
        nodes: List[BeliefNode] = []
        self._collect_nodes(self.root, nodes)

        for node in nodes:
            key = node.representative_policy
            if key is None:
                continue
            score = node.disc_q() if node.disc_visits > 0 else node.q()
            if key not in candidate_scores or score > candidate_scores[key]:
                candidate_scores[key] = score

        for edge in self.root.actions.values():
            policy = self._policy_from_forced_root_action(edge.action)
            key = policy.key()
            score = edge.q()
            if key not in candidate_scores or score > candidate_scores[key]:
                candidate_scores[key] = score

        if not candidate_scores:
            return

        scored = [
            (
                key,
                self._score_policy_key(
                    key,
                    n_eval=max(
                        1,
                        self.num_samples if self.score_samples is None else self.score_samples,
                    ),
                ),
            )
            for key in candidate_scores.keys()
        ]
        scored.sort(key=lambda kv: kv[1], reverse=True)
        new_X_hat = [key for key, _score in scored[: self.num_policies]]
        self._replace_sample_space_preserve_q(new_X_hat)

    def _update_distribution(self) -> None:
        if not self.X_hat:
            return
        E_f = self._estimate_expectation(fixed_policy_key=None)
        H = -sum(p * math.log(p) for p in self.q.values() if p > 0)
        denom = self.max_reward - self.min_reward
        beta = max(self.beta, 1e-9)
        new_q: Dict[PolicyKey, float] = {}

        for key in self.X_hat:
            E_f_given = self._estimate_expectation(fixed_policy_key=key)
            q_val = self.q.get(key, 1.0 / len(self.X_hat))
            ln_q = math.log(max(q_val, 1e-12))
            if denom > 0:
                norm_E = (E_f - self.min_reward) / denom
                norm_E_given = (E_f_given - self.min_reward) / denom
            else:
                norm_E = 0.5
                norm_E_given = 0.5
            delta = self.alpha * q_val * ((norm_E - norm_E_given) / beta + H + ln_q)
            new_q[key] = max(1e-12, q_val - delta)
        self.q = self._normalize(new_q)

    def _estimate_expectation(self, fixed_policy_key: Optional[PolicyKey]) -> float:
        total = 0.0
        for _ in range(self.num_samples):
            if fixed_policy_key is None:
                own_policy = self._sample_own_policy()
            else:
                own_policy = BeliefPolicyTree.from_key(
                    fixed_policy_key,
                    self.default_action_fn,
                    self.belief_lookup,
                )
            other_policies = self._sample_other_policies()
            state = self.model.sample_state_from_belief(self.root_belief, self.rng)
            total += self._eval_joint_policies(state, own_policy, other_policies)
        return total / self.num_samples

    def _eval_joint_policies(
        self,
        state: State,
        own_policy: BeliefPolicyTree,
        other_policies: Dict[RobotID, BeliefPolicyTree],
    ) -> float:
        beliefs = {rid: list(self.root_beliefs_by_robot[rid]) for rid in self.robot_ids}
        total = 0.0
        discount = 1.0
        for depth in range(self.horizon):
            actions: Dict[RobotID, Action] = {}
            node_keys = {
                rid: (depth, self._belief_key(beliefs[rid]))
                for rid in self.robot_ids
            }
            for rid in self.robot_ids:
                if rid == self.robot_id:
                    actions[rid] = own_policy.action(beliefs[rid], node_keys[rid])
                else:
                    actions[rid] = other_policies[rid].action(beliefs[rid], node_keys[rid])
            joint_action = self.model.joint_action_from_dict(actions)
            step = self.model.step(state, joint_action, self.rng)
            local_obs_all = self.model.split_obs(step.joint_obs)
            total += discount * step.reward
            discount *= self.gamma
            next_beliefs = dict(beliefs)
            for idx, rid in enumerate(self.robot_ids):
                next_belief, _next_key = self._update_belief_cached(
                    rid,
                    beliefs[rid],
                    joint_action,
                    local_obs_all[idx],
                )
                next_beliefs[rid] = next_belief
            beliefs = next_beliefs
            state = step.next_state
        return total

    def _sample_own_policy(self) -> BeliefPolicyTree:
        key = self._sample_from_dist(self.q)
        return BeliefPolicyTree.from_key(key, self.default_action_fn, self.belief_lookup)

    def _sample_other_policies(self) -> Dict[RobotID, BeliefPolicyTree]:
        out: Dict[RobotID, BeliefPolicyTree] = {}
        for rid in self.robot_ids:
            if rid == self.robot_id:
                continue
            default_action_fn = self._default_action_fn_for_robot(rid)
            dist = self.received_dists.get(rid, {})
            if dist:
                key = self._sample_from_dist(dist)
                out[rid] = BeliefPolicyTree.from_key(
                    key,
                    default_action_fn,
                    self.belief_lookup,
                )
            else:
                out[rid] = BeliefPolicyTree(default_action_fn=default_action_fn)
        return out

    def _sample_from_dist(self, dist: Dict[PolicyKey, float]) -> PolicyKey:
        if not dist:
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
    def _normalize(dist: Dict[PolicyKey, float]) -> Dict[PolicyKey, float]:
        total = sum(max(0.0, p) for p in dist.values())
        if total <= 0:
            n = max(len(dist), 1)
            return {k: 1.0 / n for k in dist}
        return {k: max(0.0, p) / total for k, p in dist.items()}

    def _greedy_policy_from_tree(self) -> BeliefPolicyTree:
        policy = BeliefPolicyTree(default_action_fn=self.default_action_fn)
        self._fill_greedy_policy(self.root, policy)
        return policy

    def _fill_greedy_policy(self, node: BeliefNode, policy: BeliefPolicyTree) -> None:
        if not node.actions:
            return
        best_edge = max(node.actions.values(), key=lambda e: e.q())
        policy.set_action(node.node_key, node.belief_key, node.belief, best_edge.action)
        for child in best_edge.obs_children.values():
            self._fill_greedy_policy(child, policy)

    def _collect_nodes(
        self,
        node: BeliefNode,
        out: List[BeliefNode],
        seen: Optional[set] = None,
    ) -> None:
        seen = seen if seen is not None else set()
        if id(node) in seen:
            return
        seen.add(id(node))
        out.append(node)
        for edge in node.actions.values():
            for child in edge.obs_children.values():
                self._collect_nodes(child, out, seen)

    def _collect_edges(
        self,
        node: BeliefNode,
        out: List[BeliefActionEdge],
        seen_nodes: Optional[set] = None,
        seen_edges: Optional[set] = None,
    ) -> None:
        seen_nodes = seen_nodes if seen_nodes is not None else set()
        seen_edges = seen_edges if seen_edges is not None else set()
        if id(node) in seen_nodes:
            return
        seen_nodes.add(id(node))
        for edge in node.actions.values():
            if id(edge) not in seen_edges:
                out.append(edge)
                seen_edges.add(id(edge))
            for child in edge.obs_children.values():
                self._collect_edges(child, out, seen_nodes, seen_edges)


class BeliefObsDecMCTSTeam:
    def __init__(self, planners: Dict[RobotID, BeliefObsDecMCTS]):
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

    def best_policies(self) -> Dict[RobotID, BeliefPolicyTree]:
        return {rid: planner.best_policy() for rid, planner in self.planners.items()}

    def best_actions(
        self,
        beliefs: Optional[Dict[RobotID, Belief]] = None,
        source: str = "policy",
    ) -> Dict[RobotID, Action]:
        beliefs = beliefs or {
            rid: planner.root_belief
            for rid, planner in self.planners.items()
        }
        return {
            rid: planner.best_action(beliefs.get(rid), source=source)
            for rid, planner in self.planners.items()
        }

    def entropies(self) -> Dict[RobotID, float]:
        return {
            rid: -sum(p * math.log(p) for p in planner.q.values() if p > 0)
            for rid, planner in self.planners.items()
        }


def _install_core_profiling_wrappers() -> None:
    planner_methods = [
        "iterate",
        "best_policy",
        "best_action",
        "_policy_from_forced_root_action",
        "_grow_tree_once",
        "_simulate_from_node",
        "_select_or_expand_action",
        "_rollout",
        "_score_policy_key",
        "_update_sample_space",
        "_update_distribution",
        "_estimate_expectation",
        "_eval_joint_policies",
        "_sample_own_policy",
        "_sample_other_policies",
        "_sample_from_dist",
        "_greedy_policy_from_tree",
        "_fill_greedy_policy",
        "_collect_nodes",
        "_collect_edges",
        "_update_belief_cached",
        "_get_or_create_node",
        "_belief_key",
    ]
    for name in planner_methods:
        fn = getattr(BeliefObsDecMCTS, name, None)
        if fn is not None:
            setattr(
                BeliefObsDecMCTS,
                name,
                _profiled_method(f"belief_core.BeliefObsDecMCTS.{name}", fn),
            )

    team_methods = [
        "iterate_and_communicate",
        "best_policies",
        "best_actions",
        "entropies",
    ]
    for name in team_methods:
        fn = getattr(BeliefObsDecMCTSTeam, name, None)
        if fn is not None:
            setattr(
                BeliefObsDecMCTSTeam,
                name,
                _profiled_method(f"belief_core.BeliefObsDecMCTSTeam.{name}", fn),
            )


_install_core_profiling_wrappers()
