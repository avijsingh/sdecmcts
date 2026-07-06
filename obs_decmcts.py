from __future__ import annotations

import copy
import math
import random
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Hashable, List, Optional, Sequence, Tuple


Action = Any
Observation = Any
RobotID = Hashable
State = Any

# Local history is a tuple of (own_action, own_observation) pairs.
History = Tuple[Tuple[Action, Observation], ...]

# A partial policy tree maps local histories to actions.
PolicyKey = Tuple[Tuple[History, Action], ...]
JointPolicies = Dict[RobotID, "PolicyTree"]

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


class PolicyTree:
    """
    Observation-conditioned partial policy.

    table:
        local action-observation history -> action

    If a history is missing, default_action_fn is used.
    """

    def __init__(
        self,
        table: Optional[Dict[History, Action]] = None,
        default_action_fn: Optional[Callable[[History], Action]] = None,
    ):
        self.table: Dict[History, Action] = dict(table or {})
        self.default_action_fn = default_action_fn

    def action(self, history: History) -> Action:
        if history in self.table:
            return self.table[history]
        if self.default_action_fn is not None:
            return self.default_action_fn(history)
        raise KeyError(f"No action for history {history} and no default_action_fn.")

    def set_action(self, history: History, action: Action) -> None:
        self.table[history] = action

    def copy(self) -> "PolicyTree":
        return PolicyTree(dict(self.table), self.default_action_fn)

    def key(self) -> PolicyKey:
        return tuple(sorted(self.table.items(), key=lambda x: repr(x[0])))

    @staticmethod
    def from_key(
        key: PolicyKey,
        default_action_fn: Optional[Callable[[History], Action]] = None,
    ) -> "PolicyTree":
        return PolicyTree(dict(key), default_action_fn)


class ObsActionEdge:
    def __init__(self, action: Action):
        self.action = action

        self.visits = 0
        self.value_sum = 0.0

        self.disc_visits = 0.0
        self.disc_reward = 0.0

        # own observation -> ObsNode
        self.obs_children: Dict[Observation, ObsNode] = {}

    def q(self) -> float:
        return self.value_sum / self.visits if self.visits > 0 else 0.0

    def disc_q(self) -> float:
        return self.disc_reward / self.disc_visits if self.disc_visits > 0 else 0.0

    def update_discounted(self, reward: float, visited: bool, gamma: float) -> None:
        self.disc_visits = gamma * self.disc_visits + (1.0 if visited else 0.0)
        self.disc_reward = gamma * self.disc_reward + (reward if visited else 0.0)


class ObsNode:
    """
    Node indexed by local action-observation history.

    This is the key architectural change from vanilla Dec-MCTS:
        old node = own action prefix
        new node = own local history / observation-conditioned information state
    """

    def __init__(
        self,
        history: History,
        depth: int,
        legal_actions: Sequence[Action],
    ):
        self.history = history
        self.depth = depth
        self.legal_actions = list(legal_actions)

        self.actions: Dict[Action, ObsActionEdge] = {}
        self.untried_actions = list(legal_actions)

        self.visits = 0
        self.value_sum = 0.0

        self.representative_policy: Optional[PolicyKey] = None
        self.representative_reward: float = float("-inf")

        self.disc_visits = 0.0
        self.disc_reward = 0.0

    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0

    def add_action_edge(self, action: Action) -> ObsActionEdge:
        edge = ObsActionEdge(action)
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


class ObsDecMCTS:
    """
    Observation-conditioned Dec-MCTS variant.

    This is NOT vanilla Dec-MCTS.

    Vanilla Dec-MCTS:
        q is over action sequences.

    ObsDecMCTS:
        q is over partial policy trees:
            local action-observation history -> action.

    Required model interface
    ------------------------
    model.sample_state_from_belief(belief, rng) -> state
    model.step(state, joint_action, rng) -> StepResult
        where StepResult has next_state, joint_obs, reward
    model.split_obs(joint_obs) -> tuple/list of local observations
    model.joint_action_from_dict({rid: action}) -> joint_action

    You can adapt thin wrappers around your TigerModel to expose these methods.
    """

    def __init__(
        self,
        robot_id: RobotID,
        robot_ids: Sequence[RobotID],
        root_belief: Sequence[float],
        model: Any,
        legal_actions_fn: Callable[[History, int], Sequence[Action]],
        default_action_fn: Callable[[History], Action],
        default_action_fns_by_robot: Optional[
            Dict[RobotID, Callable[[History], Action]]
        ] = None,
        *,
        gamma: float = 1.0,
        cp: float = 1.0,
        horizon: int = 8,
        tau: int = 100,
        num_policies: int = 10,
        num_samples: int = 30,
        beta_init: float = 2.0,
        beta_decay: float = 0.995,
        alpha: float = 0.01,
        prefer_default_expansion: bool = False,
        seed: Optional[int] = None,
    ):
        self.robot_id = robot_id
        self.robot_ids = list(robot_ids)
        self.root_belief = list(root_belief)
        self.model = model

        self.legal_actions_fn = legal_actions_fn
        self.default_action_fn = default_action_fn
        self.default_action_fns_by_robot = dict(default_action_fns_by_robot or {})
        self.default_action_fns_by_robot.setdefault(
            self.robot_id,
            self.default_action_fn,
        )

        self.gamma = gamma
        self.cp = cp
        self.horizon = horizon
        self.tau = tau
        self.num_policies = num_policies
        self.num_samples = num_samples

        self.beta_init = beta_init
        self.beta = beta_init
        self.beta_decay = beta_decay
        self.alpha = alpha
        self.prefer_default_expansion = prefer_default_expansion

        self.rng = random.Random(seed)

        self.root = ObsNode(
            history=(),
            depth=0,
            legal_actions=self.legal_actions_fn((), 0),
        )
        # Teammate distributions over policy trees.
        self.received_dists: Dict[RobotID, Dict[PolicyKey, float]] = {
            rid: {} for rid in self.robot_ids if rid != self.robot_id
        }

        # Own sparse support and distribution over partial policy trees.
        self.X_hat: List[PolicyKey] = []
        self.q: Dict[PolicyKey, float] = {}
        self.sample_space_history: List[Dict[str, Any]] = []
        self.action_tiebreak_history: List[Dict[str, Any]] = []
        self.action_margin = 0.05

        self.min_reward = float("inf")
        self.max_reward = float("-inf")

        # REMOVE
        self.debug_root_teammate_actions: Dict[Action, int] = {}
        self.debug_root_joint_actions: Dict[Tuple[Action, Action], int] = {}

    def _default_action_fn_for_robot(
        self,
        robot_id: RobotID,
    ) -> Callable[[History], Action]:
        return self.default_action_fns_by_robot.get(
            robot_id,
            self.default_action_fn,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def iterate(self, n_outer: int = 1) -> None:
        """
        Paper-style outer schedule:
            1. update sample space from current tree
            2. grow tree tau times
            3. update q over the already-selected sample space
            4. cool beta

        New policies discovered during tree growth enter X_hat next iteration.
        """
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

    def best_policy(self) -> PolicyTree:
        if self.q:
            best_key = max(self.q, key=self.q.get)
            return PolicyTree.from_key(best_key, self.default_action_fn)

        return self._greedy_policy_from_tree()

    def _policy_from_forced_root_action(self, root_action: Action) -> PolicyTree:
        policy = PolicyTree(default_action_fn=self.default_action_fn)
        policy.set_action((), root_action)

        edge = self.root.actions.get(root_action)
        if edge is None:
            return policy

        for child in edge.obs_children.values():
            self._fill_greedy_policy(child, policy)

        return policy

    # def best_action(self, history: History = ()) -> Action:
    #     """
    #     Execute action at the current local history.
    #     Usually history=() at each fresh online replanning call.
    #     """
    #     return self.best_policy().action(history)
    def _find_node_for_history(self, node: ObsNode, history: History) -> Optional[ObsNode]:
        if node.history == history:
            return node

        for edge in node.actions.values():
            for child in edge.obs_children.values():
                found = self._find_node_for_history(child, history)
                if found is not None:
                    return found

        return None

    # def best_action(self, history: History = ()) -> Action:
    #     node = self._find_node_for_history(self.root, history)
    #     if node is not None and node.actions:
    #         return max(
    #             node.actions.values(),
    #             key=lambda e: e.q(),
    #         ).action

    #     return self.best_policy().action(history)

    def best_action(self, history: History = (), source: str = "policy_marginal") -> Action:
        """
        Execute using the single ObsDecMCTS extraction rule: marginal root/local
        action mass under q over observation-conditioned policy trees.
        """
        # TESTING: previous extraction alternatives are commented out while we
        # standardize on policy_marginal for all ObsDecMCTS benchmarks.
        #
        # if source == "policy":
        #     return self.best_policy().action(history)
        #
        node = self._find_node_for_history(self.root, history)
        if source == "visits" and node is not None and node.actions:
            return max(node.actions.values(), key=lambda e: e.visits).action

        # TESTING: remaining previous extraction alternatives stay commented
        # while visits is temporarily re-enabled as a diagnostic for Tiger.
        #
        # if node is not None and node.actions:
        #     if source == "tree":
        #         return max(node.actions.values(), key=lambda e: e.q()).action
        #
        #     if source == "disc_tree":
        #         return max(node.actions.values(), key=lambda e: e.disc_q()).action
        #
        #     if source == "visits":
        #         return max(node.actions.values(), key=lambda e: e.visits).action
        #
        #     if source == "policy_value":
        #         return self.best_policy_by_value().action(history)
        #
        #     raise ValueError(f"Unknown action source: {source}")
        #
        # return self.best_policy().action(history)
        return self._best_action_by_policy_marginal(history)

    def _best_action_by_policy_marginal(self, history: History = ()) -> Action:
        if not self.q:
            return self.best_policy().action(history)

        masses: Dict[Action, float] = {}
        for key, p in self.q.items():
            policy = PolicyTree.from_key(key, self.default_action_fn)
            action = policy.action(history)
            masses[action] = masses.get(action, 0.0) + p

        if not masses:
            return self.best_policy().action(history)

        ranked = sorted(masses.items(), key=lambda kv: kv[1], reverse=True)
        top_action, top_mass = ranked[0]
        second_mass = ranked[1][1] if len(ranked) > 1 else 0.0

        # TESTING: old direct marginal argmax, too sensitive when top masses are
        # separated only by Monte Carlo noise.
        #
        # return top_action
        if top_mass - second_mass >= self.action_margin:
            return top_action

        tied_actions = [
            action for action, mass in ranked
            if top_mass - mass <= self.action_margin
        ]
        chosen = self._break_action_mass_tie(history, tied_actions, top_action)
        self.action_tiebreak_history.append({
            "history": history,
            "masses": dict(masses),
            "candidates": list(tied_actions),
            "chosen": chosen,
        })
        return chosen

    def _break_action_mass_tie(
        self,
        history: History,
        tied_actions: Sequence[Action],
        fallback_action: Action,
    ) -> Action:
        node = self._find_node_for_history(self.root, history)
        if node is not None and node.actions:
            available = [
                action for action in tied_actions
                if action in node.actions
            ]
            if available:
                return max(
                    available,
                    key=lambda action: node.actions[action].q(),
                )

        return fallback_action

    # def best_action(self, history: History = (), source: str = "tree") -> Action:
    #     # return self.best_policy().action(history)
    #     return max(node.actions.values(), key=lambda e: e.q()).action

    # ------------------------------------------------------------------
    # Tree growth
    # ------------------------------------------------------------------

    # def _grow_tree_once(self) -> None:
    #     state = self.model.sample_state_from_belief(self.root_belief, self.rng)

    #     own_policy = PolicyTree(default_action_fn=self.default_action_fn)
    #     other_policies = self._sample_other_policies()

    #     visited_edges: List[ObsActionEdge] = []
    #     visited_nodes: List[ObsNode] = []

    #     total_return = self._simulate_from_node(
    #         node=self.root,
    #         state=state,
    #         histories={rid: () for rid in self.robot_ids},
    #         own_policy=own_policy,
    #         other_policies=other_policies,
    #         visited_nodes=visited_nodes,
    #         visited_edges=visited_edges,
    #         depth=0,
    #     )

    #     self.min_reward = min(self.min_reward, total_return)
    #     self.max_reward = max(self.max_reward, total_return)

    #     # Visited edges
    #     visited_edge_ids = {id(e) for e in visited_edges}
    #     all_edges: List[ObsActionEdge] = []
    #     self._collect_edges(self.root, all_edges)

    #     for edge in all_edges:
    #         on_path = id(edge) in visited_edge_ids

    #         if on_path:
    #             edge.visits += 1
    #             edge.value_sum += total_return
            
    #         edge.update_discounted(total_return, visited=on_path, gamma=self.gamma)

    #     # for node in visited_nodes:
    #     #     node.visits += 1
    #     #     node.value_sum += total_return

    #     #     if total_return > node.representative_reward:
    #     #         node.representative_reward = total_return
    #     #         node.representative_policy = own_policy.key()

    #     # visited_edge_ids = {id(e) for e in visited_edges}
    #     # all_edges: List[ObsActionEdge] = []
    #     # self._collect_edges(self.root, all_edges)

    #     # for edge in all_edges:
    #     #     on_path = id(edge) in visited_edge_ids
    #     #     if on_path:
    #     #         edge.visits += 1
    #     #         edge.value_sum += total_return
    #     #     edge.update_discounted(total_return, visited=on_path, gamma=self.gamma)

    #     visited_node_ids = {id(n) for n in visited_nodes}
    #     all_nodes: List[ObsNode] = []
    #     self._collect_nodes(self.root, all_nodes)

    #     for node in all_nodes:
    #         on_path = id(node) in visited_node_ids

    #         if on_path:
    #             node.visits += 1
    #             node.value_sum += total_return

    #             if total_return > node.representative_reward:
    #                 node.representative_reward = total_return
    #                 node.representative_policy = own_policy.key()

    #         node.update_discounted(total_return, visited=on_path, gamma=self.gamma)

    def _grow_tree_once(self) -> None:
        state = self.model.sample_state_from_belief(self.root_belief, self.rng)

        own_policy = PolicyTree(default_action_fn=self.default_action_fn)
        other_policies = self._sample_other_policies()

        visited_edges: List[ObsActionEdge] = []
        visited_nodes: List[ObsNode] = []

        total_return = self._simulate_from_node(
            node=self.root,
            state=state,
            histories={rid: () for rid in self.robot_ids},
            own_policy=own_policy,
            other_policies=other_policies,
            visited_nodes=visited_nodes,
            visited_edges=visited_edges,
            depth=0,
        )

        self.min_reward = min(self.min_reward, total_return)
        self.max_reward = max(self.max_reward, total_return)

        visited_edge_ids = {id(e) for e in visited_edges}
        all_edges: List[ObsActionEdge] = []
        self._collect_edges(self.root, all_edges)

        for edge in all_edges:
            on_path = id(edge) in visited_edge_ids

            if on_path:
                edge.visits += 1
                edge.value_sum += total_return

            edge.update_discounted(
                reward=total_return,
                visited=on_path,
                gamma=self.gamma,
            )

        # Update nodes once.
        visited_node_ids = {id(n) for n in visited_nodes}
        all_nodes: List[ObsNode] = []
        self._collect_nodes(self.root, all_nodes)

        for node in all_nodes:
            on_path = id(node) in visited_node_ids

            if on_path:
                node.visits += 1
                node.value_sum += total_return

                if total_return > node.representative_reward:
                    node.representative_reward = total_return
                    node.representative_policy = own_policy.key()

            node.update_discounted(
                reward=total_return,
                visited=on_path,
                gamma=self.gamma,
            )
    
    # REMOVE ONCE FIXED
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

    def _simulate_from_node(
        self,
        node: ObsNode,
        state: State,
        histories: Dict[RobotID, History],
        own_policy: PolicyTree,
        other_policies: Dict[RobotID, PolicyTree],
        visited_nodes: List[ObsNode],
        visited_edges: List[ObsActionEdge],
        depth: int,
    ) -> float:
        if depth >= self.horizon:
            return 0.0

        visited_nodes.append(node)

        action = self._select_or_expand_action(node)
        edge = node.actions[action]
        visited_edges.append(edge)
        own_policy.set_action(histories[self.robot_id], action)

        actions: Dict[RobotID, Action] = {}
        for rid in self.robot_ids:
            if rid == self.robot_id:
                actions[rid] = action
            else:
                actions[rid] = other_policies[rid].action(histories[rid])

        joint_action = self.model.joint_action_from_dict(actions)

        # REMOVE
        if depth == 0:
            other_id = next(rid for rid in self.robot_ids if rid != self.robot_id)

            other_action = actions[other_id]
            self.debug_root_teammate_actions[other_action] = (
                self.debug_root_teammate_actions.get(other_action, 0) + 1
            )

            own_action = actions[self.robot_id]
            pair = (own_action, other_action)
            self.debug_root_joint_actions[pair] = (
                self.debug_root_joint_actions.get(pair, 0) + 1
            )

            #

        step = self.model.step(state, joint_action, self.rng)
        local_obs_all = self.model.split_obs(step.joint_obs)

        next_histories = dict(histories)
        for idx, rid in enumerate(self.robot_ids):
            obs_i = local_obs_all[idx]
            act_i = actions[rid]
            next_histories[rid] = histories[rid] + ((act_i, obs_i),)

        own_obs = local_obs_all[self.robot_ids.index(self.robot_id)]

        if own_obs not in edge.obs_children:
            next_history = next_histories[self.robot_id]
            edge.obs_children[own_obs] = ObsNode(
                history=next_history,
                depth=depth + 1,
                legal_actions=self.legal_actions_fn(next_history, depth + 1),
            )
            # Roll out after first newly-created observation node.
            future = self._rollout(
                state=step.next_state,
                histories=next_histories,
                own_policy=own_policy,
                other_policies=other_policies,
                depth=depth + 1,
            )
        else:
            child = edge.obs_children[own_obs]
            future = self._simulate_from_node(
                node=child,
                state=step.next_state,
                histories=next_histories,
                own_policy=own_policy,
                other_policies=other_policies,
                visited_nodes=visited_nodes,
                visited_edges=visited_edges,
                depth=depth + 1,
            )

        return step.reward + self.gamma * future

    def _select_or_expand_action(self, node: ObsNode) -> Action:
        if node.untried_actions:
            # TESTING: previous behavior expanded a uniformly random untried
            # action. That is formally generic, but it wastes short online
            # budgets at newly reached observation nodes where the benchmark
            # already provides an observation-conditioned default policy.
            #
            # action = self.rng.choice(node.untried_actions)
            default_action = (
                self.default_action_fn(node.history)
                if self.prefer_default_expansion
                else None
            )
            if default_action is not None and default_action in node.untried_actions:
                action = default_action
            else:
                action = self.rng.choice(node.untried_actions)
            node.add_action_edge(action)
            return action

        return max(
            node.actions.values(),
            key=lambda edge: self._ucb(edge, node),
        ).action

    def _ucb(self, edge: ObsActionEdge, node: ObsNode) -> float:
        if edge.disc_visits <= 0:
            return float("inf")

        q = edge.disc_q()
        if self.max_reward > self.min_reward:
            q = (q - self.min_reward) / (self.max_reward - self.min_reward)
        else:
            q = 0.5

        # parent_count = max(node.visits, 1.0000001)
        parent_count = max(node.disc_visits, 1.0000001)
        bonus = self.cp * math.sqrt(math.log(parent_count) / edge.disc_visits)
        return q + bonus

    def _rollout(
        self,
        state: State,
        histories: Dict[RobotID, History],
        own_policy: PolicyTree,
        other_policies: Dict[RobotID, PolicyTree],
        depth: int,
    ) -> float:
        total = 0.0
        discount = 1.0

        for t in range(depth, self.horizon):
            actions: Dict[RobotID, Action] = {}

            for rid in self.robot_ids:
                if rid == self.robot_id:
                    a = self.default_action_fn(histories[rid])
                    own_policy.set_action(histories[rid], a)
                    actions[rid] = a
                else:
                    actions[rid] = other_policies[rid].action(histories[rid])

            joint_action = self.model.joint_action_from_dict(actions)
            step = self.model.step(state, joint_action, self.rng)
            local_obs_all = self.model.split_obs(step.joint_obs)

            total += discount * step.reward
            discount *= self.gamma

            next_histories = dict(histories)
            for idx, rid in enumerate(self.robot_ids):
                obs_i = local_obs_all[idx]
                act_i = actions[rid]
                next_histories[rid] = histories[rid] + ((act_i, obs_i),)

            histories = next_histories
            state = step.next_state

        return total
    
    def best_policy_by_value(self) -> PolicyTree:
        if not self.X_hat:
            return self._greedy_policy_from_tree()

        best_key = max(
            self.X_hat,
            key=lambda key: self._estimate_expectation(fixed_policy_key=key),
        )
        return PolicyTree.from_key(best_key, self.default_action_fn)

    # ------------------------------------------------------------------
    # Sparse policy distribution update
    # ------------------------------------------------------------------

    # def _update_sample_space(self) -> None:
    #     nodes: List[ObsNode] = []
    #     self._collect_nodes(self.root, nodes)

    #     candidates = [
    #         n for n in nodes
    #         if n.representative_policy is not None
    #     ]
    #     if not candidates:
    #         return

    #     # TODO: Check which implementation is correct
    #    # candidates.sort(key=lambda n: n.q(), reverse=True)
    #     candidates.sort(key=lambda n: n.disc_q(), reverse=True)
    #     # candidates.sort(key=lambda n: n.representative_reward, reverse=True)

    #     new_X_hat: List[PolicyKey] = []
    #     seen = set()

    #     for n in candidates:
    #         key = n.representative_policy
    #         if key is None or key in seen:
    #             continue
    #         new_X_hat.append(key)
    #         seen.add(key)

    #         if len(new_X_hat) >= self.num_policies:
    #             break

    #     if not new_X_hat:
    #         return

    #     if set(new_X_hat) != set(self.X_hat):
    #         self.X_hat = new_X_hat
    #         self.q = {key: 1.0 / len(new_X_hat) for key in new_X_hat}
    #         self.beta = self.beta_init
        
    #     # if set(new_X_hat) != set(self.X_hat):
    #     #     self.X_hat = new_X_hat

    #     #     if self.preserve_q_on_sample_space_change:
    #     #         fallback = 1.0 / len(new_X_hat)
    #     #         self.q = self._normalize({
    #     #             key: self.q.get(key, fallback)
    #     #             for key in new_X_hat
    #     #         })
    #     #     else:
    #     #         self.q = {key: 1.0 / len(new_X_hat) for key in new_X_hat}
    #     #         self.beta = self.beta_init

    # def _update_sample_space(self) -> None:
    #     """
    #     DEBUG / safer variant: REMOVE
    #     Build candidate policies from root action values instead of from
    #     max-sampled representative trajectories.
    #     """

    #     if not self.root.actions:
    #         return

    #     ranked_edges = sorted(
    #         self.root.actions.values(),
    #         key=lambda e: e.q(),
    #         reverse=True,
    #     )

    #     new_X_hat: List[PolicyKey] = []

    #     for edge in ranked_edges:
    #         policy = PolicyTree(default_action_fn=self.default_action_fn)
    #         policy.set_action((), edge.action)
    #         key = policy.key()
    #         new_X_hat.append(key)

    #         if len(new_X_hat) >= self.num_policies:
    #             break

    #     if not new_X_hat:
    #         return

    #     if set(new_X_hat) != set(self.X_hat):
    #         self.X_hat = new_X_hat
    #         self.q = {key: 1.0 / len(new_X_hat) for key in new_X_hat}
    #         self.beta = self.beta_init

    # def _update_sample_space(self) -> None:
    #     """
    #     Build one policy candidate per legal root action. REMOVE

    #     Each candidate forces a different root action, then fills the rest of the
    #     observation-conditioned policy greedily from the current tree.

    #     This avoids max-sampled representative trajectories while preserving
    #     non-root observation-conditioned policy structure.
    #     """
    #     if not self.root.actions:
    #         return

    #     new_X_hat: List[PolicyKey] = []
    #     seen = set()

    #     # for action in self.root.legal_actions:
    #     #     policy = self._policy_from_forced_root_action(action)
    #     #     key = policy.key()

    #     #     if key in seen:
    #     #         continue

    #     #     new_X_hat.append(key)
    #     #     seen.add(key)

    #     # REMOVE
    #     for action in self.root.legal_actions:
    #         policy = self._policy_from_forced_root_action(action)
    #         key = policy.key()

    #         if key in seen:
    #             continue

    #         new_X_hat.append(key)
    #         seen.add(key)

    #         if len(new_X_hat) >= self.num_policies:
    #             break

    #             if not new_X_hat:
    #                 return

    #             if set(new_X_hat) != set(self.X_hat):
    #                 self.X_hat = new_X_hat
    #                 self.q = {key: 1.0 / len(new_X_hat) for key in new_X_hat}
    #                 self.beta = self.beta_init
        
    #     # if set(new_X_hat) != set(self.X_hat):
    #     #     old_mass_by_root = {}

    #     #     for old_key, old_p in self.q.items():
    #     #         root_a = self._root_action_of_key(old_key)
    #     #         if root_a is not None:
    #     #             old_mass_by_root[root_a] = old_mass_by_root.get(root_a, 0.0) + old_p

    #     #     new_q = {}
    #     #     for new_key in new_X_hat:
    #     #         root_a = self._root_action_of_key(new_key)
    #     #         if root_a is None:
    #     #             new_q[new_key] = 1.0 / len(new_X_hat)
    #     #         else:
    #     #             new_q[new_key] = old_mass_by_root.get(root_a, 1.0 / len(new_X_hat))

    #     #     self.X_hat = new_X_hat
    #     #     self.q = self._normalize(new_q)

    # def _update_sample_space(self) -> None:
    #     """
    #     Generic sample-space update.

    #     Candidate policies come from:
    #     1. representative policies discovered during tree search
    #     2. root-action-forced policies from explored root edges

    #     We then rank all candidates by estimated/tree value and keep the top
    #     self.num_policies. This avoids domain-specific behavior and avoids
    #     arbitrary dependence on legal action ordering.
    #     """
    #     candidate_scores: Dict[PolicyKey, float] = {}

    #     # 1. Representative policies from tree nodes.
    #     nodes: List[ObsNode] = []
    #     self._collect_nodes(self.root, nodes)

    #     for node in nodes:
    #         key = node.representative_policy
    #         if key is None:
    #             continue

    #         # Prefer discounted value because your current tree uses disc stats.
    #         score = node.disc_q() if node.disc_visits > 0 else node.q()

    #         if key not in candidate_scores or score > candidate_scores[key]:
    #             candidate_scores[key] = score

    #     # 2. Root-action-forced policies from explored root actions.
    #     for edge in self.root.actions.values():
    #         policy = self._policy_from_forced_root_action(edge.action)
    #         key = policy.key()

    #         score = edge.disc_q() if edge.disc_visits > 0 else edge.q()

    #         if key not in candidate_scores or score > candidate_scores[key]:
    #             candidate_scores[key] = score

    #     if not candidate_scores:
    #         return

    #     ranked_keys = [
    #         key
    #         for key, _score in sorted(
    #             candidate_scores.items(),
    #             key=lambda kv: kv[1],
    #             reverse=True,
    #         )
    #     ]

    #     new_X_hat = ranked_keys[: self.num_policies]

    #     if not new_X_hat:
    #         return

    #     if set(new_X_hat) != set(self.X_hat):
    #         self.X_hat = new_X_hat
    #         self.q = {key: 1.0 / len(new_X_hat) for key in new_X_hat}
    #         self.beta = self.beta_init

    def _replace_sample_space_preserve_q(
        self,
        new_X_hat: List[PolicyKey],
        scores: Optional[Dict[PolicyKey, float]] = None,
    ) -> None:
        if not new_X_hat:
            return

        old_q = self.q
        old_support = set(self.X_hat)
        new_support = set(new_X_hat)

        # If support is unchanged, preserve q exactly.
        if new_support == old_support:
            self.X_hat = new_X_hat
            self.q = self._normalize({
                key: old_q.get(key, 0.0)
                for key in new_X_hat
            })
            return

        # TESTING: old behavior preserved surviving policy mass and gave newly
        # discovered policies a tiny prior. This can trap good new policies near
        # zero probability when X_hat changes late in a short-horizon solve.
        #
        # eps = 1e-3
        # new_q = {}
        #
        # for key in new_X_hat:
        #     if key in old_q:
        #         new_q[key] = old_q[key]
        #     else:
        #         # Small prior for newly discovered policies.
        #         new_q[key] = eps
        #
        # self.X_hat = new_X_hat
        # self.q = self._normalize(new_q)
        #
        # # Do not reset beta every time; changing support is normal.
        # # self.beta = self.beta_init

        # TESTING: simple preserve/uniform blend. This reduced lock-in in one
        # isolated case but made Tiger policy consistency worse overall.
        #
        # uniform = 1.0 / len(new_X_hat)
        # blend = 0.30
        # preserved_q = self._normalize({
        #     key: old_q.get(key, 0.0)
        #     for key in new_X_hat
        # })
        #
        # self.X_hat = new_X_hat
        # self.q = self._normalize({
        #     key: (1.0 - blend) * preserved_q.get(key, 0.0) + blend * uniform
        #     for key in new_X_hat
        # })

        value_q = self._value_initialized_q(new_X_hat, scores or {})
        preserved_q = self._normalize({
            key: old_q.get(key, 0.0)
            for key in new_X_hat
        })
        # TESTING: a 0.50 value blend fixed a targeted stale-q case but still
        # left too much room for stale support probabilities in Tiger batches.
        #
        # blend = 0.50
        blend = 0.80

        self.X_hat = new_X_hat
        self.q = self._normalize({
            key: (1.0 - blend) * preserved_q.get(key, 0.0)
            + blend * value_q.get(key, 0.0)
            for key in new_X_hat
        })
        self._record_sample_space_update(scores or {})

    def _value_initialized_q(
        self,
        keys: List[PolicyKey],
        scores: Dict[PolicyKey, float],
    ) -> Dict[PolicyKey, float]:
        if not keys:
            return {}
        if not scores:
            return {key: 1.0 / len(keys) for key in keys}

        vals = [scores.get(key, 0.0) for key in keys]
        lo = min(vals)
        hi = max(vals)
        if hi <= lo:
            return {key: 1.0 / len(keys) for key in keys}

        temperature = 0.25
        weights = {}
        for key in keys:
            normalized_score = (scores.get(key, lo) - lo) / (hi - lo)
            weights[key] = math.exp(normalized_score / temperature)

        return self._normalize(weights)
    
    # CHECK REMOVE
    def _score_policy_key(self, key: PolicyKey, n_eval: int = 5) -> float:
        total = 0.0

        for _ in range(n_eval):
            own_policy = PolicyTree.from_key(key, self.default_action_fn)
            other_policies = self._sample_other_policies()
            state = self.model.sample_state_from_belief(self.root_belief, self.rng)

            total += self._eval_joint_policies(
                state=state,
                own_policy=own_policy,
                other_policies=other_policies,
            )

        return total / n_eval
    ##

    def _update_sample_space(self) -> None:
        candidate_scores: Dict[PolicyKey, float] = {}

        nodes: List[ObsNode] = []
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
            (key, self._score_policy_key(key, n_eval=max(1, self.num_samples)))
            for key in candidate_scores.keys()
        ]

        scored.sort(key=lambda kv: kv[1], reverse=True)
        new_X_hat = [key for key, _score in scored[: self.num_policies]]

        if not new_X_hat:
            return

        self._replace_sample_space_preserve_q(new_X_hat, scores=dict(scored))
    
    def _root_action_of_key(self, key: PolicyKey) -> Optional[Action]:
        for hist, action in key:
            if hist == ():
                return action
        return None

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

            delta = self.alpha * q_val * (
                (norm_E - norm_E_given) / beta + H + ln_q
            )
            new_q[key] = max(1e-12, q_val - delta)

        self.q = self._normalize(new_q)

    def policy_distribution_debug(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []

        for key in self.X_hat:
            policy = PolicyTree.from_key(key, self.default_action_fn)
            root_action: Optional[Action]
            try:
                root_action = policy.action(())
            except Exception:
                root_action = None

            rows.append({
                "key": key,
                "root_action": root_action,
                "prob": self.q.get(key, 0.0),
                "num_entries": len(key),
            })

        rows.sort(key=lambda row: row["prob"], reverse=True)
        return rows

    def _record_sample_space_update(
        self,
        scores: Dict[PolicyKey, float],
    ) -> None:
        root_masses: Dict[Action, float] = {}
        top_policies = []

        for row in self.policy_distribution_debug():
            key = row["key"]
            root_action = row["root_action"]
            prob = row["prob"]

            if root_action is not None:
                root_masses[root_action] = root_masses.get(root_action, 0.0) + prob

            top_policies.append({
                "root_action": root_action,
                "prob": prob,
                "score": scores.get(key),
                "num_entries": row["num_entries"],
            })

        self.sample_space_history.append({
            "size": len(self.X_hat),
            "root_masses": root_masses,
            "top_policies": top_policies[:5],
        })

    def _estimate_expectation(
        self,
        fixed_policy_key: Optional[PolicyKey],
    ) -> float:
        total = 0.0

        for _ in range(self.num_samples):
            if fixed_policy_key is None:
                own_policy = self._sample_own_policy()
            else:
                own_policy = PolicyTree.from_key(
                    fixed_policy_key,
                    self.default_action_fn,
                )

            other_policies = self._sample_other_policies()
            state = self.model.sample_state_from_belief(self.root_belief, self.rng)

            total += self._eval_joint_policies(
                state=state,
                own_policy=own_policy,
                other_policies=other_policies,
            )

        return total / self.num_samples

    def _eval_joint_policies(
        self,
        state: State,
        own_policy: PolicyTree,
        other_policies: Dict[RobotID, PolicyTree],
    ) -> float:
        histories = {rid: () for rid in self.robot_ids}
        total = 0.0
        discount = 1.0

        for _ in range(self.horizon):
            actions: Dict[RobotID, Action] = {}

            for rid in self.robot_ids:
                if rid == self.robot_id:
                    actions[rid] = own_policy.action(histories[rid])
                else:
                    actions[rid] = other_policies[rid].action(histories[rid])

            joint_action = self.model.joint_action_from_dict(actions)
            step = self.model.step(state, joint_action, self.rng)
            local_obs_all = self.model.split_obs(step.joint_obs)

            total += discount * step.reward
            discount *= self.gamma

            next_histories = dict(histories)
            for idx, rid in enumerate(self.robot_ids):
                obs_i = local_obs_all[idx]
                act_i = actions[rid]
                next_histories[rid] = histories[rid] + ((act_i, obs_i),)

            histories = next_histories
            state = step.next_state

        return total

    # ------------------------------------------------------------------
    # Sampling helpers
    # ------------------------------------------------------------------

    def _sample_own_policy(self) -> PolicyTree:
        key = self._sample_from_dist(self.q)
        return PolicyTree.from_key(key, self.default_action_fn)

    def _sample_other_policies(self) -> Dict[RobotID, PolicyTree]:
        out = {}

        for rid in self.robot_ids:
            if rid == self.robot_id:
                continue

            dist = self.received_dists.get(rid, {})
            default_action_fn = self._default_action_fn_for_robot(rid)
            if dist:
                key = self._sample_from_dist(dist)
                out[rid] = PolicyTree.from_key(key, default_action_fn)
            else:
                out[rid] = PolicyTree(default_action_fn=default_action_fn)

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

    # ------------------------------------------------------------------
    # Tree extraction / traversal
    # ------------------------------------------------------------------

    def _greedy_policy_from_tree(self) -> PolicyTree:
        policy = PolicyTree(default_action_fn=self.default_action_fn)
        self._fill_greedy_policy(self.root, policy)
        return policy

    def _fill_greedy_policy(self, node: ObsNode, policy: PolicyTree) -> None:
        if not node.actions:
            return

        best_edge = max(
            node.actions.values(),
            key=lambda e: e.q(),
        )

        policy.set_action(node.history, best_edge.action)

        for child in best_edge.obs_children.values():
            self._fill_greedy_policy(child, policy)

    def _collect_nodes(self, node: ObsNode, out: List[ObsNode]) -> None:
        out.append(node)
        for edge in node.actions.values():
            for child in edge.obs_children.values():
                self._collect_nodes(child, out)

    def _collect_edges(self, node: ObsNode, out: List[ObsActionEdge]) -> None:
        for edge in node.actions.values():
            out.append(edge)
            for child in edge.obs_children.values():
                self._collect_edges(child, out)


class ObsDecMCTSTeam:
    def __init__(self, planners: Dict[RobotID, ObsDecMCTS]):
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

    def best_policies(self) -> Dict[RobotID, PolicyTree]:
        return {
            rid: planner.best_policy()
            for rid, planner in self.planners.items()
        }

    # def best_actions(self, histories: Optional[Dict[RobotID, History]] = None) -> Dict[RobotID, Action]:
    #     histories = histories or {rid: () for rid in self.planners}

    #     return {
    #         rid: planner.best_action(histories.get(rid, ()))
    #         for rid, planner in self.planners.items()
    #     }

    # REMOVE/CHANGE
    def best_actions(
        self,
        histories: Optional[Dict[RobotID, History]] = None,
        source: str = "tree",
    ) -> Dict[RobotID, Action]:
        histories = histories or {rid: () for rid in self.planners}

        return {
            rid: planner.best_action(histories.get(rid, ()), source=source)
            for rid, planner in self.planners.items()
        }

    def entropies(self) -> Dict[RobotID, float]:
        out = {}

        for rid, planner in self.planners.items():
            out[rid] = -sum(
                p * math.log(p)
                for p in planner.q.values()
                if p > 0
            )

        return out


def _install_core_profiling_wrappers() -> None:
    planner_methods = [
        "iterate",
        "best_policy",
        "best_policy_by_value",
        "best_action",
        "_find_node_for_history",
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
    ]
    for name in planner_methods:
        fn = getattr(ObsDecMCTS, name, None)
        if fn is not None:
            setattr(
                ObsDecMCTS,
                name,
                _profiled_method(f"obs_core.ObsDecMCTS.{name}", fn),
            )

    team_methods = [
        "iterate_and_communicate",
        "best_policies",
        "best_actions",
        "entropies",
    ]
    for name in team_methods:
        fn = getattr(ObsDecMCTSTeam, name, None)
        if fn is not None:
            setattr(
                ObsDecMCTSTeam,
                name,
                _profiled_method(f"obs_core.ObsDecMCTSTeam.{name}", fn),
            )


_install_core_profiling_wrappers()
