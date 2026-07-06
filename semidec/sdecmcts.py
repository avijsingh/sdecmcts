from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple

from .comm_state import CommState, SemiMarkovCommModel


Action = Any
Belief = Sequence[float]
History = Tuple[Tuple[Action, Any], ...]
JointAction = Tuple[Action, ...]
JointObs = Any
LocalInfoState = Tuple[int, History]
RobotID = Hashable
State = Any


@dataclass(frozen=True)
class StepResult:
    next_state: State
    joint_obs: JointObs
    reward: float


class LocalPolicy:
    """Executable local policy extracted from a centralized search tree."""

    def __init__(
        self,
        table: Optional[Mapping[LocalInfoState, Action]] = None,
        default_action: Optional[Action] = None,
    ):
        self.table = dict(table or {})
        self.default_action = default_action

    def action(self, depth: int, history: History) -> Action:
        key = (depth, history)
        if key in self.table:
            return self.table[key]
        if self.default_action is not None:
            if callable(self.default_action):
                return self.default_action(depth, history)
            return self.default_action
        raise KeyError(f"No action for local information state {key!r}.")


class SemiDecPolicy:
    """Factored policy, one local policy per robot."""

    def __init__(self, policies: Mapping[RobotID, LocalPolicy]):
        self.policies = dict(policies)

    def action(self, robot_id: RobotID, depth: int, history: History) -> Action:
        return self.policies[robot_id].action(depth, history)

    def joint_action_from_histories(
        self,
        robot_ids: Sequence[RobotID],
        depth: int,
        histories: Mapping[RobotID, History],
    ) -> Dict[RobotID, Action]:
        return {
            rid: self.action(rid, depth, histories[rid])
            for rid in robot_ids
        }


class ActionStats:
    def __init__(self, joint_action: JointAction):
        self.joint_action = joint_action
        self.visits = 0
        self.value_sum = 0.0
        self.obs_children: Dict[JointObs, BeliefNode] = {}

    def q(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


class BeliefNode:
    def __init__(
        self,
        belief: Belief,
        depth: int,
        histories: Mapping[RobotID, History],
        comm_state: CommState,
        legal_joint_actions: Sequence[JointAction],
    ):
        self.belief = list(belief)
        self.depth = depth
        self.histories = dict(histories)
        self.comm_state = comm_state
        self.legal_joint_actions = list(legal_joint_actions)
        self.untried_joint_actions = list(legal_joint_actions)
        self.actions: Dict[JointAction, ActionStats] = {}
        self.visits = 0
        self.value_sum = 0.0

    def q(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


class SDecMCTS:
    """
    Centralized Planning, Semi-Decentralized Execution MCTS.

    The inner search is centralized over joint actions and joint observations
    from a synchronized belief. Extraction projects the visited centralized
    tree into factored local policies indexed by each agent's private history.
    """

    def __init__(
        self,
        robot_ids: Sequence[RobotID],
        root_belief: Belief,
        model: Any,
        comm_model: SemiMarkovCommModel,
        *,
        horizon: int,
        gamma: float = 1.0,
        cp: float = 1.0,
        rollout_policy: Optional[Any] = None,
        leaf_value_fn: Optional[Any] = None,
        comm_transition_fn: Optional[Any] = None,
        seed: Optional[int] = None,
        max_tree_depth: Optional[int] = None,
    ):
        self.robot_ids = list(robot_ids)
        if not self.robot_ids:
            raise ValueError("robot_ids cannot be empty.")
        self.root_belief = list(root_belief)
        self.model = model
        self.comm_model = comm_model
        self.horizon = horizon
        self.gamma = gamma
        self.cp = cp
        self.rollout_policy = rollout_policy
        self.leaf_value_fn = leaf_value_fn
        self.comm_transition_fn = comm_transition_fn
        self.max_tree_depth = max_tree_depth
        self.rng = random.Random(seed)
        self.min_return = float("inf")
        self.max_return = float("-inf")

        root_comm_state = self.comm_model.initial_state()
        root_histories = {rid: tuple() for rid in self.robot_ids}
        self.root = BeliefNode(
            belief=self.root_belief,
            depth=0,
            histories=root_histories,
            comm_state=root_comm_state,
            legal_joint_actions=self._legal_joint_actions(self.root_belief, 0),
        )
        self._all_nodes: List[BeliefNode] = [self.root]

    def run(self, n_iter: int) -> None:
        for _ in range(n_iter):
            state = (
                None
                if hasattr(self.model, "sample_belief_step")
                else self.model.sample_state_from_belief(self.root_belief, self.rng)
            )
            visited_nodes: List[BeliefNode] = []
            visited_actions: List[ActionStats] = []
            total_return = self._simulate(
                self.root,
                state,
                visited_nodes,
                visited_actions,
            )
            self.min_return = min(self.min_return, total_return)
            self.max_return = max(self.max_return, total_return)
            for node in visited_nodes:
                node.visits += 1
                node.value_sum += total_return
            for edge in visited_actions:
                edge.visits += 1
                edge.value_sum += total_return

    def extract_policy(
        self,
        default_actions: Optional[Mapping[RobotID, Action]] = None,
        *,
        force_root_joint_action: bool = True,
        min_edge_visits: int = 1,
    ) -> SemiDecPolicy:
        decisions = self._collect_extraction_decisions(min_edge_visits)
        default_actions = dict(default_actions or self._root_default_actions())
        forced_assignment: Dict[Tuple[RobotID, LocalInfoState], Action] = {}
        if force_root_joint_action and self.root.actions:
            root_edge = max(self.root.actions.values(), key=lambda e: (e.q(), e.visits))
            for idx, rid in enumerate(self.robot_ids):
                forced_assignment[(rid, (0, tuple()))] = root_edge.joint_action[idx]
        if not decisions:
            return SemiDecPolicy({
                rid: LocalPolicy(default_action=default_actions.get(rid))
                for rid in self.robot_ids
            })

        candidate_actions: Dict[RobotID, Dict[LocalInfoState, List[Action]]] = {
            rid: {} for rid in self.robot_ids
        }
        for decision in decisions:
            for idx, rid in enumerate(self.robot_ids):
                local_state = decision["local_states"][rid]
                action = decision["joint_action"][idx]
                actions = candidate_actions[rid].setdefault(local_state, [])
                if action not in actions:
                    actions.append(action)

        policies = {}
        for rid in self.robot_ids:
            table = {}
            for local_state in candidate_actions[rid]:
                table[local_state] = default_actions.get(rid)
            policies[rid] = LocalPolicy(table, default_actions.get(rid))

        local_keys = [
            (rid, local_state)
            for rid in self.robot_ids
            for local_state in candidate_actions[rid]
            if (rid, local_state) not in forced_assignment
        ]

        max_assignments = 200_000
        total_assignments = 1
        for rid, local_state in local_keys:
            total_assignments *= len(candidate_actions[rid][local_state])
            if total_assignments > max_assignments:
                return self._extract_policy_greedy(decisions, candidate_actions, default_actions)

        best_score = float("-inf")
        best_assignment: Dict[Tuple[RobotID, LocalInfoState], Action] = {}
        domains = [
            candidate_actions[rid][local_state]
            for rid, local_state in local_keys
        ]
        for values in itertools.product(*domains):
            assignment = dict(forced_assignment)
            assignment.update(dict(zip(local_keys, values)))
            score = self._score_assignment(decisions, assignment)
            if score > best_score:
                best_score = score
                best_assignment = assignment

        return self._policy_from_assignment(best_assignment, candidate_actions, default_actions)

    def best_joint_action(self) -> Optional[Dict[RobotID, Action]]:
        if not self.root.actions:
            return None
        edge = max(self.root.actions.values(), key=lambda e: (e.q(), e.visits))
        return self._action_tuple_to_dict(edge.joint_action)

    def _simulate(
        self,
        node: BeliefNode,
        state: State,
        visited_nodes: List[BeliefNode],
        visited_actions: List[ActionStats],
    ) -> float:
        if node.depth >= self.horizon:
            return 0.0

        visited_nodes.append(node)
        edge = self._select_or_expand_action(node)
        visited_actions.append(edge)

        joint_action_obj = self.model.joint_action_from_dict(
            self._action_tuple_to_dict(edge.joint_action)
        )
        if hasattr(self.model, "sample_belief_step"):
            next_belief, joint_obs, immediate_reward = self.model.sample_belief_step(
                node.belief,
                joint_action_obj,
                self.rng,
            )
            next_state = None
        else:
            step = self.model.step(state, joint_action_obj, self.rng)
            next_belief = None
            joint_obs = step.joint_obs
            immediate_reward = step.reward
            next_state = step.next_state
        local_obs = tuple(self.model.split_obs(joint_obs))

        if joint_obs not in edge.obs_children:
            child = self._make_child(
                node,
                edge.joint_action,
                joint_action_obj,
                joint_obs,
                local_obs,
                next_belief=next_belief,
            )
            edge.obs_children[joint_obs] = child
            self._all_nodes.append(child)
            future = self._rollout(next_state, child)
        else:
            child = edge.obs_children[joint_obs]
            at_depth_cap = (
                self.max_tree_depth is not None
                and child.depth >= self.max_tree_depth
            )
            if at_depth_cap:
                future = self._rollout(next_state, child)
            else:
                future = self._simulate(child, next_state, visited_nodes, visited_actions)

        return immediate_reward + self.gamma * future

    def _rollout(self, state: State, node: BeliefNode) -> float:
        if self.leaf_value_fn is not None:
            return float(self.leaf_value_fn(node.belief, node.depth))

        total = 0.0
        discount = 1.0
        belief = list(node.belief)
        depth = node.depth
        while depth < self.horizon:
            joint_action = self._rollout_joint_action(belief, depth)
            joint_action_obj = self.model.joint_action_from_dict(
                self._action_tuple_to_dict(joint_action)
            )
            if hasattr(self.model, "sample_belief_step"):
                next_belief, _joint_obs, reward = self.model.sample_belief_step(
                    belief,
                    joint_action_obj,
                    self.rng,
                )
                belief = list(next_belief)
            else:
                step = self.model.step(state, joint_action_obj, self.rng)
                reward = step.reward
                belief = list(self.model.update_joint_belief(belief, joint_action_obj, step.joint_obs))
                state = step.next_state
            total += discount * reward
            discount *= self.gamma
            depth += 1
        return total

    def _select_or_expand_action(self, node: BeliefNode) -> ActionStats:
        if node.untried_joint_actions:
            joint_action = self.rng.choice(node.untried_joint_actions)
            node.untried_joint_actions.remove(joint_action)
            edge = ActionStats(joint_action)
            node.actions[joint_action] = edge
            return edge
        return max(node.actions.values(), key=lambda edge: self._ucb(node, edge))

    def _ucb(self, node: BeliefNode, edge: ActionStats) -> float:
        if edge.visits <= 0:
            return float("inf")
        q = edge.q()
        if self.max_return > self.min_return:
            q = (q - self.min_return) / (self.max_return - self.min_return)
        parent_visits = max(node.visits, 1)
        return q + self.cp * math.sqrt(math.log(parent_visits + 1) / edge.visits)

    def _make_child(
        self,
        node: BeliefNode,
        joint_action: JointAction,
        joint_action_obj: Any,
        joint_obs: JointObs,
        local_obs: Sequence[Any],
        next_belief: Optional[Belief] = None,
    ) -> BeliefNode:
        if next_belief is None:
            next_belief = list(self.model.update_joint_belief(node.belief, joint_action_obj, joint_obs))
        else:
            next_belief = list(next_belief)
        next_histories = {}
        for idx, rid in enumerate(self.robot_ids):
            next_histories[rid] = node.histories[rid] + ((joint_action[idx], local_obs[idx]),)
        next_depth = node.depth + 1
        if self.comm_transition_fn is None:
            next_comm_state = self.comm_model.step(node.comm_state)
        else:
            next_comm_state = self.comm_transition_fn(
                node.comm_state,
                self._action_tuple_to_dict(joint_action),
                joint_action_obj,
                joint_obs,
                next_belief,
                next_depth,
            )
        was_isolated = not node.comm_state.is_fully_connected(self.robot_ids)
        now_connected = next_comm_state.is_fully_connected(self.robot_ids)
        if was_isolated and now_connected:
            next_histories = {rid: tuple() for rid in self.robot_ids}
        return BeliefNode(
            belief=next_belief,
            depth=next_depth,
            histories=next_histories,
            comm_state=next_comm_state,
            legal_joint_actions=self._legal_joint_actions(next_belief, next_depth),
        )

    def _legal_joint_actions(self, belief: Belief, depth: int) -> List[JointAction]:
        per_robot = [
            list(self.model.legal_actions(belief, rid, depth))
            for rid in self.robot_ids
        ]
        if any(not actions for actions in per_robot):
            return []
        return [tuple(actions) for actions in itertools.product(*per_robot)]

    def _rollout_joint_action(self, belief: Belief, depth: int) -> JointAction:
        if self.rollout_policy is not None:
            action_dict = self.rollout_policy(belief, depth, self.rng)
            return tuple(action_dict[rid] for rid in self.robot_ids)
        legal = self._legal_joint_actions(belief, depth)
        return self.rng.choice(legal)

    def _collect_extraction_decisions(self, min_edge_visits: int = 1) -> List[Dict[str, Any]]:
        decisions = []
        for node in self._all_nodes:
            if node.depth >= self.horizon:
                continue
            local_states = {
                rid: (node.depth, node.histories[rid])
                for rid in self.robot_ids
            }
            for edge in node.actions.values():
                if edge.visits < min_edge_visits:
                    continue
                decisions.append({
                    "weight": max(1, node.visits),
                    "local_states": local_states,
                    "joint_action": edge.joint_action,
                    "q": edge.q(),
                })
        return decisions

    def _score_assignment(
        self,
        decisions: Sequence[Mapping[str, Any]],
        assignment: Mapping[Tuple[RobotID, LocalInfoState], Action],
    ) -> float:
        score = 0.0
        for decision in decisions:
            compatible = True
            for idx, rid in enumerate(self.robot_ids):
                local_state = decision["local_states"][rid]
                if assignment[(rid, local_state)] != decision["joint_action"][idx]:
                    compatible = False
                    break
            if compatible:
                score += decision["weight"] * decision["q"]
        return score

    def _extract_policy_greedy(
        self,
        decisions: Sequence[Mapping[str, Any]],
        candidate_actions: Mapping[RobotID, Mapping[LocalInfoState, List[Action]]],
        default_actions: Mapping[RobotID, Action],
    ) -> SemiDecPolicy:
        assignment: Dict[Tuple[RobotID, LocalInfoState], Action] = {}
        if self.root.actions:
            root_edge = max(self.root.actions.values(), key=lambda e: (e.q(), e.visits))
            for idx, rid in enumerate(self.robot_ids):
                assignment[(rid, (0, tuple()))] = root_edge.joint_action[idx]
        for rid in self.robot_ids:
            for local_state, actions in candidate_actions[rid].items():
                if (rid, local_state) in assignment:
                    continue
                action_scores = {action: 0.0 for action in actions}
                for decision in decisions:
                    if decision["local_states"][rid] != local_state:
                        continue
                    idx = self.robot_ids.index(rid)
                    action_scores[decision["joint_action"][idx]] += (
                        decision["weight"] * decision["q"]
                    )
                assignment[(rid, local_state)] = max(action_scores, key=action_scores.get)
        return self._policy_from_assignment(assignment, candidate_actions, default_actions)

    def _policy_from_assignment(
        self,
        assignment: Mapping[Tuple[RobotID, LocalInfoState], Action],
        candidate_actions: Mapping[RobotID, Mapping[LocalInfoState, List[Action]]],
        default_actions: Mapping[RobotID, Action],
    ) -> SemiDecPolicy:
        policies = {}
        for rid in self.robot_ids:
            table = {
                local_state: assignment[(rid, local_state)]
                for local_state in candidate_actions[rid]
            }
            policies[rid] = LocalPolicy(table, default_actions.get(rid))
        return SemiDecPolicy(policies)

    def _root_default_actions(self) -> Dict[RobotID, Action]:
        defaults = {}
        for rid, action in self._action_tuple_to_dict(
            self.root.legal_joint_actions[0] if self.root.legal_joint_actions else tuple()
        ).items():
            defaults[rid] = action
        return defaults

    def _action_tuple_to_dict(self, joint_action: JointAction) -> Dict[RobotID, Action]:
        return {
            rid: joint_action[idx]
            for idx, rid in enumerate(self.robot_ids)
        }
