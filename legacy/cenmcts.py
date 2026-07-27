import math
import random

class CenMCTSNode:
    """Node in the joint centralized tree. Tracks which robot moves next."""

    def __init__(self, states, robot_paths, turn, parent=None, action=None):
        self.states      = states       # {rid: OrienteeringState}
        self.robot_paths = robot_paths  # {rid: [vertex, ...]}
        self.turn        = turn         # which robot acts next
        self.parent      = parent
        self.action      = action
        self.children    = []
        self.visits      = 0
        self.cum_reward  = 0.0

    def is_terminal(self):
        return all(s.is_terminal_state() for s in self.states.values())

    def is_fully_expanded(self):
        return len(self.children) == len(self.states[self.turn].get_legal_actions())

    def q(self):
        return self.cum_reward / self.visits if self.visits else 0.0


class CenMCTS:
    """
    Centralized MCTS baseline: single joint tree interleaving robot actions.
    Robot turns cycle 0, 1, ..., R-1, 0, 1, ... until all budgets exhausted.
    Upper bound reference for Dec-MCTS quality comparison.
    """

    def __init__(self, init_states, global_obj, starts,
                 exploration_const=math.sqrt(2), rollout_depth=50):
        robot_paths = {rid: [s.vertex] for rid, s in init_states.items()}
        self.robot_ids    = sorted(init_states.keys())
        self.root         = CenMCTSNode(init_states, robot_paths, turn=self.robot_ids[0])
        self.global_obj   = global_obj
        self.starts       = starts
        self.Cp           = exploration_const
        self.rollout_depth = rollout_depth

    def run(self, n_iter):
        for _ in range(n_iter):
            node   = self._select(self.root)
            child  = self._expand(node)
            reward = self._rollout(child)
            self._backprop(child, reward)
        return self._best_paths()

    def _select(self, node):
        while not node.is_terminal() and node.is_fully_expanded():
            if not node.children:
                break
            node = max(node.children, key=lambda c: (
                c.q() + self.Cp * math.sqrt(math.log(node.visits) / c.visits)
                if c.visits > 0 else float('inf')
            ))
        return node

    def _expand(self, node):
        if node.is_terminal():
            return node
        legal = node.states[node.turn].get_legal_actions()
        tried = {c.action for c in node.children}
        untried = [a for a in legal if a not in tried]
        if not untried:
            return node
        action = random.choice(untried)

        new_states = dict(node.states)
        new_states[node.turn] = node.states[node.turn].take_action(action)

        new_paths = dict(node.robot_paths)
        new_paths[node.turn] = node.robot_paths[node.turn] + [action]

        # Advance turn (skip robots that are terminal)
        R = len(self.robot_ids)
        turn_idx = self.robot_ids.index(node.turn)
        for _ in range(R):
            turn_idx = (turn_idx + 1) % R
            if not new_states[self.robot_ids[turn_idx]].is_terminal_state():
                break
        next_robot = self.robot_ids[turn_idx]

        child = CenMCTSNode(new_states, new_paths, next_robot,
                            parent=node, action=action)
        node.children.append(child)
        return child

    def _rollout(self, node):
        states = dict(node.states)
        paths  = dict(node.robot_paths)
        turn_idx = self.robot_ids.index(node.turn)
        depth = 0
        while depth < self.rollout_depth:
            active = [r for r in self.robot_ids
                      if not states[r].is_terminal_state()]
            if not active:
                break
            rid = active[turn_idx % len(active)]
            legal = states[rid].get_legal_actions()
            if not legal:
                turn_idx += 1
                depth += 1
                continue
            a = random.choice(legal)
            states[rid] = states[rid].take_action(a)
            paths[rid]  = paths[rid] + [a]
            turn_idx += 1
            depth += 1
        return self.global_obj(paths)

    def _backprop(self, node, reward):
        while node is not None:
            node.visits     += 1
            node.cum_reward += reward
            node = node.parent

    def _best_paths(self):
        """Greedy descent by Q-value to extract best joint plan."""
        node = self.root
        while node.children:
            visited = [c for c in node.children if c.visits > 0]
            if not visited:
                break
            node = max(visited, key=lambda c: c.q())
        return node.robot_paths