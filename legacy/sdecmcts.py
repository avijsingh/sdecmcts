"""
sdecmcts.py
-----------
SDecMCTS: Semi-Decentralized MCTS.

ONE joint tree whose nodes carry a communication partition.  Within that
single tree, each turn is handled by one of two expansion modes:

  CenMCTS mode  (node.turn is in a connected group, |group| > 1)
    Standard UCB over global joint reward.  The robot acts in coordination
    with its comm group — same logic as CenMCTS.

  DecMCTS mode  (node.turn is isolated, |group| == 1)
    Standard UCB, but the reward signal comes from the robot's local utility
    function (if provided) rather than the full joint reward.  The robot acts
    without coordination — same spirit as DecMCTS.

Both modes produce child nodes in the SAME tree.  No sub-planner objects,
no separate per-agent trees.

Partition transitions
---------------------
The comm partition is stable within a round (one full cycle through all
active robots).  At the end of each round, a new partition is sampled from
CommGraph and attached to the first node of the next round.

Execution
---------
  Full comm  →  joint_action() / best_paths()   (CenMCTS-style extraction)
  Isolated r →  best_action(r) / factored_policy(r)
                (marginalise visits over all nodes where r acted:
                 π_r(a) ∝ Σ{n : n.turn==r, child.action==a} child.visits)
"""

import math
import random


# ── COMMUNICATION GRAPH ───────────────────────────────────────────────────────

class CommGraph:
    """
    Stochastic communication topology for a robot team.

    Each pair of robots (i, j) has an independent link that is UP with
    probability p_link[i][j].  Connected components form the comm partition.

    Parameters
    ----------
    robot_ids     : ordered list of robot identifiers
    p_link        : float  — uniform link-up probability for every pair, OR
                   dict {(ri, rj): float} — per-pair probabilities
    init_partition: frozenset of frozensets — starting partition for the root.
                   If None, defaults to all robots in one group (full comm).
    """

    def __init__(self, robot_ids, p_link=0.8, init_partition=None):
        self.robot_ids = list(robot_ids)

        if isinstance(p_link, (int, float)):
            self._p = {
                (min(ri, rj), max(ri, rj)): float(p_link)
                for i, ri in enumerate(robot_ids)
                for j, rj in enumerate(robot_ids)
                if j > i
            }
        else:
            self._p = {
                (min(k[0], k[1]), max(k[0], k[1])): v
                for k, v in p_link.items()
            }

        if init_partition is None:
            self.init_partition = frozenset({frozenset(robot_ids)})
        else:
            self.init_partition = init_partition

    def sample_partition(self):
        """
        Sample link states independently then return connected components
        as a frozenset of frozensets.
        """
        parent = {rid: rid for rid in self.robot_ids}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[rx] = ry

        for i, ri in enumerate(self.robot_ids):
            for j, rj in enumerate(self.robot_ids):
                if j <= i:
                    continue
                key = (min(ri, rj), max(ri, rj))
                if random.random() < self._p.get(key, 0.0):
                    union(ri, rj)

        components = {}
        for rid in self.robot_ids:
            root = find(rid)
            components.setdefault(root, set()).add(rid)

        return frozenset(frozenset(c) for c in components.values())

    def link_prob(self, ri, rj):
        key = (min(ri, rj), max(ri, rj))
        return self._p.get(key, 0.0)


# ── JOINT TREE NODE ───────────────────────────────────────────────────────────

class SDecMCTSNode:
    """
    Node in the single joint tree.

    Each edge = one action taken by robot `turn`.
    The `partition` field records the comm state active at this node.

    Nodes whose active robot is in a connected group are "CenMCTS nodes".
    Nodes whose active robot is isolated are "DecMCTS nodes".
    Both live in the same tree and use the same UCB formula; the difference
    is which reward signal backs them up (joint vs. local).

    Fields
    ------
    states      : {robot_id: state}         — joint state at this node
    robot_paths : {robot_id: [vertex, ...]} — accumulated paths (incl. start)
    turn        : robot_id                  — which robot acts next
    partition   : frozenset of frozensets   — comm groups at this node
    parent      : SDecMCTSNode | None
    action      : action taken by parent.turn to reach this node
    """

    def __init__(self, states, robot_paths, turn, partition,
                 parent=None, action=None):
        self.states      = states
        self.robot_paths = robot_paths
        self.turn        = turn
        self.partition   = partition
        self.parent      = parent
        self.action      = action
        self.children    = []
        self.visits      = 0
        self.cum_reward  = 0.0

    # ── helpers ──────────────────────────────────────────────────────────────

    def is_terminal(self):
        return all(s.is_terminal_state() for s in self.states.values())

    def is_fully_expanded(self):
        return (len(self.children) ==
                len(self.states[self.turn].get_legal_actions()))

    def q(self):
        return self.cum_reward / self.visits if self.visits else 0.0

    def ucb(self, Cp):
        if self.visits == 0:
            return float("inf")
        if self.parent is None or self.parent.visits == 0:
            return float("inf")
        return self.q() + Cp * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )

    # ── comm-mode queries ─────────────────────────────────────────────────────

    def comm_group(self):
        """Return the comm group that contains self.turn."""
        for group in self.partition:
            if self.turn in group:
                return group
        return frozenset({self.turn})   # fallback: treat as isolated

    def is_in_comm(self):
        """True if the active robot is in a connected group (CenMCTS mode)."""
        return len(self.comm_group()) > 1

    def is_isolated(self):
        """True if the active robot has no communication (DecMCTS mode)."""
        return not self.is_in_comm()


# ── SDECMCTS PLANNER ──────────────────────────────────────────────────────────

class SDecMCTS:
    """
    Single-tree semi-decentralized MCTS.

    The tree contains both CenMCTS-mode nodes (connected robots, joint UCB)
    and DecMCTS-mode nodes (isolated robots, local UCB).  The comm partition
    transitions stochastically at each round boundary via CommGraph.

    Parameters
    ----------
    robot_ids     : ordered list of robot identifiers
    init_states   : dict {robot_id: state}
    global_obj    : callable dict{robot_id: path} -> float
                   Used for CenMCTS-mode backprop and as fallback.
    local_obj_fns : dict {robot_id: callable}  (optional)
                   f(joint_seqs) -> float, used for DecMCTS-mode backprop.
                   If omitted, global_obj is used for all nodes.
    comm_graph    : CommGraph — drives partition transitions.
                   Defaults to CommGraph(robot_ids, p_link=1.0).
    Cp            : UCB exploration constant (default: sqrt(2))
    rollout_depth : max steps in the random rollout
    """

    def __init__(
        self,
        robot_ids,
        init_states,
        global_obj,
        local_obj_fns=None,
        comm_graph=None,
        Cp=math.sqrt(2),
        rollout_depth=50,
    ):
        self.robot_ids     = list(robot_ids)
        self.global_obj    = global_obj
        self.local_obj_fns = local_obj_fns or {}
        self.comm_graph    = comm_graph or CommGraph(robot_ids, p_link=1.0)
        self.Cp            = Cp
        self.rollout_depth = rollout_depth

        robot_paths = {rid: [s.vertex] for rid, s in init_states.items()}
        self.root = SDecMCTSNode(
            states      = init_states,
            robot_paths = robot_paths,
            turn        = self.robot_ids[0],
            partition   = self.comm_graph.init_partition,
        )

    # ── PUBLIC API ────────────────────────────────────────────────────────────

    def run(self, n_iter):
        """Run n_iter MCTS iterations (select → expand → rollout → backprop)."""
        for _ in range(n_iter):
            node   = self._select(self.root)
            child  = self._expand(node)
            reward = self._rollout(child)
            self._backprop(child, reward)

    def best_paths(self):
        """
        Full-comm execution: greedy Q-descent through joint tree.
        Returns {robot_id: [vertex, ...]} including the starting vertex.
        """
        node = self.root
        while node.children:
            visited = [c for c in node.children if c.visits > 0]
            if not visited:
                break
            node = max(visited, key=lambda c: c.q())
        return {rid: list(node.robot_paths[rid]) for rid in self.robot_ids}

    def factored_policy(self, robot_id):
        """
        Isolated execution: per-agent marginal policy from the joint tree.

        Aggregates visit counts across all nodes where robot_id was active:
            π_r(a) ∝ Σ{n : n.turn == robot_id, child.action == a} child.visits

        Falls back to uniform over root legal actions if unexplored.
        """
        action_counts = {}

        def dfs(node):
            if node.turn == robot_id:
                for child in node.children:
                    if child.visits > 0:
                        a = child.action
                        action_counts[a] = action_counts.get(a, 0) + child.visits
            for child in node.children:
                dfs(child)

        dfs(self.root)

        total = sum(action_counts.values())
        if total == 0:
            legal = self.root.states[robot_id].get_legal_actions()
            n = len(legal)
            return {a: 1.0 / n for a in legal} if n else {}
        return {a: v / total for a, v in action_counts.items()}

    def best_action(self, robot_id):
        """Best first action for robot_id when isolated (DecMCTS mode)."""
        policy = self.factored_policy(robot_id)
        return max(policy, key=policy.get) if policy else None

    def joint_action(self):
        """
        Best joint first action for all robots (full-comm / CenMCTS mode).
        Descends the greedy Q path collecting one action per robot.
        """
        actions = {}
        node    = self.root
        seen    = set()
        while node.children and len(seen) < len(self.robot_ids):
            visited = [c for c in node.children if c.visits > 0]
            if not visited:
                break
            best = max(visited, key=lambda c: c.q())
            if node.turn not in seen:
                actions[node.turn] = best.action
                seen.add(node.turn)
            node = best
        return actions

    # ── SELECTION ─────────────────────────────────────────────────────────────

    def _select(self, node):
        while not node.is_terminal() and node.is_fully_expanded():
            if not node.children:
                break
            node = max(node.children, key=lambda c: c.ucb(self.Cp))
        return node

    # ── EXPANSION ─────────────────────────────────────────────────────────────

    def _expand(self, node):
        """
        Expand one untried action for node.turn.

        CenMCTS mode (in comm): action selected among untried, backed by
          global_obj reward — robot coordinates within its group.

        DecMCTS mode (isolated): same selection logic, but the reward signal
          in backprop will come from local_obj_fns[turn] if available.

        Partition transitions: at round boundaries (turn wraps back to the
          first robot), a new partition is sampled from CommGraph and
          attached to the child — this is where comm topology changes.
        """
        if node.is_terminal():
            return node

        legal   = node.states[node.turn].get_legal_actions()
        tried   = {c.action for c in node.children}
        untried = [a for a in legal if a not in tried]
        if not untried:
            return node

        action = random.choice(untried)

        new_states            = dict(node.states)
        new_states[node.turn] = node.states[node.turn].take_action(action)

        new_paths             = dict(node.robot_paths)
        new_paths[node.turn]  = node.robot_paths[node.turn] + [action]

        # Advance turn (skip terminal robots)
        R        = len(self.robot_ids)
        turn_idx = self.robot_ids.index(node.turn)
        next_idx = turn_idx
        for _ in range(R):
            next_idx = (next_idx + 1) % R
            if not new_states[self.robot_ids[next_idx]].is_terminal_state():
                break
        next_robot = self.robot_ids[next_idx]

        # Sample new comm partition at round boundaries (turn wraps around).
        # Within a round, robots stay in the same partition — same comm state
        # for the whole coordinated/isolated episode.
        if next_idx <= turn_idx:
            new_partition = self.comm_graph.sample_partition()
        else:
            new_partition = node.partition

        child = SDecMCTSNode(
            states      = new_states,
            robot_paths = new_paths,
            turn        = next_robot,
            partition   = new_partition,
            parent      = node,
            action      = action,
        )
        node.children.append(child)
        return child

    # ── ROLLOUT ───────────────────────────────────────────────────────────────

    def _rollout(self, node):
        """
        Random rollout that respects the comm partition.

        CenMCTS-mode robots (connected group): act in random turn-by-turn
          order within the group — they have access to joint state.

        DecMCTS-mode robots (isolated): act independently — they can only
          use their own local state (same random policy here, but the
          partition boundary is the hook for learned policies in SDecZero).

        Partition is refreshed at each round boundary.
        """
        states    = dict(node.states)
        paths     = dict(node.robot_paths)
        partition = node.partition
        depth     = 0

        while depth < self.rollout_depth:
            active = [r for r in self.robot_ids
                      if not states[r].is_terminal_state()]
            if not active:
                break

            # One full round: iterate over comm groups in partition order.
            # CenMCTS groups act in sorted turn order (coordinated).
            # Isolated robots act independently.
            for group in sorted(partition, key=lambda g: min(g)):
                for rid in sorted(group):
                    if states[rid].is_terminal_state():
                        continue
                    legal = states[rid].get_legal_actions()
                    if legal:
                        a           = random.choice(legal)
                        states[rid] = states[rid].take_action(a)
                        paths[rid]  = paths[rid] + [a]

            # Sample new partition for the next round
            partition = self.comm_graph.sample_partition()
            depth    += 1

        return self.global_obj(paths)

    # ── BACKPROPAGATION ───────────────────────────────────────────────────────

    def _backprop(self, node, reward):
        """
        Backpropagate reward through the path to the root.

        For DecMCTS-mode nodes (isolated robot), the reward is re-evaluated
        using the robot's local_obj_fn if available.  This gives isolated
        robots a reward signal grounded in their own marginal utility rather
        than the full joint reward — matching the DecMCTS philosophy.

        For CenMCTS-mode nodes (connected group), the full joint reward is
        used unchanged.
        """
        while node is not None:
            node.visits += 1
            if node.is_isolated() and node.turn in self.local_obj_fns:
                # DecMCTS mode: credit local utility to this node
                local_r = self.local_obj_fns[node.turn](node.robot_paths)
                node.cum_reward += local_r
            else:
                # CenMCTS mode: credit full joint reward
                node.cum_reward += reward
            node = node.parent
