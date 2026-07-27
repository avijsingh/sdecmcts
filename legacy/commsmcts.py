"""
commsmcts.py
------------
Semi-decentralized MCTS for cooperative multi-robot teams with intermittent
communication.

Communication regimes
---------------------
  FULL    — instant, lossless comms → CenMCTS-style joint planning
              (one robot acts per level, round-robin; global team reward)
  DELAYED — comms with stale-message lag → DecMCTS-style planning biased
              by the last-received per-robot distributions
  NONE    — no communication → DecMCTS-style fully independent planning

Architecture
------------
A single tree is rooted at the shared initial joint state.  Each node
carries a CommRegime tag that controls:
  • how children are created     (single-robot edge vs. joint-action edge)
  • which UCB formula is used    (standard UCB1 vs. D-UCT)
  • how the reward is computed   (global_obj vs. sum of local objectives)

Communication regime transitions are sampled from a CommModel (Markov chain)
during both expansion and rollout, so every simulated trajectory experiences
a realistic mixture of conditions.

Interface contract for state objects
-------------------------------------
  state.get_legal_actions()  -> list of actions (vertex ints in the test)
  state.take_action(action)  -> new state
  state.is_terminal_state()  -> bool
  state.vertex               -> int  (start vertex, used to seed robot_paths)
"""

import math
import random
import copy
from enum import Enum


# ─────────────────────────────────────────────────────────
# PART 1 — COMMUNICATION REGIME
# ─────────────────────────────────────────────────────────

class CommRegime(Enum):
    FULL    = "full"     # instant, lossless comms
    DELAYED = "delayed"  # comms with stale-message lag
    NONE    = "none"     # no communication


# ─────────────────────────────────────────────────────────
# PART 2 — COMMUNICATION MODEL
# ─────────────────────────────────────────────────────────

class CommModel:
    """
    Markov model of communication regime transitions and message-delay
    distribution.

    Parameters
    ----------
    transition : dict[CommRegime -> dict[CommRegime -> float]]
        P(next_regime | curr_regime). Each row must sum to 1.
        If None, a sensible default is used (mostly FULL, with occasional
        degradation and recovery).

    delay_distribution : dict[int -> float]
        P(message_lag = k steps) when regime is DELAYED.
        Default: {1: 1.0}  (messages are exactly one step stale).

    initial_regime : CommRegime
        Regime assigned to the root node.
    """

    _DEFAULT_TRANSITION = {
        CommRegime.FULL: {
            CommRegime.FULL:    0.7,
            CommRegime.DELAYED: 0.2,
            CommRegime.NONE:    0.1,
        },
        CommRegime.DELAYED: {
            CommRegime.FULL:    0.4,
            CommRegime.DELAYED: 0.4,
            CommRegime.NONE:    0.2,
        },
        CommRegime.NONE: {
            CommRegime.FULL:    0.2,
            CommRegime.DELAYED: 0.3,
            CommRegime.NONE:    0.5,
        },
    }

    def __init__(
        self,
        transition=None,
        delay_distribution=None,
        initial_regime=CommRegime.FULL,
    ):
        self.transition     = transition or self._DEFAULT_TRANSITION
        self.delay_dist     = delay_distribution or {1: 1.0}
        self.initial_regime = initial_regime

    def sample_next_regime(self, current):
        """Sample the next regime from the Markov row for `current`."""
        row = self.transition[current]
        r   = random.random()
        cum = 0.0
        for regime, p in row.items():
            cum += p
            if r <= cum:
                return regime
        return list(row.keys())[-1]

    def sample_delay(self):
        """Sample a message-lag (number of steps) for the DELAYED regime."""
        r   = random.random()
        cum = 0.0
        for steps, p in self.delay_dist.items():
            cum += p
            if r <= cum:
                return steps
        return max(self.delay_dist.keys())


# ─────────────────────────────────────────────────────────
# PART 3 — UNIFIED TREE NODE
# ─────────────────────────────────────────────────────────

class CommMCTSNode:
    """
    Node in the unified semi-decentralized MCTS tree.

    FULL regime
    -----------
    • `action` = int    (destination vertex for robot `turn`)
    • `turn`   = robot_id of the robot that acts to produce this node's children
    • Selection: standard UCB1

    NONE / DELAYED regime
    ---------------------
    • `action` = dict {robot_id: vertex}   (simultaneous joint action)
    • `turn`   = None
    • Selection: D-UCT (discounted visit / reward statistics), matching
      the D-UCT formula in DecMCTS

    Parameters
    ----------
    joint_states  : dict {robot_id: state}
    robot_paths   : dict {robot_id: [vertex, ...]}
    regime        : CommRegime
    turn          : robot_id whose turn it is (FULL only; None otherwise)
    stale_dists   : dict {robot_id: {seq_tuple: prob}}
                    Snapshot of per-robot distributions at the moment this
                    node was created.  Used in DELAYED mode to guide action
                    sampling.  Never mutated after creation.
    parent        : CommMCTSNode or None
    action        : int (FULL) or dict[robot_id, vertex] (NONE / DELAYED)
    """

    def __init__(
        self,
        joint_states,
        robot_paths,
        regime,
        turn=None,
        stale_dists=None,
        parent=None,
        action=None,
    ):
        self.joint_states = joint_states
        self.robot_paths  = robot_paths
        self.regime       = regime
        self.turn         = turn
        self.stale_dists  = stale_dists if stale_dists is not None else {}
        self.parent       = parent
        self.action       = action
        self.children     = []

        # Standard (undiscounted) statistics — used for final plan extraction
        self.visits     = 0
        self.cum_reward = 0.0

        # D-UCT discounted statistics — used for selection in NONE/DELAYED nodes
        self.disc_visits = 0.0
        self.disc_reward = 0.0

    # ── Structural queries ───────────────────────────────────────────────────

    def is_terminal(self):
        return all(s.is_terminal_state() for s in self.joint_states.values())

    def is_fully_expanded(self, max_dec_children):
        """
        FULL: fully expanded when all legal actions for `turn` have children.
        NONE/DELAYED: capped at max_dec_children (joint action space is huge).
        """
        if self.regime == CommRegime.FULL:
            if self.turn is None:
                return True
            legal = self.joint_states[self.turn].get_legal_actions()
            tried = {c.action for c in self.children}
            return all(a in tried for a in legal)
        return len(self.children) >= max_dec_children

    # ── Statistics ──────────────────────────────────────────────────────────

    def q(self):
        """Undiscounted empirical mean — used for greedy plan extraction."""
        return self.cum_reward / self.visits if self.visits else 0.0

    def disc_q(self):
        """Discounted empirical mean (Equation 6 of Dec-MCTS paper)."""
        return self.disc_reward / self.disc_visits if self.disc_visits > 0 else 0.0

    # ── UCB variants ────────────────────────────────────────────────────────

    def ucb_score(self, Cp):
        """Standard UCB1 — used for FULL-regime nodes."""
        if self.visits == 0:
            return float("inf")
        if self.parent is None or self.parent.visits == 0:
            return float("inf")
        return self.q() + Cp * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )

    def d_ucb(self, Cp, gamma):
        """
        D-UCT score — used for NONE/DELAYED-regime nodes.
        Mirrors the d_ucb formula in DecMCTSNode.
        """
        if self.disc_visits == 0:
            return float("inf")
        if self.parent is None or self.parent.disc_visits <= 1.0:
            return float("inf")
        explore = 2.0 * Cp * math.sqrt(
            math.log(self.parent.disc_visits) / self.disc_visits
        )
        return self.disc_q() + explore

    def update_discounted(self, reward, visited, gamma):
        """
        Apply D-UCT gamma decay and fold in a new sample.
        Must be called once per MCTS iteration for every NONE/DELAYED node.
        """
        self.disc_visits = gamma * self.disc_visits + (1.0 if visited else 0.0)
        self.disc_reward = gamma * self.disc_reward + (reward if visited else 0.0)


# ─────────────────────────────────────────────────────────
# PART 4 — UNIFIED PLANNER
# ─────────────────────────────────────────────────────────

class CommMCTS:
    """
    Semi-decentralized MCTS for a robot team with intermittent communication.

    A single tree is grown from the shared root regardless of comm regime.
    From each node, expansion and selection switch based on the node's regime:

        FULL    → CenMCTS-style: one robot acts per level (round-robin),
                  standard UCB1, reward = global_obj(paths)

        NONE    → DecMCTS-style: all robots act simultaneously per level,
                  joint actions capped at max_dec_children per node,
                  D-UCT selection, reward = Σ local_obj_fns[r](paths)

        DELAYED → DecMCTS-style with stale distributions used to bias each
                  robot's action sample; same UCB and reward as NONE.

    Rollout samples future comm regimes from the CommModel, so each simulated
    trajectory experiences a realistic mixture of conditions.

    Parameters
    ----------
    robot_ids       : list — ordered list of robot identifiers
    init_states     : dict {robot_id: state}
    global_obj      : callable  dict{robot_id: path_list} -> float
                      Team-level objective (CenMCTS-compatible signature).
    local_obj_fns   : dict {robot_id: callable}
                      Each fn: dict{robot_id: path_list} -> float
                      Marginal contribution of that robot given all paths.
                      Used in NONE / DELAYED regime.
                      If None, global_obj is used for all regimes.
    comm_model      : CommModel  (if None, a default model is constructed)
    Cp              : UCB / D-UCT exploration constant
    gamma           : D-UCT discount factor ∈ (0.5, 1)
    rollout_depth   : max steps per rollout simulation
    max_dec_children: branching-factor cap for NONE / DELAYED nodes
    """

    def __init__(
        self,
        robot_ids,
        init_states,
        global_obj,
        local_obj_fns=None,
        comm_model=None,
        Cp=1.0 / math.sqrt(2),
        gamma=0.9,
        rollout_depth=50,
        max_dec_children=20,
    ):
        self.robot_ids        = list(robot_ids)
        self.global_obj       = global_obj
        self.local_obj_fns    = local_obj_fns or {}
        self.comm_model       = comm_model or CommModel()
        self.Cp               = Cp
        self.gamma            = gamma
        self.rollout_depth    = rollout_depth
        self.max_dec_children = max_dec_children

        # Build the root node
        init_paths = {rid: [s.vertex] for rid, s in init_states.items()}
        self.root  = CommMCTSNode(
            joint_states = init_states,
            robot_paths  = init_paths,
            regime       = self.comm_model.initial_regime,
            turn         = self.robot_ids[0],
        )

        # Flat list of all tree nodes — maintained for O(1)-amortized D-UCT
        # full-tree updates (avoids DFS every backprop call).
        self._all_nodes = [self.root]

        # Latest distributions received from other robots (for DELAYED mode).
        # Updated externally via receive_distribution().
        self._current_dists = {rid: {} for rid in self.robot_ids}

    # ── PUBLIC API ───────────────────────────────────────────────────────────

    def run(self, n_iter):
        """Run n_iter MCTS iterations from the root."""
        for _ in range(n_iter):
            path  = []
            leaf  = self._select(self.root, path)
            child = self._expand(leaf, path)
            reward = self._rollout(child)
            self._backprop(path, reward)

    def best_paths(self):
        """
        Greedy Q-value descent from the root.
        Returns the best joint plan as {robot_id: [vertex, ...]}.
        """
        node = self.root
        while node.children:
            visited = [c for c in node.children if c.visits > 0]
            if not visited:
                break
            node = max(visited, key=lambda c: c.q())
        return {rid: list(node.robot_paths[rid]) for rid in self.robot_ids}

    def best_regime_sequence(self):
        """
        Return the sequence of CommRegime values along the greedy best path.
        Useful for inspecting which comm conditions the best plan depends on.
        """
        node    = self.root
        regimes = [node.regime]
        while node.children:
            visited = [c for c in node.children if c.visits > 0]
            if not visited:
                break
            node = max(visited, key=lambda c: c.q())
            regimes.append(node.regime)
        return regimes

    def receive_distribution(self, robot_id, dist_dict):
        """
        Store the latest distribution received from a robot.
        dist_dict: {seq_tuple: probability}
        Used by DELAYED-regime expansion and rollout.
        Does not modify existing tree nodes (they keep their historical
        stale snapshot); only affects future expansions.
        """
        self._current_dists[robot_id] = copy.copy(dist_dict)

    def current_regime(self):
        """Comm regime at the root of the current search."""
        return self.root.regime

    # ── SELECTION ────────────────────────────────────────────────────────────

    def _select(self, root, path):
        """
        Descend the tree via UCB until we reach an unexpanded or terminal node.
        Appends every visited node to `path`.

        FULL nodes   → standard UCB1  (ucb_score)
        NONE/DELAYED → D-UCT          (d_ucb)
        """
        node = root
        path.append(node)
        while not node.is_terminal():
            if not node.is_fully_expanded(self.max_dec_children):
                return node
            if not node.children:
                return node
            node = self._best_child(node)
            path.append(node)
        return node

    def _best_child(self, node):
        """Select the best child according to this node's regime UCB formula."""
        if node.regime == CommRegime.FULL:
            return max(node.children, key=lambda c: c.ucb_score(self.Cp))
        # NONE or DELAYED → D-UCT
        return max(node.children, key=lambda c: c.d_ucb(self.Cp, self.gamma))

    # ── EXPANSION ────────────────────────────────────────────────────────────

    def _expand(self, node, path):
        """
        Add one new child to `node` (if not terminal and expandable).
        Samples the child's comm regime from the CommModel.
        Appends the new child to `path`.
        """
        if node.is_terminal():
            return node

        next_regime = self.comm_model.sample_next_regime(node.regime)

        if node.regime == CommRegime.FULL:
            child = self._expand_full(node, next_regime)
        else:
            child = self._expand_dec(node, next_regime)

        if child is not node:
            self._all_nodes.append(child)
            path.append(child)
        return child

    def _expand_full(self, node, next_regime):
        """
        CenMCTS-style expansion: one untried action for the robot whose turn
        it is.  Edge action = int (destination vertex).

        Stale distribution handling on the child:
          FULL → FULL:    stale_dists = {}  (full real-time coordination)
          FULL → DELAYED: snapshot self._current_dists into child
          FULL → NONE:    stale_dists = {}  (robots are independent)
        """
        legal   = node.joint_states[node.turn].get_legal_actions()
        tried   = {c.action for c in node.children}
        untried = [a for a in legal if a not in tried]
        if not untried:
            return node

        action = random.choice(untried)

        new_states            = dict(node.joint_states)
        new_states[node.turn] = node.joint_states[node.turn].take_action(action)

        new_paths             = dict(node.robot_paths)
        new_paths[node.turn]  = node.robot_paths[node.turn] + [action]

        next_turn = (
            self._next_turn(node.turn, new_states)
            if next_regime == CommRegime.FULL
            else None
        )

        if next_regime == CommRegime.DELAYED:
            stale = {rid: copy.copy(d) for rid, d in self._current_dists.items()}
        else:
            stale = {}

        child = CommMCTSNode(
            joint_states = new_states,
            robot_paths  = new_paths,
            regime       = next_regime,
            turn         = next_turn,
            stale_dists  = stale,
            parent       = node,
            action       = action,
        )
        node.children.append(child)
        return child

    def _expand_dec(self, node, next_regime):
        """
        DecMCTS-style expansion: sample one joint action (all robots act
        simultaneously).  Edge action = dict {robot_id: vertex}.

        In DELAYED mode, each robot's action is biased by node.stale_dists.
        In NONE mode, each robot acts uniformly at random.

        Duplicate joint actions are rejected (up to 5 resample attempts).

        Stale distribution handling on the child:
          NONE/DELAYED → FULL:    snapshot self._current_dists (comms restored)
          NONE/DELAYED → DELAYED: age parent's stale_dists by one step
          NONE/DELAYED → NONE:    stale_dists = {}
        """
        tried_frozen = {
            frozenset(c.action.items())
            for c in node.children
            if isinstance(c.action, dict)
        }

        joint_action = None
        src_dists    = node.stale_dists if node.regime == CommRegime.DELAYED else {}
        for _ in range(5):
            candidate = self._sample_joint_action(node.joint_states, src_dists)
            key = frozenset(
                (rid, a) for rid, a in candidate.items() if a is not None
            )
            if key not in tried_frozen:
                joint_action = candidate
                break
        if joint_action is None:
            return node  # all resamples were duplicates — caller will retry

        new_states = {}
        new_paths  = {}
        for rid in self.robot_ids:
            a = joint_action.get(rid)
            if a is not None:
                new_states[rid] = node.joint_states[rid].take_action(a)
                new_paths[rid]  = node.robot_paths[rid] + [a]
            else:
                new_states[rid] = node.joint_states[rid]
                new_paths[rid]  = node.robot_paths[rid]

        if next_regime == CommRegime.FULL:
            stale     = {rid: copy.copy(d) for rid, d in self._current_dists.items()}
            next_turn = self._first_active(new_states)
        elif next_regime == CommRegime.DELAYED:
            stale     = self._advance_stale_dists(node.stale_dists, steps=1)
            next_turn = None
        else:  # NONE
            stale     = {}
            next_turn = None

        child = CommMCTSNode(
            joint_states = new_states,
            robot_paths  = new_paths,
            regime       = next_regime,
            turn         = next_turn,
            stale_dists  = stale,
            parent       = node,
            action       = joint_action,
        )
        node.children.append(child)
        return child

    # ── ROLLOUT ──────────────────────────────────────────────────────────────

    def _rollout(self, node):
        """
        Random rollout from node.

        At each step:
          1. Sample next regime from CommModel.
          2. Advance robot states according to the regime:
               FULL:    one robot acts (round-robin), uniform random action.
               NONE:    all robots act simultaneously, uniform random actions.
               DELAYED: all robots act simultaneously, actions sampled from
                        stale distributions (falls back to random if empty).
          3. Age stale distributions by one step when in DELAYED regime.

        Reward is evaluated under the leaf node's starting regime, so the
        tree node's value reflects the planning quality under that regime
        regardless of how regimes evolved during the rollout.
        """
        states   = dict(node.joint_states)
        paths    = {rid: list(p) for rid, p in node.robot_paths.items()}
        regime   = node.regime
        stale    = dict(node.stale_dists)
        turn_idx = (
            self.robot_ids.index(node.turn)
            if node.turn is not None and node.turn in self.robot_ids
            else 0
        )
        depth = 0

        while depth < self.rollout_depth:
            active = [
                rid for rid in self.robot_ids
                if not states[rid].is_terminal_state()
            ]
            if not active:
                break

            regime = self.comm_model.sample_next_regime(regime)

            if regime == CommRegime.FULL:
                # CenMCTS-style: one robot acts per rollout step
                rid   = active[turn_idx % len(active)]
                legal = states[rid].get_legal_actions()
                if legal:
                    a           = random.choice(legal)
                    states[rid] = states[rid].take_action(a)
                    paths[rid]  = paths[rid] + [a]
                turn_idx += 1

            else:
                # DecMCTS-style: all active robots act simultaneously
                for rid in active:
                    legal = states[rid].get_legal_actions()
                    if not legal:
                        continue
                    if regime == CommRegime.DELAYED and stale.get(rid):
                        a = self._sample_from_stale(stale[rid], legal)
                    else:
                        a = random.choice(legal)
                    states[rid] = states[rid].take_action(a)
                    paths[rid]  = paths[rid] + [a]

                if regime == CommRegime.DELAYED:
                    stale = self._advance_stale_dists(stale, steps=1)

            depth += 1

        return self._evaluate_reward(paths, node.regime)

    # ── BACKPROPAGATION ──────────────────────────────────────────────────────

    def _backprop(self, path, reward):
        """
        Propagate reward back through the path from root to leaf.

        Standard undiscounted update (visits, cum_reward):
            Applied to every node on the path, regardless of regime.

        D-UCT discounted update (disc_visits, disc_reward):
            Applied to every NONE/DELAYED node in the entire tree — not just
            those on the path.  Off-path nodes receive gamma decay only
            (visited=False), which is what makes D-UCT devalue stale stats.
            FULL-regime nodes are excluded; they rely on standard UCB stats.
        """
        path_set = {id(n) for n in path}

        # Standard undiscounted update for all path nodes
        for node in path:
            node.visits     += 1
            node.cum_reward += reward

        # D-UCT full-tree decay for every NONE/DELAYED node
        for node in self._all_nodes:
            if node.regime in (CommRegime.NONE, CommRegime.DELAYED):
                on_path = id(node) in path_set
                node.update_discounted(reward, visited=on_path, gamma=self.gamma)

    # ── REWARD EVALUATION ────────────────────────────────────────────────────

    def _evaluate_reward(self, paths, regime):
        """
        Evaluate team reward for a completed set of paths.

        FULL:    global_obj(paths)          — full coordination, global view
        NONE:    Σ local_obj_fns[r](paths)  — sum of marginal utilities
        DELAYED: same as NONE (stale info biased action selection but does
                 not alter the reward model — we still judge by true utility)

        Falls back to global_obj if local_obj_fns was not provided.
        """
        if regime == CommRegime.FULL or not self.local_obj_fns:
            return self.global_obj(paths)
        return sum(
            self.local_obj_fns[rid](paths)
            for rid in self.robot_ids
            if rid in self.local_obj_fns
        )

    # ── HELPERS ──────────────────────────────────────────────────────────────

    def _sample_joint_action(self, states, stale_dists):
        """
        Sample one joint action {robot_id: vertex} with robots acting
        independently.

        Each robot's action is drawn by marginalizing its stale distribution
        down to a first-action probability and filtering to legal actions.
        Falls back to uniform random for robots with empty or incompatible dists.
        Terminal robots are assigned None.
        """
        joint = {}
        for rid in self.robot_ids:
            state = states[rid]
            if state.is_terminal_state():
                joint[rid] = None
                continue
            legal = state.get_legal_actions()
            if not legal:
                joint[rid] = None
                continue
            dist = stale_dists.get(rid, {})
            joint[rid] = self._sample_from_stale(dist, legal)
        return joint

    def _sample_from_stale(self, dist, legal_actions):
        """
        Sample an action from a stale distribution dict, restricted to
        legal_actions.

        Marginalizes over sequences: sums P(seq) for all sequences whose
        first element equals each candidate action.  Falls back to uniform
        random if no legal overlap exists.
        """
        if not dist:
            return random.choice(legal_actions)

        legal_set    = set(legal_actions)
        action_probs = {}
        for seq_tup, prob in dist.items():
            if seq_tup and seq_tup[0] in legal_set:
                first = seq_tup[0]
                action_probs[first] = action_probs.get(first, 0.0) + prob

        if not action_probs:
            return random.choice(legal_actions)

        total = sum(action_probs.values())
        r     = random.random() * total
        cum   = 0.0
        for action, prob in action_probs.items():
            cum += prob
            if r <= cum:
                return action
        return list(action_probs.keys())[-1]

    @staticmethod
    def _advance_stale_dists(stale_dists, steps):
        """
        Age stale distributions by `steps` timesteps.

        A distribution {(v0, v1, v2): p} aged by 2 steps becomes {(v2,): p}
        (remove the first `steps` elements from each sequence key).
        Sequences shorter than `steps` are dropped.  Result is renormalized.
        """
        result = {}
        for rid, dist in stale_dists.items():
            aged = {}
            for seq_tup, prob in dist.items():
                trimmed = seq_tup[steps:]
                if trimmed:
                    aged[trimmed] = aged.get(trimmed, 0.0) + prob
            if aged:
                total = sum(aged.values())
                result[rid] = {k: v / total for k, v in aged.items()}
        return result

    def _next_turn(self, current_turn, states):
        """
        Round-robin to the next non-terminal robot (matches CenMCTS logic).
        If all robots are terminal, still cycles to prevent infinite loops.
        """
        idx = self.robot_ids.index(current_turn)
        R   = len(self.robot_ids)
        for i in range(1, R + 1):
            nxt = self.robot_ids[(idx + i) % R]
            if not states[nxt].is_terminal_state():
                return nxt
        return self.robot_ids[(idx + 1) % R]

    def _first_active(self, states):
        """Return the first non-terminal robot, or robot_ids[0] if all terminal."""
        for rid in self.robot_ids:
            if not states[rid].is_terminal_state():
                return rid
        return self.robot_ids[0]
