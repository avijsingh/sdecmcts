"""
cpde_decmcts.py
---------------
Centralized Planning, Decentralized Execution (CPDE) via POMCP-style
observation-branching MCTS.

Planning (centralized, done once):
  Each agent builds an observation-branching policy tree from the initial
  joint belief.  Tree edges are (action, local_obs) pairs so the extracted
  policy is a pure function of the agent's *own* observation history — no
  inter-agent communication required during execution.

Execution (decentralized):
  Each agent maintains its local obs history, looks up the current action
  in its ConditionalPolicy, executes it, receives its local observation,
  and records (action, local_obs) for the next step.

Communication gating:
  Agents may replan (re-call plan() + extract_policy()) whenever a
  communication opportunity arises.  Otherwise they execute the
  pre-computed policy unchanged.

Model interface (all three benchmark models satisfy this after our edit):
  model.n_states      : int
  model.n_obs         : int    (joint obs count)
  model.act_per_agent : int
  model.obs_per_agent : int    (per-agent obs count)
  model.sample_next_state(s, joint_a) -> int
  model.reward(joint_a, s)            -> float
  model.O[joint_a * n_states * n_obs + s_next * n_obs + joint_o] -> float

Joint action / obs encoding (same convention across all benchmarks):
  joint_a = a0 + act_per_agent * a1
  joint_o = o0 + obs_per_agent * o1   (o0 = agent-0 local obs)
"""

import math
import random


# ── LOCAL OBS HELPERS ─────────────────────────────────────────────────────────

def _sample_local_obs(model, joint_a, s_next, agent_id):
    """
    Sample agent `agent_id`'s local obs from the marginal distribution
    P(local_o | s_next, joint_a) = Σ_{other_o} P(joint_o | s_next, joint_a).
    """
    probs = []
    for lo in range(model.obs_per_agent):
        p = 0.0
        for oo in range(model.obs_per_agent):
            if agent_id == 0:
                joint_o = lo + model.obs_per_agent * oo
            else:
                joint_o = oo + model.obs_per_agent * lo
            p += model.O[joint_a * model.n_states * model.n_obs
                         + s_next * model.n_obs + joint_o]
        probs.append(p)

    r = random.random()
    cum = 0.0
    for lo, p in enumerate(probs):
        cum += p
        if r <= cum:
            return lo
    return model.obs_per_agent - 1


def _marginal_obs_weight(model, joint_a, s_next, local_obs, agent_id):
    """
    P(local_obs | s_next, joint_a) = Σ_{other_o} P(joint_o | s_next, joint_a).
    Used to weight particles during decentralized execution.
    """
    p = 0.0
    for oo in range(model.obs_per_agent):
        if agent_id == 0:
            joint_o = local_obs + model.obs_per_agent * oo
        else:
            joint_o = oo + model.obs_per_agent * local_obs
        p += model.O[joint_a * model.n_states * model.n_obs
                     + s_next * model.n_obs + joint_o]
    return p


def marginal_pf_update(particles, action, local_obs, model, agent_id,
                        n_resample=None):
    """
    Particle filter update for decentralized execution: propagate through T
    (other agent's action = uniform random), weight by marginal obs likelihood,
    resample.

    Parameters
    ----------
    particles  : list[int]   current belief particles
    action     : int         this agent's action
    local_obs  : int         this agent's local observation
    model      : model obj   with n_states, act_per_agent, obs_per_agent, O, T
    agent_id   : int         0 or 1
    n_resample : int | None  target particle count (default = len(particles))

    Returns
    -------
    list[int]  resampled particles representing updated belief
    """
    if n_resample is None:
        n_resample = len(particles)

    new_states = []
    weights    = []
    for s in particles:
        other   = random.randint(0, model.act_per_agent - 1)
        joint_a = (action + model.act_per_agent * other if agent_id == 0
                   else other + model.act_per_agent * action)
        s_next  = model.sample_next_state(s, joint_a)
        w       = _marginal_obs_weight(model, joint_a, s_next, local_obs, agent_id)
        new_states.append(s_next)
        weights.append(w)

    total = sum(weights)
    if total <= 1e-12:
        return new_states[:n_resample]

    probs = [w / total for w in weights]

    # Systematic resampling
    resampled = []
    r   = random.random() / n_resample
    cum = 0.0
    idx = 0
    for j in range(n_resample):
        target = r + j / n_resample
        while cum < target and idx < len(probs):
            cum += probs[idx]
            idx += 1
        resampled.append(new_states[max(idx - 1, 0)])
    return resampled


# ── POMCP TREE NODE ───────────────────────────────────────────────────────────

class CPDENode:
    """
    Node in agent r's obs-branching policy tree.

    Each node represents a local observation history h = [(a0,o0),(a1,o1),...].
    Children are keyed by (action, local_obs) pairs.
    Particles approximate P(s | h).

    UCT action statistics are stored per-action (not per child node), matching
    the POMCP convention: action selection uses counts and values accumulated
    across all (action, *) children that share the same action.
    """

    __slots__ = ("particles", "step", "max_steps", "obs_history",
                 "children", "action_stats", "visits", "_act_per_agent")

    def __init__(self, particles, step, max_steps, act_per_agent,
                 parent=None, action=None, obs=None):
        self.particles     = list(particles)
        self.step          = step
        self.max_steps     = max_steps
        self._act_per_agent = act_per_agent
        # obs_history = sequence of (action, local_obs) pairs from root
        if parent is None:
            self.obs_history = []
        else:
            self.obs_history = parent.obs_history + [(action, obs)]
        # Children: {(action, local_obs): CPDENode}
        self.children      = {}
        # Per-action UCT stats: {action: [visits, cum_reward]}
        self.action_stats  = {}
        # Node visit count (for UCT parent term)
        self.visits        = 0

    def is_terminal(self):
        return self.step >= self.max_steps

    def untried_actions(self):
        """Actions not yet tried from this node."""
        return [a for a in range(self._act_per_agent)
                if a not in self.action_stats]


# ── CPDE DEC-MCTS PLANNER ────────────────────────────────────────────────────

class CPDEDecMCTS:
    """
    POMCP-style obs-branching planner for a single agent in CPDE mode.

    Builds a policy tree from the initial belief (particles) using POMCP.
    After plan() is called, extract_policy() returns a ConditionalPolicy
    that maps observation histories to actions for decentralized execution.

    Parameters
    ----------
    robot_id       : int   (0 or 1)
    init_particles : list[int]  initial belief particles
    model          : model obj  with CPDE interface (see module docstring)
    max_steps      : int        planning horizon
    Cp             : float      UCT exploration constant
    rollout_depth  : int        max random-rollout depth from leaf
    """

    def __init__(self, robot_id, init_particles, model, max_steps,
                 Cp=1.0 / math.sqrt(2), rollout_depth=20):
        self.robot_id      = robot_id
        self.model         = model
        self.max_steps     = max_steps
        self.Cp            = Cp
        self.rollout_depth = rollout_depth

        self.root = CPDENode(
            particles    = init_particles,
            step         = 0,
            max_steps    = max_steps,
            act_per_agent = model.act_per_agent,
        )

    def plan(self, n_iterations):
        """Run `n_iterations` POMCP simulations from the initial belief."""
        # Mark root as initialized so first call descends into the tree
        if self.root.visits == 0:
            self.root.visits = 1
        for _ in range(n_iterations):
            if not self.root.particles:
                break
            s = random.choice(self.root.particles)
            self._simulate(self.root, s)

    # ── POMCP SIMULATION ──────────────────────────────────────────────────────

    def _simulate(self, node, s):
        """
        Recursive POMCP simulation (standard POMCP tree structure).

        A node with visits == 0 has never been expanded — do a rollout
        immediately and mark it as initialized.  On subsequent visits,
        select an action via UCT, sample the generative model, recurse
        into the child, and update the parent's action statistics.

        Returns cumulative (undiscounted) reward from `node` to terminal.
        """
        if node.is_terminal():
            return 0.0

        # First visit: initialize this node with a rollout
        if node.visits == 0:
            node.visits = 1
            return self._rollout(s, node.step)

        action = self._select_action(node)

        # Generative model: sample next state and this agent's local obs.
        # Other agent's action is sampled uniformly at random.
        other   = random.randint(0, self.model.act_per_agent - 1)
        joint_a = (action + self.model.act_per_agent * other
                   if self.robot_id == 0
                   else other + self.model.act_per_agent * action)
        s_next    = self.model.sample_next_state(s, joint_a)
        r         = self.model.reward(joint_a, s)
        local_obs = _sample_local_obs(self.model, joint_a, s_next, self.robot_id)

        # Navigate to (or create) child node
        key = (action, local_obs)
        if key not in node.children:
            node.children[key] = CPDENode(
                particles     = [],
                step          = node.step + 1,
                max_steps     = node.max_steps,
                act_per_agent = self.model.act_per_agent,
                parent        = node,
                action        = action,
                obs           = local_obs,
            )
        child = node.children[key]
        child.particles.append(s_next)   # lazy particle accumulation

        total_r = r + self._simulate(child, s_next)

        # UCT backprop: update action statistics at this node
        node.visits += 1
        stats = node.action_stats.setdefault(action, [0, 0.0])
        stats[0] += 1        # action visit count
        stats[1] += total_r  # cumulative reward for this action

        return total_r

    def _select_action(self, node):
        """UCT action selection. Untried actions get priority (→ inf UCB)."""
        untried = node.untried_actions()
        if untried:
            return random.choice(untried)

        log_n = math.log(node.visits) if node.visits > 0 else 0.0

        def ucb(a):
            v, cr = node.action_stats[a]
            q       = cr / v if v > 0 else 0.0
            explore = (self.Cp * math.sqrt(log_n / v)
                       if v > 0 else float("inf"))
            return q + explore

        return max(range(self.model.act_per_agent), key=ucb)

    def _rollout(self, s, step):
        """
        Random rollout from concrete state `s` at `step`.
        Both agents select actions uniformly at random.
        Returns total (undiscounted) reward over remaining steps.
        """
        total = 0.0
        for d in range(self.rollout_depth):
            if step + d >= self.max_steps:
                break
            action  = random.randint(0, self.model.act_per_agent - 1)
            other   = random.randint(0, self.model.act_per_agent - 1)
            joint_a = (action + self.model.act_per_agent * other
                       if self.robot_id == 0
                       else other + self.model.act_per_agent * action)
            total  += self.model.reward(joint_a, s)
            s       = self.model.sample_next_state(s, joint_a)
        return total

    # ── POLICY EXTRACTION ─────────────────────────────────────────────────────

    def extract_policy(self, default_action=0):
        """
        Walk the tree and build a {obs_history: action} mapping.

        At each node, the chosen action = the action with the highest
        visit count (most reliable estimate after many simulations).

        Returns
        -------
        ConditionalPolicy
        """
        policy = {}
        self._extract_recursive(self.root, policy)
        return ConditionalPolicy(policy, default_action=default_action)

    def _extract_recursive(self, node, policy):
        if not node.action_stats:
            return
            
        # [BUG FIX]: Thin-Tree Extraction Defense
        # If this node was barely visited during planning, the "best_action" 
        # is just noise (often the first random action tested). We stop extracting here 
        # so the execution policy falls back to the safe `default_action` (LISTEN) 
        # instead of blindly opening tiger doors.
        if node.visits < 10:
            return
            
        # Greedy by visit count (robust to reward scale differences)
        best_action = max(node.action_stats,
                          key=lambda a: node.action_stats[a][0])
        policy[tuple(node.obs_history)] = best_action
        for child in node.children.values():
            self._extract_recursive(child, policy)


# ── CONDITIONAL POLICY ────────────────────────────────────────────────────────

class ConditionalPolicy:
    """
    Decentralized execution policy extracted from a CPDE planning tree.

    Maintains the agent's local observation history and maps it to actions.
    If the current history was not reached during planning, falls back to
    `default_action`.

    Usage
    -----
    policy.reset()                      # start of each episode
    action = policy.get_action()        # get action for current history
    policy.record(action, local_obs)    # update history after step
    """

    def __init__(self, policy_dict, default_action=0):
        self._policy        = policy_dict   # {tuple([(a,o),...]): action}
        self._default       = default_action
        self._obs_history   = []

    def reset(self):
        """Reset observation history for a new episode."""
        self._obs_history = []

    def get_action(self):
        """Return the action prescribed by the current observation history."""
        key = tuple(self._obs_history)
        return self._policy.get(key, self._default)

    def record(self, action, local_obs):
        """Append (action, local_obs) to history after a step."""
        self._obs_history.append((action, local_obs))

    def history(self):
        """Return current obs history as a list of (action, obs) pairs."""
        return list(self._obs_history)


# ── MULTI-AGENT COORDINATOR ───────────────────────────────────────────────────

class CPDEDecMCTSTeam:
    """
    Runs CPDE planning for a team of agents and extracts their policies.

    Agents plan independently from their respective initial particles.
    (Proper centralized planning with inter-agent distribution sharing
    can be added later; independent POMCP is the baseline CPDE.)

    Parameters
    ----------
    planners : {robot_id: CPDEDecMCTS}
    """

    def __init__(self, planners):
        self.planners = planners

    def plan(self, n_iterations):
        """Run n_iterations independent POMCP simulations for every agent."""
        for p in self.planners.values():
            p.plan(n_iterations)

    def plan_joint(self, n_iterations):
        """
        Run n_iterations JOINT POMCP simulations.

        Both agents' obs-branching trees are trained on the same simulated
        trajectories.  At each step of a simulation, agent 0 selects its
        action from its own UCT tree and agent 1 selects from its own tree —
        the joint action is their combined choice.

        This is centralized planning (full joint reward signal, full joint
        state/observation sampling) that extracts decoupled per-agent
        conditional policies — the defining property of CPDE.

        Contrast with plan() (independent): there each agent assumes the other
        always uses null_action.  With joint simulation:
          - Both listen → −2  (same as null assumption)
          - Both open correct door → +20 (vs +9 under null assumption)
          - One opens, one listens → −101 trains the trees to avoid this
        The trees therefore naturally learn to coordinate: open together when
        confident, listen together when uncertain.
        """
        rids = sorted(self.planners.keys())
        p0   = self.planners[rids[0]]
        p1   = self.planners[rids[1]]

        # Initialise roots (same convention as single-agent plan())
        if p0.root.visits == 0:
            p0.root.visits = 1
        if p1.root.visits == 0:
            p1.root.visits = 1

        for _ in range(n_iterations):
            particles = p0.root.particles
            if not particles:
                break
            s = random.choice(particles)
            self._joint_simulate(p0, p1, p0.root, p1.root, s)

    # ── JOINT SIMULATION ──────────────────────────────────────────────────────

    def _joint_simulate(self, p0, p1, node0, node1, s):
        """
        Recurse jointly through both trees.

        node0 / node1  : current nodes in each agent's tree
        s              : current concrete state

        Returns cumulative (undiscounted) reward from this point to terminal.
        """
        if node0.is_terminal():
            return 0.0

        # First visit to either node → joint rollout to initialise
        if node0.visits == 0 or node1.visits == 0:
            node0.visits = max(node0.visits, 1)
            node1.visits = max(node1.visits, 1)
            return self._joint_rollout(p0, p1, s, node0.step)

        model   = p0.model
        action0 = p0._select_action(node0)
        action1 = p1._select_action(node1)

        joint_a    = action0 + model.act_per_agent * action1
        s_next     = model.sample_next_state(s, joint_a)
        r          = model.reward(joint_a, s)
        local_obs0 = _sample_local_obs(model, joint_a, s_next, 0)
        local_obs1 = _sample_local_obs(model, joint_a, s_next, 1)

        # Navigate / create children in each tree
        key0 = (action0, local_obs0)
        key1 = (action1, local_obs1)

        if key0 not in node0.children:
            node0.children[key0] = CPDENode(
                particles     = [],
                step          = node0.step + 1,
                max_steps     = node0.max_steps,
                act_per_agent = model.act_per_agent,
                parent        = node0,
                action        = action0,
                obs           = local_obs0,
            )
        if key1 not in node1.children:
            node1.children[key1] = CPDENode(
                particles     = [],
                step          = node1.step + 1,
                max_steps     = node1.max_steps,
                act_per_agent = model.act_per_agent,
                parent        = node1,
                action        = action1,
                obs           = local_obs1,
            )

        child0 = node0.children[key0]
        child1 = node1.children[key1]
        child0.particles.append(s_next)
        child1.particles.append(s_next)

        total_r = r + self._joint_simulate(p0, p1, child0, child1, s_next)

        # Backprop to both trees — same joint reward signal
        node0.visits += 1
        node1.visits += 1
        stats0 = node0.action_stats.setdefault(action0, [0, 0.0])
        stats1 = node1.action_stats.setdefault(action1, [0, 0.0])
        stats0[0] += 1;  stats0[1] += total_r
        stats1[0] += 1;  stats1[1] += total_r

        return total_r

    def _joint_rollout(self, p0, p1, s, step):
        """
        Joint rollout: both agents select actions uniformly at random.
        Returns total reward for steps [step, max_steps).
        """
        model = p0.model
        total = 0.0
        max_d = max(p0.rollout_depth, p1.rollout_depth)
        for d in range(max_d):
            if step + d >= p0.max_steps:
                break
            a0      = random.randint(0, model.act_per_agent - 1)
            a1      = random.randint(0, model.act_per_agent - 1)
            joint_a = a0 + model.act_per_agent * a1
            total  += model.reward(joint_a, s)
            s       = model.sample_next_state(s, joint_a)
        return total

    def extract_policies(self, default_action=0):
        """Return {robot_id: ConditionalPolicy}."""
        return {
            rid: p.extract_policy(default_action=default_action)
            for rid, p in self.planners.items()
        }
