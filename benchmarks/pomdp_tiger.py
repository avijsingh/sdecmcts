"""
pomdp_tiger.py
--------------
Tiger POMDP benchmark for Dec-MCTS, using Mahdi's exact T/O/R formulation.

Reference values from RS-SDA* paper (AAMAS 2026):
  h=8:  RS-MAA* 12.217  RS-SDA* 27.215  Cen 47.717
  h=9:  RS-MAA* 15.572  RS-SDA* 30.905  Cen 53.474
  h=10: RS-MAA* 15.184  RS-SDA* 34.724  Cen 60.510

Dec-MCTS should land between RS-MAA* (lower) and RS-SDA* (upper).

Problem:
  2 states  : TIGER_LEFT=0, TIGER_RIGHT=1
  3 actions : OPEN_LEFT=0, OPEN_RIGHT=1, LISTEN=2   (per agent)
  9 joint actions, 4 joint observations
  init_belief: [0.5, 0.5]

Belief-state interface
----------------------
TigerBeliefState wraps a particle filter over concrete states.
This lets the existing Dec-MCTS tree (D-UCT, distribution learning,
communication) run unchanged -- only the state interface and reward
evaluation differ from the MDP version.

Reward estimation uses Monte Carlo sampling over the particle set:
  local_obj_fn(joint_seqs) = (1/N) * sum_i R(episode_i | init_state ~ belief)
"""

import random

# ── CONSTANTS ──────────────────────────────────────────────────────────────────

OPEN_LEFT     = 0
OPEN_RIGHT    = 1
LISTEN        = 2

N_AGENTS      = 2
ACT_PER_AGENT = 3
OBS_PER_AGENT = 2
N_STATES      = 2
N_ACTS        = ACT_PER_AGENT ** N_AGENTS   # 9
N_OBS         = OBS_PER_AGENT ** N_AGENTS   # 4

DEFAULT_HORIZON = 9
N_PARTICLES     = 200   # particles per belief node
N_MC_SAMPLES    = 50    # MC draws per local_obj_fn call


# ── TIGER POMDP MODEL ──────────────────────────────────────────────────────────

class TigerModel:
    """
    T/O/R matrices using Mahdi's exact formulation (MOD='C', TRIG_SEMI).

    Indexing:
      T[a * N_STATES^2 + s * N_STATES + s'] = P(s' | s, a)
      O[a * N_STATES * N_OBS + s' * N_OBS + o] = P(o | a, s')
      R[a * N_STATES + s]                       = R(s, a)

    Joint action encoding:
      joint_a = a0 + a1 * ACT_PER_AGENT
      → a0 = joint_a % ACT_PER_AGENT   (agent 0)
      → a1 = joint_a // ACT_PER_AGENT  (agent 1)

    Joint obs encoding:
      joint_o = o0 + o1 * OBS_PER_AGENT
    """

    def __init__(self):
        nsq = N_STATES ** 2
        nso = N_STATES * N_OBS
        self.T = [0.0] * (nsq * N_ACTS)
        self.O = [0.0] * (nso * N_ACTS)
        self.R = [0.0] * (N_STATES * N_ACTS)
        self.init_belief = [1.0 / N_STATES] * N_STATES
        # Attributes required by cpde_decmcts.py
        self.n_states      = N_STATES
        self.n_obs         = N_OBS
        self.act_per_agent = ACT_PER_AGENT
        self.obs_per_agent = OBS_PER_AGENT
        self._build()

    def _set_o(self, a, sp, o, val):
        self.O[a * N_STATES * N_OBS + sp * N_OBS + o] = val

    def _build(self):
        def a0(a): return a % ACT_PER_AGENT
        def a1(a): return a // ACT_PER_AGENT

        for a in range(N_ACTS):
            if a == 8:  # both LISTEN (a0=2, a1=2)
                for s in range(N_STATES):
                    self.R[a * N_STATES + s] = -2.0
                    self.T[a * N_STATES**2 + s * N_STATES + s] = 1.0  # tiger stays
                # Informative observations: 85% accurate per agent
                for sp in range(N_STATES):
                    for o in range(N_OBS):
                        b0, b1 = o % OBS_PER_AGENT, o // OBS_PER_AGENT
                        p0 = 0.85 if b0 == sp else 0.15
                        p1 = 0.85 if b1 == sp else 0.15
                        self._set_o(8, sp, o, p0 * p1)
            else:
                # Any open action: tiger resets uniformly, uniform obs
                for s in range(N_STATES):
                    r = (
                        - 101 * ((a0(a) == s) * (a1(a) == 2) +
                                 (a1(a) == s) * (a0(a) == 2))
                        -  50 * ((a0(a) == s) * (a1(a) == s))
                        - 100 * ((a0(a) != 2) * (a0(a) != s) * (a1(a) == s) +
                                 (a1(a) != 2) * (a1(a) != s) * (a0(a) == s))
                        +   9 * ((a0(a) != 2) * (a0(a) != s) * (a1(a) == 2) +
                                 (a1(a) != 2) * (a1(a) != s) * (a0(a) == 2))
                        +  20 * ((a0(a) != 2) * (a0(a) != s) *
                                 (a1(a) != 2) * (a1(a) != s))
                    )
                    self.R[a * N_STATES + s] = float(r)
                    for sp in range(N_STATES):
                        self.T[a * N_STATES**2 + s * N_STATES + sp] = 1.0 / N_STATES
                for sp in range(N_STATES):
                    for o in range(N_OBS):
                        self._set_o(a, sp, o, 1.0 / N_OBS)

        # MOD='C': single-listener observation corrections
        # Actions where exactly one agent listens (2,5,6,7)
        for a in [2, 5, 6, 7]:
            for sp in range(N_STATES):
                for o in range(N_OBS):
                    b0, b1 = o % OBS_PER_AGENT, o // OBS_PER_AGENT
                    if a0(a) == 2 and a1(a) != 2:
                        # Agent 0 listens, agent 1 opens
                        p_listen = 0.85 if b0 == sp else 0.15
                        self._set_o(a, sp, o, p_listen * 0.5)
                    elif a1(a) == 2 and a0(a) != 2:
                        # Agent 1 listens, agent 0 opens
                        p_listen = 0.85 if b1 == sp else 0.15
                        self._set_o(a, sp, o, 0.5 * p_listen)

    def sample_next_state(self, s, joint_a):
        """Sample s' ~ T(· | s, joint_a)."""
        r = random.random()
        cum = 0.0
        for sp in range(N_STATES):
            cum += self.T[joint_a * N_STATES**2 + s * N_STATES + sp]
            if r <= cum:
                return sp
        return N_STATES - 1

    def reward(self, joint_a, s):
        return self.R[joint_a * N_STATES + s]


# ── BELIEF STATE ───────────────────────────────────────────────────────────────

class TigerBeliefState:
    """
    Particle-filter belief state for POMCP-style Dec-MCTS.

    Each node in robot r's local tree holds a list of concrete state
    samples (particles).  take_action() propagates the particles through
    the generative model, treating the other agent's action as uniform
    random (replaced by sampled distribution once comms are established).

    The vertex attribute is a hashable summary of the belief, used by
    global_obj to identify the start of each robot's path.
    """

    __slots__ = ("particles", "step", "max_steps", "model", "agent_id", "vertex")

    def __init__(self, particles, step, max_steps, model, agent_id):
        self.particles = particles
        self.step      = step
        self.max_steps = max_steps
        self.model     = model
        self.agent_id  = agent_id
        # Hashable belief summary: step + quantized belief.
        # Quantize to 10 bins so similar beliefs share tree nodes — coarse
        # discretization is essential for Dec-MCTS tree re-use (if every
        # belief gets a unique vertex the tree stays 1-node-wide forever).
        n = len(particles) or 1
        p_left = particles.count(0) / n
        self.vertex = (step, round(p_left * 10) / 10)

    def get_legal_actions(self):
        if self.step >= self.max_steps:
            return []
        return [OPEN_LEFT, OPEN_RIGHT, LISTEN]

    def take_action(self, my_action):
        """
        Advance particle set by my_action.
        Other agent's action sampled uniformly (no prior).
        Particles pass through T; obs conditioning is deferred to
        MC evaluation in local_obj_fn.
        """
        m = self.model
        new_particles = []
        for s in self.particles:
            other = random.randint(0, ACT_PER_AGENT - 1)
            joint_a = (my_action + other * ACT_PER_AGENT
                       if self.agent_id == 0
                       else other + my_action * ACT_PER_AGENT)
            new_particles.append(m.sample_next_state(s, joint_a))

        return TigerBeliefState(new_particles, self.step + 1,
                                self.max_steps, m, self.agent_id)

    def is_terminal_state(self):
        return self.step >= self.max_steps


# ── OBJECTIVE FUNCTIONS ────────────────────────────────────────────────────────

def _simulate_episode(init_state, joint_action_seqs, model, null_action=LISTEN):
    """
    Simulate one concrete episode from init_state under joint_action_seqs.
    Missing actions are padded with null_action.
    Returns total reward.
    """
    s      = init_state
    seqs   = {rid: list(seq) for rid, seq in joint_action_seqs.items()}
    length = max((len(v) for v in seqs.values()), default=0)

    total = 0.0
    for t in range(length):
        a0 = seqs.get(0, [])[t] if t < len(seqs.get(0, [])) else null_action
        a1 = seqs.get(1, [])[t] if t < len(seqs.get(1, [])) else null_action
        joint_a = a0 + a1 * ACT_PER_AGENT
        total  += model.reward(joint_a, s)
        s       = model.sample_next_state(s, joint_a)
    return total


def make_tiger_pomdp_objectives(model, n_mc=N_MC_SAMPLES):
    """
    Build global_obj and local_obj_fns for all three planners.

    Both functions estimate expected reward by Monte Carlo sampling
    from the uniform initial belief.

    global_obj(joint_paths)
        joint_paths = {rid: [vertex, a1, a2, ...]}
        Strips vertex, evaluates action sequences via MC.

    local_obj_fns[rid](joint_action_seqs)
        joint_action_seqs = {rid: [a1, a2, ...]}
        Evaluates directly via MC.
    """
    init_belief = model.init_belief

    def _sample_init():
        r = random.random()
        cum = 0.0
        for s, p in enumerate(init_belief):
            cum += p
            if r <= cum:
                return s
        return N_STATES - 1

    def global_obj(joint_paths):
        seqs  = {rid: list(path[1:]) for rid, path in joint_paths.items()}
        total = sum(_simulate_episode(_sample_init(), seqs, model)
                    for _ in range(n_mc))
        return total / n_mc

    def make_local_fn(rid):
        def local_fn(joint_action_seqs):
            total = sum(_simulate_episode(_sample_init(), joint_action_seqs, model)
                        for _ in range(n_mc))
            null_seqs = dict(joint_action_seqs)
            null_seqs[rid] = []
            null_total = sum(_simulate_episode(_sample_init(), null_seqs, model)
                             for _ in range(n_mc))
            return (total - null_total) / n_mc
        return local_fn

    local_obj_fns = {rid: make_local_fn(rid) for rid in range(N_AGENTS)}
    return global_obj, local_obj_fns


# ── ONLINE POMDP HELPERS ───────────────────────────────────────────────────────

def sample_obs(model, joint_a, true_state):
    """Sample joint observation o ~ O(· | true_state, joint_a)."""
    r = random.random()
    cum = 0.0
    for o in range(N_OBS):
        cum += model.O[joint_a * N_STATES * N_OBS + true_state * N_OBS + o]
        if r <= cum:
            return o
    return N_OBS - 1


def update_particles(particles, joint_a, joint_obs, model, n_resample=None):
    """
    Particle filter update: propagate particles through T, then weight by
    O(joint_obs | s', joint_a) and resample.

    After any OPEN action the tiger resets (T is uniform → particles also reset
    to uniform, which the transition already handles).  Weight by the
    observation likelihood to sharpen the LISTEN case.
    """
    if n_resample is None:
        n_resample = len(particles)

    # Propagate each particle through T, accumulate observation weight
    new_states  = []
    obs_weights = []
    for s in particles:
        s_next  = model.sample_next_state(s, joint_a)
        w       = model.O[joint_a * N_STATES * N_OBS + s_next * N_OBS + joint_obs]
        new_states.append(s_next)
        obs_weights.append(w)

    total = sum(obs_weights)
    if total <= 1e-12:
        # Degenerate — keep propagated particles unweighted
        return new_states[:n_resample]

    probs = [w / total for w in obs_weights]

    # Systematic resampling
    resampled = []
    r = random.random() / n_resample
    cum = 0.0
    idx = 0
    for j in range(n_resample):
        target = r + j / n_resample
        while cum < target and idx < len(probs):
            cum += probs[idx]
            idx += 1
        resampled.append(new_states[max(idx - 1, 0)])
    return resampled


def make_objectives_from_particles(particles, model, n_mc=N_MC_SAMPLES):
    """
    Build global_obj and local_obj_fns that sample from `particles` (the
    current-step belief) rather than from the initial uniform belief.

    Used at each step in the online/receding-horizon setting.
    """
    def _sample_belief():
        return random.choice(particles)

    def global_obj(joint_paths):
        seqs  = {rid: list(path[1:]) for rid, path in joint_paths.items()}
        total = sum(_simulate_episode(_sample_belief(), seqs, model)
                    for _ in range(n_mc))
        return total / n_mc

    def make_local_fn(rid):
        def local_fn(joint_action_seqs):
            total = sum(_simulate_episode(_sample_belief(), joint_action_seqs, model)
                        for _ in range(n_mc))
            null_seqs = dict(joint_action_seqs)
            null_seqs[rid] = []
            null_total = sum(_simulate_episode(_sample_belief(), null_seqs, model)
                             for _ in range(n_mc))
            return (total - null_total) / n_mc
        return local_fn

    local_obj_fns = {rid: make_local_fn(rid) for rid in range(N_AGENTS)}
    return global_obj, local_obj_fns


# ── COMMUNICATION MODEL ────────────────────────────────────────────────────────

def can_communicate(s0, s1):
    """Tiger: agents always in the same room — comm unrestricted."""
    return True


# ── BENCHMARK FACTORY ──────────────────────────────────────────────────────────

def make_tiger_pomdp_benchmark(horizon=DEFAULT_HORIZON,
                                n_particles=N_PARTICLES,
                                n_mc=N_MC_SAMPLES):
    """
    Returns everything needed to run Dec-MCTS on the Tiger POMDP.

    Returns
    -------
    robot_ids     : [0, 1]
    init_states   : {rid: TigerBeliefState}   uniform belief, step=0
    global_obj    : callable  (path format, MC estimated)
    local_obj_fns : {rid: callable}  (action-seq format, MC estimated)
    can_communicate : callable (s0, s1) -> bool
    """
    model     = TigerModel()
    robot_ids = [0, 1]

    # Uniform initial particles
    base = n_particles // N_STATES
    init_particles = [s for s in range(N_STATES) for _ in range(base)]

    init_states = {
        rid: TigerBeliefState(
            particles = list(init_particles),
            step      = 0,
            max_steps = horizon,
            model     = model,
            agent_id  = rid,
        )
        for rid in robot_ids
    }

    global_obj, local_obj_fns = make_tiger_pomdp_objectives(model, n_mc)
    return robot_ids, init_states, global_obj, local_obj_fns, can_communicate


# import sys
# import os
# import random
# import math
# import time

# # Dynamic pathing to find decmcts.py in the root folder
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from decmcts import DecMCTS, DecMCTSTeam

# # Actions: 0 = LISTEN, 1 = OPEN_LEFT, 2 = OPEN_RIGHT
# LISTEN, OPEN_LEFT, OPEN_RIGHT = 0, 1, 2

# class TigerSearchState:
#     def __init__(self, depth=0, max_horizon=8):
#         self.depth = depth
#         self.max_horizon = max_horizon

#     def get_legal_actions(self):
#         return [] if self.is_terminal_state() else [LISTEN, OPEN_LEFT, OPEN_RIGHT]

#     def take_action(self, action):
#         return TigerSearchState(self.depth + 1, self.max_horizon)

#     def is_terminal_state(self):
#         return self.depth >= self.max_horizon

# class TigerUtility:
#     def __init__(self, horizon=8, n_samples=40):
#         self.horizon = horizon
#         self.n_samples = n_samples

#     def g(self, joint_sequences):
#         """Global objective function g(x)[cite: 164]."""
#         if not joint_sequences: return 0.0
#         total_reward = 0.0
#         robot_ids = list(joint_sequences.keys())
        
#         for _ in range(self.n_samples):
#             tiger_pos = random.randint(0, 1) # Hidden state Psi [cite: 399]
#             sample_reward = 0.0
#             for t in range(self.horizon):
#                 actions = {rid: joint_sequences[rid][t] 
#                            if t < len(joint_sequences[rid]) else LISTEN 
#                            for rid in robot_ids}
                
#                 opened = False
#                 for rid, act in actions.items():
#                     if act == LISTEN: sample_reward -= 1
#                     elif act == OPEN_LEFT:
#                         opened = True
#                         sample_reward += 10 if tiger_pos == 1 else -100
#                     elif act == OPEN_RIGHT:
#                         opened = True
#                         sample_reward += 10 if tiger_pos == 0 else -100
                
#                 if opened: tiger_pos = random.randint(0, 1)
#             total_reward += sample_reward
            
#         return total_reward / self.n_samples

#     def local_utility_fn(self, robot_id, joint_sequences):
#         """Local utility f^r = g(x) - g(x_phi)[cite: 201, 202]."""
#         g_joint = self.g(joint_sequences)
#         no_reward_seqs = joint_sequences.copy()
#         no_reward_seqs[robot_id] = []
#         return g_joint - self.g(no_reward_seqs)

# # --- QMDP HEURISTIC PATCH ---
# def qmdp_rollout(self, node):
#     """
#     Injected rollout policy: replaces random simulation with QMDP logic[cite: 114].
#     It samples the hidden state and follows the MDP-optimal action.
#     """
#     state = node.state
#     actions = []
#     depth = node.depth if hasattr(node, 'depth') else 0
    
#     # Sample true state for this rollout simulation [cite: 274, 400]
#     tiger_pos = random.randint(0, 1) 
    
#     while not state.is_terminal_state() and len(node.action_sequence) + len(actions) < state.max_horizon:
#         # QMDP Heuristic: If I 'know' state, I take the optimal MDP action
#         # Tiger Left (0) -> Open Right (2) | Tiger Right (1) -> Open Left (1)
#         a = OPEN_RIGHT if tiger_pos == 0 else OPEN_LEFT
#         actions.append(a)
#         state = state.take_action(a)
#     return actions

# def run_benchmark():
#     HORIZON = 8
#     ROBOT_IDS = ['agent_1', 'agent_2']
#     utility = TigerUtility(horizon=HORIZON, n_samples=50)

#     print(f"--- SDEC-TIGER (Horizon {HORIZON}) WITH QMDP HEURISTIC ---", flush=True)
    
#     planners = {}
#     for rid in ROBOT_IDS:
#         init_state = TigerSearchState(max_horizon=HORIZON)
#         def f_r(joint, rid=rid): return utility.local_utility_fn(rid, joint)

#         planner = DecMCTS(
#             robot_id=rid, robot_ids=ROBOT_IDS, init_state=init_state,
#             local_utility_fn=f_r, Cp=0.5, tau=800, num_samples=1000, num_seq=10,
#             gamma=0.95, beta_init=2.0, beta_decay=0.99, alpha=0.05
#         )
        
#         # Monkey-patch: Replace random rollout with QMDP [cite: 271, 275]
#         planner._rollout = qmdp_rollout.__get__(planner, DecMCTS)
#         planners[rid] = planner

#     team = DecMCTSTeam(planners)
#     start_time = time.time()

#     for n in range(41):
#         team.iterate_and_communicate(n_outer=1)
#         if n % 5 == 0:
#             elapsed = time.time() - start_time
#             print(f"Iter {n} | Time: {elapsed:.1f}s | Entropies: {team.entropies()}", flush=True)

#     print("\n--- RESULTS ---", flush=True)
#     best_plans = team.best_sequences()
#     final_joint = {rid: list(plan) for rid, plan in best_plans.items()}
    
#     for rid, seq in final_joint.items():
#         print(f"{rid} Plan: {seq}")
    
#     print("\n--- POLICY EXTRACTION ---", flush=True)
#     for rid, planner in planners.items():
#         print(f"\nTop 3 Plans for {rid}:")
#         policy = planner.get_full_policy()
#         for seq, prob in policy:
#             print(f"  Prob {prob:.4f}: {list(seq)}")
        
#     print(f"\nFinal Expected Value: {utility.g(final_joint):.5f}")
#     print(f"Target Value: 12.21726")

# if __name__ == "__main__":
#     run_benchmark()