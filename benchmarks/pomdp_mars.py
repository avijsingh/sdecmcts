"""
pomdp_mars.py
-------------
Mars Rover Dec-POMDP benchmark for Dec-MCTS.

T/O/R matrices loaded from mars.data (Amato et al. domain), exactly as in
Mahdi Al-Husseini's RSSDA sdec_mars.py. Mapped to the belief-state Dec-MCTS
interface following the same pattern as pomdp_tiger.py.

State space  : joint rover positions on 4x4 grid — 256 states (16 x 16)
Actions      : 6 per agent (36 joint), loaded from mars.data
Observations : 8 per agent (64 joint), loaded from mars.data
Init belief  : both rovers at grid position 0 (top-left corner)

mars.data format (one entry per line):
  T: a1 a2 : start_s : end_s : prob
  O: a1 a2 : end_s : obs1 obs2 : prob
  R: a1 a2 start_s ... reward
"""

import os
import random

# ── CONSTANTS ──────────────────────────────────────────────────────────────────

N_AGENTS      = 2
N_STATES      = 256
ACT_PER_AGENT = 6
OBS_PER_AGENT = 8
N_ACTS        = ACT_PER_AGENT ** N_AGENTS   # 36
N_OBS         = OBS_PER_AGENT ** N_AGENTS   # 64

N_PARTICLES   = 200
N_MC_SAMPLES  = 50
DEFAULT_HORIZON = 6

_DATA_FILE = os.path.join(os.path.dirname(__file__), "mars.data")


# ── MARS POMDP MODEL ──────────────────────────────────────────────────────────

class MarsModel:
    """
    Loads T/O/R from mars.data and exposes sampling interface.

    mars.data line formats (Mahdi's parser in MarsProblemLoader):
      T: a1 a2 : start_s : end_s : prob
        → act = a1 + ACT_PER_AGENT * a2
        → T[act * N_STATES^2 + start_s * N_STATES + end_s] = prob

      O: a1 a2 : end_s : obs1 obs2 : prob
        → act = a1 + ACT_PER_AGENT * a2
        → o   = obs1 + OBS_PER_AGENT * obs2
        → O[act * N_STATES * N_OBS + end_s * N_OBS + o] = prob

      R: a1 a2 : start_s : ... : reward   (reward is the last token)
        → R[act * N_STATES + start_s] = reward
    """

    def __init__(self, data_file=_DATA_FILE):
        nsq = N_STATES * N_STATES
        nso = N_STATES * N_OBS
        self.T = [0.0] * (nsq  * N_ACTS)
        self.O = [0.0] * (nso  * N_ACTS)
        self.R = [0.0] * (N_STATES * N_ACTS)

        # Both rovers start at grid position 0
        self.init_belief = [0.0] * N_STATES
        self.init_belief[0] = 1.0

        # Attributes required by cpde_decmcts.py
        self.n_states      = N_STATES
        self.n_obs         = N_OBS
        self.act_per_agent = ACT_PER_AGENT
        self.obs_per_agent = OBS_PER_AGENT

        self._load(data_file)

    def _load(self, path):
        with open(path, "r") as f:
            for line in f:
                d = line.split()
                if not d:
                    continue
                kind = d[0][0]
                if kind == "T":
                    # T: a1 a2 : start_s : end_s : prob
                    a1, a2  = int(d[1]), int(d[2])
                    act     = a1 + ACT_PER_AGENT * a2
                    start_s = int(d[4])
                    end_s   = int(d[6])
                    prob    = float(d[8])
                    self.T[act * N_STATES * N_STATES + start_s * N_STATES + end_s] = prob

                elif kind == "O":
                    # O: a1 a2 : end_s : obs1 obs2 : prob
                    a1, a2  = int(d[1]), int(d[2])
                    act     = a1 + ACT_PER_AGENT * a2
                    end_s   = int(d[4])
                    obs1, obs2 = int(d[6]), int(d[7])
                    o       = obs1 + OBS_PER_AGENT * obs2
                    prob    = float(d[9])
                    self.O[act * N_STATES * N_OBS + end_s * N_OBS + o] = prob

                elif kind == "R":
                    # R: a1 a2 : start_s : ... : reward  (reward = last token)
                    a1, a2  = int(d[1]), int(d[2])
                    act     = a1 + ACT_PER_AGENT * a2
                    start_s = int(d[4])
                    reward  = float(d[-1])
                    self.R[act * N_STATES + start_s] = reward

    def sample_next_state(self, s, joint_a):
        r   = random.random()
        cum = 0.0
        base = joint_a * N_STATES * N_STATES + s * N_STATES
        for sp in range(N_STATES):
            cum += self.T[base + sp]
            if r <= cum:
                return sp
        return N_STATES - 1

    def reward(self, joint_a, s):
        return self.R[joint_a * N_STATES + s]


# ── PARTICLE FILTER HELPERS ───────────────────────────────────────────────────

def sample_obs(model, joint_a, true_state):
    """Sample joint observation o ~ O(· | true_state, joint_a)."""
    base = joint_a * N_STATES * N_OBS + true_state * N_OBS
    r    = random.random()
    cum  = 0.0
    for o in range(N_OBS):
        cum += model.O[base + o]
        if r <= cum:
            return o
    return N_OBS - 1


def update_particles(particles, joint_a, joint_obs, model, n_resample=None):
    """Particle filter update: propagate through T, weight by O, resample."""
    if n_resample is None:
        n_resample = len(particles)

    new_states  = []
    obs_weights = []
    for s in particles:
        s_next = model.sample_next_state(s, joint_a)
        w      = model.O[joint_a * N_STATES * N_OBS + s_next * N_OBS + joint_obs]
        new_states.append(s_next)
        obs_weights.append(w)

    total = sum(obs_weights)
    if total <= 1e-12:
        return new_states[:n_resample]

    probs = [w / total for w in obs_weights]

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


# ── BELIEF STATE ──────────────────────────────────────────────────────────────

class MarsBeliefState:
    """
    Particle-filter belief state for Mars Dec-MCTS.

    vertex = (step, quantized_dominant_state) — coarse discretization of
    the most likely joint rover position, allowing tree node reuse.
    """
    __slots__ = ("particles", "step", "max_steps", "model", "agent_id", "vertex")

    def __init__(self, particles, step, max_steps, model, agent_id):
        self.particles = particles
        self.step      = step
        self.max_steps = max_steps
        self.model     = model
        self.agent_id  = agent_id
        # Coarse belief summary: step + modal state (quantized to 5-state bins)
        n = len(particles) or 1
        counts = {}
        for s in particles:
            counts[s] = counts.get(s, 0) + 1
        modal = max(counts, key=counts.get) if counts else 0
        # Bin modal state into groups of 5 to allow node reuse
        self.vertex = (step, (modal // 5) * 5)

    def get_legal_actions(self):
        if self.step >= self.max_steps:
            return []
        return list(range(ACT_PER_AGENT))

    def take_action(self, my_action):
        m = self.model
        new_particles = []
        for s in self.particles:
            other   = random.randint(0, ACT_PER_AGENT - 1)
            joint_a = (my_action + other * ACT_PER_AGENT
                       if self.agent_id == 0
                       else other + my_action * ACT_PER_AGENT)
            new_particles.append(m.sample_next_state(s, joint_a))
        return MarsBeliefState(new_particles, self.step + 1,
                               self.max_steps, m, self.agent_id)

    def is_terminal_state(self):
        return self.step >= self.max_steps


# ── OBJECTIVE FUNCTIONS ───────────────────────────────────────────────────────

def _simulate_episode(init_state, joint_action_seqs, model, null_action=0):
    """Simulate one episode from init_state under joint_action_seqs."""
    s      = init_state
    seqs   = {rid: list(seq) for rid, seq in joint_action_seqs.items()}
    length = max((len(v) for v in seqs.values()), default=0)
    total  = 0.0
    for t in range(length):
        a0 = seqs.get(0, [])[t] if t < len(seqs.get(0, [])) else null_action
        a1 = seqs.get(1, [])[t] if t < len(seqs.get(1, [])) else null_action
        joint_a = a0 + a1 * ACT_PER_AGENT
        total  += model.reward(joint_a, s)
        s       = model.sample_next_state(s, joint_a)
    return total


def make_objectives_from_particles(particles, model, n_mc=N_MC_SAMPLES):
    """
    Build global_obj and local_obj_fns from current particle set.
    Used at each step in online replanning.
    """
    def _sample():
        return random.choice(particles)

    def global_obj(joint_paths):
        seqs  = {rid: list(path[1:]) for rid, path in joint_paths.items()}
        total = sum(_simulate_episode(_sample(), seqs, model) for _ in range(n_mc))
        return total / n_mc

    def make_local_fn(rid):
        def local_fn(joint_action_seqs):
            total = sum(_simulate_episode(_sample(), joint_action_seqs, model)
                        for _ in range(n_mc))
            null_seqs = dict(joint_action_seqs)
            null_seqs[rid] = []
            null_total = sum(_simulate_episode(_sample(), null_seqs, model)
                             for _ in range(n_mc))
            return (total - null_total) / n_mc
        return local_fn

    local_obj_fns = {rid: make_local_fn(rid) for rid in range(N_AGENTS)}
    return global_obj, local_obj_fns


# ── BENCHMARK FACTORY ─────────────────────────────────────────────────────────

def make_mars_pomdp_benchmark(horizon=DEFAULT_HORIZON,
                               n_particles=N_PARTICLES,
                               n_mc=N_MC_SAMPLES):
    """
    Returns everything needed to run online Dec-MCTS on Mars Dec-POMDP.

    Returns
    -------
    robot_ids, init_states, global_obj, local_obj_fns, model, init_particles
    """
    model     = MarsModel()
    robot_ids = [0, 1]

    # Both rovers start at state 0 (grid position 0)
    init_particles = [0] * n_particles

    belief_states = {
        rid: MarsBeliefState(list(init_particles), 0, horizon, model, rid)
        for rid in robot_ids
    }

    global_obj, local_obj_fns = make_objectives_from_particles(
        init_particles, model, n_mc
    )
    return robot_ids, belief_states, global_obj, local_obj_fns, model, init_particles
