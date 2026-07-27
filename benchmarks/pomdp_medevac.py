"""
pomdp_medevac.py
----------------
Maritime MedEvac Dec-POMDP benchmark for Dec-MCTS.

Exact T/O/R formulation from Mahdi Al-Husseini's RSSDA implementation
(maritimemedevac.py). Mapped to the belief-state Dec-MCTS interface
following the same pattern as pomdp_tiger.py.

State space  : (helo_x, helo_y, ship_x, ship_y, carry) — 512 states
Actions      : WAIT=0, ADVANCE=1, EXCHANGE=2  (per agent, 9 joint)
Observations : at-target=1, not-at-target=0   (per agent, 4 joint)

Key dynamics:
  - Helicopter moves with p=0.95, ship with p=0.85 (stochastic)
  - Pickup requires both agents at patient AND both choose EXCHANGE
  - Dropoff requires both agents at hospital AND both choose EXCHANGE
  - Failed solo exchange: penalty -6.0
"""

import random

# ── CONSTANTS ─────────────────────────────────────────────────────────────────

G             = 4
N_AGENTS      = 2
ACT_PER_AGENT = 3   # WAIT=0, ADVANCE=1, EXCHANGE=2
OBS_PER_AGENT = 2   # not-at-target=0, at-target=1
N_STATES      = G * G * G * G * 2   # 512
N_ACTS        = ACT_PER_AGENT ** N_AGENTS   # 9
N_OBS         = OBS_PER_AGENT ** N_AGENTS   # 4

PATIENT  = (1, 1)
HOSPITAL = (3, 3)
HELO_START = (0, 1)
SHIP_START = (1, 0)

P_MOVE_HELO = 0.95
P_MOVE_SHIP = 0.85
P_PICKUP    = 0.95
P_DROPOFF   = 0.95

STEP_COST               = -0.3
WRONG_EXCHANGE_COST     = -1.0
PICKUP_REWARD           =  5.0
DROPOFF_REWARD          = 12.0
PICKUP_MISMATCH_PENALTY = -6.0
DROPOFF_MISMATCH_PENALTY= -6.0

N_PARTICLES  = 200
N_MC_SAMPLES = 50
DEFAULT_HORIZON = 6


# ── STATE ENCODING ────────────────────────────────────────────────────────────

def _pos_to_idx(x, y):
    return y * G + x

def _idx_to_pos(idx):
    return (idx % G, idx // G)

def _state_id(px, py, bx, by, carry):
    return carry * (G * G * G * G) + _pos_to_idx(px, py) * (G * G) + _pos_to_idx(bx, by)

def _id_to_state(s):
    carry = 1 if s >= (G * G * G * G) else 0
    rem   = s - carry * (G * G * G * G)
    ppos  = rem // (G * G)
    bpos  = rem  % (G * G)
    px, py = _idx_to_pos(ppos)
    bx, by = _idx_to_pos(bpos)
    return (px, py, bx, by, carry)


# ── MEDEVAC POMDP MODEL ───────────────────────────────────────────────────────

def _next_step_toward(x, y, tx, ty, prefer='H'):
    if (x, y) == (tx, ty):
        return (x, y)
    if prefer == 'H':
        if x < tx: return (x + 1, y)
        if y < ty: return (x, y + 1)
    else:
        if y < ty: return (x, y + 1)
        if x < tx: return (x + 1, y)
    return (x, y)

def _advance_helo(x, y, carry):
    tx, ty  = PATIENT if carry == 0 else HOSPITAL
    nx, ny  = _next_step_toward(x, y, tx, ty, 'H')
    p_succ  = P_MOVE_HELO if (nx, ny) != (x, y) else 1.0
    return nx, ny, p_succ

def _advance_ship(x, y, carry):
    tx, ty  = PATIENT if carry == 0 else HOSPITAL
    nx, ny  = _next_step_toward(x, y, tx, ty, 'V')
    p_succ  = P_MOVE_SHIP if (nx, ny) != (x, y) else 1.0
    return nx, ny, p_succ

def _at_target_for_agent(x, y, carry):
    target = PATIENT if carry == 0 else HOSPITAL
    return 1 if (x, y) == target else 0


def build_medevac_problem():
    """
    Build T, O, R matrices exactly matching Mahdi's maritimemedevac.py.

    Returns
    -------
    T : list[float], length N_ACTS * N_STATES^2
        T[a * N_STATES^2 + s * N_STATES + s'] = P(s' | s, a)
    O : list[float], length N_ACTS * N_STATES * N_OBS
        O[a * N_STATES * N_OBS + s' * N_OBS + o] = P(o | a, s')
    R : list[float], length N_ACTS * N_STATES
        R[a * N_STATES + s] = R(s, a)
    init_belief : list[float], length N_STATES  (deterministic start)
    """
    nsq = N_STATES * N_STATES
    nso = N_STATES * N_OBS

    T = [0.0] * (nsq  * N_ACTS)
    O = [0.0] * (nso  * N_ACTS)
    R = [0.0] * (N_STATES * N_ACTS)

    for a0 in range(ACT_PER_AGENT):
        for a1 in range(ACT_PER_AGENT):
            act = a0 + ACT_PER_AGENT * a1

            for s in range(N_STATES):
                px, py, bx, by, carry = _id_to_state(s)

                helo_at_pat  = (px, py) == PATIENT
                boat_at_pat  = (bx, by) == PATIENT
                helo_at_hosp = (px, py) == HOSPITAL
                boat_at_hosp = (bx, by) == HOSPITAL

                # Reward
                r = STEP_COST
                if a0 == 2 and not (helo_at_pat or helo_at_hosp):
                    r += WRONG_EXCHANGE_COST
                if a1 == 2 and not (boat_at_pat or boat_at_hosp):
                    r += WRONG_EXCHANGE_COST

                if carry == 0:
                    if helo_at_pat and boat_at_pat and a0 == 2 and a1 == 2:
                        r += P_PICKUP * PICKUP_REWARD
                    if ((a0 == 2 and helo_at_pat and not (a1 == 2 and boat_at_pat)) or
                            (a1 == 2 and boat_at_pat and not (a0 == 2 and helo_at_pat))):
                        r += PICKUP_MISMATCH_PENALTY
                else:
                    if helo_at_hosp and boat_at_hosp and a0 == 2 and a1 == 2:
                        r += P_DROPOFF * DROPOFF_REWARD
                    if ((a0 == 2 and helo_at_hosp and not (a1 == 2 and boat_at_hosp)) or
                            (a1 == 2 and boat_at_hosp and not (a0 == 2 and helo_at_hosp))):
                        r += DROPOFF_MISMATCH_PENALTY

                R[act * N_STATES + s] = r

                # Transition: compute (nx_p, ny_p) and (nx_b, ny_b) with stochastic success
                if a0 == 1:
                    nx_p, ny_p, p_succ_p = _advance_helo(px, py, carry)
                else:
                    nx_p, ny_p, p_succ_p = px, py, 1.0

                if a1 == 1:
                    nx_b, ny_b, p_succ_b = _advance_ship(bx, by, carry)
                else:
                    nx_b, ny_b, p_succ_b = bx, by, 1.0

                # Four movement outcome combinations
                cases = [
                    (nx_p, ny_p, nx_b, ny_b, p_succ_p * p_succ_b),
                    (nx_p, ny_p, bx,   by,   p_succ_p * (1.0 - p_succ_b)),
                    (px,   py,   nx_b, ny_b, (1.0 - p_succ_p) * p_succ_b),
                    (px,   py,   bx,   by,   (1.0 - p_succ_p) * (1.0 - p_succ_b)),
                ]
                # EXCHANGE = stay in place
                if a0 == 2:
                    cases = [(px, py, xb, yb, p) if (xp == px and yp == py) else (px, py, xb, yb, 0.0)
                             for (xp, yp, xb, yb, p) in cases]
                if a1 == 2:
                    cases = [(xp, yp, bx, by, p) if (xb == bx and yb == by) else (xp, yp, bx, by, 0.0)
                             for (xp, yp, xb, yb, p) in cases]

                for (px2, py2, bx2, by2, p_case) in cases:
                    if p_case == 0.0:
                        continue
                    if carry == 0 and helo_at_pat and boat_at_pat and a0 == 2 and a1 == 2:
                        s_succ = _state_id(px2, py2, bx2, by2, 1)
                        s_fail = _state_id(px2, py2, bx2, by2, 0)
                        T[act * nsq + s * N_STATES + s_succ] += p_case * P_PICKUP
                        T[act * nsq + s * N_STATES + s_fail] += p_case * (1.0 - P_PICKUP)
                    elif carry == 1 and helo_at_hosp and boat_at_hosp and a0 == 2 and a1 == 2:
                        s_succ = _state_id(px2, py2, bx2, by2, 0)
                        s_fail = _state_id(px2, py2, bx2, by2, 1)
                        T[act * nsq + s * N_STATES + s_succ] += p_case * P_DROPOFF
                        T[act * nsq + s * N_STATES + s_fail] += p_case * (1.0 - P_DROPOFF)
                    else:
                        s2 = _state_id(px2, py2, bx2, by2, carry)
                        T[act * nsq + s * N_STATES + s2] += p_case

            # Observation matrix: deterministic — agents observe whether they're at target
            for s2 in range(N_STATES):
                px2, py2, bx2, by2, carry2 = _id_to_state(s2)
                o0 = _at_target_for_agent(px2, py2, carry2)
                o1 = _at_target_for_agent(bx2, by2, carry2)
                o  = o0 + OBS_PER_AGENT * o1
                O[act * nso + s2 * N_OBS + o] = 1.0

    init_belief = [0.0] * N_STATES
    init_belief[_state_id(HELO_START[0], HELO_START[1],
                          SHIP_START[0], SHIP_START[1], 0)] = 1.0

    return T, O, R, init_belief


# ── PARTICLE FILTER HELPERS ───────────────────────────────────────────────────

class MedEvacModel:
    """Holds pre-built T/O/R matrices for sampling."""
    def __init__(self):
        self.T, self.O, self.R, self.init_belief = build_medevac_problem()
        # Attributes required by cpde_decmcts.py
        self.n_states      = N_STATES
        self.n_obs         = N_OBS
        self.act_per_agent = ACT_PER_AGENT
        self.obs_per_agent = OBS_PER_AGENT

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

class MedEvacBeliefState:
    """
    Particle-filter belief state for MedEvac Dec-MCTS.

    vertex = (step, quantized carry probability) — coarse enough to allow
    tree node reuse across similar beliefs.
    """
    __slots__ = ("particles", "step", "max_steps", "model", "agent_id", "vertex")

    def __init__(self, particles, step, max_steps, model, agent_id):
        self.particles = particles
        self.step      = step
        self.max_steps = max_steps
        self.model     = model
        self.agent_id  = agent_id
        n = len(particles) or 1
        p_carry = sum(1 for s in particles if _id_to_state(s)[4] == 1) / n
        self.vertex = (step, round(p_carry * 10) / 10)

    def get_legal_actions(self):
        if self.step >= self.max_steps:
            return []
        return list(range(ACT_PER_AGENT))   # WAIT, ADVANCE, EXCHANGE

    def take_action(self, my_action):
        m = self.model
        new_particles = []
        for s in self.particles:
            other = random.randint(0, ACT_PER_AGENT - 1)
            joint_a = (my_action + other * ACT_PER_AGENT
                       if self.agent_id == 0
                       else other + my_action * ACT_PER_AGENT)
            new_particles.append(m.sample_next_state(s, joint_a))
        return MedEvacBeliefState(new_particles, self.step + 1,
                                  self.max_steps, m, self.agent_id)

    def is_terminal_state(self):
        return self.step >= self.max_steps


# ── OBJECTIVE FUNCTIONS ───────────────────────────────────────────────────────

def _simulate_episode(init_state, joint_action_seqs, model, null_action=0):
    """Simulate one episode from init_state under joint_action_seqs (WAIT padding)."""
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

def make_medevac_pomdp_benchmark(horizon=DEFAULT_HORIZON,
                                  n_particles=N_PARTICLES,
                                  n_mc=N_MC_SAMPLES):
    """
    Returns everything needed to run online Dec-MCTS on MedEvac Dec-POMDP.

    Returns
    -------
    robot_ids, init_states, global_obj, local_obj_fns, model, init_particles
    """
    model     = MedEvacModel()
    robot_ids = [0, 1]

    # Deterministic initial state — both at starts, no patient
    init_s = _state_id(HELO_START[0], HELO_START[1],
                       SHIP_START[0], SHIP_START[1], 0)
    init_particles = [init_s] * n_particles

    belief_states = {
        rid: MedEvacBeliefState(list(init_particles), 0, horizon, model, rid)
        for rid in robot_ids
    }

    global_obj, local_obj_fns = make_objectives_from_particles(
        init_particles, model, n_mc
    )
    return robot_ids, belief_states, global_obj, local_obj_fns, model, init_particles
