"""
medevac.py
----------
Maritime MedEvac benchmark for SDecMCTS (MDP relaxation: deterministic nav).

Based on: RSSDA maritime medical evacuation domain.

2 agents (helicopter + ship), 4×4 grid, fixed horizon.

Mission:
  1. Helicopter (robot 0) navigates to PATIENT_LOC  → pickup reward
  2. Helicopter and Ship meet at the same cell       → handoff reward
  3. Ship (robot 1) navigates to HOSPITAL            → delivery reward

Per-step cost applied to total path length.

Actions: move to adjacent cell OR stay  (position tuples, like orienteering)

NOTE — Two function signatures (same convention as tiger.py / mars.py):

  global_obj(joint_paths)
      joint_paths = {0: [(r,c),...], 1: [(r,c),...]}  ← position sequences

  local_obj_fns[rid](joint_action_seqs)
      joint_action_seqs = {rid: [(r,c),...]}           ← actions only
"""

GRID              = 4
DEFAULT_HORIZON   = 25

PATIENT_LOC       = (1, 1)
HOSPITAL          = (3, 3)
HELO_START        = (0, 1)   # robot 0 (helicopter)
SHIP_START        = (1, 0)   # robot 1 (ship)

STEP_COST         = -0.3
PICKUP_REWARD     =  5.0
HANDOFF_REWARD    =  3.0
DELIVERY_REWARD   = 12.0


# ── STATE ─────────────────────────────────────────────────────────────────────

class MedEvacState:
    """
    Per-robot state: grid position + step counter.
    Actions = destination (row, col) tuples (adjacent cells or stay).
    vertex  = (row, col).
    """
    __slots__ = ("vertex", "row", "col", "step", "max_steps")

    def __init__(self, row, col, step=0, max_steps=DEFAULT_HORIZON):
        self.row       = row
        self.col       = col
        self.step      = step
        self.max_steps = max_steps
        self.vertex    = (row, col)

    def get_legal_actions(self):
        if self.step >= self.max_steps:
            return []
        r, c = self.row, self.col
        moves = [(r, c)]                         # wait
        if r > 0:        moves.append((r-1, c))
        if r < GRID - 1: moves.append((r+1, c))
        if c > 0:        moves.append((r, c-1))
        if c < GRID - 1: moves.append((r, c+1))
        return moves

    def take_action(self, action):
        r, c = action
        return MedEvacState(r, c, self.step + 1, self.max_steps)

    def is_terminal_state(self):
        return self.step >= self.max_steps


# ── REWARD ────────────────────────────────────────────────────────────────────

def _score_medevac(helo_path, ship_path):
    """
    Compute MedEvac reward from two position-sequence lists.

    Sequence: [start_pos, pos_1, pos_2, ...]
    """
    n = max(len(helo_path), len(ship_path))

    # Pad shorter path by repeating last position (robot stopped)
    hp = list(helo_path)
    sp = list(ship_path)
    while len(hp) < n: hp.append(hp[-1])
    while len(sp) < n: sp.append(sp[-1])

    reward = STEP_COST * (n - 1)   # per-step cost (n-1 transitions)

    # 1. Helicopter picks up patient
    patient_step = next(
        (i for i, pos in enumerate(hp) if pos == PATIENT_LOC), None
    )
    if patient_step is None:
        return reward   # no pickup — no further rewards possible

    reward += PICKUP_REWARD

    # 2. Handoff: first step where both robots at same cell AFTER patient picked up
    handoff_step = None
    for t in range(patient_step, n):
        if hp[t] == sp[t]:
            handoff_step = t
            reward += HANDOFF_REWARD
            break

    if handoff_step is None:
        return reward   # no handoff

    # 3. Delivery: ship visits hospital at any step AFTER handoff
    if HOSPITAL in sp[handoff_step:]:
        reward += DELIVERY_REWARD

    return reward


# ── OBJECTIVES ────────────────────────────────────────────────────────────────

def make_medevac_objective(init_positions):
    """
    Parameters
    ----------
    init_positions : {0: (r,c), 1: (r,c)}

    Returns
    -------
    global_obj    : {rid: path_list} -> float
    local_obj_fns : {rid: callable} (pure action-seq format for DecMCTS)
    """
    robot_ids = [0, 1]

    # ── global_obj (path format) ──────────────────────────────────────────────
    def global_obj(joint_paths):
        helo = list(joint_paths[0])
        ship = list(joint_paths[1])
        return _score_medevac(helo, ship)

    # ── local_obj_fns (pure action-seq format) ────────────────────────────────
    def make_local_fn(rid):
        def local_fn(joint_action_seqs):
            joint_paths = {}
            for r, seq in joint_action_seqs.items():
                ip = init_positions.get(r, init_positions[rid])
                joint_paths[r] = [ip] + list(seq)
            for r in robot_ids:
                if r not in joint_paths:
                    joint_paths[r] = [init_positions[r]]

            # Marginal contribution: G(x^r ∪ x^{-r}) - G(∅ ∪ x^{-r})
            # Null path for rid = stay at initial position (no actions taken).
            null_paths = dict(joint_paths)
            null_paths[rid] = [init_positions[rid]]
            return global_obj(joint_paths) - global_obj(null_paths)

        return local_fn

    local_obj_fns = {rid: make_local_fn(rid) for rid in robot_ids}
    return global_obj, local_obj_fns


# ── BENCHMARK FACTORY ─────────────────────────────────────────────────────────

def can_communicate(s0, s1):
    """
    MedEvac: helicopter and ship can communicate when they are at the
    same cell (short-range radio / line-of-sight at handoff point).
    """
    return s0.vertex == s1.vertex


def make_medevac_benchmark(horizon=DEFAULT_HORIZON):
    """
    Returns
    -------
    robot_ids, init_states, global_obj, local_obj_fns, can_communicate
    """
    starts = {0: HELO_START, 1: SHIP_START}
    robot_ids = [0, 1]
    init_states = {
        rid: MedEvacState(r, c, 0, horizon)
        for rid, (r, c) in starts.items()
    }
    global_obj, local_obj_fns = make_medevac_objective(starts)
    return robot_ids, init_states, global_obj, local_obj_fns, can_communicate
