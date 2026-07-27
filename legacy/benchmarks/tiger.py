"""
tiger.py
--------
Tiger benchmark for SDecMCTS (MDP relaxation: true tiger position known).

Based on: Nair et al. (2003), "Taming Decentralized POMDPs"

2 agents, discrete actions, fixed horizon.

State space  : tiger position ∈ {LEFT=0, RIGHT=1}  (shared global truth)
Action space : OPEN_LEFT=0, OPEN_RIGHT=1, LISTEN=2
Horizon      : configurable (default 5)

Reward (per step, summed over both agents):
  - Both listen             : -2
  - Open safe door (each)   : +10
  - Open tiger's door (each): -100

NOTE — Two function signatures are required by SDecMCTS:

  global_obj(joint_paths)
      joint_paths = {rid: [tiger_pos, a1, a2, ...]}   ← path format (SDecMCTS/CenMCTS)

  local_obj_fns[rid](joint_action_seqs)
      joint_action_seqs = {rid: [a1, a2, ...]}         ← pure action seqs (DecMCTS)
"""

OPEN_LEFT, OPEN_RIGHT, LISTEN = 0, 1, 2
TIGER_LEFT, TIGER_RIGHT = 0, 1
DEFAULT_HORIZON = 5


# ── STATE ─────────────────────────────────────────────────────────────────────

class TigerState:
    """
    Per-robot state for Tiger.

    vertex   = tiger_pos  (used as the initial path element in SDecMCTS/CenMCTS)
    Actions  = integers: OPEN_LEFT=0, OPEN_RIGHT=1, LISTEN=2

    MDP relaxation: tiger position is fixed throughout the episode (no
    stochastic reset after door open).  This gives an upper-bound planner
    that knows exactly where the tiger is and never loses that information.
    """
    __slots__ = ("vertex", "tiger_pos", "step", "max_steps")

    def __init__(self, tiger_pos, step=0, max_steps=DEFAULT_HORIZON):
        self.tiger_pos = tiger_pos
        self.step      = step
        self.max_steps = max_steps
        self.vertex    = tiger_pos   # path element

    def get_legal_actions(self):
        if self.step >= self.max_steps:
            return []
        return [OPEN_LEFT, OPEN_RIGHT, LISTEN]

    def take_action(self, action):
        # Tiger stays in place (MDP relaxation)
        return TigerState(self.tiger_pos, self.step + 1, self.max_steps)

    def is_terminal_state(self):
        return self.step >= self.max_steps


# ── REWARD ────────────────────────────────────────────────────────────────────

def _step_reward(tiger_pos, a0, a1):
    """Immediate joint reward for one time step."""
    r = 0.0
    for a in (a0, a1):
        if   a == LISTEN:     r -= 1.0
        elif a == OPEN_LEFT:  r += -100.0 if tiger_pos == TIGER_LEFT  else 10.0
        elif a == OPEN_RIGHT: r += -100.0 if tiger_pos == TIGER_RIGHT else 10.0
    return r


def _reward_from_seqs(tiger_pos, seq0, seq1, null_action=LISTEN):
    """
    Sum reward over zip(seq0, seq1).
    Pads the shorter sequence with null_action (default: LISTEN).
    """
    max_len = max(len(seq0), len(seq1)) if (seq0 or seq1) else 0
    s0 = list(seq0) + [null_action] * (max_len - len(seq0))
    s1 = list(seq1) + [null_action] * (max_len - len(seq1))
    return sum(_step_reward(tiger_pos, a0, a1) for a0, a1 in zip(s0, s1))


# ── OBJECTIVES ────────────────────────────────────────────────────────────────

def make_tiger_objective(tiger_pos, horizon=DEFAULT_HORIZON):
    """
    Returns
    -------
    global_obj : callable  {0: [v0,a1,...], 1: [v0,a1,...]} -> float
        For SDecMCTS / CenMCTS rollout evaluation.
        Path format: first element = initial vertex (tiger_pos), rest = actions.

    local_obj_fns : dict  {rid: callable}
        For DecMCTS local utility.
        Receives pure action sequences {rid: [a1, a2, ...]} (no initial vertex).
        Missing or short sequences are padded with LISTEN.
    """
    null_actions = [LISTEN] * horizon

    # ── global_obj (path format) ──────────────────────────────────────────────
    def global_obj(joint_paths):
        seq0 = list(joint_paths[0][1:])  # skip initial vertex
        seq1 = list(joint_paths[1][1:])
        return _reward_from_seqs(tiger_pos, seq0, seq1)

    # ── local_obj_fns (pure action-seq format) ────────────────────────────────
    def make_local_fn(rid):
        other = 1 - rid

        def local_fn(joint_action_seqs):
            my_seq    = list(joint_action_seqs.get(rid,   []))
            other_seq = list(joint_action_seqs.get(other, []))
            
            # Absolute reward
            if rid == 0:
                total = _reward_from_seqs(tiger_pos, my_seq, other_seq)
                null_total = _reward_from_seqs(tiger_pos, [], other_seq) # Empty means all padded with LISTEN
            else:
                total = _reward_from_seqs(tiger_pos, other_seq, my_seq)
                null_total = _reward_from_seqs(tiger_pos, other_seq, [])
                
            return total - null_total

        return local_fn

    local_obj_fns = {0: make_local_fn(0), 1: make_local_fn(1)}
    return global_obj, local_obj_fns


# ── BENCHMARK FACTORY ─────────────────────────────────────────────────────────

def can_communicate(s0, s1):
    """
    Tiger: agents are always in the same room — communication is unrestricted.
    """
    return True


def make_tiger_benchmark(tiger_pos=TIGER_LEFT, horizon=DEFAULT_HORIZON):
    """
    Returns everything needed to run SDecMCTS / CenMCTS / DecMCTS on Tiger.

    Parameters
    ----------
    tiger_pos : TIGER_LEFT (0) or TIGER_RIGHT (1)
    horizon   : episode length

    Returns
    -------
    robot_ids     : [0, 1]
    init_states   : {rid: TigerState}
    global_obj    : callable (path format)
    local_obj_fns : {rid: callable} (action-seq format)
    can_comm_fn   : callable (s0, s1) -> bool
    """
    robot_ids   = [0, 1]
    init_states = {rid: TigerState(tiger_pos, 0, horizon) for rid in robot_ids}
    global_obj, local_obj_fns = make_tiger_objective(tiger_pos, horizon)
    return robot_ids, init_states, global_obj, local_obj_fns, can_communicate
