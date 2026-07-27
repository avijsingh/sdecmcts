"""
mars.py
-------
Mars Rover benchmark for SDecMCTS (MDP relaxation: deterministic grid nav).

Loosely based on: Spaan & Melo (2008) and RSSDA mars domain.

2 rovers, 4×4 grid, fixed horizon.
Rovers navigate to rock locations to sample them, then return to base
to transmit.  Reward is awarded for rocks sampled and successfully
transmitted (base visited after sampling).

State  : grid position (row, col)      — one per robot, independent
Actions: move to adjacent cell OR stay  (tuples, like orienteering)
Reward :
  rock_value   per unique rock visited across the team
  transmit_bonus per rock that the SAME robot visits and then returns to base

NOTE — Two function signatures (same convention as tiger.py):

  global_obj(joint_paths)
      joint_paths = {rid: [(r0,c0), (r1,c1), ...]}   ← position sequence

  local_obj_fns[rid](joint_action_seqs)
      joint_action_seqs = {rid: [(r1,c1), (r2,c2), ...]}   ← actions only
"""

GRID            = 4
DEFAULT_HORIZON = 20

# Rock positions and their sample rewards
ROCKS = {
    (1, 1): 8.0,
    (2, 3): 6.0,
    (3, 0): 5.0,
}
BASE            = (0, 0)   # transmission / start location
TRANSMIT_BONUS  = 1.5      # per-rock multiplier for returning to base


# ── STATE ─────────────────────────────────────────────────────────────────────

class MarsState:
    """
    Per-robot state: grid position + step counter.
    Actions = destination (row, col) tuples (adjacent cells or stay).
    vertex  = (row, col)  — used as the initial path element.
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
        moves = [(r, c)]                         # stay
        if r > 0:        moves.append((r-1, c))  # up
        if r < GRID - 1: moves.append((r+1, c))  # down
        if c > 0:        moves.append((r, c-1))  # left
        if c < GRID - 1: moves.append((r, c+1))  # right
        return moves

    def take_action(self, action):
        r, c = action
        return MarsState(r, c, self.step + 1, self.max_steps)

    def is_terminal_state(self):
        return self.step >= self.max_steps


# ── REWARD ────────────────────────────────────────────────────────────────────

def _score_path(path_list):
    """
    Score a single robot's path.
      rock_reward      : sum of ROCKS[pos] for every unique rock in path
      transmit_reward  : rocks × TRANSMIT_BONUS if BASE visited after last rock
    """
    visited = set(path_list)
    rocks_hit = [pos for pos in ROCKS if pos in visited]

    rock_reward = sum(ROCKS[pos] for pos in rocks_hit)

    transmit_reward = 0.0
    if rocks_hit:
        last_rock_step = max(
            next(i for i, v in enumerate(path_list) if v == pos)
            for pos in rocks_hit
        )
        if any(v == BASE for v in path_list[last_rock_step + 1:]):
            transmit_reward = len(rocks_hit) * TRANSMIT_BONUS

    return rock_reward + transmit_reward


def _global_from_paths(joint_paths):
    """Compute global reward from a {rid: path_list} dict."""
    # De-duplicate rock credit across robots (first visitor scores)
    credited_rocks = set()
    total = 0.0

    for path in joint_paths.values():
        path_list = list(path)
        visited   = set(path_list)

        # Only credit rocks not already credited by another robot
        new_rocks = [pos for pos in ROCKS if pos in visited and pos not in credited_rocks]
        credited_rocks.update(new_rocks)
        rock_reward = sum(ROCKS[pos] for pos in new_rocks)

        # Transmit bonus for this robot's own round-trip
        transmit_reward = 0.0
        if new_rocks:
            last_step = max(
                next(i for i, v in enumerate(path_list) if v == pos)
                for pos in new_rocks
            )
            if any(v == BASE for v in path_list[last_step + 1:]):
                transmit_reward = len(new_rocks) * TRANSMIT_BONUS

        total += rock_reward + transmit_reward

    return total


# ── OBJECTIVES ────────────────────────────────────────────────────────────────

def make_mars_objective(init_positions):
    """
    Parameters
    ----------
    init_positions : {rid: (row, col)}  — starting grid positions

    Returns
    -------
    global_obj    : {rid: [(r0,c0),(r1,c1),...]} -> float
    local_obj_fns : {rid: callable} (pure action-seq format for DecMCTS)
    """
    robot_ids = list(init_positions.keys())

    # ── global_obj (path format) ──────────────────────────────────────────────
    def global_obj(joint_paths):
        return _global_from_paths({rid: list(p) for rid, p in joint_paths.items()})

    # ── local_obj_fns (pure action-seq format) ────────────────────────────────
    def make_local_fn(rid):

        def local_fn(joint_action_seqs):
            # Convert action sequences to full paths (prepend initial position)
            joint_paths = {}
            for r, seq in joint_action_seqs.items():
                ip = init_positions.get(r, init_positions[rid])
                joint_paths[r] = [ip] + list(seq)
            # Robots absent from joint_action_seqs: just their start position
            for r in robot_ids:
                if r not in joint_paths:
                    joint_paths[r] = [init_positions[r]]

            # Marginal contribution: G(x^r ∪ x^{-r}) - G(∅ ∪ x^{-r})
            # Null path for rid = stay at initial position (no actions taken).
            # Essential for submodular coverage: without the baseline, both
            # robots see full team reward regardless of overlap, incentivising
            # clustering rather than complementary rock coverage.
            null_paths = dict(joint_paths)
            null_paths[rid] = [init_positions[rid]]
            return global_obj(joint_paths) - global_obj(null_paths)

        return local_fn

    local_obj_fns = {rid: make_local_fn(rid) for rid in robot_ids}
    return global_obj, local_obj_fns


# ── BENCHMARK FACTORY ─────────────────────────────────────────────────────────

# Default start positions
HELO_START = (0, 0)   # rover 0
SHIP_START = (0, 3)   # rover 1

def can_communicate(s0, s1):
    """
    Mars: rovers can communicate when they are at the same grid cell
    (line-of-sight contact), OR when at least one is at BASE (relay station).
    """
    return s0.vertex == s1.vertex or s0.vertex == BASE or s1.vertex == BASE


def make_mars_benchmark(
    starts=None,
    horizon=DEFAULT_HORIZON,
):
    """
    Parameters
    ----------
    starts  : {rid: (row, col)} or None (uses defaults)
    horizon : episode length

    Returns
    -------
    robot_ids, init_states, global_obj, local_obj_fns, can_communicate
    """
    if starts is None:
        starts = {0: HELO_START, 1: SHIP_START}
    robot_ids   = list(starts.keys())
    init_states = {
        rid: MarsState(r, c, 0, horizon)
        for rid, (r, c) in starts.items()
    }
    global_obj, local_obj_fns = make_mars_objective(starts)
    return robot_ids, init_states, global_obj, local_obj_fns, can_communicate
