"""
labyrinth.py
------------
Labyrinth benchmark for SDecMCTS (MDP relaxation: target location known).

Based on: RSSDA labyrinth domain.

2 agents navigate a graph.  Each must visit the target node then return
to the start node.  Reward is awarded per robot that completes the round
trip, with a coordination bonus if BOTH complete.

Default graph (6 nodes):

    0 ── 1 ── 2
    │         │
    3 ── 4 ── 5

    Start  = 0
    Target = 5

Actions: move to an adjacent node OR stay at current node  (node IDs)
vertex  = current node ID  (integer)

NOTE — Two function signatures (same convention as other benchmarks):

  global_obj(joint_paths)
      joint_paths = {rid: [node_0, node_1, ...]}   ← node-ID sequences

  local_obj_fns[rid](joint_action_seqs)
      joint_action_seqs = {rid: [node_1, node_2, ...]}  ← actions only
"""

DEFAULT_HORIZON = 20

# Default 6-node graph (adjacency list)
DEFAULT_GRAPH = {
    0: [1, 3],
    1: [0, 2, 4],
    2: [1, 5],
    3: [0, 4],
    4: [1, 3, 5],
    5: [2, 4],
}
DEFAULT_START  = 0
DEFAULT_TARGET = 5

ROUNDTRIP_REWARD     = 100.0
COORDINATION_BONUS   =  50.0   # added if ALL robots complete


# ── STATE ─────────────────────────────────────────────────────────────────────

class LabyrinthState:
    """
    Per-robot state: current node + step counter.
    Actions = adjacent nodes OR stay  (node IDs, matching vertex format).
    vertex  = current node ID.
    """
    __slots__ = ("vertex", "node", "step", "max_steps", "_graph")

    def __init__(self, node, step=0, max_steps=DEFAULT_HORIZON, graph=None):
        self.node      = node
        self.step      = step
        self.max_steps = max_steps
        self._graph    = graph if graph is not None else DEFAULT_GRAPH
        self.vertex    = node

    def get_legal_actions(self):
        if self.step >= self.max_steps:
            return []
        return [self.node] + self._graph.get(self.node, [])   # stay + neighbors

    def take_action(self, action):
        return LabyrinthState(action, self.step + 1, self.max_steps, self._graph)

    def is_terminal_state(self):
        return self.step >= self.max_steps


# ── REWARD ────────────────────────────────────────────────────────────────────

def _completed_roundtrip(path_list, target, start):
    """
    Returns True if `path_list` visits `target` at some step t, then
    visits `start` at some step > t.
    """
    target_step = next(
        (i for i, v in enumerate(path_list) if v == target), None
    )
    if target_step is None:
        return False
    return any(v == start for v in path_list[target_step + 1:])


# ── OBJECTIVES ────────────────────────────────────────────────────────────────

def make_labyrinth_objective(
    init_positions,
    graph=None,
    start=DEFAULT_START,
    target=DEFAULT_TARGET,
):
    """
    Parameters
    ----------
    init_positions : {rid: node_id}
    graph          : adjacency dict (or None → DEFAULT_GRAPH)
    start          : start / home node ID
    target         : target node ID

    Returns
    -------
    global_obj    : {rid: path_list} -> float
    local_obj_fns : {rid: callable} (pure action-seq format for DecMCTS)
    """
    if graph is None:
        graph = DEFAULT_GRAPH
    robot_ids = list(init_positions.keys())

    # ── global_obj (path format) ──────────────────────────────────────────────
    def global_obj(joint_paths):
        n_completed = sum(
            1 for path in joint_paths.values()
            if _completed_roundtrip(list(path), target, start)
        )
        reward = n_completed * ROUNDTRIP_REWARD
        if n_completed == len(joint_paths):
            reward += COORDINATION_BONUS
        return reward

    # ── local_obj_fns (pure action-seq format) ────────────────────────────────
    def make_local_fn(rid):
        def local_fn(joint_action_seqs):
            joint_paths = {}
            for r, seq in joint_action_seqs.items():
                ip = init_positions.get(r, start)
                joint_paths[r] = [ip] + list(seq)
            for r in robot_ids:
                if r not in joint_paths:
                    joint_paths[r] = [init_positions[r]]

            # Marginal contribution: G(x^r ∪ x^{-r}) - G(∅ ∪ x^{-r})
            # Null path for rid = stay at start node (no roundtrip attempted).
            # This correctly isolates the COORDINATION_BONUS as robot r's
            # marginal contribution when its teammate would already complete alone.
            null_paths = dict(joint_paths)
            null_paths[rid] = [init_positions[rid]]
            return global_obj(joint_paths) - global_obj(null_paths)

        return local_fn

    local_obj_fns = {rid: make_local_fn(rid) for rid in robot_ids}
    return global_obj, local_obj_fns


# ── BENCHMARK FACTORY ─────────────────────────────────────────────────────────

def can_communicate(s0, s1):
    """
    Labyrinth: agents can communicate when they are at the same node
    (co-located contact), or when either is at the start node (hub with
    relay infrastructure).
    """
    return s0.node == s1.node or s0.node == DEFAULT_START or s1.node == DEFAULT_START


def make_labyrinth_benchmark(
    horizon=DEFAULT_HORIZON,
    graph=None,
    start=DEFAULT_START,
    target=DEFAULT_TARGET,
):
    """
    Returns
    -------
    robot_ids, init_states, global_obj, local_obj_fns, can_communicate
    """
    if graph is None:
        graph = DEFAULT_GRAPH
    robot_ids      = [0, 1]
    init_positions = {rid: start for rid in robot_ids}
    init_states    = {
        rid: LabyrinthState(start, 0, horizon, graph)
        for rid in robot_ids
    }
    global_obj, local_obj_fns = make_labyrinth_objective(
        init_positions, graph, start, target
    )
    return robot_ids, init_states, global_obj, local_obj_fns, can_communicate
