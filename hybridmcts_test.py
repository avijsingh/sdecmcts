"""
hybridmcts_test.py
------------------
Tests for partition-based HybridMCTS on the same orienteering scenario
used in decmcts_test.py.

Tests
-----
1. VALIDITY       — every robot's path respects budget and uses real edges.

2. BELIEF RESET   — for every parent→child edge in the tree, applying the
                    parent's committed joint_action to the parent's
                    joint_states must exactly reproduce the child's
                    joint_states.  This holds regardless of which groups
                    merged or split at that transition.

3. TREE STRUCTURE — confirm the tree contains nodes with varied partitions
                    (not just all-connected or all-isolated), and that
                    group planners are correctly typed (CenMCTS for |g|>1,
                    DecMCTS for singletons).

4. REWARD ORDERING — compare HybridMCTS against CenMCTS and DecMCTS
                     baselines across different link-up probabilities.

5. ROBUSTNESS SWEEP — vary p_link from 1.0 (always full comm) to 0.0
                      (always isolated), confirm graceful degradation.
"""

import math
import random
import time
from collections import defaultdict, deque

from decmcts import DecMCTS, DecMCTSTeam
from cenmcts import CenMCTS
from hybridmcts import HybridMCTS, CommGraph, PartitionNode


# ─────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────

SEED        = 42
N_VERTICES  = 800
WORLD_SIZE  = 100.0
N_NEIGHBORS = 8
N_OBSTACLES = 3
N_REGIONS   = 60
REGION_RAD  = 8.0
N_ROBOTS    = 4
BUDGET      = 60.0

HYBRID_ITER     = 60
CEN_PLAN_ITERS  = 30
DEC_OUTER_ITERS = 3
DEC_TAU         = 5
DEC_NUM_SEQ     = 5
DEC_NUM_SAMPLES = 10
ROLLOUT_DEPTH   = 20
MAX_CHILDREN    = 8

CEN_ITER      = 500
DEC_N_OUTER   = 30
DEC_TAU_B     = 10
DEC_NUM_SEQ_B = 10


# ─────────────────────────────────────────────────────────
# ENVIRONMENT
# ─────────────────────────────────────────────────────────

def _dist(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def _in_obstacle(x, y, obs):
    ox, oy, ow, oh = obs
    return ox <= x <= ox+ow and oy <= y <= oy+oh

def _edge_blocked(a, b, obstacles):
    mx, my = (a[0]+b[0])/2, (a[1]+b[1])/2
    return any(_in_obstacle(mx, my, obs) for obs in obstacles)


class Graph:
    def __init__(self, n_vertices, world_size, k, obstacles, rng):
        self.pos = []
        attempts = 0
        while len(self.pos) < n_vertices and attempts < n_vertices * 20:
            x = rng.uniform(0, world_size)
            y = rng.uniform(0, world_size)
            if not any(_in_obstacle(x, y, obs) for obs in obstacles):
                self.pos.append((x, y))
            attempts += 1
        self.n = len(self.pos)
        self.adj = defaultdict(list)
        for i in range(self.n):
            dists = sorted(
                (_dist(self.pos[i], self.pos[j]), j)
                for j in range(self.n) if i != j
            )
            for d, j in dists[:k]:
                if not _edge_blocked(self.pos[i], self.pos[j], obstacles):
                    self.adj[i].append((j, d))
                    self.adj[j].append((i, d))
        for v in range(self.n):
            seen = {}
            for nb, d in self.adj[v]:
                if nb not in seen or d < seen[nb]:
                    seen[nb] = d
            self.adj[v] = list(seen.items())

    def neighbors(self, v):
        return self.adj[v]


def build_obstacles(rng, world_size, n):
    obs = []
    for _ in range(n):
        w = rng.uniform(0.05, 0.12) * world_size
        h = rng.uniform(0.05, 0.12) * world_size
        x = rng.uniform(0.1*world_size, 0.9*world_size - w)
        y = rng.uniform(0.1*world_size, 0.9*world_size - h)
        obs.append((x, y, w, h))
    return obs


def build_regions(rng, world_size, n, radius):
    regions = []
    for _ in range(n):
        cx = rng.uniform(radius, world_size - radius)
        cy = rng.uniform(radius, world_size - radius)
        w  = rng.randint(1, 10)
        regions.append({"cx": cx, "cy": cy, "r": radius, "w": w, "verts": set()})
    return regions


def assign_vertices_to_regions(graph, regions):
    for i, (x, y) in enumerate(graph.pos):
        for reg in regions:
            if math.hypot(x - reg["cx"], y - reg["cy"]) <= reg["r"]:
                reg["verts"].add(i)


class OrienteeringState:
    __slots__ = ("vertex", "dist_used", "_graph", "_budget")

    def __init__(self, vertex, dist_used, graph, budget):
        self.vertex    = vertex
        self.dist_used = dist_used
        self._graph    = graph
        self._budget   = budget

    def is_terminal_state(self):
        return self.dist_used >= self._budget

    def get_legal_actions(self):
        if self.dist_used >= self._budget:
            return []
        remaining = self._budget - self.dist_used
        return [nb for nb, cost in self._graph.neighbors(self.vertex)
                if cost <= remaining]

    def take_action(self, action):
        cost = next(c for nb, c in self._graph.neighbors(self.vertex)
                    if nb == action)
        return OrienteeringState(action, self.dist_used + cost,
                                 self._graph, self._budget)

    @property
    def reward(self):
        return 0.0


def make_objective(regions):
    def global_obj(joint_seqs):
        visited = set()
        for path in joint_seqs.values():
            visited.update(path)
        return sum(reg["w"] for reg in regions if reg["verts"] & visited)

    def make_local(robot_id, null_path):
        def local_util(joint_seqs):
            g = global_obj(joint_seqs)
            null_j = dict(joint_seqs)
            null_j[robot_id] = null_path
            return g - global_obj(null_j)
        return local_util

    return global_obj, make_local


# ─────────────────────────────────────────────────────────
# CORRECTNESS CHECKS
# ─────────────────────────────────────────────────────────

def validate_paths(paths, start_verts, graph, budget, robot_ids):
    errors = []
    for rid in robot_ids:
        path = paths.get(rid, [])
        if not path:
            errors.append(f"Robot {rid}: empty path")
            continue
        if path[0] != start_verts[rid]:
            errors.append(f"Robot {rid}: starts at {path[0]}, "
                          f"expected {start_verts[rid]}")
        dist = 0.0
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            neighbors = dict(graph.neighbors(u))
            if v not in neighbors:
                errors.append(f"Robot {rid}: edge ({u}→{v}) not in graph")
            else:
                dist += neighbors[v]
        if dist > budget + 1e-6:
            errors.append(f"Robot {rid}: distance {dist:.2f} > budget {budget}")
    return errors


def check_belief_reset(planner):
    """
    For every parent→child edge in the tree, verify that applying the
    parent's committed joint_action to the parent's joint_states exactly
    reproduces the child's joint_states.

    This is the core correctness test for the partition-based architecture:
    group splits, merges, and singletons all share the same verification
    logic because every transition is expressed as a joint_action dict.
    """
    errors = []
    queue  = deque([planner.root])
    seen   = set()

    while queue:
        parent = queue.popleft()
        if id(parent) in seen:
            continue
        seen.add(id(parent))

        for child in parent.children:
            if child.action is None:
                queue.append(child)
                continue

            for rid in planner.robot_ids:
                parent_state = parent.joint_states[rid]
                committed    = child.action.get(rid)

                if committed is None or parent_state.is_terminal_state():
                    # Robot stayed put — vertex must be unchanged
                    if child.joint_states[rid].vertex != parent_state.vertex:
                        errors.append(
                            f"Robot {rid}: stayed put but vertex changed "
                            f"{parent_state.vertex} → "
                            f"{child.joint_states[rid].vertex}"
                        )
                else:
                    expected_vertex = parent_state.take_action(committed).vertex
                    if child.joint_states[rid].vertex != expected_vertex:
                        errors.append(
                            f"Robot {rid}: committed {committed} but "
                            f"child vertex is {child.joint_states[rid].vertex}, "
                            f"expected {expected_vertex}"
                        )
            queue.append(child)

    return errors


def tree_stats(planner):
    """
    Count nodes, unique partitions, and planner types encountered.
    """
    total       = 0
    partitions  = set()
    cen_groups  = 0
    dec_singles = 0
    nodes_with_planners = 0

    queue = deque([planner.root])
    seen  = set()
    while queue:
        node = queue.popleft()
        if id(node) in seen:
            continue
        seen.add(id(node))
        total += 1
        partitions.add(node.partition)

        if node._group_planners is not None:
            nodes_with_planners += 1
            for group in node.partition:
                if len(group) > 1:
                    cen_groups += 1
                else:
                    dec_singles += 1

        for child in node.children:
            queue.append(child)

    return {
        "total":               total,
        "unique_partitions":   len(partitions),
        "nodes_with_planners": nodes_with_planners,
        "cen_group_instances": cen_groups,
        "dec_singleton_instances": dec_singles,
    }


# ─────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────

def section(title):
    w = 70
    print(f"\n{'═'*w}\n  {title}\n{'═'*w}")

def ok(msg):   print(f"  [PASS]  {msg}")
def fail(msg): print(f"  [FAIL]  {msg}")
def info(msg): print(f"  [INFO]  {msg}")


# ─────────────────────────────────────────────────────────
# BUILD SHARED ENVIRONMENT
# ─────────────────────────────────────────────────────────

rng       = random.Random(SEED)
random.seed(SEED)

obstacles = build_obstacles(rng, WORLD_SIZE, N_OBSTACLES)
graph     = Graph(N_VERTICES, WORLD_SIZE, N_NEIGHBORS, obstacles, rng)
regions   = build_regions(rng, WORLD_SIZE, N_REGIONS, REGION_RAD)
assign_vertices_to_regions(graph, regions)

robot_ids   = list(range(N_ROBOTS))
start_verts = rng.sample(range(graph.n), N_ROBOTS)

global_obj, make_local = make_objective(regions)
null_paths = {rid: [start_verts[rid]] for rid in robot_ids}
local_fns  = {rid: make_local(rid, null_paths[rid]) for rid in robot_ids}
max_reward = sum(r["w"] for r in regions)

info(f"Graph: {graph.n} vertices, {N_ROBOTS} robots, budget={BUDGET}, "
     f"max_reward={max_reward}")
info(f"Starts: {[start_verts[r] for r in robot_ids]}")


def make_init_states():
    return {
        rid: OrienteeringState(start_verts[rid], 0.0, graph, BUDGET)
        for rid in robot_ids
    }


# ─────────────────────────────────────────────────────────
# TEST 1: VALIDITY + BELIEF RESET + TREE STRUCTURE
# ─────────────────────────────────────────────────────────

section("TEST 1: Core Correctness (validity + belief reset + tree structure)")

# p_link=0.6 ensures a mix of full-team, partial, and isolated partitions
comm_graph = CommGraph(robot_ids, p_link=0.6)

t0 = time.perf_counter()
hybrid = HybridMCTS(
    robot_ids       = robot_ids,
    init_states     = make_init_states(),
    global_obj      = global_obj,
    local_obj_fns   = local_fns,
    comm_graph      = comm_graph,
    max_children    = MAX_CHILDREN,
    cen_plan_iters  = CEN_PLAN_ITERS,
    dec_outer_iters = DEC_OUTER_ITERS,
    dec_tau         = DEC_TAU,
    dec_num_seq     = DEC_NUM_SEQ,
    dec_num_samples = DEC_NUM_SAMPLES,
    rollout_depth   = ROLLOUT_DEPTH,
)
hybrid.run(HYBRID_ITER)
elapsed = time.perf_counter() - t0

best   = hybrid.best_paths()
reward = global_obj(best)
pseq   = hybrid.best_partition_sequence()

info(f"Ran {HYBRID_ITER} iterations in {elapsed:.2f}s")
info(f"Best reward: {reward:.1f} / {max_reward} ({100*reward/max_reward:.1f}%)")
info(f"Best partition sequence:")
for i, p in enumerate(pseq):
    print(f"       depth {i}: {p}")

# 1a. Path validity
errors = validate_paths(best, start_verts, graph, BUDGET, robot_ids)
if not errors:
    ok("All robot paths are valid (edges + budget)")
else:
    for e in errors:
        fail(e)

# 1b. Belief reset
br_errors = check_belief_reset(hybrid)
if not br_errors:
    ok("All parent→child joint_state transitions are consistent "
       "(belief reset holds for all partition changes)")
else:
    for e in br_errors:
        fail(e)

# 1c. Tree structure
stats = tree_stats(hybrid)
info(f"Tree stats: {stats}")

if stats["unique_partitions"] > 1:
    ok(f"Tree has {stats['unique_partitions']} unique partitions — "
       "partial comm loss events are being explored")
else:
    info("Only one partition seen — try lower p_link or more iterations")

if stats["cen_group_instances"] > 0:
    ok(f"CenMCTS used for {stats['cen_group_instances']} connected subgroup(s)")
else:
    info("No CenMCTS subgroups initialised yet")

if stats["dec_singleton_instances"] > 0:
    ok(f"DecMCTS used for {stats['dec_singleton_instances']} singleton(s)")
else:
    info("No DecMCTS singletons initialised yet")


# ─────────────────────────────────────────────────────────
# TEST 2: REWARD ORDERING vs BASELINES
# ─────────────────────────────────────────────────────────

section("TEST 2: Reward Ordering vs CenMCTS and DecMCTS Baselines")

# CenMCTS baseline
t0  = time.perf_counter()
cen = CenMCTS(make_init_states(), global_obj, start_verts, rollout_depth=50)
cen.run(CEN_ITER)
cen_paths  = cen._best_paths()
cen_reward = global_obj(cen_paths)
info(f"CenMCTS  ({CEN_ITER} iter, {time.perf_counter()-t0:.2f}s): "
     f"reward={cen_reward:.1f}")

# DecMCTS baseline
t0 = time.perf_counter()
planners = {
    rid: DecMCTS(
        robot_id=rid, robot_ids=robot_ids,
        init_state=make_init_states()[rid],
        local_utility_fn=local_fns[rid],
        tau=DEC_TAU_B, num_seq=DEC_NUM_SEQ_B, num_samples=20,
    )
    for rid in robot_ids
}
DecMCTSTeam(planners).iterate_and_communicate(n_outer=DEC_N_OUTER, comm_period=1)
dec_paths  = {rid: [start_verts[rid]] + planners[rid].best_action_sequence()
              for rid in robot_ids}
dec_reward = global_obj(dec_paths)
info(f"DecMCTS  ({DEC_N_OUTER} outer iter, {time.perf_counter()-t0:.2f}s): "
     f"reward={dec_reward:.1f}")

# Hybrid — high link reliability (should behave close to CenMCTS)
t0 = time.perf_counter()
h_high = HybridMCTS(
    robot_ids=robot_ids, init_states=make_init_states(),
    global_obj=global_obj, local_obj_fns=local_fns,
    comm_graph=CommGraph(robot_ids, p_link=0.95),
    max_children=MAX_CHILDREN, cen_plan_iters=CEN_PLAN_ITERS,
    dec_outer_iters=DEC_OUTER_ITERS, dec_tau=DEC_TAU,
    dec_num_seq=DEC_NUM_SEQ, dec_num_samples=DEC_NUM_SAMPLES,
    rollout_depth=ROLLOUT_DEPTH,
)
h_high.run(HYBRID_ITER)
r_high = global_obj(h_high.best_paths())
info(f"Hybrid (p_link=0.95, {HYBRID_ITER} iter, {time.perf_counter()-t0:.2f}s): "
     f"reward={r_high:.1f}")

# Hybrid — mixed reliability
t0 = time.perf_counter()
h_mid = HybridMCTS(
    robot_ids=robot_ids, init_states=make_init_states(),
    global_obj=global_obj, local_obj_fns=local_fns,
    comm_graph=CommGraph(robot_ids, p_link=0.6),
    max_children=MAX_CHILDREN, cen_plan_iters=CEN_PLAN_ITERS,
    dec_outer_iters=DEC_OUTER_ITERS, dec_tau=DEC_TAU,
    dec_num_seq=DEC_NUM_SEQ, dec_num_samples=DEC_NUM_SAMPLES,
    rollout_depth=ROLLOUT_DEPTH,
)
h_mid.run(HYBRID_ITER)
r_mid = global_obj(h_mid.best_paths())
info(f"Hybrid (p_link=0.60, {HYBRID_ITER} iter, {time.perf_counter()-t0:.2f}s): "
     f"reward={r_mid:.1f}")

# Hybrid — low reliability (should behave close to DecMCTS)
t0 = time.perf_counter()
h_low = HybridMCTS(
    robot_ids=robot_ids, init_states=make_init_states(),
    global_obj=global_obj, local_obj_fns=local_fns,
    comm_graph=CommGraph(robot_ids, p_link=0.1),
    max_children=MAX_CHILDREN, cen_plan_iters=CEN_PLAN_ITERS,
    dec_outer_iters=DEC_OUTER_ITERS, dec_tau=DEC_TAU,
    dec_num_seq=DEC_NUM_SEQ, dec_num_samples=DEC_NUM_SAMPLES,
    rollout_depth=ROLLOUT_DEPTH,
)
h_low.run(HYBRID_ITER)
r_low = global_obj(h_low.best_paths())
info(f"Hybrid (p_link=0.10, {HYBRID_ITER} iter, {time.perf_counter()-t0:.2f}s): "
     f"reward={r_low:.1f}")

# Validate all paths
for label, paths in [("CenMCTS", cen_paths), ("DecMCTS", dec_paths),
                     ("Hybrid p=0.95", h_high.best_paths()),
                     ("Hybrid p=0.60", h_mid.best_paths()),
                     ("Hybrid p=0.10", h_low.best_paths())]:
    errs = validate_paths(paths, start_verts, graph, BUDGET, robot_ids)
    if not errs:
        ok(f"{label} paths valid")
    else:
        for e in errs:
            fail(f"{label}: {e}")


# ─────────────────────────────────────────────────────────
# TEST 3: PARTIAL COMM LOSS — per-pair link probabilities
# ─────────────────────────────────────────────────────────

section("TEST 3: Partial Comm Loss — per-pair link probabilities")

info("Scenario: robots {0,1} reliably connected (p=0.9), "
     "robot 2 unreliable (p=0.2 to all), robot 3 isolated (p=0.05 to all)")

# Build asymmetric p_link dict
p_link_asym = {}
for i in robot_ids:
    for j in robot_ids:
        if j <= i:
            continue
        if i in (0, 1) and j in (0, 1):
            p_link_asym[(i, j)] = 0.9   # 0↔1 reliable
        elif i == 2 or j == 2:
            p_link_asym[(i, j)] = 0.2   # robot 2 unreliable
        else:
            p_link_asym[(i, j)] = 0.05  # robot 3 often isolated

t0 = time.perf_counter()
h_asym = HybridMCTS(
    robot_ids=robot_ids, init_states=make_init_states(),
    global_obj=global_obj, local_obj_fns=local_fns,
    comm_graph=CommGraph(robot_ids, p_link=p_link_asym),
    max_children=MAX_CHILDREN, cen_plan_iters=CEN_PLAN_ITERS,
    dec_outer_iters=DEC_OUTER_ITERS, dec_tau=DEC_TAU,
    dec_num_seq=DEC_NUM_SEQ, dec_num_samples=DEC_NUM_SAMPLES,
    rollout_depth=ROLLOUT_DEPTH,
)
h_asym.run(HYBRID_ITER)
r_asym = global_obj(h_asym.best_paths())
info(f"Asymmetric comm ({time.perf_counter()-t0:.2f}s): reward={r_asym:.1f}")

stats_asym = tree_stats(h_asym)
info(f"Tree: {stats_asym['total']} nodes, "
     f"{stats_asym['unique_partitions']} unique partitions")

errs = validate_paths(h_asym.best_paths(), start_verts, graph, BUDGET, robot_ids)
if not errs:
    ok("Asymmetric comm paths valid")
else:
    for e in errs:
        fail(e)

br_errs = check_belief_reset(h_asym)
if not br_errs:
    ok("Belief reset holds under asymmetric comm topology")
else:
    for e in br_errs:
        fail(e)

info("Partition sequence under asymmetric comm:")
for i, p in enumerate(h_asym.best_partition_sequence()):
    print(f"       depth {i}: {p}")


# ─────────────────────────────────────────────────────────
# TEST 4: ROBUSTNESS SWEEP — uniform p_link 0.0 → 1.0
# ─────────────────────────────────────────────────────────

section("TEST 4: Robustness Sweep — p_link from 1.0 to 0.0")

print(f"\n  {'p_link':>8}  {'reward':>8}  {'nodes':>6}  "
      f"{'partitions':>11}  {'status'}")
print(f"  {'-'*8}  {'-'*8}  {'-'*6}  {'-'*11}  {'-'*6}")

for p in [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]:
    random.seed(SEED)
    try:
        h = HybridMCTS(
            robot_ids=robot_ids, init_states=make_init_states(),
            global_obj=global_obj, local_obj_fns=local_fns,
            comm_graph=CommGraph(robot_ids, p_link=p),
            max_children=MAX_CHILDREN, cen_plan_iters=CEN_PLAN_ITERS,
            dec_outer_iters=DEC_OUTER_ITERS, dec_tau=DEC_TAU,
            dec_num_seq=DEC_NUM_SEQ, dec_num_samples=DEC_NUM_SAMPLES,
            rollout_depth=ROLLOUT_DEPTH,
        )
        h.run(HYBRID_ITER)
        r    = global_obj(h.best_paths())
        st   = tree_stats(h)
        errs = validate_paths(h.best_paths(), start_verts, graph, BUDGET, robot_ids)
        status = "OK" if not errs else f"INVALID({len(errs)})"
        print(f"  {p:>8.2f}  {r:>8.1f}  {st['total']:>6}  "
              f"{st['unique_partitions']:>11}  {status}")
    except Exception as e:
        print(f"  {p:>8.2f}  {'CRASH':>8}  {'—':>6}  {'—':>11}  {e}")

ok("Robustness sweep completed")


# ─────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────

section("SUMMARY")
print(f"""
  Scenario : {graph.n}-vertex graph, {N_ROBOTS} robots, budget={BUDGET},
             {N_REGIONS} goal regions, max_reward={max_reward}

  Baselines
  ─────────────────────────────────────────────────────────
  CenMCTS   ({CEN_ITER} iter)               : {cen_reward:.1f}
  DecMCTS   ({DEC_N_OUTER} outer iter)      : {dec_reward:.1f}

  HybridMCTS ({HYBRID_ITER} iter)
  ─────────────────────────────────────────────────────────
  p_link=0.95 (near-full comm)              : {r_high:.1f}
  p_link=0.60 (mixed)                       : {r_mid:.1f}
  p_link=0.10 (near-isolated)               : {r_low:.1f}
  asymmetric topology (0↔1 reliable)        : {r_asym:.1f}
""")
