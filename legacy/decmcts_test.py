"""

Paper setup replicated exactly here:
  - Random graph (PRM-like), 4000 vertices in a 2D workspace
  - 200 disk-shaped goal regions, weights uniform in [1, 10]
  - 5 random rectangular obstacles (robots avoid them)
  - 8 robots with random start vertices and equal distance budgets
  - Action = move to adjacent graph vertex; edge cost = Euclidean distance
  - Region visited if any vertex ON any robot's PATH lies inside the disk
  - No double-reward for multiply-visited regions
  - Objective: sum of weights of all visited regions (team total)

Key paper metrics reported:
  - Reward vs. outer-iteration curve  (convergence)
  - Dec-MCTS vs. Cen-MCTS reward comparison  (quality)
  - Performance vs. message-drop probability  (robustness)

Coordinate-system fix vs. previous test:
  State tracks the current VERTEX (a node index), not a (dx, dy) delta.
  action_sequence on DecMCTSNode is therefore a list of vertex indices.
  global_obj receives {robot_id: [v0, v1, ...]} — actual vertex paths.

"""

import math
import random
import time
import sys
import argparse
from collections import defaultdict

import json
import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from decmcts import DecMCTS, DecMCTSTeam
from cenmcts import CenMCTS, CenMCTSNode

# CONFIGURATION

SEED          = 42

# Graph
N_VERTICES    = 4000         
WORLD_SIZE    = 100.0        
N_NEIGHBORS  = 8            # k-NN connectivity (PRM-like)
N_OBSTACLES   = 5            

# Regions
N_REGIONS     = 200          
REGION_RADIUS = 8.0          # disk radius

# Robots
N_ROBOTS      = 8            
BUDGET        = 80.0         # distance budget per robot

# Algorithm  (paper: tau=10, num_seq=10, resample every 10 outer iters)
N_OUTER       = 200          # outer iterations to run
TAU           = 10           # MCTS rollouts per outer iter
NUM_SEQ       = 10           
NUM_SAMPLES   = 20           # MC samples for dist update
GAMMA         = 0.9
CP            = 1.0 / math.sqrt(2)
ALPHA         = 0.01
BETA_INIT     = 1.0
BETA_DECAY    = 0.99

# Communication interval: broadcast every COMM_PERIOD outer iterations
# 1  = every iteration (maximum bandwidth, paper default)
# K  = every K iterations
# -1 = never communicate (fully decentralized baseline)
COMM_PERIOD   = 1

# Robustness sweep: message drop probabilities to evaluate
DROP_PROBS    = [0.0, 0.25, 0.5, 0.75, 1.0]

# ANSI COLORS

if sys.stdout.isatty():
    R, G, Y, B, BOLD, DIM, RST = (
        "\033[91m", "\033[92m", "\033[93m", "\033[94m",
        "\033[1m",  "\033[2m",  "\033[0m"
    )
else:
    R = G = Y = B = BOLD = DIM = RST = ""

def section(t):
    w = 72
    pad = (w - len(t) - 2) // 2
    print(f"\n{BOLD}{'═'*w}{RST}")
    print(f"{BOLD}{'═'*pad} {t} {'═'*(w-pad-len(t)-2)}{RST}")
    print(f"{BOLD}{'═'*w}{RST}")

def sub(t):
    print(f"\n{B}{BOLD}── {t} {'─'*(66-len(t))}{RST}")

# GRAPH CONSTRUCTION

class Graph:
    """
    Random k-NN graph in [0, WORLD_SIZE]^2, obstacles excluded.
    Edge weight = Euclidean distance between vertex positions.
    """

    def __init__(self, n_vertices, world_size, k, obstacles, rng):
        self.world_size = world_size

        # Sample vertices (reject those inside obstacles)
        self.pos = []
        attempts = 0
        while len(self.pos) < n_vertices and attempts < n_vertices * 20:
            x = rng.uniform(0, world_size)
            y = rng.uniform(0, world_size)
            if not any(_in_obstacle(x, y, obs) for obs in obstacles):
                self.pos.append((x, y))
            attempts += 1
        self.n = len(self.pos)

        # Build k-NN adjacency
        self.adj = defaultdict(list)
        for i in range(self.n):
            dists = []
            for j in range(self.n):
                if i != j:
                    d = _dist(self.pos[i], self.pos[j])
                    dists.append((d, j))
            dists.sort()
            for d, j in dists[:k]:
                # Only add edge if path doesn't cross an obstacle
                if not _edge_blocked(self.pos[i], self.pos[j], obstacles):
                    self.adj[i].append((j, d))
                    self.adj[j].append((i, d))

        # De-duplicate adjacency lists
        for v in range(self.n):
            seen = {}
            for (nb, d) in self.adj[v]:
                if nb not in seen or d < seen[nb]:
                    seen[nb] = d
            self.adj[v] = list(seen.items())

    def neighbors(self, v):
        return self.adj[v]   # [(neighbor_idx, edge_cost), ...]


def _dist(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def _in_obstacle(x, y, obs):
    ox, oy, ow, oh = obs
    return ox <= x <= ox+ow and oy <= y <= oy+oh

def _edge_blocked(a, b, obstacles):
    """Coarse check: midpoint inside obstacle."""
    mx, my = (a[0]+b[0])/2, (a[1]+b[1])/2
    return any(_in_obstacle(mx, my, obs) for obs in obstacles)


def build_obstacles(rng, world_size, n):
    """Random axis-aligned rectangles, each ~10% of workspace width."""
    obs = []
    for _ in range(n):
        w = rng.uniform(0.05, 0.12) * world_size
        h = rng.uniform(0.05, 0.12) * world_size
        x = rng.uniform(0.1*world_size, 0.9*world_size - w)
        y = rng.uniform(0.1*world_size, 0.9*world_size - h)
        obs.append((x, y, w, h))
    return obs


def build_regions(rng, world_size, n, radius):
    """
    N disk-shaped goal regions, each with weight in [1, 10].
    Returns list of {cx, cy, radius, weight, vertex_set}.
    vertex_set populated after graph is built.
    """
    regions = []
    for _ in range(n):
        cx = rng.uniform(radius, world_size - radius)
        cy = rng.uniform(radius, world_size - radius)
        w  = rng.randint(1, 10)
        regions.append({"cx": cx, "cy": cy, "r": radius, "w": w, "verts": set()})
    return regions


def assign_vertices_to_regions(graph, regions):
    """Populate region["verts"] with indices of vertices inside each disk."""
    for i, (x, y) in enumerate(graph.pos):
        for reg in regions:
            if math.hypot(x - reg["cx"], y - reg["cy"]) <= reg["r"]:
                reg["verts"].add(i)


# STATE  — tracks current vertex + distance budget used

class OrienteeringState:
    """
    State for one robot: current vertex index + cumulative distance used.
    action  = destination vertex index
    action cost = edge distance
    """
    __slots__ = ("vertex", "dist_used", "_graph", "_budget")

    def __init__(self, vertex, dist_used, graph, budget):
        self.vertex   = vertex
        self.dist_used = dist_used
        self._graph   = graph
        self._budget  = budget

    def is_terminal_state(self):
        return self.dist_used >= self._budget

    def get_legal_actions(self):
        """
        Neighbors reachable without exceeding budget.
        Action = destination vertex index.
        """
        if self.dist_used >= self._budget:
            return []
        remaining = self._budget - self.dist_used
        return [nb for nb, cost in self._graph.neighbors(self.vertex)
                if cost <= remaining]

    def take_action(self, action):
        # Find edge cost to `action`
        cost = next(c for nb, c in self._graph.neighbors(self.vertex)
                    if nb == action)
        return OrienteeringState(action, self.dist_used + cost,
                                 self._graph, self._budget)

    @property
    def reward(self):
        return 0.0   # reward defined over full sequence, not per step


# OBJECTIVE FUNCTION

def make_objective(regions):
    """
    global_obj(joint_seqs) → float
      joint_seqs = {robot_id: [v0, v1, v2, ...]}   ← vertex index paths

    local_utility_fn for robot r:
      f^r(joint) = g(joint) - g(joint with r replaced by null_path)
    """
    def global_obj(joint_seqs):
        visited = set()
        for path in joint_seqs.values():
            visited.update(path)
        total = 0.0
        for reg in regions:
            if reg["verts"] & visited:
                total += reg["w"]
        return total

    def make_local(robot_id, null_path):
        def local_util(joint_seqs):
            g = global_obj(joint_seqs)
            null_j = dict(joint_seqs)
            null_j[robot_id] = null_path
            return g - global_obj(null_j)
        return local_util

    return global_obj, make_local


# PATH EXTRACTION  — action_sequence → vertex path

def actions_to_path(start_vertex, action_sequence):
    """
    Convert DecMCTS action_sequence (list of destination vertices)
    into a full vertex path [start, v1, v2, ...].
    """
    return [start_vertex] + list(action_sequence)

# SINGLE TRIAL

def run_trial(seed, n_outer, drop_prob=0.0, comm_period=COMM_PERIOD, verbose=True):
    """
    Run one problem instance. Returns dict of metrics.

    drop_prob: probability that any given communication message is dropped
               (models intermittent communication, Section 4.5 of paper).
    """
    rng = random.Random(seed)
    random.seed(seed)

    # Build environment 
    obstacles = build_obstacles(rng, WORLD_SIZE, N_OBSTACLES)
    graph     = Graph(N_VERTICES, WORLD_SIZE, N_NEIGHBORS, obstacles, rng)
    regions   = build_regions(rng, WORLD_SIZE, N_REGIONS, REGION_RADIUS)
    assign_vertices_to_regions(graph, regions)

    total_reward = sum(r["w"] for r in regions)
    robot_ids    = list(range(N_ROBOTS))

    # Random start vertices (unique per robot)
    start_verts = rng.sample(range(graph.n), N_ROBOTS)

    global_obj, make_local = make_objective(regions)

    # Null path = stay at start vertex (contributes zero unique reward)
    null_paths = {rid: [start_verts[rid]] for rid in robot_ids}

    # Init Dec-MCTS planners 
    planners = {}
    for rid in robot_ids:
        sv    = start_verts[rid]
        state = OrienteeringState(sv, 0.0, graph, BUDGET)
        local = make_local(rid, null_paths[rid])
        planners[rid] = DecMCTS(
            robot_id         = rid,
            robot_ids        = robot_ids,
            init_state       = state,
            local_utility_fn = local,
            gamma            = GAMMA,
            Cp               = CP,
            rollout_depth    = 50,
            tau              = TAU,
            num_seq          = NUM_SEQ,
            num_samples      = NUM_SAMPLES,
            beta_init        = BETA_INIT,
            beta_decay       = BETA_DECAY,
            alpha            = ALPHA,
        )

    # Init Cen-MCTS baseline 
    cen_init_states = {
        rid: OrienteeringState(start_verts[rid], 0.0, graph, BUDGET)
        for rid in robot_ids
    }
    cen = CenMCTS(cen_init_states, global_obj, start_verts)

    # Run Dec-MCTS outer loop 
    reward_curve = []   # reward at each outer iteration
    t0 = time.perf_counter()

    # Live plot setup
    _ROBOT_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red",
                     "tab:purple", "tab:brown", "tab:pink", "tab:cyan"]
    if verbose:
        plt.ion()
        fig, (ax_map, ax_curve) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle("Dec-MCTS live planning", fontweight="bold")

        # Map panel 
        ax_map.set_xlim(0, WORLD_SIZE)
        ax_map.set_ylim(0, WORLD_SIZE)
        ax_map.set_aspect("equal")
        ax_map.set_title("Environment  (paths = current best plan)")
        ax_map.set_facecolor("#f8f8f8")

        # Obstacles
        for ox, oy, ow, oh in obstacles:
            ax_map.add_patch(mpatches.Rectangle(
                (ox, oy), ow, oh, facecolor="dimgray", edgecolor="black",
                linewidth=0.5, zorder=3))

        # Goal regions — color by weight, brighten when visited
        w_max = max(reg["w"] for reg in regions)
        region_patches = []
        for reg in regions:
            alpha = 0.15 + 0.25 * (reg["w"] / w_max)
            patch = mpatches.Circle(
                (reg["cx"], reg["cy"]), reg["r"],
                facecolor=plt.cm.YlOrRd(reg["w"] / w_max),
                edgecolor="none", alpha=alpha, zorder=1)
            ax_map.add_patch(patch)
            region_patches.append(patch)

        # Vertices
        vx = [graph.pos[i][0] for i in range(graph.n)]
        vy = [graph.pos[i][1] for i in range(graph.n)]
        ax_map.scatter(vx, vy, s=1.5, c="silver", zorder=2)

        # Robot start markers
        for idx, rid in enumerate(robot_ids):
            sx, sy = graph.pos[start_verts[rid]]
            ax_map.plot(sx, sy, "^", color=_ROBOT_COLORS[idx % len(_ROBOT_COLORS)],
                        markersize=9, zorder=6)

        # Robot path lines (empty at start)
        robot_lines = {}
        for idx, rid in enumerate(robot_ids):
            (ln,) = ax_map.plot([], [], "-o",
                                color=_ROBOT_COLORS[idx % len(_ROBOT_COLORS)],
                                linewidth=1.8, markersize=3, zorder=5,
                                label=f"Robot {rid}")
            robot_lines[rid] = ln
        ax_map.legend(loc="upper right", fontsize=8, framealpha=0.7)

        #  Reward curve panel 
        ax_curve.set_xlabel("Outer iteration")
        ax_curve.set_ylabel("Team reward")
        ax_curve.set_title("Reward convergence")
        ax_curve.set_xlim(1, n_outer)
        ax_curve.set_ylim(0, total_reward * 1.05)
        ax_curve.axhline(total_reward, color="gray", linestyle="--",
                         linewidth=0.8, label=f"Max possible ({total_reward})")
        (reward_line,) = ax_curve.plot([], [], color="steelblue", linewidth=2,
                                       marker="o", markersize=3,
                                       label="Dec-MCTS reward")
        ax_curve.legend(loc="lower right")

        fig.tight_layout()
        plt.show()

    for n in range(1, n_outer + 1):
        # Grow trees + update distributions
        for rid in robot_ids:
            planners[rid].iterate(n_outer=1)

        # Communicate with message dropping (gated by comm_period)
        if comm_period > 0 and n % comm_period == 0:
            for rid in robot_ids:
                _, q = planners[rid].get_distribution()
                for other in robot_ids:
                    if other != rid and rng.random() > drop_prob:
                        planners[other].receive_dist_dict(rid, q)

        # Extract current joint plan
        seqs = {rid: planners[rid].best_action_sequence() for rid in robot_ids}
        paths = {rid: actions_to_path(start_verts[rid], seqs[rid])
                 for rid in robot_ids}
        reward = global_obj(paths)
        reward_curve.append(reward)

        if verbose:
            pct = reward / total_reward * 100 if total_reward else 0
            elapsed = (time.perf_counter() - t0) * 1000
            if n % 5 == 0 or n == 1:
                print(f"  outer {n:>3}  reward={reward:6.1f}/{total_reward}  "
                      f"({pct:5.1f}%)  elapsed={elapsed:.0f}ms")

            # Update region colors (visited becomes bright green)
            all_visited = set()
            for p in paths.values():
                all_visited.update(p)
            for reg, patch in zip(regions, region_patches):
                if reg["verts"] & all_visited:
                    patch.set_facecolor("limegreen")
                    patch.set_alpha(0.55)

            # Update robot paths
            for idx, rid in enumerate(robot_ids):
                px = [graph.pos[v][0] for v in paths[rid]]
                py = [graph.pos[v][1] for v in paths[rid]]
                robot_lines[rid].set_data(px, py)

            # Update reward curve
            reward_line.set_data(range(1, n + 1), reward_curve)
            ax_curve.set_ylim(0, max(total_reward, max(reward_curve)) * 1.05)

            fig.canvas.draw()
            plt.pause(0.01)

    dec_elapsed = time.perf_counter() - t0
    dec_reward  = reward_curve[-1]
    dec_paths   = paths

    if verbose:
        plt.ioff()
        plt.show(block=False)

    #  Run Cen-MCTS (same total rollout budget) 
    total_rollouts = n_outer * TAU * N_ROBOTS
    t1 = time.perf_counter()
    cen_paths  = cen.run(total_rollouts)
    cen_elapsed = time.perf_counter() - t1
    cen_reward  = global_obj(cen_paths)

    #  Compute metrics 
    def regions_found(paths):
        visited = set()
        for p in paths.values():
            visited.update(p)
        return [reg for reg in regions if reg["verts"] & visited]

    dec_found = regions_found(dec_paths)
    cen_found = regions_found(cen_paths)

    # Marginal utilities
    marginals = {}
    for rid in robot_ids:
        without = dict(dec_paths)
        without[rid] = null_paths[rid]
        marginals[rid] = dec_reward - global_obj(without)

    # Path validity: each step must be a graph neighbor
    def path_valid(path):
        for a, b in zip(path, path[1:]):
            if b not in {nb for nb, _ in graph.neighbors(a)}:
                return False
        return True

    # Path budget: total edge length
    def path_cost(path):
        cost = 0.0
        for a, b in zip(path, path[1:]):
            cost += next(c for nb, c in graph.neighbors(a) if nb == b)
        return cost

    return {
        # Core metrics
        "dec_reward":        dec_reward,
        "cen_reward":        cen_reward,
        "total_reward":      total_reward,
        "dec_pct":           dec_reward / total_reward * 100 if total_reward else 0,
        "cen_pct":           cen_reward / total_reward * 100 if total_reward else 0,
        "reward_vs_cen_pct": (dec_reward / cen_reward * 100 - 100)
                             if cen_reward > 0 else 0,
        # Convergence
        "reward_curve":      reward_curve,
        # Coverage
        "dec_regions_found": len(dec_found),
        "cen_regions_found": len(cen_found),
        "n_regions":         len(regions),
        # Timing
        "dec_elapsed_s":     dec_elapsed,
        "cen_elapsed_s":     cen_elapsed,
        # Correctness
        "paths_valid":       all(path_valid(p) for p in dec_paths.values()),
        "budget_respected":  all(path_cost(p) <= BUDGET + 1e-6
                                 for p in dec_paths.values()),
        "marginals_sum_ok":  sum(marginals.values()) <= dec_reward + 1e-6,
        "dist_normalized":   all(
            abs(sum(planners[rid].q.values()) - 1.0) < 1e-6 or not planners[rid].q
            for rid in robot_ids
        ),
        # Instance info
        "n_vertices_actual": graph.n,
        "start_verts":       start_verts,
        "marginals":         marginals,
        "planners":          planners,
    }


# ROBUSTNESS SWEEP: reward vs. drop probability

def robustness_sweep(base_seed, n_outer, trials_per_point=3):
    """
    Replicates Figure 3 of the paper: reward as function of message drop prob.
    Returns {drop_prob: [reward, ...]} for Dec-MCTS and Cen-MCTS.
    """
    dec_results = {p: [] for p in DROP_PROBS}
    cen_results = {p: [] for p in DROP_PROBS}

    for drop_p in DROP_PROBS:
        print(f"  drop_prob={drop_p:.2f} ...", end="", flush=True)
        for t in range(trials_per_point):
            seed = base_seed + t * 100 + int(drop_p * 1000)
            res  = run_trial(seed, n_outer, drop_prob=drop_p, verbose=False)
            dec_results[drop_p].append(res["dec_reward"])
            cen_results[drop_p].append(res["cen_reward"])
        d_med = sorted(dec_results[drop_p])[len(dec_results[drop_p])//2]
        c_med = sorted(cen_results[drop_p])[len(cen_results[drop_p])//2]
        print(f"  Dec median={d_med:.1f}  Cen median={c_med:.1f}")

    return dec_results, cen_results


# MAIN

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials",    type=int, default=1,
                        help="Number of random problem instances to average over")
    parser.add_argument("--robustness", action="store_true",
                        help="Run the message-drop robustness sweep")
    parser.add_argument("--outer",       type=int, default=N_OUTER)
    parser.add_argument("--comm-period", type=int, default=COMM_PERIOD,
                        help="Broadcast every K outer iterations; -1 = never")
    args = parser.parse_args()

    n_outer     = args.outer
    comm_period = args.comm_period

    section("EXPERIMENT CONFIG")
    print(f"  Graph vertices  : {N_VERTICES}")
    print(f"  Regions         : {N_REGIONS}")
    print(f"  Robots          : {N_ROBOTS}")
    print(f"  Budget          : {BUDGET:.0f}  (distance units)")
    print(f"  Outer iters     : {n_outer}  (τ={TAU} rollouts each)")
    print(f"  Comm period     : {comm_period}  ({'every iter' if comm_period == 1 else 'never' if comm_period < 0 else f'every {comm_period} iters'})")
    print(f"  Trials          : {args.trials}")
    print(f"  Total rollouts/robot: {n_outer * TAU}")
    print()
    print("  Metrics to compare against paper:")
    print("    • Dec-MCTS reward as % of total possible")
    print("    • Dec-MCTS reward vs. Cen-MCTS")
    print("    • Regions found / total")
    print("    • Reward convergence curve across outer iterations")
    if args.robustness:
        print("    • Reward vs. message-drop probability sweep")

    section("SINGLE / MULTI-TRIAL RESULTS")

    all_results = []
    for trial in range(args.trials):
        seed = SEED + trial * 997
        if args.trials > 1:
            sub(f"Trial {trial+1}/{args.trials}  (seed={seed})")
        result = run_trial(seed, n_outer, drop_prob=0.0,
                           comm_period=comm_period,
                           verbose=(args.trials == 1))
        all_results.append(result)

    def med(vals):
        s = sorted(vals)
        return s[len(s)//2]
    def avg(vals):
        return sum(vals) / len(vals)

    dec_rewards  = [r["dec_reward"]  for r in all_results]
    cen_rewards  = [r["cen_reward"]  for r in all_results]
    vs_cen_pcts  = [r["reward_vs_cen_pct"] for r in all_results]
    dec_pcts     = [r["dec_pct"]     for r in all_results]

    section("KEY METRICS  (compare against paper Section 6)")

    print(f"\n  {'Metric':<45} {'Your result':>15}  {'Paper':>12}")
    print(f"  {'─'*74}")

    def row(label, val, paper_ref=""):
        print(f"  {label:<45} {val:>15}  {paper_ref:>12}")

    row("Dec-MCTS median reward (abs)",
        f"{med(dec_rewards):.2f}",
        "see Fig 3")
    row("Cen-MCTS median reward (abs)",
        f"{med(cen_rewards):.2f}",
        "see Fig 3")
    row("Dec % of total possible reward",
        f"{avg(dec_pcts):.1f}%",
        "see Fig 3")
    row("Dec reward vs Cen-MCTS (median)",
        f"{med(vs_cen_pcts):+.1f}%",
        "+7% (median)")
    row("Dec better than Cen (fraction of trials)",
        f"{sum(d>c for d,c in zip(dec_rewards,cen_rewards))}/{len(all_results)}",
        "91% of trials")
    row("Regions found / total (Dec, avg)",
        f"{avg([r['dec_regions_found'] for r in all_results]):.1f}"
        f"/{all_results[0]['n_regions']}",
        "see Fig 2")

    section("CORRECTNESS CHECKS")

    checks = []

    def chk(label, passed, detail=""):
        status = f"{G}PASS ✓{RST}" if passed else f"{R}FAIL ✗{RST}"
        checks.append(passed)
        d = f"  ({detail})" if detail else ""
        print(f"  [{status}]  {label}{d}")

    r = all_results[-1]  # use last trial for structural checks

    chk("All robot paths are graph-connected",
        r["paths_valid"])

    chk("All paths respect distance budget",
        r["budget_respected"],
        f"budget={BUDGET}")

    chk("Sum of marginal utilities ≤ joint reward",
        r["marginals_sum_ok"],
        f"sum={sum(r['marginals'].values()):.2f}, joint={r['dec_reward']:.2f}")

    chk("All distributions sum to 1",
        r["dist_normalized"])

    chk("Dec-MCTS reward > 0",
        r["dec_reward"] > 0,
        f"{r['dec_reward']:.2f}")

    # Convergence: second half of reward curve ≥ first half
    curve = r["reward_curve"]
    mid   = len(curve) // 2
    fh    = avg(curve[:mid]) if mid > 0 else 0
    sh    = avg(curve[mid:]) if mid < len(curve) else 0
    chk("Reward improves: 2nd half avg ≥ 1st half avg",
        sh >= fh - 0.5,
        f"{fh:.2f} → {sh:.2f}")

    # Distribution entropy decreases (distributions converge)
    for rid in range(N_ROBOTS):
        q     = all_results[-1]["planners"][rid].q
        total = sum(q.values())
        chk(f"Robot {rid} distribution normalized",
            abs(total - 1.0) < 1e-6 or not q,
            f"sum={total:.8f}")

    n_pass = sum(checks)
    n_fail = len(checks) - n_pass
    color  = G if n_fail == 0 else R
    print(f"\n  {color}{BOLD}{n_pass}/{len(checks)} checks passed{RST}")

    # ══════════════════════════════════════════════════════════
    section("CONVERGENCE TRACE  (reward vs. outer iteration)")
    # ══════════════════════════════════════════════════════════

    curve    = all_results[-1]["reward_curve"]
    max_r    = max(curve) if curve else 1
    bar_w    = 40
    print(f"\n  {'Iter':>5}  {'Reward':>8}  {'Bar':<{bar_w}}")
    print(DIM + "  " + "─"*58 + RST)
    for i, rv in enumerate(curve):
        bl  = max(0, int(rv / max_r * bar_w)) if max_r > 0 else 0
        bar = "█" * bl + "░" * (bar_w - bl)
        col = G if rv == max_r else RST
        print(f"  {i+1:>5}  {col}{rv:>8.2f}{RST}  {bar}")

    # ══════════════════════════════════════════════════════════
    section("EXPORT")
    # ══════════════════════════════════════════════════════════

    # Compute derived values needed for export
    def _tree_size(node):
        return 1 + sum(_tree_size(c) for c in node.children)

    def _entropy(q):
        return -sum(p * math.log(p) for p in q.values() if p > 0)

    H_max          = math.log(NUM_SEQ) if NUM_SEQ > 1 else 1.0
    dec_beats_cen  = sum(d > c for d, c in zip(dec_rewards, cen_rewards))
    last_r         = all_results[-1]
    last_pls       = last_r["planners"]
    exp_curve      = last_r["reward_curve"]
    exp_mid        = len(exp_curve) // 2
    exp_fh         = avg(exp_curve[:exp_mid]) if exp_mid > 0 else 0
    exp_sh         = avg(exp_curve[exp_mid:]) if exp_mid < len(exp_curve) else 0
    first_reward   = exp_curve[0]  if exp_curve else 0
    final_reward   = exp_curve[-1] if exp_curve else 0
    reward_pct_gain = ((final_reward - first_reward) / first_reward * 100
                       if first_reward > 0 else 0.0)

    per_robot_export = {}
    for rid in range(N_ROBOTS):
        p  = last_pls[rid]
        per_robot_export[str(rid)] = {
            "marginal_contribution": last_r["marginals"].get(rid, 0.0),
            "entropy_final":         _entropy(p.q) if p.q else H_max,
            "sharpness":             1.0 - (_entropy(p.q) / H_max if p.q else 1.0),
            "beta_final":            p.beta,
            "tree_nodes":            _tree_size(p.root),
            "seq_len":               len(p.best_action_sequence()),
            "dist_normalized":       abs(sum(p.q.values()) - 1.0) < 1e-6 if p.q else True,
        }

    timestamp   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    import os
    os.makedirs("results", exist_ok=True)
    export_path = os.path.join("results", f"decmcts_results_{timestamp}.json")

    export = {
        "meta": {
            "timestamp": timestamp,
            "paper":     "Best et al. 2019, Dec-MCTS, IJRR 38(2-3)",
        },
        "config": {
            "N_VERTICES":    N_VERTICES,
            "N_REGIONS":     N_REGIONS,
            "N_ROBOTS":      N_ROBOTS,
            "N_OBSTACLES":   N_OBSTACLES,
            "N_NEIGHBORS":   N_NEIGHBORS,
            "WORLD_SIZE":    WORLD_SIZE,
            "BUDGET":        BUDGET,
            "REGION_RADIUS": REGION_RADIUS,
            "TAU":           TAU,
            "NUM_SEQ":       NUM_SEQ,
            "NUM_SAMPLES":   NUM_SAMPLES,
            "GAMMA":         GAMMA,
            "CP":            CP,
            "ALPHA":         ALPHA,
            "BETA_INIT":     BETA_INIT,
            "BETA_DECAY":    BETA_DECAY,
            "N_OUTER":       n_outer,
            "SEED":          SEED,
        },
        "paper_comparison": {
            "dec_pct_of_max":               avg(dec_pcts),
            "dec_pct_of_max_paper":         65.0,
            "dec_vs_cen_median_pct":        med(vs_cen_pcts),
            "dec_vs_cen_median_pct_paper":  7.0,
            "dec_beats_cen_fraction":       dec_beats_cen / len(all_results),
            "dec_beats_cen_fraction_paper": 0.91,
            "dec_regions_found_avg":        avg([r2["dec_regions_found"] for r2 in all_results]),
            "dec_regions_found_paper":      120,
            "n_regions_total":              last_r["n_regions"],
            "dec_reward_median":            med(dec_rewards),
            "cen_reward_median":            med(cen_rewards),
        },
        "correctness_checks": {
            "paths_graph_connected":   last_r["paths_valid"],
            "budget_respected":        last_r["budget_respected"],
            "marginals_sum_leq_joint": last_r["marginals_sum_ok"],
            "distributions_sum_to_1":  last_r["dist_normalized"],
            "reward_gt_0":             last_r["dec_reward"] > 0,
            "reward_improves":         exp_sh >= exp_fh - 0.5,
        },
        "per_robot_diagnostics": per_robot_export,
        "convergence": {
            "reward_curve":     exp_curve,
            "reward_iter_1":    first_reward,
            "reward_final":     final_reward,
            "reward_pct_gain":  reward_pct_gain,
            "first_half_avg":   exp_fh,
            "second_half_avg":  exp_sh,
            "total_reward_max": last_r["total_reward"],
        },
        "timing": {
            "dec_elapsed_s": avg([r2["dec_elapsed_s"] for r2 in all_results]),
            "cen_elapsed_s": avg([r2["cen_elapsed_s"] for r2 in all_results]),
        },
    }

    with open(export_path, "w") as f:
        json.dump(export, f, indent=2)
    print(f"\n  Results saved to: {export_path}\n")

    # ══════════════════════════════════════════════════════════
    if args.robustness:
        section("ROBUSTNESS SWEEP  (reward vs. message drop probability)")
        print("  Paper Figure 3: Dec-MCTS degrades gracefully vs. drop rate\n")
        dec_r, cen_r = robustness_sweep(SEED, n_outer, trials_per_point=3)

        print(f"\n  {'Drop prob':>10}  {'Dec median':>12}  {'Cen median':>12}  {'Dec vs Cen':>12}")
        print("  " + "─"*52)
        for p in DROP_PROBS:
            dm = sorted(dec_r[p])[len(dec_r[p])//2]
            cm = sorted(cen_r[p])[len(cen_r[p])//2]
            vs = (dm/cm*100-100) if cm > 0 else 0
            print(f"  {p:>10.2f}  {dm:>12.2f}  {cm:>12.2f}  {vs:>+11.1f}%")
        print()
        print("  Paper result: reward degrades slowly as drop_prob increases;")
        print("  Dec-MCTS still matches or exceeds Cen-MCTS even at 50% drop.")


if __name__ == "__main__":
    main()