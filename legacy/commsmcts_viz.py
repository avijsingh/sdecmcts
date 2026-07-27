"""
commsmcts_viz.py
----------------
Visualize the CommMCTS tree after N iterations on a toy two-robot problem.

Each node is colored by its communication regime:
  Blue   = FULL    (CenMCTS-style, single-robot edges, standard UCB)
  Orange = DELAYED (stale-distribution-guided, D-UCT)
  Red    = NONE    (fully independent, D-UCT)

Node size is proportional to visit count.
The gold path highlights the greedy best plan.
Edge style distinguishes action type:
  Solid  = single-robot action (FULL parent)
  Dashed = joint action (NONE/DELAYED parent)
"""

import math
import random
from collections import deque

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

from commsmcts import CommMCTS, CommModel, CommRegime


# ─────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────

SEED    = 7
N_ITER  = 80        # MCTS iterations to run
BUDGET  = 3         # steps per robot (keeps tree small enough to visualize)

# Comm model: transitions that guarantee all three regimes appear
TRANSITION = {
    CommRegime.FULL:    {CommRegime.FULL: 0.55, CommRegime.DELAYED: 0.25, CommRegime.NONE: 0.20},
    CommRegime.DELAYED: {CommRegime.FULL: 0.30, CommRegime.DELAYED: 0.40, CommRegime.NONE: 0.30},
    CommRegime.NONE:    {CommRegime.FULL: 0.20, CommRegime.DELAYED: 0.30, CommRegime.NONE: 0.50},
}

# Visual style
REGIME_COLOR = {
    CommRegime.FULL:    "#4C9BE8",
    CommRegime.DELAYED: "#F5A623",
    CommRegime.NONE:    "#E84C4C",
}
REGIME_LABEL = {
    CommRegime.FULL:    "FULL — CenMCTS style",
    CommRegime.DELAYED: "DELAYED — stale-guided",
    CommRegime.NONE:    "NONE — DecMCTS style",
}


# ─────────────────────────────────────────────────────────
# TOY STATE: linear graph  0 — 1 — 2 — … — 9
# ─────────────────────────────────────────────────────────

N_V     = 10
REWARDS = {2: 4.0, 5: 6.0, 8: 3.0}


class LinearState:
    """
    Robot on a 1-D integer graph 0..N_V-1 with a step budget.
    Legal actions: move one vertex left or right (if budget > 0).
    """
    def __init__(self, vertex, budget):
        self.vertex = vertex
        self.budget = budget
        self.reward = REWARDS.get(vertex, 0.0)

    def get_legal_actions(self):
        if self.budget <= 0:
            return []
        acts = []
        if self.vertex > 0:
            acts.append(self.vertex - 1)
        if self.vertex < N_V - 1:
            acts.append(self.vertex + 1)
        return acts

    def take_action(self, action):
        return LinearState(action, self.budget - 1)

    def is_terminal_state(self):
        return self.budget <= 0


def global_obj(paths):
    """Union reward: sum of weights of all vertices visited by any robot."""
    visited = set()
    for p in paths.values():
        visited.update(p)
    return sum(REWARDS.get(v, 0.0) for v in visited)


def make_local_obj(my_id):
    """Marginal contribution of robot my_id (Dec-MCTS style local utility)."""
    def f(paths):
        my_verts    = set(paths.get(my_id, []))
        other_verts = set()
        for rid, p in paths.items():
            if rid != my_id:
                other_verts.update(p)
        return sum(REWARDS.get(v, 0.0) for v in my_verts if v not in other_verts)
    return f


# ─────────────────────────────────────────────────────────
# RUN CommMCTS
# ─────────────────────────────────────────────────────────

random.seed(SEED)

robot_ids   = [0, 1]
init_states = {
    0: LinearState(0, BUDGET),
    1: LinearState(9, BUDGET),
}
local_fns = {rid: make_local_obj(rid) for rid in robot_ids}

comm_model = CommModel(
    transition=TRANSITION,
    initial_regime=CommRegime.FULL,
)

planner = CommMCTS(
    robot_ids       = robot_ids,
    init_states     = init_states,
    global_obj      = global_obj,
    local_obj_fns   = local_fns,
    comm_model      = comm_model,
    rollout_depth   = 6,
    max_dec_children = 4,
)

planner.run(N_ITER)
best = planner.best_paths()
regime_seq = planner.best_regime_sequence()
print(f"Tree: {len(planner._all_nodes)} nodes after {N_ITER} iterations")
print(f"Best paths: {best}")
print(f"Best regime sequence: {[r.value for r in regime_seq]}")


# ─────────────────────────────────────────────────────────
# TREE LAYOUT  (BFS hierarchical, centred per level)
# ─────────────────────────────────────────────────────────

def layout_tree(root):
    """
    Assign (x, y) positions using a two-pass BFS layout.

    Pass 1: assign each node to a (depth, slot) in BFS order.
    Pass 2: centre each depth-level horizontally.
    Returns dict {id(node): (x, y)}.
    """
    levels  = {}   # depth -> [node, ...]
    order   = {}   # id(node) -> depth

    queue = deque([(root, 0)])
    while queue:
        node, depth = queue.popleft()
        if id(node) in order:
            continue
        order[id(node)] = depth
        levels.setdefault(depth, []).append(node)
        for child in node.children:
            queue.append((child, depth + 1))

    positions = {}
    X_SPREAD  = 2.2   # horizontal spacing between siblings
    Y_SPREAD  = 2.0   # vertical spacing between levels

    for depth, nodes in levels.items():
        n = len(nodes)
        for i, node in enumerate(nodes):
            x = (i - (n - 1) / 2.0) * X_SPREAD
            y = -depth * Y_SPREAD
            positions[id(node)] = (x, y)
    return positions


positions = layout_tree(planner.root)


# ─────────────────────────────────────────────────────────
# COLLECT BEST PATH
# ─────────────────────────────────────────────────────────

best_path = []
node = planner.root
best_path.append(node)
while node.children:
    visited = [c for c in node.children if c.visits > 0]
    if not visited:
        break
    node = max(visited, key=lambda c: c.q())
    best_path.append(node)
best_path_ids = {id(n) for n in best_path}


# ─────────────────────────────────────────────────────────
# DRAW
# ─────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(18, 11))
ax.axis("off")

# ── edges ────────────────────────────────────────────────

def draw_edges(root, positions, ax, best_path_set):
    queue = deque([root])
    seen  = set()
    while queue:
        parent = queue.popleft()
        if id(parent) in seen:
            continue
        seen.add(id(parent))
        px, py = positions[id(parent)]
        for child in parent.children:
            cx, cy = positions[id(child)]
            on_best = id(parent) in best_path_set and id(child) in best_path_set

            # Edge style: solid for FULL-parent (single-robot), dashed for NONE/DELAYED
            ls  = "-"  if parent.regime == CommRegime.FULL else "--"
            lw  = 2.5  if on_best else 0.7
            col = "gold" if on_best else "#888888"
            zord = 3   if on_best else 1
            alpha = 1.0 if on_best else 0.45

            ax.plot([px, cx], [py, cy],
                    linestyle=ls, color=col, linewidth=lw,
                    alpha=alpha, zorder=zord)
            queue.append(child)

draw_edges(planner.root, positions, ax, best_path_ids)

# ── nodes ────────────────────────────────────────────────

def draw_nodes(root, positions, ax, best_path_set):
    queue = deque([root])
    seen  = set()
    while queue:
        node = queue.popleft()
        if id(node) in seen:
            continue
        seen.add(id(node))

        x, y   = positions[id(node)]
        color  = REGIME_COLOR[node.regime]
        on_best = id(node) in best_path_set
        size   = max(180, node.visits * 60)
        ec     = "gold" if on_best else "white"
        lw     = 2.0    if on_best else 0.6
        zord   = 4      if on_best else 2

        ax.scatter(x, y, s=size, c=color, zorder=zord,
                   edgecolors=ec, linewidths=lw)

        # Visit count label (only for nodes with visits)
        if node.visits > 0:
            ax.text(x, y, str(node.visits),
                    ha="center", va="center",
                    fontsize=6.5, fontweight="bold",
                    color="white", zorder=zord + 1)

        # Regime annotation for root
        if node.parent is None:
            ax.text(x, y - 0.6, "ROOT", ha="center", va="top",
                    fontsize=7, color="#333333")

        for child in node.children:
            queue.append(child)

draw_nodes(planner.root, positions, ax, best_path_ids)

# ── regime annotation on best-path nodes ─────────────────

for i, node in enumerate(best_path):
    x, y = positions[id(node)]
    label = node.regime.value.upper()
    ax.text(x + 0.15, y + 0.55, label,
            ha="center", va="bottom",
            fontsize=6, color="#555555",
            style="italic")

# ── depth labels on left margin ───────────────────────────

depths = {}
queue = deque([(planner.root, 0)])
seen = set()
while queue:
    node, d = queue.popleft()
    if id(node) in seen:
        continue
    seen.add(id(node))
    depths[id(node)] = d
    for c in node.children:
        queue.append((c, d + 1))

max_depth = max(depths.values()) if depths else 0
x_min = min(x for x, y in positions.values()) - 1.0
for d in range(max_depth + 1):
    y = -d * 2.0
    ax.text(x_min, y, f"depth {d}", ha="right", va="center",
            fontsize=8, color="#777777")

# ── legend ───────────────────────────────────────────────

regime_patches = [
    mpatches.Patch(color=REGIME_COLOR[r], label=REGIME_LABEL[r])
    for r in CommRegime
]
solid_line  = mlines.Line2D([], [], color="#888888", lw=1.5, linestyle="-",
                             label="Single-robot edge (FULL parent)")
dashed_line = mlines.Line2D([], [], color="#888888", lw=1.5, linestyle="--",
                             label="Joint-action edge (NONE/DELAYED parent)")
gold_line   = mlines.Line2D([], [], color="gold", lw=2.5,
                             label="Best plan (greedy Q descent)")

ax.legend(
    handles=regime_patches + [solid_line, dashed_line, gold_line],
    loc="upper right",
    fontsize=9,
    framealpha=0.9,
    title="CommMCTS Legend",
    title_fontsize=9,
)

# ── title ────────────────────────────────────────────────

ax.set_title(
    f"CommMCTS Tree — {N_ITER} iterations, {len(planner._all_nodes)} nodes\n"
    f"2 robots on linear graph  |  budget={BUDGET}  |  node size ∝ visits",
    fontsize=13,
    pad=12,
)

plt.tight_layout()
out = "commsmcts_tree.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved {out}")
