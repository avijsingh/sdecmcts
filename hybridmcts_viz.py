"""
hybridmcts_viz.py
-----------------
Visualize the HybridMCTS meta-tree after N iterations on a toy problem.

Blue  nodes = CEN  (CenMCTS subtree — full communication)
Red   nodes = DEC  (DecMCTS subtrees — comm lost)

Node size ∝ visit count.  Gold path = greedy best plan.
Solid edges  = single-robot action  (from CEN parent)
Dashed edges = joint action         (from DEC parent)
"""

import random
from collections import deque

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

from hybridmcts import HybridMCTS, CommModel, NodeType


# ─────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────

SEED    = 7
N_ITER  = 100
BUDGET  = 4      # steps per robot

# Comm: moderate chance of comm loss, good chance of restoration
COMM_MODEL = CommModel(p_loss=0.3, p_restore=0.5, init_type=NodeType.CEN)

NODE_COLOR = {
    NodeType.CEN: "#4C9BE8",
    NodeType.DEC: "#E84C4C",
}
NODE_LABEL = {
    NodeType.CEN: "CEN — CenMCTS (full comm)",
    NodeType.DEC: "DEC — DecMCTS (comm lost)",
}


# ─────────────────────────────────────────────────────────
# TOY STATE: linear graph 0—1—…—9
# ─────────────────────────────────────────────────────────

N_V     = 10
REWARDS = {2: 4.0, 5: 6.0, 8: 3.0}


class LinearState:
    def __init__(self, vertex, budget):
        self.vertex = vertex
        self.budget = budget
        self.reward = REWARDS.get(vertex, 0.0)

    def get_legal_actions(self):
        if self.budget <= 0:
            return []
        acts = []
        if self.vertex > 0:          acts.append(self.vertex - 1)
        if self.vertex < N_V - 1:    acts.append(self.vertex + 1)
        return acts

    def take_action(self, action):
        return LinearState(action, self.budget - 1)

    def is_terminal_state(self):
        return self.budget <= 0


def global_obj(paths):
    visited = set()
    for p in paths.values():
        visited.update(p)
    return sum(REWARDS.get(v, 0.0) for v in visited)


def make_local_obj(my_id):
    def f(joint_seqs):
        my_acts    = set(joint_seqs.get(my_id, []))
        other_acts = set()
        for rid, acts in joint_seqs.items():
            if rid != my_id:
                other_acts.update(acts)
        return sum(REWARDS.get(v, 0.0) for v in my_acts if v not in other_acts)
    return f


# ─────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────

random.seed(SEED)

robot_ids   = [0, 1]
init_states = {0: LinearState(0, BUDGET), 1: LinearState(9, BUDGET)}
local_fns   = {rid: make_local_obj(rid) for rid in robot_ids}

planner = HybridMCTS(
    robot_ids        = robot_ids,
    init_states      = init_states,
    global_obj       = global_obj,
    local_obj_fns    = local_fns,
    comm_model       = COMM_MODEL,
    cen_rollout_depth = 6,
    max_dec_children  = 5,
    dec_outer_iters   = 2,
    dec_tau           = 3,
    dec_num_seq       = 4,
    dec_num_samples   = 6,
    dec_rollout_iters = 3,
)
planner.run(N_ITER)

best   = planner.best_paths()
types  = planner.best_node_type_sequence()
print(f"Best paths:          {best}")
print(f"Best type sequence:  {[t.value for t in types]}")


# ─────────────────────────────────────────────────────────
# TREE LAYOUT (BFS centred per depth level)
# ─────────────────────────────────────────────────────────

def layout_tree(root):
    levels = {}
    queue  = deque([(root, 0)])
    seen   = set()
    while queue:
        node, d = queue.popleft()
        if id(node) in seen:
            continue
        seen.add(id(node))
        levels.setdefault(d, []).append(node)
        for c in node.children:
            queue.append((c, d + 1))

    pos = {}
    for d, nodes in levels.items():
        n = len(nodes)
        for i, node in enumerate(nodes):
            pos[id(node)] = ((i - (n - 1) / 2.0) * 2.2, -d * 2.0)
    return pos


positions = layout_tree(planner.root)

# Greedy best path
best_path = [planner.root]
node = planner.root
while node.children:
    visited = [c for c in node.children if c.visits > 0]
    if not visited:
        break
    node = max(visited, key=lambda c: c.q())
    best_path.append(node)
best_ids = {id(n) for n in best_path}


# ─────────────────────────────────────────────────────────
# DRAW
# ─────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(18, 11))
ax.axis("off")

# ── edges ────────────────────────────────────────────────
queue = deque([planner.root])
seen  = set()
while queue:
    parent = queue.popleft()
    if id(parent) in seen:
        continue
    seen.add(id(parent))
    px, py = positions[id(parent)]
    for child in parent.children:
        cx, cy   = positions[id(child)]
        on_best  = id(parent) in best_ids and id(child) in best_ids
        ls       = "-"    if parent.node_type == NodeType.CEN else "--"
        lw       = 2.5    if on_best else 0.7
        col      = "gold" if on_best else "#888888"
        alpha    = 1.0    if on_best else 0.45
        zord     = 3      if on_best else 1
        ax.plot([px, cx], [py, cy],
                linestyle=ls, color=col, linewidth=lw,
                alpha=alpha, zorder=zord)
        queue.append(child)

# ── nodes ────────────────────────────────────────────────
queue = deque([planner.root])
seen  = set()
while queue:
    node = queue.popleft()
    if id(node) in seen:
        continue
    seen.add(id(node))
    x, y    = positions[id(node)]
    on_best = id(node) in best_ids
    color   = NODE_COLOR[node.node_type]
    size    = max(200, node.visits * 65)
    ec      = "gold"  if on_best else "white"
    lw      = 2.0     if on_best else 0.6
    zord    = 4       if on_best else 2

    ax.scatter(x, y, s=size, c=color, zorder=zord, edgecolors=ec, linewidths=lw)

    if node.visits > 0:
        ax.text(x, y, str(node.visits),
                ha="center", va="center",
                fontsize=7, fontweight="bold", color="white", zorder=zord + 1)

    # Type label on best-path nodes
    if on_best:
        ax.text(x + 0.1, y + 0.55, node.node_type.value.upper(),
                ha="center", va="bottom", fontsize=6,
                color="#444444", style="italic")

    # Mark root
    if node.parent is None:
        ax.text(x, y - 0.65, "ROOT", ha="center", va="top",
                fontsize=7, color="#333333")

    for child in node.children:
        queue.append(child)

# ── depth labels ─────────────────────────────────────────
depths = {}
queue = deque([(planner.root, 0)])
seen  = set()
while queue:
    node, d = queue.popleft()
    if id(node) in seen:
        continue
    seen.add(id(node))
    depths[id(node)] = d
    for c in node.children:
        queue.append((c, d + 1))

max_d = max(depths.values()) if depths else 0
x_min = min(x for x, _ in positions.values()) - 1.2
for d in range(max_d + 1):
    ax.text(x_min, -d * 2.0, f"depth {d}",
            ha="right", va="center", fontsize=8, color="#777777")

# ── legend ───────────────────────────────────────────────
patches = [
    mpatches.Patch(color=NODE_COLOR[t], label=NODE_LABEL[t])
    for t in NodeType
]
solid  = mlines.Line2D([], [], color="#888888", lw=1.5, ls="-",
                        label="Single-robot edge (CEN parent)")
dashed = mlines.Line2D([], [], color="#888888", lw=1.5, ls="--",
                        label="Joint-action edge (DEC parent)")
gold   = mlines.Line2D([], [], color="gold", lw=2.5,
                        label="Best plan (greedy Q)")
ax.legend(handles=patches + [solid, dashed, gold],
          loc="upper right", fontsize=9, framealpha=0.9,
          title="HybridMCTS Legend", title_fontsize=9)

# ── title ────────────────────────────────────────────────
n_cen = sum(1 for n in depths if True)  # total nodes
all_nodes = list(depths.keys())
n_cen_nodes = sum(
    1 for nid in all_nodes
    for node in [next(n for n in [planner.root] + [c for c in planner.root.children]
                      if id(n) == nid)]
    if node.node_type == NodeType.CEN
) if False else "?"  # skip expensive recount for title

ax.set_title(
    f"HybridMCTS Meta-Tree — {N_ITER} iterations\n"
    f"Blue = CEN subtree (CenMCTS, full comm) · "
    f"Red = DEC subtree (DecMCTS, comm lost) · "
    f"node size ∝ visits",
    fontsize=12, pad=12,
)

plt.tight_layout()
out = "hybridmcts_tree.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved {out}")
