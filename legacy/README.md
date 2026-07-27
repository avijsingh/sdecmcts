# legacy/

Code that is not on the paper's code path but was kept because it is runnable
and plausibly useful again (extra baselines, earlier designs, alternative
domain formulations). Nothing here is imported by `benchmarks/semidec_*.py`.

**These files are archived, not maintained.** Because they moved out of the
repository root, they need the root on `PYTHONPATH`. Two invocations are needed,
both verified after the move:

Most files run from the repository root:

```bash
PYTHONPATH=. python3 -m legacy.benchmarks.run_pomdp_benchmarks
```

`decmcts_test.py` and `commsmcts_viz.py` import their dependencies as bare
names (`cenmcts`, `commsmcts`), so run those from inside `legacy/`:

```bash
cd legacy && PYTHONPATH=.. python3 decmcts_test.py
```

One import fix was applied during archiving: `benchmarks/run_benchmarks.py` now
imports its sibling domain modules as `from tiger import ...` rather than
`from benchmarks.tiger import ...`, because the live top-level `benchmarks/`
package shadows `legacy/benchmarks/`. Its behavior is unchanged.

## Baselines and predecessors

| File | What it is | Why it was kept |
|---|---|---|
| `cpde_decmcts.py` | Centralized Planning, Decentralized Execution via POMCP-style observation-branching MCTS | A comparison class reviewers commonly ask about. Its model interface still matches the live `benchmarks/pomdp_*.py` domains. |
| `commsmcts.py`, `commsmcts_viz.py` | Semi-decentralized MCTS over FULL/DELAYED/NONE communication regimes, with D-UCT | An earlier design of the same idea as SDecMCTS; useful for related-work framing or an ablation. |
| `sdecmcts.py` | The original SDecMCTS: **one** joint tree whose nodes carry a communication partition | Direct predecessor of `semidec/sdecmcts.py`, which uses a different (per-agent) architecture. Kept for design provenance. |
| `cenmcts.py` | Centralized joint-tree MCTS baseline (orienteering) | Small, and the baseline `decmcts_test.py` and `commsmcts.py` compare against. |

## Harnesses and drivers

| File | What it is |
|---|---|
| `decmcts_test.py` | Multi-agent orienteering scenario with validity/metrics checks for vanilla DecMCTS. Runnable — `decmcts.py` is still live. |
| `benchmarks/run_benchmarks.py` | Driver for the offline orienteering-style formulations below. |
| `benchmarks/run_pomdp_benchmarks.py`, `benchmarks/run_cpde_offline.py` | Drivers for the DecMCTS / CPDE baselines against the live `benchmarks/pomdp_*.py` domains. |
| `benchmarks/{tiger,mars,medevac,labyrinth}.py` | Offline orienteering-style formulations of the four domains. Distinct from the live `pomdp_*.py` / `*_online.py` formulations — do not confuse them. |
| `parse_results.py`, `results/` | Parser and stored JSON for the April 2026 DecMCTS orienteering runs. |

## Known-unrunnable

`benchmarks/medtest.py` imports `RSSDA` and `decPOMDP`, which were **never**
committed to this repository, so it cannot run as-is. It was kept anyway
because its `triggers_none()` / `triggers_semi()` / `triggers_full()`
synchronization-trigger definitions and its `to_sparse_format()` exporter are
the starting point for comparing against an exact Dec-POMDP solver. Note that
its MEDEVAC domain itself is a duplicate of the live
`benchmarks/pomdp_medevac.py`.

## Deleted rather than archived

For the record, these were removed outright (recoverable from git history):

- `hybridmcts_test.py`, `hybridmcts_viz.py` — both import a `hybridmcts`
  module that was never committed, so neither can ever run. Their orienteering
  harness is duplicated in `decmcts_test.py`, and the partition-based design
  they describe survives in `sdecmcts.py`.
- `mcts.py`, `2048/`, `SingleAgentDroneSearch/` — textbook single-agent MCTS
  demos, unrelated to the multi-agent work.
- Stale figures (`hybridmcts_tree.png`, `decmcts_sar.png`,
  `decmcts_sar_anim.gif`, `decmcts_medevac.png`, `decmcts_ablation_stale.png`)
  and the April `results_*.txt` run logs. `commsmcts_tree.png` was kept here
  instead, since it is the output of `commsmcts_viz.py`.
- `best2018decmcts.pdf` — a third-party paper. It is still present in git
  history; see the note in the top-level cleanup summary if this repository is
  ever published.
