"""
run_benchmarks.py
-----------------
Evaluates Dec-MCTS and SDecMCTS on four standard multi-agent benchmarks:
  Tiger, Mars Rover, Maritime MedEvac, Labyrinth

For each benchmark, the following planners are compared:
  CenMCTS        — centralized joint planning (upper bound)
  DecMCTS        — fully decentralized, free communication (lower bound)
  DecMCTS-gated  — Dec-MCTS with communication restricted to the overlap of:
                     (a) the fixed Dec-MCTS comm period  AND
                     (b) when the benchmark physically allows communication
                   Run at comm_period ∈ {1, 5, ∞}
  SDecMCTS       — partition-based semi-decentralized (this work)
                   tested at three comm link probabilities: 1.0, 0.6, 0.1

NOTE: All benchmarks use MDP relaxations (true state known to all planners).
This is a standard upper-bound baseline; RSSDA uses POMDP solvers with
partial observability.

Communication models per benchmark:
  Tiger    — always allowed (agents share a room)
  Mars     — allowed when same cell OR at BASE relay station
  MedEvac  — allowed when helicopter and ship are co-located (handoff point)
  Labyrinth— allowed when co-located OR at the start (hub) node

Usage:
    python run_benchmarks.py [--benchmarks tiger mars medevac labyrinth]
                             [--sdec-iters N] [--cen-iters N] [--trials K]
                             [--seed S]
"""

import sys
import os
import math
import time
import random
import argparse

# Allow importing legacy/ (cenmcts, sdecmcts) and this directory's own domain
# modules. The latter are imported as bare names rather than as benchmarks.*
# because the live top-level benchmarks/ package shadows this one.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cenmcts  import CenMCTS
from decmcts  import DecMCTS, DecMCTSTeam
from sdecmcts import SDecMCTS, CommGraph

from tiger     import make_tiger_benchmark
from mars      import make_mars_benchmark
from medevac   import make_medevac_benchmark
from labyrinth import make_labyrinth_benchmark


# ── DEFAULTS ──────────────────────────────────────────────────────────────────

SDEC_ITERS       = 300
CEN_ITERS        = 1000
DEC_OUTER_ITERS  = 100
DEC_TAU          = 10
COMM_PROBS       = [1.0, 0.6, 0.1]
# comm_period values for gated Dec-MCTS:
#   1  = try to communicate every outer iteration (most frequent)
#   5  = try every 5 outer iterations
#  -1  = never communicate
GATED_COMM_PERIODS = [1, 5, -1]
TRIALS           = 5
SEED             = 42

# DecMCTS hyperparams (used by run_decmcts / run_decmcts_gated)
SDEC_DEC_NUM_SEQ     = 10
SDEC_DEC_NUM_SAMPLES = 20

# SDecMCTS hyperparams
# Single-tree design: only rollout_depth is needed (no sub-planner knobs).
# Set to 2*R*H to cover full horizon for all robots (turn-by-turn tree).
SDEC_ROLLOUT_DEPTH   = 50


# ── FORMATTING ────────────────────────────────────────────────────────────────

def section(title):
    w = 70
    print(f"\n{'═'*w}")
    print(f"  {title}")
    print(f"{'═'*w}")

def row(label, *vals, width=28):
    vals_str = "  ".join(f"{v:>10}" for v in vals)
    print(f"  {label:<{width}}  {vals_str}")


# ── PLANNER RUNNERS ───────────────────────────────────────────────────────────

def run_cenmcts(robot_ids, init_states, global_obj, n_iter, seed):
    random.seed(seed)
    starts = {rid: s.vertex for rid, s in init_states.items()}
    cen = CenMCTS(
        init_states   = init_states,
        global_obj    = global_obj,
        starts        = starts,
        rollout_depth = SDEC_ROLLOUT_DEPTH,
    )
    t0 = time.perf_counter()
    cen.run(n_iter)
    elapsed = time.perf_counter() - t0
    paths   = cen._best_paths()
    reward  = global_obj(paths)
    return reward, elapsed


def run_decmcts(robot_ids, init_states, global_obj, local_obj_fns,
                n_outer, tau, seed):
    random.seed(seed)
    planners = {
        rid: DecMCTS(
            robot_id         = rid,
            robot_ids        = robot_ids,
            init_state       = init_states[rid],
            local_utility_fn = local_obj_fns[rid],
            tau              = tau,
            num_seq          = SDEC_DEC_NUM_SEQ,
            num_samples      = SDEC_DEC_NUM_SAMPLES,
        )
        for rid in robot_ids
    }
    t0 = time.perf_counter()
    DecMCTSTeam(planners).iterate_and_communicate(n_outer=n_outer, comm_period=1)
    elapsed = time.perf_counter() - t0

    # Build paths: [initial_vertex] + best_action_sequence
    paths = {
        rid: [init_states[rid].vertex] + planners[rid].best_action_sequence()
        for rid in robot_ids
    }
    reward = global_obj(paths)
    return reward, elapsed


def run_decmcts_gated(robot_ids, init_states, global_obj, local_obj_fns,
                      can_communicate, n_outer, tau, comm_period, seed):
    """
    Offline Dec-MCTS with communication gated by BOTH:
      (1) the fixed Dec-MCTS period  (comm_period)
      (2) the benchmark's physical comm constraint (can_communicate)

    At outer iteration i, communication is attempted only when
      i % comm_period == 0
    AND
      can_communicate holds at any step along the current best joint plan.

    This matches the paper setting: "only update/communicate at the overlap of
    the Dec-MCTS fixed period and when the benchmark actually allows it."

    comm_period=-1 disables communication entirely.
    """
    random.seed(seed)
    planners = {
        rid: DecMCTS(
            robot_id         = rid,
            robot_ids        = robot_ids,
            init_state       = init_states[rid],
            local_utility_fn = local_obj_fns[rid],
            tau              = tau,
            num_seq          = SDEC_DEC_NUM_SEQ,
            num_samples      = SDEC_DEC_NUM_SAMPLES,
        )
        for rid in robot_ids
    }

    def _can_comm_along_plan():
        """
        True if can_communicate holds at any step of the current best joint plan.

        Simulates each robot's state through its planned action sequence and
        checks comm eligibility at each step (including the initial positions).
        This is a much better proxy than checking only after the first action,
        which almost always returns False when robots start apart.
        """
        seqs   = {rid: planners[rid].best_action_sequence() for rid in robot_ids}
        states = {rid: init_states[rid] for rid in robot_ids}
        if can_communicate(states[robot_ids[0]], states[robot_ids[1]]):
            return True
        max_len = max((len(s) for s in seqs.values()), default=0)
        for step in range(max_len):
            for rid in robot_ids:
                if step < len(seqs[rid]):
                    states[rid] = states[rid].take_action(seqs[rid][step])
            if can_communicate(states[robot_ids[0]], states[robot_ids[1]]):
                return True
        return False

    t0 = time.perf_counter()

    for i in range(1, n_outer + 1):
        for p in planners.values():
            p.iterate(1)

        if comm_period > 0 and i % comm_period == 0:
            if _can_comm_along_plan():
                for rid, planner in planners.items():
                    _, q = planner.get_distribution()
                    for other_id, other_planner in planners.items():
                        if other_id != rid:
                            other_planner.receive_dist_dict(rid, q)

    elapsed = time.perf_counter() - t0
    paths = {
        rid: [init_states[rid].vertex] + planners[rid].best_action_sequence()
        for rid in robot_ids
    }
    reward = global_obj(paths)
    return reward, elapsed


def run_sdecmcts(robot_ids, init_states, global_obj, local_obj_fns,
                 p_link, n_iter, seed):
    random.seed(seed)
    comm_graph = CommGraph(robot_ids, p_link=p_link)
    sdec = SDecMCTS(
        robot_ids      = robot_ids,
        init_states    = init_states,
        global_obj     = global_obj,
        local_obj_fns  = local_obj_fns,
        comm_graph     = comm_graph,
        rollout_depth  = SDEC_ROLLOUT_DEPTH,
    )
    t0 = time.perf_counter()
    sdec.run(n_iter)
    elapsed = time.perf_counter() - t0
    paths   = sdec.best_paths()
    reward  = global_obj(paths)
    return reward, elapsed


# ── SINGLE BENCHMARK ─────────────────────────────────────────────────────────

def run_benchmark(name, factory_fn, factory_kwargs,
                  sdec_iters, cen_iters, dec_outer, dec_tau,
                  trials, seed):
    """
    Run one benchmark across all planners and comm probabilities.
    Returns a results dict.
    """
    section(f"BENCHMARK: {name.upper()}")

    cen_rewards, cen_times   = [], []
    dec_rewards, dec_times   = [], []
    gated_results = {cp: {"rewards": [], "times": []} for cp in GATED_COMM_PERIODS}
    sdec_results  = {p:  {"rewards": [], "times": []} for p  in COMM_PROBS}

    for t in range(trials):
        trial_seed = seed + t * 137

        # Fresh problem instance per trial (5-tuple: includes can_communicate)
        robot_ids, init_states, global_obj, local_obj_fns, can_comm = \
            factory_fn(**factory_kwargs)

        # CenMCTS
        r, elapsed = run_cenmcts(robot_ids, init_states, global_obj,
                                  cen_iters, trial_seed)
        cen_rewards.append(r)
        cen_times.append(elapsed)

        # DecMCTS (free communication — benchmark's can_communicate not applied)
        r, elapsed = run_decmcts(robot_ids, init_states, global_obj,
                                  local_obj_fns, dec_outer, dec_tau,
                                  trial_seed)
        dec_rewards.append(r)
        dec_times.append(elapsed)

        # DecMCTS-gated: communication at overlap of fixed period + benchmark constraint
        for cp in GATED_COMM_PERIODS:
            r, elapsed = run_decmcts_gated(
                robot_ids, init_states, global_obj, local_obj_fns,
                can_comm, dec_outer, dec_tau, cp, trial_seed,
            )
            gated_results[cp]["rewards"].append(r)
            gated_results[cp]["times"].append(elapsed)

        # SDecMCTS at each comm probability
        for p in COMM_PROBS:
            r, elapsed = run_sdecmcts(robot_ids, init_states, global_obj,
                                       local_obj_fns, p, sdec_iters,
                                       trial_seed)
            sdec_results[p]["rewards"].append(r)
            sdec_results[p]["times"].append(elapsed)

    def med(xs):
        s = sorted(xs)
        return s[len(s) // 2]

    def avg(xs):
        return sum(xs) / len(xs) if xs else 0.0

    # ── Print results table ───────────────────────────────────────────────────
    print(f"\n  Trials: {trials}   CenMCTS iters: {cen_iters}   "
          f"DecMCTS outer: {dec_outer}   SDecMCTS iters: {sdec_iters}\n")

    header_vals = ["Median R", "Avg R", "Avg t(s)"]
    row("Planner", *header_vals)
    print("  " + "─" * 62)

    row("CenMCTS (upper bound)",
        f"{med(cen_rewards):.2f}",
        f"{avg(cen_rewards):.2f}",
        f"{avg(cen_times):.2f}s")

    row("DecMCTS (free comm)",
        f"{med(dec_rewards):.2f}",
        f"{avg(dec_rewards):.2f}",
        f"{avg(dec_times):.2f}s")

    for cp in GATED_COMM_PERIODS:
        rs = gated_results[cp]["rewards"]
        ts = gated_results[cp]["times"]
        if cp == -1:
            label = "DecMCTS-gated (no comm)"
        else:
            label = f"DecMCTS-gated (period={cp})"
        row(label,
            f"{med(rs):.2f}",
            f"{avg(rs):.2f}",
            f"{avg(ts):.2f}s")

    for p in COMM_PROBS:
        rs = sdec_results[p]["rewards"]
        ts = sdec_results[p]["times"]
        label = f"SDecMCTS (p_link={p:.1f})"
        row(label,
            f"{med(rs):.2f}",
            f"{avg(rs):.2f}",
            f"{avg(ts):.2f}s")

    # ── Summary relative to CenMCTS ───────────────────────────────────────────
    cen_med = med(cen_rewards) if med(cen_rewards) != 0 else 1.0
    print(f"\n  Relative to CenMCTS median ({med(cen_rewards):.2f}):")
    row(f"  DecMCTS (free comm)",
        f"{100*med(dec_rewards)/cen_med:.1f}%", "", "")
    for cp in GATED_COMM_PERIODS:
        rs = gated_results[cp]["rewards"]
        label = "no comm" if cp == -1 else f"period={cp}"
        row(f"  DecMCTS-gated ({label})",
            f"{100*med(rs)/cen_med:.1f}%", "", "")
    for p in COMM_PROBS:
        rs = sdec_results[p]["rewards"]
        row(f"  SDecMCTS p={p:.1f}",
            f"{100*med(rs)/cen_med:.1f}%", "", "")

    return {
        "cen":   {"rewards": cen_rewards,  "times": cen_times},
        "dec":   {"rewards": dec_rewards,  "times": dec_times},
        "gated": gated_results,
        "sdec":  sdec_results,
    }


# ── MAIN ──────────────────────────────────────────────────────────────────────

BENCHMARK_REGISTRY = {
    "tiger":    (make_tiger_benchmark,    {}),
    "mars":     (make_mars_benchmark,     {}),
    "medevac":  (make_medevac_benchmark,  {}),
    "labyrinth":(make_labyrinth_benchmark,{}),
}


def main():
    parser = argparse.ArgumentParser(description="SDecMCTS benchmark runner")
    parser.add_argument(
        "--benchmarks", nargs="+",
        choices=list(BENCHMARK_REGISTRY.keys()) + ["all"],
        default=["all"],
        help="Which benchmarks to run (default: all)",
    )
    parser.add_argument("--sdec-iters", type=int, default=SDEC_ITERS)
    parser.add_argument("--cen-iters",  type=int, default=CEN_ITERS)
    parser.add_argument("--dec-outer",  type=int, default=DEC_OUTER_ITERS)
    parser.add_argument("--dec-tau",    type=int, default=DEC_TAU)
    parser.add_argument("--trials",     type=int, default=TRIALS)
    parser.add_argument("--seed",       type=int, default=SEED)
    args = parser.parse_args()

    benchmarks = (
        list(BENCHMARK_REGISTRY.keys())
        if "all" in args.benchmarks
        else args.benchmarks
    )

    section("SDecMCTS BENCHMARK SUITE")
    print(f"  Benchmarks  : {', '.join(benchmarks)}")
    print(f"  Trials      : {args.trials}")
    print(f"  SDecMCTS    : {args.sdec_iters} iterations")
    print(f"  CenMCTS     : {args.cen_iters} iterations")
    print(f"  DecMCTS     : {args.dec_outer} outer × {args.dec_tau} tau")
    print(f"  p_link sweep: {COMM_PROBS}")
    print(f"  Seed        : {args.seed}")
    print()
    print("  NOTE: All benchmarks use MDP relaxations (oracle state known).")
    print("  RSSDA uses POMDP solvers — direct numeric comparison requires")
    print("  caution; qualitative coordination behaviour is the key signal.")

    all_results = {}
    for name in benchmarks:
        factory_fn, factory_kwargs = BENCHMARK_REGISTRY[name]
        results = run_benchmark(
            name         = name,
            factory_fn   = factory_fn,
            factory_kwargs = factory_kwargs,
            sdec_iters   = args.sdec_iters,
            cen_iters    = args.cen_iters,
            dec_outer    = args.dec_outer,
            dec_tau      = args.dec_tau,
            trials       = args.trials,
            seed         = args.seed,
        )
        all_results[name] = results

    # ── Cross-benchmark summary ───────────────────────────────────────────────
    section("CROSS-BENCHMARK SUMMARY")
    print()
    row("Benchmark",
        "Cen", "Dec(free)",
        "Dec(p=1)", "Dec(p=5)", "Dec(none)",
        "SDec 1.0", "SDec 0.6", "SDec 0.1",
        width=12)
    print("  " + "─" * 96)

    def med(xs):
        s = sorted(xs)
        return s[len(s) // 2]

    for name, res in all_results.items():
        cen_med   = med(res["cen"]["rewards"])
        dec_med   = med(res["dec"]["rewards"])
        gated_meds = [med(res["gated"][cp]["rewards"]) for cp in GATED_COMM_PERIODS]
        sdec_meds  = [med(res["sdec"][p]["rewards"])   for p  in COMM_PROBS]
        row(name,
            f"{cen_med:.1f}",
            f"{dec_med:.1f}",
            *[f"{v:.1f}" for v in gated_meds],
            *[f"{v:.1f}" for v in sdec_meds],
            width=12)

    print()


if __name__ == "__main__":
    main()
