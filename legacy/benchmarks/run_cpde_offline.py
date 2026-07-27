"""
run_cpde_offline.py
-------------------
Offline CPDE (Centralized Planning, Decentralized Execution) runner.

Protocol (three phases, executed once per trial):
  1. PLAN  — CPDEDecMCTSTeam.plan_joint() runs n_iterations POMCP simulations
             from the initial particles.  Both agents' obs-branching trees are
             trained on the same joint trajectories.  No replanning ever.
  2. EXTRACT — team.extract_policies() walks each agent's tree and builds a
               ConditionalPolicy: obs_history → action.  Policies are static
               from this point on.
  3. EXECUTE — each agent calls policy.get_action(), executes it, receives its
               local observation, and calls policy.record(action, local_obs).
               No communication, no replanning, no particle filter updates.

Reference values (AAMAS 2026, RS-SDA* paper, Tiger Dec-POMDP):
  h=8:  RS-MAA* 12.217  RS-SDA* 27.215  Cen 47.717
  h=9:  RS-MAA* 15.572  RS-SDA* 30.905  Cen 53.474
  h=10: RS-MAA* 15.184  RS-SDA* 34.724  Cen 60.510

Usage
-----
  python benchmarks/run_cpde_offline.py [--benchmark tiger|medevac|mars]
                                        [--horizons 8 9 10]
                                        [--trials K]
                                        [--iters N]
                                        [--rollout D]
                                        [--Cp C]
                                        [--n-particles P]
                                        [--seed S]
"""

import sys
import os
import time
import random
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cpde_decmcts import CPDEDecMCTS, CPDEDecMCTSTeam


# ── POLICY DIAGNOSTICS ────────────────────────────────────────────────────────

def print_policy_tree(policy, label="", act_names=None, obs_names=None):
    """
    Print every entry in a ConditionalPolicy's internal dict, sorted by depth
    then obs history.  Shows: obs_history -> action, plus tree size.

    act_names : dict  {action_int: str}  e.g. {0:"OL", 1:"OR", 2:"LS"}
    obs_names : dict  {obs_int: str}     e.g. {0:"L", 1:"R"}
    """
    d = policy._policy

    def fmt_act(a):
        return act_names.get(a, str(a)) if act_names else str(a)

    def fmt_obs(o):
        return obs_names.get(o, str(o)) if obs_names else str(o)

    def fmt_hist(h):
        if not h:
            return "(root)"
        return " -> ".join(f"{fmt_act(a)}+{fmt_obs(o)}" for a, o in h)

    print(f"\n  {'─'*60}")
    print(f"  Policy tree{(' — ' + label) if label else ''}  "
          f"({len(d)} nodes)")
    print(f"  {'─'*60}")

    entries = sorted(d.items(), key=lambda kv: (len(kv[0]), kv[0]))
    for hist, action in entries:
        indent = "  " + "  " * len(hist)
        print(f"{indent}{fmt_hist(hist)}  =>  {fmt_act(action)}")
    print(f"  {'─'*60}")


def trace_episode(spec, policies, horizon, true_state, act_names=None, obs_names=None):
    """
    Re-run one episode with full per-step tracing.
    Returns total_reward.
    """
    model     = spec["model"]
    robot_ids = spec["robot_ids"]

    for pol in policies.values():
        pol.reset()

    def fmt_act(a):
        return act_names.get(a, str(a)) if act_names else str(a)

    def fmt_obs(o):
        return obs_names.get(o, str(o)) if obs_names else str(o)

    def fmt_hist(h):
        if not h:
            return "()"
        return "[" + ", ".join(f"{fmt_act(a)}+{fmt_obs(o)}" for a, o in h) + "]"

    state_names = getattr(spec.get("model"), "_state_names", None)
    def fmt_state(s):
        if state_names:
            return state_names.get(s, str(s))
        return f"s={s}"

    print(f"\n  {'─'*68}")
    print(f"  Step-by-step trace  (true_init={fmt_state(true_state)})")
    print(f"  {'─'*68}")
    print(f"  {'t':>2}  {'A0':>4} {'A1':>4}  {'O0':>3} {'O1':>3}  "
          f"{'R':>7}  {'s_next':>6}  "
          f"{'hist0 -> act0':>20}  {'hist1 -> act1':>20}")
    print(f"  {'─'*68}")

    total_reward = 0.0
    for t in range(horizon):
        hist0 = tuple(policies[0].history())
        hist1 = tuple(policies[1].history())

        in_tree0 = hist0 in policies[0]._policy
        in_tree1 = hist1 in policies[1]._policy

        a0 = policies[0].get_action()
        a1 = policies[1].get_action()
        joint_a = a0 + a1 * spec["act_per_agent"]

        r = model.reward(joint_a, true_state)
        total_reward += r

        joint_obs  = spec["sample_obs"](model, joint_a, true_state)
        true_state = model.sample_next_state(true_state, joint_a)

        o0 = joint_obs % spec["obs_per_agent"]
        o1 = joint_obs // spec["obs_per_agent"]

        tag0 = "" if in_tree0 else "*"   # * = fell back to default
        tag1 = "" if in_tree1 else "*"

        print(f"  {t:>2}  "
              f"{fmt_act(a0)+tag0:>4} {fmt_act(a1)+tag1:>4}  "
              f"{fmt_obs(o0):>3} {fmt_obs(o1):>3}  "
              f"{r:>7.1f}  {fmt_state(true_state):>6}  "
              f"{fmt_hist(hist0)+' -> '+fmt_act(a0)+tag0:>20}  "
              f"{fmt_hist(hist1)+' -> '+fmt_act(a1)+tag1:>20}")

        for rid in robot_ids:
            lo = o0 if rid == 0 else o1
            policies[rid].record([a0, a1][rid], lo)

    print(f"  {'─'*68}")
    print(f"  Total reward: {total_reward:.1f}"
          f"   (* = default action, tree miss)")
    return total_reward

import benchmarks.pomdp_tiger   as _tiger
import benchmarks.pomdp_medevac as _medevac
import benchmarks.pomdp_mars    as _mars


# ── BENCHMARK SPECS ────────────────────────────────────────────────────────────

def _tiger_spec(n_particles):
    model = _tiger.TigerModel()
    base  = n_particles // _tiger.N_STATES
    return dict(
        model          = model,
        init_particles = [s for s in range(_tiger.N_STATES)
                          for _ in range(base)],
        null_action    = _tiger.LISTEN,
        act_per_agent  = _tiger.ACT_PER_AGENT,
        obs_per_agent  = _tiger.OBS_PER_AGENT,
        sample_obs     = _tiger.sample_obs,
        robot_ids      = [0, 1],
    )

def _medevac_spec(n_particles):
    model  = _medevac.MedEvacModel()
    init_s = _medevac._state_id(
        _medevac.HELO_START[0], _medevac.HELO_START[1],
        _medevac.SHIP_START[0], _medevac.SHIP_START[1], 0
    )
    return dict(
        model          = model,
        init_particles = [init_s] * n_particles,
        null_action    = 0,
        act_per_agent  = _medevac.ACT_PER_AGENT,
        obs_per_agent  = _medevac.OBS_PER_AGENT,
        sample_obs     = _medevac.sample_obs,
        robot_ids      = [0, 1],
    )

def _mars_spec(n_particles):
    model = _mars.MarsModel()
    return dict(
        model          = model,
        init_particles = [0] * n_particles,
        null_action    = 0,
        act_per_agent  = _mars.ACT_PER_AGENT,
        obs_per_agent  = _mars.OBS_PER_AGENT,
        sample_obs     = _mars.sample_obs,
        robot_ids      = [0, 1],
    )

SPECS = {
    "tiger":   _tiger_spec,
    "medevac": _medevac_spec,
    "mars":    _mars_spec,
}

TIGER_REFS = {
    8:  dict(rs_maa=12.217, rs_sda=27.215, cen=47.717),
    9:  dict(rs_maa=15.572, rs_sda=30.905, cen=53.474),
    10: dict(rs_maa=15.184, rs_sda=34.724, cen=60.510),
}


# ── CORE OFFLINE CPDE TRIAL ───────────────────────────────────────────────────

def run_trial(spec, horizon, n_iters, Cp, rollout_depth, seed, trace=False,
              snap_iters=None, act_names=None, obs_names=None):
    """
    One offline CPDE trial.

    snap_iters : list[int] | None
        If provided, plan in increments and print the extracted policy tree
        after each cumulative iteration count.  Only runs for trial 0.

    Returns
    -------
    dict with keys:
        plan_time   : float   seconds spent in phase 1 (planning)
        exec_time   : float   seconds spent in phase 3 (execution)
        total_reward: float   cumulative reward over `horizon` steps
        actions     : list[tuple]  (a0, a1) at each step
        obs         : list[tuple]  (local_o0, local_o1) at each step
    """
    random.seed(seed)
    model     = spec["model"]
    robot_ids = spec["robot_ids"]

    # Sample true initial state from init_belief
    init_belief = model.init_belief
    r = random.random()
    cum = 0.0
    true_state = len(init_belief) - 1
    for s, p in enumerate(init_belief):
        cum += p
        if r <= cum:
            true_state = s
            break

    # ── PHASE 1: PLAN ONCE ────────────────────────────────────────────────────
    t0 = time.perf_counter()

    planners = {
        rid: CPDEDecMCTS(
            robot_id       = rid,
            init_particles = list(spec["init_particles"]),
            model          = model,
            max_steps      = horizon,
            Cp             = Cp,
            rollout_depth  = rollout_depth,
        )
        for rid in robot_ids
    }
    team = CPDEDecMCTSTeam(planners)

    if snap_iters:
        # Plan incrementally; print policy snapshot at each checkpoint
        snaps = sorted(set(snap_iters))
        done  = 0
        for snap in snaps:
            delta = snap - done
            if delta > 0:
                team.plan_joint(delta)
                done = snap
            policies_snap = team.extract_policies(
                default_action=spec["null_action"]
            )
            print(f"\n  ── Policy snapshot @ {snap} iters  (h={horizon}) ──")
            for rid in robot_ids:
                print_policy_tree(policies_snap[rid], label=f"Agent {rid}",
                                  act_names=act_names, obs_names=obs_names)
        # Continue to final count if needed
        if done < n_iters:
            team.plan_joint(n_iters - done)
    else:
        team.plan_joint(n_iters)

    plan_time = time.perf_counter() - t0

    # ── PHASE 2: EXTRACT CONDITIONAL POLICIES ─────────────────────────────────
    policies = team.extract_policies(default_action=spec["null_action"])
    for pol in policies.values():
        pol.reset()

    if trace:
        for rid in robot_ids:
            print_policy_tree(policies[rid], label=f"Agent {rid}",
                              act_names=act_names, obs_names=obs_names)
        trace_episode(spec, policies, horizon, true_state,
                      act_names=act_names, obs_names=obs_names)
        for pol in policies.values():
            pol.reset()

    # ── PHASE 3: EXECUTE PASSIVELY ────────────────────────────────────────────
    t0 = time.perf_counter()

    total_reward = 0.0
    step_actions = []
    step_obs     = []

    for _ in range(horizon):
        # Each agent consults its static ConditionalPolicy
        actions  = {rid: policies[rid].get_action() for rid in robot_ids}
        joint_a  = actions[0] + actions[1] * spec["act_per_agent"]

        total_reward += model.reward(joint_a, true_state)

        joint_obs  = spec["sample_obs"](model, joint_a, true_state)
        true_state = model.sample_next_state(true_state, joint_a)

        # Decode per-agent local observations
        local_obs = {
            0: joint_obs % spec["obs_per_agent"],
            1: joint_obs // spec["obs_per_agent"],
        }

        # Update each agent's obs history — no inter-agent communication
        for rid in robot_ids:
            policies[rid].record(actions[rid], local_obs[rid])

        step_actions.append((actions[0], actions[1]))
        step_obs.append((local_obs[0], local_obs[1]))

    exec_time = time.perf_counter() - t0

    return dict(
        plan_time    = plan_time,
        exec_time    = exec_time,
        total_reward = total_reward,
        actions      = step_actions,
        obs          = step_obs,
    )


# ── HORIZON RUNNER ────────────────────────────────────────────────────────────

def run_horizon(benchmark, spec, horizon, trials, n_iters, Cp,
                rollout_depth, seed, verbose=False,
                trace=False, trace_trial=0, snap_iters=None,
                act_names=None, obs_names=None):
    w = 72
    print(f"\n{'═'*w}")
    print(f"  CPDE Offline — {benchmark.upper()}  h={horizon}")
    print(f"{'═'*w}")

    refs = TIGER_REFS.get(horizon, {}) if benchmark == "tiger" else {}
    if refs:
        print(f"  Paper refs:  RS-MAA*={refs['rs_maa']:.3f}  "
              f"RS-SDA*={refs['rs_sda']:.3f}  Cen={refs['cen']:.3f}")

    print(f"  iters={n_iters}  rollout={rollout_depth}  "
          f"Cp={Cp:.4f}  trials={trials}")
    print()

    rewards    = []
    plan_times = []

    for t in range(trials):
        trial_seed = seed + t * 137
        result     = run_trial(spec, horizon, n_iters, Cp,
                               rollout_depth, trial_seed,
                               trace=(trace and t == trace_trial),
                               snap_iters=(snap_iters if t == 0 else None),
                               act_names=act_names, obs_names=obs_names)
        rewards.append(result["total_reward"])
        plan_times.append(result["plan_time"])

        if verbose:
            acts = result["actions"]
            act_names = {0: "OL", 1: "OR", 2: "LS"}
            seq = " ".join(
                f"({act_names.get(a0,'?')},{act_names.get(a1,'?')})"
                for a0, a1 in acts
            )
            print(f"  trial {t+1:>2}/{trials}  "
                  f"R={result['total_reward']:>8.1f}  "
                  f"plan={result['plan_time']:.2f}s  "
                  f"actions: {seq}")
        else:
            print(f"  trial {t+1:>2}/{trials}  "
                  f"R={result['total_reward']:>8.1f}  "
                  f"plan={result['plan_time']:.2f}s",
                  flush=True)

    def med(xs):
        s = sorted(xs)
        return s[len(s) // 2]

    def avg(xs):
        return sum(xs) / len(xs)

    print()
    print(f"  {'Median R':<12} {med(rewards):>10.3f}")
    print(f"  {'Mean R':<12} {avg(rewards):>10.3f}")
    print(f"  {'Min R':<12} {min(rewards):>10.3f}")
    print(f"  {'Max R':<12} {max(rewards):>10.3f}")
    print(f"  {'Avg plan t':<12} {avg(plan_times):>10.2f}s")

    if refs:
        lo, hi = refs["rs_maa"], refs["rs_sda"]
        m = med(rewards)
        print()
        if m < lo:
            print(f"  Gap: {lo - m:.3f} below RS-MAA* lower bound")
        elif m > hi:
            print(f"  Gap: {m - hi:.3f} above RS-SDA* (exceeds semi-dec UB)")
        else:
            pct = 100 * (m - lo) / (hi - lo)
            print(f"  Within target [{lo:.3f}, {hi:.3f}]  "
                  f"({pct:.1f}% from RS-MAA* to RS-SDA*)")

    return dict(rewards=rewards, plan_times=plan_times)


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    pa = argparse.ArgumentParser(
        description="Offline CPDE runner — plan once, execute passively"
    )
    pa.add_argument("--benchmark",   default="tiger",
                    choices=list(SPECS.keys()))
    pa.add_argument("--horizons",    nargs="+", type=int, default=[8, 9, 10])
    pa.add_argument("--trials",      type=int,   default=10)
    pa.add_argument("--iters",       type=int,   default=5000,
                    help="POMCP joint simulations per trial (default: 5000)")
    pa.add_argument("--rollout",     type=int,   default=20,
                    help="Rollout depth inside POMCP (default: 20)")
    pa.add_argument("--Cp",          type=float, default=1.0 / (2**0.5),
                    help="UCT exploration constant (default: 1/sqrt(2))")
    pa.add_argument("--n-particles", type=int,   default=500,
                    help="Initial belief particles (default: 500)")
    pa.add_argument("--seed",        type=int,   default=42)
    pa.add_argument("--verbose",     action="store_true",
                    help="Print per-step action sequences")
    pa.add_argument("--trace",       action="store_true",
                    help="Print policy tree + step-by-step trace for one trial")
    pa.add_argument("--trace-trial", type=int,   default=0,
                    help="Which trial (0-indexed) to trace (default: 0)")
    pa.add_argument("--snap-iters",  nargs="+",  type=int, default=None,
                    help="Print extracted policy tree at these cumulative "
                         "iteration counts during trial 0 (e.g. --snap-iters 100 500 2000)")
    args = pa.parse_args()

    w = 72
    print(f"\n{'═'*w}")
    print(f"  Offline CPDE — {args.benchmark.upper()} Dec-POMDP")
    print(f"{'═'*w}")
    print(f"  Benchmark   : {args.benchmark}")
    print(f"  Horizons    : {args.horizons}")
    print(f"  Trials      : {args.trials}")
    print(f"  POMCP iters : {args.iters}")
    print(f"  Rollout     : {args.rollout}")
    print(f"  Cp          : {args.Cp:.4f}")
    print(f"  n_particles : {args.n_particles}")
    print(f"  Seed        : {args.seed}")
    print()
    print("  Phase 1 (PLAN)    : CPDEDecMCTSTeam.plan_joint() — once per trial")
    print("  Phase 2 (EXTRACT) : extract_policies() — static ConditionalPolicy")
    print("  Phase 3 (EXECUTE) : follow policy passively, no replanning")

    spec = SPECS[args.benchmark](args.n_particles)

    # Tiger-specific human-readable names
    act_names = {0: "OL", 1: "OR", 2: "LS"} if args.benchmark == "tiger" else None
    obs_names = {0: "L",  1: "R"}            if args.benchmark == "tiger" else None

    all_results = {}
    for h in args.horizons:
        all_results[h] = run_horizon(
            benchmark    = args.benchmark,
            spec         = spec,
            horizon      = h,
            trials       = args.trials,
            n_iters      = args.iters,
            Cp           = args.Cp,
            rollout_depth = args.rollout,
            seed         = args.seed,
            verbose      = args.verbose,
            trace        = args.trace,
            trace_trial  = args.trace_trial,
            snap_iters   = args.snap_iters,
            act_names    = act_names,
            obs_names    = obs_names,
        )

    # Cross-horizon summary
    print(f"\n{'═'*w}")
    print(f"  CROSS-HORIZON SUMMARY")
    print(f"{'═'*w}")
    print(f"  {'h':>4}  {'Median':>10}  {'Mean':>10}  {'Min':>10}  {'Max':>10}")
    print(f"  {'─'*50}")

    def med(xs):
        s = sorted(xs)
        return s[len(s) // 2]

    for h in args.horizons:
        r = all_results[h]["rewards"]
        refs = TIGER_REFS.get(h, {}) if args.benchmark == "tiger" else {}
        ref_str = (f"  [MAA*={refs['rs_maa']:.1f}, SDA*={refs['rs_sda']:.1f}]"
                   if refs else "")
        print(f"  {h:>4}  "
              f"{med(r):>10.3f}  "
              f"{sum(r)/len(r):>10.3f}  "
              f"{min(r):>10.3f}  "
              f"{max(r):>10.3f}"
              f"{ref_str}")
    print()


if __name__ == "__main__":
    main()
