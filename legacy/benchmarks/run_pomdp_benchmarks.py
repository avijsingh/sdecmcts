"""
run_pomdp_benchmarks.py
-----------------------
Evaluates online Dec-MCTS on Dec-POMDP benchmarks using Mahdi Al-Husseini's
exact T/O/R formulations (AAMAS 2026, RS-SDA* paper).

Benchmarks
----------
  tiger    — Multi-agent Tiger (MOD='C', TRIG_SEMI)
  medevac  — Maritime MedEvac (stochastic movement, carry flag)
  mars     — Mars Rover (loaded from mars.data, Amato et al. domain)

Reference values from RS-SDA* paper (Tiger, AAMAS 2026):
  h=8:  RS-MAA* 12.217   RS-SDA* 27.215   Cen 47.717
  h=9:  RS-MAA* 15.572   RS-SDA* 30.905   Cen 53.474
  h=10: RS-MAA* 15.184   RS-SDA* 34.724   Cen 60.510

Usage:
    python run_pomdp_benchmarks.py [--benchmark tiger|medevac|mars]
                                   [--horizons 8 9 10] [--trials K]
                                   [--online-outer N] [--seed S]
                                   [--n-particles P] [--n-mc M]
                                   [--skip-offline]
"""

import sys
import os
import time
import random
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from decmcts import DecMCTS, DecMCTSTeam
from cpde_decmcts import CPDEDecMCTS, CPDEDecMCTSTeam

# ── BENCHMARK IMPORTS ─────────────────────────────────────────────────────────

import benchmarks.pomdp_tiger as _tiger
import benchmarks.pomdp_medevac as _medevac
import benchmarks.pomdp_mars as _mars


# ── BENCHMARK REGISTRY ────────────────────────────────────────────────────────

def _make_tiger_spec(n_particles, n_mc):
    model = _tiger.TigerModel()
    base  = n_particles // _tiger.N_STATES
    init_particles = [s for s in range(_tiger.N_STATES) for _ in range(base)]

    def make_init_state(rid, particles, remaining):
        return _tiger.TigerBeliefState(list(particles), 0, remaining, model, rid)

    return dict(
        model            = model,
        init_particles   = init_particles,
        null_action      = _tiger.LISTEN,
        act_per_agent    = _tiger.ACT_PER_AGENT,
        sample_obs_fn    = _tiger.sample_obs,
        update_pf_fn     = _tiger.update_particles,
        make_obj_fn      = _tiger.make_objectives_from_particles,
        make_init_state  = make_init_state,
        n_mc             = n_mc,
        n_particles      = n_particles,
    )

def _make_medevac_spec(n_particles, n_mc):
    model = _medevac.MedEvacModel()
    init_s = _medevac._state_id(
        _medevac.HELO_START[0], _medevac.HELO_START[1],
        _medevac.SHIP_START[0], _medevac.SHIP_START[1], 0
    )
    init_particles = [init_s] * n_particles

    def make_init_state(rid, particles, remaining):
        return _medevac.MedEvacBeliefState(list(particles), 0, remaining, model, rid)

    return dict(
        model            = model,
        init_particles   = init_particles,
        null_action      = 0,   # WAIT
        act_per_agent    = _medevac.ACT_PER_AGENT,
        sample_obs_fn    = _medevac.sample_obs,
        update_pf_fn     = _medevac.update_particles,
        make_obj_fn      = _medevac.make_objectives_from_particles,
        make_init_state  = make_init_state,
        n_mc             = n_mc,
        n_particles      = n_particles,
    )

def _make_mars_spec(n_particles, n_mc):
    model          = _mars.MarsModel()
    init_particles = [0] * n_particles   # both rovers start at state 0

    def make_init_state(rid, particles, remaining):
        return _mars.MarsBeliefState(list(particles), 0, remaining, model, rid)

    return dict(
        model            = model,
        init_particles   = init_particles,
        null_action      = 0,   # action 0 (stay / first action)
        act_per_agent    = _mars.ACT_PER_AGENT,
        sample_obs_fn    = _mars.sample_obs,
        update_pf_fn     = _mars.update_particles,
        make_obj_fn      = _mars.make_objectives_from_particles,
        make_init_state  = make_init_state,
        n_mc             = n_mc,
        n_particles      = n_particles,
    )

BENCHMARK_SPECS = {
    "tiger":   _make_tiger_spec,
    "medevac": _make_medevac_spec,
    "mars":    _make_mars_spec,
}

# Tiger-only paper reference values (AAMAS 2026)
TIGER_PAPER_REFS = {
    8:  {"rs_maa_star": 12.217, "rs_sda_star": 27.215, "cen": 47.717},
    9:  {"rs_maa_star": 15.572, "rs_sda_star": 30.905, "cen": 53.474},
    10: {"rs_maa_star": 15.184, "rs_sda_star": 34.724, "cen": 60.510},
}


# ── DEFAULTS ──────────────────────────────────────────────────────────────────

DEFAULT_BENCHMARK    = "tiger"
DEFAULT_HORIZONS     = [8, 9, 10]
DEFAULT_TRIALS       = 5
DEFAULT_SEED         = 42
DEFAULT_ONLINE_OUTER = 50
DEFAULT_DEC_OUTER    = 80
DEFAULT_DEC_TAU      = 10
DEFAULT_NUM_SEQ      = 10
DEFAULT_NUM_SAMPLES  = 20
DEFAULT_N_PARTICLES  = 200
DEFAULT_N_MC         = 50
GATED_COMM_PERIODS   = [1, 5, -1]
DEFAULT_CPDE_ITERS   = 500   # POMCP simulations per agent at planning time
DEFAULT_CPDE_ROLLOUT = 20
DEFAULT_CPDE_CP      = 1.0 / (2 ** 0.5)


# ── FORMATTING ────────────────────────────────────────────────────────────────

def section(title):
    w = 72
    print(f"\n{'═'*w}")
    print(f"  {title}")
    print(f"{'═'*w}")

def row(label, *vals, width=36):
    vals_str = "  ".join(f"{v:>10}" for v in vals)
    print(f"  {label:<{width}}  {vals_str}")


# ── GENERIC ONLINE DEC-MCTS ───────────────────────────────────────────────────

def _make_planners(robot_ids, belief_states, local_obj_fns, tau,
                   num_seq, num_samples, seed):
    random.seed(seed)
    return {
        rid: DecMCTS(
            robot_id         = rid,
            robot_ids        = robot_ids,
            init_state       = belief_states[rid],
            local_utility_fn = local_obj_fns[rid],
            tau              = tau,
            num_seq          = num_seq,
            num_samples      = num_samples,
        )
        for rid in robot_ids
    }


def _local_pf_update(particles, action, local_obs, model, agent_id, n_resample=None):
    """
    Per-agent particle filter update using only the agent's local observation.

    Used in no-comm mode where agent i cannot see agent j's observation.
    Marginalises over the other agent's action (assumed uniform) and their
    unobserved local observation.

    Parameters
    ----------
    particles  : list[int]   current belief particles for this agent
    action     : int         this agent's executed action
    local_obs  : int         this agent's local observation (o_i, not joint)
    model      : model obj   with n_states, act_per_agent, obs_per_agent, n_obs, O
    agent_id   : int         0 or 1
    n_resample : int | None  target particle count (default = len(particles))
    """
    if n_resample is None:
        n_resample = len(particles)

    new_states = []
    weights    = []
    for s in particles:
        other   = random.randint(0, model.act_per_agent - 1)
        joint_a = (action + model.act_per_agent * other if agent_id == 0
                   else other + model.act_per_agent * action)
        s_next  = model.sample_next_state(s, joint_a)

        # Marginal likelihood: P(local_obs | s_next, joint_a)
        #   = sum_{other_obs} P(joint_obs | s_next, joint_a)
        w = 0.0
        for other_obs in range(model.obs_per_agent):
            joint_o = (local_obs + model.obs_per_agent * other_obs if agent_id == 0
                       else other_obs + model.obs_per_agent * local_obs)
            w += model.O[joint_a * model.n_states * model.n_obs
                         + s_next * model.n_obs + joint_o]
        new_states.append(s_next)
        weights.append(w)

    total = sum(weights)
    if total <= 1e-12:
        return new_states[:n_resample]

    probs = [w / total for w in weights]

    # Systematic resampling
    resampled = []
    r   = random.random() / n_resample
    cum = 0.0
    idx = 0
    for j in range(n_resample):
        target = r + j / n_resample
        while cum < target and idx < len(probs):
            cum += probs[idx]
            idx += 1
        resampled.append(new_states[max(idx - 1, 0)])
    return resampled


def run_decmcts_online(spec, horizon, n_outer_per_step, tau,
                       num_seq, num_samples, comm_period, seed):
    """
    Online (receding-horizon) Dec-MCTS for any Dec-POMDP benchmark.

    Each agent maintains its own particle filter:
      - free-comm (comm_period >= 1): agents share observations at each step,
        so both PFs are updated with the joint observation — beliefs stay in sync.
      - no-comm (comm_period == -1): each agent updates its PF with only its own
        local observation, marginalising over the other agent's unseen obs.

    At each of the `horizon` decision steps:
      1. Re-plan from each agent's current belief.
      2. Execute the first action from each agent's best sequence.
      3. Sample the true joint observation from O(·|true_state, joint_a).
      4. Update each agent's particle filter (joint or local obs per comm mode).
      5. Transition true state through T.

    `spec` is a benchmark specification dict produced by _make_*_spec().
    comm_period=-1 disables inter-agent communication during planning.
    """
    random.seed(seed)
    robot_ids = [0, 1]
    model     = spec["model"]

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

    # Per-agent particle filters — each agent starts from the same prior
    particles = {rid: list(spec["init_particles"]) for rid in robot_ids}
    total_reward = 0.0

    for step in range(horizon):
        remaining = horizon - step

        # Build per-agent objectives from each agent's own belief
        local_obj_fns = {}
        for rid in robot_ids:
            _, fns = spec["make_obj_fn"](particles[rid], model, spec["n_mc"])
            local_obj_fns[rid] = fns[rid]

        belief_states = {
            rid: spec["make_init_state"](rid, particles[rid], remaining)
            for rid in robot_ids
        }

        planners = _make_planners(
            robot_ids, belief_states, local_obj_fns,
            tau, num_seq, num_samples, seed=seed + step * 7
        )

        for i in range(1, n_outer_per_step + 1):
            for p in planners.values():
                p.iterate(1)
            if comm_period > 0 and i % comm_period == 0:
                for rid, planner in planners.items():
                    _, q = planner.get_distribution()
                    for other_id, other_planner in planners.items():
                        if other_id != rid:
                            other_planner.receive_dist_dict(rid, q)

        # Extract first action; fall back to null_action if sequence is empty
        actions = {}
        null = spec["null_action"]
        for rid in robot_ids:
            seq = planners[rid].best_action_sequence()
            actions[rid] = seq[0] if seq else null

        joint_a = actions[0] + actions[1] * spec["act_per_agent"]

        total_reward += model.reward(joint_a, true_state)

        obs        = spec["sample_obs_fn"](model, joint_a, true_state)
        true_state = model.sample_next_state(true_state, joint_a)

        # Update each agent's particle filter
        if comm_period >= 0:
            # Free-comm: agents share observations — update both with joint obs
            updated = spec["update_pf_fn"](
                particles[0], joint_a, obs, model, n_resample=spec["n_particles"]
            )
            particles[0] = updated
            particles[1] = list(updated)
        else:
            # No-comm: each agent updates with its own local observation only
            local_obs = {
                0: obs % model.obs_per_agent,
                1: obs // model.obs_per_agent,
            }
            for rid in robot_ids:
                particles[rid] = _local_pf_update(
                    particles[rid], actions[rid], local_obs[rid],
                    model, rid, n_resample=spec["n_particles"]
                )

    return total_reward


# ── CPDE RUNNER ───────────────────────────────────────────────────────────────

def run_cpde_decmcts(spec, horizon, n_iterations, Cp, rollout_depth, seed):
    """
    CPDE: plan once from initial belief, execute via conditional policy.

    Planning: each agent builds a POMCP obs-branching tree from the initial
    particles.  The other agent's action is sampled uniformly (no prior).
    Execution: each agent looks up its action from its ConditionalPolicy
    using its accumulated local obs history — no inter-agent communication.

    Parameters
    ----------
    spec          : benchmark spec dict (from _make_*_spec)
    horizon       : int   planning / execution horizon
    n_iterations  : int   total POMCP simulations per agent at planning time
    Cp            : float UCT exploration constant
    rollout_depth : int   random-rollout depth inside POMCP
    seed          : int   random seed

    Returns
    -------
    float  total reward accumulated over `horizon` steps
    """
    random.seed(seed)
    robot_ids = [0, 1]
    model     = spec["model"]

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

    # Build one POMCP planner per agent and run planning
    planners = {
        rid: CPDEDecMCTS(
            robot_id      = rid,
            init_particles = list(spec["init_particles"]),
            model         = model,
            max_steps     = horizon,
            Cp            = Cp,
            rollout_depth = rollout_depth,
        )
        for rid in robot_ids
    }
    team = CPDEDecMCTSTeam(planners)
    team.plan_joint(n_iterations)   # joint simulation → coordinated policies

    # Extract conditional policies (obs-history → action)
    null = spec["null_action"]
    policies = team.extract_policies(default_action=null)
    for pol in policies.values():
        pol.reset()

    # Execute: agents follow their own policy using local obs only
    total_reward = 0.0
    for step in range(horizon):
        actions = {rid: policies[rid].get_action() for rid in robot_ids}
        joint_a = actions[0] + actions[1] * spec["act_per_agent"]

        total_reward += model.reward(joint_a, true_state)

        joint_obs  = spec["sample_obs_fn"](model, joint_a, true_state)
        true_state = model.sample_next_state(true_state, joint_a)

        # Decode local obs per agent and update policy history
        for rid in robot_ids:
            if rid == 0:
                local_obs = joint_obs % model.obs_per_agent
            else:
                local_obs = joint_obs // model.obs_per_agent
            policies[rid].record(actions[rid], local_obs)

    return total_reward


# ── SINGLE HORIZON RUN ────────────────────────────────────────────────────────

def run_horizon(benchmark_name, spec, horizon, trials, n_outer_per_step,
                dec_tau, num_seq, num_samples, seed,
                cpde_iters=None, cpde_Cp=DEFAULT_CPDE_CP,
                cpde_rollout=DEFAULT_CPDE_ROLLOUT):
    section(f"{benchmark_name.upper()} POMDP  h={horizon}")

    refs = TIGER_PAPER_REFS.get(horizon, {}) if benchmark_name == "tiger" else {}
    if refs:
        print(f"  Paper refs (AAMAS 2026):  "
              f"RS-MAA*={refs['rs_maa_star']:.3f}  "
              f"RS-SDA*={refs['rs_sda_star']:.3f}  "
              f"Cen={refs['cen']:.3f}")
        print(f"  Target range for Dec-MCTS: "
              f"[{refs['rs_maa_star']:.3f}, {refs['rs_sda_star']:.3f}]")
    print(f"\n  Trials={trials}  online_outer={n_outer_per_step}  tau={dec_tau}  "
          f"n_particles={spec['n_particles']}  n_mc={spec['n_mc']}")
    run_cpde = cpde_iters is not None and cpde_iters > 0
    if run_cpde:
        print(f"  CPDE iters={cpde_iters}  cpde_rollout={cpde_rollout}")
    print()

    online_free_rewards,   online_free_times   = [], []
    online_nocomm_rewards, online_nocomm_times = [], []
    cpde_rewards,          cpde_times          = [], []

    print(f"  Running {trials} trials ...", flush=True)

    for t in range(trials):
        trial_seed = seed + t * 137

        t0 = time.perf_counter()
        r = run_decmcts_online(spec, horizon, n_outer_per_step,
                               dec_tau, num_seq, num_samples,
                               comm_period=1, seed=trial_seed)
        online_free_rewards.append(r)
        online_free_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        r = run_decmcts_online(spec, horizon, n_outer_per_step,
                               dec_tau, num_seq, num_samples,
                               comm_period=-1, seed=trial_seed)
        online_nocomm_rewards.append(r)
        online_nocomm_times.append(time.perf_counter() - t0)

        cpde_str = ""
        if run_cpde:
            t0 = time.perf_counter()
            r = run_cpde_decmcts(spec, horizon, cpde_iters,
                                 cpde_Cp, cpde_rollout, seed=trial_seed)
            cpde_rewards.append(r)
            cpde_times.append(time.perf_counter() - t0)
            cpde_str = f"  cpde={cpde_rewards[-1]:>8.2f}"

        print(f"  trial {t+1:>3}/{trials}  "
              f"online_free={online_free_rewards[-1]:>8.2f}  "
              f"online_nocomm={online_nocomm_rewards[-1]:>8.2f}"
              f"{cpde_str}",
              flush=True)

    def med(xs):
        s = sorted(xs)
        return s[len(s) // 2]

    def avg(xs):
        return sum(xs) / len(xs) if xs else 0.0

    row("Planner", "Median R", "Avg R", "Avg t(s)")
    print("  " + "─" * 70)

    print(f"  [ONLINE — replanning every step, {n_outer_per_step} iters/step, belief updated]")
    row("  DecMCTS online (free comm)",
        f"{med(online_free_rewards):.3f}",
        f"{avg(online_free_rewards):.3f}",
        f"{avg(online_free_times):.2f}s")
    row("  DecMCTS online (no comm)",
        f"{med(online_nocomm_rewards):.3f}",
        f"{avg(online_nocomm_rewards):.3f}",
        f"{avg(online_nocomm_times):.2f}s")

    if run_cpde:
        print(f"  [CPDE — plan once ({cpde_iters} POMCP iters/agent), "
              f"decentralized execution]")
        row("  CPDE DecMCTS",
            f"{med(cpde_rewards):.3f}",
            f"{avg(cpde_rewards):.3f}",
            f"{avg(cpde_times):.2f}s")

    if refs:
        print("  " + "─" * 70)
        row("── Paper refs ──", "", "", "")
        row("RS-MAA* (lower bound)",   f"{refs['rs_maa_star']:.3f}", "", "")
        row("RS-SDA* (semi-dec UB)",   f"{refs['rs_sda_star']:.3f}", "", "")
        row("Centralized (paper)",     f"{refs['cen']:.3f}",         "", "")

        online_med = med(online_free_rewards)
        lo, hi = refs["rs_maa_star"], refs["rs_sda_star"]
        print(f"\n  Gap analysis (online free-comm median = {online_med:.3f}):")
        if online_med < lo:
            print(f"    BELOW RS-MAA* by {lo - online_med:.3f}  "
                  f"(try more iters/step)")
        elif online_med > hi:
            print(f"    ABOVE RS-SDA* by {online_med - hi:.3f}  "
                  f"(exceeds semi-dec UB — MC noise or lucky trial)")
        else:
            pct = 100 * (online_med - lo) / (hi - lo) if hi != lo else 0
            print(f"    Within target range  ({pct:.1f}% of the way from "
                  f"RS-MAA* to RS-SDA*)")

    result = {
        "online":        {"rewards": online_free_rewards,   "times": online_free_times},
        "online_nocomm": {"rewards": online_nocomm_rewards, "times": online_nocomm_times},
    }
    if run_cpde:
        result["cpde"] = {"rewards": cpde_rewards, "times": cpde_times}
    return result


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Online Dec-MCTS on Dec-POMDP benchmarks (Mahdi's T/O/R formulations)"
    )
    parser.add_argument("--benchmark",    type=str, default=DEFAULT_BENCHMARK,
                        choices=list(BENCHMARK_SPECS.keys()),
                        help="Which benchmark to run (default: tiger)")
    parser.add_argument("--horizons",     nargs="+", type=int,
                        default=DEFAULT_HORIZONS)
    parser.add_argument("--trials",       type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--online-outer", type=int, default=DEFAULT_ONLINE_OUTER,
                        help="Dec-MCTS iters per step in online mode")
    parser.add_argument("--dec-tau",      type=int, default=DEFAULT_DEC_TAU)
    parser.add_argument("--num-seq",      type=int, default=DEFAULT_NUM_SEQ)
    parser.add_argument("--num-samples",  type=int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument("--n-particles",  type=int, default=DEFAULT_N_PARTICLES)
    parser.add_argument("--n-mc",         type=int, default=DEFAULT_N_MC)
    parser.add_argument("--seed",         type=int, default=DEFAULT_SEED)
    parser.add_argument("--cpde-iters",   type=int, default=0,
                        help="POMCP simulations per agent for CPDE planning "
                             "(0 = skip CPDE, default: 0)")
    parser.add_argument("--cpde-rollout", type=int, default=DEFAULT_CPDE_ROLLOUT,
                        help=f"CPDE rollout depth (default: {DEFAULT_CPDE_ROLLOUT})")
    parser.add_argument("--cpde-cp",      type=float, default=DEFAULT_CPDE_CP,
                        help="CPDE UCT exploration constant")
    parser.add_argument("--output",       type=str, default=None,
                        help="Save console output to this file (in addition to stdout)")
    args = parser.parse_args()

    if args.output:
        class _Tee:
            def __init__(self, *streams): self.streams = streams
            def write(self, s):
                for st in self.streams: st.write(s)
            def flush(self):
                for st in self.streams: st.flush()
        _f = open(args.output, "w")
        sys.stdout = _Tee(sys.__stdout__, _f)

    section(f"Online Dec-MCTS — {args.benchmark.upper()} Dec-POMDP")
    print(f"  Benchmark   : {args.benchmark}")
    print(f"  Horizons    : {args.horizons}")
    print(f"  Trials      : {args.trials}")
    print(f"  Online iters/step: {args.online_outer}")
    print(f"  tau         : {args.dec_tau}")
    print(f"  n_particles : {args.n_particles}   n_mc: {args.n_mc}")
    print(f"  Seed        : {args.seed}")
    if args.cpde_iters > 0:
        print(f"  CPDE iters  : {args.cpde_iters}  rollout: {args.cpde_rollout}  Cp: {args.cpde_cp:.4f}")
    print()
    print("  Belief states use a particle filter; reward estimated by MC sampling.")
    if args.benchmark == "tiger":
        print("  Dec-MCTS should land BETWEEN RS-MAA* and RS-SDA* reference values.")
    if args.cpde_iters > 0:
        print("  CPDE: plan once from initial belief, execute via obs-conditional policy.")

    print(f"\n  Building {args.benchmark} model...", flush=True)
    spec = BENCHMARK_SPECS[args.benchmark](args.n_particles, args.n_mc)
    print(f"  Model ready.")

    all_results = {}
    for h in args.horizons:
        all_results[h] = run_horizon(
            benchmark_name   = args.benchmark,
            spec             = spec,
            horizon          = h,
            trials           = args.trials,
            n_outer_per_step = args.online_outer,
            dec_tau          = args.dec_tau,
            num_seq          = args.num_seq,
            num_samples      = args.num_samples,
            seed             = args.seed,
            cpde_iters       = args.cpde_iters if args.cpde_iters > 0 else None,
            cpde_Cp          = args.cpde_cp,
            cpde_rollout     = args.cpde_rollout,
        )

    # Cross-horizon summary
    section("CROSS-HORIZON SUMMARY")
    print()
    run_cpde = args.cpde_iters > 0
    hdr = ("h", "Online-free", "Online-nocomm") + (("CPDE",) if run_cpde else ())
    row(*hdr, width=4)
    print("  " + "─" * 50)

    def med(xs):
        s = sorted(xs)
        return s[len(s) // 2]

    for h in args.horizons:
        res  = all_results[h]
        r    = TIGER_PAPER_REFS.get(h, {})
        lo   = f"{r.get('rs_maa_star', '?'):.3f}" if r else "?"
        hi   = f"{r.get('rs_sda_star', '?'):.3f}" if r else "?"
        extra = f"  [{lo}, {hi}]" if r else ""
        vals  = [str(h),
                 f"{med(res['online']['rewards']):.3f}",
                 f"{med(res['online_nocomm']['rewards']):.3f}"]
        if run_cpde and "cpde" in res:
            vals.append(f"{med(res['cpde']['rewards']):.3f}")
        row(*vals, width=4)
        if extra:
            print(f"          paper range: {extra}")

    print()


if __name__ == "__main__":
    main()
