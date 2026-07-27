"""
parse_results.py — Dec-MCTS result visualizer

Usage:
    python parse_results.py                        # load latest file in results/
    python parse_results.py results/foo.json       # load specific file
    python parse_results.py results/a.json results/b.json  # compare multiple runs
"""

import json
import sys
import os
import glob
import math
import argparse
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


# ── helpers ──────────────────────────────────────────────────────────────────

def load(path):
    with open(path) as f:
        return json.load(f)

def check_mark(val):
    return "✓" if val else "✗"

def pct_bar(pct, width=20):
    filled = round(pct / 100 * width)
    return "[" + "█" * filled + "░" * (width - filled) + f"] {pct:.1f}%"

def fmt_time(secs):
    m, s = divmod(int(secs), 60)
    return f"{m}m {s:02d}s" if m else f"{s:.1f}s"


# ── terminal summary ──────────────────────────────────────────────────────────

def print_summary(data, path):
    meta   = data["meta"]
    cfg    = data["config"]
    pc     = data["paper_comparison"]
    cc     = data["correctness_checks"]
    conv   = data["convergence"]
    timing = data["timing"]
    robots = data["per_robot_diagnostics"]

    SEP  = "═" * 72
    SEP2 = "─" * 72

    print(f"\n{SEP}")
    print(f"  Dec-MCTS Results  —  {meta['timestamp']}")
    print(f"  {path}")
    print(SEP)

    # Config
    print(f"\n  CONFIG")
    print(SEP2)
    print(f"  Robots: {cfg['N_ROBOTS']}   Budget: {cfg['BUDGET']}   "
          f"Vertices: {cfg['N_VERTICES']}   Regions: {cfg['N_REGIONS']}   "
          f"Obstacles: {cfg['N_OBSTACLES']}")
    print(f"  Outer iters: {cfg['N_OUTER']}   TAU: {cfg['TAU']}   "
          f"NUM_SEQ: {cfg['NUM_SEQ']}   GAMMA: {cfg['GAMMA']}   Seed: {cfg['SEED']}")

    # Paper comparison
    print(f"\n  PAPER COMPARISON  (Best et al. 2019)")
    print(SEP2)
    rows = [
        ("Dec % of max reward",  f"{pc['dec_pct_of_max']:.1f}%",     f"{pc['dec_pct_of_max_paper']:.1f}%"),
        ("Dec vs Cen median %",  f"{pc['dec_vs_cen_median_pct']:.2f}%", f"{pc['dec_vs_cen_median_pct_paper']:.1f}%"),
        ("Dec beats Cen frac",   f"{pc['dec_beats_cen_fraction']:.2f}", f"{pc['dec_beats_cen_fraction_paper']:.2f}"),
        ("Regions found (avg)",  f"{pc['dec_regions_found_avg']:.1f}/{pc['n_regions_total']}",
                                 f"{pc['dec_regions_found_paper']}/{pc['n_regions_total']}"),
        ("Reward Dec (median)",  f"{pc['dec_reward_median']:.0f}",    "—"),
        ("Reward Cen (median)",  f"{pc['cen_reward_median']:.0f}",    "—"),
    ]
    print(f"  {'Metric':<28} {'This run':>12}  {'Paper':>10}")
    print(f"  {'─'*28} {'─'*12}  {'─'*10}")
    for label, val, paper in rows:
        print(f"  {label:<28} {val:>12}  {paper:>10}")

    # Convergence
    print(f"\n  CONVERGENCE")
    print(SEP2)
    print(f"  First reward : {conv['reward_iter_1']:.0f}")
    print(f"  Final reward : {conv['reward_final']:.0f}   (max possible: {conv['total_reward_max']})")
    print(f"  % gain       : {conv['reward_pct_gain']:.1f}%")
    print(f"  1st-half avg : {conv['first_half_avg']:.1f}   2nd-half avg: {conv['second_half_avg']:.1f}")
    print(f"  Progress     : {pct_bar(pc['dec_pct_of_max'])}")

    # Correctness checks
    print(f"\n  CORRECTNESS CHECKS")
    print(SEP2)
    for k, v in cc.items():
        label = k.replace("_", " ").capitalize()
        print(f"  {check_mark(v)}  {label}")

    # Per-robot diagnostics
    print(f"\n  PER-ROBOT DIAGNOSTICS")
    print(SEP2)
    header = f"  {'Robot':>5}  {'Contribution':>13}  {'Entropy':>8}  {'Sharpness':>10}  {'SeqLen':>7}  {'TreeNodes':>10}"
    print(header)
    print(f"  {'─'*5}  {'─'*13}  {'─'*8}  {'─'*10}  {'─'*7}  {'─'*10}")
    total_contribution = sum(d["marginal_contribution"] for d in robots.values())
    for rid, d in sorted(robots.items(), key=lambda x: int(x[0])):
        pct = 100 * d["marginal_contribution"] / total_contribution if total_contribution else 0
        print(f"  {rid:>5}  {d['marginal_contribution']:>8.0f} ({pct:4.1f}%)  "
              f"{d['entropy_final']:>8.3f}  {d['sharpness']:>10.4f}  "
              f"{d['seq_len']:>7}  {d['tree_nodes']:>10}")

    # Timing
    print(f"\n  TIMING")
    print(SEP2)
    print(f"  Dec-MCTS : {fmt_time(timing['dec_elapsed_s'])}  "
          f"({timing['dec_elapsed_s']:.1f}s)")
    print(f"  Cen-MCTS : {fmt_time(timing['cen_elapsed_s'])}  "
          f"({timing['cen_elapsed_s']:.1f}s)")
    ratio = timing["dec_elapsed_s"] / timing["cen_elapsed_s"] if timing["cen_elapsed_s"] else float("inf")
    print(f"  Ratio    : {ratio:.1f}x slower (Dec vs Cen)")
    print()


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_single(data, path, save=None):
    conv   = data["convergence"]
    pc     = data["paper_comparison"]
    robots = data["per_robot_diagnostics"]
    cc     = data["correctness_checks"]
    timing = data["timing"]
    ts     = data["meta"]["timestamp"]

    curve = conv["reward_curve"]
    iters = list(range(1, len(curve) + 1))

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f"Dec-MCTS Results  —  {ts}", fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.38)

    # 1. Convergence curve
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(iters, curve, lw=1.8, color="#2196F3", label="Dec-MCTS reward")
    ax1.axhline(conv["total_reward_max"], ls="--", color="#F44336", lw=1.2, label="Max possible")
    ax1.axhline(pc["cen_reward_median"], ls=":", color="#FF9800", lw=1.2, label=f"Cen-MCTS median ({pc['cen_reward_median']:.0f})")
    midpoint = len(curve) // 2
    ax1.axvline(midpoint, ls="--", color="#9E9E9E", lw=0.8, alpha=0.6)
    ax1.fill_between(iters, curve, alpha=0.12, color="#2196F3")
    ax1.set_xlabel("Outer iteration")
    ax1.set_ylabel("Team reward")
    ax1.set_title("Convergence")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # 2. Per-robot contributions
    ax2 = fig.add_subplot(gs[0, 2])
    rids = sorted(robots.keys(), key=int)
    contribs = [robots[r]["marginal_contribution"] for r in rids]
    colors = plt.cm.tab10(np.linspace(0, 1, len(rids)))
    bars = ax2.barh(rids, contribs, color=colors)
    ax2.set_xlabel("Marginal contribution")
    ax2.set_title("Per-robot contribution")
    ax2.bar_label(bars, fmt="%.0f", padding=3, fontsize=8)
    ax2.grid(True, axis="x", alpha=0.3)

    # 3. Entropy & Sharpness scatter
    ax3 = fig.add_subplot(gs[1, 0])
    entropies  = [robots[r]["entropy_final"] for r in rids]
    sharpnesses = [robots[r]["sharpness"] for r in rids]
    sc = ax3.scatter(entropies, sharpnesses, c=np.arange(len(rids)), cmap="tab10", s=80, zorder=3)
    for i, rid in enumerate(rids):
        ax3.annotate(f"R{rid}", (entropies[i], sharpnesses[i]),
                     textcoords="offset points", xytext=(5, 3), fontsize=7)
    ax3.set_xlabel("Final entropy")
    ax3.set_ylabel("Sharpness")
    ax3.set_title("Entropy vs. Sharpness (per robot)")
    ax3.grid(True, alpha=0.3)

    # 4. Paper comparison bar chart
    ax4 = fig.add_subplot(gs[1, 1])
    metrics = ["Dec % of\nmax reward", "Dec beats\nCen frac×100"]
    ours   = [pc["dec_pct_of_max"],           pc["dec_beats_cen_fraction"] * 100]
    paper  = [pc["dec_pct_of_max_paper"],     pc["dec_beats_cen_fraction_paper"] * 100]
    x = np.arange(len(metrics))
    w = 0.35
    ax4.bar(x - w/2, ours,  w, label="This run", color="#2196F3")
    ax4.bar(x + w/2, paper, w, label="Paper",    color="#FF9800", alpha=0.8)
    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics, fontsize=9)
    ax4.set_ylabel("Value (%)")
    ax4.set_title("vs. Paper")
    ax4.legend(fontsize=8)
    ax4.grid(True, axis="y", alpha=0.3)
    ax4.set_ylim(0, 115)

    # 5. Summary text panel
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis("off")
    checks_text = "\n".join(
        f"{'✓' if v else '✗'}  {k.replace('_', ' ')}"
        for k, v in cc.items()
    )
    timing_text = (f"Dec: {fmt_time(timing['dec_elapsed_s'])}\n"
                   f"Cen: {fmt_time(timing['cen_elapsed_s'])}\n"
                   f"Ratio: {timing['dec_elapsed_s']/max(timing['cen_elapsed_s'],0.001):.1f}×")
    summary = (
        f"CORRECTNESS\n{checks_text}\n\n"
        f"TIMING\n{timing_text}\n\n"
        f"CONVERGENCE\n"
        f"Start → Final: {conv['reward_iter_1']:.0f} → {conv['reward_final']:.0f}\n"
        f"Gain: {conv['reward_pct_gain']:.1f}%\n"
        f"Regions found: {pc['dec_regions_found_avg']:.0f}/{pc['n_regions_total']}"
    )
    ax5.text(0.05, 0.97, summary, transform=ax5.transAxes,
             fontsize=8.5, va="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#F5F5F5", edgecolor="#BDBDBD"))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save:
        plt.savefig(save, dpi=150, bbox_inches="tight")
        print(f"  Plot saved to: {save}")
    else:
        plt.show()


def plot_comparison(datasets, paths, save=None):
    """Overlay convergence curves and key metrics for multiple runs."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Dec-MCTS Multi-Run Comparison", fontsize=13, fontweight="bold")

    colors = plt.cm.tab10(np.linspace(0, 1, len(datasets)))

    # Convergence overlay
    ax = axes[0]
    for i, (data, path) in enumerate(zip(datasets, paths)):
        label = os.path.basename(path).replace("decmcts_results_", "").replace(".json", "")
        curve = data["convergence"]["reward_curve"]
        ax.plot(range(1, len(curve)+1), curve, lw=1.6, color=colors[i], label=label)
    ax.set_xlabel("Outer iteration")
    ax.set_ylabel("Team reward")
    ax.set_title("Convergence curves")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Key metrics bar comparison
    ax2 = axes[1]
    labels = [os.path.basename(p).replace("decmcts_results_", "").replace(".json", "")
              for p in paths]
    metrics = {
        "Dec % of max": [d["paper_comparison"]["dec_pct_of_max"] for d in datasets],
        "Dec vs Cen %": [d["paper_comparison"]["dec_vs_cen_median_pct"] for d in datasets],
        "Regions found %": [
            100 * d["paper_comparison"]["dec_regions_found_avg"] / d["paper_comparison"]["n_regions_total"]
            for d in datasets
        ],
    }
    x = np.arange(len(labels))
    n_metrics = len(metrics)
    w = 0.8 / n_metrics
    for j, (mname, vals) in enumerate(metrics.items()):
        offset = (j - n_metrics/2 + 0.5) * w
        ax2.bar(x + offset, vals, w, label=mname)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax2.set_ylabel("Value (%)")
    ax2.set_title("Key metrics per run")
    ax2.legend(fontsize=8)
    ax2.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=150, bbox_inches="tight")
        print(f"  Comparison plot saved to: {save}")
    else:
        plt.show()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Visualize Dec-MCTS result JSON files")
    parser.add_argument("files", nargs="*", help="JSON result files (default: latest in results/)")
    parser.add_argument("--save", metavar="PATH", help="Save plot to file instead of showing")
    parser.add_argument("--no-plot", action="store_true", help="Print summary only, no plots")
    args = parser.parse_args()

    # Resolve file paths
    if args.files:
        paths = args.files
    else:
        candidates = sorted(glob.glob("results/decmcts_results_*.json"))
        if not candidates:
            print("No result files found in results/. Run decmcts_test.py first.")
            sys.exit(1)
        paths = [candidates[-1]]
        print(f"  Loading latest: {paths[0]}")

    for p in paths:
        if not os.path.exists(p):
            print(f"File not found: {p}")
            sys.exit(1)

    datasets = [load(p) for p in paths]

    # Print summaries
    for data, path in zip(datasets, paths):
        print_summary(data, path)

    if args.no_plot:
        return

    # Plot
    if len(datasets) == 1:
        save_path = args.save or None
        plot_single(datasets[0], paths[0], save=save_path)
    else:
        save_path = args.save or None
        plot_comparison(datasets, paths, save=save_path)


if __name__ == "__main__":
    main()
