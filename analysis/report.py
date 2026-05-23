"""Top-level driver: load both data sources, compute every metric, dump tables/figures.

Usage:
    python -m analysis.report                # use defaults
    python -m analysis.report --out analysis/out --skip-figures

Designed to be re-run on a growing subset without code changes.
"""

from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

# allow `python analysis/report.py` from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis import loaders, metrics, figures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollout-root", default=loaders.ROLLOUT_ROOT)
    ap.add_argument("--flat-root", default=loaders.FLAT_ROOT)
    ap.add_argument("--out", default="analysis/out")
    ap.add_argument("--skip-figures", action="store_true")
    args = ap.parse_args()

    out_dir = args.out
    figures.ensure_dir(out_dir)

    # ---- load -----------------------------------------------------------
    print(f"[load] rollouts from {args.rollout_root}")
    gz = loaders.load_rollouts(args.rollout_root, keep_raw=True)
    print(f"       loaded {len(gz)} rollout files ({sum(1 for r in gz if r['has_data'])} with data)")
    print(f"[load] kimina flat from {args.flat_root}")
    km = loaders.load_kimina(args.flat_root)
    print(f"       loaded {len(km)} kimina problem dirs ({sum(1 for r in km if r['has_data'])} with data)")

    joined = loaders.align_by_name(gz, km)

    # ---- per-source metrics --------------------------------------------
    gz_overall = metrics.solve_rate(gz)
    gz_split = metrics.solve_rate_by_split(gz)
    km_overall = metrics.solve_rate(km)
    km_split = metrics.solve_rate_by_split(km)
    comparison = metrics.system_comparison(joined)
    km_agg = metrics.kimina_aggregates(km)

    # GammaZero hierarchy / cost / funnel
    hb = metrics.hierarchical_breakdown(gz)
    depth = metrics.depth_of_solution(gz)
    cost = metrics.search_cost(gz)
    funnel = metrics.skeleton_funnel(gz)
    traj = metrics.trajectory_stats(gz)

    # Reward signal
    and_rows = metrics.and_node_metrics(gz)
    sep = metrics.reward_separability(and_rows)
    root_q = metrics.root_q_vs_outcome(gz)
    rdist = metrics.reward_distribution(and_rows)

    # ---- tables ---------------------------------------------------------
    figures.write_main_table(out_dir, gz_split, km_split, comparison)
    figures.write_hierarchy_table(out_dir, hb, depth)
    figures.write_cost_table(out_dir, cost)
    figures.write_funnel_table(out_dir, funnel)
    figures.write_reward_table(out_dir, sep, root_q)
    figures.write_trajectory_table(out_dir, traj)

    # ---- figures --------------------------------------------------------
    if not args.skip_figures:
        try:
            figures.fig_system_compare(out_dir, gz_split, km_split)
            figures.fig_depth_hist(out_dir, depth)
            figures.fig_hierarchy_pie(out_dir, hb)
            figures.fig_reward_kde(out_dir, and_rows)
            figures.fig_funnel(out_dir, funnel)
            figures.fig_cost_solved_failed(out_dir, gz)
        except Exception as e:
            print(f"[warn] figure generation failed: {e}")

    # ---- structured dump for downstream use ----------------------------
    figures.dump_json(out_dir, "summary.json", {
        "n_gammazero_files": len(gz),
        "n_gammazero_with_data": gz_overall["n"],
        "n_kimina_dirs": len(km),
        "n_kimina_with_data": km_overall["n"],
        "gammazero_overall": gz_overall,
        "gammazero_by_split": gz_split,
        "kimina_overall": km_overall,
        "kimina_by_split": km_split,
        "comparison": comparison,
        "hierarchy": hb,
        "depth": depth,
        "cost": cost,
        "skeleton_funnel": funnel,
        "trajectory": {k: v for k, v in traj.items() if k != "per_problem"},
        "reward_separability": sep,
        "root_q_vs_outcome": root_q,
        "reward_distributions": rdist,
        "kimina_aggregates": km_agg,
    })

    # ---- top-level summary printout ------------------------------------
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"GammaZero:   {gz_overall['solved']}/{gz_overall['n']} solved ({gz_overall['rate']*100:.2f}%)")
    for k, v in sorted(gz_split.items()):
        print(f"   split={k:6s}  {v['solved']}/{v['n']}  ({v['rate']*100:.2f}%)")
    print(f"Kimina:      {km_overall['solved']}/{km_overall['n']} solved ({km_overall['rate']*100:.2f}%)")
    for k, v in sorted(km_split.items()):
        print(f"   split={k:6s}  {v['solved']}/{v['n']}  ({v['rate']*100:.2f}%)")
    print()
    print(f"Hierarchy:   root-only={hb['solved_root_only']}  "
          f"via-skel={hb['solved_via_skeleton']}  other={hb['solved_other']}")
    print(f"Aligned:     both={comparison['both']}  "
          f"only-gz={comparison['only_gammazero']}  "
          f"only-km={comparison['only_kimina']}  "
          f"neither={comparison['neither']}  "
          f"(N={comparison['n_aligned']})")
    print()
    print(f"AND nodes:   {len(and_rows)} rows  "
          f"({sum(1 for r in and_rows if r['status']=='SOLVED')} SOLVED, "
          f"{sum(1 for r in and_rows if r['status']=='FAILED')} FAILED)")
    if "Q|all" in sep:
        s = sep["Q|all"]
        print(f"   Q AUROC (SOLVED vs FAILED): {s['auroc']:.3f}  Δmean={s['delta_mean']:+.3f}")
    if root_q["n"]:
        print(f"   root-Q→solve AUROC: {root_q['auroc']:.3f}  "
              f"(mean Q solved={root_q['mean_q_solved']:.3f}, failed={root_q['mean_q_failed']:.3f})")
    print()
    print(f"Outputs in: {out_dir}/")


if __name__ == "__main__":
    main()
