"""Paper-ready figures and tables.

Uses matplotlib (no seaborn). Each function takes already-computed records or
metric dicts and writes one figure or table file under `out_dir`.
"""

from __future__ import annotations
import os
import json
from collections import defaultdict


# Lazy matplotlib import so unit tests / pure-metric runs don't require it.
def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def ensure_dir(d: str) -> str:
    os.makedirs(d, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Tables (markdown + simple LaTeX)
# ---------------------------------------------------------------------------

def write_main_table(out_dir: str, gz_split: dict, km_split: dict, comparison: dict) -> None:
    ensure_dir(out_dir)
    lines = ["| System | Split | N | Solved | Solve rate |",
             "|---|---|---:|---:|---:|"]
    for split, rec in sorted(gz_split.items()):
        lines.append(f"| GammaZero | {split} | {rec['n']} | {rec['solved']} | {rec['rate']*100:.2f}% |")
    for split, rec in sorted(km_split.items()):
        lines.append(f"| Kimina (flat) | {split} | {rec['n']} | {rec['solved']} | {rec['rate']*100:.2f}% |")
    lines.append("")
    lines.append("**Per-problem head-to-head (aligned set):**")
    lines.append("")
    lines.append("| both | only GammaZero | only Kimina | neither | aligned |")
    lines.append("|---:|---:|---:|---:|---:|")
    lines.append(
        f"| {comparison['both']} | {comparison['only_gammazero']} | "
        f"{comparison['only_kimina']} | {comparison['neither']} | {comparison['n_aligned']} |"
    )
    with open(os.path.join(out_dir, "table_main.md"), "w") as f:
        f.write("\n".join(lines))


def write_hierarchy_table(out_dir: str, hb: dict, depth: dict) -> None:
    ensure_dir(out_dir)
    lines = [
        "**Hierarchical contribution (GammaZero only):**",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Problems with data | {hb['total_with_data']} |",
        f"| Solved | {hb['solved']} |",
        f"| Solved at root only | {hb['solved_root_only']} |",
        f"| Solved via skeleton expansion | {hb['solved_via_skeleton']} |",
        f"| Solved (other) | {hb['solved_other']} |",
        f"| Root-only rate | {hb['root_only_rate']*100:.2f}% |",
        f"| Skeleton marginal gain | +{hb['skeleton_marginal_gain']*100:.2f} pp |",
        "",
        "**Depth distribution (max depth reached):**",
        "",
        "| depth | all problems | solved |",
        "|---:|---:|---:|",
    ]
    keys = sorted(set(depth["hist_all"].keys()) | set(depth["hist_solved"].keys()))
    for k in keys:
        lines.append(f"| {k} | {depth['hist_all'].get(k,0)} | {depth['hist_solved'].get(k,0)} |")
    with open(os.path.join(out_dir, "table_hierarchy.md"), "w") as f:
        f.write("\n".join(lines))


def write_cost_table(out_dir: str, cost: dict) -> None:
    ensure_dir(out_dir)
    lines = ["**Search cost (GammaZero):**", "",
             "| metric | mean | median | std | max | sum |",
             "|---|---:|---:|---:|---:|---:|"]
    for k in ["n_nodes", "n_or", "n_and", "n_tactic_and", "n_skeleton_and",
              "used_total", "lean_verify_calls", "patch_verify_calls"]:
        c = cost[k]
        lines.append(f"| {k} | {c['mean']:.2f} | {c['median']:.2f} | {c['std']:.2f} | {c['max']} | {c['sum']} |")
    svf = cost["solved_vs_failed"]
    lines += ["",
              "**Solved vs failed problems:**", "",
              "| metric | solved (mean) | failed (mean) |",
              "|---|---:|---:|",
              f"| n_nodes | {svf['n_nodes_solved_mean']:.2f} | {svf['n_nodes_failed_mean']:.2f} |",
              f"| lean_verify_calls | {svf['lean_calls_solved_mean']:.2f} | {svf['lean_calls_failed_mean']:.2f} |"]
    with open(os.path.join(out_dir, "table_cost.md"), "w") as f:
        f.write("\n".join(lines))


def write_funnel_table(out_dir: str, funnel: dict) -> None:
    ensure_dir(out_dir)
    lines = ["**Skeleton pipeline funnel (sum across problems):**", "",
             "| stage | count |",
             "|---|---:|"]
    p = funnel["pipeline"]
    order = ["requested", "raw_verify_success", "raw_verify_failed",
             "patch_attempted", "patch_scored", "patch_failed",
             "feedback_generated", "inserted_raw", "selected_by_beam",
             "rejected_by_beam", "children_new", "children_duplicate"]
    for k in order:
        lines.append(f"| {k} | {p.get(k,0)} |")
    lines += ["", "**Commitment outcomes:**", "",
              "| stage | count |", "|---|---:|"]
    for k, v in funnel["commitment"].items():
        lines.append(f"| {k} | {v} |")
    if funnel["rates"]:
        lines += ["", "**Rates:**", "", "| rate | value |", "|---|---:|"]
        for k, v in funnel["rates"].items():
            lines.append(f"| {k} | {v*100:.2f}% |")
    with open(os.path.join(out_dir, "table_skeleton_funnel.md"), "w") as f:
        f.write("\n".join(lines))


def write_reward_table(out_dir: str, sep: dict, root_q: dict) -> None:
    ensure_dir(out_dir)
    lines = ["**Reward separability — AUROC for distinguishing SOLVED vs FAILED AND nodes:**", "",
             "| component | action | n_solved | n_failed | AUROC | Δ mean |",
             "|---|---|---:|---:|---:|---:|"]
    for key, v in sep.items():
        comp, action = key.split("|")
        lines.append(
            f"| {comp} | {action} | {v['n_pos']} | {v['n_neg']} | "
            f"{v['auroc']:.3f} | {v['delta_mean']:+.3f} |"
        )
    lines += ["", "**Per-problem root-Q vs eventual solve:**", "",
              "| metric | value |", "|---|---:|",
              f"| n problems | {root_q['n']} |",
              f"| AUROC (max-root-Q vs solve) | {root_q['auroc']:.3f} |",
              f"| mean root-Q (solved) | {root_q['mean_q_solved']:.3f} |",
              f"| mean root-Q (failed) | {root_q['mean_q_failed']:.3f} |"]
    with open(os.path.join(out_dir, "table_reward.md"), "w") as f:
        f.write("\n".join(lines))


def write_trajectory_table(out_dir: str, traj: dict) -> None:
    ensure_dir(out_dir)
    lines = ["**Trajectory dataset yield:**", "",
             "| metric | value |", "|---|---:|",
             f"| problems analyzed | {traj['n_problems']} |",
             f"| positive transitions (SOLVED AND) | {traj['total_pos_transitions']} |",
             f"| negative transitions (FAILED AND) | {traj['total_neg_transitions']} |",
             f"| reserved skeletons | {traj['total_reserved']} |",
             f"| positives per problem (mean) | {traj['pos_per_problem_mean']:.2f} |",
             f"| negatives per problem (mean) | {traj['neg_per_problem_mean']:.2f} |"]
    with open(os.path.join(out_dir, "table_trajectory.md"), "w") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_depth_hist(out_dir: str, depth: dict) -> None:
    plt = _mpl()
    ensure_dir(out_dir)
    keys = sorted(set(depth["hist_all"].keys()) | set(depth["hist_solved"].keys()))
    xs = list(keys)
    all_vals = [depth["hist_all"].get(k, 0) for k in keys]
    sol_vals = [depth["hist_solved"].get(k, 0) for k in keys]
    fig, ax = plt.subplots(figsize=(6.0, 3.5))
    w = 0.4
    ax.bar([x - w / 2 for x in xs], all_vals, width=w, label="all", color="#9ecae1")
    ax.bar([x + w / 2 for x in xs], sol_vals, width=w, label="solved", color="#3182bd")
    ax.set_xlabel("max depth reached")
    ax.set_ylabel("# problems")
    ax.set_title("Search depth distribution (GammaZero)")
    ax.set_xticks(xs)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_depth.pdf"))
    fig.savefig(os.path.join(out_dir, "fig_depth.png"), dpi=150)
    plt.close(fig)


def fig_reward_kde(out_dir: str, and_rows: list[dict]) -> None:
    plt = _mpl()
    ensure_dir(out_dir)
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.2), sharey=True)
    components = ["r_env", "r_dep", "Q"]
    for ax, comp in zip(axes, components):
        solved = [r[comp] for r in and_rows if r["status"] == "SOLVED" and r[comp] is not None]
        failed = [r[comp] for r in and_rows if r["status"] == "FAILED" and r[comp] is not None]
        if solved:
            ax.hist(solved, bins=30, alpha=0.55, label=f"SOLVED (n={len(solved)})", color="#2ca25f", density=True)
        if failed:
            ax.hist(failed, bins=30, alpha=0.55, label=f"FAILED (n={len(failed)})", color="#de2d26", density=True)
        ax.set_title(comp)
        ax.set_xlabel("value")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("density")
    fig.suptitle("Reward distributions by AND-node outcome")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_reward_dist.pdf"))
    fig.savefig(os.path.join(out_dir, "fig_reward_dist.png"), dpi=150)
    plt.close(fig)


def fig_funnel(out_dir: str, funnel: dict) -> None:
    plt = _mpl()
    ensure_dir(out_dir)
    p = funnel["pipeline"]
    sc = funnel["commitment"]
    stages = [
        ("requested", p.get("requested", 0)),
        ("raw_verify_success", p.get("raw_verify_success", 0)),
        ("patch_scored", p.get("patch_scored", 0)),
        ("selected_by_beam", p.get("selected_by_beam", 0)),
        ("committed", sc.get("committed", 0)),
        ("committed_solved", sc.get("committed_solved", 0)),
    ]
    labels = [s for s, _ in stages]
    vals = [v for _, v in stages]
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    ax.barh(labels[::-1], vals[::-1], color="#756bb1")
    ax.set_xlabel("count (sum across problems)")
    ax.set_title("Skeleton pipeline funnel")
    for i, v in enumerate(vals[::-1]):
        ax.text(v, i, f" {v}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_funnel.pdf"))
    fig.savefig(os.path.join(out_dir, "fig_funnel.png"), dpi=150)
    plt.close(fig)


def fig_system_compare(out_dir: str, gz_split: dict, km_split: dict) -> None:
    plt = _mpl()
    ensure_dir(out_dir)
    splits = sorted(set(gz_split.keys()) | set(km_split.keys()))
    gz_vals = [gz_split.get(s, {"rate": 0.0})["rate"] * 100 for s in splits]
    km_vals = [km_split.get(s, {"rate": 0.0})["rate"] * 100 for s in splits]
    x = list(range(len(splits)))
    w = 0.35
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.bar([i - w / 2 for i in x], gz_vals, w, label="GammaZero", color="#3182bd")
    ax.bar([i + w / 2 for i in x], km_vals, w, label="Kimina (flat)", color="#e6550d")
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.set_ylabel("solve rate (%)")
    ax.set_title("System solve-rate comparison")
    ax.legend()
    for i, v in enumerate(gz_vals):
        ax.text(i - w / 2, v + 0.5, f"{v:.1f}", ha="center", fontsize=8)
    for i, v in enumerate(km_vals):
        ax.text(i + w / 2, v + 0.5, f"{v:.1f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_system_compare.pdf"))
    fig.savefig(os.path.join(out_dir, "fig_system_compare.png"), dpi=150)
    plt.close(fig)


def fig_hierarchy_pie(out_dir: str, hb: dict) -> None:
    plt = _mpl()
    ensure_dir(out_dir)
    labels = ["root only", "via skeleton", "other"]
    sizes = [hb["solved_root_only"], hb["solved_via_skeleton"], hb["solved_other"]]
    if sum(sizes) == 0:
        return
    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    ax.pie(sizes, labels=[f"{l}\n({s})" for l, s in zip(labels, sizes)],
           colors=["#a1d99b", "#41ab5d", "#cccccc"],
           autopct="%1.1f%%", startangle=90)
    ax.set_title("How solved problems were solved (GammaZero)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_hierarchy.pdf"))
    fig.savefig(os.path.join(out_dir, "fig_hierarchy.png"), dpi=150)
    plt.close(fig)


def fig_cost_solved_failed(out_dir: str, gz_records: list[dict]) -> None:
    plt = _mpl()
    ensure_dir(out_dir)
    solved = [r["n_nodes"] for r in gz_records if r.get("has_data") and r.get("solved")]
    failed = [r["n_nodes"] for r in gz_records if r.get("has_data") and not r.get("solved")]
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    if solved:
        ax.hist(solved, bins=20, alpha=0.6, label=f"solved (n={len(solved)})", color="#2ca25f")
    if failed:
        ax.hist(failed, bins=20, alpha=0.6, label=f"failed (n={len(failed)})", color="#de2d26")
    ax.set_xlabel("# nodes expanded")
    ax.set_ylabel("# problems")
    ax.set_title("Search cost by outcome")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_cost.pdf"))
    fig.savefig(os.path.join(out_dir, "fig_cost.png"), dpi=150)
    plt.close(fig)


def dump_json(out_dir: str, name: str, obj) -> None:
    ensure_dir(out_dir)

    def default(o):
        if isinstance(o, set):
            return list(o)
        if isinstance(o, float) and (o != o):  # NaN
            return None
        return str(o)

    with open(os.path.join(out_dir, name), "w") as f:
        json.dump(obj, f, indent=2, default=default)
