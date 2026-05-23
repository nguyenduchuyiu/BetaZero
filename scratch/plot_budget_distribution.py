import os
import json
import glob
import matplotlib.pyplot as plt
import numpy as np

def generate_plot():
    test_dir = "/workspace/npthai/BetaZero/outputs/rollouts/gemini3flash/miniF2F-test"
    json_files = glob.glob(os.path.join(test_dir, "*.json"))
    
    data_points = []
    
    for path in json_files:
        with open(path, "r") as f:
            try:
                data = json.load(f)
            except Exception:
                continue
                
        root_id = data.get("root_id", "state_0")
        nodes = data.get("nodes", [])
        
        # Check solve status
        root_solved = False
        for n in nodes:
            if n["id"] == root_id:
                root_solved = (n["status"] == "SOLVED")
                break
                
        # Get budget used
        meta = data.get("search_metadata", {})
        budget = meta.get("budget", {})
        used_total = budget.get("used_total", 0)
        
        data_points.append({
            "name": os.path.basename(path).replace(".json", ""),
            "solved": root_solved,
            "budget_used": used_total
        })
        
    # Sort data points by budget_used (ascending)
    data_points.sort(key=lambda x: (x["budget_used"], not x["solved"]))
    
    # Extract values for plotting
    budgets = [x["budget_used"] for x in data_points]
    colors = ["#10b981" if x["solved"] else "#f43f5e" for x in data_points]  # Emerald green vs Crimson rose
    
    # Create the plot with high resolution and premium layout
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial"]
    
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    
    # Plot as a bar chart
    x_indices = np.arange(len(budgets))
    bars = ax.bar(x_indices, budgets, color=colors, width=0.8, edgecolor="none", alpha=0.9)
    
    # Highlight the budget limit
    ax.axhline(y=512, color="#94a3b8", linestyle="--", linewidth=1.2, label="Max Node Budget (512)")
    
    # Customizing axes and grids
    ax.set_title("GammaZero Budget Usage Distribution across miniF2F-test (N=244)", fontsize=14, fontweight="bold", pad=15, color="#1e293b")
    ax.set_xlabel("Theorems (Sorted by Node Usage)", fontsize=11, labelpad=10, color="#475569")
    ax.set_ylabel("Expanded Action Nodes (AND Nodes)", fontsize=11, labelpad=10, color="#475569")
    
    ax.set_xlim(-1, len(budgets))
    ax.set_ylim(0, 550)
    
    ax.grid(axis="y", linestyle=":", alpha=0.6, color="#cbd5e1")
    
    # Remove top and right spines
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#94a3b8")
    ax.spines["bottom"].set_color("#94a3b8")
    
    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#10b981", label="SOLVED (Early Termination)"),
        Patch(facecolor="#f43f5e", label="FAILED (Budget Exhausted / Failed)"),
        plt.Line2D([0], [0], color="#94a3b8", linestyle="--", linewidth=1.2, label="Max Search Budget (512)")
    ]
    ax.legend(handles=legend_elements, loc="upper left", frameon=True, facecolor="white", edgecolor="#e2e8f0", fontsize=10)
    
    # Annotate summary stats on the plot
    total_solved = sum(1 for x in data_points if x["solved"])
    success_rate = (total_solved / len(data_points)) * 100
    avg_solved = np.mean([x["budget_used"] for x in data_points if x["solved"]])
    avg_failed = np.mean([x["budget_used"] for x in data_points if not x["solved"]])
    
    stats_text = (
        f"Total Problems: {len(data_points)}\n"
        f"Solved: {total_solved} ({success_rate:.1f}%)\n"
        f"Avg Solved Nodes: {avg_solved:.1f}\n"
        f"Avg Failed Nodes: {avg_failed:.1f}"
    )
    
    ax.text(
        0.97, 0.25, stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8fafc", edgecolor="#e2e8f0", alpha=0.9)
    )
    
    plt.tight_layout()
    
    # Save paths
    paths = [
        "/workspace/npthai/BetaZero/scratch/budget_usage_distribution.png",
        "/workspace/npthai/BetaZero/budget_usage_distribution.png",
        "/home/npthai/.gemini/antigravity/brain/5a1fc07e-dc22-4f49-bac2-0c1f69b4f581/budget_usage_distribution.png"
    ]
    
    # Ensure directories exist
    for p in paths:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        plt.savefig(p, bbox_inches="tight", dpi=300)
        
    print(f"Plot successfully saved to all locations.")

if __name__ == "__main__":
    generate_plot()
