import os
import json
import numpy as np
import matplotlib.pyplot as plt

def analyze_directory(directory):
    solved_proofs = {}
    if not os.path.exists(directory):
        return solved_proofs
        
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            path = os.path.join(directory, filename)
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                
                if "nodes" in data:
                    nodes = data["nodes"]
                    state_0 = None
                    if isinstance(nodes, list):
                        state_0 = next((n for n in nodes if n.get("id") == "state_0"), None)
                    elif isinstance(nodes, dict):
                        state_0 = nodes.get("state_0")
                    
                    if state_0 and state_0.get("status") == "SOLVED":
                        content = state_0.get("content")
                        if isinstance(content, dict) and "proof_body" in content:
                            pb = content["proof_body"]
                            if pb:
                                lines = len(pb.strip().split("\n"))
                                solved_proofs[filename] = lines
                
                elif "samples" in data and "summary" in data:
                    summary = data.get("summary", {})
                    if summary.get("solved") == True or summary.get("passed_count", 0) > 0:
                        samples = data.get("samples", [])
                        passed_sample = None
                        for s in samples:
                            ver = s.get("verification")
                            if isinstance(ver, dict) and ver.get("pass") == True:
                                passed_sample = s
                                break
                        
                        if passed_sample:
                            code = passed_sample.get("extracted_code")
                            if code:
                                total_lines = len(code.strip().split("\n"))
                                proof_body = ""
                                if ":= by" in code:
                                    proof_body = code.split(":= by", 1)[1]
                                elif ":=" in code:
                                    proof_body = code.split(":=", 1)[1]
                                
                                if proof_body:
                                    pb_lines = len(proof_body.strip().split("\n"))
                                else:
                                    pb_lines = total_lines
                                solved_proofs[filename] = pb_lines
            except Exception as e:
                pass
    return solved_proofs

baseline = analyze_directory("outputs/baseline_logs")
valid = analyze_directory("outputs/rollouts/gemini3flash/miniF2F-valid")
test = analyze_directory("outputs/rollouts/gemini3flash/miniF2F-test")
scaffold = {**valid, **test}

intersection = sorted(list(set(baseline.keys()) & set(scaffold.keys())))
N = len(intersection)

ns_vals = [baseline[f] for f in intersection]
s_vals = [scaffold[f] for f in intersection]

# Sort by Non-Scaffold proof length for clean visual display
sorted_indices = np.argsort(ns_vals)
intersection_sorted = [intersection[i] for i in sorted_indices]
ns_sorted = [ns_vals[i] for i in sorted_indices]
s_sorted = [s_vals[i] for i in sorted_indices]

# Set premium styling parameters
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
plt.rcParams['figure.facecolor'] = '#1a1c23'  # Slate dark background
plt.rcParams['axes.facecolor'] = '#1a1c23'
plt.rcParams['axes.edgecolor'] = '#4e5569'
plt.rcParams['axes.labelcolor'] = '#e2e8f0'
plt.rcParams['xtick.color'] = '#a0aec0'
plt.rcParams['ytick.color'] = '#a0aec0'
plt.rcParams['grid.color'] = '#2d3748'
plt.rcParams['text.color'] = '#e2e8f0'

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# --- PLOT 1: Line comparison sorted by Non-Scaffold proof length ---
ax1.plot(np.arange(N), ns_sorted, label='Non-Scaffold-Aware Baseline', 
         color='#ff5e62', linewidth=2.5, alpha=0.9, marker='o', markersize=4)
ax1.plot(np.arange(N), s_sorted, label='Scaffold-Aware MCTS', 
         color='#00f2fe', linewidth=2.5, alpha=0.9, marker='s', markersize=4)

# Fill the area between them to highlight the reduction
ax1.fill_between(np.arange(N), ns_sorted, s_sorted, color='#ff5e62', alpha=0.15, label='Proof Bloat Reduction')

ax1.set_xlabel('Solved Problem Index (Sorted by Non-Scaffold Length)', fontsize=12, fontweight='bold', labelpad=10)
ax1.set_ylabel('Proof Length (Lines of Code)', fontsize=12, fontweight='bold', labelpad=10)
ax1.set_title(f'Proof Length Across Solved Problems (N = {N})', fontsize=15, fontweight='bold', pad=15, color='#ffffff')
ax1.legend(loc='upper left', facecolor='#1e222b', edgecolor='#4e5569', fontsize=11)
ax1.grid(True, linestyle='--', alpha=0.3)
ax1.set_xlim(-1, N)

# --- PLOT 2: CDF (Cumulative Distribution Function) ---
# Calculate CDF data
ns_cdf_x = np.sort(ns_vals)
ns_cdf_y = np.arange(1, len(ns_cdf_x) + 1) / len(ns_cdf_x)

s_cdf_x = np.sort(s_vals)
s_cdf_y = np.arange(1, len(s_cdf_x) + 1) / len(s_cdf_x)

ax2.step(ns_cdf_x, ns_cdf_y, label='Non-Scaffold-Aware Baseline', color='#ff5e62', linewidth=3, where='post')
ax2.step(s_cdf_x, s_cdf_y, label='Scaffold-Aware MCTS', color='#00f2fe', linewidth=3, where='post')

ax2.set_xlabel('Proof Length (Lines of Code)', fontsize=12, fontweight='bold', labelpad=10)
ax2.set_ylabel('Cumulative Probability (CDF)', fontsize=12, fontweight='bold', labelpad=10)
ax2.set_title('Proof Length Cumulative Distribution (CDF)', fontsize=15, fontweight='bold', pad=15, color='#ffffff')
ax2.legend(loc='lower right', facecolor='#1e222b', edgecolor='#4e5569', fontsize=11)
ax2.grid(True, linestyle='--', alpha=0.3)
ax2.set_xlim(0, 45)  # Zoom in on main distribution

# Add text box with statistics on ax2
stats_text = (
    f"=== Summary Stats ===\n"
    f"Baseline Mean: {np.mean(ns_vals):.1f} lines\n"
    f"Scaffold Mean: {np.mean(s_vals):.1f} lines\n"
    f"Baseline Median: {np.median(ns_vals):.1f} lines\n"
    f"Scaffold Median: {np.median(s_vals):.1f} lines\n"
    f"Average Reduction: {((np.mean(ns_vals) - np.mean(s_vals)) / np.mean(ns_vals)) * 100:.1f}%\n"
    f"Median Reduction: {((np.median(ns_vals) - np.median(s_vals)) / np.median(ns_vals)) * 100:.1f}%"
)
props = dict(boxstyle='round,pad=0.8', facecolor='#1e222b', edgecolor='#4e5569', alpha=0.95)
ax2.text(0.05, 0.45, stats_text, transform=ax2.transAxes, fontsize=10,
         verticalalignment='bottom', bbox=props, color='#e2e8f0', fontfamily='monospace')

plt.tight_layout()

# Save directly to the artifacts directory
artifact_path = "/root/.gemini/antigravity/brain/4790e5a6-fe02-4b13-b916-074f3320550a/artifacts/proof_length_statistics.png"
plt.savefig(artifact_path, dpi=300, facecolor='#1a1c23', bbox_inches='tight')
print(f"Saved plot successfully to {artifact_path}")

print(f"=== STATS SUMMARY ===")
print(f"Baseline (Non-Scaffold) - Mean: {np.mean(ns_vals):.2f}, Median: {np.median(ns_vals):.2f}, Max: {np.max(ns_vals)}, Min: {np.min(ns_vals)}")
print(f"Scaffold-Aware MCTS     - Mean: {np.mean(s_vals):.2f}, Median: {np.median(s_vals):.2f}, Max: {np.max(s_vals)}, Min: {np.min(s_vals)}")
print(f"Average proof line reduction: {((np.mean(ns_vals) - np.mean(s_vals)) / np.mean(ns_vals)) * 100:.2f}%")
print(f"Median proof line reduction: {((np.median(ns_vals) - np.median(s_vals)) / np.median(ns_vals)) * 100:.2f}%")
