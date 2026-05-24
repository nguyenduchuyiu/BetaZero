import json
import matplotlib.pyplot as plt
import numpy as np

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

def get_proof_lengths(path):
    with open(path, "r") as f:
        data = json.load(f)
    
    nodes = data.get("nodes", [])
    lengths = []
    
    for n in nodes:
        if n.get("type") == "OR" and n.get("status") == "SOLVED":
            content = n.get("content")
            if isinstance(content, dict) and "proof_body" in content:
                pb = content["proof_body"]
                if pb:
                    lines = len(pb.strip().split("\n"))
                    chars = len(pb)
                    lengths.append({
                        "id": n.get("id"),
                        "lines": lines,
                        "chars": chars
                    })
    return lengths

non_scaffold = get_proof_lengths("aime_1983_p1-non-scaffold.json")
scaffold = get_proof_lengths("outputs/rollouts/gemini3flash/miniF2F-valid/aime_1983_p1.json")

# Root proof lengths (state_0)
root_ns = next((x["lines"] for x in non_scaffold if x["id"] == "state_0"), 137)
root_s = next((x["lines"] for x in scaffold if x["id"] == "state_0"), 39)

root_ns_chars = next((x["chars"] for x in non_scaffold if x["id"] == "state_0"), 5870)
root_s_chars = next((x["chars"] for x in scaffold if x["id"] == "state_0"), 1559)

# Subgoal distributions (excluding root state_0 to isolate actual component proof lengths)
sub_ns = sorted([x["lines"] for x in non_scaffold if x["id"] != "state_0"])
sub_s = sorted([x["lines"] for x in scaffold if x["id"] != "state_0"])

# Setup the figure with a dual subplot (1x2 grid)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# --- PLOT 1: Final Proof Length (Lines & Characters) ---
categories = ['Non-Scaffold-Aware\n(Erronous Solved)', 'Scaffold-Aware\n(Stitched Correct)']
bar_width = 0.35
indices = np.arange(len(categories))

# Plot lines on primary Y axis
bars_lines = ax1.bar(indices - bar_width/2, [root_ns, root_s], bar_width, 
                     color='#ff5e62', label='Lines of Code', alpha=0.9, edgecolor='#ff9a9e', linewidth=1)
# Plot characters on secondary Y axis
ax1_sec = ax1.twinx()
bars_chars = ax1_sec.bar(indices + bar_width/2, [root_ns_chars, root_s_chars], bar_width, 
                         color='#00f2fe', label='Character Count', alpha=0.8, edgecolor='#4facfe', linewidth=1)

ax1.set_ylabel('Proof Length (Lines)', color='#ff5e62', fontsize=12, fontweight='bold')
ax1_sec.set_ylabel('Proof Size (Characters)', color='#00f2fe', fontsize=12, fontweight='bold')
ax1.set_title('Root Theorem Proof Length (state_0)', fontsize=14, fontweight='bold', pad=15, color='#ffffff')
ax1.set_xticks(indices)
ax1.set_xticklabels(categories, fontsize=11, fontweight='bold')
ax1.grid(True, linestyle='--', alpha=0.3)

# Add values on top of bars
for bar in bars_lines:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, yval + 3, f"{yval} lines", 
             ha='center', va='bottom', color='#ff5e62', fontweight='bold', fontsize=10)

for bar in bars_chars:
    yval = bar.get_height()
    ax1_sec.text(bar.get_x() + bar.get_width()/2, yval + 100, f"{yval} ch", 
                 ha='center', va='bottom', color='#00f2fe', fontweight='bold', fontsize=10)

# Combine legends
lines_lbl, lines_hnd = ax1.get_legend_handles_labels()
chars_lbl, chars_hnd = ax1_sec.get_legend_handles_labels()
ax1.legend(lines_lbl + chars_lbl, lines_hnd + chars_hnd, loc='upper right', 
           facecolor='#1e222b', edgecolor='#4e5569')

# --- PLOT 2: Subgoal Proof Length Distribution ---
# Cumulative distributions to show complexity scaling
ax2.plot(np.arange(1, len(sub_ns)+1), sub_ns, label='Non-Scaffold Subgoals (N=37)', 
         color='#ff5e62', linewidth=3, marker='o', markersize=6, alpha=0.9)
ax2.plot(np.arange(1, len(sub_s)+1), sub_s, label='Scaffold Subgoals (N=9)', 
         color='#00f2fe', linewidth=3, marker='s', markersize=6, alpha=0.9)

ax2.set_xlabel('Subgoal Rank (Ordered by Length)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Proof Length (Lines)', fontsize=11, fontweight='bold')
ax2.set_title('Subgoal Proof Length Distribution', fontsize=14, fontweight='bold', pad=15, color='#ffffff')
ax2.legend(loc='upper left', facecolor='#1e222b', edgecolor='#4e5569')
ax2.grid(True, linestyle='--', alpha=0.3)

# Add highlight text in Plot 2
ax2.annotate('Verbose/Redundant\nSubgoal skeletons\n(Max: 31 lines)', xy=(len(sub_ns)-1, sub_ns[-1]), 
             xytext=(len(sub_ns)-10, sub_ns[-1]-10),
             arrowprops=dict(facecolor='#ff5e62', shrink=0.08, width=1, headwidth=6),
             color='#ff5e62', fontweight='bold', fontsize=9)

ax2.annotate('Direct/Clean\nSubgoal proofs\n(Max: 11 lines)', xy=(len(sub_s)-1, sub_s[-1]), 
             xytext=(len(sub_s)-4, sub_s[-1]+10),
             arrowprops=dict(facecolor='#00f2fe', shrink=0.08, width=1, headwidth=6),
             color='#00f2fe', fontweight='bold', fontsize=9)

plt.tight_layout()

# Save directly to the artifacts directory
artifact_path = "/root/.gemini/antigravity/brain/4790e5a6-fe02-4b13-b916-074f3320550a/artifacts/proof_length_comparison.png"
plt.savefig(artifact_path, dpi=300, facecolor='#1a1c23', bbox_inches='tight')
print(f"Saved plot successfully to {artifact_path}")
