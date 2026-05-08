import json
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_rewards(json_path, output_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    scores = [item['r_env'] for item in data]
    
    plt.figure(figsize=(10, 6))
    
    # Use a premium color
    n, bins, patches = plt.hist(scores, bins=20, color='#4A90E2', edgecolor='white', alpha=0.8)
    
    plt.title('Distribution of Environment Rewards (r_env)', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Reward Score', fontsize=12)
    plt.ylabel('Number of Theorems', fontsize=12)
    
    # Add grid
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    # Add stats box
    stats_text = (
        f"Total: {len(scores)}\n"
        f"Mean: {np.mean(scores):.4f}\n"
        f"Median: {np.median(scores):.4f}\n"
        f"Max: {np.max(scores):.4f}\n"
        f"Min: {np.min(scores):.4f}\n"
        f"Perfect (1.0): {sum(1 for s in scores if s >= 0.9999)}\n"
        f"Failed (0.0): {sum(1 for s in scores if s <= 0.0001)}"
    )
    plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    # Clean up spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    json_path = "/workspace/npthai/BetaZero/logs/batch_reward_report.json"
    output_path = "/workspace/npthai/BetaZero/logs/reward_distribution.png"
    plot_rewards(json_path, output_path)
