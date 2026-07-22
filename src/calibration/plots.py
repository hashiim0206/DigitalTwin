"""
plots.py
Generates microscopic trajectory fidelity plots:
- XY Overlays (Real vs Sim)
- Space-Time Diagrams
- Histograms (RMSE, TD)
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt

def plot_histograms(metrics_data, out_dir):
    rmses = [v['rmse'] for v in metrics_data['vehicles']]
    tds = [v['td'] for v in metrics_data['vehicles']]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.hist(rmses, bins=20, color='skyblue', edgecolor='black')
    ax1.set_title('Spatial RMSE Distribution')
    ax1.set_xlabel('Normalized RMSE')
    ax1.set_ylabel('Vehicle Count')
    
    ax2.hist(tds, bins=20, color='lightgreen', edgecolor='black')
    ax2.set_title('Timing Deviation (TD) Distribution')
    ax2.set_xlabel('Normalized TD')
    ax2.set_ylabel('Vehicle Count')
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fidelity_histograms.png'))
    plt.close()

def plot_xy_overlays(matched_data, out_dir, num_samples=12):
    # Select up to num_samples vehicles to plot
    samples = matched_data[:num_samples]
    
    # Calculate grid size (e.g. 3x4 for 12 samples)
    cols = 4
    rows = (len(samples) + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3*rows))
    axes = axes.flatten()
    
    for i, veh in enumerate(samples):
        ax = axes[i]
        
        real = np.array(veh['real_positions'])
        sim = np.array([p[1:] for p in veh['sim_positions']])
        
        # Normalize for overlay visualization
        if len(real) > 1 and len(sim) > 1:
            r_min, r_max = np.min(real, axis=0), np.max(real, axis=0)
            r_range = np.where(r_max - r_min == 0, 1, r_max - r_min)
            r_norm = (real - r_min) / r_range
            
            s_min, s_max = np.min(sim, axis=0), np.max(sim, axis=0)
            s_range = np.where(s_max - s_min == 0, 1, s_max - s_min)
            s_norm = (sim - s_min) / s_range
            
            ax.plot(r_norm[:, 0], r_norm[:, 1], 'b-', label='Real', alpha=0.7)
            ax.plot(s_norm[:, 0], s_norm[:, 1], 'r--', label='Sim', alpha=0.7)
            
        ax.set_title(f"Veh {veh['real_id']} ({veh['origin']}->{veh['dest']})")
        ax.set_xticks([])
        ax.set_yticks([])
        
    # Hide empty subplots
    for i in range(len(samples), len(axes)):
        axes[i].set_visible(False)
        
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(out_dir, 'xy_overlays.png'))
    plt.close()

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out_dir = os.path.join(base_dir, 'outputs')
    
    matched_path = os.path.join(out_dir, 'matched_trajectories.json')
    metrics_path = os.path.join(out_dir, 'micro_metrics.json')
    
    if not os.path.exists(matched_path) or not os.path.exists(metrics_path):
        print("Required JSON files missing. Run match_real_sim.py and metrics.py first.")
        return
        
    with open(matched_path, 'r') as f:
        matched_data = json.load(f)
        
    with open(metrics_path, 'r') as f:
        metrics_data = json.load(f)
        
    print("Generating Histograms...")
    plot_histograms(metrics_data, out_dir)
    
    print("Generating XY Overlays...")
    plot_xy_overlays(matched_data, out_dir)
    
    print(f"Plots saved to {out_dir}/")

if __name__ == "__main__":
    main()
