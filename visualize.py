import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Set beautiful scientific style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
plt.rcParams['axes.edgecolor'] = '#CCCCCC'
plt.rcParams['axes.linewidth'] = 0.8

def plot_persistence_diagram(dgms, title="Persistence Diagram", save_path=None):
    """Plots birth-death persistence diagrams for H0 and H1."""
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Max death value for plotting diagonal limit
    max_val = 0.0
    for dim_dgms in dgms:
        if len(dim_dgms) > 0:
            finite_vals = dim_dgms[np.isfinite(dim_dgms)]
            if len(finite_vals) > 0:
                max_val = max(max_val, np.max(finite_vals))
                
    max_val = max(max_val * 1.1, 1.0)
    
    # Plot diagonal line
    ax.plot([0, max_val], [0, max_val], color='#888888', linestyle='--', alpha=0.7)
    
    colors = ['#FF4C4C', '#1F77B4', '#2CA02C']
    markers = ['o', '^', 's']
    labels = [r'$H_0$ (Components)', r'$H_1$ (Loops)', r'$H_2$ (Cavities)']
    
    for dim, dgm in enumerate(dgms):
        if len(dgm) == 0:
            continue
        # Separate infinite death points
        inf_mask = ~np.isfinite(dgm[:, 1])
        finite_pts = dgm[~inf_mask]
        inf_pts = dgm[inf_mask]
        
        # Plot finite points
        ax.scatter(finite_pts[:, 0], finite_pts[:, 1], color=colors[dim], marker=markers[dim], 
                   alpha=0.8, s=40, label=labels[dim] if dim < len(labels) else f'H{dim}')
        
        # Plot infinite points near top boundary
        if len(inf_pts) > 0:
            ax.scatter(inf_pts[:, 0], [max_val * 0.95] * len(inf_pts), color=colors[dim], 
                       marker=markers[dim], facecolors='none', edgecolors=colors[dim], 
                       alpha=0.8, s=60, linewidths=1.5, label=f'{labels[dim]} (infinite death)')
            
    ax.set_xlim(-0.02 * max_val, max_val)
    ax.set_ylim(-0.02 * max_val, max_val)
    ax.set_xlabel("Birth time (Distance)", fontsize=11, fontweight='semibold')
    ax.set_ylabel("Death time (Distance)", fontsize=11, fontweight='semibold')
    ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
    ax.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='#E0E0E0')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()

def plot_persistence_barcode(dgms, title="Persistence Barcode", save_path=None):
    """Plots persistence barcodes showing lifetime intervals for H0 and H1."""
    # Count total generators
    total_gens = sum(len(dgm) for dgm in dgms)
    if total_gens == 0:
        return
        
    fig, ax = plt.subplots(figsize=(8, max(4, total_gens * 0.15)))
    
    # Determine max value for infinite features
    max_val = 0.0
    for dim_dgms in dgms:
        if len(dim_dgms) > 0:
            finite_vals = dim_dgms[np.isfinite(dim_dgms)]
            if len(finite_vals) > 0:
                max_val = max(max_val, np.max(finite_vals))
    max_val = max(max_val * 1.1, 1.0)
    
    colors = ['#FF4C4C', '#1F77B4', '#2CA02C']
    
    y_idx = 0
    legend_handles = []
    
    for dim, dgm in enumerate(dgms):
        if len(dgm) == 0:
            continue
        
        # Sort generators by birth time for visual aesthetics
        sorted_indices = np.argsort(dgm[:, 0])
        sorted_dgm = dgm[sorted_indices]
        
        for birth, death in sorted_dgm:
            if not np.isfinite(death):
                death_plot = max_val
                # Draw arrowhead to represent infinity
                ax.arrow(birth, y_idx, death_plot - birth, 0, head_width=0.15, head_length=0.03 * max_val, 
                         fc=colors[dim], ec=colors[dim], alpha=0.8)
            else:
                death_plot = death
                ax.plot([birth, death_plot], [y_idx, y_idx], color=colors[dim], linewidth=2.5, alpha=0.8)
                
            y_idx += 1
            
        # For legend
        line, = ax.plot([], [], color=colors[dim], linewidth=2.5, label=f'$H_{dim}$')
        legend_handles.append(line)
        
    ax.set_ylim(-1, y_idx)
    ax.set_yticks([])
    ax.set_xlabel("Filtration Value (Distance)", fontsize=11, fontweight='semibold')
    ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
    ax.legend(handles=legend_handles, loc='upper right', frameon=True, facecolor='white', edgecolor='#E0E0E0')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()

def plot_embedding_projection(embeddings, labels, title="Layer Embedding Projection", method='pca', save_path=None):
    """Plots 2D projection of activations, colored by label."""
    if embeddings.shape[1] > 2:
        if method == 'tsne':
            # Use perplexity min of 30 or N-1
            perp = min(30, max(5, len(embeddings) // 10))
            projector = TSNE(n_components=2, perplexity=perp, random_state=42)
        else:
            projector = PCA(n_components=2, random_state=42)
        projected = projector.fit_transform(embeddings)
    else:
        projected = embeddings
        
    fig, ax = plt.subplots(figsize=(6, 6))
    scatter = ax.scatter(projected[:, 0], projected[:, 1], c=labels, cmap='coolwarm', alpha=0.8, s=30, edgecolors='none')
    
    # Legend
    classes = np.unique(labels)
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.coolwarm(c / max(classes) if max(classes) > 0 else 0), 
                          markersize=8, label=f'Class {int(c)}') for c in classes]
    ax.legend(handles=handles, frameon=True, edgecolor='#E0E0E0')
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel("Component 1", fontsize=10)
    ax.set_ylabel("Component 2", fontsize=10)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()

def plot_layer_metrics(layers, trained_metrics, untrained_metrics, metric_name="Persistence Entropy", save_path=None):
    """Compares evolution of a metric across layers for trained vs untrained network."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    x = np.arange(len(layers))
    
    ax.plot(x, trained_metrics, marker='o', linewidth=2.5, color='#4A90E2', label='Trained')
    ax.plot(x, untrained_metrics, marker='x', linewidth=2, linestyle='--', color='#9B9B9B', label='Untrained (Random)')
    
    ax.set_xticks(x)
    ax.set_xticklabels(layers, rotation=30, ha='right')
    ax.set_xlabel("Layer", fontsize=11, fontweight='semibold')
    ax.set_ylabel(metric_name, fontsize=11, fontweight='semibold')
    ax.set_title(f"Evolution of {metric_name} Across Layers", fontsize=13, fontweight='bold', pad=15)
    ax.legend(frameon=True, facecolor='white', edgecolor='#E0E0E0')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()
