#!/usr/bin/env python3
"""
Pixel Embedding Visualization Demo
==================================
Standalone demo to illustrate the visualization experiment for proving 
that Pixel Embedding (with multi-dimensional info from INP / channel encoding)
improves LLM's "understanding" of pixels compared to standard LLM Tokenizer path.

- 140 fake pixels across 7 rainbow color zones (ROYGBIV reference)
- Two main views:
  1. t-SNE scatter plots (side-by-side comparison)  → 保存为 pixel_embedding_scatter.png
  2. Cosine similarity heatmaps (block structure)   → 保存为 pixel_embedding_heatmap.png
- Quantitative metrics: Silhouette score + intra/inter-zone cosine similarity ratio
- Replace the "simulate_embeddings" part with real extraction logic from trained models.

Usage:
    python viz_pixel_embedding_demo.py
    (Requires: pip install numpy matplotlib seaborn scikit-learn)
    Optional: pip install umap-learn

This script now saves two separate high-quality figures for papers.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from matplotlib.colors import ListedColormap
import os

# Optional UMAP
try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("[Info] umap-learn not found. Using t-SNE only.")

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# ====================== 1. Generate Fake Data ======================
np.random.seed(42)
N_ZONES = 7
N_PER_ZONE = 20
N = N_ZONES * N_PER_ZONE

zones = np.repeat(np.arange(N_ZONES), N_PER_ZONE)

pixel_values = np.concatenate([
    np.random.randint(z * 36, (z + 1) * 36, N_PER_ZONE) for z in range(N_ZONES)
])

RAINBOW_COLORS = ['#E31A1C', '#FF7F00', '#FFFF00', '#33A02C', 
                  '#1F78B4', '#6A3D9A', '#B15928']
cmap_rainbow = ListedColormap(RAINBOW_COLORS)
zone_names = ['Red Zone', 'Orange Zone', 'Yellow Zone', 'Green Zone', 
              'Blue Zone', 'Indigo/Purple Zone', 'Violet/Brown Zone']

print(f"Generated {N} fake pixels across {N_ZONES} rainbow zones.")

# ====================== 2. Simulate Embeddings ======================
DIM = 512

def simulate_embeddings():
    embs_base = np.random.randn(N, DIM).astype(np.float32) * 1.8
    embs_base += (pixel_values / 255.0 * 2.5).reshape(-1, 1)

    zone_centers = np.random.randn(N_ZONES, DIM).astype(np.float32) * 5.5
    embs_new = zone_centers[zones] + np.random.randn(N, DIM).astype(np.float32) * 1.3

    pos_mod = np.sin(np.linspace(0, 3 * np.pi, N)).reshape(-1, 1) * np.random.randn(1, DIM) * 0.8
    channel_mod = (zones % 3).reshape(-1, 1) * np.random.randn(1, DIM) * 0.6
    embs_new = embs_new + pos_mod + channel_mod

    return embs_base, embs_new

embs_base, embs_new = simulate_embeddings()

# ====================== 3. Dimensionality Reduction ======================
def reduce_dim(embs, method='tsne'):
    if method == 'umap' and HAS_UMAP:
        reducer = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
        return reducer.fit_transform(embs)
    else:
        reducer = TSNE(n_components=2, perplexity=12, learning_rate='auto', 
                       max_iter=1500, random_state=42, init='pca')
        return reducer.fit_transform(embs)

embs_base_2d = reduce_dim(embs_base, 'tsne')
embs_new_2d  = reduce_dim(embs_new,  'umap' if HAS_UMAP else 'tsne')

# ====================== 4. Metrics ======================
def compute_intra_inter_ratio(embs, zones):
    embs_n = embs / np.linalg.norm(embs, axis=1, keepdims=True)
    sim_matrix = embs_n @ embs_n.T
    intra, inter = [], []
    for z in range(N_ZONES):
        mask_in = (zones == z)
        mask_out = ~mask_in
        if mask_in.sum() > 1:
            intra.append(sim_matrix[np.ix_(mask_in, mask_in)].mean())
        inter.append(sim_matrix[np.ix_(mask_in, mask_out)].mean())
    intra_avg = np.mean(intra)
    inter_avg = np.mean(inter)
    return intra_avg, inter_avg, intra_avg / (inter_avg + 1e-8)

def print_metrics(name, embs):
    sil = silhouette_score(embs, zones)
    intra, inter, ratio = compute_intra_inter_ratio(embs, zones)
    print(f"{name:12s} | Silhouette: {sil:.3f} | Intra/Inter ratio: {ratio:.3f}")

print("\n=== Quantitative Metrics ===")
print_metrics("Baseline", embs_base)
print_metrics("Proposed", embs_new)

# ====================== 5. Plotting: Two Separate Figures ======================

# ---- Figure 1: Scatter Plots ----
fig_scatter = plt.figure(figsize=(14, 6))

ax1 = fig_scatter.add_subplot(1, 2, 1)
sc1 = ax1.scatter(embs_base_2d[:, 0], embs_base_2d[:, 1], 
                  c=zones, cmap=cmap_rainbow, s=55, alpha=0.85, 
                  edgecolors='black', linewidths=0.4)
ax1.set_title('Figure (a): Baseline — LLM Tokenizer Path\n'
              '(Clusters weak / mostly numerical)', fontsize=11, pad=8)
ax1.set_xlabel('Dim 1 (t-SNE/UMAP)')
ax1.set_ylabel('Dim 2 (t-SNE/UMAP)')
ax1.grid(True, alpha=0.3)

ax2 = fig_scatter.add_subplot(1, 2, 2)
sc2 = ax2.scatter(embs_new_2d[:, 0], embs_new_2d[:, 1], 
                  c=zones, cmap=cmap_rainbow, s=55, alpha=0.85, 
                  edgecolors='black', linewidths=0.4)
ax2.set_title('Figure (b): Proposed — Pixel Embedding + Multi-dim Info\n'
              '(Tighter, well-separated color zone clusters)', fontsize=11, pad=8)
ax2.set_xlabel('Dim 1 (t-SNE/UMAP)')
ax2.set_ylabel('Dim 2 (t-SNE/UMAP)')
ax2.grid(True, alpha=0.3)

cbar_ax = fig_scatter.add_axes([0.92, 0.15, 0.015, 0.7])
cbar = fig_scatter.colorbar(sc2, cax=cbar_ax, ticks=range(N_ZONES))
cbar.set_ticklabels([f'{i}: {zone_names[i]}' for i in range(N_ZONES)])
cbar.set_label('Rainbow Color Zone', rotation=270, labelpad=12)

fig_scatter.suptitle('t-SNE/UMAP Projection of Pixel Embeddings\n'
                     '140 Synthetic Pixels | 7 Rainbow Color Zones', 
                     fontsize=13, y=0.98)

plt.tight_layout(rect=[0, 0, 0.90, 0.95])

scatter_path = os.path.join(os.getcwd(), 'pixel_embedding_scatter.png')
fig_scatter.savefig(scatter_path, bbox_inches='tight', facecolor='white', dpi=300)
print(f"\n[Saved] Scatter figure: {scatter_path}")

# ---- Figure 2: Cosine Similarity Heatmaps ----
def plot_sim_heatmap(embs, title, ax):
    embs_n = embs / np.linalg.norm(embs, axis=1, keepdims=True)
    sim = embs_n @ embs_n.T
    sns.heatmap(sim, ax=ax, cmap='RdBu_r', center=0, vmin=-0.3, vmax=1.0,
                cbar=False, xticklabels=False, yticklabels=False)
    ax.set_title(title, fontsize=10, pad=6)
    for i in range(1, N_ZONES):
        pos = i * N_PER_ZONE
        ax.axhline(pos, color='white', linewidth=1.2, alpha=0.9)
        ax.axvline(pos, color='white', linewidth=1.2, alpha=0.9)
    ax.set_xlabel('Pixel index (grouped by zone)')
    ax.set_ylabel('Pixel index (grouped by zone)')

fig_heatmap = plt.figure(figsize=(14, 6))

ax3 = fig_heatmap.add_subplot(1, 2, 1)
plot_sim_heatmap(embs_base, 
                 'Figure (c): Baseline\nWeak block-diagonal structure', ax3)

ax4 = fig_heatmap.add_subplot(1, 2, 2)
plot_sim_heatmap(embs_new, 
                 'Figure (d): Proposed\nStrong block-diagonal structure', ax4)

fig_heatmap.suptitle('Cosine Similarity Heatmaps\n'
                     '140 Synthetic Pixels | 7 Rainbow Color Zones', 
                     fontsize=13, y=0.98)

plt.tight_layout(rect=[0, 0, 0.90, 0.95])

heatmap_path = os.path.join(os.getcwd(), 'pixel_embedding_heatmap.png')
fig_heatmap.savefig(heatmap_path, bbox_inches='tight', facecolor='white', dpi=300)
print(f"[Saved] Heatmap figure: {heatmap_path}")

print("\n=== Done ===")
print("Two figures have been saved successfully for your paper.")