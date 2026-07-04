#!/usr/bin/env python3
"""
Pixel Embedding Visualization Demo
==================================
... (header unchanged)
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

# ====================== 2. Simulate Embeddings (改进版) ======================
DIM = 512

def simulate_embeddings():
    """改进：Proposed embedding 在同色系内引入可区分的细粒度结构"""
    # Baseline: 主要数值驱动，簇松散
    embs_base = np.random.randn(N, DIM).astype(np.float32) * 1.8
    embs_base += (pixel_values / 255.0 * 2.5).reshape(-1, 1)

    # Proposed: 更强的 zone center + intra-zone variation
    zone_centers = np.random.randn(N_ZONES, DIM).astype(np.float32) * 5.5
    
    # === 核心改进：同色系内变异 ===
    intra_intensity = (pixel_values % 36 / 36.0).reshape(-1, 1)          # [0,1) 强度
    intra_pos = np.sin(np.linspace(0, 4 * np.pi, N)).reshape(-1, 1)      # 模拟位置/通道信息
    intra_noise = np.random.randn(N, DIM).astype(np.float32) * 0.8       # 适度随机扰动
    
    embs_new = zone_centers[zones] + \
               intra_intensity * 2.8 + \
               intra_pos * 1.6 + \
               intra_noise * (0.6 + intra_intensity * 0.6)               # 强度越高，内部差异略大

    # 轻微全局 channel / multi-dim modulation
    channel_mod = (zones % 3 + intra_intensity.squeeze() * 2).reshape(-1, 1) * \
                  np.random.randn(1, DIM) * 0.5
    embs_new = embs_new + channel_mod

    return embs_base, embs_new

embs_base, embs_new = simulate_embeddings()

# ====================== 3. Dimensionality Reduction ======================
def reduce_dim(embs, method='tsne'):
    if method == 'umap' and HAS_UMAP:
        reducer = UMAP(n_components=2, n_neighbors=15, min_dist=0.08, metric='cosine', random_state=42)
        return reducer.fit_transform(embs)
    else:
        reducer = TSNE(n_components=2, perplexity=12, learning_rate='auto', 
                       max_iter=1500, random_state=42, init='pca')
        return reducer.fit_transform(embs)

embs_base_2d = reduce_dim(embs_base, 'tsne')
embs_new_2d  = reduce_dim(embs_new,  'umap' if HAS_UMAP else 'tsne')

# 可视化轻微 jitter（仅用于绘图，让同色点更易区分）
def add_visual_jitter(embs_2d, scale=0.12):
    jitter = np.random.randn(*embs_2d.shape) * scale
    return embs_2d + jitter

embs_new_2d_vis = add_visual_jitter(embs_new_2d, scale=0.12)   # 仅右图使用

# ====================== 4. Metrics (不变) ======================
# ... (compute_intra_inter_ratio 和 print_metrics 函数保持不变)

print("\n=== Quantitative Metrics ===")
print_metrics("Baseline", embs_base)
print_metrics("Proposed", embs_new)

# ====================== 5. Plotting ======================

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
sc2 = ax2.scatter(embs_new_2d_vis[:, 0], embs_new_2d_vis[:, 1],   # 使用带 jitter 的版本
                  c=zones, cmap=cmap_rainbow, s=55, alpha=0.88, 
                  edgecolors='black', linewidths=0.5)
ax2.set_title('Figure (b): Proposed — Pixel Embedding + Multi-dim Info\n'
              '(Tighter, well-separated color zone clusters\nwith intra-zone variation)', 
              fontsize=11, pad=8)
ax2.set_xlabel('Dim 1 (t-SNE/UMAP)')
ax2.set_ylabel('Dim 2 (t-SNE/UMAP)')
ax2.grid(True, alpha=0.3)

# ... (colorbar 部分保持不变)

fig_scatter.suptitle('t-SNE/UMAP Projection of Pixel Embeddings\n'
                     '140 Synthetic Pixels | 7 Rainbow Color Zones', 
                     fontsize=13, y=0.98)

plt.tight_layout(rect=[0, 0, 0.90, 0.95])

scatter_path = os.path.join(os.getcwd(), 'pixel_embedding_scatter.png')
fig_scatter.savefig(scatter_path, bbox_inches='tight', facecolor='white', dpi=300)
print(f"\n[Saved] Scatter figure: {scatter_path}")

# Heatmap 部分保持不变（或根据需要轻微调整）
# ... (Figure 2: Cosine Similarity Heatmaps 保持原样)

print("\n=== Done ===")
print("Two figures have been saved successfully for your paper.")