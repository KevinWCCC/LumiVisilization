#!/usr/bin/env python3
"""
Pixel Embedding Visualization Demo
==================================
Standalone demo to illustrate the visualization experiment for proving 
that Pixel Embedding (with multi-dimensional info from INP / channel encoding)
improves LLM's "understanding" of pixels compared to standard LLM Tokenizer path.

- 140 fake pixels across 7 rainbow color zones (ROYGBIV reference)
- Two main views:
  1. t-SNE scatter plots (side-by-side comparison)
  2. Cosine similarity heatmaps (block structure shows intra-zone similarity)
- Quantitative metrics: Silhouette score + intra/inter-zone cosine similarity ratio
- Replace the "simulate_embeddings" part with your real extraction logic from trained models.

Usage:
    python viz_pixel_embedding_demo.py
    (Requires: pip install numpy matplotlib seaborn scikit-learn)
    Optional: pip install umap-learn  (then set USE_UMAP=True for better global structure)

This produces pixel_embedding_viz.png (or .pdf) ready for papers/slides.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from matplotlib.colors import ListedColormap
import os  # for robust output directory creation

# Optional UMAP (recommended for LLM embedding viz - preserves global structure better)
try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("[Info] umap-learn not found. Using t-SNE only. Install with: pip install umap-learn")

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# ====================== 1. Generate Fake Data (140 pixels, 7 rainbow zones) ======================
np.random.seed(42)
N_ZONES = 7
N_PER_ZONE = 20
N = N_ZONES * N_PER_ZONE

# Zone labels 0-6 (for coloring and metrics)
zones = np.repeat(np.arange(N_ZONES), N_PER_ZONE)

# Fake pixel values: grouped by zone (0-35, 36-71, ..., 216-251)
# This simulates intensity bands; in real experiment you can use actual pixel values from images
pixel_values = np.concatenate([
    np.random.randint(z * 36, (z + 1) * 36, N_PER_ZONE) for z in range(N_ZONES)
])

# Rainbow-inspired discrete colors (ROYGBIV reference, perceptually reasonable)
RAINBOW_COLORS = ['#E31A1C', '#FF7F00', '#FFFF00', '#33A02C', '#1F78B4', '#6A3D9A', '#B15928']
cmap_rainbow = ListedColormap(RAINBOW_COLORS)
zone_names = ['Red Zone', 'Orange Zone', 'Yellow Zone', 'Green Zone', 
              'Blue Zone', 'Indigo/Purple Zone', 'Violet/Brown Zone']

print(f"Generated {N} fake pixels across {N_ZONES} rainbow zones.")
print(f"Pixel value range: {pixel_values.min()} - {pixel_values.max()}")

# ====================== 2. Simulate Embeddings (REPLACE THIS SECTION WITH REAL EXTRACTION) ======================
DIM = 512  # Typical hidden size; use your model's H (e.g. 4096 for LLaMA-3.1-8B)

def simulate_embeddings():
    """
    DEMO ONLY: Simulate what real embeddings would look like.
    
    REAL VERSION (in your project):
    - Baseline (Tokenizer path):
        ids = tokenizer.encode(str(p), add_special_tokens=False)
        embs = model.get_input_embeddings()(torch.tensor(ids, device=dev))
        baseline_embs[i] = embs.mean(0).cpu().numpy()   # or last token if preferred
    - Proposed (Pixel Embedding path):
        pix_t = torch.tensor(pixel_values, device=dev, dtype=torch.long)
        pix_emb = pixel_emb(pix_t)                      # [N, H]
        if intra_pos is not None:
            token_idx = torch.arange(N, device=dev)
            pos = intra_pos(token_idx).unsqueeze(0)     # or per-patch logic
            pix_emb = pix_emb + pos
        if channel_emb is not None:                     # RGB channel encoding
            ch = torch.tensor(fake_channels, device=dev)
            pix_emb = pix_emb + channel_emb(ch)
        proposed_embs = pix_emb.detach().cpu().numpy()
    
    The key hypothesis: Proposed embeddings (learned end-to-end + multi-dim INP/channel)
    will show MUCH stronger clustering by color zone than general-purpose tokenizer embeddings.
    """
    # Baseline: mostly random + weak correlation with numerical pixel value (typical for text token embs of numbers)
    embs_base = np.random.randn(N, DIM).astype(np.float32) * 1.8
    embs_base += (pixel_values / 255.0 * 2.5).reshape(-1, 1)   # slight numerical bias

    # Proposed: strong zone-aware clusters (effect of task-specific training + multi-dim encoding)
    # + positional modulation (simulates INP effect) + slight extra structure
    zone_centers = np.random.randn(N_ZONES, DIM).astype(np.float32) * 5.5
    embs_new = zone_centers[zones] + np.random.randn(N, DIM).astype(np.float32) * 1.3

    # Simulate multi-dimensional addition (INP positional + channel-like effect)
    # Sequential positions create smooth modulation within/between zones
    pos_mod = np.sin(np.linspace(0, 3 * np.pi, N)).reshape(-1, 1) * np.random.randn(1, DIM) * 0.8
    channel_mod = (zones % 3).reshape(-1, 1) * np.random.randn(1, DIM) * 0.6   # fake RGB channel effect
    embs_new = embs_new + pos_mod + channel_mod

    return embs_base, embs_new

embs_base, embs_new = simulate_embeddings()
print(f"Embeddings shape: baseline {embs_base.shape}, proposed {embs_new.shape}")

# ====================== 3. Dimensionality Reduction & Visualization ======================
def reduce_dim(embs, method='tsne'):
    if method == 'umap' and HAS_UMAP:
        reducer = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
        return reducer.fit_transform(embs)
    else:
        reducer = TSNE(n_components=2, perplexity=12, learning_rate='auto', 
                       max_iter=1500, random_state=42, init='pca')
        return reducer.fit_transform(embs)

print("\nReducing dimensions (t-SNE/UMAP)...")
embs_base_2d = reduce_dim(embs_base, 'tsne')
embs_new_2d  = reduce_dim(embs_new,  'umap' if HAS_UMAP else 'tsne')

# ====================== 4. Metrics (Quantitative Proof) ======================
def compute_intra_inter_ratio(embs, zones):
    """Average cosine similarity: intra-zone / inter-zone. Higher = better semantic grouping."""
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
    print(f"{name:12s} | Silhouette: {sil:.3f} | Intra/Inter cos ratio: {ratio:.3f} (intra={intra:.3f}, inter={inter:.3f})")
    return sil, ratio

print("\n=== Quantitative Metrics (higher is better for proposed) ===")
print_metrics("Baseline", embs_base)
print_metrics("Proposed", embs_new)

# ====================== 5. Plotting: Two-panel t-SNE + Two-panel Cosine Heatmap ======================
fig = plt.figure(figsize=(16, 12))

# ---- Top row: t-SNE / UMAP scatters ----
ax1 = fig.add_subplot(2, 2, 1)
sc1 = ax1.scatter(embs_base_2d[:, 0], embs_base_2d[:, 1], 
                  c=zones, cmap=cmap_rainbow, s=55, alpha=0.85, 
                  edgecolors='black', linewidths=0.4)
ax1.set_title('Figure 1 (a): Baseline — LLM Tokenizer Path\n'
              'General-purpose text token embeddings for pixel values\n'
              '(Clusters weak / mostly numerical)', fontsize=11, pad=10)
ax1.set_xlabel('Dim 1 (t-SNE/UMAP)')
ax1.set_ylabel('Dim 2 (t-SNE/UMAP)')
ax1.grid(True, alpha=0.3)

ax2 = fig.add_subplot(2, 2, 2)
sc2 = ax2.scatter(embs_new_2d[:, 0], embs_new_2d[:, 1], 
                  c=zones, cmap=cmap_rainbow, s=55, alpha=0.85, 
                  edgecolors='black', linewidths=0.4)
ax2.set_title('Figure 1 (b): Proposed — Pixel Embedding + Multi-dim Info\n'
              'Learned PixelInputEmbedding + INP (pos) + RGB Channel Encoding\n'
              '(Tighter, well-separated color zone clusters)', fontsize=11, pad=10)
ax2.set_xlabel('Dim 1 (t-SNE/UMAP)')
ax2.set_ylabel('Dim 2 (t-SNE/UMAP)')
ax2.grid(True, alpha=0.3)

# Shared colorbar for scatters
cbar_ax = fig.add_axes([0.92, 0.55, 0.015, 0.35])
cbar = fig.colorbar(sc2, cax=cbar_ax, ticks=range(N_ZONES))
cbar.set_ticklabels([f'{i}: {zone_names[i]}' for i in range(N_ZONES)])
cbar.set_label('Rainbow Color Zone', rotation=270, labelpad=15)

# ---- Bottom row: Cosine similarity heatmaps (reordered by zone) ----
def plot_sim_heatmap(embs, title, ax):
    embs_n = embs / np.linalg.norm(embs, axis=1, keepdims=True)
    sim = embs_n @ embs_n.T
    sns.heatmap(sim, ax=ax, cmap='RdBu_r', center=0, vmin=-0.3, vmax=1.0,
                cbar=False, xticklabels=False, yticklabels=False)
    ax.set_title(title, fontsize=10, pad=6)
    # Draw zone block boundaries
    for i in range(1, N_ZONES):
        pos = i * N_PER_ZONE
        ax.axhline(pos, color='white', linewidth=1.2, alpha=0.9)
        ax.axvline(pos, color='white', linewidth=1.2, alpha=0.9)
    ax.set_xlabel('Pixel index (grouped by zone)')
    ax.set_ylabel('Pixel index (grouped by zone)')

ax3 = fig.add_subplot(2, 2, 3)
plot_sim_heatmap(embs_base, 
                 'Figure 2 (a): Cosine Similarity (Baseline)\n'
                 'Weak block-diagonal structure — limited color zone awareness', ax3)

ax4 = fig.add_subplot(2, 2, 4)
plot_sim_heatmap(embs_new, 
                 'Figure 2 (b): Cosine Similarity (Proposed)\n'
                 'Strong block-diagonal — pixels in same rainbow zone have high similarity\n'
                 '(Direct evidence of improved semantic understanding via multi-dim encoding)', ax4)

plt.suptitle('Visualization Experiment: Proving Pixel Embedding Improves LLM Pixel Understanding\n'
             '140 Synthetic Pixels | 7 Rainbow Color Zones (ROYGBIV) | Direct Comparison of Representation Methods',
             fontsize=13, y=0.98)

# plt.tight_layout(rect=[0, 0, 0.91, 0.96])

# out_path = '/home/workdir/artifacts/pixel_embedding_viz_demo.png'
# plt.savefig(out_path, bbox_inches='tight', facecolor='white')
# print(f"\n[Saved] Visualization saved to: {out_path}")
# print("You can also change to .pdf for vector graphics in papers: plt.savefig(..., format='pdf')")



plt.tight_layout(rect=[0, 0, 0.91, 0.96])

# Robust output path: save to current working directory (portable across machines)
out_path = os.path.join(os.getcwd(), 'pixel_embedding_viz_demo.png')
os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)

plt.savefig(out_path, bbox_inches='tight', facecolor='white')
print(f"\n[Saved] Visualization saved to: {out_path}")
print("You can also change to .pdf for vector graphics in papers: plt.savefig(..., format='pdf')")



# Optional: show plot if running interactively
# plt.show()

print("\n=== How to adapt to your real models ===")
print("1. Copy the data generation + plotting functions into your project.")
print("2. Replace simulate_embeddings() with real loading + forward:")
print("   - Load tokenizer + raw_llm (as in your train_prefix_...py)")
print("   - Load pixel_emb = PixelInputEmbedding.load('models/pixel_emb_xxx_stage1_best.pt')")
print("   - (Optional) intra_pos = load_intra_pos(...)")
print("   - Then compute embs_base and embs_new exactly as described in the function docstring.")
print("3. For RGB channel encoding experiment: add a small nn.Embedding(3, H) and compare with/without it.")
print("4. Run on your real trained checkpoints (Stage 1 best recommended) for convincing results.")
print("5. For paper: use PDF output, adjust titles/captions in Chinese if needed, add your method name (LUMI/P²-LLM).")

print("\nDone. This demo shows the expected visual outcome: Proposed method yields clearly superior clustering and similarity structure.")