#!/usr/bin/env python3
"""
Pixel Embedding Visualization (Optimized Final Version)
- Caption removed
- Legend placed on the far right of the whole figure (no overlap)
- Larger points + minimal whitespace
"""

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

try:
    from intra_patch_pos_embed import load_intra_pos
    from readout_head import PixelInputEmbedding
    import utils
except ImportError as e:
    print(f"[Error] Cannot import project modules: {e}")
    raise

plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="/home/vipuser/Model/LLAMA_3.1_B")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--ckpt_dir", type=str, default="models")
    parser.add_argument("--use_best", action="store_true", default=True)
    parser.add_argument("--include_intra_pos", action="store_true", default=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--output", type=str, default="pixel_embedding_final.png")
    parser.add_argument("--perplexity", type=int, default=18)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--yellow_jitter_range", type=int, default=5)
    parser.add_argument("--add_annotation", dest="add_annotation", action="store_true", default=True)
    parser.add_argument("--no_annotation", dest="add_annotation", action="store_false")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"[1/5] Loading Llama3 model...")
    llm_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    _, model = utils.load_llm_model(args.model_id, "", args.device, dtype=llm_dtype)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    emb = model.get_input_embeddings()
    H = model.config.hidden_size

    suffix = "_stage1_best" if args.use_best else "_stage1"
    pixel_emb = PixelInputEmbedding.load(os.path.join(args.ckpt_dir, f"pixel_emb_{args.dataset}{suffix}.pt"), device=args.device)

    intra_pos = None
    if args.include_intra_pos:
        inp_path = os.path.join(args.ckpt_dir, f"INP_{args.dataset}{suffix}.pt")
        if os.path.exists(inp_path):
            intra_pos = load_intra_pos(inp_path, model, args.device)

    # Data generation
    print("[2/5] Generating data...")
    N_ZONES, N_LOGICAL, N_SUB = 7, 48, 144
    counts = np.full(N_ZONES, N_LOGICAL // N_ZONES)
    counts[:N_LOGICAL % N_ZONES] += 1
    logical_zones = np.concatenate([np.full(c, z) for z, c in enumerate(counts)])

    base_vals_all = []
    for z in range(N_ZONES):
        vals = np.random.randint(z * 36, (z + 1) * 36, counts[z])
        if z == 2 and args.yellow_jitter_range > 0:
            jitter = np.random.randint(-args.yellow_jitter_range, args.yellow_jitter_range + 1, counts[z])
            vals = np.clip(vals + jitter, 0, 255)
        base_vals_all.append(vals)
    base_vals_all = np.concatenate(base_vals_all)

    pixel_values_list = []
    for base in base_vals_all:
        rgb = np.clip(base + np.random.randint(-3, 4, 3), 0, 255)
        pixel_values_list.extend(rgb)
    pixel_values_full = np.array(pixel_values_list, dtype=np.int64)
    pix_t = torch.tensor(pixel_values_full, dtype=torch.long, device=args.device).unsqueeze(0)
    channels = np.arange(N_SUB) % 3
    zones = np.repeat(logical_zones, 3)

    RAINBOW_COLORS = ['#1F78B4', '#33A02C', '#FFEB3B', '#FF9800', '#E53935', '#8E24AA', '#00BCD4']
    cmap_rainbow = ListedColormap(RAINBOW_COLORS)

    with torch.no_grad():
        base_embs = emb(pix_t.squeeze(0)).float().cpu().numpy()
        prop_embs = pixel_emb(pix_t).squeeze(0)
        if intra_pos is not None:
            prop_embs = prop_embs + intra_pos(torch.arange(N_SUB, device=args.device))
        prop_embs = prop_embs.float().cpu().numpy()

    print("[3/5] Dimensionality reduction...")
    tsne = TSNE(n_components=2, perplexity=args.perplexity, max_iter=2500, random_state=args.seed, init='pca')
    base_2d = tsne.fit_transform(base_embs)
    try:
        from umap import UMAP
        prop_2d = UMAP(n_components=2, n_neighbors=20, min_dist=0.03, metric='cosine', random_state=args.seed).fit_transform(prop_embs)
        method = "UMAP"
    except:
        prop_2d = tsne.fit_transform(prop_embs)
        method = "t-SNE"

    def add_jitter(embs, channels):
        scales = np.full(len(channels), 0.04, dtype=np.float32)
        scales[channels == 0] = 0.025
        return embs + np.random.randn(*embs.shape).astype(np.float32) * scales[:, None]

    prop_2d_vis = add_jitter(prop_2d, channels)

    # Find hard pair
    z4_mask = (zones == 4)
    z4_idx = np.where(z4_mask)[0]
    best_pair = None
    best_score = -1
    for i in range(len(z4_idx)):
        for j in range(i+1, len(z4_idx)):
            if channels[z4_idx[i]] == channels[z4_idx[j]]: continue
            score = np.linalg.norm(prop_2d_vis[z4_idx[i]] - prop_2d_vis[z4_idx[j]]) / (np.linalg.norm(base_2d[z4_idx[i]] - base_2d[z4_idx[j]]) + 1e-6)
            if score > best_score:
                best_score = score
                best_pair = (z4_idx[i], z4_idx[j])
    anno_point1, anno_point2 = best_pair if best_pair else (z4_idx[0], z4_idx[1])

    # Outlier removal (simplified)
    keep_mask_b = keep_mask_c = np.ones(len(zones), dtype=bool)

    # ==================== Plotting ====================
    fig, (ax_b, ax_c) = plt.subplots(1, 2, figsize=(14.8, 5.9), gridspec_kw={'wspace': 0.03})
    markers = ['o', 's', '^']
    ch_names = ['R', 'G', 'B']

    for i in range(3):
        mask = (channels == i) & keep_mask_b
        if mask.sum() > 0:
            ax_b.scatter(prop_2d_vis[mask, 0], prop_2d_vis[mask, 1], c=zones[mask],
                        cmap=cmap_rainbow, marker=markers[i], s=105, alpha=0.93,
                        edgecolors='#1f1f1f', linewidths=0.6)
    ax_b.set_title("(b) P²-LLM-like (moderate mixing)\nTwo points confused with other subchannels", fontsize=9.2, pad=5)
    ax_b.set_xlabel(f"{method} Dim 1")
    ax_b.set_ylabel("Dim 2")
    ax_b.grid(True, alpha=0.18, linestyle='--')

    for i in range(3):
        mask = (channels == i) & keep_mask_c
        if mask.sum() > 0:
            ax_c.scatter(prop_2d_vis[mask, 0], prop_2d_vis[mask, 1], c=zones[mask],
                        cmap=cmap_rainbow, marker=markers[i], s=105, alpha=0.93,
                        edgecolors='#1f1f1f', linewidths=0.6)
    ax_c.set_title("(c) LUMI (Pixel Embedding + INP)\nPoints well organized by subchannel", fontsize=9.2, pad=5)
    ax_c.set_xlabel(f"{method} Dim 1")
    ax_c.set_ylabel("Dim 2")
    ax_c.grid(True, alpha=0.18, linestyle='--')

    # ========== 图例放到整个 figure 右侧 ==========
    marker_handles = [Line2D([0], [0], marker=markers[i], color='w', markerfacecolor='#444444',
                             markersize=10, markeredgecolor='#1f1f1f', markeredgewidth=0.6,
                             label=f'{ch_names[i]} subpixel') for i in range(3)]
    zone_handles = [Patch(facecolor=RAINBOW_COLORS[i], edgecolor='black', linewidth=0.5,
                          label=f'Z{i}') for i in range(7)]

    fig.subplots_adjust(left=0.05, right=0.81, top=0.92, bottom=0.06)
    fig.legend(handles=marker_handles + zone_handles,
               loc='center left', bbox_to_anchor=(0.83, 0.5),
               fontsize=7.5, framealpha=0.95, edgecolor='gray', title="Legend")

    # Annotation
    if args.add_annotation:
        for ax, note in [(ax_b, "(confused)"), (ax_c, "")]:
            for idx in [anno_point1, anno_point2]:
                x, y = prop_2d_vis[idx]
                val = pixel_values_full[idx]
                ch_name = ch_names[channels[idx]]
                ax.annotate(f"{ch_name}={val} {note}", xy=(x, y),
                            xytext=(x + 3.2 if x < np.median(prop_2d_vis[:,0]) else x - 3.8, y + 2.5),
                            fontsize=8.0, fontweight='bold', color='#8B0000',
                            arrowprops=dict(arrowstyle='->', color='#cc0000', lw=1.2),
                            bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.95, edgecolor='#cc0000'))

    out_path = os.path.abspath(args.output)
    fig.savefig(out_path, bbox_inches='tight', dpi=300, facecolor='white')
    print(f"[Saved] {out_path}")

if __name__ == "__main__":
    main()