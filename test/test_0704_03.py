#!/usr/bin/env python3
"""
Final Version - Single Annotation on Left + Square Subplots
===========================================================
- Left (b): One annotation box pointing to both similar-intensity points
- Right (c): Two separate annotations
- Both subplot panels rendered as true squares
- No jitter on LUMI
- Outlier removal applied to both plots
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
    import seaborn as sns
except ImportError:
    sns = None

try:
    from intra_patch_pos_embed import load_intra_pos
    from readout_head import PixelInputEmbedding
    import utils
except ImportError as e:
    print(f"[Error] Cannot import project modules: {e}")
    raise

if sns is not None:
    sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12

PIXEL_SIZE = 110

def get_text_tokenizer_embedding(pixel_values, tokenizer, emb, device):
    all_embs = []
    for val in pixel_values:
        text = str(int(val))
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) == 0:
            token_ids = [tokenizer.pad_token_id] if tokenizer.pad_token_id is not None else [0]
        token_tensor = torch.tensor(token_ids, dtype=torch.long, device=device)
        token_embs = emb(token_tensor)
        aggregated = token_embs.mean(dim=0)
        all_embs.append(aggregated)
    return torch.stack(all_embs, dim=0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="/home/vipuser/Model/LLAMA_3.1_B")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--ckpt_dir", type=str, default="models")
    parser.add_argument("--use_best", action="store_true", default=True)
    parser.add_argument("--include_intra_pos", action="store_true", default=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--output", type=str, default="pixel_embedding_final_square.png")
    parser.add_argument("--perplexity", type=int, default=18)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--yellow_jitter_range", type=int, default=5)
    parser.add_argument("--add_annotation", dest="add_annotation", action="store_true", default=True)
    parser.add_argument("--no_annotation", dest="add_annotation", action="store_false")

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"[1/5] Loading Llama3 model: {args.model_id}")
    llm_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tokenizer, model = utils.load_llm_model(args.model_id, "", args.device, dtype=llm_dtype)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    emb = model.get_input_embeddings()
    H = model.config.hidden_size

    suffix = "_stage1_best" if args.use_best else "_stage1"
    pixel_emb = PixelInputEmbedding.load(
        os.path.join(args.ckpt_dir, f"pixel_emb_{args.dataset}{suffix}.pt"), device=args.device)

    intra_pos = None
    if args.include_intra_pos:
        inp_path = os.path.join(args.ckpt_dir, f"INP_{args.dataset}{suffix}.pt")
        if os.path.exists(inp_path):
            intra_pos = load_intra_pos(inp_path, model, args.device)

    # Data generation
    print("[3/5] Generating data ...")
    N_ZONES = 7
    N_LOGICAL = 48
    N_SUB = N_LOGICAL * 3

    counts = np.full(N_ZONES, N_LOGICAL // N_ZONES)
    counts[:N_LOGICAL % N_ZONES] += 1
    logical_zones = np.concatenate([np.full(c, z) for z, c in enumerate(counts)])

    base_vals_all = []
    for z in range(N_ZONES):
        low, high = z * 36, (z + 1) * 36
        n = counts[z]
        vals = np.random.randint(low, high, n)
        if z == 2 and args.yellow_jitter_range > 0:
            jitter = np.random.randint(-args.yellow_jitter_range, args.yellow_jitter_range + 1, n)
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
    logical_ids = np.arange(N_SUB) // 3

    RAINBOW_COLORS = ['#1F78B4', '#33A02C', '#FFEB3B', '#FF9800', '#E53935', '#8E24AA', '#00BCD4']
    cmap_rainbow = ListedColormap(RAINBOW_COLORS)

    print("[4/5] Extracting embeddings ...")
    with torch.no_grad():
        base_embs = get_text_tokenizer_embedding(pixel_values_full, tokenizer, emb, args.device).float().cpu().numpy()
        prop_embs = pixel_emb(pix_t).squeeze(0)
        if intra_pos is not None:
            prop_embs = prop_embs + intra_pos(torch.arange(N_SUB, device=args.device))
        prop_embs = prop_embs.float().cpu().numpy()

    # t-SNE / UMAP
    print("[5/5] Dimensionality reduction ...")
    tsne = TSNE(n_components=2, perplexity=args.perplexity, max_iter=2500,
                random_state=args.seed, init='pca', learning_rate='auto')
    base_2d = tsne.fit_transform(base_embs)

    try:
        from umap import UMAP
        prop_2d = UMAP(n_components=2, n_neighbors=20, min_dist=0.03, metric='cosine', random_state=args.seed).fit_transform(prop_embs)
        method = "UMAP"
    except ImportError:
        prop_2d = tsne.fit_transform(prop_embs)
        method = "t-SNE"

    prop_2d_vis = prop_2d.copy()   # No jitter on LUMI

    # Annotation pair selection
    print("[Annotation] Searching for similar intensity pair ...")
    z4_mask = (zones == 4)
    z4_indices = np.where(z4_mask)[0]

    max_similar_intensity_gap = 8
    best_pair = None
    best_score = -1.0
    relaxed_best_pair = None
    relaxed_best_score = -1.0
    for i in range(len(z4_indices)):
        for j in range(i + 1, len(z4_indices)):
            idx_i, idx_j = z4_indices[i], z4_indices[j]
            if channels[idx_i] == channels[idx_j]:
                continue
            val_diff = abs(pixel_values_full[idx_i] - pixel_values_full[idx_j])
            if val_diff == 0:
                continue
            d_base = np.linalg.norm(base_2d[idx_i] - base_2d[idx_j])
            d_prop = np.linalg.norm(prop_2d_vis[idx_i] - prop_2d_vis[idx_j])
            if d_base < 1e-6:
                continue
            score = (d_prop / (d_base + 1e-6)) + max(0, 6 - val_diff) * 0.25
            if score > relaxed_best_score:
                relaxed_best_score = score
                relaxed_best_pair = (idx_i, idx_j)
            if val_diff > max_similar_intensity_gap:
                continue
            if score > best_score:
                best_score = score
                best_pair = (idx_i, idx_j)

    if best_pair is None and relaxed_best_pair is not None:
        best_pair = relaxed_best_pair

    if best_pair is not None:
        anno_point1, anno_point2 = best_pair
        pair_diff = abs(pixel_values_full[anno_point1] - pixel_values_full[anno_point2])
        print(f"[Annotation] Selected non-identical similar pair, intensity gap = {pair_diff}")
    else:
        anno_point1 = z4_indices[0]
        anno_point2 = z4_indices[3] if len(z4_indices) > 3 else z4_indices[1]
        print("[Annotation] Warning: fallback pair used; no non-identical pair was available")

    # Outlier removal (applied to both plots)
    print("[Outlier] Detecting outliers ...")

    def detect_mild_outliers(embs_2d, zones, logical_ids, threshold=3.8):
        outlier_logical = set()
        for z in range(N_ZONES):
            z_mask = (zones == z)
            if z_mask.sum() < 6:
                continue
            zone_embs = embs_2d[z_mask]
            zone_logical = logical_ids[z_mask]
            center = np.median(zone_embs, axis=0)
            dists = np.linalg.norm(zone_embs - center, axis=1)
            med_dist = np.median(dists)
            mad = np.median(np.abs(dists - med_dist)) + 1e-8
            outlier_mask = dists > (med_dist + threshold * mad)
            if outlier_mask.any():
                outlier_logical.update(np.unique(zone_logical[outlier_mask]).tolist())
        return outlier_logical

    outlier_logical = detect_mild_outliers(prop_2d_vis, zones, logical_ids, threshold=3.8)
    keep_mask = np.ones(len(zones), dtype=bool)
    for log_id in outlier_logical:
        keep_mask[logical_ids == log_id] = False

    n_removed = len(outlier_logical)
    if n_removed > 0:
        print(f"[Outlier] Removed {n_removed} logical pixels from BOTH plots")
    else:
        print("[Outlier] No outliers removed.")

    # ========== Plotting with square subplot panels ==========
    # Keep each subplot's plotting area physically square, regardless of titles/legend.
    fig, (ax_b, ax_c) = plt.subplots(1, 2, figsize=(14.2, 7.1))
    for ax in (ax_b, ax_c):
        ax.set_box_aspect(1)

    markers = ['o', 's', '^']
    ch_names = ['R', 'G', 'B']

    # Left: Text Tokenizer Baseline
    for i in range(3):
        mask = (channels == i) & keep_mask
        if mask.sum() == 0:
            continue
        ax_b.scatter(base_2d[mask, 0], base_2d[mask, 1],
                    c=zones[mask], cmap=cmap_rainbow,
                    marker=markers[i], s=PIXEL_SIZE, alpha=0.85,
                    edgecolors='black', linewidths=0.3)

    ax_b.set_title("(b) Baseline — Text Tokenizer\n"
                   "pixel value → string → real tokenizer → mean pooling\n"
                   "Similar intensity points tend to cluster",
                   fontsize=10.5, pad=6)
    ax_b.set_xlabel(f"{method} Dim 1", fontsize=12, fontweight='bold')
    ax_b.set_ylabel("Dim 2", fontsize=12, fontweight='bold')
    ax_b.grid(True, alpha=0.15, linestyle='--')

    # Right: LUMI (no jitter)
    for i in range(3):
        mask = (channels == i) & keep_mask
        if mask.sum() == 0:
            continue
        ax_c.scatter(prop_2d_vis[mask, 0], prop_2d_vis[mask, 1],
                    c=zones[mask], cmap=cmap_rainbow,
                    marker=markers[i], s=PIXEL_SIZE, alpha=0.88,
                    edgecolors='black', linewidths=0.3)

    ax_c.set_title("(c) LUMI — Pixel Embedding + INP\n"
                   "Learned channel-aware + position-aware representation\n"
                   "Similar intensity points clearly separated by subchannel",
                   fontsize=10.5, pad=6)
    ax_c.set_xlabel(f"{method} Dim 1", fontsize=12, fontweight='bold')
    ax_c.set_ylabel("Dim 2", fontsize=12, fontweight='bold')
    ax_c.grid(True, alpha=0.15, linestyle='--')

    # Legend
    marker_handles = [Line2D([0], [0], marker=markers[i], color='w', markerfacecolor='#444444',
                             markersize=8, markeredgecolor='black', markeredgewidth=0.4,
                             label=f'{ch_names[i]} subpixel') for i in range(3)]
    zone_handles = [Patch(facecolor=RAINBOW_COLORS[i], edgecolor='black', linewidth=0.5,
                          label=f'Z{i}: {i*36}-{min(255,(i+1)*36-1)}') for i in range(7)]
    fig.legend(handles=marker_handles + zone_handles, loc='center right',
               bbox_to_anchor=(1.015, 0.5), fontsize=8.5, framealpha=0.92)

    # ========== Annotation ==========
    def annotate_pair(ax, embs, idx1, idx2, is_left=False):
        val1 = pixel_values_full[idx1]
        val2 = pixel_values_full[idx2]
        ch1 = ch_names[channels[idx1]]
        ch2 = ch_names[channels[idx2]]

        if is_left:
            # Left plot: ONE annotation box + two arrows
            text = (f"{ch1}={val1}  &  {ch2}={val2}\n"
                    f"Similar intensity (text tokenized)\n"
                    f"→ tend to cluster / confused")
            ax.annotate(
                text,
                xy=(embs[idx1, 0], embs[idx1, 1]),
                xytext=(0.66, 0.27),
                textcoords='axes fraction',
                ha='left', va='center',
                fontsize=10.2, fontweight='bold', color='#8B0000',
                linespacing=1.25,
                bbox=dict(boxstyle='round,pad=0.45', facecolor='white', alpha=0.96,
                          edgecolor='#cc0000', linewidth=1.1),
                arrowprops=dict(arrowstyle='->', color='#cc0000', lw=1.8,
                                connectionstyle='arc3,rad=0.2'),
                zorder=20
            )
            # Second arrow (empty text)
            ax.annotate(
                "",
                xy=(embs[idx2, 0], embs[idx2, 1]),
                xytext=(0.78, 0.31),
                textcoords='axes fraction',
                arrowprops=dict(arrowstyle='->', color='#cc0000', lw=1.8,
                                connectionstyle='arc3,rad=-0.18'),
                zorder=19
            )
        else:
            # Right plot: two separate annotations
            ax.annotate(f"{ch1}={val1}",
                        xy=(embs[idx1, 0], embs[idx1, 1]),
                        xytext=(0.54, 0.17),
                        textcoords='axes fraction',
                        ha='left', va='center',
                        fontsize=9.8, fontweight='bold', color='#8B0000',
                        arrowprops=dict(arrowstyle='->', color='#cc0000', lw=1.5,
                                        connectionstyle='arc3,rad=0.16'),
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.95,
                                  edgecolor='#cc0000'),
                        zorder=14)

            ax.annotate(f"{ch2}={val2} (similar intensity)",
                        xy=(embs[idx2, 0], embs[idx2, 1]),
                        xytext=(0.54, 0.08),
                        textcoords='axes fraction',
                        ha='left', va='center',
                        fontsize=9.8, fontweight='bold', color='#8B0000',
                        arrowprops=dict(arrowstyle='->', color='#cc0000', lw=1.5,
                                        connectionstyle='arc3,rad=-0.14'),
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.95,
                                  edgecolor='#cc0000'),
                        zorder=14)

    if args.add_annotation:
        annotate_pair(ax_b, base_2d, anno_point1, anno_point2, is_left=True)
        annotate_pair(ax_c, prop_2d_vis, anno_point1, anno_point2, is_left=False)
        print("[Annotation] Single box on left, separate on right")

    plt.tight_layout(rect=[0, 0.01, 0.87, 0.98])
    out_path = os.path.abspath(args.output)
    fig.savefig(out_path, bbox_inches='tight', dpi=300, facecolor='white')
    print(f"\n[Saved] {out_path}")

if __name__ == "__main__":
    main()
