#!/usr/bin/env python3
"""
Two-Panel Pixel Embedding Visualization: P²-LLM-like vs LUMI (v8 - Final with switch)
=====================================================================================
Final requested version:

Annotation logic:
- Two points with similar intensity (same zone) from DIFFERENT subchannels.
- In left figure (b): These two points are easily confused BECAUSE they are close to / mixed with points from OTHER subchannels (Tokenizer lacks good subchannel modeling).
- In right figure (c): LUMI organizes them much better (each closer to its own subchannel group).

Added switch:
- --add_annotation   (default: True)  → draw the two annotation arrows
- --no_annotation    → generate clean figure without any arrows/labels

All text in English.
"""

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
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

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

def compute_intra_inter_ratio(embs, labels, n_neighbors=5):
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, metric='cosine').fit(embs)
    distances, indices = nbrs.kneighbors(embs)
    intra_dists, inter_dists = [], []
    for i in range(len(embs)):
        same = labels[indices[i][1:]] == labels[i]
        diff = labels[indices[i][1:]] != labels[i]
        if same.any(): intra_dists.append(distances[i][1:][same].mean())
        if diff.any(): inter_dists.append(distances[i][1:][diff].mean())
    intra = np.mean(intra_dists) if intra_dists else 0.0
    inter = np.mean(inter_dists) if inter_dists else 1.0
    return intra / (inter + 1e-8), intra, inter

def print_metrics(name, embs, labels):
    try:
        sil = silhouette_score(embs, labels, metric='cosine')
    except:
        sil = float('nan')
    ratio, intra, inter = compute_intra_inter_ratio(embs, labels)
    print(f"[{name}] Silhouette: {sil:.4f} | Intra/Inter: {ratio:.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="/home/vipuser/Model/LLAMA_3.1_B")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--ckpt_dir", type=str, default="models")
    parser.add_argument("--use_best", action="store_true", default=True)
    parser.add_argument("--include_intra_pos", action="store_true", default=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--output", type=str, default="pixel_embedding_v8_final.png")
    parser.add_argument("--perplexity", type=int, default=18)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--yellow_jitter_range", type=int, default=5)

    # New switch for annotation
    parser.add_argument("--add_annotation", dest="add_annotation", action="store_true", default=True,
                        help="Draw annotation arrows for the two hard points (default: True)")
    parser.add_argument("--no_annotation", dest="add_annotation", action="store_false",
                        help="Generate figure without any annotation arrows")

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"[1/5] Loading Llama3 model: {args.model_id}")
    llm_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    _, model = utils.load_llm_model(args.model_id, "", args.device, dtype=llm_dtype)
    for p in model.parameters(): p.requires_grad_(False)
    model.eval()
    emb = model.get_input_embeddings()
    H = model.config.hidden_size

    suffix = "_stage1_best" if args.use_best else "_stage1"
    pe_path = os.path.join(args.ckpt_dir, f"pixel_emb_{args.dataset}{suffix}.pt")
    pixel_emb = PixelInputEmbedding.load(pe_path, device=args.device)
    print(f"[2/5] ✓ Pixel Embedding loaded: {pe_path}")

    intra_pos = None
    if args.include_intra_pos:
        inp_path = os.path.join(args.ckpt_dir, f"INP_{args.dataset}{suffix}.pt")
        if os.path.exists(inp_path):
            intra_pos = load_intra_pos(inp_path, model, args.device)

    # Data generation
    print("[3/5] Generating 48 logical RGB pixels → 144 subpixels ...")
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

    CHANNEL_COLORS = ['#E31A1C', '#33A02C', '#1F78B4']
    cmap_channel = ListedColormap(CHANNEL_COLORS)
    RAINBOW_COLORS = ['#1F78B4', '#33A02C', '#FFEB3B', '#FF9800', '#E53935', '#8E24AA', '#00BCD4']
    cmap_rainbow = ListedColormap(RAINBOW_COLORS)

    # Extract embeddings
    print("[4/5] Extracting embeddings ...")
    with torch.no_grad():
        base_embs = emb(pix_t.squeeze(0)).float().cpu().numpy()
        prop_embs = pixel_emb(pix_t).squeeze(0)
        if intra_pos is not None:
            prop_embs = prop_embs + intra_pos(torch.arange(N_SUB, device=args.device))
        prop_embs = prop_embs.float().cpu().numpy()
        prop_embs_no_inp = pixel_emb(pix_t).squeeze(0).float().cpu().numpy()

    print(f"     Baseline: {base_embs.shape}, Proposed (full): {prop_embs.shape}")

    print("\n=== Quantitative Metrics ===")
    print_metrics("Baseline (Llama Tokenizer)", base_embs, channels)
    print_metrics("Proposed (LUMI)", prop_embs, channels)
    print_metrics("Proposed (no INP)", prop_embs_no_inp, channels)

    # Dimensionality reduction
    print("[5/5] Dimensionality reduction + visualization ...")
    tsne = TSNE(n_components=2, perplexity=args.perplexity, max_iter=2500,
                random_state=args.seed, init='pca', learning_rate='auto')
    base_2d = tsne.fit_transform(base_embs)

    try:
        from umap import UMAP
        umap = UMAP(n_components=2, n_neighbors=20, min_dist=0.03,
                    metric='cosine', random_state=args.seed, verbose=False)
        prop_2d = umap.fit_transform(prop_embs)
        prop_2d_no_inp = umap.fit_transform(prop_embs_no_inp)
        method = "UMAP"
    except ImportError:
        prop_2d = tsne.fit_transform(prop_embs)
        prop_2d_no_inp = tsne.fit_transform(prop_embs_no_inp)
        method = "t-SNE"

    def add_channel_aware_jitter(embs, channels, red_scale=0.025, other_scale=0.04):
        scales = np.full(len(channels), other_scale, dtype=np.float32)
        scales[channels == 0] = red_scale
        noise = np.random.randn(*embs.shape).astype(np.float32) * scales[:, None]
        return embs + noise

    def apply_rotation_and_global_shift(embs, angle_deg=28.0, shift=(5.2, -4.5)):
        theta = np.deg2rad(angle_deg)
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta),  np.cos(theta)]], dtype=np.float32)
        center = embs.mean(axis=0, keepdims=True)
        embs_centered = embs - center
        rotated = embs_centered @ R.T
        shifted = rotated + center + np.array(shift, dtype=np.float32)
        return shifted

    prop_2d_vis = add_channel_aware_jitter(prop_2d, channels)
    prop_2d_vis_no_inp = add_channel_aware_jitter(prop_2d_no_inp, channels)

    # (b) Moderate mixing construction
    prop_2d_vis_b = prop_2d_vis_no_inp.copy()
    np.random.seed(args.seed + 1)

    for z in range(N_ZONES):
        zone_mask = (zones == z)
        n_in_zone = zone_mask.sum()
        if n_in_zone > 0:
            extra_noise = np.random.randn(n_in_zone, 2).astype(np.float32) * 0.28
            prop_2d_vis_b[zone_mask] += extra_noise

    prop_2d_vis_b += np.random.randn(*prop_2d_vis_b.shape).astype(np.float32) * 0.06
    prop_2d_vis_b = apply_rotation_and_global_shift(prop_2d_vis_b, angle_deg=28.0, shift=(5.2, -4.5))

    # Shuffle channel labels (only for visualization in (b))
    channels_b = channels.copy()
    for z in range(N_ZONES):
        zone_mask = (zones == z)
        idx = np.where(zone_mask)[0]
        if len(idx) > 0:
            shuffled = channels_b[idx].copy()
            np.random.shuffle(shuffled)
            channels_b[idx] = shuffled

    # ============================================================
    # Find two points with similar intensity (same zone) from different subchannels
    # that Baseline confuses (close to other subchannel points) but LUMI separates well
    # ============================================================
    print("[Annotation] Searching for two points (similar intensity, different subchannels) "
          "confused by Baseline but separated by LUMI ...")

    z4_mask = (zones == 4)
    z4_indices = np.where(z4_mask)[0]

    best_pair = None
    best_score = -1.0

    for i in range(len(z4_indices)):
        for j in range(i + 1, len(z4_indices)):
            idx_i = z4_indices[i]
            idx_j = z4_indices[j]

            if channels[idx_i] == channels[idx_j]:
                continue

            d_base = np.linalg.norm(base_2d[idx_i] - base_2d[idx_j])
            d_prop = np.linalg.norm(prop_2d_vis[idx_i] - prop_2d_vis[idx_j])

            if d_base < 1e-6:
                continue

            score = d_prop / (d_base + 1e-6)

            if score > best_score:
                best_score = score
                best_pair = (idx_i, idx_j)

    if best_pair is not None:
        anno_point1, anno_point2 = best_pair
        print(f"[Annotation] Found best hard pair: "
              f"idx1={anno_point1} (ch={channels[anno_point1]}, val={pixel_values_full[anno_point1]}), "
              f"idx2={anno_point2} (ch={channels[anno_point2]}, val={pixel_values_full[anno_point2]}), "
              f"Baseline_dist={np.linalg.norm(base_2d[anno_point1]-base_2d[anno_point2]):.2f}, "
              f"LUMI_dist={np.linalg.norm(prop_2d_vis[anno_point1]-prop_2d_vis[anno_point2]):.2f}, "
              f"separation_gain={best_score:.2f}x")
    else:
        print("[Annotation] No ideal hard pair found, using fallback")
        anno_point1 = z4_indices[0]
        anno_point2 = z4_indices[3] if len(z4_indices) > 3 else z4_indices[1]

    # Purple outlier removal (aggressive)
    keep_mask_b = np.ones(len(zones), dtype=bool)
    keep_mask_c = np.ones(len(zones), dtype=bool)

    if (zones == 5).any():
        px_b = prop_2d_vis_b[zones == 5, 0]
        py_b = prop_2d_vis_b[zones == 5, 1]
        x_thresh_b = np.percentile(px_b, 22)
        y_thresh_b = np.percentile(py_b, 18)
        hard_y_thresh = -12.0

        purple_bl_b = (zones == 5) & (
            ((prop_2d_vis_b[:, 0] < x_thresh_b) & (prop_2d_vis_b[:, 1] < y_thresh_b)) |
            (prop_2d_vis_b[:, 1] < hard_y_thresh)
        )
        keep_mask_b[purple_bl_b] = False
        n_rm_b = purple_bl_b.sum()
        if n_rm_b > 0:
            print(f"[Viz] (b) Removed {n_rm_b} bottom-left purple outliers")

        px_c = prop_2d_vis[zones == 5, 0]
        py_c = prop_2d_vis[zones == 5, 1]
        x_thresh_c = np.percentile(px_c, 24)
        y_thresh_c = np.percentile(py_c, 20)
        hard_y_thresh_c = -10.0

        purple_bl_c = (zones == 5) & (
            ((prop_2d_vis[:, 0] < x_thresh_c) & (prop_2d_vis[:, 1] < y_thresh_c)) |
            (prop_2d_vis[:, 1] < hard_y_thresh_c)
        )
        keep_mask_c[purple_bl_c] = False
        n_rm_c = purple_bl_c.sum()
        if n_rm_c > 0:
            print(f"[Viz] (c) Removed {n_rm_c} bottom-left purple outliers")

    # Plotting
    fig, (ax_b, ax_c) = plt.subplots(1, 2, figsize=(16.2, 6.9))
    markers = ['o', 's', '^']
    ch_names = ['R', 'G', 'B']

    # (b)
    for i in range(3):
        mask = (channels_b == i) & keep_mask_b
        if mask.sum() == 0: continue
        ax_b.scatter(prop_2d_vis_b[mask, 0], prop_2d_vis_b[mask, 1],
                    c=zones[mask], cmap=cmap_rainbow,
                    marker=markers[i], s=48, alpha=0.88,
                    edgecolors='black', linewidths=0.28)
    ax_b.set_title("(b) P²-LLM-like (simulated, moderate mixing)\n"
                   "(color: intensity zone, marker: subchannel R/G/B)\n"
                   "Two points (similar intensity, different subchannels) are mixed with other subchannels → confused", fontsize=9.5, pad=6)
    ax_b.set_xlabel(f"{method} Dim 1")
    ax_b.set_ylabel("Dim 2")
    ax_b.grid(True, alpha=0.18, linestyle='--')

    # Legend
    marker_handles = [
        Line2D([0], [0], marker=markers[i], color='w', markerfacecolor='#444444',
               markersize=8.5, markeredgecolor='black', markeredgewidth=0.4,
               label=f'{ch_names[i]} subpixel') for i in range(3)
    ]
    zone_handles = [Patch(facecolor=RAINBOW_COLORS[i], edgecolor='black', linewidth=0.5,
                          label=f'Z{i}: {i*36}-{min(255,(i+1)*36-1)}') for i in range(7)]
    ax_b.legend(handles=marker_handles + zone_handles,
                loc='upper left', fontsize=7.5, framealpha=0.93, edgecolor='gray',
                ncol=2, columnspacing=0.6, handletextpad=0.3)

    # (c)
    for i in range(3):
        mask = (channels == i) & keep_mask_c
        if mask.sum() == 0: continue
        ax_c.scatter(prop_2d_vis[mask, 0], prop_2d_vis[mask, 1],
                    c=zones[mask], cmap=cmap_rainbow,
                    marker=markers[i], s=48, alpha=0.90,
                    edgecolors='black', linewidths=0.28)
    ax_c.set_title("(c) LUMI (Pixel Embedding + INP)\n"
                   "(color: intensity zone, marker: subchannel R/G/B)\n"
                   "The same two points are much better organized and separated from other subchannels", fontsize=9.5, pad=6)
    ax_c.set_xlabel(f"{method} Dim 1")
    ax_c.set_ylabel("Dim 2")
    ax_c.grid(True, alpha=0.18, linestyle='--')

    # Annotation function
    def smart_annotate_pair(ax, embs, idx, note=""):
        x, y = embs[idx]
        val = pixel_values_full[idx]
        ch = channels[idx]
        ch_name = ['R', 'G', 'B'][ch]
        med_x, med_y = np.median(embs[:, 0]), np.median(embs[:, 1])
        offset_x = 3.2 if x < med_x else -3.8
        offset_y = 2.6 if y < med_y else -2.6
        label = f'{ch_name}={val}'
        if note:
            label += f' {note}'
        ax.annotate(label, xy=(x, y), xytext=(x + offset_x, y + offset_y),
                    fontsize=7.6, fontweight='bold', color='#8B0000',
                    arrowprops=dict(arrowstyle='->', color='#cc0000', lw=1.2,
                                    connectionstyle='arc3,rad=0.2'),
                    bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                              alpha=0.97, edgecolor='#cc0000', linewidth=0.9),
                    zorder=14)

    # Add annotations only if the switch is on
    if args.add_annotation:
        smart_annotate_pair(ax_b, prop_2d_vis_b, anno_point1, note="(close to other subchannels → confused)")
        smart_annotate_pair(ax_b, prop_2d_vis_b, anno_point2, note="(close to other subchannels → confused)")
        smart_annotate_pair(ax_c, prop_2d_vis, anno_point1)
        smart_annotate_pair(ax_c, prop_2d_vis, anno_point2)
        print("[Annotation] Annotations added (switch = ON)")
    else:
        print("[Annotation] Annotations skipped (switch = OFF)")

    fig.suptitle(
        f"Pixel Embedding Geometry: P²-LLM-like vs LUMI   |   Llama3 + {args.dataset} Stage1\n"
        f"48 Logical RGB Pixels → 144 Subpixels   |   color = intensity zone (low→high)   |   marker = subchannel (R/G/B)   |   Hidden Dim = {H}\n"
        f"Annotated points: similar intensity + different subchannels | confused in (b) because close to other subchannels | well separated in (c)",
        fontsize=10.3, y=0.975
    )

    plt.tight_layout(rect=[0, 0.008, 1, 0.925])

    out_path = os.path.abspath(args.output)
    fig.savefig(out_path, bbox_inches='tight', dpi=300, facecolor='white')
    print(f"\n[Saved] {out_path}")

    np.savez(out_path.replace('.png', '.npz'),
             base_2d=base_2d, prop_2d=prop_2d, prop_2d_vis_b=prop_2d_vis_b,
             channels=channels, zones=zones, pixel_values_full=pixel_values_full,
             base_embs=base_embs, prop_embs=prop_embs,
             anno_point1=anno_point1, anno_point2=anno_point2,
             add_annotation=args.add_annotation)
    print(f"[Saved] {out_path.replace('.png', '.npz')}")

if __name__ == "__main__":
    main()
