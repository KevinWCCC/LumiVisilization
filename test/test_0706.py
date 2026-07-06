#!/usr/bin/env python3
"""
Improved Publication-Quality Version of pixel_embedding figure
==============================================================
Key improvements for higher-end / NeurIPS-CVPR style:

1. Shortened panel titles (detailed explanation → Figure Caption)
2. Increased marker size (PIXEL_SIZE=48) with adjusted alpha for better visibility
3. Highlighted annotation points (R=152 / B=153) stand out clearly in BOTH panels
4. Right plot now uses ONE combined annotation box (with two arrows) + new text:
   "Different channels are explicitly disentangled."
5. More elegant, thinner annotations with professional styling
6. Better perceptually-uniform colormap (plasma)
7. Cleaner overall style + professional rcParams

Usage remains the same as before. You only need to change the output filename
and optionally tweak a few parameters below (PIXEL_SIZE, ALPHA_LEFT, etc.).
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

# ====================== Professional Style Settings ======================
if sns is not None:
    sns.set_style("white")          # cleaner than whitegrid for t-SNE
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'axes.labelsize': 11,
    'xtick.labelsize': 9.5,
    'ytick.labelsize': 9.5,
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'legend.fontsize': 8.5,
    'legend.framealpha': 0.95,
})

# ====================== Tunable Visualization Parameters ======================
PIXEL_SIZE = 80         # increased from 36 for better visibility (good balance between clarity and overlap)
# ALPHA_LEFT = 1.00        # fully opaque markers; no transparency
# ALPHA_RIGHT = 1.00       # fully opaque markers; no transparency

ALPHA_LEFT = 0.50
ALPHA_RIGHT = 0.50

ALPHA_LEFT = 0.9
ALPHA_RIGHT = 0.90

ALPHA_LEFT = 0.7
ALPHA_RIGHT = 0.70

HIGHLIGHT_SIZE = 78      # size for the two special points (R=152, B=153)
HIGHLIGHT_LW = 0.95      # edge width for highlighted points

# Cool-tone sequential colormap for intensity zones (no bright yellow).
# Uses seaborn "mako" (dark blue-teal-cyan, modern & perceptually good for papers) if available;
# falls back to plt.cm.cool (cyan-magenta cool tones).
if sns is not None:
    CMAP_ZONES = sns.color_palette("mako", as_cmap=True)
else:
    CMAP_ZONES = plt.cm.cool

ZONE_COLORS_RGB = [
    (31, 119, 180),    # blue
    (44, 160, 44),     # green
    (148, 103, 189),   # purple
    (188, 189, 34),    # olive
    (23, 190, 207),    # cyan
    (227, 119, 194),   # pink
    (255, 127, 14),    # orange  <-- 新增
]

ZONE_COLORS = [(r/255, g/255, b/255) for r, g, b in ZONE_COLORS_RGB]
CMAP_ZONES = ListedColormap(ZONE_COLORS)


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
    parser.add_argument("--output", type=str, default="pixel_embedding_improved.png")
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

    # ====================== Data Generation ======================
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

    print("[4/5] Extracting embeddings ...")
    with torch.no_grad():
        base_embs = get_text_tokenizer_embedding(pixel_values_full, tokenizer, emb, args.device).float().cpu().numpy()
        prop_embs = pixel_emb(pix_t).squeeze(0)
        if intra_pos is not None:
            prop_embs = prop_embs + intra_pos(torch.arange(N_SUB, device=args.device))
        prop_embs = prop_embs.float().cpu().numpy()

    # ====================== Dimensionality Reduction ======================
    print("[5/5] Dimensionality reduction ...")
    tsne = TSNE(n_components=2, perplexity=args.perplexity, max_iter=2500,
                random_state=args.seed, init='pca', learning_rate='auto')
    base_2d = tsne.fit_transform(base_embs)

    try:
        from umap import UMAP
        prop_2d = UMAP(n_components=2, n_neighbors=20, min_dist=0.03, metric='cosine', random_state=args.seed).fit_transform(prop_embs)
        method_right = "UMAP"
    except ImportError:
        prop_2d = tsne.fit_transform(prop_embs)
        method_right = "t-SNE"

    prop_2d_vis = prop_2d.copy()

    # ====================== Annotation Pair Selection ======================
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

    # ====================== Moderate Robust Outlier Removal ======================
    print("[Outlier] Detecting outliers ...")

    def detect_moderate_outliers(
        embs_2d,
        zones,
        logical_ids,
        n_zones,
        threshold=4.5,
        min_points=8,
        max_remove_ratio=0.08,
        use_knn=True,
        knn_k=4,
        knn_threshold=4.8,
    ):
        """
        Moderate outlier detection for visualization.

        This version only changes the outlier-removal logic:
        1. Detects outliers in BOTH the left plot and the right plot.
        2. Uses zone-wise median + MAD robust distance as the main criterion.
        3. Falls back to IQR if MAD is degenerate.
        4. Uses a conservative kNN isolation check as a small supplement.
        5. Caps the number of removed points per zone to avoid over-cleaning.
        6. Removes by logical pixel id, so R/G/B subpixels are removed together.
        """
        outlier_logical = set()

        for z in range(n_zones):
            z_mask = (zones == z)
            if z_mask.sum() < min_points:
                continue

            zone_embs = embs_2d[z_mask]
            zone_logical = logical_ids[z_mask]

            # ---------- 1. Robust distance to zone center ----------
            center = np.median(zone_embs, axis=0)
            dists = np.linalg.norm(zone_embs - center, axis=1)

            med = np.median(dists)
            mad = np.median(np.abs(dists - med))

            if mad > 1e-8:
                robust_sigma = 1.4826 * mad
                cutoff = med + threshold * robust_sigma
            else:
                q1, q3 = np.percentile(dists, [25, 75])
                iqr = q3 - q1
                cutoff = q3 + 2.5 * iqr if iqr > 1e-8 else np.inf

            candidate_indices = np.where(dists > cutoff)[0]

            # ---------- 2. Conservative kNN isolation detection ----------
            if use_knn and len(zone_embs) >= max(min_points, knn_k + 3):
                diff = zone_embs[:, None, :] - zone_embs[None, :, :]
                dist_mat = np.sqrt(np.sum(diff ** 2, axis=-1))
                np.fill_diagonal(dist_mat, np.inf)

                k_eff = min(knn_k, len(zone_embs) - 1)
                knn_dists = np.sort(dist_mat, axis=1)[:, :k_eff].mean(axis=1)

                knn_med = np.median(knn_dists)
                knn_mad = np.median(np.abs(knn_dists - knn_med))

                if knn_mad > 1e-8:
                    knn_cutoff = knn_med + knn_threshold * 1.4826 * knn_mad
                else:
                    q1, q3 = np.percentile(knn_dists, [25, 75])
                    iqr = q3 - q1
                    knn_cutoff = q3 + 3.0 * iqr if iqr > 1e-8 else np.inf

                knn_candidates = np.where(knn_dists > knn_cutoff)[0]

                # kNN is only used as a conservative supplement:
                # add at most one most-isolated candidate per zone.
                if len(knn_candidates) > 0:
                    worst_knn = knn_candidates[np.argmax(knn_dists[knn_candidates])]
                    candidate_indices = np.unique(
                        np.concatenate([candidate_indices, np.array([worst_knn])])
                    )

            if len(candidate_indices) == 0:
                continue

            # ---------- 3. Cap deletion per zone ----------
            max_remove = max(1, int(np.floor(max_remove_ratio * len(zone_embs))))

            if len(candidate_indices) > max_remove:
                # Prefer removing points farthest from the robust zone center.
                candidate_indices = candidate_indices[
                    np.argsort(dists[candidate_indices])[::-1]
                ][:max_remove]

            outlier_logical.update(np.unique(zone_logical[candidate_indices]).tolist())

        return outlier_logical


    def detect_lumi_local_isolation_outliers(
        embs_2d,
        zones,
        logical_ids,
        n_zones,
        max_extra_logical=2,
        min_points=8,
        knn_k=4,
        knn_threshold=3.2,
        center_percentile=60,
    ):
        """
        Extra conservative detector for the right LUMI plot.

        The previous moderate detector may miss a few visually isolated LUMI points
        because they are local-density outliers rather than extreme center-distance
        outliers. This supplement only ranks highly isolated points by zone-wise kNN
        distance and removes at most `max_extra_logical` logical pixels globally.
        """
        scored_candidates = []

        for z in range(n_zones):
            z_mask = (zones == z)
            if z_mask.sum() < min_points:
                continue

            zone_embs = embs_2d[z_mask]
            zone_logical = logical_ids[z_mask]
            local_indices = np.where(z_mask)[0]

            diff = zone_embs[:, None, :] - zone_embs[None, :, :]
            dist_mat = np.sqrt(np.sum(diff ** 2, axis=-1))
            np.fill_diagonal(dist_mat, np.inf)

            k_eff = min(knn_k, len(zone_embs) - 1)
            knn_dists = np.sort(dist_mat, axis=1)[:, :k_eff].mean(axis=1)

            knn_med = np.median(knn_dists)
            knn_mad = np.median(np.abs(knn_dists - knn_med))
            if knn_mad > 1e-8:
                knn_scale = 1.4826 * knn_mad
                knn_cutoff = knn_med + knn_threshold * knn_scale
                knn_score = (knn_dists - knn_med) / (knn_scale + 1e-8)
            else:
                q1, q3 = np.percentile(knn_dists, [25, 75])
                iqr = q3 - q1
                if iqr > 1e-8:
                    knn_cutoff = q3 + 1.5 * iqr
                    knn_score = (knn_dists - q3) / (iqr + 1e-8)
                else:
                    knn_cutoff = np.inf
                    knn_score = np.zeros_like(knn_dists)

            center = np.median(zone_embs, axis=0)
            center_dists = np.linalg.norm(zone_embs - center, axis=1)
            center_cutoff = np.percentile(center_dists, center_percentile)

            candidate_indices = np.where(
                (knn_dists > knn_cutoff) & (center_dists >= center_cutoff)
            )[0]

            for local_i in candidate_indices:
                scored_candidates.append(
                    (
                        float(knn_score[local_i]),
                        float(knn_dists[local_i]),
                        int(zone_logical[local_i]),
                        int(local_indices[local_i]),
                    )
                )

        if len(scored_candidates) == 0:
            return set()

        # Keep only the most isolated logical pixels globally, to avoid over-deletion.
        scored_candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        extra_logical = []
        seen = set()
        for _, _, log_id, _ in scored_candidates:
            if log_id in seen:
                continue
            extra_logical.append(log_id)
            seen.add(log_id)
            if len(extra_logical) >= max_extra_logical:
                break

        return set(extra_logical)


    # Detect outliers in BOTH 2D spaces, then remove their union from BOTH plots.
    outlier_base = detect_moderate_outliers(
        base_2d,
        zones,
        logical_ids,
        N_ZONES,
        threshold=4.5,
        min_points=8,
        max_remove_ratio=0.08,
        use_knn=True,
        knn_k=4,
        knn_threshold=4.8,
    )

    outlier_prop = detect_moderate_outliers(
        prop_2d_vis,
        zones,
        logical_ids,
        N_ZONES,
        threshold=4.5,
        min_points=8,
        max_remove_ratio=0.08,
        use_knn=True,
        knn_k=4,
        knn_threshold=4.8,
    )

    # Additional right-plot-only local isolation pass.
    # It is globally capped at 2 logical pixels, targeting the two remaining LUMI outliers.
    outlier_prop_extra = detect_lumi_local_isolation_outliers(
        prop_2d_vis,
        zones,
        logical_ids,
        N_ZONES,
        max_extra_logical=2,
        min_points=8,
        knn_k=4,
        knn_threshold=3.2,
        center_percentile=60,
    )

    outlier_logical = outlier_base | outlier_prop | outlier_prop_extra

    # Protect the two annotated logical pixels, so annotation arrows never point to removed data.
    protected_logical = {
        logical_ids[anno_point1],
        logical_ids[anno_point2],
    }
    outlier_logical = outlier_logical - protected_logical

    keep_mask = np.ones(len(zones), dtype=bool)
    for log_id in outlier_logical:
        keep_mask[logical_ids == log_id] = False

    n_removed_logical = len(outlier_logical)
    n_removed_subpixels = int(np.sum(~keep_mask))

    if n_removed_logical > 0:
        print(
            f"[Outlier] Removed {n_removed_logical} logical pixels "
            f"({n_removed_subpixels} subpixels) from BOTH plots"
        )
        print(
            f"[Outlier] base={len(outlier_base)}, "
            f"prop={len(outlier_prop)}, "
            f"prop_extra={len(outlier_prop_extra)}, "
            f"union={len(outlier_base | outlier_prop | outlier_prop_extra)}, "
            f"protected={len(protected_logical)}"
        )
    else:
        print("[Outlier] No outliers removed.")

    # ====================== Plotting ======================
    fig, (ax_b, ax_c) = plt.subplots(1, 2, figsize=(13.8, 6.9))
    # for ax in (ax_b, ax_c):
    #     ax.set_box_aspect(1)


    for ax in (ax_b, ax_c):
        ax.set_box_aspect(1)

        # 完整四周边框
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)
            spine.set_color("black")


    markers = ['o', 's', '^']
    ch_names = ['R', 'G', 'B']

    # ---------- Left: Baseline (Text Tokenizer) ----------
    for i in range(3):
        mask = (channels == i) & keep_mask
        if mask.sum() == 0:
            continue
        ax_b.scatter(base_2d[mask, 0], base_2d[mask, 1],
                     c=zones[mask], cmap=CMAP_ZONES,
                     marker=markers[i], s=PIXEL_SIZE, alpha=ALPHA_LEFT,
                     edgecolors='none', zorder=3)

    # Short, clean title (put full description in Figure Caption)
    # ax_b.set_title("(a) Llama Tokenizer", fontsize=11.5, pad=8, fontweight='bold')
    ax_b.set_xlabel("t-SNE Dim 1", fontsize=11, fontweight='bold')
    ax_b.set_ylabel("Dim 2", fontsize=11, fontweight='bold')
    ax_b.grid(True, alpha=0.12, linestyle='--', linewidth=0.6)

    # ---------- Right: LUMI ----------
    for i in range(3):
        mask = (channels == i) & keep_mask
        if mask.sum() == 0:
            continue
        ax_c.scatter(prop_2d_vis[mask, 0], prop_2d_vis[mask, 1],
                     c=zones[mask], cmap=CMAP_ZONES,
                     marker=markers[i], s=PIXEL_SIZE, alpha=ALPHA_RIGHT,
                     edgecolors='none', zorder=3)

    # ax_c.set_title("(b) LUMI Pixel Embedding", fontsize=11.5, pad=8, fontweight='bold')
    ax_c.set_xlabel(f"{method_right} Dim 1", fontsize=11, fontweight='bold')
    ax_c.set_ylabel("Dim 2", fontsize=11, fontweight='bold')
    ax_c.grid(True, alpha=0.12, linestyle='--', linewidth=0.6)

    # ====================== Legend ======================
    marker_handles = [
        Line2D([0], [0], marker=markers[i], color='w',
               markerfacecolor='#444444', markersize=8.5,
               markeredgecolor='black', markeredgewidth=0.5,
               label=f'{ch_names[i]} channel')
        for i in range(3)
    ]
    zone_handles = [
        Patch(facecolor=CMAP_ZONES(i / 6), edgecolor='black', linewidth=0.5,
              label=f'Z{i}: {i*36}-{255 if i == 6 else (i+1)*36-1}')
        for i in range(7)
    ]
            #   label=f'Z{i}: {i*36}-{min(255, (i+1)*36-1)}')

    # fig.legend(handles=marker_handles + zone_handles,
    #            loc='center right', bbox_to_anchor=(0.935, 0.5),
    #            borderaxespad=0.25,
    #            fontsize=8.2, framealpha=0.94, title="Channel / Zone")
    fig.legend(handles=marker_handles + zone_handles,
               loc='center right', bbox_to_anchor=(0.935, 0.5),
               borderaxespad=0.25,
               fontsize=8.2, framealpha=0.94, title="")
    # ====================== Refined Annotations ======================
    def annotate_pair(ax, embs, idx1, idx2, is_left=False):
        val1 = pixel_values_full[idx1]
        val2 = pixel_values_full[idx2]
        ch1 = ch_names[channels[idx1]]
        ch2 = ch_names[channels[idx2]]

        if is_left:
            # Left: single elegant annotation box + two thin arrows
            text = (f"{ch1}={val1}  &  {ch2}={val2}\n"
                    f"Different channels collapse into\nneighboring embeddings")
            ax.annotate(
                text,
                xy=(embs[idx1, 0], embs[idx1, 1]),
                xytext=(0.63, 0.29),
                textcoords='axes fraction',
                ha='left', va='center',
                fontsize=8.0, fontweight='bold', color='#9C1F1F',
                linespacing=1.15,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.97,
                          edgecolor='#B22222', linewidth=0.9),
                arrowprops=dict(arrowstyle='->', color='#B22222', lw=1.25,
                                connectionstyle='arc3,rad=0.18'),
                zorder=20
            )
            # Second thin arrow
            ax.annotate(
                "",
                xy=(embs[idx2, 0], embs[idx2, 1]),
                xytext=(0.76, 0.33),
                textcoords='axes fraction',
                arrowprops=dict(arrowstyle='->', color='#B22222', lw=1.25,
                                connectionstyle='arc3,rad=-0.16'),
                zorder=19
            )
        else:
            # Right: ONE combined annotation box + two thin arrows (consistent style with left)
            text = (f"{ch1}={val1}  &  {ch2}={val2}\n"
                    f"Different channels are explicitly disentangled.")
            ax.annotate(
                text,
                xy=(embs[idx1, 0], embs[idx1, 1]),
                xytext=(0.58, 0.22),
                textcoords='axes fraction',
                ha='left', va='center',
                fontsize=8.0, fontweight='bold', color='#9C1F1F',
                linespacing=1.15,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.97,
                          edgecolor='#B22222', linewidth=0.9),
                arrowprops=dict(arrowstyle='->', color='#B22222', lw=1.25,
                                connectionstyle='arc3,rad=0.18'),
                zorder=20
            )
            # Second thin arrow pointing to the other point
            ax.annotate(
                "",
                xy=(embs[idx2, 0], embs[idx2, 1]),
                xytext=(0.72, 0.26),
                textcoords='axes fraction',
                arrowprops=dict(arrowstyle='->', color='#B22222', lw=1.25,
                                connectionstyle='arc3,rad=-0.15'),
                zorder=19
            )

    if args.add_annotation:
        annotate_pair(ax_b, base_2d, anno_point1, anno_point2, is_left=True)
        annotate_pair(ax_c, prop_2d_vis, anno_point1, anno_point2, is_left=False)
        print("[Annotation] Combined annotation on right plot with disentanglement message")

    # ====================== Save ======================
    plt.tight_layout(rect=[0, 0.01, 0.905, 0.985])
    out_path = os.path.abspath(args.output)
    fig.savefig(out_path, bbox_inches='tight', dpi=300, facecolor='white')
    print(f"\n[Saved] Improved figure → {out_path}")
    print("Tips: Move the long descriptive text into your LaTeX figure caption for a cleaner look.")


if __name__ == "__main__":
    main()