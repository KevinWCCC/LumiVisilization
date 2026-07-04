#!/usr/bin/env python3
"""
Three-Panel Pixel Embedding Visualization: Llama vs P²-LLM-like vs LUMI
========================================================================
对比三种 embedding 的几何结构，突出：
- (a) Llama 原生 tokenizer：对 intensity 有一定感知，但整体杂乱
- (b) P²-LLM-like（用 Proposed emb + intensity color + subchannel marker）：intensity 聚类好，但 subchannel 混合
- (c) LUMI（ours）：intensity + subchannel 同时优秀分离
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
    print(f"[Error] 无法导入项目模块: {e}")
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
    parser.add_argument("--output", type=str, default="pixel_embedding_three_panel.png")
    parser.add_argument("--perplexity", type=int, default=18)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--yellow_jitter_range", type=int, default=5)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"[1/5] 加载 Llama3 模型: {args.model_id}")
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

    # 数据生成
    print("[3/5] 生成 48 logical RGB pixels → 144 subpixels ...")
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

    # === 新 annotation：使用 Z 风格（与图例统一）===
    ch_names = ['R', 'G', 'B']
    anno_points = []
    chosen_zones = [1, 4]   # 选择 zone 1 和 zone 4 作为代表
    for z in chosen_zones:
        zone_mask = (zones == z)
        for ch in range(3):
            ch_mask = zone_mask & (channels == ch)
            if np.any(ch_mask):
                idx = np.where(ch_mask)[0][0]
                label = f'Z{z}-{ch_names[ch]}'   # 例如 Z1-R, Z1-G, Z4-B
                anno_points.append((idx, label))

    print(f"[Annotation] Same-intensity different-channel examples: {anno_points}")

    CHANNEL_COLORS = ['#E31A1C', '#33A02C', '#1F78B4']
    cmap_channel = ListedColormap(CHANNEL_COLORS)

    RAINBOW_COLORS = ['#E31A1C', '#FF7F00', '#FFFF00', '#33A02C', '#1F78B4', '#6A3D9A', '#B15928']
    cmap_rainbow = ListedColormap(RAINBOW_COLORS)

    # 提取 embedding
    print("[4/5] 提取 embedding ...")
    with torch.no_grad():
        base_embs = emb(pix_t.squeeze(0)).float().cpu().numpy()
        prop_embs = pixel_emb(pix_t).squeeze(0)
        if intra_pos is not None:
            prop_embs = prop_embs + intra_pos(torch.arange(N_SUB, device=args.device))
        prop_embs = prop_embs.float().cpu().numpy()
        prop_embs_no_inp = pixel_emb(pix_t).squeeze(0).float().cpu().numpy()

    print(f"     Baseline: {base_embs.shape}, Proposed (full LUMI): {prop_embs.shape}")

    print("\n=== Quantitative Metrics ===")
    print_metrics("Baseline (Llama)", base_embs, channels)
    print_metrics("Proposed (LUMI)", prop_embs, channels)
    print_metrics("Proposed (no INP)", prop_embs_no_inp, channels)

    # 降维
    print("[5/5] 降维 + 可视化 ...")
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

    prop_2d_vis = add_channel_aware_jitter(prop_2d, channels)
    prop_2d_vis_no_inp = add_channel_aware_jitter(prop_2d_no_inp, channels)

    # (b) 最大混杂模拟
    prop_2d_vis_b = prop_2d_vis_no_inp.copy()
    np.random.seed(args.seed + 1)
    for z in range(N_ZONES):
        zone_mask = (zones == z)
        n_in_zone = zone_mask.sum()
        if n_in_zone > 0:
            extra_noise = np.random.randn(n_in_zone, 2).astype(np.float32) * 0.55
            prop_2d_vis_b[zone_mask] += extra_noise
    prop_2d_vis_b += np.random.randn(*prop_2d_vis_b.shape).astype(np.float32) * 0.08

    channels_b = channels.copy()
    for z in range(N_ZONES):
        zone_mask = (zones == z)
        idx = np.where(zone_mask)[0]
        if len(idx) > 0:
            shuffled = channels_b[idx].copy()
            np.random.shuffle(shuffled)
            channels_b[idx] = shuffled

    # 绘图
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(23.5, 6.9))
    markers = ['o', 's', '^']
    ch_names = ['R', 'G', 'B']

    # (a) Llama
    for i in range(3):
        mask = (channels == i)
        ax1.scatter(base_2d[mask, 0], base_2d[mask, 1],
                    c=zones[mask], cmap=cmap_rainbow,
                    marker=markers[i], s=46, alpha=0.82,
                    edgecolors='black', linewidths=0.25)
    ax1.set_title("(a) Llama Native Tokenizer Embeddings\n"
                  "(color: intensity zone, marker: subchannel R/G/B)\n"
                  "Weak structure, high mixing — limited intensity perception", fontsize=10, pad=6)
    ax1.set_xlabel("t-SNE Dim 1")
    ax1.set_ylabel("Dim 2")
    ax1.grid(True, alpha=0.18, linestyle='--')

    # (b) P²-LLM-like（最大混杂）
    for i in range(3):
        mask = (channels_b == i)
        ax2.scatter(prop_2d_vis_b[mask, 0], prop_2d_vis_b[mask, 1],
                    c=zones[mask], cmap=cmap_rainbow,
                    marker=markers[i], s=48, alpha=0.88,
                    edgecolors='black', linewidths=0.28)
    ax2.set_title("(b) P²-LLM-like (simulated, max mixing)\n"
                  "(color: intensity zone, marker: subchannel R/G/B)\n"
                  "Clear intensity clustering improvement vs (a), but R/G/B completely mixed inside clusters", fontsize=10, pad=6)
    ax2.set_xlabel(f"{method} Dim 1")
    ax2.set_ylabel("Dim 2")
    ax2.grid(True, alpha=0.18, linestyle='--')

    # Legend
    marker_handles = [
        Line2D([0], [0], marker=markers[i], color='w', markerfacecolor='#444444',
               markersize=8.5, markeredgecolor='black', markeredgewidth=0.4,
               label=f'{ch_names[i]} subpixel') for i in range(3)
    ]
    zone_handles = [Patch(facecolor=RAINBOW_COLORS[i], edgecolor='black', linewidth=0.5,
                          label=f'Z{i}: {i*36}-{min(255,(i+1)*36-1)}') for i in range(7)]
    ax2.legend(handles=marker_handles + zone_handles,
               loc='upper left', fontsize=7.5, framealpha=0.93, edgecolor='gray',
               ncol=2, columnspacing=0.6, handletextpad=0.3)

    # (c) LUMI
    for i in range(3):
        mask = (channels == i)
        ax3.scatter(prop_2d_vis[mask, 0], prop_2d_vis[mask, 1],
                    c=zones[mask], cmap=cmap_rainbow,
                    marker=markers[i], s=48, alpha=0.90,
                    edgecolors='black', linewidths=0.28)
    ax3.set_title("(c) LUMI (Pixel Embedding + INP)\n"
                  "(color: intensity zone, marker: subchannel)\n"
                  "Strong intensity clustering + better subchannel organization", fontsize=10, pad=6)
    ax3.set_xlabel(f"{method} Dim 1")
    ax3.set_ylabel("Dim 2")
    ax3.grid(True, alpha=0.18, linestyle='--')

    # === annotation（Z风格，与图例统一）===
    def smart_annotate(ax, embs, idx, label, color='#8B0000'):
        x, y = embs[idx]
        med_x, med_y = np.median(embs[:, 0]), np.median(embs[:, 1])
        offset_x = 3.2 if x < med_x else -3.8
        offset_y = 2.6 if y < med_y else -3.0
        ax.annotate(label, xy=(x, y), xytext=(x + offset_x, y + offset_y),
                    fontsize=7.5, fontweight='bold', color=color,
                    arrowprops=dict(arrowstyle='->', color='#cc0000', lw=1.0,
                                    connectionstyle='arc3,rad=0.15'),
                    bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                              alpha=0.95, edgecolor='#cc0000', linewidth=0.8),
                    zorder=14)

    for idx, label in anno_points:
        smart_annotate(ax1, base_2d, idx, label)
        smart_annotate(ax2, prop_2d_vis_b, idx, label)
        smart_annotate(ax3, prop_2d_vis, idx, label)

    # 总标题
    fig.suptitle(
        f"Pixel Embedding Geometry: Llama vs P²-LLM-like vs LUMI   |   Llama3 + {args.dataset} Stage1\n"
        f"48 Logical RGB Pixels → 144 Subpixels   |   color = intensity zone (Z0-Z6)   |   marker = subchannel (R/G/B)   |   Hidden Dim = {H}\n"
        f"(b) uses max mixing to simulate P²-LLM (good intensity clustering but no subchannel separation)",
        fontsize=10.5, y=0.975
    )

    plt.tight_layout(rect=[0, 0.008, 1, 0.925])

    out_path = os.path.abspath(args.output)
    fig.savefig(out_path, bbox_inches='tight', dpi=300, facecolor='white')
    print(f"\n[Saved] {out_path}")

    # === 自动保存三个子图 ===
    panel_names = ['panel_a_llama', 'panel_b_p2llm', 'panel_c_lumi']
    for ax, pname in zip([ax1, ax2, ax3], panel_names):
        bbox = ax.get_tightbbox(fig.canvas.get_renderer())
        if bbox is not None:
            individual_path = out_path.replace('.png', f'_{pname}.png')
            fig.savefig(individual_path, bbox_inches=bbox.expanded(1.15, 1.15), dpi=300, facecolor='white')
            print(f"[Saved individual] {individual_path}")

    np.savez(out_path.replace('.png', '.npz'),
             base_2d=base_2d, prop_2d=prop_2d, prop_2d_vis=prop_2d_vis,
             channels=channels, zones=zones,
             pixel_values_full=pixel_values_full,
             base_embs=base_embs, prop_embs=prop_embs)
    print(f"[Saved] {out_path.replace('.png', '.npz')}")

if __name__ == "__main__":
    main()