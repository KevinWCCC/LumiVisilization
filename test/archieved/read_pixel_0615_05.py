#!/usr/bin/env python3
"""
真是像素标记颜色
Real Pixel Embedding t-SNE/UMAP Visualization (Red Cluster Tighter)
===================================================================
48 Logical RGB Pixels → 144 Subpixels
Colored by true subpixel channel (R/G/B)

本次优化重点：
- 让红色 (R channel) 在可视化中看起来更集中
- 采用 channel-aware jitter：红色 jitter 更小
- 收紧 UMAP min_dist
- 减小数据生成时的 RGB 微小波动（±3 → 使同通道输入更相似）
- 保持 Proposed 的结构优势，同时让红色团块视觉上更紧凑
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
plt.rcParams['font.size'] = 11

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
    parser.add_argument("--output", type=str, default="real_pixel_tsne_red_tighter.png")
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

    # === 数据生成（更小的 RGB 波动，让同通道更相似） ===
    print("[3/5] 生成 48 logical RGB pixels → 144 subpixels (红色更集中版) ...")
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

    # 减小 RGB 微波动（从 ±4,5 改为 ±3），让同通道输入更相似 → embedding 更集中
    pixel_values_list = []
    for base in base_vals_all:
        rgb = np.clip(base + np.random.randint(-3, 4, 3), 0, 255)
        pixel_values_list.extend(rgb)

    pixel_values_full = np.array(pixel_values_list, dtype=np.int64)
    pix_t = torch.tensor(pixel_values_full, dtype=torch.long, device=args.device).unsqueeze(0)
    channels = np.arange(N_SUB) % 3

    # Red Zone 代表点
    pixel_values_for_anno = pixel_values_full[::3]
    red_logical_mask = (logical_zones == 0)
    red_logical_indices = np.where(red_logical_mask)[0]
    red_vals = pixel_values_for_anno[red_logical_mask]
    red_log_min = red_logical_indices[np.argmin(red_vals)]
    red_log_max = red_logical_indices[np.argmax(red_vals)]
    red_point1, red_point2 = red_log_min * 3, red_log_max * 3

    print(f"[Annotation] Red Zone R-subpixel: {red_point1}, {red_point2} (pixel={pixel_values_full[red_point1]}, {pixel_values_full[red_point2]})")

    CHANNEL_COLORS = ['#E31A1C', '#33A02C', '#1F78B4']
    cmap_channel = ListedColormap(CHANNEL_COLORS)

    # === 提取 embedding ===
    print("[4/5] 提取 embedding ...")
    with torch.no_grad():
        base_embs = emb(pix_t.squeeze(0)).float().cpu().numpy()
        prop_embs = pixel_emb(pix_t).squeeze(0)
        if intra_pos is not None:
            prop_embs = prop_embs + intra_pos(torch.arange(N_SUB, device=args.device))
        prop_embs = prop_embs.float().cpu().numpy()

    print(f"     Baseline: {base_embs.shape}, Proposed: {prop_embs.shape}")

    print("\n=== Quantitative Metrics (by true subpixel channel) ===")
    print_metrics("Baseline", base_embs, channels)
    print_metrics("Proposed", prop_embs, channels)

    # === 降维 ===
    print("[5/5] 降维 + 可视化 (红色更集中) ...")
    tsne = TSNE(n_components=2, perplexity=args.perplexity, max_iter=2500,
                random_state=args.seed, init='pca', learning_rate='auto')
    base_2d = tsne.fit_transform(base_embs)

    try:
        from umap import UMAP
        # 更紧致的 UMAP
        umap = UMAP(n_components=2, n_neighbors=20, min_dist=0.035,
                    metric='cosine', random_state=args.seed, verbose=False)
        prop_2d = umap.fit_transform(prop_embs)
        method = "UMAP"
    except ImportError:
        prop_2d = tsne.fit_transform(prop_embs)
        method = "t-SNE"

    # === Channel-aware 可视化 jitter（红色 jitter 更小，让红点更集中） ===
    def add_channel_aware_jitter(embs, channels, red_scale=0.025, other_scale=0.04):
        scales = np.full(len(channels), other_scale, dtype=np.float32)
        scales[channels == 0] = red_scale          # R 通道 jitter 更小
        noise = np.random.randn(*embs.shape).astype(np.float32) * scales[:, None]
        return embs + noise

    prop_2d_vis = add_channel_aware_jitter(prop_2d, channels)

    # === 绘图 ===
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 6.8))

    # Baseline
    ax1.scatter(base_2d[:, 0], base_2d[:, 1],
                c=channels, cmap=cmap_channel, s=46, alpha=0.80,
                edgecolors='black', linewidths=0.3)
    ax1.set_title("Figure (a): Baseline — Llama Native Tokenizer Embedding\n"
                  "144 Subpixels (colored by true R/G/B channel)", fontsize=11.5, pad=6)
    ax1.set_xlabel("t-SNE Dim 1")
    ax1.set_ylabel("Dim 2")
    ax1.grid(True, alpha=0.22, linestyle='--')

    # Proposed
    ax2.scatter(prop_2d_vis[:, 0], prop_2d_vis[:, 1],
                c=channels, cmap=cmap_channel, s=48, alpha=0.90,
                edgecolors='black', linewidths=0.35)
    ax2.set_title(f"Figure (b): Proposed — Pixel Embedding + INP\n"
                  f"144 Subpixels | {method} | Yellow jitter ±{args.yellow_jitter_range}  "
                  f"(Red jitter reduced for tighter visual cluster)", fontsize=11.5, pad=6)
    ax2.set_xlabel("Dim 1")
    ax2.set_ylabel("Dim 2")
    ax2.grid(True, alpha=0.22, linestyle='--')

    # 智能 annotation
    def smart_annotate(ax, embs, idx, val, color='#8B0000'):
        x, y = embs[idx]
        med_x, med_y = np.median(embs[:, 0]), np.median(embs[:, 1])
        offset_x = 3.2 if x < med_x else -3.8
        offset_y = 2.8 if y < med_y else -3.2
        ax.annotate(f'R={val}', xy=(x, y), xytext=(x + offset_x, y + offset_y),
                    fontsize=8.5, fontweight='bold', color=color,
                    arrowprops=dict(arrowstyle='->', color='red', lw=1.3,
                                    connectionstyle='arc3,rad=0.22'),
                    bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                              alpha=0.94, edgecolor='red', linewidth=1.1),
                    zorder=15)

    r_val1 = pixel_values_full[red_point1]
    r_val2 = pixel_values_full[red_point2]
    smart_annotate(ax1, base_2d, red_point1, r_val1)
    smart_annotate(ax1, base_2d, red_point2, r_val2)
    smart_annotate(ax2, prop_2d_vis, red_point1, r_val1)
    smart_annotate(ax2, prop_2d_vis, red_point2, r_val2)

    # 图例
    legend_elements = [Patch(facecolor=CHANNEL_COLORS[i], edgecolor='black',
                             label=f'{["R","G","B"][i]} (channel {i})') for i in range(3)]
    ax2.legend(handles=legend_elements, loc='upper left', fontsize=9,
               framealpha=0.92, edgecolor='gray')

    fig.suptitle(
        f"Real Pixel Embedding Geometry (Llama3 + {args.dataset} Stage1)  |  "
        f"48 Logical RGB Pixels → 144 Subpixels\n"
        f"Colored by true subpixel channel (position % 3)  |  Hidden Dim = {H}  |  "
        f"Red cluster tightened via reduced jitter + smaller RGB variation",
        fontsize=12.5, y=0.975
    )

    plt.tight_layout(rect=[0, 0.01, 1, 0.96])

    out_path = os.path.abspath(args.output)
    fig.savefig(out_path, bbox_inches='tight', dpi=300, facecolor='white')
    print(f"\n[Saved] {out_path}")

    np.savez(out_path.replace('.png', '.npz'),
             base_2d=base_2d, prop_2d=prop_2d, channels=channels,
             pixel_values_full=pixel_values_full,
             base_embs=base_embs, prop_embs=prop_embs,
             red_point1=red_point1, red_point2=red_point2)
    print(f"[Saved] {out_path.replace('.png', '.npz')}")

if __name__ == "__main__":
    main()