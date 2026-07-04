#!/usr/bin/env python3
"""
Real Pixel Embedding t-SNE Visualization (48 Logical RGB Pixels → 144 Subpixels)
===============================================================================
轻量版 + 真实 subpixel 通道着色版

核心改进：
- 保持 48 logical RGB pixels → 144 subpixels 的真实生成逻辑
  （每个 logical pixel 生成 R/G/B 三个 subpixel，并加入小幅真实波动）
- 现在散点颜色严格按照真实 subpixel 通道着色：
    position % 3 == 0 → R (红色)
    position % 3 == 1 → G (绿色)
    position % 3 == 2 → B (蓝色)
- 这与 PixelInputEmbedding.forward 中 ch_oh = one_hot(arange(T) % 3) 完全一致
- 同时保留 Red Zone 代表点的 R subpixel 标注功能

对比：
- Baseline (左): Llama3 原生 token embedding
- Proposed (右): 训练好的 PixelInputEmbedding (7D特征) + IntraPatchPositionEmbedding

运行示例:
python real_pixel_embedding_tsne_144subpixel_by_channel.py \
    --model_id /home/vipuser/Model/LLAMA_3.1_B \
    --dataset DIV2K \
    --ckpt_dir models \
    --use_best \
    --output real_pixel_tsne_144sub_by_channel.png
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
    from soft_prefix import SoftPrefix, PrefixConfig
    from intra_patch_pos_embed import IntraPatchPositionEmbedding, load_intra_pos
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
    parser.add_argument("--output", type=str, default="real_pixel_tsne_144sub_by_channel.png")
    parser.add_argument("--perplexity", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--yellow_jitter_range", type=int, default=8)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 1. 加载模型
    print(f"[1/5] 加载 Llama3 模型: {args.model_id}")
    llm_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    _, model = utils.load_llm_model(args.model_id, "", args.device, dtype=llm_dtype)
    for p in model.parameters(): p.requires_grad_(False)
    model.eval()
    emb = model.get_input_embeddings()
    H = model.config.hidden_size

    # 2. 加载 Pixel Embedding
    suffix = "_stage1_best" if args.use_best else "_stage1"
    pe_path = os.path.join(args.ckpt_dir, f"pixel_emb_{args.dataset}{suffix}.pt")
    pixel_emb = PixelInputEmbedding.load(pe_path, device=args.device)
    print(f"[2/5] ✓ Pixel Embedding loaded: {pe_path}")

    intra_pos = None
    if args.include_intra_pos:
        inp_path = os.path.join(args.ckpt_dir, f"INP_{args.dataset}{suffix}.pt")
        if os.path.exists(inp_path):
            intra_pos = load_intra_pos(inp_path, model, args.device)
            print(f"     ✓ IntraPatch Positional Embedding loaded")

    # 3. 生成 48 logical RGB pixels → 144 subpixels（真实通道循环）
    print("[3/5] 生成 48 logical RGB pixels → 144 subpixels ...")
    N_ZONES = 7
    N_LOGICAL = 48
    N_SUB = N_LOGICAL * 3             # 144

    # logical zone 分配（用于 Red Zone 标注）
    logical_zones = np.repeat(np.arange(N_ZONES), N_LOGICAL // N_ZONES)
    extra = N_LOGICAL - len(logical_zones)
    if extra > 0:
        logical_zones = np.concatenate([logical_zones, np.arange(extra)])

    # 生成 144 个 subpixel value
    pixel_values_list = []
    for z in range(N_ZONES):
        low, high = z * 36, (z + 1) * 36
        n_in_zone = N_LOGICAL // N_ZONES + (z < extra)
        base_vals = np.random.randint(low, high, n_in_zone)
        if z == 2 and args.yellow_jitter_range > 0:
            jitter = np.random.randint(-args.yellow_jitter_range, args.yellow_jitter_range + 1, n_in_zone)
            base_vals = np.clip(base_vals + jitter, 0, 255)
        for base in base_vals:
            # 每个 logical pixel 生成 R/G/B 三个 subpixel，加入真实小波动
            rgb = np.clip(base + np.random.randint(-4, 5, 3), 0, 255)
            pixel_values_list.extend(rgb)

    pixel_values_full = np.array(pixel_values_list, dtype=np.int64)  # (144,)
    pix_t = torch.tensor(pixel_values_full, dtype=torch.long, device=args.device).unsqueeze(0)  # (1, 144)

    # 真实 subpixel 通道标签（与模型内部完全一致）
    channels = np.arange(N_SUB) % 3   # 0=R, 1=G, 2=B

    # 用于 Red Zone 标注（取每个 logical pixel 的 R subpixel）
    pixel_values_for_anno = pixel_values_full[::3]  # (48,)

    # 选择 Red Zone 两个代表 logical pixel
    red_logical_mask = (logical_zones == 0)
    red_logical_indices = np.where(red_logical_mask)[0]
    red_vals = pixel_values_for_anno[red_logical_mask]
    red_log_min = red_logical_indices[np.argmin(red_vals)]
    red_log_max = red_logical_indices[np.argmax(red_vals)]
    # 映射到 144 序列中的 R subpixel 位置 (0, 3, 6, ...)
    red_point1 = red_log_min * 3
    red_point2 = red_log_max * 3

    print(f"[Annotation] Red Zone 两个代表点（R subpixel 位置）: {red_point1}, {red_point2}")
    print(f"             对应 pixel value: {pixel_values_full[red_point1]}, {pixel_values_full[red_point2]}")

    # 真实 subpixel 颜色映射
    CHANNEL_COLORS = ['#E31A1C', '#33A02C', '#1F78B4']  # R / G / B
    cmap_channel = ListedColormap(CHANNEL_COLORS)
    channel_names = ['R (channel 0)', 'G (channel 1)', 'B (channel 2)']

    # 4. 提取 embedding（144 个 subpixel 直接走 7D 展开 + INP）
    print("[4/5] 提取 144 个 subpixel 的 embedding（真实 T=144 通道循环） ...")
    with torch.no_grad():
        base_embs = emb(pix_t.squeeze(0)).float().cpu().numpy()           # (144, H)
        prop_embs = pixel_emb(pix_t).squeeze(0)                           # (144, H)
        if intra_pos is not None:
            token_idx = torch.arange(N_SUB, device=args.device)
            prop_embs = prop_embs + intra_pos(token_idx)
        prop_embs = prop_embs.float().cpu().numpy()

    print(f"     Baseline: {base_embs.shape}, Proposed: {prop_embs.shape}")

    # 5. 定量指标（按真实 subpixel channel 计算）
    print("\n=== Quantitative Metrics (grouped by true subpixel channel R/G/B) ===")
    print_metrics("Baseline (Llama Tokenizer)", base_embs, channels)
    print_metrics("Proposed (PixelEmb + INP)", prop_embs, channels)

    # 6. 降维 + 可视化
    print("[5/5] t-SNE / UMAP 降维 (144 points) ...")
    tsne = TSNE(n_components=2, perplexity=args.perplexity, max_iter=2000, random_state=args.seed, init='pca')
    base_2d = tsne.fit_transform(base_embs)

    try:
        from umap import UMAP
        prop_2d = UMAP(n_components=2, n_neighbors=12, min_dist=0.1, metric='cosine', random_state=args.seed).fit_transform(prop_embs)
        method = "UMAP"
    except:
        prop_2d = tsne.fit_transform(prop_embs)
        method = "t-SNE"

    # 可视化 jitter（可选按 channel 调整）
    def add_channel_jitter(embs, channels, base_scale=0.06):
        scales = np.full(len(channels), base_scale, dtype=np.float32)
        # Yellow zone 相关 jitter 已体现在数据生成，这里只做显示 jitter
        noise = np.random.randn(*embs.shape).astype(np.float32) * scales[:, None]
        return embs + noise

    prop_2d_vis = add_channel_jitter(prop_2d, channels)

    # 绘图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.2))

    sc1 = ax1.scatter(base_2d[:, 0], base_2d[:, 1],
                      c=channels, cmap=cmap_channel, s=42, alpha=0.85,
                      edgecolors='black', linewidths=0.3)
    ax1.set_title("Figure (a): Baseline — Llama Native Tokenizer Embedding\n144 Subpixels (colored by true R/G/B channel)", fontsize=11)
    ax1.set_xlabel("t-SNE / UMAP Dim 1")
    ax1.set_ylabel("Dim 2")
    ax1.grid(True, alpha=0.3)

    sc2 = ax2.scatter(prop_2d_vis[:, 0], prop_2d_vis[:, 1],
                      c=channels, cmap=cmap_channel, s=42, alpha=0.88,
                      edgecolors='black', linewidths=0.3)
    jitter_note = f" + Yellow jitter ±{args.yellow_jitter_range}" if args.yellow_jitter_range > 0 else ""
    ax2.set_title(f"Figure (b): Proposed — Pixel Embedding + INP{jitter_note}\n144 Subpixels | {method} | colored by true subpixel channel", fontsize=11)
    ax2.set_xlabel("Dim 1")
    ax2.set_ylabel("Dim 2")
    ax2.grid(True, alpha=0.3)

    # Red Zone R subpixel 标注
    def annotate_r(ax, embs, idx, val):
        x, y = embs[idx]
        ax.annotate(f'R={val}', xy=(x, y), xytext=(x + 2.8, y + 2.8),
                    fontsize=8, fontweight='bold', color='#8B0000',
                    arrowprops=dict(arrowstyle='->', color='red', lw=1.3),
                    bbox=dict(boxstyle='round,pad=0.35', facecolor='white', alpha=0.92, edgecolor='red'),
                    zorder=12)

    r_val1 = pixel_values_full[red_point1]
    r_val2 = pixel_values_full[red_point2]
    annotate_r(ax1, base_2d, red_point1, r_val1)
    annotate_r(ax1, base_2d, red_point2, r_val2)
    annotate_r(ax2, prop_2d_vis, red_point1, r_val1)
    annotate_r(ax2, prop_2d_vis, red_point2, r_val2)

    # 添加 R/G/B 图例
    legend_elements = [Patch(facecolor=CHANNEL_COLORS[i], edgecolor='black', label=channel_names[i])
                       for i in range(3)]
    ax2.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.92)

    fig.suptitle(
        f"Real Pixel Embedding Geometry | 48 Logical RGB Pixels → 144 Subpixels\n"
        f"Colored by true subpixel channel (R/G/B via position % 3, exactly as in PixelInputEmbedding) | "
        f"Hidden Dim = {H}",
        fontsize=12, y=0.97
    )

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    out_path = os.path.abspath(args.output)
    fig.savefig(out_path, bbox_inches='tight', dpi=300, facecolor='white')
    print(f"\n[Saved] {out_path}")

    # 保存 npz
    np.savez(out_path.replace('.png', '.npz'),
             base_2d=base_2d,
             prop_2d=prop_2d,
             channels=channels,
             zones=logical_zones,           # logical zone，仅供参考
             pixel_values_full=pixel_values_full,
             base_embs=base_embs,
             prop_embs=prop_embs,
             red_point1=red_point1,
             red_point2=red_point2)
    print(f"[Saved] {out_path.replace('.png', '.npz')} (含 channels 标签)")


if __name__ == "__main__":
    main()