#!/usr/bin/env python3
"""
Real Pixel Embedding t-SNE Visualization (48 Logical Pixels → 144 Subpixels)
=========================================================================
轻量版：48 logical RGB pixels × 3 subpixels = 144 subpixels
直接对 144 个 subpixel 做 7D 展开 + t-SNE/UMAP 可视化
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
    parser.add_argument("--output", type=str, default="real_pixel_tsne_48logical.png")
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

    # 3. 生成 48 logical pixels → 144 subpixels
    print("[3/5] 生成 48 logical RGB pixels（共 144 subpixels）...")
    N_ZONES = 7
    N_LOGICAL = 48
    N_SUB = N_LOGICAL * 3             # 144

    # zones 扩展为 144（每个 logical pixel 重复 3 次）
    logical_zones = np.repeat(np.arange(N_ZONES), N_LOGICAL // N_ZONES)
    # 补充剩余的 logical pixels 使总数达到 48
    extra = N_LOGICAL - len(logical_zones)
    if extra > 0:
        logical_zones = np.concatenate([logical_zones, np.arange(extra)])
    zones = np.repeat(logical_zones, 3)   # length=144

    pixel_values_list = []
    for z in range(N_ZONES):
        low, high = z * 36, (z + 1) * 36
        base_vals = np.random.randint(low, high, N_LOGICAL // N_ZONES + (z < extra))
        if z == 2 and args.yellow_jitter_range > 0:
            jitter = np.random.randint(-args.yellow_jitter_range, args.yellow_jitter_range + 1, len(base_vals))
            base_vals = np.clip(base_vals + jitter, 0, 255)
        for base in base_vals:
            rgb = np.clip(base + np.random.randint(-3, 4, 3), 0, 255)
            pixel_values_list.extend(rgb)

    pixel_values_full = np.array(pixel_values_list, dtype=np.int64)  # (144,)
    pix_t = torch.tensor(pixel_values_full, dtype=torch.long, device=args.device).unsqueeze(0)  # (1, 144)

    # 用于 Red Zone 标注（取每个 logical pixel 的 R 值）
    pixel_values_for_anno = pixel_values_full[::3]  # (48,)

    # 选择 Red Zone 两个代表点（logical index）
    red_logical_mask = (logical_zones == 0)
    red_logical_indices = np.where(red_logical_mask)[0]
    red_vals = pixel_values_for_anno[red_logical_mask]
    red_log_min = red_logical_indices[np.argmin(red_vals)]
    red_log_max = red_logical_indices[np.argmax(red_vals)]
    # 映射到 144 序列中的 R subpixel 位置
    red_point1 = red_log_min * 3
    red_point2 = red_log_max * 3

    print(f"[Annotation] Red Zone 两个代表点（R subpixel 位置）: {red_point1}, {red_point2}")

    RAINBOW_COLORS = ['#E31A1C', '#FF7F00', '#FFFF00', '#33A02C', '#1F78B4', '#6A3D9A', '#B15928']
    cmap_rainbow = ListedColormap(RAINBOW_COLORS)

    # 4. 提取 embedding（144 个 subpixel 直接走 7D 展开）
    print("[4/5] 提取 144 个 subpixel 的 embedding ...")
    with torch.no_grad():
        base_embs = emb(pix_t.squeeze(0)).float().cpu().numpy()           # (144, H)
        prop_embs = pixel_emb(pix_t).squeeze(0)                           # (144, H)
        if intra_pos is not None:
            token_idx = torch.arange(144, device=args.device)
            prop_embs = prop_embs + intra_pos(token_idx)
        prop_embs = prop_embs.float().cpu().numpy()

    print(f"     Baseline: {base_embs.shape}, Proposed: {prop_embs.shape}")

    # 5. 定量指标
    print("\n=== Quantitative Metrics ===")
    print_metrics("Baseline", base_embs, zones)
    print_metrics("Proposed", prop_embs, zones)

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

    # 可视化 jitter
    def add_jitter(embs, zones):
        scales = np.full(len(zones), 0.06)
        scales[zones == 2] = 0.15   # Yellow zone 更大 jitter
        return embs + np.random.randn(*embs.shape) * scales[:, None]

    prop_2d_vis = add_jitter(prop_2d, zones)

    # 绘图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.scatter(base_2d[:, 0], base_2d[:, 1], c=zones, cmap=cmap_rainbow, s=45, alpha=0.85, edgecolors='black', linewidths=0.3)
    ax1.set_title("Baseline (Llama Native Embedding)\n144 Subpixels", fontsize=11)
    ax1.set_xlabel("Dim 1"); ax1.set_ylabel("Dim 2"); ax1.grid(True, alpha=0.3)

    ax2.scatter(prop_2d_vis[:, 0], prop_2d_vis[:, 1], c=zones, cmap=cmap_rainbow, s=45, alpha=0.88, edgecolors='black', linewidths=0.3)
    ax2.set_title(f"Proposed (PixelEmb + 7D Features)\n144 Subpixels | {method}", fontsize=11)
    ax2.set_xlabel("Dim 1"); ax2.set_ylabel("Dim 2"); ax2.grid(True, alpha=0.3)

    # Red Zone 标注（标注 R subpixel）
    def annotate(ax, embs, idx, r_val):
        x, y = embs[idx]
        ax.annotate(f'R={r_val}', xy=(x, y), xytext=(x+2.5, y+2.5),
                    fontsize=8, fontweight='bold', color='#8B0000',
                    arrowprops=dict(arrowstyle='->', color='red', lw=1.2),
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='red'))

    r_val1 = pixel_values_full[red_point1]
    r_val2 = pixel_values_full[red_point2]
    annotate(ax1, base_2d, red_point1, r_val1)
    annotate(ax1, base_2d, red_point2, r_val2)
    annotate(ax2, prop_2d_vis, red_point1, r_val1)
    annotate(ax2, prop_2d_vis, red_point2, r_val2)

    fig.suptitle(f"Pixel Embedding Geometry | 48 Logical RGB Pixels → 144 Subpixels | 7 Zones | Yellow Jitter ±{args.yellow_jitter_range}", 
                 fontsize=12, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = os.path.abspath(args.output)
    fig.savefig(out_path, bbox_inches='tight', dpi=300, facecolor='white')
    print(f"\n[Saved] {out_path}")

    # 保存 npz
    np.savez(out_path.replace('.png', '.npz'),
             base_2d=base_2d, prop_2d=prop_2d, zones=zones,
             pixel_values_full=pixel_values_full,
             base_embs=base_embs, prop_embs=prop_embs,
             red_point1=red_point1, red_point2=red_point2)
    print(f"[Saved] {out_path.replace('.png', '.npz')}")


if __name__ == "__main__":
    main()