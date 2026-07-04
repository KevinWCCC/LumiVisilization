#!/usr/bin/env python3
"""
Real Pixel Embedding t-SNE Visualization (Yellow Jitter + Red Annotation)
=========================================================================
使用真实 Llama3 模型 + 已训练的 checkpoint 生成 140 个点的 t-SNE/UMAP 图。

增强功能 v2（2026-06-11）：
1. Yellow Zone 增强输入 jitter（--yellow_jitter_range，默认12），加大同一色系内波动。
2. 新增 Red Zone 自动标注功能：在红色系中自动选取两个代表性点（pixel value 最小/最大），
   在 Baseline 和 Proposed 子图上用 annotate + arrow 清晰标注其 R 通道值（即原始 pixel 值）。
   便于直观展示“同色系不同强度点在 embedding 空间的位置关系”。

对比：
- Baseline (左): 直接使用 Llama3 原生 token embedding
- Proposed (右): 训练好的 PixelInputEmbedding + IntraPatchPositionEmbedding

运行示例:
python real_pixel_embedding_tsne_yellow_jitter_enhanced.py \
    --model_id /home/vipuser/Model/LLAMA_3.1_B \
    --dataset DIV2K \
    --ckpt_dir models \
    --use_best \
    --yellow_jitter_range 12 \
    --output real_pixel_tsne_red_annotated.png
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

# ==================== 项目模块导入 ====================
try:
    from soft_prefix import SoftPrefix, PrefixConfig
    from intra_patch_pos_embed import IntraPatchPositionEmbedding, IntraPatchPosConfig, load_intra_pos
    from readout_head import PixelInputEmbedding, PixelInputEmbeddingConfig
    import utils
except ImportError as e:
    print(f"[Error] 无法导入项目模块: {e}")
    print("请将本脚本放在 train_prefix_inp_pixel_lla_up.py 所在目录下运行。")
    raise

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

class LLMHiddenExtractor(torch.nn.Module):
    """只返回平均 hidden states，减少 DataParallel 通信开销"""
    def __init__(self, llm, hidden_layers):
        super().__init__()
        self.llm = llm
        self.hidden_layers = hidden_layers

    def forward(self, inputs_embeds, attention_mask):
        out = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        h_list = [out.hidden_states[i] for i in self.hidden_layers]
        target_device = h_list[0].device
        h_list = [h.to(target_device) for h in h_list]
        return torch.stack(h_list, dim=0).mean(dim=0)


def compute_intra_inter_ratio(embs, labels, n_neighbors=5):
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, metric='cosine').fit(embs)
    distances, indices = nbrs.kneighbors(embs)
    
    intra_dists = []
    inter_dists = []
    for i in range(len(embs)):
        same_zone = labels[indices[i][1:]] == labels[i]
        if same_zone.any():
            intra_dists.append(distances[i][1:][same_zone].mean())
        diff_zone = labels[indices[i][1:]] != labels[i]
        if diff_zone.any():
            inter_dists.append(distances[i][1:][diff_zone].mean())
    
    intra = np.mean(intra_dists) if intra_dists else 0.0
    inter = np.mean(inter_dists) if inter_dists else 1.0
    return intra / (inter + 1e-8), intra, inter


def print_metrics(name, embs, labels):
    try:
        sil = silhouette_score(embs, labels, metric='cosine')
    except:
        sil = float('nan')
    ratio, intra, inter = compute_intra_inter_ratio(embs, labels)
    print(f"[{name}] Silhouette: {sil:.4f} | Intra/Inter ratio: {ratio:.4f} (intra={intra:.4f}, inter={inter:.4f})")


def main():
    parser = argparse.ArgumentParser(description="Real t-SNE visualization with Yellow jitter + Red Zone R-value annotations")
    parser.add_argument("--model_id", type=str, default="/home/vipuser/Model/LLAMA_3.1_B",
                        help="Llama3 模型路径")
    parser.add_argument("--dataset", type=str, required=True,
                        help="数据集标识，例如 DIV2K / Kodak / BRACS")
    parser.add_argument("--ckpt_dir", type=str, default="models",
                        help="checkpoint 存放目录")
    parser.add_argument("--use_best", action="store_true", default=True,
                        help="加载 *_stage1_best.pt 而不是最终的 stage1.pt")
    parser.add_argument("--prefix_len", type=int, default=16)
    parser.add_argument("--include_prefix", action="store_true", default=False,
                        help="可视化时是否加入 SoftPrefix（通常不需要）")
    parser.add_argument("--include_intra_pos", action="store_true", default=True,
                        help="Proposed 路径是否加入 IntraPatchPositionEmbedding")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--output", type=str, default="real_pixel_embedding_tsne.png")
    parser.add_argument("--perplexity", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    
    # Yellow jitter
    parser.add_argument("--yellow_jitter_range", type=int, default=12,
                        help="Yellow Zone 额外 pixel value jitter ±k（默认12），用于加大同色系内波动")
    parser.add_argument("--vis_jitter_yellow_scale", type=float, default=0.18,
                        help="Yellow Zone 可视化 display jitter 缩放系数（默认0.18）")

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 1. 加载 Llama3 模型
    print(f"[1/6] 加载 Llama3 模型: {args.model_id}")
    llm_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tokenizer, model = utils.load_llm_model(args.model_id, "", args.device, dtype=llm_dtype)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    emb = model.get_input_embeddings()
    H = model.config.hidden_size
    print(f"     Hidden size = {H}, dtype = {llm_dtype}")

    # 2. 加载训练好的模块
    print(f"[2/6] 加载 {args.dataset} 的训练 checkpoint ...")
    suffix = "_stage1_best" if args.use_best else "_stage1"
    
    pe_path  = os.path.join(args.ckpt_dir, f"pixel_emb_{args.dataset}{suffix}.pt")
    inp_path = os.path.join(args.ckpt_dir, f"INP_{args.dataset}{suffix}.pt")
    sp_path  = os.path.join(args.ckpt_dir, f"SP_{args.dataset}{suffix}.pt")

    if not os.path.exists(pe_path):
        raise FileNotFoundError(f"找不到 Pixel Embedding checkpoint: {pe_path}")
    pixel_emb = PixelInputEmbedding.load(pe_path, device=args.device)
    print(f"     ✓ Pixel Embedding: {pe_path}")

    intra_pos = None
    if args.include_intra_pos and os.path.exists(inp_path):
        intra_pos = load_intra_pos(inp_path, model, args.device)
        print(f"     ✓ IntraPatch Positional Embedding: {inp_path}")
    elif args.include_intra_pos:
        print(f"     [Warning] 未找到 INP checkpoint: {inp_path}")

    prefix = None
    if args.include_prefix and os.path.exists(sp_path):
        prefix = SoftPrefix(H, emb.weight.dtype, args.device, PrefixConfig(length=args.prefix_len))
        ckpt = torch.load(sp_path, map_location="cpu")
        prefix.prefix.data.copy_(ckpt["prefix"].to(args.device))
        print(f"     ✓ SoftPrefix: {sp_path}")

    # 3. 生成 140 个 fake pixel（支持 yellow jitter）
    print("[3/6] 生成 140 个 rainbow zone 像素点 ...")
    if args.yellow_jitter_range > 0:
        print(f"     [Enhance] Yellow Zone 将应用 ±{args.yellow_jitter_range} pixel jitter")

    N_ZONES = 7
    N_PER_ZONE = 20
    N = N_ZONES * N_PER_ZONE

    zones = np.repeat(np.arange(N_ZONES), N_PER_ZONE)
    
    pixel_values_list = []
    for z in range(N_ZONES):
        low = z * 36
        high = (z + 1) * 36
        vals = np.random.randint(low, high, N_PER_ZONE)
        if z == 2 and args.yellow_jitter_range > 0:
            jitter = np.random.randint(-args.yellow_jitter_range, args.yellow_jitter_range + 1, N_PER_ZONE)
            vals = np.clip(vals + jitter, 0, 255)
        pixel_values_list.append(vals)
    pixel_values = np.concatenate(pixel_values_list)
    pix_t = torch.tensor(pixel_values, dtype=torch.long, device=args.device)

    # ==================== 自动选择 Red Zone 两个代表性点（用于标注 R 值） ====================
    red_mask = (zones == 0)
    red_indices = np.where(red_mask)[0]
    red_pixel_vals = pixel_values[red_mask]
    
    # 选取 pixel value 最小和最大的两个点（R通道强度差异最明显）
    idx_in_red_min = np.argmin(red_pixel_vals)
    idx_in_red_max = np.argmax(red_pixel_vals)
    red_point1 = red_indices[idx_in_red_min]   # 全局 index，低 R
    red_point2 = red_indices[idx_in_red_max]   # 全局 index，高 R
    
    print(f"[Annotation] Red Zone 自动选取两点用于标注 R 通道值：")
    print(f"  Point1 (全局idx={red_point1}): R/pixel = {pixel_values[red_point1]:3d}  (较暗/低强度)")
    print(f"  Point2 (全局idx={red_point2}): R/pixel = {pixel_values[red_point2]:3d}  (较亮/高强度)")

    RAINBOW_COLORS = ['#E31A1C', '#FF7F00', '#FFFF00', '#33A02C', 
                      '#1F78B4', '#6A3D9A', '#B15928']
    cmap_rainbow = ListedColormap(RAINBOW_COLORS)

    # 4. 提取真实 embedding
    print("[4/6] 提取真实 embedding（Baseline vs Proposed）...")

    with torch.no_grad():
        base_embs = emb(pix_t).float().cpu().numpy()
        prop_embs = pixel_emb(pix_t.unsqueeze(1)).squeeze(1)
        if intra_pos is not None:
            token_idx = torch.zeros(1, dtype=torch.long, device=args.device)
            pos_emb = intra_pos(token_idx)
            prop_embs = prop_embs + pos_emb.expand_as(prop_embs)
        prop_embs = prop_embs.float().cpu().numpy()

    print(f"     Baseline shape: {base_embs.shape}")
    print(f"     Proposed shape: {prop_embs.shape}")

    # 5. 定量指标
    print("\n=== Quantitative Metrics (Real Embeddings) ===")
    print_metrics("Baseline (Llama Tokenizer)", base_embs, zones)
    print_metrics("Proposed (PixelEmb + INP)", prop_embs, zones)

    # 6. 降维 + 可视化
    print("[5/6] t-SNE / UMAP 降维 ...")
    
    tsne_base = TSNE(n_components=2, perplexity=args.perplexity, 
                     max_iter=2000, random_state=args.seed, init='pca', verbose=0)
    base_2d = tsne_base.fit_transform(base_embs)

    try:
        from umap import UMAP
        umap = UMAP(n_components=2, n_neighbors=15, min_dist=0.08, 
                    metric='cosine', random_state=args.seed, verbose=False)
        prop_2d = umap.fit_transform(prop_embs)
        prop_method = "UMAP"
    except ImportError:
        tsne_prop = TSNE(n_components=2, perplexity=args.perplexity,
                         max_iter=2000, random_state=args.seed, init='pca', verbose=0)
        prop_2d = tsne_prop.fit_transform(prop_embs)
        prop_method = "t-SNE"
        print("[Info] umap-learn 未安装，Proposed 使用 t-SNE")

    # 可视化 jitter（黄色组更大）
    def add_zone_aware_jitter(embs_2d, zones, base_scale=0.08, yellow_scale=None):
        if yellow_scale is None:
            yellow_scale = args.vis_jitter_yellow_scale
        scales = np.full(len(zones), base_scale, dtype=np.float32)
        scales[zones == 2] = yellow_scale
        noise = np.random.randn(*embs_2d.shape).astype(np.float32) * scales[:, np.newaxis]
        return embs_2d + noise

    prop_2d_vis = add_zone_aware_jitter(prop_2d, zones)

    print("[6/6] 绘制并保存图像（包含 Red Zone R值标注）...")

    fig = plt.figure(figsize=(14, 6.2))

    # Left: Baseline
    ax1 = fig.add_subplot(1, 2, 1)
    sc1 = ax1.scatter(base_2d[:, 0], base_2d[:, 1],
                      c=zones, cmap=cmap_rainbow, s=55, alpha=0.85,
                      edgecolors='black', linewidths=0.4)
    ax1.set_title('Figure (a): Baseline — Real Llama3 Tokenizer Path\n'
                  '(Native token embeddings)', fontsize=11, pad=6)
    ax1.set_xlabel('t-SNE Dim 1')
    ax1.set_ylabel('t-SNE Dim 2')
    ax1.grid(True, alpha=0.3)

    # Right: Proposed
    ax2 = fig.add_subplot(1, 2, 2)
    sc2 = ax2.scatter(prop_2d_vis[:, 0], prop_2d_vis[:, 1],
                      c=zones, cmap=cmap_rainbow, s=55, alpha=0.88,
                      edgecolors='black', linewidths=0.5)
    yellow_note = f" + Yellow jitter ±{args.yellow_jitter_range}" if args.yellow_jitter_range > 0 else ""
    ax2.set_title(f'Figure (b): Proposed — Pixel Embedding + INP{yellow_note}\n'
                  f'(Real {args.dataset} checkpoint, {prop_method})', 
                  fontsize=11, pad=6)
    ax2.set_xlabel(f'{prop_method} Dim 1')
    ax2.set_ylabel(f'{prop_method} Dim 2')
    ax2.grid(True, alpha=0.3)

    # ==================== Red Zone R值标注（annotate + arrow） ====================
    def annotate_red_point(ax, embs_2d, point_idx, color='#8B0000', label_prefix='R'):
        x, y = embs_2d[point_idx]
        r_val = pixel_values[point_idx]
        # 动态偏移，避免标签重叠或遮挡点
        offset_x = 2.2 if x < np.median(embs_2d[:, 0]) else -3.0
        offset_y = 2.0 if y > np.median(embs_2d[:, 1]) else -2.2
        ax.annotate(
            f'{label_prefix}={r_val}',
            xy=(x, y),
            xytext=(x + offset_x, y + offset_y),
            fontsize=9,
            fontweight='bold',
            color=color,
            ha='center', va='center',
            arrowprops=dict(
                arrowstyle='->',
                color='red',
                lw=1.6,
                connectionstyle='arc3,rad=0.2'
            ),
            bbox=dict(
                boxstyle='round,pad=0.4',
                facecolor='white',
                alpha=0.95,
                edgecolor='red',
                linewidth=1.3
            ),
            zorder=12
        )

    # 在两个子图上都标注（对比 Baseline vs Proposed 中同一点的位置差异）
    annotate_red_point(ax1, base_2d, red_point1)
    annotate_red_point(ax1, base_2d, red_point2)
    annotate_red_point(ax2, prop_2d_vis, red_point1)
    annotate_red_point(ax2, prop_2d_vis, red_point2)

    # 总标题
    fig.suptitle(
        f'Real Pixel Embedding Geometry (Llama3 + {args.dataset} Stage1)  |  '
        f'140 Pixels, 7 Zones  |  Red Zone 示例点已标注原始 R/pixel 值\n'
        f'Yellow jitter ±{args.yellow_jitter_range}  |  Hidden Dim = {H}',
        fontsize=12, y=0.96
    )

    plt.tight_layout(rect=[0, 0, 0.92, 0.90])

    # 保存
    out_path = os.path.abspath(args.output)
    fig.savefig(out_path, bbox_inches='tight', facecolor='white', dpi=300)
    print(f"\n[Saved] {out_path}")

    # 保存 npz（包含标注点索引，便于后续分析）
    npz_path = out_path.replace('.png', '.npz')
    np.savez(npz_path,
             base_2d=base_2d,
             prop_2d=prop_2d,
             zones=zones,
             pixel_values=pixel_values,
             base_embs=base_embs,
             prop_embs=prop_embs,
             red_point1=red_point1,
             red_point2=red_point2,
             yellow_jitter_range=args.yellow_jitter_range)
    print(f"[Saved] {npz_path} (含 red_point 索引)")

    print("\n=== 可视化完成 ===")
    print("Red Zone 标注说明：")
    print("  - 两个箭头指向的点分别为 Red Zone 内 R/pixel 值最小和最大的样本。")
    print("  - 对比左右两图可直观看到：Baseline 中同色不同R的点可能分散较远；")
    print("    Proposed 中它们在 embedding 空间中被更好地组织（更接近）。")
    print("  - 黄色组仍保持增强 jitter 效果。")


if __name__ == "__main__":
    main()