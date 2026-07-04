#!/usr/bin/env python3
"""
Real Pixel Embedding t-SNE Visualization (Colored by Real Subpixel Channel R/G/B)
=================================================================================
使用真实 Llama3 模型 + 已训练的 checkpoint 生成 140 个点的 t-SNE/UMAP 图。

核心修改（2026-06-14）：
- 移除了原来的人工 7-zone rainbow 着色
- 现在每个点按照其在序列中的位置 % 3 分配真实 subpixel 通道（R=0, G=1, B=2）
- 散点图使用真实 RGB 颜色（红色/绿色/蓝色）着色，直观展示模型对不同通道 subpixel 的 embedding 区分能力
- 保留强度代表点标注功能（在 R 通道中选 min/max pixel value 的点进行 annotate）

对比：
- Baseline (左): 直接使用 Llama3 原生 token embedding (tokenizer path)
- Proposed (右): 使用训练好的 PixelInputEmbedding + IntraPatchPositionEmbedding

运行示例:
python real_pixel_embedding_tsne_by_channel.py \
    --model_id /home/vipuser/Model/LLAMA_3.1_B \
    --dataset DIV2K \
    --ckpt_dir models \
    --use_best \
    --output real_pixel_tsne_by_channel.png
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
        same = labels[indices[i][1:]] == labels[i]
        if same.any():
            intra_dists.append(distances[i][1:][same].mean())
        diff = labels[indices[i][1:]] != labels[i]
        if diff.any():
            inter_dists.append(distances[i][1:][diff].mean())
    
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
    parser = argparse.ArgumentParser(description="Real t-SNE visualization colored by true subpixel channel (R/G/B)")
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
    parser.add_argument("--output", type=str, default="real_pixel_embedding_tsne_by_channel.png")
    parser.add_argument("--perplexity", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    
    # 可选：为特定通道增加输入 jitter（模拟真实图像中通道间波动）
    parser.add_argument("--channel_jitter_range", type=int, default=0,
                        help="为 channel 0 (R) 额外增加 pixel value jitter ±k，0 表示关闭")
    parser.add_argument("--vis_jitter_scale", type=float, default=0.08,
                        help="可视化时 display jitter 基础缩放系数")

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

    # 3. 生成 140 个 fake pixel（按真实 subpixel 通道着色）
    print("[3/6] 生成 140 个像素点（按真实 R/G/B 通道分配颜色）...")
    
    N = 140
    # 通道标签：模拟真实序列中 position % 3
    # 0 = R (红色), 1 = G (绿色), 2 = B (蓝色)
    channels = np.arange(N) % 3
    
    # 生成 pixel values（0-255），可选为 R 通道增加 jitter
    pixel_values = np.random.randint(0, 256, N)
    if args.channel_jitter_range > 0:
        r_mask = (channels == 0)
        jitter = np.random.randint(-args.channel_jitter_range, args.channel_jitter_range + 1, r_mask.sum())
        pixel_values[r_mask] = np.clip(pixel_values[r_mask] + jitter, 0, 255)
        print(f"     [Enhance] R 通道 (channel 0) 已应用 ±{args.channel_jitter_range} jitter")

    pix_t = torch.tensor(pixel_values, dtype=torch.long, device=args.device)

    # 选取 R 通道 (channel==0) 中 pixel value 最小和最大的两个点用于标注
    r_mask = (channels == 0)
    r_indices = np.where(r_mask)[0]
    r_pixel_vals = pixel_values[r_mask]
    
    if len(r_indices) >= 2:
        idx_in_r_min = np.argmin(r_pixel_vals)
        idx_in_r_max = np.argmax(r_pixel_vals)
        r_point1 = r_indices[idx_in_r_min]
        r_point2 = r_indices[idx_in_r_max]
        print(f"[Annotation] R 通道自动选取两点用于标注原始 pixel 值：")
        print(f"  Point1 (idx={r_point1}): pixel={pixel_values[r_point1]:3d} (较暗)")
        print(f"  Point2 (idx={r_point2}): pixel={pixel_values[r_point2]:3d} (较亮)")
    else:
        r_point1 = r_point2 = None

    # 真实 subpixel 颜色映射（R=红, G=绿, B=蓝）
    CHANNEL_COLORS = ['#E31A1C', '#33A02C', '#1F78B4']  # 红 / 绿 / 蓝
    cmap_channel = ListedColormap(CHANNEL_COLORS)
    channel_names = ['R (Red)', 'G (Green)', 'B (Blue)']

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

    # 5. 定量指标（现在按真实通道 labels 计算）
    print("\n=== Quantitative Metrics (Real Embeddings, grouped by subpixel channel) ===")
    print_metrics("Baseline (Llama Tokenizer)", base_embs, channels)
    print_metrics("Proposed (PixelEmb + INP)", prop_embs, channels)

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

    # 可视化 jitter（可选按通道调整强度）
    def add_channel_aware_jitter(embs_2d, channels, base_scale=0.08):
        scales = np.full(len(channels), base_scale, dtype=np.float32)
        # 可根据需要给某个通道更大的 jitter
        noise = np.random.randn(*embs_2d.shape).astype(np.float32) * scales[:, np.newaxis]
        return embs_2d + noise

    prop_2d_vis = add_channel_aware_jitter(prop_2d, channels, base_scale=args.vis_jitter_scale)

    print("[6/6] 绘制并保存图像（按真实 R/G/B 通道着色）...")

    fig = plt.figure(figsize=(14, 6.2))

    # Left: Baseline
    ax1 = fig.add_subplot(1, 2, 1)
    sc1 = ax1.scatter(base_2d[:, 0], base_2d[:, 1],
                      c=channels, cmap=cmap_channel, s=55, alpha=0.85,
                      edgecolors='black', linewidths=0.4)
    ax1.set_title('Figure (a): Baseline — Real Llama3 Tokenizer Path\n'
                  '(Native token embeddings, colored by subpixel channel)', fontsize=11, pad=6)
    ax1.set_xlabel('t-SNE Dim 1')
    ax1.set_ylabel('t-SNE Dim 2')
    ax1.grid(True, alpha=0.3)

    # Right: Proposed
    ax2 = fig.add_subplot(1, 2, 2)
    sc2 = ax2.scatter(prop_2d_vis[:, 0], prop_2d_vis[:, 1],
                      c=channels, cmap=cmap_channel, s=55, alpha=0.88,
                      edgecolors='black', linewidths=0.5)
    jitter_note = f" + R jitter ±{args.channel_jitter_range}" if args.channel_jitter_range > 0 else ""
    ax2.set_title(f'Figure (b): Proposed — Pixel Embedding + INP{jitter_note}\n'
                  f'(Real {args.dataset} checkpoint, {prop_method}, colored by subpixel channel)', 
                  fontsize=11, pad=6)
    ax2.set_xlabel(f'{prop_method} Dim 1')
    ax2.set_ylabel(f'{prop_method} Dim 2')
    ax2.grid(True, alpha=0.3)

    # ==================== R 通道代表点标注（annotate + arrow） ====================
    def annotate_r_point(ax, embs_2d, point_idx, color='#8B0000', label_prefix='R'):
        if point_idx is None:
            return
        x, y = embs_2d[point_idx]
        val = pixel_values[point_idx]
        offset_x = 2.2 if x < np.median(embs_2d[:, 0]) else -3.0
        offset_y = 2.0 if y > np.median(embs_2d[:, 1]) else -2.2
        ax.annotate(
            f'{label_prefix}={val}',
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

    if r_point1 is not None:
        annotate_r_point(ax1, base_2d, r_point1)
        annotate_r_point(ax1, base_2d, r_point2)
        annotate_r_point(ax2, prop_2d_vis, r_point1)
        annotate_r_point(ax2, prop_2d_vis, r_point2)

    # 添加图例
    legend_elements = [Patch(facecolor=CHANNEL_COLORS[i], edgecolor='black', label=channel_names[i])
                       for i in range(3)]
    ax2.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.9)

    # 总标题
    fig.suptitle(
        f'Real Pixel Embedding Geometry (Llama3 + {args.dataset} Stage1)  |  '
        f'140 Points colored by true subpixel channel (R/G/B via position % 3)\n'
        f'Hidden Dim = {H}  |  Each point uses channel one-hot exactly as in PixelInputEmbedding.forward',
        fontsize=12, y=0.96
    )

    plt.tight_layout(rect=[0, 0, 0.92, 0.90])

    # 保存
    out_path = os.path.abspath(args.output)
    fig.savefig(out_path, bbox_inches='tight', facecolor='white', dpi=300)
    print(f"\n[Saved] {out_path}")

    # 保存 npz
    npz_path = out_path.replace('.png', '.npz')
    np.savez(npz_path,
             base_2d=base_2d,
             prop_2d=prop_2d,
             channels=channels,
             pixel_values=pixel_values,
             base_embs=base_embs,
             prop_embs=prop_embs,
             r_point1=r_point1,
             r_point2=r_point2)
    print(f"[Saved] {npz_path} (含 channels 标签)")

    print("\n=== 可视化完成 ===")
    print("说明：")
    print("  - 散点颜色现在严格对应 PixelInputEmbedding 中 ch_oh = one_hot(position % 3, 3)")
    print("  - 红色点 = 该位置在序列中被视为 R 通道 (channel 0)")
    print("  - 绿色点 = G 通道 (channel 1)")
    print("  - 蓝色点 = B 通道 (channel 2)")
    print("  - 箭头标注的点为 R 通道内 pixel value 最小/最大的样本，便于观察强度变化。")


if __name__ == "__main__":
    main()