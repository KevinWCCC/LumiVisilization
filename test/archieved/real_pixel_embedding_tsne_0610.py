#!/usr/bin/env python3
"""
Real Pixel Embedding t-SNE Visualization
========================================
使用真实 Llama3 模型 + 已训练的 checkpoint 生成 140 个点的 t-SNE/UMAP 图。

对比：
- Baseline (左): 直接使用 Llama3 原生 token embedding (tokenizer path)
- Proposed (右): 使用训练好的 PixelInputEmbedding + IntraPatchPositionEmbedding

运行示例:
python real_pixel_embedding_tsne.py \
    --model_id /home/vipuser/Model/LLAMA_3.1_B \
    --dataset DIV2K \
    --ckpt_dir models \
    --use_best \
    --output real_pixel_tsne_DIV2K.png
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
# 请确保此脚本与 train_prefix_inp_pixel_lla_up.py 放在同一目录
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

# ==================== LLMHiddenExtractor (从训练脚本复制) ====================
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
    """简化版 intra/inter 比率计算"""
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
    parser = argparse.ArgumentParser(description="Real t-SNE visualization of pixel embeddings using trained checkpoints")
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
    parser.add_argument("--output", type=str, default="real_pixel_embedding_tsne_0615.png")
    parser.add_argument("--perplexity", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ====================== 1. 加载 Llama3 模型 ======================
    print(f"[1/6] 加载 Llama3 模型: {args.model_id}")
    llm_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tokenizer, model = utils.load_llm_model(args.model_id, "", args.device, dtype=llm_dtype)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    emb = model.get_input_embeddings()
    H = model.config.hidden_size
    print(f"     Hidden size = {H}, dtype = {llm_dtype}")

    # ====================== 2. 加载训练好的模块 ======================
    print(f"[2/6] 加载 {args.dataset} 的训练 checkpoint ...")
    suffix = "_stage1_best" if args.use_best else "_stage1"
    
    pe_path  = os.path.join(args.ckpt_dir, f"pixel_emb_{args.dataset}{suffix}.pt")
    inp_path = os.path.join(args.ckpt_dir, f"INP_{args.dataset}{suffix}.pt")
    sp_path  = os.path.join(args.ckpt_dir, f"SP_{args.dataset}{suffix}.pt")

    print(f"{pe_path}")
    # Pixel Embedding
    if not os.path.exists(pe_path):
        raise FileNotFoundError(f"找不到 Pixel Embedding checkpoint: {pe_path}")
    pixel_emb = PixelInputEmbedding.load(pe_path, device=args.device)
    print(f"     ✓ Pixel Embedding: {pe_path}")

    # Intra-Patch Positional Embedding
    intra_pos = None
    if args.include_intra_pos and os.path.exists(inp_path):
        intra_pos = load_intra_pos(inp_path, model, args.device)
        print(f"     ✓ IntraPatch Positional Embedding: {inp_path}")
    elif args.include_intra_pos:
        print(f"     [Warning] 未找到 INP checkpoint: {inp_path}，将不使用 intra_pos")

    # Soft Prefix（可选）
    prefix = None
    if args.include_prefix and os.path.exists(sp_path):
        prefix = SoftPrefix(H, emb.weight.dtype, args.device,
                            PrefixConfig(length=args.prefix_len))
        ckpt = torch.load(sp_path, map_location="cpu")
        prefix.prefix.data.copy_(ckpt["prefix"].to(args.device))
        print(f"     ✓ SoftPrefix: {sp_path}")
    elif args.include_prefix:
        print(f"     [Warning] 未找到 SP checkpoint: {sp_path}")

    # ====================== 3. 生成 140 个 fake pixel（与 test_0609.py 完全一致） ======================
    print("[3/6] 生成 140 个 rainbow zone 像素点 ...")
    N_ZONES = 7
    N_PER_ZONE = 20
    N = N_ZONES * N_PER_ZONE

    zones = np.repeat(np.arange(N_ZONES), N_PER_ZONE)
    pixel_values = np.concatenate([
        np.random.randint(z * 36, (z + 1) * 36, N_PER_ZONE) for z in range(N_ZONES)
    ])
    pix_t = torch.tensor(pixel_values, dtype=torch.long, device=args.device)  # [140]

    RAINBOW_COLORS = ['#E31A1C', '#FF7F00', '#FFFF00', '#33A02C', 
                      '#1F78B4', '#6A3D9A', '#B15928']
    cmap_rainbow = ListedColormap(RAINBOW_COLORS)
    zone_names = ['Red Zone', 'Orange Zone', 'Yellow Zone', 'Green Zone', 
                  'Blue Zone', 'Indigo/Purple Zone', 'Violet/Brown Zone']

    # ====================== 4. 提取真实 embedding ======================
    print("[4/6] 提取真实 embedding（Baseline vs Proposed）...")

    with torch.no_grad():
        # --- Baseline: Llama3 原生 Tokenizer Embedding ---
        base_embs = emb(pix_t).float().cpu().numpy()          # [140, H]

        # --- Proposed: 训练好的 Pixel Embedding + INP ---
        prop_embs = pixel_emb(pix_t.unsqueeze(1)).squeeze(1)   # [140, H]  (T=1)
        if intra_pos is not None:
            # 为每个独立像素添加 position 0 的 intra-patch embedding
            token_idx = torch.zeros(1, dtype=torch.long, device=args.device)
            pos_emb = intra_pos(token_idx)                     # [1, H]
            prop_embs = prop_embs + pos_emb.expand_as(prop_embs)
        prop_embs = prop_embs.float().cpu().numpy()

    print(f"     Baseline shape: {base_embs.shape}")
    print(f"     Proposed shape: {prop_embs.shape}")

    # ====================== 5. 定量指标 ======================
    print("\n=== Quantitative Metrics (Real Embeddings) ===")
    print_metrics("Baseline (Llama Tokenizer)", base_embs, zones)
    print_metrics("Proposed (PixelEmb + INP)", prop_embs, zones)

    # ====================== 6. 降维 + 可视化 ======================
    print("[5/6] t-SNE / UMAP 降维 ...")
    
    # Baseline 用 t-SNE
    tsne_base = TSNE(n_components=2, perplexity=args.perplexity, 
                     max_iter=2000, random_state=args.seed, init='pca', verbose=0)
    base_2d = tsne_base.fit_transform(base_embs)

    # Proposed 优先用 UMAP（如果可用）
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
        print("[Info] umap-learn 未安装，Proposed 也使用 t-SNE")

    # 轻微 jitter（仅可视化用，让同色点更易区分）
    def add_jitter(embs_2d, scale=0.08):
        return embs_2d + np.random.randn(*embs_2d.shape) * scale
    prop_2d_vis = add_jitter(prop_2d, scale=0.08)

    print("[6/6] 绘制并保存图像 ...")
    fig = plt.figure(figsize=(14, 6))

    # Left: Baseline
    ax1 = fig.add_subplot(1, 2, 1)
    sc1 = ax1.scatter(base_2d[:, 0], base_2d[:, 1],
                      c=zones, cmap=cmap_rainbow, s=55, alpha=0.85,
                      edgecolors='black', linewidths=0.4)
    ax1.set_title('Figure (a): Baseline — Real Llama3 Tokenizer Path\n'
                  '(Native token embeddings for pixel values 0-255)', fontsize=11, pad=8)
    ax1.set_xlabel('t-SNE Dim 1')
    ax1.set_ylabel('t-SNE Dim 2')
    ax1.grid(True, alpha=0.3)

    # Right: Proposed
    ax2 = fig.add_subplot(1, 2, 2)
    sc2 = ax2.scatter(prop_2d_vis[:, 0], prop_2d_vis[:, 1],
                      c=zones, cmap=cmap_rainbow, s=55, alpha=0.88,
                      edgecolors='black', linewidths=0.5)
    ax2.set_title(f'Figure (b): Proposed — Trained Pixel Embedding + INP\n'
                  f'(Real forward through {args.dataset} checkpoint, {prop_method})', 
                  fontsize=11, pad=8)
    ax2.set_xlabel(f'{prop_method} Dim 1')
    ax2.set_ylabel(f'{prop_method} Dim 2')
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f'Real Pixel Embedding Geometry (Llama3 + {args.dataset} Stage1)\n'
                 f'140 Synthetic Pixels | 7 Rainbow Color Zones | Hidden Dim = {H}',
                 fontsize=13, y=0.98)

    plt.tight_layout(rect=[0, 0, 0.92, 0.95])

    # 保存
    out_path = os.path.abspath(args.output)
    fig.savefig(out_path, bbox_inches='tight', facecolor='white', dpi=300)
    print(f"\n[Saved] {out_path}")

    # 额外保存 numpy 结果，方便后续分析
    npz_path = out_path.replace('.png', '.npz')
    np.savez(npz_path,
             base_2d=base_2d,
             prop_2d=prop_2d,
             zones=zones,
             pixel_values=pixel_values,
             base_embs=base_embs,
             prop_embs=prop_embs)
    print(f"[Saved] {npz_path} (可用于后续分析)")

    print("\n=== 可视化完成 ===")
    print("提示：如果 Proposed 聚类效果不明显，请检查：")
    print("  1. 是否使用了正确的 --dataset 和 --use_best")
    print("  2. Stage 1 训练是否已经收敛（BPSP 是否明显下降）")
    print("  3. 可以尝试加上 --include_intra_pos（默认已开启）")


if __name__ == "__main__":
    main()