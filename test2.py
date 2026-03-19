#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tsne_pe_comparison.py
对比同一个 16×16 patch 在 Without PE vs With PE 下的 t-SNE 图
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import torch
import os
from datetime import datetime
# ====================== 项目依赖导入 ======================
from readout_head import PixelInputEmbedding, PixelInputEmbeddingConfig
import utils  # 您的项目 utils.py（加载模型用）

def load_one_patch(jsonl_path, patch_size=16, target_idx=0):
    """读取 channel-separated jsonl（您的格式）"""
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i != target_idx:
                continue
            data = json.loads(line.strip())
            r = np.array(data['R'], dtype=np.float32)
            g = np.array(data['G'], dtype=np.float32)
            b = np.array(data['B'], dtype=np.float32)
            patch = np.stack([r, g, b], axis=-1).reshape(patch_size, patch_size, 3) / 255.0
            return patch, data.get('meta', {})
    raise IndexError(f"样本索引 {target_idx} 不存在")

def extract_features(patch, pixel_emb=None, device="cuda"):
    """返回两种特征"""
    h, w, _ = patch.shape
    n = h * w  # 256
    
    # === Without PE: raw RGB (3维) ===
    rgb = patch.reshape(n, 3)
    
    # === With PE: 使用 PixelInputEmbedding ===
    if pixel_emb is None:
        raise ValueError("请提供 PixelInputEmbedding")
    
    pix_t = torch.tensor(patch.reshape(n, 3).flatten(), dtype=torch.long, device=device).unsqueeze(0)  # [1, 768]
    with torch.no_grad():
        pe_emb = pixel_emb(pix_t).squeeze(0).cpu().numpy()  # [256, H]
    
    return rgb, pe_emb  # (256,3) 和 (256, H)

def plot_comparison(embed_no, embed_with, colors, title_prefix=""):
    """并排绘制两张 t-SNE 图"""
    fig, axs = plt.subplots(1, 2, figsize=(16, 7), dpi=150)
    
    for ax, embed, name in zip(axs, [embed_no, embed_with], ["Without PE", "With PE"]):
        ax.scatter(embed[:, 0], embed[:, 1], c=colors, cmap='tab10', s=60, alpha=0.85)
        ax.set_title(f"{title_prefix}\n{name}", fontsize=14)
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=str, help="jsonl 文件路径")
    parser.add_argument("--idx", type=int, default=0, help="样本索引")
    parser.add_argument("--model_id", type=str, default="/home/vipuser/Model/LLAMA_3.1_B")
    parser.add_argument("--pixel_emb_ckpt", type=str, default="models/pixel_emb_init.pt")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save", type=str, default="tsne_pe_comparison.pdf")
    args = parser.parse_args()

    # 1. 加载 patch
    patch, meta = load_one_patch(args.jsonl, 16, args.idx)
    print(f"加载 patch shape: {patch.shape}")

    # 2. 加载 PixelInputEmbedding（复用您的 evaluate 逻辑）
    device = torch.device(args.device)
    H = 4096 if "Llama-3.1" in args.model_id else 4096  # 根据您的模型调整
    if os.path.exists(args.pixel_emb_ckpt):
        pixel_emb = PixelInputEmbedding.load(args.pixel_emb_ckpt, device=device)
        print(f"加载 PE from {args.pixel_emb_ckpt}")
    else:
        pixel_emb = PixelInputEmbedding(PixelInputEmbeddingConfig(hidden_size=H)).to(device)
        print("使用随机初始化的 PE（请先训练）")

    # 3. 提取两种特征
    rgb_features, pe_features = extract_features(patch, pixel_emb, device)

    print(f"开始 制图")

    # 4. t-SNE（使用 max_iter 兼容新版 sklearn）
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000,
                init='pca', random_state=42, learning_rate='auto')
    
    embed_no = tsne.fit_transform(rgb_features)
    embed_with = tsne.fit_transform(pe_features)

        # ================== 修复 colors 生成 ==================
    # 原代码是错的，会少一个
    # colors = np.repeat([0, 1, 2], 256//3)[:256]          # ← 删除这行

    # 推荐写法：按主导通道着色（0=R, 1=G, 2=B）
    rgb_flat = patch.reshape(256, 3)
    colors = np.argmax(rgb_flat, axis=1)                   # shape: (256,)

    print(f"embed_no.shape   = {embed_no.shape}")
    print(f"embed_with.shape = {embed_with.shape}")
    print(f"colors len       = {len(colors)}")

    # 长度对齐（以防万一模型输出多 token）
    n = len(colors)
    embed_no   = embed_no[:n]
    embed_with = embed_with[:n]

    # 如果你确定是 ViT 风格且第一个是 CLS，可以这样写（但目前看起来不是）：
    # if embed_no.shape[0] == n + 1:
    #     embed_no = embed_no[1:]
    # if embed_with.shape[0] == n + 1:
    #     embed_with = embed_with[1:]
    # ======================================================

    fig = plot_comparison(
        embed_no, embed_with, colors,
        title_prefix=f"Sample #{args.idx} - Kodak 16×16 Patch"
    )
    # # 6. 绘图 & 保存
    # fig = plot_comparison(embed_no, embed_with, colors,
    #                       title_prefix=f"Sample #{args.idx} - Kodak 16×16 Patch")
    # 自动创建目录（如果不存在）
    save_dir = os.path.dirname(args.save)          # 提取目录部分，例如 "results"
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)       # 创建目录，exist_ok 防止重复创建报错
        print(f"已创建目录: {save_dir}")

    # 保存为 JPEG（质量较高，适合大多数场景）
    jpeg_path = args.save.replace('.pdf', '.jpg')   # 或 .jpeg 也行

    # fig.savefig(args.save, bbox_inches='tight', format='pdf')      # 保留 PDF 如果你还想要
    # fig.savefig(jpeg_path, bbox_inches='tight', dpi=300, format='jpg', quality=95)

    # 保存 JPEG，使用 pil_kwargs 控制质量
    fig.savefig(
        jpeg_path,
        bbox_inches='tight',
        dpi=300,                # 分辨率（可调 150~400）
        format='jpg',
        pil_kwargs={
            'quality': 92,      # 85~95 通常是性价比最高的范围
            'optimize': True,   # 优化文件大小
            'progressive': True #  progressive JPEG，网页加载更快（可选）
        }
    )

    print(f"✅ 保存成功:")
    print(f"  PDF:  {args.save}")
    print(f"  JPEG: {jpeg_path}")

    # 7. 量化指标（论文必备）
    print(f"Without PE Silhouette: {silhouette_score(embed_no, colors):.4f}")
    print(f"With PE Silhouette:    {silhouette_score(embed_with, colors):.4f}")

if __name__ == "__main__":
    main()