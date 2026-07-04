#!/usr/bin/env python3
"""
t-SNE 可视化脚本（已修复版）
对比：Tokenizer Embedding (ASCII Tokens 0~127) vs 您的 PixelInputEmbedding (Numerical Tokens 0~255)
"""

import os
import argparse
import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM

# ====================== 自动导入您的模块 ======================
try:
    from readout_head import PixelInputEmbedding, PixelInputEmbeddingConfig
    HAS_CUSTOM_MODULES = True
except ImportError:
    print("[Warning] 未找到 readout_head.py，使用 fallback 简单 Embedding（仅 demo）")
    HAS_CUSTOM_MODULES = False
    import torch.nn as nn

    class PixelInputEmbeddingConfig:
        def __init__(self, hidden_size: int):
            self.hidden_size = hidden_size

    class PixelInputEmbedding(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.embedding = nn.Embedding(256, config.hidden_size)

        def forward(self, x):
            return self.embedding(x)

        @classmethod
        def load(cls, path, device="cpu"):
            state = torch.load(path, map_location=device)
            cfg = PixelInputEmbeddingConfig(2048)
            model = cls(cfg)
            model.load_state_dict(state)
            return model.to(device)


def main():
    parser = argparse.ArgumentParser(description="t-SNE 对比 Tokenizer vs Pixel Embedding（已修复版）")
    parser.add_argument("--model_id", type=str, default="/home/vipuser/Model/LLAMA_3.1_B")
    parser.add_argument("--pixel_emb_ckpt", type=str, default="models/pixel_emb_dataset_stage1_best.pt")
    parser.add_argument("--output", type=str, default="tsne_ascii_vs_pixel.png")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--max_iter", type=int, default=1000, help="t-SNE 最大迭代次数（新版sklearn用max_iter）")
    parser.add_argument("--ascii_range", type=int, nargs=2, default=[0, 128])
    args = parser.parse_args()

    device = args.device
    print(f"[Info] 设备: {device} | 模型: {args.model_id}")

    # ====================== 1. 加载模型 ======================
    print("[1/5] 加载 Tokenizer + LLM ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,  # ← 已修复
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    emb = model.get_input_embeddings()
    H = emb.weight.shape[1]
    print(f"     Hidden size = {H}, Vocab size = {emb.weight.shape[0]}")

    # ====================== 2. ASCII Tokens ======================
    print("[2/5] 提取 ASCII Tokens embedding ...")
    start_id, end_id = args.ascii_range
    ascii_ids = list(range(start_id, end_id))
    with torch.no_grad():
        ascii_embs = emb.weight[ascii_ids].float().cpu().numpy()
    print(f"     ASCII embeddings shape: {ascii_embs.shape}")

    # ====================== 3. Pixel Embedding ======================
    print("[3/5] 加载 PixelInputEmbedding ...")
    pixel_cfg = PixelInputEmbeddingConfig(hidden_size=H)

    if HAS_CUSTOM_MODULES and os.path.exists(args.pixel_emb_ckpt):
        pixel_emb = PixelInputEmbedding.load(args.pixel_emb_ckpt, device=device)
        print(f"     ✓ 已加载: {args.pixel_emb_ckpt}")
    else:
        if os.path.exists(args.pixel_emb_ckpt):
            pixel_emb = PixelInputEmbedding(pixel_cfg).to(device)
            pixel_emb.load_state_dict(torch.load(args.pixel_emb_ckpt, map_location=device))
            print(f"     ✓ 已加载 checkpoint（fallback模式）")
        else:
            pixel_emb = PixelInputEmbedding(pixel_cfg).to(device)
            print(f"     ⚠ Checkpoint不存在，使用随机初始化（请检查 --pixel_emb_ckpt 路径！）")

    pixel_emb.eval()
    with torch.no_grad():
        pix_tensor = torch.arange(256, device=device).unsqueeze(0)
        pixel_embs = pixel_emb(pix_tensor).squeeze(0).float().cpu().numpy()
    print(f"     Pixel embeddings shape: {pixel_embs.shape}")

    # ====================== 4. t-SNE（已修复 n_iter → max_iter） ======================
    print("[4/5] 执行 t-SNE 降维 ...")
    tsne_params = {
        "n_components": 2,
        "perplexity": args.perplexity,
        "max_iter": args.max_iter,   # ← 已修复
        "random_state": 42,
        "init": "pca",
        "verbose": 1,
    }
    ascii_2d = TSNE(**tsne_params).fit_transform(ascii_embs)

    tsne_params["random_state"] = 43
    pixel_2d = TSNE(**tsne_params).fit_transform(pixel_embs)

    # ====================== 5. 绘图 ======================
    print("[5/5] 绘制并保存图片 ...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)

    # (a) ASCII
    ax1 = axes[0]
    scatter1 = ax1.scatter(ascii_2d[:, 0], ascii_2d[:, 1],
                           c=ascii_ids, cmap="viridis", s=35, alpha=0.85, edgecolors="none")
    ax1.set_title("(a) Tokenizer Embedding", fontsize=13, fontweight="bold", pad=10)
    ax1.set_xlabel("t-SNE Dimension 1", fontsize=10)
    ax1.set_ylabel("t-SNE Dimension 2", fontsize=10)
    ax1.grid(True, alpha=0.25, linestyle="--")
    cbar1 = fig.colorbar(scatter1, ax=ax1, shrink=0.75, pad=0.02)
    cbar1.set_label("Token ID (0-127)", fontsize=9)

    # (b) Pixel
    ax2 = axes[1]
    scatter2 = ax2.scatter(pixel_2d[:, 0], pixel_2d[:, 1],
                           c=np.arange(256), cmap="viridis", s=28, alpha=0.85, edgecolors="none")
    ax2.set_title("With 7-D Pixel Embedding", fontsize=13, fontweight="bold", pad=10)
    ax2.set_xlabel("t-SNE Dimension 1", fontsize=10)
    ax2.set_ylabel("t-SNE Dimension 2", fontsize=10)
    ax2.grid(True, alpha=0.25, linestyle="--")
    cbar2 = fig.colorbar(scatter2, ax=ax2, shrink=0.75, pad=0.02)
    cbar2.set_label("Pixel Value (0-255)", fontsize=9)

    fig.suptitle(
        f"t-SNE Visualization: Tokenizer vs Pixel Embedding\n"
        f"Model: {os.path.basename(args.model_id)}  |  "
        f"Pixel CKPT: {os.path.basename(args.pixel_emb_ckpt) if os.path.exists(args.pixel_emb_ckpt) else 'random init'}",
        fontsize=11, y=0.98
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(args.output, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"\n✅ 已保存: {os.path.abspath(args.output)}")
    plt.close()


if __name__ == "__main__":
    main()