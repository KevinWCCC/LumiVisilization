#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
从 jsonl 数据集中读取一个 16×16 patch，
对所有 sub-pixel 使用 t-SNE 降维并可视化
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.cm as cm


def load_one_patch(jsonl_path, patch_size=16, target_idx=0):
    """
    从按通道分离的 jsonl 文件中读取第 target_idx 个 16×16 patch，
    返回 shape=(H,W,3) 的 numpy array，值域 [0,1]
    """
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i != target_idx:
                continue
            data = json.loads(line.strip())
            
            required_keys = {'R', 'G', 'B'}
            if not required_keys.issubset(data):
                raise KeyError(f"缺少必要字段: {required_keys - set(data)}")
            
            r = np.array(data['R'], dtype=np.float32)
            g = np.array(data['G'], dtype=np.float32)
            b = np.array(data['B'], dtype=np.float32)
            
            expected_len = patch_size * patch_size
            if not (len(r) == len(g) == len(b) == expected_len):
                raise ValueError(
                    f"通道长度不匹配或不为 {expected_len}: "
                    f"R={len(r)}, G={len(g)}, B={len(b)}"
                )
            
            # stack 成 (H, W, 3)，row-major
            patch = np.stack([r, g, b], axis=-1).reshape(patch_size, patch_size, 3)
            
            # 归一化到 [0,1]
            patch = patch / 255.0
            
            meta = {
                k: data[k] for k in data 
                if k not in {'R','G','B'}
            }
            return patch, meta
    
    raise IndexError(f"文件中样本数量少于 {target_idx+1} 个")

def extract_features(patch, feature_type="rgb"):
    """
    为每个 sub-pixel 提取特征向量
    
    参数:
        patch: (16,16,3) numpy array, 值域 [0,1]
        feature_type:
            "rgb"       → 直接使用 RGB 值 (3维)
            "position"  → 使用像素的 (i,j) 坐标 (2维)
            "rgb+pos"   → 拼接 RGB + (i/15, j/15) 归一化坐标 (5维)
            "channel"   → 只用通道索引 (1维，主要是为了着色)
    
    返回:
        features: (256, dim) 特征矩阵
        colors:   (256,) 用于着色的值（通常 0~255 或通道索引）
    """
    h, w, c = patch.shape
    n_pixels = h * w
    i, j = np.mgrid[0:h, 0:w]
    i = i.ravel()
    j = j.ravel()
    
    rgb = patch.reshape(n_pixels, 3)          # (256,3)
    
    if feature_type == "rgb":
        features = rgb
        colors = np.arange(n_pixels) % 256       # 随便着色，或后续改
    elif feature_type == "position":
        features = np.stack([i/15.0, j/15.0], axis=-1)   # 归一化到 [0,1]
        colors = np.sqrt(i**2 + j**2)            # 距离中心远近
    elif feature_type == "rgb+pos":
        pos = np.stack([i/15.0, j/15.0], axis=-1)
        features = np.concatenate([rgb, pos], axis=-1)   # (256,5)
        colors = np.arange(n_pixels) % 256
    elif feature_type == "channel":
        features = rgb
        colors = np.tile([0,1,2], n_pixels//3)[:n_pixels]  # R=0,G=1,B=2
    else:
        raise ValueError(f"不支持的 feature_type: {feature_type}")
    
    return features, colors


def plot_tsne_2d(embed_2d, colors, title, cmap="tab10", s=60, alpha=0.85):
    """
    绘制 t-SNE 散点图
    """
    plt.figure(figsize=(9, 7), dpi=120)
    
    scatter = plt.scatter(
        embed_2d[:, 0], embed_2d[:, 1],
        c=colors,
        cmap=cmap,
        s=s,
        alpha=alpha,
        edgecolors='none'
    )
    
    plt.colorbar(scatter, label="color encoding")
    plt.title(title, fontsize=14, pad=12)
    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="t-SNE 可视化单个 16×16 patch 的像素点")
    parser.add_argument("--jsonl_path", type=str, help="处理后的 .jsonl 文件路径")
    parser.add_argument("--idx", type=int, default=0, help="读取第几个样本（从0开始）")
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--feature", choices=["rgb", "position", "rgb+pos", "channel"],
                        default="rgb", help="用于 t-SNE 的特征类型")
    parser.add_argument("--perplexity", type=float, default=30.0,
                        help="t-SNE perplexity 参数（建议 5~50）")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--save", type=str, default=None,
                        help="如果指定路径，则保存图片而非显示")

    args = parser.parse_args()

    # 1. 读取 patch
    print(f"读取文件: {args.jsonl_path}  第 {args.idx} 个样本...")
    patch, meta = load_one_patch(args.jsonl_path, args.patch_size, args.idx)
    print("patch shape:", patch.shape)
    print("min/max value:", patch.min(), patch.max())   # 应接近 0 和 1
    print("meta keys:", list(meta.keys()))
    if 'long_caption' in meta:
        print("caption:", meta['long_caption'])
    print("patch shape:", patch.shape)
    if "meta" in meta:
        print("meta:", meta)

    # 2. 提取特征
    features, colors = extract_features(patch, args.feature)
    print(f"特征维度: {features.shape[1]}   像素数量: {features.shape[0]}")

    # 3. t-SNE 降维
    print("执行 t-SNE ...")
    tsne = TSNE(
        n_components=2,
        perplexity=args.perplexity,
        learning_rate="auto",
        max_iter=1000,                # ← 这里改成 max_iter
        n_iter_without_progress=300,  # 可选，保持原样
        min_grad_norm=1e-07,
        metric='euclidean',
        init="pca",                   # 通常比 random 更稳定
        random_state=args.random_state
    )
    embed_2d = tsne.fit_transform(features)
    print("t-SNE 完成，形状:", embed_2d.shape)

    # 4. 绘图
    title = (
        f"t-SNE of 16×16 patch pixels\n"
        f"file: {args.jsonl_path}  sample #{args.idx}  "
        f"feature={args.feature}  perplexity={args.perplexity}"
    )
    
    plot_tsne_2d(embed_2d, colors, title)
    
    if args.save:
        plt.savefig(args.save, dpi=300, bbox_inches="tight")
        print(f"已保存到: {args.save}")
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    main()