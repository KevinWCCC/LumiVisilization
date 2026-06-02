#!/usr/bin/env python
"""
t-SNE Visualization: Original LLM vs Trained Pixel Model (Flexible Components)
支持灵活传入组件：
  - 只传 --pixel_emb_ckpt          → 对比 Original vs PE(+Heading)
  - 同时传 --pixel_emb_ckpt + --intra_pos_ckpt → 对比 Original vs PE+INP(+Heading)
  - 可额外传入 --prefix_ckpt
"""

import os
import torch
import numpy as np
import argparse
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import utils
from readout_head import PixelInputEmbedding
from soft_prefix import load_prefix
from intra_patch_pos_embed import load_intra_pos
from data_loader import PixelJsonlDataset, load_data_format_config

torch.set_grad_enabled(False)


def parse_args():
    parser = argparse.ArgumentParser(description="t-SNE 可视化 - 支持灵活组件加载")
    parser.add_argument("--model_id", type=str, default="/home/vipuser/Model/LLAMA_3.1_B")
    parser.add_argument("--eval_jsonl", type=str, default="data/kodak_p16_7-24.jsonl")
    parser.add_argument("--format_config", type=str, default="configs/default_format.yaml")
    
    # 必传
    parser.add_argument("--pixel_emb_ckpt", type=str, required=True,
                        help="PixelInputEmbedding 的 checkpoint（必须提供）")
    
    # 可选组件
    parser.add_argument("--intra_pos_ckpt", type=str, default=None,
                        help="IntraPatchPositionEmbedding 的 checkpoint（可选）")
    parser.add_argument("--prefix_ckpt", type=str, default=None,
                        help="SoftPrefix 的 checkpoint（可选）")
    
    # 可配置参数
    parser.add_argument("--num_patches", type=int, default=8, help="要处理的 patch 数量")
    parser.add_argument("--max_points", type=int, default=12000, help="t-SNE 最大点数")
    parser.add_argument("--perplexity", type=int, default=40)
    parser.add_argument("--max_iter", type=int, default=2000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default="tsne_comparison.png")
    parser.add_argument("--seed", type=int, default=42)
    
    return parser.parse_args()


@torch.no_grad()
def extract_pixel_hidden_states(model, pixel_emb, prefix_tau, intra_pos,
                                prompt_ids_list, pixel_values_list, device):
    prompt_t = torch.tensor([prompt_ids_list], dtype=torch.long, device=device)
    pix_t = torch.tensor([pixel_values_list], dtype=torch.long, device=device)
    
    emb = model.get_input_embeddings()
    tok_embs = emb(prompt_t)
    
    if pixel_emb is not None:
        pix_embs = pixel_emb(pix_t)
        if intra_pos is not None:
            token_idx = torch.arange(len(pixel_values_list), device=device)
            pos = intra_pos(token_idx).unsqueeze(0)
            pix_embs = pix_embs + pos
        if prefix_tau is not None:
            tau_b = prefix_tau.unsqueeze(0).to(device)
            inputs_embeds = torch.cat([tau_b, tok_embs, pix_embs], dim=1)
            base = prefix_tau.size(0)
        else:
            inputs_embeds = torch.cat([tok_embs, pix_embs], dim=1)
            base = 0
    else:
        T = len(pixel_values_list)
        pix_embs = torch.zeros((1, T, tok_embs.size(-1)), device=device, dtype=tok_embs.dtype)
        inputs_embeds = torch.cat([tok_embs, pix_embs], dim=1)
        base = 0

    attn = torch.ones((1, inputs_embeds.size(1)), device=device, dtype=torch.long)
    out = model(inputs_embeds=inputs_embeds, attention_mask=attn,
                output_hidden_states=True, use_cache=False)
    hs = out.hidden_states[-1]
    
    T = len(pixel_values_list)
    return hs[0, base-1 : base-1 + T, :].cpu().numpy()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device
    
    print(f"[INFO] Loading original model: {args.model_id}")
    tokenizer, original_llm = utils.load_llm_model(args.model_id, "", device)
    original_llm.eval()
    
    # ==================== 加载组件（灵活支持） ====================
    print(f"[INFO] Loading PixelInputEmbedding from {args.pixel_emb_ckpt}")
    pixel_emb = PixelInputEmbedding.load(args.pixel_emb_ckpt, device=device)
    pixel_emb.eval()
    
    intra_pos = None
    if args.intra_pos_ckpt and os.path.exists(args.intra_pos_ckpt):
        print(f"[INFO] Loading IntraPatchPositionEmbedding from {args.intra_pos_ckpt}")
        intra_pos = load_intra_pos(args.intra_pos_ckpt, original_llm, device)
    
    prefix_tau = None
    if args.prefix_ckpt and os.path.exists(args.prefix_ckpt):
        print(f"[INFO] Loading SoftPrefix from {args.prefix_ckpt}")
        prefix_tau, _ = load_prefix(args.prefix_ckpt, device=device, model=original_llm)
    
    # 动态生成配置名称
    config_name = "PE"
    if intra_pos is not None:
        config_name += "+INP"
    if prefix_tau is not None:
        config_name += "+Prefix"
    print(f"[INFO] Current trained configuration: {config_name}")
    
    # ==================== 数据加载 ====================
    format_args = load_data_format_config(args.format_config)
    dataset = PixelJsonlDataset(args.eval_jsonl.split(','), tokenizer, format_args,
                                shuffle=False, with_full_img=False, with_neighbor_patches=False)
    
    all_hidden = []
    all_labels = []
    all_pixel_values = []
    
    print(f"[INFO] 提取 {args.num_patches} 个 patch 的 hidden states...")
    for idx in tqdm(range(min(args.num_patches, len(dataset)))):
        prompt_ids, pixel_values, _, _, _ = dataset[idx]
        
        # Original LLM
        hs_orig = extract_pixel_hidden_states(original_llm, None, None, None,
                                              prompt_ids, pixel_values, device)
        
        # Trained 配置（可灵活开关 INP 和 Prefix）
        hs_trained = extract_pixel_hidden_states(original_llm, pixel_emb, prefix_tau, intra_pos,
                                                 prompt_ids, pixel_values, device)
        
        all_hidden.append(hs_orig)
        all_hidden.append(hs_trained)
        all_labels.extend(["Original"] * len(hs_orig) + [config_name] * len(hs_trained))
        all_pixel_values.extend(pixel_values + pixel_values)
    
    X = np.vstack(all_hidden)
    labels = np.array(all_labels)
    pixel_vals = np.array(all_pixel_values)
    
    # ================== 子采样 ==================
    if len(X) > args.max_points:
        print(f"[INFO] 总点数 {len(X)} 超过上限 {args.max_points}，随机子采样...")
        indices = np.random.choice(len(X), size=args.max_points, replace=False)
        X = X[indices]
        labels = labels[indices]
        pixel_vals = pixel_vals[indices]
    else:
        print(f"[INFO] 当前总点数 {len(X)}（小于上限）")
    
    print(f"[INFO] Running t-SNE on {len(X)} points...")
    tsne = TSNE(
        n_components=2,
        perplexity=args.perplexity,
        max_iter=args.max_iter,
        random_state=args.seed,
        init='pca'
    )
    X_2d = tsne.fit_transform(X)
    
    # ================== 可视化 ==================
    plt.figure(figsize=(14, 10))
    sns.scatterplot(x=X_2d[:, 0], y=X_2d[:, 1], hue=labels, alpha=0.7, s=60,
                    palette={"Original": "#1f77b4", config_name: "#d62728"})
    
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=pixel_vals, cmap='viridis',
                          alpha=0.6, s=25, edgecolors='none')
    plt.colorbar(scatter, label="Pixel Value (0-255)")
    
    plt.title(f"t-SNE of Pixel Hidden States\nOriginal LLM vs {config_name}", fontsize=16)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(title="Configuration", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f"[INFO] Saved to: {args.output}")
    plt.show()


if __name__ == "__main__":
    main()