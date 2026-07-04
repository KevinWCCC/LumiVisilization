#!/usr/bin/env python
"""
t-SNE Visualization: Original LLM vs Trained Pixel Model
最终修复版：长度严格对齐 + 可控点数 + 调试信息
支持 tokenizer 原始模式（不吃 PE/INP）
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
    parser = argparse.ArgumentParser(description="t-SNE 可视化 - 控制点数")
    parser.add_argument("--model_id", type=str, default="/home/vipuser/Model/LLAMA_3.1_B")
    parser.add_argument("--eval_jsonl", type=str, default="data/kodak_p16_7-24.jsonl")
    parser.add_argument("--format_config", type=str, default="configs/default_format.yaml")
    
    parser.add_argument("--pixel_emb_ckpt", type=str, default=None,
                        help="Path to trained PixelInputEmbedding ckpt. If None or not exist, "
                             "run in original tokenizer mode (no PE/INP).")
    parser.add_argument("--prefix_ckpt", type=str, default=None)
    parser.add_argument("--intra_pos_ckpt", type=str, default=None)
    
    # ================== 可配置参数 ==================
    parser.add_argument("--num_patches", type=int, default=8, help="要处理的 patch 数量（默认8个，快速调试）")
    parser.add_argument("--max_points", type=int, default=12000, help="t-SNE 最终使用的最大点数")
    parser.add_argument("--perplexity", type=int, default=40)
    parser.add_argument("--max_iter", type=int, default=2000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default="tsne_pixel_hidden_comparison.png")
    parser.add_argument("--seed", type=int, default=42)
    
    return parser.parse_args()


@torch.no_grad()
def extract_pixel_hidden_states(model, pixel_emb, prefix_tau, intra_pos,
                                prompt_ids_list, pixel_values_list, device, tokenizer=None):
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
        # --- tokenizer方式的原始模式（不吃PE/INP）---
        if tokenizer is None:
            raise RuntimeError("tokenizer mode requires tokenizer to be passed to extract_pixel_hidden_states")
        _, id2val = utils.build_numeric_maps(tokenizer)
        val2tid = {v: tid for tid, v in id2val.items()}
        pixel_tids = [val2tid[int(v)] for v in pixel_values_list]
        full_ids = prompt_ids_list + pixel_tids
        prompt_t = torch.tensor([full_ids], dtype=torch.long, device=device)
        out = model(input_ids=prompt_t, output_hidden_states=True, use_cache=False)
        hs = out.hidden_states[-1]
        T = len(pixel_values_list)
        return hs[0, -T:, :].cpu().numpy()

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
    
    print(f"[INFO] Loading trained components...")
    pixel_emb = None
    if args.pixel_emb_ckpt and os.path.exists(args.pixel_emb_ckpt):
        pixel_emb = PixelInputEmbedding.load(args.pixel_emb_ckpt, device=device)
        pixel_emb.eval()
        print(f"   Loaded pixel_emb: {args.pixel_emb_ckpt}")
    else:
        print("[INFO] pixel_emb_ckpt not provided or not found → using original tokenizer mode (no PE/INP)")
    
    prefix_tau = None
    if args.prefix_ckpt and os.path.exists(args.prefix_ckpt):
        prefix_tau, _ = load_prefix(args.prefix_ckpt, device=device, model=original_llm)
        print(f"   Loaded prefix: {args.prefix_ckpt}")
    
    intra_pos = None
    if args.intra_pos_ckpt and os.path.exists(args.intra_pos_ckpt):
        intra_pos = load_intra_pos(args.intra_pos_ckpt, original_llm, device)
        print(f"   Loaded intra_pos: {args.intra_pos_ckpt}")
    
    format_args = load_data_format_config(args.format_config)
    dataset = PixelJsonlDataset(args.eval_jsonl.split(','), tokenizer, format_args,
                                shuffle=False, with_full_img=False, with_neighbor_patches=False)
    
    all_hidden = []
    all_labels = []
    all_pixel_values = []
    
    mode_str = "tokenizer-original (no PE/INP)" if pixel_emb is None else "pixel-vs-trained"
    print(f"[INFO] 提取 {args.num_patches} 个 patch 的 hidden states (mode: {mode_str})...")
    
    for idx in tqdm(range(min(args.num_patches, len(dataset)))):
        prompt_ids, pixel_values, _, _, _ = dataset[idx]
        
        hs_orig = extract_pixel_hidden_states(original_llm, None, None, None,
                                              prompt_ids, pixel_values, device, tokenizer=tokenizer)
        all_hidden.append(hs_orig)
        all_labels.extend(["Original"] * len(hs_orig))
        all_pixel_values.extend(pixel_values)
        
        if pixel_emb is not None:
            hs_trained = extract_pixel_hidden_states(original_llm, pixel_emb, prefix_tau, intra_pos,
                                                     prompt_ids, pixel_values, device, tokenizer=tokenizer)
            all_hidden.append(hs_trained)
            all_labels.extend(["Trained"] * len(hs_trained))
            all_pixel_values.extend(pixel_values)
    
    X = np.vstack(all_hidden)
    labels = np.array(all_labels)
    pixel_vals = np.array(all_pixel_values)
    
    # 安全检查：长度必须严格一致
    if len(X) != len(pixel_vals):
        raise RuntimeError(
            f"严重错误：hidden states 数量 ({len(X)}) 与 pixel_vals 数量 ({len(pixel_vals)}) 不一致！\n"
            f"请检查循环中 append/extend 是否匹配。"
        )
    
    # ================== 严格控制点数 ==================
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
                    palette={"Original": "#1f77b4", "Trained": "#d62728"})
    
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=pixel_vals, cmap='viridis',
                          alpha=0.6, s=25, edgecolors='none')
    plt.colorbar(scatter, label="Pixel Value (0-255)")
    
    plt.title("t-SNE of Pixel Hidden States\nOriginal LLM vs Trained Pixel Model", fontsize=16)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f"[INFO] Saved to: {args.output}")
    # plt.show()


if __name__ == "__main__":
    main()