# 文件名：visualize_full_image_bpp.py
# 功能：从 kodak_p16 jsonl 中取出连续多个 16×16 patch → 拼接成完整大图 → 计算整图 per-pixel BPP → 绘制原图 + 完整 BPP 热力图
# 使用示例：
#   python visualize_full_image_bpp.py \
#     --eval_jsonl data/kodak_p16_7-24.jsonl \
#     --start_patch 0 \
#     --grid_rows 32 \
#     --grid_cols 32 \
#     --tag kodim01

import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

from readout_head import NumericReadoutHead, PixelInputEmbedding
from soft_prefix import load_prefix
from intra_patch_pos_embed import load_intra_pos
import utils
from data_loader import PixelJsonlDataset, load_data_format_config


@torch.no_grad()
def compute_full_bpp(
    model,
    pixel_emb,
    readout_head,
    prompt_ids_list,
    all_pixel_values,     # 完整图像展平后的所有 sub-pixel 值
    prefix_tau=None,
    intra_pos=None,
    device="cuda"
):
    T = len(all_pixel_values)
    prompt_t = torch.tensor(prompt_ids_list, dtype=torch.long, device=device).unsqueeze(0)
    pix_t = torch.tensor(all_pixel_values, dtype=torch.long, device=device).unsqueeze(0)

    Lp = prompt_t.size(1)

    emb = model.get_input_embeddings()
    dtype = emb.weight.dtype

    prompt_emb = emb(prompt_t).to(dtype)
    pix_emb = pixel_emb(pix_t).to(dtype)

    if intra_pos is not None:
        token_idx = torch.arange(T, device=device, dtype=torch.long)
        pos = intra_pos(token_idx).unsqueeze(0).to(dtype)
        pix_emb = pix_emb + pos

    if prefix_tau is not None:
        tau_b = prefix_tau.to(device=device, dtype=dtype).unsqueeze(0)
        seq = torch.cat([tau_b, prompt_emb, pix_emb], dim=1)
        prefix_len = tau_b.size(1)
    else:
        seq = torch.cat([prompt_emb, pix_emb], dim=1)
        prefix_len = 0

    attn = torch.ones((1, seq.size(1)), dtype=torch.long, device=device)

    base = getattr(model, "model", None)
    if base is not None:
        hs = base(inputs_embeds=seq, attention_mask=attn, use_cache=False).last_hidden_state
    else:
        out = model(inputs_embeds=seq, attention_mask=attn,
                    output_hidden_states=True, use_cache=False)
        hs = out.hidden_states[-1]

    rows = torch.arange(prefix_len + Lp - 1, prefix_len + Lp - 1 + T, device=device)
    h_rows = hs[:, rows, :].float().squeeze(0)

    logits = readout_head(h_rows)
    logp = F.log_softmax(logits, dim=-1)
    tgt = pix_t.squeeze(0)
    logp_true = logp.gather(1, tgt.unsqueeze(1)).squeeze(1)
    bits_per_subpixel = (-logp_true) / np.log(2.0)

    bits_per_pixel = bits_per_subpixel.reshape(-1, 3).sum(dim=1)
    return bits_per_pixel.reshape(full_h, full_w).cpu().numpy()


def patches_to_full_image(patches_list, grid_rows, grid_cols):
    """把多个 16×16 patch 拼接成完整图像"""
    patch_h = patch_w = 16
    full_h = grid_rows * patch_h
    full_w = grid_cols * patch_w
    full_img = np.zeros((full_h, full_w, 3), dtype=np.uint8)

    idx = 0
    for r in range(grid_rows):
        for c in range(grid_cols):
            if idx >= len(patches_list):
                break
            patch = np.array(patches_list[idx]).reshape(16, 16, 3).astype(np.uint8)
            full_img[r*16:(r+1)*16, c*16:(c+1)*16] = patch
            idx += 1
    return full_img


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="拼接多个 16×16 patch 成完整图像 + per-pixel BPP 热力图")

    parser.add_argument("--eval_jsonl", type=str, required=True)
    parser.add_argument("--format_config", type=str, default="configs/default_format.yaml")
    parser.add_argument("--start_patch", type=int, default=0, help="从第几个 patch 开始拼接")
    parser.add_argument("--grid_rows", type=int, required=True, help="拼接后的图像行数（patch 行数）")
    parser.add_argument("--grid_cols", type=int, required=True, help="拼接后的图像列数（patch 列数）")
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)

    parser.add_argument("-m", "--model_id", type=str, default="/home/vipuser/Model/LLAMA_3.1_B")

    # parser.add_argument("--model_id", type=str, default="/home/vipuser/Model/QWEN_3_0.6B")
    parser.add_argument("--pixel_emb", type=str, default="models/pixel_emb_k.pt")
    parser.add_argument("--readout_head", type=str, default="models/HeadPixel_k.pt")
    parser.add_argument("--prefix", type=str, default=None)
    parser.add_argument("--intra_pos", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    # 加载模型与模块
    tokenizer, model = utils.load_llm_model(args.model_id, "", args.device)
    model.eval()
    model = model.float()

    pixel_emb = PixelInputEmbedding.load(args.pixel_emb, device=args.device).eval()
    readout_head = NumericReadoutHead.load(args.readout_head, device=args.device).to(torch.float32).eval()

    prefix_tau = None
    if args.prefix and os.path.exists(args.prefix):
        prefix_tau, _ = load_prefix(args.prefix, device=args.device, model=model)

    intra_pos_full = None
    if args.intra_pos and os.path.exists(args.intra_pos):
        intra_pos_full = load_intra_pos(args.intra_pos, model, args.device)

    format_args = load_data_format_config(args.format_config)
    dataset = PixelJsonlDataset(args.eval_jsonl.split(','), tokenizer, format_args,
                                shuffle=False, with_full_img=False, with_neighbor_patches=False)

    # 取出连续的 patch
    all_patches = []
    all_pixel_values = []
    for i in range(args.start_patch, args.start_patch + args.grid_rows * args.grid_cols):
        if i >= len(dataset):
            break
        _, pixel_values, _, _, _ = dataset[i]
        all_patches.append(pixel_values)
        all_pixel_values.extend(pixel_values)

    print(f"已取出 {len(all_patches)} 个 16×16 patch，拼接成 {args.grid_rows}×{args.grid_cols} 大图")

    # 拼接成完整图像
    full_image = patches_to_full_image(all_patches, args.grid_rows, args.grid_cols)

    # 计算完整图像的 per-pixel BPP
    bpp_map = compute_full_bpp(
        model, pixel_emb, readout_head,
        dataset[args.start_patch][0],  # 用第一个 patch 的 prompt
        all_pixel_values,
        prefix_tau=prefix_tau,
        intra_pos=intra_pos_full,
        device=args.device
    )

    # 绘图（Figure 7 风格）
    fig, axs = plt.subplots(1, 2, figsize=(14, 7), dpi=200)

    axs[0].imshow(full_image)
    axs[0].set_title("Reconstructed Full Image", fontsize=14)
    axs[0].axis("off")

    im = axs[1].imshow(bpp_map, cmap="inferno", interpolation="nearest")
    axs[1].set_title("Per-Pixel BPP Heatmap", fontsize=14)
    axs[1].axis("off")

    cbar = fig.colorbar(im, ax=axs[1], fraction=0.046, pad=0.04)
    cbar.set_label("Bits per Pixel", rotation=270, labelpad=20)

    mean_bpp = bpp_map.mean()
    fig.suptitle(f"Full Image BPP Visualization | Avg BPP = {mean_bpp:.4f}", fontsize=16, y=0.95)

    plt.tight_layout()

    base_name = args.output or f"full_bpp_{Path(args.eval_jsonl).stem}_start{args.start_patch}_grid{args.grid_rows}x{args.grid_cols}.png"
    out_path = base_name
    if args.tag:
        stem, ext = os.path.splitext(base_name)
        out_path = f"{stem}_tag-{args.tag}{ext}"

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"完整图像 + BPP 热力图已保存 → {out_path}")
    print("完成！")