# 文件名：visualize_patch_bpp.py
# 功能：从 PixelJsonlDataset 中取一张图像的 16×16 patch，计算每个像素的 BPP 并可视化
# 使用示例：
#   python visualize_patch_bpp.py \
#     --eval_jsonl data/kodak_p16_7-24.jsonl \
#     --pixel_emb models/pixel_emb_k.pt \
#     --readout_head models/HeadPixel_k.pt \
#     --prefix models/SP_k.pt \
#     --intra_pos models/INP_k.pt \
#     --model_id /home/vipuser/Model/QWEN_3_0.6B \
#     --format_config configs/default_format.yaml \
#     --sample_idx 0 \
#     --patch_start_pixel 0 \
#     --output patch_bpp_sample0.png

import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# 项目模块（根据你的代码结构调整导入路径）
from readout_head import NumericReadoutHead, PixelInputEmbedding
from soft_prefix import load_prefix
from intra_patch_pos_embed import load_intra_pos
import utils
from data_loader import PixelJsonlDataset, load_data_format_config


@torch.no_grad()
def compute_patch_bpp(
    model,
    pixel_emb,
    readout_head,
    prompt_ids_list,
    pixel_values_patch,       # list of int, length=768 (16×16×3)
    prefix_tau=None,
    intra_pos=None,
    device="cuda"
):
    """
    计算单个 16×16 patch 的 per-pixel BPP map
    输入：pixel_values_patch 必须正好 768 个值
    输出：(16,16) 的 numpy array
    """
    if len(pixel_values_patch) != 768:
        raise ValueError(f"期望 768 个 sub-pixel 值，实际得到 {len(pixel_values_patch)}")

    prompt_t = torch.tensor(prompt_ids_list, dtype=torch.long, device=device).unsqueeze(0)
    pix_t    = torch.tensor(pixel_values_patch, dtype=torch.long, device=device).unsqueeze(0)

    Lp = prompt_t.size(1)
    T  = pix_t.size(1)  # 768

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
    h_rows = hs[:, rows, :].float().squeeze(0)   # [768, H]

    logits = readout_head(h_rows)                 # [768, 256]
    logp = F.log_softmax(logits, dim=-1)

    tgt = pix_t.squeeze(0)
    logp_true = logp.gather(1, tgt.unsqueeze(1)).squeeze(1)
    bits_per_subpixel = (-logp_true) / np.log(2.0)   # [768]

    # 按像素求和（RGB）
    bits_per_pixel = bits_per_subpixel.reshape(-1, 3).sum(dim=1)   # [256]
    bpp_map = bits_per_pixel.reshape(16, 16).cpu().numpy()

    return bpp_map


def plot_patch_bpp(patch_pixels_list, bpp_map, sample_idx, patch_start, save_path=None):
    """
    patch_pixels_list: list of 768 ints (0~255)
    """
    patch_img = np.array(patch_pixels_list).reshape(16, 16, 3).astype(np.uint8)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5), dpi=180)

    # 原 patch
    axs[0].imshow(patch_img)
    axs[0].set_title("16×16 Patch", fontsize=13)
    axs[0].axis("off")

    # BPP 热力图
    im = axs[1].imshow(bpp_map, cmap="inferno", interpolation="nearest")
    axs[1].set_title("Bit Consumption Map\n(redder = higher BPP)", fontsize=13)
    axs[1].axis("off")

    cbar = fig.colorbar(im, ax=axs[1], fraction=0.046, pad=0.04)
    cbar.set_label("Bits per Pixel", rotation=270, labelpad=15)

    mean_bpp = bpp_map.mean()
    fig.suptitle(
        f"Sample {sample_idx} | Patch start pixel {patch_start}\n"
        f"Avg BPP = {mean_bpp:.4f}",
        fontsize=14, y=0.98
    )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"保存到：{save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="可视化数据集单张图像中一个 16×16 patch 的 BPP 热力图")
    parser.add_argument("--eval_jsonl", type=str, required=True, help="PixelJsonlDataset 的 jsonl 文件路径")
    parser.add_argument("--format_config", type=str, default="configs/default_format.yaml")
    parser.add_argument("--sample_idx", type=int, default=0, help="从 dataset 中取第几张图像")
    parser.add_argument("--patch_start_pixel", type=int, default=0,
                        help="patch 在该图像像素序列中的起始像素索引（0 ~ (总像素数-256)）")
    parser.add_argument("--output", type=str, default=None, help="输出图片路径（默认自动生成）")

    # 模型相关
    parser.add_argument("-m", "--model_id", type=str, default="/home/vipuser/Model/LLAMA_3.1_B")

    # parser.add_argument("--model_id", type=str, default="/home/vipuser/Model/QWEN_3_0.6B")
    parser.add_argument("--pixel_emb", type=str, default="models/pixel_emb_k.pt")
    parser.add_argument("--readout_head", type=str, default="models/HeadPixel_k.pt")
    parser.add_argument("--prefix", type=str, default=None)
    parser.add_argument("--intra_pos", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    # 1. 加载 tokenizer & model
    tokenizer, model = utils.load_llm_model(args.model_id, "", args.device)
    model.eval()
    model = model.float()  # 根据你的训练设置

    # 2. 加载辅助模块
    pixel_emb = PixelInputEmbedding.load(args.pixel_emb, device=args.device).eval()

    readout_head = NumericReadoutHead.load(args.readout_head, device=args.device)
    readout_head = readout_head.to(torch.float32).eval()

    prefix_tau = None
    if args.prefix and os.path.exists(args.prefix):
        prefix_tau, _ = load_prefix(args.prefix, device=args.device, model=model)
        print(f"加载 prefix: {args.prefix}")

    intra_pos = None
    if args.intra_pos and os.path.exists(args.intra_pos):
        intra_pos = load_intra_pos(args.intra_pos, model, args.device)
        print(f"加载 intra_pos: {args.intra_pos}")

    # 3. 加载 dataset 配置 & dataset
    format_args = load_data_format_config(args.format_config)

    dataset = PixelJsonlDataset(
        args.eval_jsonl.split(','),
        tokenizer,
        format_args,
        shuffle=False,
        with_full_img=False,
        with_neighbor_patches=False
    )

    if args.sample_idx < 0 or args.sample_idx >= len(dataset):
        raise ValueError(f"sample_idx 超出范围：0 ~ {len(dataset)-1}")

    # 取样本
    sample = dataset[args.sample_idx]
    prompt_ids_list, pixel_values_list, _, _, meta = sample

    print(f"样本 {args.sample_idx}：{meta.get('filename', 'unknown')}，总像素数 = {len(pixel_values_list)//3}")

    # 4. 切出 16×16 patch
    patch_size = 16
    num_pixels = patch_size * patch_size  # 256
    num_subpixels = num_pixels * 3        # 768

    max_start = (len(pixel_values_list) // 3) - num_pixels
    if args.patch_start_pixel < 0 or args.patch_start_pixel > max_start:
        raise ValueError(f"patch_start_pixel 应在 0 ~ {max_start} 之间")

    start_idx = args.patch_start_pixel * 3
    pixel_values_patch = pixel_values_list[start_idx : start_idx + num_subpixels]

    # 5. 计算 BPP map
    bpp_map = compute_patch_bpp(
        model=model,
        pixel_emb=pixel_emb,
        readout_head=readout_head,
        prompt_ids_list=prompt_ids_list,
        pixel_values_patch=pixel_values_patch,
        prefix_tau=prefix_tau,
        intra_pos=intra_pos,
        device=args.device
    )

    # 6. 可视化
    if args.output is None:
        stem = Path(args.eval_jsonl).stem
        args.output = f"patch_bpp_{stem}_s{args.sample_idx}_p{args.patch_start_pixel}.png"

    plot_patch_bpp(pixel_values_patch, bpp_map, args.sample_idx, args.patch_start_pixel, args.output)