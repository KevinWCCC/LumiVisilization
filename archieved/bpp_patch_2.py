# 文件名：visualize_patch_bpp_ablation_v4.py
import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

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
    pixel_values_patch,
    prefix_tau=None,
    intra_pos=None,
    device="cuda"
):
    num_subpixels = len(pixel_values_patch)
    if num_subpixels % 3 != 0:
        raise ValueError(f"pixel_values_patch 长度必须是 3 的倍数，实际 {num_subpixels}")

    num_pixels = num_subpixels // 3
    side = int(np.sqrt(num_pixels))
    if side * side != num_pixels:
        raise ValueError(f"像素数 {num_pixels} 不是完美平方，无法形成 {side}×{side} patch")

    prompt_t = torch.tensor(prompt_ids_list, dtype=torch.long, device=device).unsqueeze(0)
    pix_t = torch.tensor(pixel_values_patch, dtype=torch.long, device=device).unsqueeze(0)

    Lp = prompt_t.size(1)
    T = pix_t.size(1)

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
    bpp_map = bits_per_pixel.reshape(side, side).cpu().numpy()

    return bpp_map, side


def plot_single(patch_pixels, bpp_map, side, title, save_path=None):
    patch_img = np.array(patch_pixels).reshape(side, side, 3).astype(np.uint8)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5), dpi=180)

    axs[0].imshow(patch_img)
    axs[0].set_title(f"{side}×{side} Patch", fontsize=13)
    axs[0].axis("off")

    im = axs[1].imshow(bpp_map, cmap="inferno", interpolation="nearest")
    axs[1].set_title("Bit Consumption Map", fontsize=13)
    axs[1].axis("off")

    cbar = fig.colorbar(im, ax=axs[1], fraction=0.046, pad=0.04)
    cbar.set_label("Bits per Pixel", rotation=270, labelpad=15)

    mean_bpp = bpp_map.mean()
    fig.suptitle(f"{title}\nAvg BPP = {mean_bpp:.4f}", fontsize=14)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"保存到：{save_path}")


def plot_comparison(results, patch_pixels, side, save_path):
    fig, axs = plt.subplots(2, 2, figsize=(14, 12), dpi=180)
    axs = axs.flatten()
    for i, (name, bpp_map, avg_bpp) in enumerate(results):
        patch_img = np.array(patch_pixels).reshape(side, side, 3).astype(np.uint8)
        axs[i].imshow(patch_img)
        im = axs[i].imshow(bpp_map, cmap="inferno", alpha=0.85)
        axs[i].set_title(f"{name}\nAvg BPP = {avg_bpp:.4f}", fontsize=13)
        axs[i].axis("off")
    cbar = fig.colorbar(im, ax=axs, fraction=0.02, pad=0.04)
    cbar.set_label("Bits per Pixel", rotation=270, labelpad=15)
    fig.suptitle(f"Ablation Comparison - {side}×{side} Patch", fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"对比图已保存 → {save_path}")


def add_tag_to_filename(base_name, tag):
    """在文件名末尾加上 _tag-xxx（如果有 tag）"""
    if not tag:
        return base_name
    stem, ext = os.path.splitext(base_name)
    return f"{stem}_tag-{tag}{ext}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="可视化任意大小 patch 的 BPP 热力图（支持 ablation 对比 & tag）")
    parser.add_argument("--eval_jsonl", type=str, required=True)
    parser.add_argument("--format_config", type=str, default="configs/default_format.yaml")
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--patch_start_pixel", type=int, default=0)
    parser.add_argument("--patch_size", type=int, default=16,
                        help="patch 边长（像素），必须使像素数为 k*k，例如 8/16/32/64")
    parser.add_argument("--compare", action="store_true", help="开启 ablation 对比模式")
    parser.add_argument("--tag", type=str, default=None, help="自定义 tag，保存文件名时会加在末尾")
    parser.add_argument("--output", type=str, default=None)

    # 模型参数
    # parser.add_argument("--model_id", type=str, default="/home/vipuser/Model/QWEN_3_0.6B")
    parser.add_argument("-m", "--model_id", type=str, default="/home/vipuser/Model/LLAMA_3.1_B")

    parser.add_argument("--pixel_emb", type=str, default="models/pixel_emb_k.pt")
    parser.add_argument("--readout_head", type=str, default="models/HeadPixel_k.pt")
    parser.add_argument("--prefix", type=str, default=None)
    parser.add_argument("--intra_pos", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    # 加载模型
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

    # 加载 dataset
    format_args = load_data_format_config(args.format_config)
    dataset = PixelJsonlDataset(
        args.eval_jsonl.split(','),
        tokenizer,
        format_args,
        shuffle=False,
        with_full_img=False,
        with_neighbor_patches=False
    )

    sample = dataset[args.sample_idx]
    prompt_ids_list, pixel_values_list, _, _, meta = sample

    total_pixels = len(pixel_values_list) // 3
    print(f"样本 {args.sample_idx}：总像素数 = {total_pixels}")

    # 计算所需长度
    patch_size = args.patch_size
    num_pixels_needed = patch_size * patch_size
    num_subpixels_needed = num_pixels_needed * 3

    if args.patch_start_pixel + num_pixels_needed > total_pixels:
        max_start = total_pixels - num_pixels_needed
        raise ValueError(
            f"起始位置 {args.patch_start_pixel} + patch {patch_size}×{patch_size} "
            f"({num_pixels_needed} 像素) 超出图像边界（总像素 {total_pixels}）\n"
            f"建议最大起始像素 = {max_start}（或更小）"
        )

    start_idx = args.patch_start_pixel * 3
    pixel_values_patch = pixel_values_list[start_idx : start_idx + num_subpixels_needed]

    print(f"截取 {patch_size}×{patch_size} patch（{num_pixels_needed} 像素，{num_subpixels_needed} sub-pixels）")

    # 计算
    if args.compare:
        configs = [
            ("Full (PE+INP+Prefix)", prefix_tau, intra_pos_full),
            ("PE only", prefix_tau, None),
            ("INP only", None, intra_pos_full),
            ("None", None, None),
        ]
        results = []
        for name, p_tau, i_pos in configs:
            bpp_map, side = compute_patch_bpp(
                model, pixel_emb, readout_head, prompt_ids_list, pixel_values_patch,
                prefix_tau=p_tau, intra_pos=i_pos, device=args.device
            )
            avg = bpp_map.mean()
            results.append((name, bpp_map, avg))
            print(f"{name:18} Avg BPP = {avg:.4f}")

        base_name = args.output or f"ablation_{Path(args.eval_jsonl).stem}_s{args.sample_idx}_p{args.patch_start_pixel}_size{patch_size}.png"
        out_path = add_tag_to_filename(base_name, args.tag)
        plot_comparison(results, pixel_values_patch, patch_size, out_path)
    else:
        bpp_map, side = compute_patch_bpp(
            model, pixel_emb, readout_head, prompt_ids_list, pixel_values_patch,
            prefix_tau=prefix_tau, intra_pos=intra_pos_full, device=args.device
        )
        avg = bpp_map.mean()
        print(f"Full Avg BPP = {avg:.4f}")

        base_name = args.output or f"patch_bpp_{Path(args.eval_jsonl).stem}_s{args.sample_idx}_p{args.patch_start_pixel}_size{patch_size}.png"
        out_path = add_tag_to_filename(base_name, args.tag)
        plot_single(pixel_values_patch, bpp_map, patch_size, "Full Model", out_path)

    print("完成！")