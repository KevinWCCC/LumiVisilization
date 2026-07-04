# 文件名：visualize_bpp_map.py
# 功能：读取一张普通彩色图像 → 计算每个像素的局部 BPP（bit consumption）→ 生成热力图（颜色越红/黄 = 消耗越高）
# 效果完全对标 Figure 7 右边的可视化（原图 | Bit Consumption Heatmap）
# 使用方式：python visualize_bpp_map.py --image data/kodak/kodim01.png --output results/kodim01_bpp.png

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

# ====================== 项目模块导入 ======================
from readout_head import NumericReadoutHead, PixelInputEmbedding
from soft_prefix import load_prefix
from intra_patch_pos_embed import load_intra_pos
import utils


@torch.no_grad()
def compute_per_pixel_bpp_map(
    model,
    pixel_emb,
    readout_head,
    prefix_tau=None,
    intra_pos=None,
    image_path: str = None,
    device: str = "cuda",
    prompt_ids_list: list = None,   # 可自定义（默认为空）
):
    """
    输入一张图像，返回 (H, W) 的 per-pixel BPP map（单位：bits per pixel）
    每个像素的 BPP = 该像素 3 个 sub-pixel（R/G/B）消耗的 bits 之和
    """
    if image_path is None:
        raise ValueError("必须提供 --image 路径")

    # 1. 读取图像并展平为 pixel_values_list（和你的 PixelJsonlDataset 完全一致）
    img = Image.open(image_path).convert("RGB")
    W, H = img.size
    pixels = np.array(img)                    # (H, W, 3) uint8
    pixel_values_list = pixels.reshape(-1, 3).flatten().tolist()  # [R0,G0,B0, R1,G1,B1, ...]

    # 2. prompt（如果你训练时用了特定 template，这里可以改）
    if prompt_ids_list is None:
        prompt_ids_list = [model.config.pad_token_id] * 8   # 8 个 pad 占位（最常见做法）

    # 3. 构造 tensor（完全复用 evaluate.py + run_est_bpp_pixels_fast 的逻辑）
    prompt_t = torch.tensor(prompt_ids_list, dtype=torch.long, device=device).unsqueeze(0)  # [1, Lp]
    pix_t = torch.tensor(pixel_values_list, dtype=torch.long, device=device).unsqueeze(0)    # [1, T]
    Lp = prompt_t.size(1)
    T = pix_t.size(1)

    emb = model.get_input_embeddings()
    dtype = emb.weight.dtype

    prompt_emb = emb(prompt_t).to(dtype)
    pix_emb = pixel_emb(pix_t).to(dtype)

    # intra-patch pos embed（如果你用了）
    if intra_pos is not None:
        token_idx = torch.arange(T, device=device, dtype=torch.long)
        pos = intra_pos(token_idx).unsqueeze(0).to(dtype)
        pix_emb = pix_emb + pos

    # 拼接 prefix + prompt + pixels
    if prefix_tau is not None:
        tau_b = prefix_tau.to(device=device, dtype=dtype).unsqueeze(0)   # [1, P, H]
        seq = torch.cat([tau_b, prompt_emb, pix_emb], dim=1)
        prefix_len = tau_b.size(1)
    else:
        seq = torch.cat([prompt_emb, pix_emb], dim=1)
        prefix_len = 0

    attn = torch.ones((1, seq.size(1)), dtype=torch.long, device=device)

    # forward（和你的 fast estimate 一模一样）
    base = getattr(model, "model", None)
    if base is not None:
        hs = base(inputs_embeds=seq, attention_mask=attn, use_cache=False).last_hidden_state
    else:
        hs = model(inputs_embeds=seq, attention_mask=attn,
                   output_hidden_states=True, use_cache=False).hidden_states[-1]

    # 取出预测每个 sub-pixel 的 hidden 位置
    rows = torch.arange(prefix_len + Lp - 1, prefix_len + Lp - 1 + T, device=device)
    h_rows = hs[:, rows, :].float().squeeze(0)          # [T, H]

    logits = readout_head(h_rows)                       # [T, 256]
    logp = F.log_softmax(logits, dim=-1)

    # 计算每个 sub-pixel 的 bits（-log2 p）
    tgt = pix_t.squeeze(0)
    logp_true = logp.gather(1, tgt.unsqueeze(1)).squeeze(1)
    bits_per_subpixel = (-logp_true) / np.log(2.0)      # [T]

    # === 关键：转成 per-pixel BPP（3 个 sub-pixel 求和）===
    bits_per_pixel = bits_per_subpixel.reshape(-1, 3).sum(dim=1)   # [H*W]
    bpp_map = bits_per_pixel.reshape(H, W).cpu().numpy()

    return bpp_map, pixels, (H, W)


def plot_bpp_visualization(image_path: str, bpp_map: np.ndarray, save_path: str = None):
    """生成 Figure 7 同款对比图：原图 | 热力图（带 colorbar）"""
    orig = Image.open(image_path).convert("RGB")

    fig, axs = plt.subplots(1, 2, figsize=(14, 7), dpi=200)

    # 左：原图
    axs[0].imshow(orig)
    axs[0].set_title("Original Image", fontsize=14)
    axs[0].axis("off")

    # 右：BPP 热力图（颜色越暖 = 消耗越高）
    im = axs[1].imshow(bpp_map, cmap="inferno", interpolation="nearest")   # inferno/hot 效果最好
    axs[1].set_title("Bit Consumption Map\n(redder = higher BPP)", fontsize=14)
    axs[1].axis("off")

    # colorbar（精确对标 Figure 7）
    cbar = fig.colorbar(im, ax=axs[1], fraction=0.046, pad=0.04)
    cbar.set_label("Bits per Pixel", rotation=270, labelpad=20, fontsize=12)

    # 统计信息
    mean_bpp = bpp_map.mean()
    fig.suptitle(f"Average BPP = {mean_bpp:.4f} | Image: {Path(image_path).name}",
                 fontsize=16, y=0.95)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✅ 保存成功 → {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成 Figure 7 风格的局部 BPP 热力图")
    parser.add_argument("--image", type=str, required=True, help="输入图像路径（PNG/JPG）")
    # parser.add_argument("--model_id", type=str, default="/home/vipuser/Model/QWEN_3_0.6B")
    parser.add_argument("-m", "--model_id", type=str, default="/home/vipuser/Model/LLAMA_3.1_B")
    parser.add_argument("--pixel_emb", type=str, default="models/pixel_emb_k.pt", help="PixelInputEmbedding ckpt")
    parser.add_argument("--readout_head", type=str, default="models/HeadPixel_k.pt")
    parser.add_argument("--prefix", type=str, default="models/SP_k.pt", help="Soft Prefix（可选）")
    parser.add_argument("--intra_pos", type=str, default="models/INP_k.pt", help="Intra-patch pos（可选）")
    parser.add_argument("--output", type=str, default=None, help="输出图片路径（默认自动生成）")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # ====================== 加载模型 ======================
    tokenizer, model = utils.load_llm_model(args.model_id, "", args.device)
    model.eval()
    model = model.float()

    pixel_emb = PixelInputEmbedding.load(args.pixel_emb, device=args.device).eval()
    readout_head = NumericReadoutHead.load(args.readout_head, device=args.device).to(torch.float32).eval()

    # Soft Prefix
    prefix_tau = None
    if Path(args.prefix).exists():
        prefix_tau, _ = load_prefix(args.prefix, device=args.device, model=model)
        print(f"已加载 Soft Prefix: {args.prefix}")

    # Intra Pos
    intra_pos = None
    if Path(args.intra_pos).exists():
        intra_pos = load_intra_pos(args.intra_pos, model, args.device)
        print(f"已加载 IntraPatchPositionEmbedding: {args.intra_pos}")

    # ====================== 计算 + 可视化 ======================
    print("🚀 正在计算局部 BPP map...")
    bpp_map, _, shape = compute_per_pixel_bpp_map(
        model=model,
        pixel_emb=pixel_emb,
        readout_head=readout_head,
        prefix_tau=prefix_tau,
        intra_pos=intra_pos,
        image_path=args.image,
        device=args.device
    )

    print(f"图像尺寸: {shape[0]}x{shape[1]}")
    print(f"BPP 范围: {bpp_map.min():.3f} ~ {bpp_map.max():.3f}")
    print(f"全图平均 BPP: {bpp_map.mean():.4f}")

    save_path = args.output or str(Path(args.image).with_name(Path(args.image).stem + "_bpp_map.png"))
    plot_bpp_visualization(args.image, bpp_map, save_path=save_path)