#!/usr/bin/env python3
"""
Focused Subchannel Similarity Matrix (CVPR / IEEE Style - Final Cool Tone Version)
==================================================================================
最终版本：
- 配色：冷色调（Baseline 浅冷蓝灰 + LUMI 柔和淡薄荷绿 #B8E0D2）
- 柱子宽度：0.65（更紧凑）
- 风格：匹配论文参考图片（浅色填充 + 蓝色边框 + 虚线分隔 + 顶部标注）
- 无内部 title，左侧标注最小化
"""

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from intra_patch_pos_embed import load_intra_pos
    from readout_head import PixelInputEmbedding
    import utils
except ImportError as e:
    print(f"[Error] Cannot import project modules: {e}")
    raise


def set_publication_style():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
        'font.size': 9,
        'axes.labelsize': 9,
        'axes.titlesize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'figure.titlesize': 11,
        'axes.linewidth': 0.8,
        'grid.linewidth': 0.5,
        'patch.linewidth': 0.8,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.facecolor': 'white',
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


sns.set_style("white")
set_publication_style()


def compute_cosine_similarity(embs):
    embs = embs.astype(np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8
    return (embs / norms) @ (embs / norms).T


def add_block_lines(ax, n_per_group, color='#2F2F2F', lw=0.9):
    pos = 0
    for i, n in enumerate(n_per_group):
        pos += n
        if i < len(n_per_group) - 1:
            ax.axhline(pos - 0.5, color=color, linewidth=lw, alpha=0.85, zorder=12)
            ax.axvline(pos - 0.5, color=color, linewidth=lw, alpha=0.85, zorder=12)


def plot_subchannel_matrix(sim, ax, title, ch_names, n_per_ch, cmap='Oranges', vmin=0.0, vmax=1.0):
    im = ax.imshow(sim, cmap=cmap, vmin=vmin, vmax=vmax,
                   aspect='auto', interpolation='nearest', origin='upper')
    add_block_lines(ax, n_per_ch)
    centers = np.cumsum([0] + n_per_ch[:-1]) + np.array(n_per_ch) / 2.0
    ax.set_xticks(centers)
    ax.set_xticklabels(ch_names, fontsize=8.5, fontweight='bold', rotation=25, ha='right')
    ax.set_yticks(centers)
    ax.set_yticklabels(ch_names, fontsize=8.5, fontweight='bold')
    ax.set_xlabel("Subchannel (grouped)", fontsize=9)
    ax.set_ylabel("Subchannel (grouped)", fontsize=9)
    ax.set_title(title, fontsize=9.5, pad=5, fontweight='medium')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.025, shrink=0.78)
    cbar.set_label('Cosine Similarity', fontsize=7.5)
    cbar.ax.tick_params(labelsize=7)
    ax.grid(False)
    return im


def compute_channel_pairwise_sims(sim, ch_labels):
    means = np.zeros((3, 3), dtype=np.float32)
    for i in range(3):
        for j in range(3):
            mask_i = (ch_labels == i)
            mask_j = (ch_labels == j)
            if mask_i.sum() > 0 and mask_j.sum() > 0:
                block = sim[np.ix_(mask_i, mask_j)]
                means[i, j] = float(np.mean(block))
            else:
                means[i, j] = np.nan
    return means


def main():
    parser = argparse.ArgumentParser(
        description="Final cool-tone version matching reference paper figure style."
    )
    parser.add_argument("--model_id", type=str, default="/home/vipuser/Model/LLAMA_3.1_B")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--ckpt_dir", type=str, default="models")
    parser.add_argument("--use_best", action="store_true", default=True)
    parser.add_argument("--include_intra_pos", action="store_true", default=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--output_prefix", type=str, default="focus_zone_subchannel")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--yellow_jitter_range", type=int, default=5)
    parser.add_argument("--focus_zone", type=int, default=4)
    parser.add_argument("--cmap", type=str, default="Oranges")
    parser.add_argument("--vmin", type=float, default=0.0)
    parser.add_argument("--vmax", type=float, default=1.0)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"[1/5] Loading Llama3 model: {args.model_id}")
    llm_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    _, model = utils.load_llm_model(args.model_id, "", args.device, dtype=llm_dtype)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    emb = model.get_input_embeddings()
    H = model.config.hidden_size

    suffix = "_stage1_best" if args.use_best else "_stage1"
    pe_path = os.path.join(args.ckpt_dir, f"pixel_emb_{args.dataset}{suffix}.pt")
    pixel_emb = PixelInputEmbedding.load(pe_path, device=args.device)
    print(f"[2/5] ✓ Pixel Embedding loaded")

    intra_pos = None
    if args.include_intra_pos:
        inp_path = os.path.join(args.ckpt_dir, f"INP_{args.dataset}{suffix}.pt")
        if os.path.exists(inp_path):
            intra_pos = load_intra_pos(inp_path, model, args.device)

    # Data generation
    print(f"[3/5] Generating data, focusing on Zone Z{args.focus_zone} ...")
    N_ZONES = 7
    N_LOGICAL = 48
    N_SUB = N_LOGICAL * 3

    counts = np.full(N_ZONES, N_LOGICAL // N_ZONES)
    counts[:N_LOGICAL % N_ZONES] += 1
    logical_zones = np.concatenate([np.full(c, z) for z, c in enumerate(counts)])

    base_vals_all = []
    for z in range(N_ZONES):
        low, high = z * 36, (z + 1) * 36
        n = counts[z]
        vals = np.random.randint(low, high, n)
        if z == 2 and args.yellow_jitter_range > 0:
            jitter = np.random.randint(-args.yellow_jitter_range, args.yellow_jitter_range + 1, n)
            vals = np.clip(vals + jitter, 0, 255)
        base_vals_all.append(vals)
    base_vals_all = np.concatenate(base_vals_all)

    pixel_values_list = []
    for base in base_vals_all:
        rgb = np.clip(base + np.random.randint(-3, 4, 3), 0, 255)
        pixel_values_list.extend(rgb)

    pixel_values_full = np.array(pixel_values_list, dtype=np.int64)
    pix_t = torch.tensor(pixel_values_full, dtype=torch.long, device=args.device).unsqueeze(0)
    channels = np.arange(N_SUB) % 3
    zones = np.repeat(logical_zones, 3)

    with torch.no_grad():
        base_embs = emb(pix_t.squeeze(0)).float().cpu().numpy()
        prop_embs = pixel_emb(pix_t).squeeze(0)
        if intra_pos is not None:
            prop_embs = prop_embs + intra_pos(torch.arange(N_SUB, device=args.device))
        prop_embs = prop_embs.float().cpu().numpy()

    z_mask = (zones == args.focus_zone)
    base_z = base_embs[z_mask]
    prop_z = prop_embs[z_mask]
    ch_z = channels[z_mask]

    order = np.argsort(ch_z)
    base_z_ord = base_z[order]
    prop_z_ord = prop_z[order]
    ch_z_ord = ch_z[order]

    n_per_ch = [(ch_z_ord == c).sum() for c in range(3)]
    ch_names = ['R', 'G', 'B']

    sim_base = compute_cosine_similarity(base_z_ord)
    sim_prop = compute_cosine_similarity(prop_z_ord)

    base_pairwise = compute_channel_pairwise_sims(sim_base, ch_z_ord)
    lumi_pairwise = compute_channel_pairwise_sims(sim_prop, ch_z_ord)

    print(f"\n[Zone Z{args.focus_zone}] Channel Pairwise Cosine Similarity (3×3):")
    print("Baseline (Tokenizer):")
    print(np.array2string(base_pairwise, precision=4, suppress_small=True, floatmode='fixed'))
    print("LUMI (Ours):")
    print(np.array2string(lumi_pairwise, precision=4, suppress_small=True, floatmode='fixed'))
    print("Delta (LUMI − Baseline):")
    print(np.array2string(lumi_pairwise - base_pairwise, precision=4, suppress_small=True, floatmode='fixed'))

    # ========== Figure 1: Matrix ==========
    print("[4/5] Saving matrix figure ...")
    fig_mat, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 5.2))

    plot_subchannel_matrix(
        sim_base, ax1,
        title=f"(a) Baseline — Zone Z{args.focus_zone}\nWeak within-subchannel blocks",
        ch_names=ch_names, n_per_ch=n_per_ch,
        cmap=args.cmap, vmin=args.vmin, vmax=args.vmax
    )

    plot_subchannel_matrix(
        sim_prop, ax2,
        title=f"(b) LUMI (Ours) — Zone Z{args.focus_zone}\nStrong within-subchannel blocks (tighter alignment)",
        ch_names=ch_names, n_per_ch=n_per_ch,
        cmap=args.cmap, vmin=args.vmin, vmax=args.vmax
    )

    fig_mat.suptitle(
        f"Cosine Similarity Matrix within Zone Z{args.focus_zone} (grouped by Subchannel R/G/B)   |   "
        f"Llama3 + {args.dataset} Stage1   |   Hidden Dim = {H}",
        fontsize=10.5, y=0.98, fontweight='medium'
    )
    plt.tight_layout(rect=[0.01, 0.02, 0.99, 0.935])

    mat_path = os.path.abspath(f"{args.output_prefix}_Z{args.focus_zone}_matrix.png")
    fig_mat.savefig(mat_path, bbox_inches='tight', dpi=300, facecolor='white')
    plt.close(fig_mat)
    print(f"[Saved] {mat_path}")

    # ========== Figure 2: Bar chart (冷色调 + 更紧凑) ==========
    print("[5/5] Saving bar chart (cool tone, compact bars)...")
    fig_bar, ax = plt.subplots(figsize=(10.2, 4.4))

    inter_mask = ~np.eye(3, dtype=bool)
    base_intra = np.diag(base_pairwise)
    base_inter_mean = float(np.mean(base_pairwise[inter_mask]))
    base_vals = np.append(base_intra, base_inter_mean)

    lumi_intra = np.diag(lumi_pairwise)
    lumi_inter_mean = float(np.mean(lumi_pairwise[inter_mask]))
    lumi_vals = np.append(lumi_intra, lumi_inter_mean)

    intra_labels = ['R intra', 'G intra', 'B intra', 'Mean Inter']
    x_base = np.arange(4)
    x_lumi = np.arange(4) + 4.0          # 控制两组间距
    width = 0.65                         # 柱子宽度（在这里修改）

    # === 冷色调配色 ===
    color_base_fill = '#E8EEF2'          # 浅冷蓝灰
    color_base_edge = '#2E75B6'
    color_lumi_fill = '#B8E0D2'          # 柔和淡薄荷绿（冷色、不鲜艳）
    color_lumi_edge = '#2E75B6'

    bars_base = ax.bar(x_base, base_vals, width,
                       label='Baseline', color=color_base_fill, edgecolor=color_base_edge,
                       linewidth=1.5, alpha=0.95, zorder=3)

    bars_lumi = ax.bar(x_lumi, lumi_vals, width,
                       label='LUMI (Ours)', color=color_lumi_fill, edgecolor=color_lumi_edge,
                       linewidth=1.5, alpha=0.95, zorder=3)

    # 虚线分隔
    ax.axvline(x=3.55, color='#333333', linestyle='--', linewidth=1.2, alpha=0.85, zorder=2)

    # 顶部标注
    # ax.text(0.02, 1.13, 'Mean Cosine Similarity ↑', fontsize=9, fontweight='bold',
    #         color='#37474f', transform=ax.transAxes, va='bottom')

    ax.plot([0, 1], [1.09, 1.09], color='#546e7a', linewidth=1.0,
            linestyle='--', alpha=0.75, zorder=2, transform=ax.transAxes)

    ax.text(0.22, 1.135, 'Baseline (Classical) ←', fontsize=8, fontweight='bold',
            color='#455a64', transform=ax.transAxes, ha='center', va='bottom')
    ax.text(0.78, 1.135, 'LUMI (Proposed) →', fontsize=8, fontweight='bold',
            color='#0288d1', transform=ax.transAxes, ha='center', va='bottom')

    ax.set_ylabel('Mean Cosine Similarity ↑', fontsize=8, labelpad=2)

    ax.set_xticks(list(x_base) + list(x_lumi))
    ax.set_xticklabels(intra_labels + intra_labels, fontsize=8, fontweight='medium', rotation=28, ha='right')

    ax.set_ylim(0, 1.22)
    ax.set_xlim(-0.7, 9.0)
    ax.grid(axis='y', alpha=0.18, linestyle='--', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for bars, txt_color in [(bars_base, '#1F4E79'), (bars_lumi, '#1B5E20')]:
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h):
                ax.annotate(f'{h:.3f}',
                            xy=(bar.get_x() + bar.get_width()/2, h),
                            xytext=(0, 2), textcoords="offset points",
                            ha='center', va='bottom', fontsize=6.5, fontweight='bold',
                            color=txt_color, zorder=5)

    plt.tight_layout(pad=0.5, rect=[0.01, 0.03, 0.99, 0.93])
    bar_path = os.path.abspath(f"{args.output_prefix}_Z{args.focus_zone}_bar.png")
    fig_bar.savefig(bar_path, bbox_inches='tight', dpi=300, facecolor='white')
    plt.close(fig_bar)
    print(f"[Saved] {bar_path}")

    print("\n=== Done (Final Cool Tone + Compact Version) ===")
    print("Two files generated:")
    print(f"  1. {mat_path}")
    print(f"  2. {bar_path}")


if __name__ == "__main__":
    main()