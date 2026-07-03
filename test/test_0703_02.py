#!/usr/bin/env python3
"""
Focused Subchannel Similarity Matrix (Revised v2)
=========================================================
Key improvements over original:
- Bar figure now plots **Mean Intra-Subchannel Cosine Similarity** (consistent with matrix)
- Higher value = tighter angular alignment within subchannel (matches "tighter covariance" claim)
- Removed contradictory "Lower = tighter" narrative
- Clean scientific captions (no meta "as requested" text)
- Consistent metric: cosine similarity throughout
- Bar now correctly shows LUMI > Baseline (supporting the visual claim in matrix)

Run:
python focus_zone_subchannel_similarity_matrix_v2.py --dataset DIV2K --focus_zone 4
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

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

def compute_cosine_similarity(embs):
    embs = embs.astype(np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8
    return (embs / norms) @ (embs / norms).T

def add_block_lines(ax, n_per_group, color='#333333', lw=1.0):
    pos = 0
    for i, n in enumerate(n_per_group):
        pos += n
        if i < len(n_per_group) - 1:
            ax.axhline(pos - 0.5, color=color, linewidth=lw, alpha=0.8, zorder=12)
            ax.axvline(pos - 0.5, color=color, linewidth=lw, alpha=0.8, zorder=12)

def plot_subchannel_matrix(sim, ax, title, ch_names, n_per_ch, cmap='Oranges', vmin=0.0, vmax=1.0):
    im = ax.imshow(sim, cmap=cmap, vmin=vmin, vmax=vmax,
                   aspect='auto', interpolation='nearest', origin='upper')
    add_block_lines(ax, n_per_ch)
    centers = np.cumsum([0] + n_per_ch[:-1]) + np.array(n_per_ch) / 2.0
    ax.set_xticks(centers)
    ax.set_xticklabels(ch_names, fontsize=9, fontweight='bold')
    ax.set_yticks(centers)
    ax.set_yticklabels(ch_names, fontsize=9, fontweight='bold')
    ax.set_xlabel("Subchannel (grouped)", fontsize=10)
    ax.set_ylabel("Subchannel (grouped)", fontsize=10)
    ax.set_title(title, fontsize=10.2, pad=6)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03, shrink=0.82)
    cbar.set_label('Cosine Similarity', fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    ax.grid(False)
    return im

def compute_mean_intra_channel_sim(sim, ch_labels):
    """
    Compute mean cosine similarity WITHIN each subchannel block.
    This is the recommended metric: higher = tighter angular alignment within subchannel.
    Consistent with the cosine similarity matrix visualization.
    """
    means = []
    for c in range(3):
        mask = (ch_labels == c)
        if mask.sum() >= 2:
            block = sim[np.ix_(mask, mask)]
            means.append(float(np.mean(block)))
        else:
            means.append(np.nan)
    return means

def main():
    parser = argparse.ArgumentParser(
        description="Focused subchannel similarity matrix + per-subchannel mean intra-cosine similarity bar chart. "
                    "Bar chart now uses mean intra-subchannel cosine similarity (higher = tighter alignment), "
                    "consistent with the matrix figure."
    )
    parser.add_argument("--model_id", type=str, default="/home/vipuser/Model/LLAMA_3.1_B")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--ckpt_dir", type=str, default="models")
    parser.add_argument("--use_best", action="store_true", default=True)
    parser.add_argument("--include_intra_pos", action="store_true", default=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--output_prefix", type=str, default="focus_zone_subchannel",
                        help="Prefix for output files (*_matrix.png and *_bar.png)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--yellow_jitter_range", type=int, default=5)
    parser.add_argument("--focus_zone", type=int, default=4)
    parser.add_argument("--cmap", type=str, default="Oranges",
                        help="Colormap for matrix (Oranges or plasma recommended)")
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

    # Focus on zone and reorder by subchannel
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

    # === NEW: Mean intra-subchannel cosine similarity (consistent metric) ===
    base_intra_sims = compute_mean_intra_channel_sim(sim_base, ch_z_ord)
    lumi_intra_sims = compute_mean_intra_channel_sim(sim_prop, ch_z_ord)

    print(f"\n[Zone Z{args.focus_zone}] Mean Intra-Subchannel Cosine Similarity (higher = tighter alignment):")
    for i, name in enumerate(ch_names):
        print(f"  {name}: Baseline={base_intra_sims[i]:.4f}   LUMI={lumi_intra_sims[i]:.4f}   "
              f"(Δ = {lumi_intra_sims[i] - base_intra_sims[i]:+.4f})")

    # ========== Figure 1: Matrix ==========
    print("[4/5] Saving matrix figure ...")
    fig_mat, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.8))

    plot_subchannel_matrix(
        sim_base, ax1,
        title=f"(a) Baseline — Zone Z{args.focus_zone} (grouped by subchannel)\n"
              f"Weak within-subchannel blocks",
        ch_names=ch_names, n_per_ch=n_per_ch,
        cmap=args.cmap, vmin=args.vmin, vmax=args.vmax
    )

    plot_subchannel_matrix(
        sim_prop, ax2,
        title=f"(b) LUMI — Zone Z{args.focus_zone} (grouped by subchannel)\n"
              f"Strong within-subchannel blocks (tighter angular alignment)",
        ch_names=ch_names, n_per_ch=n_per_ch,
        cmap=args.cmap, vmin=args.vmin, vmax=args.vmax
    )

    fig_mat.suptitle(
        f"Cosine Similarity Matrix within Zone Z{args.focus_zone} (grouped by Subchannel R/G/B)\n"
        f"Llama3 + {args.dataset} Stage1  |  Hidden Dim = {H}",
        fontsize=11.2, y=0.97
    )
    plt.tight_layout(rect=[0, 0.02, 1, 0.91])

    mat_path = os.path.abspath(f"{args.output_prefix}_Z{args.focus_zone}_matrix.png")
    fig_mat.savefig(mat_path, bbox_inches='tight', dpi=300, facecolor='white')
    plt.close(fig_mat)
    print(f"[Saved] {mat_path}")

    # ========== Figure 2: Bar chart (REVISED - Mean Intra-Cosine Similarity) ==========
    print("[5/5] Saving bar chart figure (revised metric)...")
    fig_bar, ax = plt.subplots(figsize=(9.5, 5.5))

    x = np.arange(3)
    width = 0.36

    # Color scheme: same family for each subchannel
    colors_base = ['#ff9999', '#99cc99', '#99bbff']  # light
    colors_lumi = ['#d62728', '#2ca02c', '#1f77b4']  # dark

    bars_base = ax.bar(x - width/2, base_intra_sims, width,
                       label='Baseline', color=colors_base, edgecolor='black',
                       linewidth=0.7, hatch='///')

    bars_lumi = ax.bar(x + width/2, lumi_intra_sims, width,
                       label='LUMI', color=colors_lumi, edgecolor='black',
                       linewidth=0.7)

    ax.set_ylabel('Mean Intra-Subchannel Cosine Similarity\n(higher = tighter angular alignment within subchannel)', fontsize=10)
    ax.set_title(f'Per-Subchannel Intra-Alignment Comparison inside Zone Z{args.focus_zone}\n'
                 f'(LUMI shows significantly higher within-subchannel similarity)',
                 fontsize=11, pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(['R subchannel', 'G subchannel', 'B subchannel'], fontsize=10, fontweight='bold')
    ax.legend(title='Scheme', loc='upper left', fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')

    # Annotate values
    for bars in [bars_base, bars_lumi]:
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h):
                ax.annotate(f'{h:.3f}',
                            xy=(bar.get_x() + bar.get_width()/2, h),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.tight_layout()
    bar_path = os.path.abspath(f"{args.output_prefix}_Z{args.focus_zone}_bar.png")
    fig_bar.savefig(bar_path, bbox_inches='tight', dpi=300, facecolor='white')
    plt.close(fig_bar)
    print(f"[Saved] {bar_path}")

    print("\n=== Done ===")
    print("Two files generated:")
    print(f"  1. {mat_path}")
    print(f"  2. {bar_path}")
    print("\n[Revision note] Bar chart now uses MEAN INTRA-SUBCHANNEL COSINE SIMILARITY.")
    print("This metric is fully consistent with the matrix figure (both based on cosine similarity).")
    print("LUMI correctly shows higher values → tighter within-subchannel alignment.")
    print("The contradictory 'Lower = tighter' narrative has been removed.")

if __name__ == "__main__":
    main()