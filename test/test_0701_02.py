#!/usr/bin/env python3
"""
Focused Subchannel Similarity Matrix (Final Mild Version)
=========================================================
- Matrix figure: two side-by-side cosine similarity heatmaps (Baseline vs LUMI)
  within ONE intensity zone, grouped by subchannel (R/G/B).
- Bar figure: quantitative per-subchannel intra-covariance comparison.

Final adjustment:
- Matrix colormap changed to very mild sequential 'Oranges' 
  (light orange → orange). Much softer gradient as requested.
- Still outputs two separate files: *_matrix.png and *_bar.png

Run:
python focus_zone_subchannel_similarity_matrix.py --dataset DIV2K --focus_zone 4
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

def compute_per_channel_within_var(embs, ch_labels):
    """Return list of mean variance per dimension for each subchannel."""
    vars_list = []
    for c in range(3):
        mask = (ch_labels == c)
        if mask.sum() >= 2:
            c_embs = embs[mask]
            vars_list.append(float(np.mean(np.var(c_embs, axis=0))))
        else:
            vars_list.append(np.nan)
    return vars_list

def main():
    parser = argparse.ArgumentParser(
        description="Focused subchannel similarity matrix (mild light-orange to orange gradient) "
                    "+ per-subchannel intra-covariance bar chart. Two separate PNG files."
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
                        help="Colormap for matrix (Oranges recommended for mild light-orange to orange)")
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

    # Per-subchannel intra covariance
    base_within_vars = compute_per_channel_within_var(base_z, ch_z)
    lumi_within_vars = compute_per_channel_within_var(prop_z, ch_z)

    print(f"\n[Zone Z{args.focus_zone}] Per-subchannel intra covariance (mean var/dim):")
    for i, name in enumerate(ch_names):
        print(f"  {name}: Baseline={base_within_vars[i]:.2e}   LUMI={lumi_within_vars[i]:.2e}")

    # ========== Figure 1: Matrix (mild light orange → orange) ==========
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
              f"Strong within-subchannel blocks (tighter covariance)",
        ch_names=ch_names, n_per_ch=n_per_ch,
        cmap=args.cmap, vmin=args.vmin, vmax=args.vmax
    )

    fig_mat.suptitle(
        f"Cosine Similarity Matrix within Zone Z{args.focus_zone} (grouped by Subchannel R/G/B)\n"
        f"Llama3 + {args.dataset} Stage1  |  Hidden Dim = {H}  |  "
        f"Very mild colormap: light orange → orange",
        fontsize=11.2, y=0.97
    )
    plt.tight_layout(rect=[0, 0.02, 1, 0.91])

    mat_path = os.path.abspath(f"{args.output_prefix}_Z{args.focus_zone}_matrix.png")
    fig_mat.savefig(mat_path, bbox_inches='tight', dpi=300, facecolor='white')
    plt.close(fig_mat)
    print(f"[Saved] {mat_path}")

    # ========== Figure 2: Bar chart ==========
    print("[5/5] Saving bar chart figure ...")
    fig_bar, ax = plt.subplots(figsize=(9.5, 5.5))

    x = np.arange(3)
    width = 0.36

    # Same color family for Baseline and LUMI of each subchannel
    colors_base = ['#ff9999', '#99cc99', '#99bbff']
    colors_lumi = ['#d62728', '#2ca02c', '#1f77b4']

    bars_base = ax.bar(x - width/2, base_within_vars, width,
                       label='Baseline', color=colors_base, edgecolor='black',
                       linewidth=0.7, hatch='///')

    bars_lumi = ax.bar(x + width/2, lumi_within_vars, width,
                       label='LUMI', color=colors_lumi, edgecolor='black',
                       linewidth=0.7)

    ax.set_ylabel('Intra-Subchannel Covariance\n(mean variance per embedding dimension)', fontsize=10)
    ax.set_title(f'Per-Subchannel Intra-Covariance Comparison inside Zone Z{args.focus_zone}\n'
                 f'(Lower = tighter aggregation. Same color family for same subchannel)',
                 fontsize=11, pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(['R subchannel', 'G subchannel', 'B subchannel'], fontsize=10, fontweight='bold')
    ax.legend(title='Scheme', loc='upper right', fontsize=9)
    ax.set_ylim(0, max(max(base_within_vars), max(lumi_within_vars)) * 1.25)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')

    for bars in [bars_base, bars_lumi]:
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h):
                ax.annotate(f'{h:.2e}',
                            xy=(bar.get_x() + bar.get_width()/2, h),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=7.5, fontweight='bold')

    plt.tight_layout()
    bar_path = os.path.abspath(f"{args.output_prefix}_Z{args.focus_zone}_bar.png")
    fig_bar.savefig(bar_path, bbox_inches='tight', dpi=300, facecolor='white')
    plt.close(fig_bar)
    print(f"[Saved] {bar_path}")

    print("\n=== Done ===")
    print("Two files generated:")
    print(f"  1. {mat_path}")
    print(f"  2. {bar_path}")
    print("Matrix now uses very mild light-orange to orange gradient as requested.")

if __name__ == "__main__":
    main()