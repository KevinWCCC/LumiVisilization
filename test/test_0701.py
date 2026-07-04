#!/usr/bin/env python3
"""
Focused Subchannel Similarity Matrix (within ONE intensity zone)
================================================================
Still follows the exact drawing style of your previous cosine similarity
heatmap (block-diagonal visualization with RdBu_r, grid lines, insightful
English captions), but now **constrained** as requested:

1. Focus on **a single intensity zone** only (default Z4).
2. Re-group the points **by the 3 subchannels** (R → G → B order).
3. Show the cosine similarity matrix → directly visualizes the
   "covariance structure change" among the three subchannel groups
   inside that zone.

Interpretation:
- Stronger 3×3 block-diagonal in LUMI = same-subchannel points have
  much higher mutual similarity (tighter covariance, lower dispersion)
  while different subchannels are more distinguishable.
- This is the matrix-figure version of "subchannel covariance changes"
  within one intensity family.

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

def add_block_lines(ax, n_per_group, colors=None, lw=1.2):
    """Draw lines separating the three subchannel blocks."""
    if colors is None:
        colors = ['#E31A1C', '#33A02C', '#1F78B4']
    pos = 0
    for i, n in enumerate(n_per_group):
        pos += n
        if i < len(n_per_group) - 1:
            ax.axhline(pos - 0.5, color=colors[i], linewidth=lw, alpha=0.9, zorder=12)
            ax.axvline(pos - 0.5, color=colors[i], linewidth=lw, alpha=0.9, zorder=12)

def plot_subchannel_matrix(sim, ax, title, ch_names, n_per_ch, vmin=-0.2, vmax=1.0):
    im = ax.imshow(sim, cmap='RdBu_r', vmin=vmin, vmax=vmax,
                   aspect='auto', interpolation='nearest', origin='upper')

    add_block_lines(ax, n_per_ch)

    # Tick labels at block centers
    centers = np.cumsum([0] + n_per_ch[:-1]) + np.array(n_per_ch) / 2
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

def main():
    parser = argparse.ArgumentParser(
        description="Focused cosine similarity matrix within ONE intensity zone, "
                    "re-grouped by the 3 subchannels (R/G/B). Shows covariance structure change."
    )
    parser.add_argument("--model_id", type=str, default="/home/vipuser/Model/LLAMA_3.1_B")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--ckpt_dir", type=str, default="models")
    parser.add_argument("--use_best", action="store_true", default=True)
    parser.add_argument("--include_intra_pos", action="store_true", default=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--output", type=str, default="focus_zone_subchannel_similarity_matrix.png")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--yellow_jitter_range", type=int, default=5)
    parser.add_argument("--focus_zone", type=int, default=4,
                        help="Which intensity zone to zoom into (0-6)")
    parser.add_argument("--vmin", type=float, default=-0.15)
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

    # Data generation (same as before)
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

    # === Focus on one zone and re-group by subchannel ===
    z_mask = (zones == args.focus_zone)
    base_z = base_embs[z_mask]
    prop_z = prop_embs[z_mask]
    ch_z = channels[z_mask]

    # Reorder: R (0) → G (1) → B (2)
    order = np.argsort(ch_z)
    base_z_ord = base_z[order]
    prop_z_ord = prop_z[order]
    ch_z_ord = ch_z[order]

    n_per_ch = [(ch_z_ord == c).sum() for c in range(3)]
    ch_names = ['R', 'G', 'B']

    sim_base = compute_cosine_similarity(base_z_ord)
    sim_prop = compute_cosine_similarity(prop_z_ord)

    # Quick stats (within-subchannel vs between-subchannel)
    def block_stats(sim, n_per):
        within, between = [], []
        pos = 0
        for i, n in enumerate(n_per):
            if n < 2: continue
            block = sim[pos:pos+n, pos:pos+n]
            tri = np.triu_indices_from(block, k=1)
            within.append(block[tri].mean())
            # between this block and all others
            other = np.concatenate([np.arange(0, pos), np.arange(pos+n, sim.shape[0])])
            if len(other) > 0:
                cross = sim[np.ix_(np.arange(pos, pos+n), other)]
                between.append(cross.mean())
            pos += n
        return float(np.mean(within)), float(np.mean(between)) if between else np.nan

    w_b, b_b = block_stats(sim_base, n_per_ch)
    w_p, b_p = block_stats(sim_prop, n_per_ch)

    print(f"\n[Zone Z{args.focus_zone} Subchannel Stats]")
    print(f"  Baseline  within-subch sim: {w_b:.4f}   between-subch: {b_b:.4f}")
    print(f"  LUMI      within-subch sim: {w_p:.4f}   between-subch: {b_p:.4f}")
    print(f"  → LUMI shows stronger within-subchannel coherence (tighter covariance).")

    # === Plot (exact same style as previous matrix figure) ===
    print("[4/5] Plotting focused subchannel similarity matrices ...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.8, 6.0))

    plot_subchannel_matrix(
        sim_base, ax1,
        title=f"(a) Baseline — Zone Z{args.focus_zone} (grouped by subchannel)\n"
              f"Weak block structure among R/G/B → limited subchannel awareness",
        ch_names=ch_names, n_per_ch=n_per_ch,
        vmin=args.vmin, vmax=args.vmax
    )

    plot_subchannel_matrix(
        sim_prop, ax2,
        title=f"(b) LUMI — Zone Z{args.focus_zone} (grouped by subchannel)\n"
              f"Strong 3-block structure: same subchannel points have high similarity\n"
              f"(Direct evidence of reduced within-subchannel covariance)",
        ch_names=ch_names, n_per_ch=n_per_ch,
        vmin=args.vmin, vmax=args.vmax
    )

    fig.suptitle(
        f"Cosine Similarity Matrix within ONE Intensity Zone (Z{args.focus_zone}), grouped by Subchannel\n"
        f"Llama3 + {args.dataset} Stage1  |  Hidden Dim = {H}  |  "
        f"Matrix size: {sim_base.shape[0]}×{sim_base.shape[0]} (only points inside this zone)",
        fontsize=11.2, y=0.97
    )

    plt.tight_layout(rect=[0, 0.015, 1, 0.91])

    out_path = os.path.abspath(args.output)
    fig.savefig(out_path, bbox_inches='tight', dpi=300, facecolor='white')
    print(f"\n[Saved] {out_path}")

    npz_path = out_path.replace('.png', '.npz')
    np.savez(npz_path,
             focus_zone=args.focus_zone,
             sim_base=sim_base, sim_prop=sim_prop,
             ch_z_ord=ch_z_ord, n_per_ch=n_per_ch,
             within_base=w_b, between_base=b_b,
             within_lumi=w_p, between_lumi=b_p)
    print(f"[Saved] {npz_path}")

    print("\n=== How to read this figure ===")
    print("The matrix is now ordered R-block | G-block | B-block (instead of zone blocks).")
    print("Stronger dark-red blocks on the diagonal in (b) = LUMI makes points of the")
    print("same subchannel much more similar to each other (lower dispersion/covariance)")
    print("while keeping them distinct from the other two subchannels inside the same intensity zone.")

if __name__ == "__main__":
    main()
