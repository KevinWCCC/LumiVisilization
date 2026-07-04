import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import argparse
from pathlib import Path

def compute_complexity_map(image, window_size=11):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    complexity = cv2.GaussianBlur(magnitude, (window_size, window_size), 0)
    complexity = (complexity - complexity.min()) / (complexity.max() - complexity.min() + 1e-8)
    return complexity

def create_bpp_map(complexity_map, target_bpp):
    return target_bpp * (0.5 + 1.0 * complexity_map)   # 加大复杂度差异（0.5~1.5倍）

def bpp_to_rgba(bpp_array, cmap, vmin=5.0, vmax=20.0):
    """把 bpp 值映射到 RGBA（0~1），用于 matplotlib 显示"""
    normed = np.clip((bpp_array - vmin) / (vmax - vmin), 0, 1)
    rgba = cmap(normed)
    return (rgba[..., :3] * 255).astype(np.uint8)  # 返回 RGB uint8

def main():
    parser = argparse.ArgumentParser(description='Per-pixel BPP heatmap – fixed range & strong contrast')
    parser.add_argument('--image', required=True)
    parser.add_argument('--bpps', nargs='+', type=float, default=[10, 8, 7, 6])
    parser.add_argument('--window_size', type=int, default=11)
    parser.add_argument('--output', default='bpp_results')
    args = parser.parse_args()

    Path(args.output).mkdir(exist_ok=True)
    img = cv2.imread(args.image)
    if img is None:
        print("Error: Cannot read image")
        return

    h, w = img.shape[:2]
    print(f"Image: {w}×{h}   window={args.window_size}")

    complexity_map = compute_complexity_map(img, args.window_size)
    target_bpps = sorted(args.bpps, reverse=True)

    # ── 更陡的 colormap ──
    colors_list = [
        (0.0,   '#000000'),     # 最高 bpp
        (0.25,  '#3c0000'),
        (0.40,  '#800000'),
        (0.55,  '#c00000'),     # ← 10~15 区间快速变红
        (0.70,  '#ff1a1a'),
        (0.85,  '#ff6666'),
        (1.0,   '#ffb3b3')      # 最低 bpp 亮粉红
    ]
    cmap = LinearSegmentedColormap.from_list('steep_bpp', colors_list, N=512)

    fig = plt.figure(figsize=(22, 16))
    fig.suptitle('Per-pixel BPP Heatmap – Strong Contrast (5–20 bpp range)', fontsize=18, y=0.98)

    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.06], wspace=0.03, hspace=0.08)

    FIXED_VMIN, FIXED_VMAX = 5.0, 20.0

    for idx, bpp in enumerate(target_bpps):
        row, col = divmod(idx, 2)
        ax = fig.add_subplot(gs[row, col])

        bpp_map = create_bpp_map(complexity_map, bpp)

        # 用 matplotlib colormap 直接生成 RGB（不再用 JET）
        heat_rgb = bpp_to_rgba(bpp_map, cmap, FIXED_VMIN, FIXED_VMAX)

        # 叠加（保持 65% 原图 + 35% 热力）
        overlay = cv2.addWeighted(img, 0.65, heat_rgb, 0.35, 0)

        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        ax.imshow(overlay_rgb)
        ax.set_title(f'{bpp:.1f} bpp   mean={bpp_map.mean():.2f}', fontsize=14)
        ax.axis('off')

        cv2.imwrite(f"{args.output}/overlay_{bpp:.1f}bpp.jpg", overlay)
        print(f"Saved {bpp:.1f} bpp overlay (mean {bpp_map.mean():.2f})")

    # colorbar
    cax = fig.add_subplot(gs[:, 2])
    norm = plt.Normalize(FIXED_VMIN, FIXED_VMAX)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, cax=cax, orientation='vertical')
    cbar.set_label('Bits Per Pixel', fontsize=14)
    ticks = np.arange(5, 21, 2)
    cbar.set_ticks(ticks)
    cbar.ax.set_yticklabels([f'{t}' for t in ticks[::-1]])  # 低在上 高在下

    plt.tight_layout(rect=[0, 0, 0.94, 0.96])
    combined = f"{args.output}/COMBINED_strong_contrast.png"
    plt.savefig(combined, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nDone. Main figure: {combined}")
    print("Now four images should show clear difference (no blue, stronger contrast)")

if __name__ == "__main__":
    main()