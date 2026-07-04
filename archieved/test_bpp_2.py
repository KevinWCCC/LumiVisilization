import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import argparse
from pathlib import Path

def compute_complexity_map(image, window_size=15):
    """Improved complexity: local standard deviation (dominant) + gradient
    → Doors/windows/uniform textures now get LOWER BPP"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    
    # Sobel gradient (edges)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient = np.sqrt(sobelx**2 + sobely**2)
    
    # Local variance / standard deviation (main metric)
    mean = cv2.GaussianBlur(gray, (window_size, window_size), 0)
    mean_sq = cv2.GaussianBlur(gray**2, (window_size, window_size), 0)
    local_var = mean_sq - mean**2
    local_std = np.sqrt(np.maximum(local_var, 0))
    
    # Normalize
    gradient_norm = (gradient - gradient.min()) / (gradient.max() - gradient.min() + 1e-8)
    std_norm = (local_std - local_std.min()) / (local_std.max() - local_std.min() + 1e-8)
    
    # Final complexity: local_std dominates (80%)
    complexity = 0.8 * std_norm + 0.2 * gradient_norm
    return complexity

def create_raw_bpp_map(complexity_map, center_bpp):
    """Generate raw BPP map with EVEN STRONGER regional contrast"""
    return center_bpp * (0.30 + 1.40 * complexity_map)   # 加大差异

def normalize_to_display_range(bpp_map, display_center, display_half_range):
    """Map every map to the SAME display range: center ± half_range"""
    display_min = display_center - display_half_range
    display_max = display_center + display_half_range
    bmin, bmax = bpp_map.min(), bpp_map.max()
    if bmax - bmin < 1e-6:
        return np.full_like(bpp_map, display_center)
    norm = (bpp_map - bmin) / (bmax - bmin)
    return display_min + norm * (display_max - display_min)

def bpp_to_rgb(bpp_array, cmap, vmin, vmax):
    normed = np.clip((bpp_array - vmin) / (vmax - vmin), 0, 1)
    rgba = cmap(normed)
    return (rgba[..., :3] * 255).astype(np.uint8)

def main():
    parser = argparse.ArgumentParser(
        description='Per-pixel BPP Heatmap – Center + Variance mode + STRONGER heatmap overlay'
    )
    parser.add_argument('--image', required=True, help='Input image path')
    parser.add_argument('--bpps', nargs=2, type=float, required=True,
                        help='Center and variance. Example: 14 1.5')
    parser.add_argument('--window_size', type=int, default=15,
                        help='Local window for variance (11~21 recommended)')
    parser.add_argument('--output', default='bpp_results', help='Output folder')
    args = parser.parse_args()

    center, variance = args.bpps
    display_min = center - variance
    display_max = center + variance

    Path(args.output).mkdir(exist_ok=True)
    
    img = cv2.imread(args.image)
    if img is None:
        print("Error: Cannot read the image. Please check the path.")
        return

    print(f"Image size: {img.shape[1]}×{img.shape[0]}")
    print(f"Display range: {display_min:.1f} ~ {display_max:.1f} (center={center}, variance={variance})")
    print("Heatmap overlay strength: 62% (much stronger than before)")

    # Auto-generate 4 BPP levels around center
    multipliers = [1.5, 0.5, -0.5, -1.5]
    target_bpps = sorted([center + m * variance for m in multipliers], reverse=True)

    complexity_map = compute_complexity_map(img, args.window_size)

    # Strong contrast colormap
    colors_list = ['#000000', '#2b0000', '#5c0000', '#8f0000', '#c80000', '#ff1f1f', '#ff5e5e', '#ff9e9e', '#ffc9c9']
    cmap = LinearSegmentedColormap.from_list('strong_bpp', colors_list, N=512)

    fig = plt.figure(figsize=(22, 16))
    fig.suptitle(f'Per-pixel BPP Heatmap (4 levels) - STRONG overlay\n'
                 f'Display range: {display_min:.1f} ~ {display_max:.1f} | Local-variance logic',
                 fontsize=17, y=0.98)

    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.07], wspace=0.03, hspace=0.10)

    for idx, orig_bpp in enumerate(target_bpps):
        row, col = divmod(idx, 2)
        ax = fig.add_subplot(gs[row, col])

        raw_bpp = create_raw_bpp_map(complexity_map, orig_bpp)
        display_bpp = normalize_to_display_range(raw_bpp, center, variance)

        heat_rgb = bpp_to_rgb(display_bpp, cmap, display_min, display_max)
        
        # ==================== 加大热力权重（关键修改） ====================
        # 原图 38% + 热力 62% → 热力颜色明显覆盖图像
        overlay = cv2.addWeighted(img, 0.38, heat_rgb, 0.62, 0)

        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        ax.imshow(overlay_rgb)
        ax.set_title(f'Original {orig_bpp:.2f} bpp\n'
                     f'Display mean {display_bpp.mean():.2f}', fontsize=14)
        ax.axis('off')

        cv2.imwrite(f"{args.output}/overlay_{orig_bpp:.2f}orig.jpg", overlay)
        print(f"Saved overlay for original {orig_bpp:.2f} bpp (display mean {display_bpp.mean():.2f})")

    # Unified colorbar
    cax = fig.add_subplot(gs[:, 2])
    norm = plt.Normalize(display_min, display_max)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, cax=cax, orientation='vertical')
    cbar.set_label('Displayed BPP', fontsize=14)
    ticks = np.linspace(display_min, display_max, 7)
    cbar.set_ticks(ticks)
    cbar.ax.set_yticklabels([f'{t:.1f}' for t in ticks[::-1]])

    plt.tight_layout(rect=[0, 0, 0.94, 0.96])
    combined_path = f"{args.output}/COMBINED_BPP_center{center}_var{variance}_strong_overlay.png"
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nDone! Main figure saved → {combined_path}")
    print("热力颜色现在明显覆盖在图像上（62%权重）")
    print("高bpp区域（黑） vs 低bpp区域（亮红）区别已大幅增强")

if __name__ == "__main__":
    main()