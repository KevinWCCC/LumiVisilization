#!/usr/bin/env python3
"""
模拟 t-SNE 风格的 Token 嵌入可视化图（增强版）
- (a) ASCII Tokens (0-127): 颜色分箱 + 局部聚类（聚类效果明显）
- (b) Numerical Tokens (0-255): 高密度低噪声参数曲线 + 紧致聚类 + 极淡参考线
  → 不同 intensity 的色点清晰形成「连线」或「聚类」结构
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

# 设置随机种子，保证可复现
np.random.seed(42)

def generate_ascii_data(n_points=160):
    """生成 ASCII tokens 的模拟数据：6个颜色分箱的局部聚类"""
    bins = [0, 20, 40, 60, 80, 100, 128]
    n_clusters = len(bins) - 1
    
    # 手动布置 6 个聚类中心，模拟图像中颜色分组散布的效果
    centers = np.array([
        [-32,  28],   # bin 0 (0-19) 蓝色区域
        [  8,  32],   # bin 1 (20-39) 绿色
        [ 35,  15],   # bin 2 (40-59) 紫色
        [-25, -18],   # bin 3 (60-79) 粉/紫红
        [ 22, -28],   # bin 4 (80-99) 黄色
        [-38,   5],   # bin 5 (100-127) 青色
    ])
    
    stds = [7.5, 6.5, 8.0, 7.0, 6.0, 7.8]  # 不同聚类扩散程度
    
    X_list = []
    val_list = []
    
    points_per = n_points // n_clusters
    remainder = n_points % n_clusters
    
    for i in range(n_clusters):
        n = points_per + (1 if i < remainder else 0)
        cx, cy = centers[i]
        std = stds[i]
        
        dx = np.random.normal(0, std, n)
        dy = np.random.normal(0, std * 0.85, n)
        
        X_list.append(np.column_stack([cx + dx, cy + dy]))
        
        vmin, vmax = bins[i], bins[i+1]
        vals = np.random.randint(vmin, vmax, n)
        val_list.append(vals)
    
    X = np.vstack(X_list)
    values = np.concatenate(val_list)
    
    # 轻微打乱顺序，让同色点不完全连在一起
    perm = np.random.permutation(len(X))
    return X[perm, 0], X[perm, 1], values[perm]


def generate_numerical_data(n_total=280):
    """生成 Numerical tokens 的模拟数据：
    强调「连线」效果：高密度 + 低噪声 的参数化曲线/环形结构
    + 少量紧致聚类，模拟不同 intensity 形成清晰的连线或聚类
    """
    bins = [0, 50, 100, 150, 200, 250, 256]
    n_segments = len(bins) - 1
    
    X_list = []
    val_list = []
    curve_guides = []  # 用于画极淡的参考线（增强连线视觉）
    
    for i in range(n_segments):
        vmin, vmax = bins[i], bins[i+1]
        n_pts = 52 + i * 7   # 更高密度，让连线更明显
        
        # 不同 bin 使用不同形状，更贴近原图的曲线/环结构
        if i == 0:  # 蓝色 (0-49) - 中心密集小弧 + 聚类
            t = np.linspace(0, 2.6 * np.pi, n_pts)
            x = -1.5 + 3.8 * np.sin(1.3 * t) + 0.6 * np.sin(3.5 * t)
            y =  1.0 + 2.8 * np.cos(1.1 * t) - 0.4 * np.sin(2.8 * t)
            noise_std = 0.28
        elif i == 1:  # 绿色 (50-99) - 左上 wavy 链
            t = np.linspace(-0.8, 3.2 * np.pi, n_pts)
            x = -9.5 + 4.5 * np.sin(0.95 * t) + 0.3 * t
            y =  6.5 + 3.2 * np.cos(0.85 * t) - 0.25 * t * 0.6
            noise_std = 0.26
        elif i == 2:  # 紫色 (100-149) - 右侧弧形/半环
            t = np.linspace(0.2, 2.9 * np.pi, n_pts)
            x =  6.0 + 5.5 * np.sin(1.05 * t + 0.3)
            y =  2.5 + 4.8 * np.cos(0.95 * t + 0.6) - 0.8 * np.sin(2.2 * t)
            noise_std = 0.30
        elif i == 3:  # 粉色 (150-199) - 底部大 U 形 / 环（最明显连线）
            t = np.linspace(0, 3.1 * np.pi, n_pts)
            x = -2.0 + 7.5 * np.sin(0.75 * t) + 0.4 * t * 0.5
            y = -9.5 + 5.5 * np.cos(0.68 * t) + 0.6 * np.sin(2.6 * t)
            noise_std = 0.25
        elif i == 4:  # 黄色 (200-249) - 中部偏左弧线
            t = np.linspace(-0.5, 2.8 * np.pi, n_pts)
            x = -5.5 + 4.2 * np.sin(1.15 * t + 1.2) + 0.2 * t
            y = -1.5 + 3.8 * np.cos(1.05 * t + 0.8)
            noise_std = 0.27
        else:  # 青色 (250-255) - 右上散弧 + 小环
            t = np.linspace(0.5, 2.4 * np.pi, n_pts)
            x =  8.5 + 4.0 * np.sin(1.4 * t) + 0.5 * np.cos(3.0 * t)
            y =  7.0 + 3.5 * np.cos(1.25 * t) - 0.7 * np.sin(2.5 * t)
            noise_std = 0.29
        
        # 添加少量紧致聚类（增强「聚类」效果）
        if i in [0, 2, 3]:
            n_extra = 12
            cx, cy = np.mean(x), np.mean(y) + (1.5 if i == 3 else -1.2 if i == 0 else 2.0)
            extra_x = cx + np.random.normal(0, 1.1, n_extra)
            extra_y = cy + np.random.normal(0, 0.9, n_extra)
            x = np.concatenate([x, extra_x])
            y = np.concatenate([y, extra_y])
            n_pts += n_extra
        
        # 低噪声 → 点紧贴曲线，形成清晰「连线」视觉
        x += np.random.normal(0, noise_std, len(x))
        y += np.random.normal(0, noise_std * 0.9, len(x))
        
        X_list.append(np.column_stack([x, y]))
        
        vals = np.random.randint(vmin, vmax, len(x))
        val_list.append(vals)
        
        # 保存理想曲线（用于画极淡参考线，增强连线感）
        curve_guides.append((x.copy() - np.random.normal(0, noise_std, len(x)), 
                             y.copy() - np.random.normal(0, noise_std * 0.9, len(x))))
    
    X = np.vstack(X_list)
    values = np.concatenate(val_list)
    
    perm = np.random.permutation(len(X))
    return X[perm, 0], X[perm, 1], values[perm], curve_guides


# ==================== 绘图 ====================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.8), constrained_layout=True)

# 颜色定义（从低到高：蓝 -> 绿 -> 紫 -> 粉 -> 黄 -> 青）
colors = ['#1e90ff', '#3cb371', '#9370db', '#ff69b4', '#ffd700', '#00ced1']
cmap = ListedColormap(colors)

# ---------- (a) ASCII Tokens ----------
x_a, y_a, val_a = generate_ascii_data(n_points=155)

boundaries_a = [0, 20, 40, 60, 80, 100, 128]
norm_a = BoundaryNorm(boundaries_a, ncolors=len(colors))

sc1 = ax1.scatter(x_a, y_a, c=val_a, cmap=cmap, norm=norm_a,
                  s=22, alpha=0.85, edgecolors='none', linewidths=0)

ax1.set_xlim(-58, 58)
ax1.set_ylim(-44, 44)
ax1.set_aspect('equal', adjustable='box')
ax1.tick_params(labelsize=8)
ax1.set_facecolor('white')

# Colorbar for (a)
cbar1 = fig.colorbar(sc1, ax=ax1, shrink=0.78, aspect=22, pad=0.015)
cbar1.set_ticks([10, 30, 50, 70, 90, 113])
cbar1.set_ticklabels(['0-19', '20-39', '40-59', '60-79', '80-99', '100-127'])
cbar1.ax.tick_params(labelsize=7)

# ---------- (b) Numerical Tokens ----------
x_b, y_b, val_b, curve_guides = generate_numerical_data()

boundaries_b = [0, 50, 100, 150, 200, 250, 256]
norm_b = BoundaryNorm(boundaries_b, ncolors=len(colors))

# 先画极淡的参考曲线（增强「连线」视觉效果，不影响 scatter 主体）
for i, (cx, cy) in enumerate(curve_guides):
    color_idx = min(i, len(colors)-1)
    ax2.plot(cx, cy, color=colors[color_idx], alpha=0.09, linewidth=2.2, zorder=1)

sc2 = ax2.scatter(x_b, y_b, c=val_b, cmap=cmap, norm=norm_b,
                  s=21, alpha=0.88, edgecolors='none', linewidths=0, zorder=2)

ax2.set_xlim(-16, 16)
ax2.set_ylim(-16, 16)
ax2.set_aspect('equal', adjustable='box')
ax2.tick_params(labelsize=8)
ax2.set_facecolor('white')

# Colorbar for (b)
cbar2 = fig.colorbar(sc2, ax=ax2, shrink=0.78, aspect=22, pad=0.015)
cbar2.set_ticks([25, 75, 125, 175, 225, 252])
cbar2.set_ticklabels(['0-49', '50-99', '100-149', '150-199', '200-249', '250-255'])
cbar2.ax.tick_params(labelsize=7)

# 底部标题 (模拟原图风格)
ax1.set_xlabel('(a) ASCII Tokens', fontsize=11, labelpad=6)
ax2.set_xlabel('(b) Numerical Tokens', fontsize=11, labelpad=6)

# 去掉多余的 spines 让图更干净（可选）
for ax in [ax1, ax2]:
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)

plt.savefig('./artifacts/simulated_token_tsne.png', 
            dpi=220, bbox_inches='tight', facecolor='white', edgecolor='none')
print("图像已保存到: ./artifacts/simulated_token_tsne.png")