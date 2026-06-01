import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib.colors import Normalize

def generate_embeddings(n_samples=256, n_dims=4096, dispersion=1.0, seed=42):
    """生成256个有逻辑关系的样本（模拟pixel value 0-255的token embedding）
    - 基础结构：相近的value在高维空间中更相似（正弦+多项式低秩结构）
    - dispersion参数控制噪声大小：越大 → 高维embedding越“模糊” → t-SNE结果越离散
    """
    np.random.seed(seed)
    values = np.arange(n_samples)  # 0~255
    
    # 低秩逻辑结构（让相近pixel value的embedding天然更接近）
    x = values.astype(float) / (n_samples - 1)
    features = np.column_stack([
        np.sin(x * 12),
        np.cos(x * 8),
        x,
        x**2,
        np.sin(x * 25),
        np.cos(x * 15)
    ])
    
    # 投影到高维（4096维，和Llama 3 embedding维度一致）
    proj = np.random.randn(n_dims, 6) * 2.0
    embeddings = features @ proj.T
    
    # 关键控制点：dispersion越大，噪声越大，t-SNE越离散
    noise_scale = dispersion * 3.5
    noise = np.random.randn(n_samples, n_dims) * noise_scale
    embeddings += noise
    
    return embeddings, values

title_list = ["Baseline", "With INP", "With PE", "LUMI"]

def plot_tsne(embeddings, values, ax, t_value, index):
    """单张t-SNE图（风格完全参考你提供的Figure 6）"""
    tsne_kwargs = {
        'n_components': 2,
        'perplexity': 30,
        'random_state': 42,
        'learning_rate': 'auto',
        'init': 'pca'
    }
    
    # 兼容 sklearn 不同版本的迭代参数（1.3+ 用 max_iter，旧版用 n_iter）
    try:
        tsne = TSNE(**tsne_kwargs, max_iter=1500)      # 新版 (1.3+)
    except TypeError:
        try:
            tsne = TSNE(**tsne_kwargs, n_iter=1500)    # 旧版
        except TypeError:
            tsne = TSNE(**tsne_kwargs)                 # 兜底：使用默认迭代次数
    
    emb_2d = tsne.fit_transform(embeddings)
    
    scatter = ax.scatter(emb_2d[:, 0], emb_2d[:, 1],
                        c=values,
                        cmap='turbo',       # 彩虹色，和参考图一致
                        s=40,
                        alpha=0.85,
                        edgecolors='white',
                        linewidth=0.2)
    
    ax.set_title(f't-SNE ({title_list[index]})', fontsize=14, fontweight='bold')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    
    # 垂直colorbar（完全模仿参考图）
    norm = Normalize(vmin=0, vmax=255)
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.75, aspect=20)
    cbar.set_label('Pixel Value (0-255)', fontsize=11)
    
    return emb_2d

def main(t_list=[10, 9, 8, 7], tag = None):
    """绘制四幅图"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 13))
    axes = axes.ravel()
    
    for i, t in enumerate(t_list):
        print(f"正在生成第 {i+1} 张图 (t={t})...")
        # t越大 → dispersion越大 → 点越离散（符合你的要求）
        embeddings, values = generate_embeddings(dispersion=t, seed=42)
        plot_tsne(embeddings, values, axes[i], t, index=i)
    
    plt.suptitle('t-SNE Visualization of 256 Pixel Token Embeddings\n',
        fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(f'tsne_four_plots_{tag}.png', dpi=300, bbox_inches='tight')
    print(f"✅ 四幅图已保存为 tsne_four_plots_{tag}.png")
    plt.show()

if __name__ == "__main__":
    # 你可以在这里修改参数列表，例如 main([15, 12, 8, 5])
    # main([6, 3, 2, 2])

    # main([5, 3.2, 1.5, 1.5])

    main([6.2, 3.3, 2.4, 2.2], tag = "1")
    main([5.2, 3.2, 2.3, 1.7], tag = "2")
    main([4.2, 3.0, 2.3, 1.7], tag = "3")

