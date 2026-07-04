**t-SNE 作图逻辑与您这幅图像的解读**

t-SNE（t-Distributed Stochastic Neighbor Embedding，t-分布随机邻域嵌入）是目前高维数据可视化中最常用的非线性降维方法之一，由 van der Maaten & Hinton (2008) 提出。它特别适合神经网络隐状态（hidden states）、词向量、图像特征等高维数据的 2D/3D 展示，核心思想是**在低维空间中尽可能保留高维空间中的局部邻域结构（local neighborhood structure）**，同时缓解“拥挤问题”（crowding problem）。

### 一、t-SNE 的数学逻辑与作图流程

给定高维数据 \(\mathbf{X} = \{\mathbf{x}_1, \dots, \mathbf{x}_N\} \in \mathbb{R}^D\)（在本实验中 \(D \approx 4096\)，对应 LLM hidden size），目标是找到低维嵌入 \(\mathbf{Y} = \{\mathbf{y}_1, \dots, \mathbf{y}_N\} \in \mathbb{R}^2\)。

**1. 高维空间相似度建模**
- 对每个点 \(\mathbf{x}_i\) 定义到其他点的**条件概率**（高斯核）：
  \[
  p_{j|i} = \frac{\exp\left(-\frac{\|\mathbf{x}_i - \mathbf{x}_j\|^2}{2\sigma_i^2}\right)}{\sum_{k \neq i}\exp\left(-\frac{\|\mathbf{x}_i - \mathbf{x}_k\|^2}{2\sigma_i^2}\right)}
  \]
- 带宽 \(\sigma_i\) 通过二分搜索自适应确定，使得**困惑度**（perplexity）满足 \(\text{Perp}(P_i) = 2^{H(P_i)} \approx\) 用户设定值（通常 5~50）。困惑度越大，算法考虑的“邻居范围”越大。
- 对称化得到联合概率：
  \[
  p_{ij} = \frac{p_{j|i} + p_{i|j}}{2N}
  \]

**2. 低维空间相似度建模**
- 使用自由度为 1 的 **t-分布**（重尾特性）：
  \[
  q_{ij} = \frac{(1 + \|\mathbf{y}_i - \mathbf{y}_j\|^2)^{-1}}{\sum_{k \neq l}(1 + \|\mathbf{y}_k - \mathbf{y}_l\|^2)^{-1}}
  \]
- t-分布比高斯分布尾部更重，能有效缓解高维到低维映射时的“拥挤”现象。

**3. 优化目标**
- 最小化两个分布之间的 **Kullback-Leibler (KL) 散度**：
  \[
  C = \sum_{i \neq j} p_{ij} \log \frac{p_{ij}}{q_{ij}}
  \]
- 使用带 momentum 的梯度下降进行优化（早期版本用 Barnes-Hut 近似实现 \(O(N \log N)\) 复杂度）。

**关键参数及其影响**
- `perplexity`：最重要。值越小越关注极局部结构，值越大越能反映全局结构，但可能丢失细粒度邻居关系。
- `max_iter` / `n_iter`：迭代次数，影响收敛质量。
- `init`：`'pca'`（推荐，稳定且快）或 `'random'`。
- `learning_rate`：通常设为 `'auto'` 或 200~1000。

**注意事项**（学术使用时必须强调）：
- t-SNE **更可靠地保留局部结构**，全局距离和簇间相对位置可能被严重扭曲。
- 结果具有随机性（随机初始化 + 随机优化路径），建议固定 `random_state` 并多次运行观察稳定模式。
- 不要用 t-SNE 做定量距离比较或聚类验证（可用 silhouette score 等辅助）。

### 二、您这幅 t-SNE 图像的具体解读

您这幅图是针对 **LLM 最后一层（或指定层平均）pixel token hidden states** 的可视化结果。数据构成：
- 多个 image patch（经 `--num_patches` 控制）
- 每个 patch 展平为 \(T \approx 768\) 个像素（16×16×3 典型情况）
- 同时提取 **Original LLM** 与 **Stage-1 Trained Model**（加载了 `PixelInputEmbedding`、`IntraPatchPositionEmbedding`、可选 Prefix）的隐状态
- 总点数经 `--max_points` 子采样后约 23k

**图中视觉元素**：
- 横纵轴：t-SNE 1 / t-SNE 2（仅为抽象嵌入坐标，无物理意义）。
- **双重颜色编码**：
  - 类别区分（Original vs Trained）：通过 seaborn `hue` 参数实现（典型蓝/红系）。
  - 像素值映射（0~255）：viridis colormap 叠加（颜色条从紫/蓝→黄），用于观察表示空间中像素强度的组织方式。

**核心观察与学术推断**（基于您的代码逻辑与典型运行结果）：

1. **模型间表示空间的分离（最关键观察）**  
   如果图中出现两个相对分离的点云（例如 Original 点云集中在某一区域，Trained 点云偏移到另一区域，或形成明显不同的“岛”），这表明您的 **Stage-1 训练（PixelInputEmbedding 的 7 维特征工程 + Intra-Patch Positional Embedding）成功地将像素隐状态从通用文本 token 表示空间“迁移”到了一个更适合像素自回归建模的特定流形**。这直接支持了您“pixel mode”与两阶段训练的核心假设——训练后的模型学到了与原始 LLM 显著不同的内部表征。

2. **像素值在隐空间中的组织结构**  
   在每个主要点云内部观察 viridis 颜色的分布：若低像素值（暗色）和高像素值（亮色）呈现明显梯度、子簇或有序排列，说明 hidden states 较好地编码了像素强度这一关键信息。这对下游 `NumericReadoutHead` 的 256-way cross-entropy 非常有利——相近像素值在表示空间中更接近，便于分类器学习。反之，若颜色在簇内近似随机分布，则可能表示当前表征更多捕捉了上下文/位置信息而非数值本身。

3. **簇的几何与密度特征**  
   - Trained 簇是否比 Original 更紧凑（compact）？紧凑通常意味着学到了更一致、鲁棒的像素特征。
   - 是否存在细长（elongated）或流形状结构？虽然 t-SNE 未显式建模空间关系，但若 patch 内相邻像素在图中也相对接近，可间接反映 Intra-Patch Positional Embedding 的作用。
   - 密度差异：高密度区域往往对应自然图像中常见的像素模式（中值灰度等）。

4. **重叠与边界区域**  
   若两个模型的点有少量混杂，可能是某些极端像素值（0 或 255）或特定上下文模式在两种模型中表示仍较为相似。这不一定是坏事，但值得结合 BPP 指标进一步分析。

**可视化技术细节提醒**：
您当前代码同时使用了 `sns.scatterplot(hue=...)` + `plt.scatter(c=...)` 叠加，可能导致颜色映射以 viridis 为主、类别区分被部分覆盖或产生视觉混淆。建议后续改进为：
- 两个子图分别展示 Original 与 Trained；或
- 使用不同 marker（`'o'` vs `'^'`）表示模型类型，统一用颜色表示像素值；或
- 改用 `plotly` 实现交互式可视化（hover 显示 pixel value、patch id、layer 等元信息）。

**在您研究中的意义**：
这幅图可作为**定性证据**有力支撑您的核心 claim：引入可学习的 PixelInputEmbedding 与 Intra-Patch Positional Embedding 后，LLM 对像素序列的建模能力得到实质性提升（从“文本通用表示”转向“像素专用表示”）。结合定量指标（BPP、train/eval loss、silhouette score 等），可形成完整的 ablation study 叙事（例如 with/without INP、with/without noise 的 t-SNE 对比）。

如果您能简要描述图中观察到的具体现象（例如“两个簇明显分离，Trained 更紧凑，像素值在 Trained 内呈清晰梯度”或“仍有较多重叠”），我可以给出更精准的解读与针对性改进建议。需要我帮您修改可视化代码（添加子图、3D t-SNE、UMAP 对比、定量指标计算等），随时告诉我！