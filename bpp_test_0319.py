# 文件名建议：plot_bpp.py
import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
import json
from typing import List, Tuple, Optional

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'SimHei']  # 支持中文
plt.style.use('ggplot')   # 或 'seaborn-v0_8' / 'default'


def extract_bpp_from_log(log_path: str) -> Tuple[List[int], List[float], List[float]]:
    """
    从 logger txt 或 stdout 重定向文件中提取 epoch / train_loss / bpp
    适配你目前打印格式：
        Epoch X 完成
          Avg Train Loss : xxx.xxx
          Current BPP    : x.xxx
          Best BPP       : x.xxx (epoch xx)
    """
    epochs = []
    losses = []
    bpps = []
    best_bpp = None
    best_epoch = None

    pattern_loss = re.compile(r"Avg Train Loss\s*:\s*([0-9.e+-]+)")
    pattern_bpp   = re.compile(r"Current BPP\s*:\s*([0-9.e+-]+)")
    pattern_best  = re.compile(r"Best BPP\s*:\s*([0-9.e+-]+)\s*\(epoch\s*(\d+)\)")

    with open(log_path, encoding="utf-8") as f:
        current_epoch = None
        for line in f:
            line = line.strip()
            if "=== Epoch" in line and "/" in line:
                m = re.search(r"Epoch\s*(\d+)\s*/", line)
                if m:
                    current_epoch = int(m.group(1))
            if "Avg Train Loss" in line:
                m = pattern_loss.search(line)
                if m and current_epoch is not None:
                    losses.append(float(m.group(1)))
                    epochs.append(current_epoch)
            if "Current BPP" in line:
                m = pattern_bpp.search(line)
                if m and current_epoch is not None:
                    bpps.append(float(m.group(1)))
            if "Best BPP" in line:
                m = pattern_best.search(line)
                if m:
                    best_bpp = float(m.group(1))
                    best_epoch = int(m.group(2))

    if len(epochs) != len(losses) or len(epochs) != len(bpps):
        print(f"警告：提取数量不匹配 loss={len(losses)} bpp={len(bpps)} epoch={len(epochs)}")

    return epochs, losses, bpps, best_bpp, best_epoch


def plot_training_curve(
    log_paths: List[str],
    labels: Optional[List[str]] = None,
    save_to: str = "bpp_training_curve.png",
    title: str = "Training BPP & Loss Curve",
    show_best: bool = True
):
    """画多条训练曲线对比"""
    if labels is None:
        labels = [Path(p).stem for p in log_paths]

    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=160)

    ax2 = ax1.twinx()   # 右边 y 轴

    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(log_paths))))

    for i, log_path in enumerate(log_paths):
        epochs, losses, bpps, best_bpp, best_ep = extract_bpp_from_log(log_path)

        if not epochs:
            print(f"跳过空日志：{log_path}")
            continue

        label = labels[i]
        color = colors[i % len(colors)]

        # 画 loss（左轴）
        ax1.plot(epochs, losses, color=color, linestyle="--", alpha=0.7,
                 label=f"{label} loss")
        # 画 BPP（右轴）
        ax1.plot(epochs, bpps, color=color, marker="o", markersize=4,
                 label=f"{label} BPP")

        if show_best and best_bpp is not None:
            ax2.axhline(best_bpp, color=color, linestyle=":", alpha=0.6, lw=1.5)
            ax2.text(epochs[-1]*1.02, best_bpp*1.01, f"{best_bpp:.4f}",
                     color=color, fontsize=9, va="bottom")

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Average NLL Loss", color="tab:blue")
    ax2.set_ylabel("BPP (bits per pixel)", color="tab:orange")

    ax1.tick_params(axis='y', labelcolor="tab:blue")
    ax2.tick_params(axis='y', labelcolor="tab:orange")

    lines1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(lines1, labels1, loc="upper right", fontsize=9)

    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_to:
        plt.savefig(save_to, dpi=200, bbox_inches="tight")
        print(f"已保存到：{save_to}")
    plt.show()


def plot_bpp_bar(
    methods: List[str],
    bpps: List[float],
    title="Kodak Dataset BPP Comparison",
    save_to="bpp_bar_kodak.png",
    ylim_top: float = None
):
    """简单柱状图对比多个方法的最终 BPP"""
    x = np.arange(len(methods))
    width = 0.6

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(x, bpps, width, color=plt.cm.Set2(range(len(methods))))

    ax.set_ylabel("BPP ↓")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha="right")

    # 在柱子上方标注数值
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{height:.4f}', ha='center', va='bottom', fontsize=10)

    if ylim_top:
        ax.set_ylim(0, ylim_top)

    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    if save_to:
        plt.savefig(save_to, dpi=180, bbox_inches="tight")
        print(f"保存柱状图：{save_to}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["curve", "bar"], default="curve",
                        help="curve=训练过程曲线   bar=最终BPP柱状对比")
    parser.add_argument("--logs", nargs="+", help="训练日志文件路径（curve模式使用）")
    parser.add_argument("--labels", nargs="*", help="每条曲线的图例名称")
    parser.add_argument("--methods", nargs="*", help="bar模式：方法名称")
    parser.add_argument("--bpps", nargs="*", type=float, help="bar模式：对应BPP值")
    parser.add_argument("--out", default="bpp_result.png", help="输出图片路径")
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    if args.mode == "curve":
        if not args.logs:
            print("请提供至少一个 --logs xxx.log")
            exit(1)
        plot_training_curve(
            log_paths=args.logs,
            labels=args.labels,
            save_to=args.out,
            title=args.title or "Pixel Autoregressive Compression Training Curve"
        )

    elif args.mode == "bar":
        if not args.methods or not args.bpps:
            print("bar模式需要 --methods 和 --bpps")
            print("示例：")
            print('  python plot_bpp.py --mode bar --methods "Baseline" "Ours w/ prefix=16" "Ours w/ INP" \\')
            print('    --bpps 0.92 0.78 0.71 --out kodak_bpp.png')
            exit(1)
        plot_bpp_bar(
            methods=args.methods,
            bpps=args.bpps,
            title=args.title or "BPP Comparison on Kodak",
            save_to=args.out
        )