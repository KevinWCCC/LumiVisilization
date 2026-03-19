import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

# ====================== 项目内导入 ======================
from readout_head import PixelInputEmbedding, NumericReadoutHead
from intra_patch_pos_embed import load_intra_pos
from soft_prefix import load_prefix
import utils
import data_loader
from data_loader import PixelJsonlDataset


def plot_tsne(X: np.ndarray, colors: np.ndarray, title: str, save_path: str):
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
    X_2d = tsne.fit_transform(X)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=colors, cmap='viridis', s=8, alpha=0.7)
    plt.colorbar(scatter, label='Pixel Value (0-255)')
    plt.title(title)
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 已保存: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="/home/vipuser/Model/QWEN_3_0.6B")
    parser.add_argument("--dataset", type=str, default="k", choices=["k", "s", "b"])
    parser.add_argument("--data_jsonl", type=str, required=True,
                        help="评估用的 jsonl，例如 data/kodak_p16_7-24.jsonl")
    parser.add_argument("--max_samples", type=int, default=30)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = args.device
    print("🚀 开始加载模型与组件...")

    # 1. LLM
    tokenizer, model = utils.load_llm_model(args.model_id, "", device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    H = model.config.hidden_size

    # 2. 加载四个组件（自动使用你训练时保存的最佳 ckpt）
    pixel_emb = PixelInputEmbedding.load(f"models/pixel_emb_{args.dataset}_stage1_best.pt", device=device)
    readout_head = NumericReadoutHead.load(f"models/HeadPixel_{args.dataset}_stage1_best.pt", device=device)

    intra_pos = None
    inp_path = f"models/INP_{args.dataset}_stage1_best.pt"
    if Path(inp_path).exists():
        intra_pos = load_intra_pos(inp_path, model, device)
        print(f"✅ Loaded INP from {inp_path}")

    prefix_tau = None
    sp_path = f"models/SP_{args.dataset}_stage2_best.pt"
    if Path(sp_path).exists():
        prefix_tau, _ = load_prefix(sp_path, device=device, model=model)
        print(f"✅ Loaded Soft Prefix from {sp_path}")

    # 3. 加载少量数据用于 hidden states 可视化
    format_args = data_loader.load_data_format_config("configs/default_format.yaml")
    dataset = PixelJsonlDataset(
        [args.data_jsonl], tokenizer, format_args,
        shuffle=False, with_full_img=False, with_neighbor_patches=False
    )

    all_pixel_vals = []
    reps = {
        "pixel_emb_only": [],
        "pixel_emb_inp": [],
        "hidden_none": [], "hidden_inp": [], "hidden_prefix": [], "hidden_both": [],
        "head_input": []
    }

    print(f"📊 从 {min(args.max_samples, len(dataset))} 个样本提取表征...")
    with torch.no_grad():
        for i, sample in enumerate(dataset):
            if i >= args.max_samples:
                break
            prompt_ids_list, pixel_values_list, _, _, _ = sample
            if len(pixel_values_list) < 10:
                continue

            pix_t = torch.tensor(pixel_values_list, dtype=torch.long, device=device).unsqueeze(0)
            T = pix_t.size(1)
            all_pixel_vals.extend(pixel_values_list)

            # Pixel Embedding only
            p_emb = pixel_emb(pix_t).squeeze(0).cpu().numpy()
            reps["pixel_emb_only"].append(p_emb)

            # Pixel + INP
            if intra_pos is not None:
                token_idx = torch.arange(T, device=device)
                pos = intra_pos(token_idx).unsqueeze(0).to(model.dtype)
                p_inp = (pixel_emb(pix_t) + pos).squeeze(0).cpu().numpy()
                reps["pixel_emb_inp"].append(p_inp)

            # Hidden states ablation（关键！）
            prompt_t = torch.tensor(prompt_ids_list, dtype=torch.long, device=device).unsqueeze(0)
            prompt_emb = model.get_input_embeddings()(prompt_t).to(model.dtype)
            pix_emb_raw = pixel_emb(pix_t).to(model.dtype)

            configs = [
                ("none", None, None),
                ("inp", None, intra_pos),
                ("prefix", prefix_tau, None),
                ("both", prefix_tau, intra_pos)
            ]

            for name, tau, inp_mod in configs:
                seq_parts = [prompt_emb]
                if inp_mod:
                    token_idx = torch.arange(T, device=device)
                    pos = inp_mod(token_idx).unsqueeze(0).to(model.dtype)
                    seq_parts.append(pix_emb_raw + pos)
                else:
                    seq_parts.append(pix_emb_raw)
                seq = torch.cat(seq_parts, dim=1)

                if tau is not None:
                    tau_b = tau.unsqueeze(0).to(model.dtype)
                    seq = torch.cat([tau_b, seq], dim=1)
                    prefix_len = tau_b.size(1)
                else:
                    prefix_len = 0

                attn = torch.ones((1, seq.size(1)), device=device, dtype=torch.long)

                base = getattr(model, "model", None) or model
                hs = base(inputs_embeds=seq, attention_mask=attn, use_cache=False).last_hidden_state

                Lp = prompt_emb.size(1)
                rows = torch.arange(prefix_len + Lp - 1, prefix_len + Lp - 1 + T, device=device)
                h_pred = hs[:, rows, :].squeeze(0).float().cpu().numpy()
                reps[f"hidden_{name}"].append(h_pred)

                # Head input（最终进入 Readout Head 的特征）
                if name == "both":
                    reps["head_input"].append(h_pred)

    pixel_vals = np.array(all_pixel_vals)

    # ====================== 画图 ======================
    for name, lst in reps.items():
        if not lst:
            continue
        X = np.vstack(lst)
        title = f"t-SNE — {name.replace('_', ' ').title()}"
        plot_tsne(X, pixel_vals[:len(X)], title, f"tsne_{name}_{args.dataset}.png")

    print("\n🎉 全部 t-SNE 图生成完成！")
    print("重点观察建议：")
    print("   • pixel_emb_only.png          → Pixel Embedding 是否学到数值连续性")
    print("   • pixel_emb_inp.png           → INP 是否给位置增加了结构")
    print("   • hidden_* .png               → Prefix / INP 如何让相同像素值更紧密聚类（核心作用）")
    print("   • head_input.png              → Readout Head 前的特征可分性（最终压缩效果）")


if __name__ == "__main__":
    main()