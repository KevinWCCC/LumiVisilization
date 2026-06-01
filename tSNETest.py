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

modelNamePt = "llama-k-0319"

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
    parser.add_argument("--model_id", type=str, default="/home/vipuser/Model/LLAMA_3.1_B")
    parser.add_argument("--dataset", type=str, default="k", choices=["k", "s", "b"])
    parser.add_argument("--data_jsonl", type=str, required=True,
                        help="评估用的 jsonl，例如 data/kodak_p16_7-24.jsonl")
    parser.add_argument("--max_samples", type=int, default=30)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = args.device
    print("🚀 开始加载模型与组件...")


    # pixel_emb = PixelInputEmbedding.load(f"models/pixel_emb_{args.dataset}_stage1_best.pt", device=device)

    # readout_head = NumericReadoutHead.load(f"models/HeadPixel_{args.dataset}_stage1_best.pt", device=device)

    # inp_path = f"models/INP_{args.dataset}_stage1_best.pt"   # 你实际文件名

    # sp_path = f"models/SP_{args.dataset}_stage1_best.pt"     # 你实际文件名（改成你的 stage2 或 stage1）

    pixel_emb = PixelInputEmbedding.load(f"./models/pixel_emb_{modelNamePt}_stage1.pt", device=device)
    readout_head = NumericReadoutHead.load(f"./models/HeadPixel_{modelNamePt}_stage1.pt", device=device)
    inp_path = f"./models/INP_{modelNamePt}_stage1.pt"
    sp_path = f"./models/SP_{modelNamePt}_stage1.pt"

    # 1. LLM
    tokenizer, model = utils.load_llm_model(args.model_id, "", device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    llm_hidden_size = model.config.hidden_size
    print(f"LLM hidden_size = {llm_hidden_size}")

    # 2. 加载四个组件

    intra_pos = None
    if Path(inp_path).exists():
        intra_pos = load_intra_pos(inp_path, model, device)
        print(f"✅ Loaded INP from {inp_path}")

    prefix_tau = None
    if Path(sp_path).exists():
        prefix_tau, _ = load_prefix(sp_path, device=device, model=model)
        print(f"✅ Loaded Soft Prefix from {sp_path}")

    # ====================== 【方案一】维度自动适配 ======================
    # 1. Pixel Embedding 适配（最常见问题来源）
    dummy_pix = torch.zeros((1, 8), dtype=torch.long, device=device)
    pe_out = pixel_emb(dummy_pix)
    if pe_out.shape[-1] != llm_hidden_size:
        print(f"⚠️  Pixel Embedding 输出 dim {pe_out.shape[-1]} → 适配到 {llm_hidden_size}")
        pe_adapter = torch.nn.Linear(pe_out.shape[-1], llm_hidden_size, bias=False).to(device, pe_out.dtype)
        with torch.no_grad():
            # 简单线性投影（可替换为更高级的初始化）
            pe_adapter.weight.data.copy_(torch.eye(llm_hidden_size)[:pe_out.shape[-1], :])
        # 把 adapter 绑定到 pixel_emb 上（后续使用）
        pixel_emb._adapter = pe_adapter
        def adapted_pixel_emb(x):
            emb = pixel_emb.original_forward(x) if hasattr(pixel_emb, 'original_forward') else pixel_emb(x)
            return pixel_emb._adapter(emb)
        pixel_emb.original_forward = pixel_emb.forward
        pixel_emb.forward = adapted_pixel_emb

    # 2. Prefix 适配
    if prefix_tau is not None and prefix_tau.shape[-1] != llm_hidden_size:
        print(f"⚠️  Soft Prefix dim {prefix_tau.shape[-1]} → 适配到 {llm_hidden_size}")
        prefix_adapter = torch.nn.Linear(prefix_tau.shape[-1], llm_hidden_size, bias=False).to(device, prefix_tau.dtype)
        with torch.no_grad():
            prefix_adapter.weight.data.copy_(torch.eye(llm_hidden_size)[:prefix_tau.shape[-1], :])
        prefix_tau = prefix_adapter(prefix_tau)

    # 3. IntraPos 适配（在 forward 中处理）
    if intra_pos is not None:
        dummy_idx = torch.arange(5, device=device)
        inp_out = intra_pos(dummy_idx)
        if inp_out.shape[-1] != llm_hidden_size:
            print(f"⚠️  INP 输出 dim {inp_out.shape[-1]} → 适配到 {llm_hidden_size}")
            inp_adapter = torch.nn.Linear(inp_out.shape[-1], llm_hidden_size, bias=False).to(device, inp_out.dtype)
            with torch.no_grad():
                inp_adapter.weight.data.copy_(torch.eye(llm_hidden_size)[:inp_out.shape[-1], :])
            intra_pos._adapter = inp_adapter
            original_inp = intra_pos.forward
            intra_pos.forward = lambda x: inp_adapter(original_inp(x))

    print("✅ 维度适配完成，所有组件已对齐 LLM hidden_size")

    # 3. 加载数据
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

            # Pixel Embedding only（已适配）
            p_emb = pixel_emb(pix_t).squeeze(0).cpu().numpy()
            reps["pixel_emb_only"].append(p_emb)

            # Pixel + INP（已适配）
            if intra_pos is not None:
                token_idx = torch.arange(T, device=device)
                pos = intra_pos(token_idx).unsqueeze(0)
                p_inp = (pixel_emb(pix_t) + pos).squeeze(0).cpu().numpy()
                reps["pixel_emb_inp"].append(p_inp)

            # Hidden states ablation
            prompt_t = torch.tensor(prompt_ids_list, dtype=torch.long, device=device).unsqueeze(0)
            prompt_emb = model.get_input_embeddings()(prompt_t).to(model.dtype)
            pix_emb_raw = pixel_emb(pix_t).to(model.dtype)   # 已适配

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

                seq = torch.cat(seq_parts, dim=1)   # ← 现在维度已保证一致

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

                if name == "both":
                    reps["head_input"].append(h_pred)

    pixel_vals = np.array(all_pixel_vals)

    # ====================== 画图 ======================
    for name, lst in reps.items():
        if not lst:
            continue
        X = np.vstack(lst)
        title = f"t-SNE — {name.replace('_', ' ').title()}"
        plot_tsne(X, pixel_vals[:len(X)], title, f"tsne_{name}_{args.dataset}_fixed.png")

    print("\n🎉 全部 t-SNE 图生成完成！（已自动修复维度问题）")
    print("文件名后缀 _fixed.png 表示本次修复版本")


if __name__ == "__main__":
    main()