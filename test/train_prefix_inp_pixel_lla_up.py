import os
import math
import time
import random
import argparse
from datetime import datetime
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import data_loader
from data_loader import PixelJsonlDataset
from logger import setup_logger, close_logger

# 原有模块
from soft_prefix import SoftPrefix, PrefixConfig, save_prefix_ckpt
from intra_patch_pos_embed import (
    IntraPatchPositionEmbedding, IntraPatchPosConfig,
    save_intra_pos, load_intra_pos,
)
import utils
import evaluate
from evaluate import _mean_selected_hidden

# Pixel & Head
from readout_head import NumericReadoutHead, ReadoutConfig, PixelInputEmbedding, PixelInputEmbeddingConfig

DEBUG_SAMPLE = 100


def _estimate_epoch_steps(num_samples: int, batch_size: int, max_samples: int, debug_mode: bool) -> int:
    steps = math.ceil(num_samples / max(1, batch_size))
    if max_samples > 0:
        steps = min(steps, math.ceil(max_samples / max(1, batch_size)))
    if debug_mode:
        steps = min(steps, DEBUG_SAMPLE + 1)
    return max(1, steps)


def _warmup_scale(step_idx: int, warmup_steps: int) -> float:
    if warmup_steps <= 0:
        return 1.0
    if step_idx < warmup_steps:
        return float(step_idx + 1) / float(warmup_steps)
    return 1.0


def _apply_lr_scale(optimizer: torch.optim.Optimizer, base_lrs: List[float], scale: float) -> None:
    for group, base_lr in zip(optimizer.param_groups, base_lrs):
        group["lr"] = base_lr * scale


def _format_group_lrs(optimizer: torch.optim.Optimizer) -> str:
    parts = []
    for idx, group in enumerate(optimizer.param_groups):
        name = group.get("name", f"group{idx}")
        parts.append(f"{name}={group['lr']:.3e}")
    return ", ".join(parts)


def _save_stage_ckpt(args, dataset: str, stage: int, tag: str, pixel_emb, readout_head, intra_pos, prefix) -> None:
    if prefix is not None:
        save_prefix_ckpt(f"models/SP_{dataset}{tag}.pt", prefix.prefix.detach())
    if stage == 1:
        pixel_emb.save(f"models/pixel_emb_{dataset}{tag}.pt")
        readout_head.save(f"models/HeadPixel_{dataset}{tag}.pt")
        if intra_pos is not None:
            save_intra_pos(intra_pos, f"models/INP_{dataset}{tag}.pt")

# ====================== init_prefix（保持原样） ======================
def init_prefix(model, tok, args, hidden_size):
    base_model = model.module if hasattr(model, 'module') else model
    emb = base_model.get_input_embeddings()
    emb_dtype = emb.weight.dtype
    emb_device = emb.weight.device
    prefix = SoftPrefix(
        hidden_size, emb_dtype, emb_device,
        PrefixConfig(length=args.prefix_len, init_text=args.prefix_init_text)
    )
    if args.prefix_init_text:
        prefix.init_from_text(args.prefix_init_text, tok, base_model)
    return prefix

# ====================== LLM Wrapper ======================
class LLMHiddenExtractor(nn.Module):
    """Only returns averaged hidden states — keeps DataParallel gather small."""
    def __init__(self, llm, hidden_layers):
        super().__init__()
        self.llm = llm
        self.hidden_layers = hidden_layers

    def forward(self, inputs_embeds, attention_mask):
        out = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        h_list = [out.hidden_states[i] for i in self.hidden_layers]
        target_device = h_list[0].device
        h_list = [h.to(target_device) for h in h_list]
        return torch.stack(h_list, dim=0).mean(dim=0)

# ====================== 像素模式专用 loss（true batch forward） ======================
def compute_batch_loss_pixels(
    model, emb, pixel_emb, readout_head,
    prefix, intra_pos, batch, device,
    pixel_noise_prob=0.0,
    pixel_noise_range=0,
    emb_noise_std=0.0,
    apply_noise=False,
):
    """
    True batch forward for pixel mode (Lp fixed, T fixed).
    model: LLMHiddenExtractor (or DataParallel thereof), returns averaged hs directly.
    """
    prompt_lists = [s[0] for s in batch]
    pixel_lists  = [s[1] for s in batch]

    B = len(batch)
    if B == 0:
        return torch.zeros((), device=device)

    Lp = len(prompt_lists[0])
    T  = len(pixel_lists[0])

    prompt_ids = torch.tensor(prompt_lists, dtype=torch.long, device=device)  # [B, Lp]
    pix_t      = torch.tensor(pixel_lists,  dtype=torch.long, device=device)  # [B, T]
    pix_in_t   = pix_t

    if apply_noise and pixel_noise_prob > 0.0 and pixel_noise_range > 0:
        noise_mask = torch.rand(pix_t.shape, device=device) < pixel_noise_prob
        if noise_mask.any():
            delta = torch.randint(
                -pixel_noise_range,
                pixel_noise_range + 1,
                pix_t.shape,
                device=device,
            )
            pix_in_t = torch.where(
                noise_mask,
                torch.clamp(pix_t + delta, min=0, max=255),
                pix_t,
            )

    tok_embs = emb(prompt_ids).detach()   # [B, Lp, H]
    pix_embs = pixel_emb(pix_in_t)        # [B, T,  H]

    if apply_noise and emb_noise_std > 0.0:
        pix_embs = pix_embs + torch.randn_like(pix_embs) * emb_noise_std

    if intra_pos is not None:
        token_idx = torch.arange(T, device=device, dtype=torch.long)
        pos = intra_pos(token_idx).unsqueeze(0)       # [1, T, H]
        pix_embs = pix_embs + pos

    pix_embs = pix_embs.to(dtype=tok_embs.dtype)

    if prefix is None:
        inputs_embeds = torch.cat([tok_embs, pix_embs], dim=1)  # [B, Lp+T, H]
        P = 0
    else:
        tau = prefix()                                            # [P, H]
        P = int(tau.shape[0])
        tau_b = tau.unsqueeze(0).expand(B, -1, -1)               # [B, P, H]
        inputs_embeds = torch.cat([tau_b, tok_embs, pix_embs], dim=1)  # [B, P+Lp+T, H]

    attn_full = torch.ones((B, inputs_embeds.shape[1]), dtype=torch.long, device=device)

    hs = model(inputs_embeds=inputs_embeds, attention_mask=attn_full)  # [B, S, H]

    base = P + Lp
    pred_h = hs[:, base - 1 : base - 1 + T, :].float()          # [B, T, H]

    logits256 = readout_head(pred_h)                              # [B, T, 256]
    loss = F.cross_entropy(
        logits256.reshape(-1, 256),
        pix_t.reshape(-1),
        reduction="mean",
    )
    return loss

# ====================== Main ======================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model_id", type=str, default="/home/vipuser/Model/LLAMA_3.1_B")
    # ap.add_argument("-m", "--model_id", type=str, default="/home/vipuser/Model/Qwen3-8.0B/")

    # ap.add_argument("-m", "--model_id", type=str, default="/home/vipuser/Model/QWEN_2.5_0.5B")
    # ap.add_argument("-m", "--model_id", type=str, default="/home/vipuser/Model/QWEN_3_0.6B")

    ap.add_argument("--lora_dir", type=str, default="")
    ap.add_argument('--train_jsonl', '-i', type=str, required=True)
    ap.add_argument('--eval_jsonl', '-eval', type=str, required=True)
    ap.add_argument('--max_train_samples', '-n', type=int, default=0)

    ap.add_argument('--format_config', '-format', type=str, default='configs/default_format.yaml')

    # Prefix & INP
    ap.add_argument("--prefix_len", type=int, default=16)
    ap.add_argument("--prefix_init_text", type=str, default="")
    ap.add_argument("--no_prefix", action="store_true", default=False)
    ap.add_argument("--intra_pos", "-inp", action="store_true", default=True)
    ap.add_argument("--inp_lr", type=float, default=1e-4)
    ap.add_argument("--inp_patch_h", type=int, default=16,
                    help="INP 的 patch 高度")
    ap.add_argument("--inp_patch_w", type=int, default=16,
                    help="INP 的 patch 宽度")

    # Pixel & Head
    ap.add_argument("--pixel_mode", action="store_true", default=True)
    ap.add_argument("--pixel_emb_ckpt", type=str, default="models/pixel_emb_init.pt")
    ap.add_argument("--readout_head_ckpt", type=str, default="models/HeadPixel_init.pt")
    ap.add_argument("--init_pixel_emb_if_missing", action="store_true", default=True)
    ap.add_argument("--head_lr", type=float, default=5e-4)
    ap.add_argument("--pe_lr", type=float, default=5e-4)

    # ================== 新增：两阶段控制 ==================
    ap.add_argument("--stage", type=int, choices=[1, 2], default=1,
                    help="1: 从零联合训练 PE+Head+INP；2: 自动加载 Stage1 最佳模型并冻结，仅训练 Soft Prefix")

    ap.add_argument("--dataset", type=str, required=True,
                    help="数据集标识，用于生成独立 checkpoint")
    ap.add_argument("--from_scratch", action="store_true", default=False,
                help="从零初始化 PE/Head（不加载已有 ckpt）")
    ap.add_argument("--max_eval_samples", "-en", type=int, default=3000,
                help="验证时最多评估的样本数, 0=ALL")
    ap.add_argument(
        "--hidden_layer", nargs="+", type=int, default=[-1, -8],
        help="Hidden layers to average. Example: --hidden_layer -1 -16"
    )

    # 训练参数
    ap.add_argument("--epochs", "-e", type=int, default=4)
    ap.add_argument("--train_batch_size", "-bs", type=int, default=4)
    ap.add_argument("--eval_batch_size", "-ebs", type=int, default=4)
    ap.add_argument("--save_every_steps", type=int, default=2000,
                    help="每隔多少个全局 step 额外保存一次 checkpoint，0 表示关闭")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--warmup_ratio", type=float, default=0.03,
                    help="前 warmup_ratio 比例的训练步数做线性 warmup，0 表示关闭")
    ap.add_argument("--warmup_steps", type=int, default=0,
                    help="显式指定 warmup 步数；>0 时覆盖 warmup_ratio")
    ap.add_argument("--sampling_mode", type=str, choices=["full", "split"], default="full",
                    help="full: 每个 epoch 遍历全部训练数据；"
                         "split: 将数据均分到各 epoch，跨 epoch 不重复（总样本量不变）")

    # Noise
    ap.add_argument("--pixel_noise_prob", type=float, default=0.05,
                    help="训练阶段像素输入抖动概率，0 表示关闭")
    ap.add_argument("--pixel_noise_range", type=int, default=2,
                    help="训练阶段像素输入抖动范围 k，对应离散扰动 [-k, k]")
    ap.add_argument("--emb_noise_std", type=float, default=0.0,
                    help="训练阶段 pixel embedding 高斯噪声标准差，0 表示关闭")

    # Runtime
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--gpus", "-g", type=str, default="0")
    ap.add_argument("--multi_gpu", action="store_true", default=False,
                    help="使用 DataParallel 多卡并行")
    ap.add_argument("--gradient_checkpointing", action="store_true", default=False,
                    help="启用 gradient checkpointing 以节省显存；默认关闭以提升速度")
    ap.add_argument("--exp", type=str, default="test")
    ap.add_argument("--tag", "-t", type=str, default="train")
    ap.add_argument("--debug_mode", action="store_true", default=False)
    ap.add_argument("--resume", action="store_true", default=False,
                    help="从已有 Stage 1 best checkpoint 恢复训练，不从头初始化 PE/Head/INP/SP")
    ap.add_argument("--resume_final", action="store_true", default=False,
                    help="从 Stage 1 final checkpoint（非 best）恢复训练，使用 *_stage1.pt 文件")

    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    setup_logger(tag=f"train_pixel_{args.dataset}_{args.tag}_stage{args.stage}", out_dir=args.exp)

    # 1. LLM
    # bfloat16 加载 Qwen3-8B 约 16GB，单张 A100 40GB 完全够用
    # DataParallel 复制模型到每张卡各 16GB，batch 平分，吞吐翻倍
    llm_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tokenizer, model = utils.load_llm_model(args.model_id, args.lora_dir, args.device, dtype=llm_dtype)
    for p in model.parameters():
        p.requires_grad_(False)
    # gradient checkpointing: 节省显存，但会增加反向传播开销
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("[Runtime] gradient checkpointing enabled")
    else:
        print("[Runtime] gradient checkpointing disabled")
    model.train()

    emb = model.get_input_embeddings()
    H = model.config.hidden_size

    # 保留原始 LLM 引用，init_prefix 需要它
    raw_llm = model

    # Wrapper: forward 只返回 averaged hidden states，避免 DataParallel gather 全部 37 层 + logits
    model = LLMHiddenExtractor(model, args.hidden_layer)

    n_gpus = torch.cuda.device_count()
    if args.multi_gpu and n_gpus > 1:
        print(f"[Multi-GPU] DataParallel + gradient_checkpointing，bf16 模型复制到 {n_gpus} 张卡")
        model = nn.DataParallel(model)

    # 2. checkpoint 路径（带 stage 后缀）
    sp_ckpt   = f"models/SP_{args.dataset}.pt"
    inp_ckpt  = f"models/INP_{args.dataset}.pt"
    pe_ckpt   = f"models/pixel_emb_{args.dataset}.pt"
    head_ckpt = f"models/HeadPixel_{args.dataset}.pt"

    # resume 路径：--resume 用 best，--resume_final 用 final（最后一个 epoch 末尾权重）
    _use_final = args.resume_final
    _ckpt_sfx  = f"_stage1.pt" if _use_final else f"_stage1_best.pt"
    resume_pe_ckpt   = f"models/pixel_emb_{args.dataset}{_ckpt_sfx}"
    resume_head_ckpt = f"models/HeadPixel_{args.dataset}{_ckpt_sfx}"
    resume_inp_ckpt  = f"models/INP_{args.dataset}{_ckpt_sfx}"
    resume_sp_ckpt   = f"models/SP_{args.dataset}{_ckpt_sfx}"
    if _use_final:
        print(f"[Resume] 使用 final checkpoint（*_stage1.pt）")
    # resume 生效条件：--resume 或 --resume_final 任一为真
    _do_resume = args.resume or args.resume_final

    if args.debug_mode:
        print("[Debug mode] 激活")
    # ====================== 加载 / 初始化四个模块 ======================
    prefix = None if args.no_prefix else init_prefix(raw_llm, tokenizer, args, H)
    # resume: 将已保存的 SP 权重复制到新初始化的 prefix 中
    if _do_resume and args.stage == 1 and prefix is not None:
        if os.path.exists(resume_sp_ckpt):
            _sp_ckpt = torch.load(resume_sp_ckpt, map_location="cpu")
            _tau = _sp_ckpt["prefix"].to(device=prefix.prefix.device, dtype=prefix.prefix.dtype)
            with torch.no_grad():
                prefix.prefix.copy_(_tau)
            print(f"[Resume] Soft Prefix loaded from {resume_sp_ckpt}")
        else:
            print(f"[Resume] SP ckpt not found ({resume_sp_ckpt}), using random init")

    intra_pos = None
    if args.intra_pos:
        intra_pos_cfg = IntraPatchPosConfig(
            patch_h=args.inp_patch_h,
            patch_w=args.inp_patch_w,
            use_channel_emb=False,
            init_std=0.01,
            init_scale=1e-3,
        )
        intra_pos = IntraPatchPositionEmbedding(hidden_size=H, dtype=emb.weight.dtype,
                                                device=args.device, cfg=intra_pos_cfg)
        # resume: 从 Stage 1 checkpoint 加载 INP 权重
        if _do_resume and args.stage == 1 and os.path.exists(resume_inp_ckpt):
            intra_pos = load_intra_pos(resume_inp_ckpt, raw_llm, args.device)
            print(f"[Resume] INP loaded from {resume_inp_ckpt}")
        elif _do_resume and args.stage == 1:
            print(f"[Resume] INP ckpt not found ({resume_inp_ckpt}), using random init")

    # Pixel Embedding
    if args.from_scratch or (args.stage == 1 and not _do_resume):
        pixel_emb = PixelInputEmbedding(PixelInputEmbeddingConfig(hidden_size=H)).to(args.device)
        print(f"[Init] New Pixel Embedding for {args.dataset} (Stage {args.stage})")
    elif _do_resume and args.stage == 1:
        if os.path.exists(resume_pe_ckpt):
            pixel_emb = PixelInputEmbedding.load(resume_pe_ckpt, device=args.device)
            print(f"[Resume] Pixel Embedding loaded from {resume_pe_ckpt}")
        else:
            pixel_emb = PixelInputEmbedding(PixelInputEmbeddingConfig(hidden_size=H)).to(args.device)
            print(f"[Resume] PE ckpt not found ({resume_pe_ckpt}), using random init")
    elif os.path.exists(pe_ckpt):
        pixel_emb = PixelInputEmbedding.load(pe_ckpt, device=args.device)
        print(f"[Load] Pixel Embedding loaded from {pe_ckpt}")
    else:
        pixel_emb = PixelInputEmbedding(PixelInputEmbeddingConfig(hidden_size=H)).to(args.device)
        print(f"[Init] New Pixel Embedding for {args.dataset}")

    # Readout Head
    if args.from_scratch or (args.stage == 1 and not _do_resume):
        head_cfg = ReadoutConfig(hidden_size=H, num_classes=256, layer_norm=True, mid_size=2048)
        readout_head = NumericReadoutHead(head_cfg).to(args.device, dtype=torch.float32)
        print(f"[Init] New Readout Head for {args.dataset} (Stage {args.stage})")
    elif _do_resume and args.stage == 1:
        if os.path.exists(resume_head_ckpt):
            readout_head = NumericReadoutHead.load(resume_head_ckpt, device=args.device)
            print(f"[Resume] Readout Head loaded from {resume_head_ckpt}")
        else:
            head_cfg = ReadoutConfig(hidden_size=H, num_classes=256, layer_norm=True, mid_size=2048)
            readout_head = NumericReadoutHead(head_cfg).to(args.device, dtype=torch.float32)
            print(f"[Resume] Head ckpt not found ({resume_head_ckpt}), using random init")
    elif os.path.exists(head_ckpt):
        readout_head = NumericReadoutHead.load(head_ckpt, device=args.device)
        print(f"[Load] Readout Head loaded from {head_ckpt}")
    else:
        head_cfg = ReadoutConfig(hidden_size=H, num_classes=256, layer_norm=True, mid_size=2048)
        readout_head = NumericReadoutHead(head_cfg).to(args.device, dtype=torch.float32)
        print(f"[Init] New Readout Head for {args.dataset}")

    # ====================== Stage 2 自动加载 Stage 1 最佳模型 ======================
    if args.stage == 2:
        stage1_pe   = f"models/pixel_emb_{args.dataset}_stage1_best.pt"
        stage1_head = f"models/HeadPixel_{args.dataset}_stage1_best.pt"
        stage1_inp  = f"models/INP_{args.dataset}_stage1_best.pt"

        if os.path.exists(stage1_pe):
            pixel_emb = PixelInputEmbedding.load(stage1_pe, device=args.device)
            print(f"[Stage 2 Auto-Load] PE loaded from Stage 1 best: {stage1_pe}")
        else:
            raise FileNotFoundError(f"Stage 2 需要 Stage 1 的 PE checkpoint: {stage1_pe}")
        
        if os.path.exists(stage1_head):
            readout_head = NumericReadoutHead.load(stage1_head, device=args.device)
            print(f"[Stage 2 Auto-Load] Head loaded from Stage 1 best: {stage1_head}")
        else:
            raise FileNotFoundError(f"Stage 2 需要 Stage 1 的 Head checkpoint: {stage1_head}")
        
        if intra_pos is not None and os.path.exists(stage1_inp):
            intra_pos = load_intra_pos(stage1_inp, model, args.device)
            print(f"[Stage 2 Auto-Load] INP loaded from Stage 1 best: {stage1_inp}")

    # ====================== 两阶段冻结控制 ======================
    if args.stage == 2:
        for p in pixel_emb.parameters():
            p.requires_grad = False
        for p in readout_head.parameters():
            p.requires_grad = False
        if intra_pos is not None:
            for p in intra_pos.parameters():
                p.requires_grad = False
        print("[Stage 2] 已冻结 PE + Head + INP，仅训练 Soft Prefix")
    else:
        print("[Stage 1] 联合训练 PE + Head + INP（Soft Prefix 可选）")

    # 3. 数据集
    format_args = data_loader.load_data_format_config(args.format_config)
    train_dataset = PixelJsonlDataset(args.train_jsonl.split(','), tokenizer, format_args,
                                      with_full_img=False, with_neighbor_patches=False, shuffle=True)

    # 跨 epoch 不重复抽取：预先将全部样本索引均分到各 epoch
    epoch_subsets = None
    if args.sampling_mode == "split":
        N = len(train_dataset)
        all_idx = list(range(N))
        random.shuffle(all_idx)
        chunk = math.ceil(N / args.epochs)
        epoch_subsets = [all_idx[i * chunk: min((i + 1) * chunk, N)]
                         for i in range(args.epochs)]
        print(f"[Sampling] split 模式：{N} 样本均分为 {args.epochs} 份，"
              f"每 epoch {[len(s) for s in epoch_subsets]} 样本（跨 epoch 不重复）")
    else:
        print(f"[Sampling] full 模式：每 epoch 遍历全部 {len(train_dataset)} 样本")

    train_dataloader = DataLoader(train_dataset, num_workers=0, batch_size=args.train_batch_size,
                                  shuffle=True, collate_fn=lambda b: b)

    eval_dataset = PixelJsonlDataset(args.eval_jsonl.split(','), tokenizer, format_args,
                                     with_full_img=False, with_neighbor_patches=False, shuffle=False)
    eval_dataloader = DataLoader(eval_dataset, num_workers=0, batch_size=args.eval_batch_size,
                                 shuffle=False, collate_fn=lambda b: b)

    # 4. 优化器（根据 stage 动态构建）
    opt_params = []
    if prefix is not None:
        opt_params.append({
            "params": [prefix.prefix],
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "name": "prefix",
        })

    if args.stage == 1:   # Stage 1 才训练 INP / PE / Head
        if intra_pos is not None:
            opt_params.append({"params": intra_pos.parameters(), "lr": args.inp_lr, "name": "inp"})
        opt_params.append({"params": pixel_emb.parameters(), "lr": args.pe_lr, "name": "pixel_emb"})
        opt_params.append({"params": readout_head.parameters(), "lr": args.head_lr, "name": "head"})

    opt = torch.optim.AdamW(opt_params)
    base_lrs = [group["lr"] for group in opt.param_groups]

    if epoch_subsets is not None:
        total_train_steps = sum(
            _estimate_epoch_steps(len(subset_idx), args.train_batch_size, args.max_train_samples, args.debug_mode)
            for subset_idx in epoch_subsets
        )
    else:
        total_train_steps = args.epochs * _estimate_epoch_steps(
            len(train_dataset), args.train_batch_size, args.max_train_samples, args.debug_mode
        )
    warmup_steps = args.warmup_steps if args.warmup_steps > 0 else int(total_train_steps * args.warmup_ratio)
    warmup_steps = min(total_train_steps, warmup_steps)

    # 5. 训练循环
    best_bpsp = float("inf")
    best_epoch = 0
    global_step = 0
    print(f"Tag == : {args.tag}")

    print(f"model : {args.model_id}")
    print(f"hidden_layers : {args.hidden_layer}")
    print(f"pixel_noise_prob : {args.pixel_noise_prob}")
    print(f"pixel_noise_range : {args.pixel_noise_range}")
    print(f"emb_noise_std : {args.emb_noise_std}")
    print(f"total_train_steps : {total_train_steps}")
    print(f"warmup_steps : {warmup_steps}")
    print(f"save_every_steps : {args.save_every_steps}")
    print(f"Train set : {args.train_jsonl}")
    print(f"Eval set : {args.eval_jsonl}")

    _apply_lr_scale(opt, base_lrs, _warmup_scale(global_step, warmup_steps))

    for epoch in range(args.epochs):
        # split 模式：每 epoch 用不同子集；full 模式：用全部数据
        if epoch_subsets is not None:
            subset = Subset(train_dataset, epoch_subsets[epoch])
            cur_dl = DataLoader(subset, num_workers=0, batch_size=args.train_batch_size,
                                shuffle=True, collate_fn=lambda b: b)
        else:
            cur_dl = train_dataloader

        print(f"\n=== Epoch {epoch + 1} / {args.epochs} 开始 ===")
        epoch_loss = 0.0
        n_steps = 0
        epoch_samples = 0
        total_batches = len(cur_dl)

        if prefix:
            prefix.train()
        if intra_pos and args.stage == 1:
            intra_pos.train()

        epoch_t0 = time.time()
        for local_step, batch in enumerate(cur_dl):
            if args.max_train_samples > 0 and epoch_samples >= args.max_train_samples:
                break
            if args.debug_mode and local_step > DEBUG_SAMPLE:
                print("触发debug 模式，跳出")
                break

            step_t0 = time.time()
            _apply_lr_scale(opt, base_lrs, _warmup_scale(global_step, warmup_steps))
            total = compute_batch_loss_pixels(
                model=model, emb=emb, pixel_emb=pixel_emb,
                readout_head=readout_head, prefix=prefix,
                intra_pos=intra_pos, batch=batch,
                device=args.device,
                pixel_noise_prob=args.pixel_noise_prob,
                pixel_noise_range=args.pixel_noise_range,
                emb_noise_std=args.emb_noise_std,
                apply_noise=True,
            )

            opt.zero_grad(set_to_none=True)
            total.backward()

            to_clip = []
            if prefix is not None:
                to_clip.append(prefix.prefix)
            if intra_pos is not None:
                to_clip.extend([p for p in intra_pos.parameters() if p.grad is not None])
            if args.stage == 1:
                to_clip.extend([p for p in pixel_emb.parameters() if p.grad is not None])
                to_clip.extend([p for p in readout_head.parameters() if p.grad is not None])
            if to_clip:
                torch.nn.utils.clip_grad_norm_(to_clip, 1.0)

            opt.step()
            global_step += 1

            if args.save_every_steps > 0 and global_step % args.save_every_steps == 0:
                step_tag = f"_stage{args.stage}_step{global_step}"
                _save_stage_ckpt(
                    args=args,
                    dataset=args.dataset,
                    stage=args.stage,
                    tag=step_tag,
                    pixel_emb=pixel_emb,
                    readout_head=readout_head,
                    intra_pos=intra_pos,
                    prefix=prefix,
                )
                print(f"  [Saved] step checkpoint: *_{args.dataset}{step_tag}.pt")

            epoch_loss += total.item()
            n_steps += 1
            epoch_samples += len(batch)
            step_ms = (time.time() - step_t0) * 1000

            if local_step % 50 == 0 or local_step == total_batches - 1:
                elapsed_min = (time.time() - epoch_t0) / 60
                avg_so_far = epoch_loss / max(1, n_steps)
                avg_ms = (time.time() - epoch_t0) / n_steps * 1000
                print(f"  Epoch {epoch + 1} - Step {local_step + 1} / {total_batches} "
                      f"loss={total.item():.6f}  avg={avg_so_far:.6f}  samples={epoch_samples}  "
                      f"step={step_ms:.0f}ms  avg_step={avg_ms:.0f}ms  elapsed={elapsed_min:.1f}min  "
                      f"lr=[{_format_group_lrs(opt)}]")

            # 定期清理 CUDA 显存碎片，防止 step 时间逐 epoch 恶化
            if local_step % 200 == 0 and local_step > 0:
                torch.cuda.empty_cache()

        avg_train_loss = epoch_loss / max(1, n_steps)

        torch.cuda.empty_cache()

        # ---------- Eval on eval_dataset ----------
        with torch.no_grad():
            max_eval = args.max_eval_samples
            eval_n = 0
            total_nll = 0.0
            total_T = 0

            for eval_batch in eval_dataloader:
                loss = compute_batch_loss_pixels(
                    model=model, emb=emb, pixel_emb=pixel_emb,
                    readout_head=readout_head, prefix=prefix,
                    intra_pos=intra_pos, batch=eval_batch,
                    device=args.device,
                    pixel_noise_prob=0.0,
                    pixel_noise_range=0,
                    emb_noise_std=0.0,
                    apply_noise=False,
                )
                bsz = len(eval_batch)
                if bsz == 0:
                    continue
                T = len(eval_batch[0][1])
                total_nll += float(loss.item()) * (bsz * T)
                total_T += (bsz * T)
                eval_n += bsz
                if max_eval > 0 and eval_n >= max_eval:
                    break

            avg_nll = total_nll / max(1, total_T)
            current_bpsp = avg_nll / math.log(2.0)

        print(f"  [Eval] BPSP={current_bpsp:.6f} ({eval_n} samples)")

        if current_bpsp < best_bpsp:
            best_bpsp = current_bpsp
            best_epoch = epoch + 1
            print(f"  [New Best] BPSP = {best_bpsp:.6f} at epoch {best_epoch}")
            ep_tag = f"_stage1_best_{args.tag}"
            if args.stage == 1:
                _save_stage_ckpt(
                    args=args,
                    dataset=args.dataset,
                    stage=args.stage,
                    tag=ep_tag,
                    pixel_emb=pixel_emb,
                    readout_head=readout_head,
                    intra_pos=intra_pos,
                    prefix=prefix,
                )
            else:
                _save_stage_ckpt(
                    args=args,
                    dataset=args.dataset,
                    stage=args.stage,
                    tag=ep_tag,
                    pixel_emb=pixel_emb,
                    readout_head=readout_head,
                    intra_pos=intra_pos,
                    prefix=prefix,
                )

        # per-epoch checkpoint
        if args.stage == 1:
            ep_tag = f"_stage1_ep{epoch + 1}_{args.tag}"
            _save_stage_ckpt(
                args=args,
                dataset=args.dataset,
                stage=args.stage,
                tag=ep_tag,
                pixel_emb=pixel_emb,
                readout_head=readout_head,
                intra_pos=intra_pos,
                prefix=prefix,
            )
            print(f"  [Saved] per-epoch checkpoint: *_{args.dataset}{ep_tag}.pt")

        print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"=== Epoch {epoch + 1} 完成 ===")
        print(f"  Avg Train Loss : {avg_train_loss:.6f}")
        print(f"  Eval BPSP      : {current_bpsp:.6f}")
        print(f"  Best BPSP      : {best_bpsp:.6f} (epoch {best_epoch})")

    # 最终保存
    _save_stage_ckpt(
        args=args,
        dataset=args.dataset,
        stage=args.stage,
        tag=f"_stage{args.stage}",
        pixel_emb=pixel_emb,
        readout_head=readout_head,
        intra_pos=intra_pos,
        prefix=prefix,
    )

    print(f"Train Stage {args.stage} complete!")
    print(f"Train complete, model id : {args.model_id}")

    close_logger()

if __name__ == "__main__":
    main()
