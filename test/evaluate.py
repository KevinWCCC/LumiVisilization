import os
import time
from typing import Dict, Any, Optional

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import math

from readout_head import NumericReadoutHead, PixelInputEmbedding, PixelInputEmbeddingConfig, ReadoutConfig
from soft_prefix import load_prefix
from amortized_prefix import load_amortized_prefix
from intra_patch_pos_embed import load_intra_pos


import utils
import compress
from logger import setup_logger, close_logger
import data_loader
from data_loader import TokenizedJsonlDataset, PixelJsonlDataset
from intra_patch_pos_embed import IntraPatchPositionEmbedding, IntraPatchPosConfig, load_intra_pos
HIDDEN_LAYERS = (-1, -8)

def _mean_selected_hidden(hidden_states, layers=HIDDEN_LAYERS):
    h_list = [hidden_states[i] for i in layers]
    return torch.stack(h_list, dim=0).mean(dim=0)

def load_intra_pos_any(path: str, model, device: str):
    ckpt = torch.load(path, map_location=device)
    # old format: {"cfg": ..., "state_dict": ...}
    return load_intra_pos(path, model, device)

    raise ValueError(f"Unknown intra_pos ckpt format: {path}. keys={list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)}")

def run_compression(
    model,
    tokenizer,
    data_iter,
    template_splitor,
    readout_head=None,
    prefix_tau=None,
    prefix_len: int = 0,
    max_samples=0,
    use_slow_lossless_compression: bool = False,
    verify_decode: bool = False,
    device: str = "cuda",
    rank: int = 0,
    world_size: int = 1,
    amortized_prefix=None,   # 样本自适应 prefix 网络
    intra_pos=None,          # 样本自适应 intra-patch pos embed
    rt_est_bpp: bool = False,  # 是否实时估计 bpp
    # --- tokenizer-free pixel input ---
    pixel_emb=None,          # PixelInputEmbedding or None
) -> Dict[str, Any]:

    # legacy tokenized mode需要这两项；pixel mode不需要，但保留不影响
    template_ids = utils.tokenize_like_project(template_splitor, tokenizer)
    pixel_token_ids, id2val = utils.build_numeric_maps(tokenizer)

    if amortized_prefix is not None:
        amortized_prefix.eval()
        emb_layer = model.get_input_embeddings()
    else:
        emb_layer = None

    total_est_bits = 0.0
    total_bits = 0
    total_padded_bits = 0
    total_symbols = 0
    total_pixels = 0
    processed = 0
    total_masses = []

    # 注意：这里不强制要求 data_iter 的结构一致，由 pixel_emb 是否为空决定 unpack 的格式
    for global_idx, sample in enumerate(data_iter):
        if max_samples > 0 and (global_idx >= max_samples):
            break
        if (global_idx % world_size) != rank:
            continue

        # --- unpack ---
        if pixel_emb is None:
            # legacy: (input_ids_list, nei_patches_tids, img_ids_list, meta)
            input_ids_list, nei_patches_tids, img_ids_list, meta = sample
        else:
            # pixel: (prompt_ids_list, pixel_values_list, nei_patches_tids, img_ids_list, meta)
            prompt_ids_list, pixel_values_list, nei_patches_tids, img_ids_list, meta = sample

        # --- 为本样本决定 tau 和 prefix_len ---
        tau_for_sample = prefix_tau
        effective_prefix_len = prefix_len

        if amortized_prefix is not None:
            if img_ids_list is None:
                raise ValueError("amortized_prefix is enabled but img_ids_list is None.")
            img_ids_tensor = torch.tensor(img_ids_list, dtype=torch.long, device=device)
            img_ids_b = img_ids_tensor.unsqueeze(0)                 # [1,L]
            img_attn_b = torch.ones_like(img_ids_b, device=device)  # [1,L]

            with torch.no_grad():
                tau_b = amortized_prefix(
                    input_ids=img_ids_b,
                    attention_mask=img_attn_b,
                    emb_layer=emb_layer,
                )                                                  # [1,P,H]
            tau_for_sample = tau_b[0]                               # [P,H]
            if hasattr(amortized_prefix, "length"):
                effective_prefix_len = amortized_prefix.length
            else:
                effective_prefix_len = tau_for_sample.size(0)

        # ---- compress ----
        if pixel_emb is None:
            compressed_bytes, num_padded_bits, num_sub_pixels, numeric_mass = compress.llm_compress_sequence(
                input_ids=input_ids_list,
                llm_model=model,
                tokenizer=tokenizer,
                readout_head=readout_head,
                use_slow_lossless_compression=use_slow_lossless_compression,
                tau=tau_for_sample,
                intra_pos=intra_pos,
                rt_est_bpp=rt_est_bpp,
            )
        else:
            compressed_bytes, num_padded_bits, num_sub_pixels, numeric_mass = compress.llm_compress_pixels(
                prefix_ids=prompt_ids_list,
                pixel_values=pixel_values_list,
                llm_model=model,
                readout_head=readout_head,
                pixel_emb=pixel_emb,
                use_slow_lossless_compression=use_slow_lossless_compression,
                tau=tau_for_sample,
                intra_pos=intra_pos,
                rt_est_bpp=rt_est_bpp,
            )

        total_symbols += int(num_sub_pixels)
        total_pixels += int(num_sub_pixels // 3)
        processed += 1

        if rt_est_bpp:
            total_est_bits += float(numeric_mass)
        else:
            total_bits += (len(compressed_bytes) * 8)
            total_padded_bits += int(num_padded_bits)
            total_masses.append(float(numeric_mass))

        # ---- 你要求的一模一样：decode verify ----
        if verify_decode and use_slow_lossless_compression:
            if pixel_emb is None:
                start = utils.find_completion_start(input_ids_list, template_ids)
                end = utils.find_completion_end(input_ids_list, tokenizer.pad_token_id)
                prefix_cxt_ids = input_ids_list[:start]
                true_value_tids = input_ids_list[start: end + 1]
                rec_value_tids = compress.llm_decompress_sequence(
                    llm_model=model,
                    tokenizer=tokenizer,
                    pixel_vocab_token_ids=pixel_token_ids,
                    prefix_context_ids=prefix_cxt_ids,
                    compressed_bytes=compressed_bytes,
                    num_padded_bits=num_padded_bits,
                    seq_len_values=len(true_value_tids),
                    readout_head=readout_head,
                    tau=tau_for_sample,
                    intra_pos=intra_pos,
                )
                assert true_value_tids == rec_value_tids
            else:
                rec_vals = compress.llm_decompress_pixels(
                    llm_model=model,
                    prefix_ids=prompt_ids_list,
                    compressed_bytes=compressed_bytes,
                    num_padded_bits=num_padded_bits,
                    seq_len_values=len(pixel_values_list),
                    readout_head=readout_head,
                    pixel_emb=pixel_emb,
                    tau=tau_for_sample,
                    intra_pos=intra_pos,
                )
                assert pixel_values_list == rec_vals

    # --- 统计 & Report ---
    avg_mass: float = float(sum(total_masses) / len(total_masses)) if total_masses else 0.0
    total_bits_final = float(total_est_bits) if rt_est_bpp else float(total_bits)

    bpsp = (total_bits_final / total_symbols) if total_symbols > 0 else 0.0
    bpp = (total_bits_final / max(1, total_pixels))

    report = {
        "samples": int(processed),
        "total_sub_pixels": int(total_symbols),
        "total_pixels": int(total_pixels),
        "total_bits": float(total_bits_final),
        "total_padded_bits": int(total_padded_bits),
        "avg_numeric_mass": float(avg_mass),
        "bpsp": float(bpsp),
        "bpp": float(bpp),
        "estimate": bool(rt_est_bpp),
        "pixel_mode": bool(pixel_emb is not None),
    }
    return report

@torch.no_grad()
def run_est_bpp_pixels_fast(
    model,
    tokenizer,
    data_iter,
    readout_head,
    pixel_emb,
    prefix_tau=None,         # [P,H] or None
    intra_pos=None,
    max_samples=0,
    device="cuda",
):
    """
    Pixel mode, fast BPP estimation using teacher forcing:
    One forward per sample: [tau + prompt_emb + pixel_emb(all)] -> hidden -> head -> logits256
    Then bits = -sum log2 p(true).
    """
    if readout_head is None:
        raise ValueError("fast pixel CE estimate requires readout_head (hidden->256).")

    emb = model.get_input_embeddings()
    model_dtype = emb.weight.dtype

    total_bits = 0.0
    total_symbols = 0  # sub-pixels
    processed = 0

    for idx, sample in enumerate(data_iter):
        if max_samples > 0 and processed >= max_samples:
            break

        # pixel: (prompt_ids_list, pixel_values_list, nei_patches_tids, img_ids_list, meta)
        prompt_ids_list, pixel_values_list, _, _, _ = sample
        if len(pixel_values_list) == 0:
            continue

        # tensors
        prompt_t = torch.tensor(prompt_ids_list, dtype=torch.long, device=device).unsqueeze(0)  # [1,Lp]
        pix_t = torch.tensor(pixel_values_list, dtype=torch.long, device=device).unsqueeze(0)  # [1,T]
        Lp = prompt_t.size(1)
        T = pix_t.size(1)

        # embeddings
        prompt_emb = emb(prompt_t).to(dtype=model_dtype)           # [1,Lp,H]
        pix_emb = pixel_emb(pix_t).to(dtype=model_dtype)           # [1,T,H]

        # add intra_pos to pixel embeddings (optional)
        if intra_pos is not None:
            token_idx = torch.arange(T, device=device, dtype=torch.long)  # [T]
            pos = intra_pos(token_idx).unsqueeze(0).to(dtype=model_dtype) # [1,T,H]
            pix_emb = pix_emb + pos

        # concat: [prompt, pixels]
        seq = torch.cat([prompt_emb, pix_emb], dim=1)              # [1,Lp+T,H]

        # prepend tau if provided
        if prefix_tau is not None:
            tau_b = prefix_tau.to(device=device, dtype=model_dtype).unsqueeze(0)  # [1,P,H]
            seq = torch.cat([tau_b, seq], dim=1)                                  # [1,P+Lp+T,H]
            prefix_len = tau_b.size(1)
        else:
            prefix_len = 0

        attn = torch.ones((1, seq.size(1)), dtype=torch.long, device=device)

        base = getattr(model, "model", None)
        out = model(inputs_embeds=seq, attention_mask=attn, output_hidden_states=True, use_cache=False)
        hs = _mean_selected_hidden(out.hidden_states)

        # rows that predict each pixel:
        # predict pix[i] using hidden at position (prefix_len + (Lp+i-1))
        rows = torch.arange(prefix_len + Lp - 1, prefix_len + Lp - 1 + T, device=device, dtype=torch.long)  # [T]
        h_rows = hs[:, rows, :].float().squeeze(0)  # [T,H] fp32 for head stability
        logits = readout_head(h_rows)

        # bits = -sum log2 p_true
        logp = F.log_softmax(logits, dim=-1)                          # [T,256] ln-prob
        tgt = pix_t.squeeze(0)                                        # [T]
        logp_true = logp.gather(1, tgt.unsqueeze(1)).squeeze(1)       # [T]
        nll_ln = (-logp_true).sum().item()                            # sum of -ln p
        bits = nll_ln / math.log(2.0)

        total_bits += float(bits)
        total_symbols += int(T)
        processed += 1

    total_pixels = total_symbols / 3.0
    bpsp = total_bits / max(1, total_symbols)
    bpp  = total_bits / max(1.0, total_pixels)

    return {
        "samples": int(processed),
        "total_sub_pixels": int(total_symbols),
        "total_pixels": float(total_pixels),
        "total_bits": float(total_bits),
        "bpsp": float(bpsp),
        "bpp": float(bpp),
        "estimate": True,
        "pixel_mode": True,
        "method": "ce_teacher_forcing_fast",
    }

@torch.no_grad()
def run_est_compression_batch(
    model,
    pixel_vocab_token_ids,
    eval_dataloader,
    readout_head=None,
    prefix_tau=None,
    intra_pos=None,
    s: int = 0,
    t: int = 0,
    max_samples: int = 0,
    id2val_lut: torch.Tensor = None,
    rank: int = 0,
    world_size: int = 1,
) -> Dict[str, Any]:

    device_model = next(model.parameters()).device

    pixel_vocab_token_ids_tensor = torch.as_tensor(
        pixel_vocab_token_ids, dtype=torch.long, device=device_model
    )
    if id2val_lut is None:
        raise ValueError("id2val_lut must be provided")
    id2val_lut = id2val_lut.to(device=device_model)

    total_bits = 0.0
    total_symbols = 0
    processed = 0

    t1 = time.time()

    for batch_idx, batch in enumerate(eval_dataloader):
        if (batch_idx % world_size) != rank:
            continue

        (input_ids, _), _, _, _ = batch
        B, L = input_ids.shape
        T = t - s + 1
        if not (0 <= s <= t < L):
            raise ValueError(f"invalid s,t for L={L}: s={s}, t={t}")

        input_ids = input_ids.to(device=device_model, dtype=torch.long)

        prefix_len = 0 if prefix_tau is None else prefix_tau.size(0)
        rows_start = (s + prefix_len - 1) if (prefix_len > 0) else (s - 1)
        rows_pixels = torch.arange(rows_start, rows_start + T, device=device_model, dtype=torch.long)  # [T]
        est_bits_b = compress.llm_est_bits_batch_fast256(
            input_ids=input_ids,
            llm_model=model,
            pixel_vocab_token_ids_tensor=pixel_vocab_token_ids_tensor,
            id2val_lut=id2val_lut,
            s=s,
            t=t,
            tau=prefix_tau,
            intra_pos=intra_pos,
            rows=rows_pixels
        )

        total_bits += float(est_bits_b.sum().item())

        num_sub_pixels = (t - s + 1)
        total_symbols += int(num_sub_pixels) * B
        processed += B

        if max_samples > 0 and processed >= max_samples:
            break

    bpsp = (total_bits / total_symbols) if total_symbols > 0 else 0.0
    total_pixels = total_symbols / 3.0
    bpp = (total_bits / max(1, total_pixels))

    t4 = time.time()
    print(f"wall time: {t4 - t1}")

    return {
        "samples": int(processed),
        "total_sub_pixels": int(total_symbols),
        "total_pixels": float(total_pixels),
        "total_bits": float(total_bits),
        "bpsp": float(bpsp),
        "bpp": float(bpp),
        "estimate": True
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Single-GPU compression eval")
    parser.add_argument("-m", "--model_id", type=str, default="/home/vipuser/Model/LLAMA_3.1_B")
    parser.add_argument("-lora", "--lora_dir", type=str, default="")
    parser.add_argument("-eval", "--eval_jsonl", type=str, default="data/kodak_p16_7-24.jsonl")
    parser.add_argument("-head", "--readout_head", type=str, default="")
    parser.add_argument("-prefix", "--prefix_ckpt", type=str, default="")
    parser.add_argument("-inp", "--intra_pos", type=str, default="")
    parser.add_argument('-amp', '--amortized_prefix', type=str, default='')
    parser.add_argument("-n", "--max_samples", type=int, default=0)
    parser.add_argument("-lossless", "--use_slow_lossless_compression", action="store_true", default=False)
    parser.add_argument("-v", "--verify_decode", action="store_true", default=False)

    # data format config
    parser.add_argument("-format", "--format_config", default='configs/default_format.yaml')

    # Runtime
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gpus", type=str, default="0")

    # Experiment Info
    parser.add_argument("--exp", type=str, default="test")
    parser.add_argument("-t", "--tag", type=str, default="test")
    parser.add_argument("--tqdm", action="store_true", default=False)
    parser.add_argument("--rt_est_bpp", "-est", action="store_true", default=False)
    parser.add_argument("--eval_batch_size", "-bs", type=int, default=4)

    # pixel mode
    parser.add_argument("--pixel_mode", action="store_true", default=False)
    parser.add_argument("--pixel_emb_ckpt", type=str, default="models/pixel_emb_init.pt")
    parser.add_argument("--init_pixel_emb_if_missing", action="store_true", default=False)

    args = parser.parse_args()

    # --- 设备设置 ---
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    if args.device.startswith("cuda") and torch.cuda.is_available() and args.gpus:
        args.device = f"cuda:{args.gpus.split(',')[0]}"
    device = args.device
    print(f"[main] Using device: {device}")

    # --- 日志 ---
    start_time = time.time()
    setup_logger(tag=f"eval_{args.tag}", out_dir=args.exp)
    print(args)

    # --- Data format config ---
    format_args = data_loader.load_data_format_config(args.format_config)
    print(format_args)

    # --- 加载模型 ---
    tokenizer, model = utils.load_llm_model(args.model_id, args.lora_dir, device)
    model = model.float()
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    # --- Readout Head ---
    readout_head = None
    if args.readout_head:
        readout_head = NumericReadoutHead.load(args.readout_head, device=device).to(dtype=torch.float32)
        readout_head = readout_head.to(device).eval()

    # --- Soft Prefix ---
    prefix_tau = None
    prefix_len = 0
    if args.prefix_ckpt:
        prefix_tau, prefix_len = load_prefix(args.prefix_ckpt, device=device, model=model)
        print(f"[main] Loaded prefix of length {prefix_len} from {args.prefix_ckpt}")

    # --- Amortized Prefix ---
    amortized_prefix = None
    if args.amortized_prefix:
        amortized_prefix = load_amortized_prefix(args.amortized_prefix, model=model, device=device)
        print(f"Loaded amortized prefix from {args.amortized_prefix}")

    # --- Intra Position Embedding ---
    intra_pos = None
    if args.intra_pos:
        intra_pos = load_intra_pos_any(args.intra_pos, model, device)
        print(f"Loaded IntraPosEmbedding from {args.intra_pos}")

    # 你要求的一模一样：(5.4) main/外层：构造 pixel_emb
    pixel_emb = None
    if args.pixel_mode:
        H = model.config.hidden_size
        pixel_emb = None
        if args.pixel_emb_ckpt and os.path.exists(args.pixel_emb_ckpt):
            pixel_emb = PixelInputEmbedding.load(args.pixel_emb_ckpt, device=device)
            pixel_emb = pixel_emb.to(device).eval()
            print(f"[main] Loaded pixel_emb from {args.pixel_emb_ckpt}")
        else:
            if not args.init_pixel_emb_if_missing:
                raise FileNotFoundError(
                    f"pixel_emb_ckpt not found: {args.pixel_emb_ckpt}. "
                    f"Pass --init_pixel_emb_if_missing to create one."
                )
            pixel_emb = PixelInputEmbedding(PixelInputEmbeddingConfig(hidden_size=H)).to(device).eval()
            os.makedirs(os.path.dirname(args.pixel_emb_ckpt) or ".", exist_ok=True)
            pixel_emb.save(args.pixel_emb_ckpt)
            print(f"[main] Init and saved pixel_emb to {args.pixel_emb_ckpt}")

        # 数据集用（你要求 iter(dataset)）
        dataset = PixelJsonlDataset(
            args.eval_jsonl.split(','),
            tokenizer,
            format_args,
            shuffle=False,
            with_full_img=False,
            with_neighbor_patches=False
        )
        data_iter = iter(dataset)
        if args.rt_est_bpp and (not args.use_slow_lossless_compression):
            report = run_est_bpp_pixels_fast(
                model=model,
                tokenizer=tokenizer,
                data_iter=data_iter,
                readout_head=readout_head,
                pixel_emb=pixel_emb,
                prefix_tau=prefix_tau,
                intra_pos=intra_pos,
                max_samples=args.max_samples,
                device=device,
            )
        else:
            report = run_compression(
                model=model,
                tokenizer=tokenizer,
                data_iter=data_iter,
                template_splitor=format_args.template_splitor,
                readout_head=readout_head,
                prefix_tau=prefix_tau,
                prefix_len=prefix_len,
                amortized_prefix=amortized_prefix,
                max_samples=args.max_samples,
                use_slow_lossless_compression=args.use_slow_lossless_compression,
                verify_decode=args.verify_decode,
                device=device,
                intra_pos=intra_pos,
                rt_est_bpp=args.rt_est_bpp,
                pixel_emb=pixel_emb
            )

    else:
        # legacy tokenized path（保持原本逻辑）
        pixel_token_ids, id2val = utils.build_numeric_maps(tokenizer)
        id2val_lut = utils.build_tokenid_to_pixel_lut(id2val).to(device)

        eval_dataset = TokenizedJsonlDataset(
            args.eval_jsonl.split(','),
            tokenizer, format_args,
            with_full_img=False,
            with_neighbor_patches=False,
            shuffle=False
        )
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            collate_fn=lambda b: data_loader.collate(b, tokenizer.pad_token_id)
        )
        eval_data_iter = data_loader.get_dataset_iterator(eval_dataset)
        data_sample = eval_dataset[0]
        template_ids = utils.tokenize_like_project(format_args.template_splitor, tokenizer)
        start = utils.find_completion_start(data_sample[0], template_ids)
        end = utils.find_completion_end(data_sample[0], tokenizer.pad_token_id)

        if args.rt_est_bpp and args.eval_batch_size > 1:
            report = run_est_compression_batch(
                model=model,
                pixel_vocab_token_ids=pixel_token_ids,
                eval_dataloader=eval_dataloader,
                readout_head=readout_head,
                prefix_tau=prefix_tau,
                intra_pos=intra_pos,
                s=start,
                t=end,
                max_samples=args.max_samples,
                id2val_lut=id2val_lut,
            )
        else:
            report = run_compression(
                model=model,
                tokenizer=tokenizer,
                data_iter=eval_data_iter,
                template_splitor=format_args.template_splitor,
                readout_head=readout_head,
                prefix_tau=prefix_tau,
                prefix_len=prefix_len,
                amortized_prefix=amortized_prefix,
                max_samples=args.max_samples,
                use_slow_lossless_compression=args.use_slow_lossless_compression,
                verify_decode=args.verify_decode,
                device=device,
                intra_pos=intra_pos,
                rt_est_bpp=args.rt_est_bpp,
                pixel_emb=None
            )

    exp_info = {}
    exp_info['model_id'] = args.model_id
    exp_info['lora_id'] = args.lora_dir
    exp_info['head_id'] = args.readout_head
    exp_info['prefix_id'] = args.prefix_ckpt
    exp_info['amortized_prefix_id'] = args.amortized_prefix
    exp_info['intra_pos_id'] = args.intra_pos
    exp_info['eval_jsonl'] = args.eval_jsonl
    exp_info['pixel_mode'] = bool(args.pixel_mode)

    utils.print_report(exp_info)
    utils.print_report(report)

    end_time = time.time()
    total_duration = end_time - start_time
    hours, remainder = divmod(total_duration, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Total evaluation time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    close_logger()