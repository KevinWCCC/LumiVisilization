# torchrun --nproc_per_node=4 step6_compress_eval_prefix_readout_ddp.py --gpus 0,1,2,3 \
#   --llm_model_id Llama/Meta-Llama-3.1-8B \
#   --eval_jsonl data/large_data_valid_p16_half_full.jsonl \
#   --template_text ".Completion:" \
#   --readout_head /mnt/data/readout_head.pt \
#   --prefix_ckpt /mnt/data/prefix_only_ddp.pt \
#   --use_slow_lossless_compression \
#   --rank0_tqdm

import argparse
import json
import math
from typing import Union, List, Dict, Tuple, Iterator, Optional

import os
import sys
import datetime

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from readout_head import NumericReadoutHead
from soft_prefix import load_prefix

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

import arithmetic_coder
import utils
import constants

from logger import setup_logger, close_logger
from intra_patch_pos_embed import IntraPatchPositionEmbedding

# ----------------------------
# Model utils
# ----------------------------
def load_llm_llm_model(llm_model_id: str, lora_dir: str, device: str):
    MODEL_ROOT = os.path.expanduser("~") + "/llm_models"
    llm_model_path = os.path.join(MODEL_ROOT, llm_model_id)

    tok = AutoTokenizer.from_pretrained(llm_model_path, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    dtype = torch.float16 if (torch.cuda.is_available() and device.startswith('cuda')) else torch.float32
    base = AutoModelForCausalLM.from_pretrained(
        llm_model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)

    llm_model = base
    if lora_dir:
        llm_model = PeftModel.from_pretrained(base, lora_dir)
        llm_model = llm_model.to(device)
    llm_model.eval()
    return tok, llm_model


# 通用 step 前向：给定上下文 tokens + 可选 prefix + 可选 readout，返回 256 维 pdf
def step_probs(
    llm_model,
    tokenizer,
    pixel_vocab_token_ids_t: torch.Tensor,
    tau: Union[torch.Tensor, None],
    input_ids_1d: torch.Tensor,   # [L]
    readout_head: Union[NumericReadoutHead, None]
) -> torch.Tensor:
    """
    Compute 256-way probability distribution for the NEXT token given current context.
    Context tokens are in input_ids_1d. If tau is not None, we prepend τ in embedding space.
    Return: probs [256], on CPU-friendly type (float64 done later).
    """
    device = next(llm_model.parameters()).device
    emb = llm_model.get_input_embeddings()

    if tau is not None:
        # Build [τ, tokens] embeddings
        tok_embs = emb(input_ids_1d.unsqueeze(0))          # [1,L,H]
        tau_b = tau.unsqueeze(0)                           # [1,P,H]
        inputs_embeds = torch.cat([tau_b, tok_embs], dim=1)  # [1,P+L,H]
        attn = torch.ones(inputs_embeds.size()[:2], dtype=torch.long, device=device)
        with torch.no_grad():
            out = llm_model(inputs_embeds=inputs_embeds,
                        attention_mask=attn,
                        output_hidden_states=(readout_head is not None),
                        use_cache=False)
    else:
        # Standard token-only forward
        inp = input_ids_1d.unsqueeze(0)
        with torch.no_grad():
            out = llm_model(inp, output_hidden_states=(readout_head is not None), use_cache=False)

    if readout_head is not None:
        hs = out.hidden_states[-1]           # [1, L(+P), H]
        h_last = hs[:, -1, :].float()        # [1,H]
        logits256 = readout_head(h_last)     # [1,256]
        probs = torch.softmax(logits256, dim=-1).squeeze(0)   # [256]
    else:
        logits = out.logits                  # [1, L(+P), V]
        logits_last = logits[:, -1, :].float()
        per256 = logits_last.index_select(-1, pixel_vocab_token_ids_t)  # [1,256]
        probs = torch.softmax(per256, dim=-1).squeeze(0)                # [256]

    return probs


# ----------------------------
# Compression / Decompression
# ----------------------------
def llm_compress_sequence(
    input_ids: List[int],
    llm_model,
    tokenizer,
    pixel_vocab_token_ids: List[int],
    id2val: Dict[int, int],
    template_ids: List[int],
    use_slow_lossless_compression: bool = True,
    readout_head=None,
    tau: torch.Tensor = None,
    intra_pos: IntraPatchPositionEmbedding = None,
    prefix_len: int = 0,
    rt_numeric_mass = False,
    rt_est_bpp: bool = False,
) -> Tuple[bytes, int, int]:
    """
    Compress one sequence. If use_slow_lossless_compression=True, uses autoregressive
    step-by-step probabilities (lossless). Otherwise uses a single forward pass as
    fast approximation (not decode-compatible).
    """
    start = utils.find_completion_start(input_ids, template_ids)
    end = utils.find_completion_end(input_ids, tokenizer.pad_token_id)

    prefix_ids = input_ids[:start]
    input_pixel_token_ids = input_ids[start:end+1]

    device = next(llm_model.parameters()).device
    pixel_vocab_token_ids_t = torch.tensor(pixel_vocab_token_ids, dtype=torch.long, device=device)

    if use_slow_lossless_compression:
        # True lossless: step-by-step context
        context = prefix_ids[:]
        pdfs_list = []
        for tid in input_pixel_token_ids:
            ctx_tensor = torch.tensor(context, dtype=torch.long, device=device)
            probs = step_probs(
                llm_model=llm_model,
                tokenizer=tokenizer,
                pixel_vocab_token_ids_t=pixel_vocab_token_ids_t,
                tau=tau,
                input_ids_1d=ctx_tensor,
                readout_head=readout_head,
            )
            pdfs_list.append(probs)
            context.append(tid)
        pdfs = torch.stack(pdfs_list, dim=0)   # [T,256], T=patch_w * patch_h * 3
    else:
        # Fast: single forward over full sequence (rate estimation)
        full = torch.tensor(input_ids, dtype=torch.int64, device=device)
        emb = llm_model.get_input_embeddings()
        T = len(input_pixel_token_ids)

        # 拼接 pos ebed
        tok_embs = emb(full.unsqueeze(0))      # [1,L,H]; L=template prompt+T

        if intra_pos is not None:
            token_idx = torch.arange(T, device=tok_embs.device, dtype=torch.long)  # 0..T-1
            pos_intra = intra_pos(token_idx).unsqueeze(0)  # [1,T,H]
            tok_embs[:, start:end+1, :] = tok_embs[:, start:end+1, :] + pos_intra
        inputs_embeds = tok_embs
        attn = torch.ones(inputs_embeds.size()[:2], dtype=torch.long, device=device)
        
        if tau is not None:
            tau_b = tau.unsqueeze(0)                             # [1,P,H]
            inputs_embeds = torch.cat([tau_b, tok_embs], dim=1)  # [1,P+L,H]
            attn = torch.ones(inputs_embeds.size()[:2], dtype=torch.long, device=device)
        with torch.no_grad():
            out = llm_model(inputs_embeds=inputs_embeds,
                            attention_mask=attn,
                            output_hidden_states=(readout_head is not None),
                            use_cache=False)
            logits = out.logits[0]                              # [P+L,V]
            if readout_head is not None:
                hs = out.hidden_states[-1][0]                   # [P+L,H]

        # rows where we predict pixel tokens
        if tau is not None:
            rows = torch.arange(start + prefix_len - 1,
                                start + prefix_len - 1 + T,
                                device=device)                  # [T]
        else:
            rows = torch.arange(start - 1, start - 1 + T, device=device)  # [T]

        avg_numeric_mass = 0.0
        if rt_numeric_mass:
            logits_rows = logits.index_select(0, rows)               # [T,V]
            full_probs = torch.softmax(logits_rows.float(), dim=-1)  # [T,V]
            probs_num_full = full_probs.index_select(1, pixel_vocab_token_ids_t)  # [T,256]
            C_t = probs_num_full.sum(dim=-1)                          # [T]
            avg_numeric_mass = C_t.mean().item()

        if readout_head is not None:
            h_rows = hs.index_select(0, rows).float()           # [T,H]
            with torch.no_grad():
                logits256 = readout_head(h_rows)                # [T,256]
                pdfs = torch.softmax(logits256, dim=-1)         # [T,256]
        else:
            logits_256 = logits.index_select(0, rows).index_select(1, pixel_vocab_token_ids_t)  # [T,256]
            pdfs = torch.softmax(logits_256, dim=-1)

    # Arithmetic Encoding (encode w.r.t numeric symbol 0..255)
    pdfs = pdfs.detach().cpu()
    if rt_est_bpp:
        # pdfs: torch.Tensor [T,256] on device
        # map true token ids -> symbol indices 0..255
        symbols = [id2val[tid] for tid in input_pixel_token_ids]  # length T
        sym_t = torch.tensor(symbols, dtype=torch.long, device=pdfs.device)  # [T]

        # pick p(true_symbol) for each step
        p_true = pdfs.gather(1, sym_t.unsqueeze(1)).squeeze(1)  # [T]
        eps = 1e-12
        est_bits = (-torch.log2(torch.clamp(p_true, min=eps))).sum().item()

        # return format aligned with caller:
        # compressed_data, num_padded_bits, num_sub_pixels, numeric_mass
        # but here we set compressed_data empty and store est_bits in numeric_mass slot
        return b"", 0, len(input_pixel_token_ids), est_bits

    pdfs = pdfs.numpy().astype(np.float64)
    output_bits: List[str] = []
    encoder = arithmetic_coder.Encoder(
        base=constants.ARITHMETIC_CODER_BASE,
        precision=constants.ARITHMETIC_CODER_PRECISION,
        output_fn=output_bits.append,
    )
    for pdf, tid in zip(pdfs, input_pixel_token_ids):
        if tid not in id2val:
            raise RuntimeError(f"Invalid token id: {tid}")
        symbol = id2val[tid]
        encoder.encode(utils.normalize_pdf_for_arithmetic_coding(pdf), symbol)

    encoder.terminate()
    compressed_bits = ''.join(map(str, output_bits))
    compressed_data, num_padded_bits = utils.bits_to_bytes(compressed_bits)
    return compressed_data, num_padded_bits, len(input_pixel_token_ids), avg_numeric_mass


@torch.no_grad()
def llm_est_bits_batch_fast256(
    input_ids: torch.Tensor,                 # [B, L] long (device 上)
    llm_model,                                # AutoModelForCausalLM 或类似
    pixel_vocab_token_ids_tensor: torch.Tensor,  # [256] long (device 上)
    id2val_lut: torch.Tensor,                # [V] long (device 上)
    s: int,
    t: int,
    tau: torch.Tensor = None,                # [P, H] or None (device 上)
    intra_pos=None,                          # callable(token_idx)->[T,H] or module
    rows: torch.Tensor = None,               # 可选：预先算好的 rows [T] long (device 上)
) -> torch.Tensor:

    B, L = input_ids.shape

    # pixel token 段长度
    T = t - s + 1

    # embeddings
    emb = llm_model.get_input_embeddings()
    device = emb.weight.device
    input_ids = input_ids.to(device=device, dtype=torch.long)

    tok_embs = emb(input_ids)  # [B, L, H]

    # intra_pos 只加在 pixel token 段上
    if intra_pos is not None:
        token_idx = torch.arange(T, device=device, dtype=torch.long)  # [T]
        pos_intra = intra_pos(token_idx).unsqueeze(0)                 # [1, T, H]
        tok_embs[:, s:t+1, :] = tok_embs[:, s:t+1, :] + pos_intra

    inputs_embeds = tok_embs

    # prefix tau
    if tau is not None:
        tau = tau.to(device=device)
        tau_b = tau.unsqueeze(0).expand(B, -1, -1)                    # [B, P, H]
        inputs_embeds = torch.cat([tau_b, inputs_embeds], dim=1)      # [B, P+L, H]

    attn = torch.ones((B, inputs_embeds.size(1)), dtype=torch.long, device=device)

    # 用 llm_forward_fast256 计算 logits256
    logits256 = utils.llm_forward_fast256(
        llm_model=llm_model,
        pixel_vocab_token_ids_tensor=pixel_vocab_token_ids_tensor,
        inputs_embeds=inputs_embeds,
        attn=attn,
        rows=rows,
    )  # [B, T, 256]

    # 用 log_softmax 避免 pdfs；bits = -log2(p_true) = -(log p_true)/ln(2)
    logp = F.log_softmax(logits256, dim=-1)  # [B, T, 256]

    pixel_tids = input_ids[:, s:t+1]         # [B, T]
    sym_t = id2val_lut[pixel_tids]           # [B, T] 取值应为 0..255
    logp_true = logp.gather(dim=2, index=sym_t.unsqueeze(-1)).squeeze(-1)  # [B, T]

    est_bits = (-logp_true / math.log(2.0)).sum(dim=1)  # [B]
    return est_bits


def llm_decompress_sequence(
    llm_model,
    tokenizer,
    pixel_vocab_token_ids: List[int],
    prefix_context_ids: List[int],
    compressed_bytes: bytes,
    num_padded_bits: int,
    seq_len_values: int,
    readout_head=None,
    tau: torch.Tensor = None,
) -> List[int]:
    """
    Lossless decoding: always step-by-step, using the SAME pdf route as slow encoder
    (prefix/readout_modes included).
    """
    data_iter = iter(utils.bytes_to_bits(compressed_bytes, num_padded_bits=num_padded_bits))

    def _input_fn(bit_sequence: Iterator[str] = data_iter) -> Union[int, None]:
        try:
            return int(next(bit_sequence))
        except StopIteration:
            return None

    decoder = arithmetic_coder.Decoder(
        base=constants.ARITHMETIC_CODER_BASE,
        precision=constants.ARITHMETIC_CODER_PRECISION,
        input_fn=_input_fn,
    )

    device = next(llm_model.parameters()).device
    pixel_vocab_token_ids_t = torch.tensor(pixel_vocab_token_ids, dtype=torch.long, device=device)

    prefix_ids = prefix_context_ids[:]
    reconstructed_value_tids: List[int] = []

    for _ in range(seq_len_values):
        ctx_tensor = torch.tensor(prefix_ids, dtype=torch.long, device=device)
        probs = step_probs(
            llm_model=llm_model,
            tokenizer=tokenizer,
            pixel_vocab_token_ids_t=pixel_vocab_token_ids_t,
            tau=tau,
            input_ids_1d=ctx_tensor,
            readout_head=readout_head,
        )
        pdf = probs.detach().cpu().numpy().astype(np.float64)
        pdf = utils.normalize_pdf_for_arithmetic_coding(pdf)
        symbol = decoder.decode(pdf)
        tid = pixel_vocab_token_ids[symbol]
        reconstructed_value_tids.append(tid)
        prefix_ids.append(tid)

    return reconstructed_value_tids