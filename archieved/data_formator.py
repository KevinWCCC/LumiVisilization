#!/usr/bin/env python3
# 2_tokenize_json_to_inputids.py

import argparse
import json
from transformers import AutoTokenizer
from tqdm import tqdm
import os
import utils
import string


def rgb_to_yuv_int(r: int, g: int, b: int):
    """
    整数可逆 RGB -> (Y, U', V').

    输入: R,G,B ∈ [0,255]
    定义:
        Y  = floor((R + 2G + B) / 4)
        U  = B - G           ∈ [-255,255]
        V  = R - G           ∈ [-255,255]
        U' = U + 255         ∈ [0,510]
        V' = V + 255         ∈ [0,510]

    返回: (Y, U', V') 都在 [0,511] 范围内（实际用到 0..510）
    """
    r = int(r)
    g = int(g)
    b = int(b)
    y = (r + 2 * g + b) >> 2
    u = b - g
    v = r - g
    u_p = u + 255
    v_p = v + 255
    return y, u_p, v_p

def rgb_lists_to_sequence_yuv(R, G, B):
    """
    将 R,G,B 三个列表做整数可逆变换，得到:
        [Y1, U1', V1', Y2, U2', V2', ...]
    其中 Y ∈ [0,255], U',V' ∈ [0,510] ⊂ [0,511]
    """
    assert len(R) == len(G) == len(B), "R/G/B length mismatch"
    seq = []
    for r, g, b in zip(R, G, B):
        y, u_p, v_p = rgb_to_yuv_int(r, g, b)
        seq.extend([y, u_p, v_p])
    return " ".join(map(str, seq))

def rgb_lists_to_sequence_rgb(R, G, B):
    """
    将 R, G, B 三个等长的一维列表拼成:
        [R1, G1, B1, R2, G2, B2, ...]
    """
    assert len(R) == len(G) == len(B), "R/G/B length mismatch"
    seq = []
    for r, g, b in zip(R, G, B):
        seq.extend([int(r), int(g), int(b)])
    return " ".join(map(str, seq))

def format_sample_for_channel_indep(rec: dict, channel: str):
    """
    returns textual string like:
    'Prompt: R channel of a flatten RGB image Completion: 12 34 56 ... <|end_of_text|>'
    """
    template = "Prompt: {} channel of a flatten RGB image Completion: "
    if channel == "R":
        seq = rec["R"]
    elif channel == "G":
        seq = rec["G"]
    elif channel == "B":
        seq = rec["B"]
    else:
        raise ValueError("channel must be R/G/B")
    pixel_text = " ".join(str(int(v)) for v in seq)
    text = template.format(channel) + pixel_text
    return text


def get_format_fields(s):
    """提取字符串中所有需要 format 的字段名"""
    # 创建一个解析器
    formatter = string.Formatter()
    # 解析字符串，返回一个元组列表，每个元组代表一部分
    # 我们关注的是 parse 结果中的 field_name
    field_names = [field_name for _, field_name, _, _ in formatter.parse(s) if field_name is not None]
    return field_names

def format_sample_for_channel_corr(rec: dict, args=None):
    R = rec["R"]
    G = rec["G"]
    B = rec["B"]

    # template text
    template = ""
    # pixels text
    pixels_text = rgb_lists_to_sequence_rgb(R, G, B)

    if args:
        args_dict = args.__dict__
        template = args.template
        fields = get_format_fields(template)
        format_map = {}
        for field in fields:
            if field == 'domain':
                format_map['domain']=args_dict['domain']
            if field == 'caption':
                format_map['caption']=rec[args_dict['with_caption']]
        template = template.format(**format_map)
        if args.color_space == "yuv":
            pixels_text = rgb_lists_to_sequence_yuv(R, G, B)
    text = template + pixels_text
    return text

def extract_pixel_values_rgb(rec: dict) -> list:
    """Return [R1,G1,B1,R2,G2,B2,...] with values in 0..255."""
    R = rec["R"]; G = rec["G"]; B = rec["B"]
    assert len(R) == len(G) == len(B), "R/G/B length mismatch"
    out = []
    for r, g, b in zip(R, G, B):
        out.extend([int(r), int(g), int(b)])
    return out

def build_prompt_only_ids(rec: dict, tokenizer, args=None) -> list:
    """Tokenize template text ONLY (WITHOUT pixel numbers)."""
    if tokenizer is None:
        raise ValueError("tokenizer is required for prompt tokenization")
    bos = getattr(tokenizer, "bos_token_id", None)

    template = ""
    if args:
        # 复用你原来的 template format 逻辑（domain/caption 等）
        args_dict = args.__dict__
        template = args.template
        fields = get_format_fields(template)
        format_map = {}
        for field in fields:
            if field == "domain":
                format_map["domain"] = args_dict.get("domain", "")
            if field == "caption":
                cap_key = args_dict.get("with_caption", None)
                if cap_key and isinstance(cap_key, str) and cap_key in rec:
                    format_map["caption"] = rec[cap_key]
        template = template.format(**format_map)

    prompt_ids = utils.tokenize_like_project(template, tokenizer)
    if bos is not None:
        prompt_ids.insert(0, bos)
    # 注意：这里不 append EOS，因为像素还要接在后面
    return prompt_ids

def read_rec(rec, tokenizer, args=None):
    bos = getattr(tokenizer, "bos_token_id", None) if tokenizer is not None else None
    eos = getattr(tokenizer, "eos_token_id", None) if tokenizer is not None else None
    final_input_ids = None
    if 'input_ids' in rec:
        final_input_ids = rec['input_ids']
    else:
        if 'text' in rec :
            text = rec['text']
        else:
            # channel corr
            text = format_sample_for_channel_corr(rec, args)
        # print(text[0:300])
        # input()
        input_ids = utils.tokenize_like_project(text, tokenizer)
        if bos is not None:
            input_ids.insert(0, bos)
        if eos is not None:
            input_ids.append(eos)
        final_input_ids = input_ids

    if args and (args.scan != 'raster' or args.include_coords):
        s, e = utils.find_completion_start_end(final_input_ids, '.Completion:', tokenizer)
        pixel_ids = final_input_ids[s: e+1]
        reorg_pids = utils.reorganize_flat_rgb_triplets(
            pixel_ids, (args.ph, args.pw), args.scan, args.include_coords, True, coords_tk_ids)
        if args.include_coords:
            final_input_ids = final_input_ids[:s] + reorg_pids
            final_input_ids.append(eos)
        else:
            final_input_ids[s:e+1] = reorg_pids
    return final_input_ids

def main():
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--input", default='data/step1data.jsonl')
    p.add_argument("-o", "--output",default='data/step2data.jsonl')
    p.add_argument("-c", "--color_space", default='rgb')
    p.add_argument('--template', default='')
    p.add_argument("--with_caption", action='store_true')
    p.add_argument("--tqdm", action="store_true")
    p.add_argument('--channel_indep',action='store_true')
    
    args = p.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    n_written = 0
    
    with open(args.input, "r", encoding="utf-8") as inf, \
         open(args.output, "w", encoding="utf-8") as outf :
        for line in (tqdm(inf, desc="tokenize") if args.tqdm else inf):
            rec = json.loads(line)
            # produce three records, R/G/B
            if args.channel_indep:
                for ch in ("R","G","B"):
                    text = format_sample_for_channel_indep(rec, ch)
                    outf.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                    n_written += 1
            else:
                text = format_sample_for_channel_corr(rec, args)
                outf.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                n_written += 1      
    print(f"Wrote {n_written} tokenized records to {args.output}")

if __name__ == "__main__":
    main()