# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Utility functions."""
import os
import random
import math
import chex
import torch

import numpy as np
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM


def set_seed(seed=42):
    # Python 内置 random 模块
    random.seed(seed)
    # NumPy
    np.random.seed(seed)
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)            # 为当前 GPU 设置种子
    torch.cuda.manual_seed_all(seed)        # 为所有 GPU 设置种子（如果使用多 GPU）
    # 确保 CUDA 的确定性操作（可能会降低性能）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # 注意：设为 False 以避免非确定性算法
    # 可选：设置 Python 哈希种子
    os.environ['PYTHONHASHSEED'] = str(seed)

def measure_peak_mb(fn, *args, **kwargs) -> float:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    fn(*args, **kwargs)

    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1024**2

def fmt_hms(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def find_completion_start(input_ids: List[int], template_ids: List[int]) -> int:
    if not template_ids:
        return 0
    tlen = len(template_ids)
    start = -1
    for j in range(0, len(input_ids) - tlen + 1):
        if input_ids[j : j + tlen] == template_ids:
            start = j + tlen
    if start < 0:
        raise RuntimeError("Cant find PromptTemplate token ids in `input_ids`")
    return start

# Assume input_ids: <bos>Blow are ... Completion: 34 245 3 ....45 <pad> <pad> ... <eos>
# return index of `45`
def find_completion_end(input_ids: List[int], pad_token_id) -> int:
    end =  input_ids.index(pad_token_id)
    if end < 0:
        raise RuntimeError("Cant find pad token id in `input_ids`")
    return end-1

def find_completion_start_end(input_ids: List[int], context_splitor: str, tokenizer: AutoTokenizer):
    cxt_ids = tokenize_like_project(context_splitor, tokenizer)
    start = find_completion_start(input_ids, cxt_ids)
    end = find_completion_end(input_ids, tokenizer.pad_token_id)
    return start, end

def find_pixels_ids(input_ids: List[int], context_splitor: str, tokenizer: AutoTokenizer):
    cxt_ids = tokenize_like_project(context_splitor, tokenizer)
    start = find_completion_start(input_ids, cxt_ids)
    end = find_completion_end(input_ids, tokenizer.pad_token_id)
    return input_ids[start:end+1]
    
def tokenize_like_project(text: str, tokenizer: AutoTokenizer):
    if not text:
        return []
    # use tokenizer.tokenize then filter out 'Ġ' tokens (to mimic the original generation)
    toks = tokenizer.tokenize(text, add_special_tokens=False)
    toks = [t for t in toks if t != "Ġ"]
    ids = tokenizer.convert_tokens_to_ids(toks)
    return ids

def bits_to_bytes(bits: str) -> tuple[bytes, int]:
  """Returns the bytes representation of bitstream and number of padded bits."""
  # Pad the string with zeros if the length is not a multiple of 8.
  padded_bits = bits.zfill((len(bits) + 7) // 8 * 8)
  num_padded_bits = len(padded_bits) - len(bits)

  # Split the string into 8-bit chunks.
  chunks = [padded_bits[i : i + 8] for i in range(0, len(padded_bits), 8)]

  # Convert each chunk to an integer and then to a byte.
  bytes_data = bytes([int(chunk, base=2) for chunk in chunks])

  return bytes_data, num_padded_bits


def bytes_to_bits(data: bytes, num_padded_bits: int = 0) -> str:
  """Returns the bitstream of bytes data accounting for padded bits."""
  return ''.join([bin(byte)[2:].zfill(8) for byte in data])[num_padded_bits:]


def right_shift_bytes_by_one(data: bytes) -> tuple[bytes, int]:
  """Returns right-shifted bytes, i.e., divided by 2, and the number of bytes.

  Our language models were trained on ASCII data. However, not all bytes can be
  decoded to ASCII, so we set the most significant bit (MSB) to 0, to ensure
  that we can decode the data to ASCII.

  However, for certain data types (e.g., images), masking the MSB and leaving
  the rest of the byte unchanged will destroy the structure of the data. Thus,
  we instead divide the number by two (i.e., we shift the bits to the right by
  one).

  Args:
    data: The bytes to be shifted.
  """
  return bytes([byte >> 1 for byte in data]), len(data)


def zero_most_significant_bit_if_not_ascii_decodable(
    data: bytes,
) -> tuple[bytes, int]:
  """Returns ascii-decodable data & the number of zeroed most significant bits.

  Our language models were trained on ASCII data. However, not all bytes can be
  decoded to ASCII, so we set the most significant bit (MSB) to 0, to ensure
  that we can decode the data to ASCII.

  Args:
    data: The bytes to be shifted.
  """
  masked_bits = 0
  masked_data = list()

  for byte in data:
    if chr(byte).isascii():
      masked_data.append(byte)
    else:
      masked_bits += 1
      masked_data.append(byte & 0x7F)

  return bytes(masked_data), masked_bits


def normalize_pdf_for_arithmetic_coding(pdf: chex.Array) -> chex.Array:
  """Normalizes the probabilities for arithmetic coding.

  Arithmetic coding converts the floating-point pdf to integers to avoid
  numerical issues. To that end, all pdf values need to be larger than the
  machine epsilon (to yield different integer values) and the sum of the pdf
  cannot exceed 1 (minus some precision tolerance).

  Args:
    pdf: The probabilities to be normalized.

  Returns:
    The normalized probabilities.
  """
  machine_epsilon = np.finfo(np.float32).eps
  # Normalize the probabilities to avoid floating-point errors.
  pdf = pdf / np.cumsum(pdf)[-1]
  # Ensure all probabilities are sufficiently large to yield distinct cdfs.
  pdf = (1 - 2 * pdf.shape[0] * machine_epsilon) * pdf + machine_epsilon
  return pdf



def _scan_indices_raster(h: int, w: int) -> List[int]:
    return [y * w + x for y in range(h) for x in range(w)]

def _scan_indices_zigzag(h: int, w: int) -> List[int]:
    out = []
    for y in range(h):
        row = [y * w + x for x in range(w)]
        if y % 2 == 1:
            row.reverse()
        out.extend(row)
    return out

def _part1by1(n: int) -> int:
    n &= 0x0000ffff
    n = (n | (n << 8)) & 0x00FF00FF
    n = (n | (n << 4)) & 0x0F0F0F0F
    n = (n | (n << 2)) & 0x33333333
    n = (n | (n << 1)) & 0x55555555
    return n

def _morton2D(x: int, y: int) -> int:
    return (_part1by1(y) << 1) | _part1by1(x)

def _scan_indices_morton(h: int, w: int) -> List[int]:
    coords = [(x, y) for y in range(h) for x in range(w)]
    coords.sort(key=lambda xy: _morton2D(xy[0], xy[1]))
    return [y * w + x for (x, y) in coords]

def _hilbert_rot(n: int, x: int, y: int, rx: int, ry: int):
    if ry == 0:
        if rx == 1:
            x = n - 1 - x
            y = n - 1 - y
        x, y = y, x
    return x, y

def _hilbert_d2xy(N: int, d: int):
    x = y = 0
    t = d
    s = 1
    while s < N:
        rx = 1 & (t // 2)
        ry = 1 & (t ^ rx)
        x, y = _hilbert_rot(s, x, y, rx, ry)
        x += s * rx
        y += s * ry
        t //= 4
        s *= 2
    return x, y

def _scan_indices_hilbert(h: int, w: int) -> List[int]:
    N = 1 << math.ceil(math.log2(max(h, w)))
    idxs = []
    for d in range(N * N):
        x, y = _hilbert_d2xy(N, d)
        if x < w and y < h:
            idxs.append(y * w + x)
    return idxs

_SCAN_FUNCS = {
    "raster": _scan_indices_raster,
    "zigzag": _scan_indices_zigzag,
    "morton": _scan_indices_morton,
    "hilbert": _scan_indices_hilbert,
}

def reorganize_flat_rgb_triplets(
    data: List[int],
    patch_shape: Tuple[int, int],
    scan_method: str = "raster",
    include_coords: bool = False,
    tokenized: bool = False,
    coords_tk_ids: List[int] = None
) -> List[int]:
    """
    把按逐行(raster)展平的 RGB 列表 data (长度=H*W*3) 重新组织为指定扫描顺序。
    若 include_coords=True，则每个像素后面追加 (row+1, col+1) 坐标。

    返回格式：
      [p1r, p1g, p1b, row1, col1, p2r, p2g, p2b, row2, col2, ...]
    """
    H, W = patch_shape
    N = H * W
    if len(data) != N * 3:
        raise ValueError(f"data length={len(data)} not equal to H*W*3={N*3}")
    if scan_method not in _SCAN_FUNCS:
        raise ValueError(f"Unknown scan method: {scan_method}")

    idxs = _SCAN_FUNCS[scan_method](H, W)

    # 逐行顺序直接返回
    if scan_method == "raster" and not include_coords:
        return list(data)

    out = []
    for i in idxs:
        base = 3 * i
        r, g, b = data[base], data[base + 1], data[base + 2]
        if include_coords:
            y, x = divmod(i, W)
            out.extend((r, g, b, coords_tk_ids[y], coords_tk_ids[x]))
        else:
            out.extend((r, g, b))
    return out

# ----- Data helpers -----
def build_numeric_maps(tokenizer) -> Tuple[List[int], Dict[int, int]]:
    pixel_token_ids: List[int] = []
    id2val: Dict[int, int] = {}
    for v in range(256):
        ids = tokenizer.encode(str(v), add_special_tokens=False)
        if not ids:
            raise RuntimeError(f"Tokenizer cant encode str({v})")
        tid = ids[-1]
        pixel_token_ids.append(tid)
        id2val[tid] = v
    return pixel_token_ids, id2val


def print_report(report):
    # print("\n===== Compression Report =====")
    # print(f"LLM Model:        {report['model_id']}")
    # print(f"Lora:             {report['lora_id']}")
    # print(f"Head:             {report['head_id']}")
    # print(f"Prefix:           {report['prefix_id']}")
    # print(f"AMPrefix:         {report['amortized_prefix_id']}")
    # print(f"IntraPos:         {report['intra_pos_id']}")
    # print(f"Eval data:        {report['eval_jsonl']}")
    # print("----------------------------------------------")
    # print(f"Samples (proc'd): {report['samples']}")
    # print(f"Total sub-pixels: {report['total_sub_pixels']}")
    # print(f"Total pixels:     {report['total_pixels']}")
    # print(f"Total bits:       {report['total_bits']}")
    # print(f"Pad bits:         {report['total_padded_bits']}")
    # print(f"Avg Numeric mass: {report['avg_numeric_mass']:.6f}")
    # print(f"BPSP:             {report['bpsp']:.6f}")
    # print(f"BPP (RGB pixel):  {report['bpp']:.6f}")
    d = report.copy()
    if not d:
        return
    # 计算最长的 key 的长度，用于对齐
    max_key_len = max(len(str(k)) for k in d.keys()) + 1
    
    # 遍历字典并打印
    for key, value in d.items():
        # 使用格式化字符串，让 key 左对齐，后面加上冒号和空格
        print(f"{str(key):<{max_key_len}}: {value}")
    print("==========================================\n")



def build_tokenid_to_pixel_lut(
    id2val: dict[int, int]
) -> torch.Tensor:
    max_id = max(id2val.keys())
    lut = torch.full((max_id + 1,), -1, dtype=torch.long)
    for tid, v in id2val.items():
        lut[tid] = int(v)
    return lut

def token_ids_to_pixel_symbols(
    pixel_tids: torch.Tensor,   # [B,T]
    lut: torch.Tensor,          # [max_id+1]
    *,
    validate: bool = True,
) -> torch.Tensor:
    if validate:
        if pixel_tids.max().item() >= lut.numel():
            raise RuntimeError("token_id exceeds LUT size")

    sym_t = lut[pixel_tids]
    if validate and (sym_t < 0).any():
        bad = pixel_tids[sym_t < 0].unique().tolist()
        raise RuntimeError(f"Invalid token id(s): {bad}")

    return sym_t


def load_llm_model(model_id: str, lora_dir: str, device: str, dtype: torch.dtype = torch.float32):
    MODEL_ROOT = os.path.expanduser("~") + "/models"
    model_path = os.path.join(MODEL_ROOT, model_id)

    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)

    model = base
    if lora_dir:
        model = PeftModel.from_pretrained(base, lora_dir)
        model = model.to(device)
    model.eval()
    return tok, model


def llm_forward_fast256(
    llm_model,                               # AutoModelForCausalLM 或类似
    pixel_vocab_token_ids_tensor,
    inputs_embeds: torch.Tensor,             # [B, P+L, H] (device 上)
    attn: torch.Tensor = None,               # [B, P+L] or None (device 上)
    rows: torch.Tensor = None,               # 可选：预先算好的 rows [T] long (device 上)
) -> torch.Tensor:
    # 使用 backbone，避开 llm_head 进而避免模型生成 [B,*,V] logits
    # 因为 V 很大，而我们最后只需要 256 个 pixel-val 对应的 tokenIds 对应的 logits
    # 因此我们可以先拿到最后一层的 hidden states
    # 再根据 rows 索引拿到 llm_head (W=HXV) 对应的 weights 和 bias（假如有的话），
    # 直接构造一个 新的 256 纬度的新的 `lightweight_llm_head` (W=HX256)
    backbone = llm_model.model
    out = backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=attn,
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
        )
    hs = out.last_hidden_state  # [B, P+L, H]
    h_rows = hs.index_select(1, rows)  # [B, T, H]

    W = llm_model.lm_head.weight.index_select(0, pixel_vocab_token_ids_tensor)  # [256, H]
    b = getattr(llm_model.lm_head, "bias", None)
    if b is not None:
        b256 = b.index_select(0, pixel_vocab_token_ids_tensor)  # [256]
    else:
        b256 = None

    # [B,T,H] x [H,256] -> [B,T,256]
    logits256 = torch.matmul(h_rows, W.t())
    if b256 is not None:
        logits256 = logits256 + b256.view(1, 1, -1)
    return logits256