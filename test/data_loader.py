import os
import random
import json
import urllib
import zipfile
import re 
from pathlib import Path
import yaml
import argparse

from typing import Iterator, Tuple, Optional, List, Dict, Any, Optional, Sequence, Union

import numpy as np
from PIL import Image
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import constants
import data_formator

DATASET_ROOT = os.path.expanduser("~") + '/Datasets'

DOMAINS = {
    'scid': 'screenshot',
    'div2k': 'natural',
    'kodak': 'natural',
    'bracs': 'medical',
    'mix': 'mix of above'
}


# region Utility Functions

def collate(batch, pad_id=0):
    """
    输入 batch: [(lst1, lst2, .., lstX, meta_dict), ...]
      - lst 可以是 List[int] / Tuple[int] / torch.Tensor / None / []
    输出: ((tensor1, mask1), (tensor2, mask2), ..., batch_meta)

    规则：
    1) 如果某一列里“任意一个样本”的 lst 是 [] 或 None，则这一列输出 (None, None)。
    2) 返回格式为 ( (lst1, mask1), (lst2, mask2), ..., meta )
    """
    if not batch:
        raise ValueError("Empty batch is not supported.")

    # 检查每个 sample 的长度一致
    expected_len = len(batch[0])
    for i, sample in enumerate(batch):
        if len(sample) != expected_len:
            raise ValueError(
                f"Inconsistent sample length at index {i}: "
                f"got {len(sample)}, expected {expected_len}."
            )

    # 转置：得到每一列的数据
    transposed = list(zip(*batch))
    xx_ids_lists = transposed[:-1]
    meta_list = transposed[-1]

    # 检查 meta 是否都是 dict
    for i, m in enumerate(meta_list):
        if not isinstance(m, dict):
            raise TypeError(f"meta at index {i} must be dict, got {type(m)}")

    # 处理 meta：按 key 聚合为 list
    batch_meta = {}
    keys = set().union(*(m.keys() for m in meta_list))
    for k in keys:
        batch_meta[k] = [m.get(k) for m in meta_list]

    outputs = []

    # 1/4) 处理每一列序列，性能优化
    for col_idx, ids_col in enumerate(xx_ids_lists):
        # 1) 若这一列里任意元素是 None 或 []，则整列返回 None
        if any(x is None or (isinstance(x, (list, tuple)) and len(x) == 0) for x in ids_col):
            outputs.append((None, None))
            continue

        # 支持 list/tuple/tensor；并做基本类型校验
        seq_tensors = []
        lengths = torch.empty(len(ids_col), dtype=torch.long)

        for i, x in enumerate(ids_col):
            if torch.is_tensor(x):
                t = x.to(dtype=torch.long).view(-1)
            elif isinstance(x, (list, tuple)):
                t = torch.as_tensor(x, dtype=torch.long)
            else:
                raise TypeError(
                    f"Column {col_idx} element {i} must be list/tuple/tensor/None, got {type(x)}"
                )

            if t.numel() == 0:
                # 按你的规则：出现 [] 等价于整列 None
                outputs.append((None, None))
                break

            seq_tensors.append(t)
            lengths[i] = t.numel()
        else:
            # pad_sequence 性能更好
            out = pad_sequence(seq_tensors, batch_first=True, padding_value=pad_id)
            maxlen = out.size(1)

            # 向量化生成 mask
            # mask: (B, L) where positions < length are 1
            mask = (torch.arange(maxlen).unsqueeze(0) < lengths.unsqueeze(1)).to(torch.long)

            outputs.append((out, mask))
            continue
        pass
    # 2) 返回 ((t1,m1),(t2,m2),...,meta)
    return (*outputs, batch_meta)


# endregion

# region Enwik Text Dataset
# ===========================================================
#                        Enwik Dataset
# ===========================================================

class Enwik8Dataset(Dataset):
    """Dataset for Enwik8 data."""

    def __init__(self, data_chunks) -> None:
        self.dataset = data_chunks

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> torch.Tensor:
        seq = self.dataset[idx]
        seq_ascii = np.frombuffer(seq, dtype=np.uint8)
        return torch.tensor(seq_ascii, dtype=torch.uint8)


def _get_enwik9_chunks(
        num_chunks: int = constants.NUM_CHUNKS,
        sequence_length: int = constants.CHUNK_SIZE_BYTES,
        chunk_start_idx: int = 0
) -> List[bytes]:
    enwik9_root = f'{DATASET_ROOT}/enwik9'
    if not os.path.exists(enwik9_root):
        # Downloading and extracting the dataset.
        urllib.request.urlretrieve(
            'https://mattmahoney.net/dc/enwik9.zip',
            'enwik9.zip',
        )
        with zipfile.ZipFile('data/enwik9.zip', 'r') as zip_ref:
            zip_ref.extract('data/enwik9')

    all_chunks = []
    with open(enwik9_root, 'rb') as file:
        for idx in range(chunk_start_idx + num_chunks):
            if idx < chunk_start_idx:
                continue
            all_chunks.append(file.read(sequence_length))
    return all_chunks


def get_enwik_dataloader(
        num_chunks: int = constants.NUM_CHUNKS,
        sequence_length=constants.CHUNK_SIZE_BYTES,
        batch_size: int = 32,
        shuffle: bool = False,
        num_workers: int = 0,
        pin_memory: bool = False
) -> DataLoader:
    enwik8_chunks = _get_enwik9_chunks(
        num_chunks=num_chunks,
        sequence_length=sequence_length
    )

    enwik8Dataset = get_enwik_dataset(num_chunks, sequence_length)
    enwik8DataLoader = DataLoader(enwik8Dataset, batch_size=batch_size, shuffle=shuffle)
    return enwik8DataLoader


def get_enwik_dataset(
        num_chunks: int = constants.NUM_CHUNKS,
        sequence_length=constants.CHUNK_SIZE_BYTES,
) -> DataLoader:
    enwik8_chunks = _get_enwik9_chunks(
        num_chunks=num_chunks,
        sequence_length=sequence_length
    )
    enwik8Dataset = Enwik8Dataset(enwik8_chunks)
    return enwik8Dataset


def get_enwik9_iterator(
        num_chunks: int = constants.NUM_CHUNKS,
        sequence_length: int = constants.CHUNK_SIZE_BYTES,
        chunk_start_idx: int = 0
) -> Iterator[bytes]:
    chunks = _get_enwik9_chunks(
        num_chunks=num_chunks,
        sequence_length=sequence_length,
        chunk_start_idx=chunk_start_idx
    )
    return iter(chunks)


# endregion

# region Image Patch Dataset
# ===========================================================
#                        Image Patch Dataset
# ===========================================================

def patch_to_jsonl_record(patch, meta={}):
    if isinstance(patch, torch.Tensor):
        patch = patch.detach().cpu().numpy()
    # Accept shape (H,W,3)
    flat = patch.reshape(-1, 3)
    R = flat[:, 0].astype(int).tolist()
    G = flat[:, 1].astype(int).tolist()
    B = flat[:, 2].astype(int).tolist()
    rec = {
              "R": R,
              "G": G,
              "B": B
          } | meta
    return rec


def patch_to_token_ids(patch, tok, format_args=None, meta={}):
    if isinstance(patch, torch.Tensor):
        patch = patch.detach().cpu().numpy()
    rec = patch_to_jsonl_record(patch, meta)
    tokenized_ids = data_formator.read_rec(rec, tok, format_args)
    return tokenized_ids


def imgpath_to_token_ids(path, tok, resize_shape=None, format_args=None, meta={}):
    img_np = _load_imgpath_np(path, resize_shape)
    return patch_to_token_ids(img_np, tok, format_args, meta)


def _load_imgpath_np(path: str, resize_shape=None) -> np.ndarray:
    with Image.open(path) as im:
        if resize_shape:
            im = im.resize(resize_shape)
        arr = np.asarray(im.convert("RGB"))
    return arr


def _load_image_metadata_from_json(img_path: str):
    """
    根据图片路径读取同名的 .json 文件。
    /path/001.png -> /path/001.json
    如果不存在，返回 {}。
    """
    base, _ = os.path.splitext(img_path)
    json_path = base + ".json"
    if not os.path.exists(json_path):
        return {}
    lines = []
    with open(json_path, "r", encoding="utf-8") as json_file:
        data_dict = json.load(json_file)
    return data_dict


class ImagePatchDataset(Dataset):
    def __init__(
            self,
            images_path: str,
            patch_shape: Tuple[int, int],
            num_images: Optional[int] = None,
            exts: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".webp"),
            fetch_format: str = 'tensor',  # tensor, ids, jsonl
            with_meta: bool = False,
            with_neighbor_patches=False,
            with_full_image: bool = False,
            full_image_shape: Tuple[int, int] = (0, 0),
            tokenizer=None,  # If true, will return token ids instead of tensor
            format_args=None
    ):
        root = images_path

        self.root = root
        self.h, self.w = patch_shape
        self.exts = tuple(x.lower() for x in exts)
        self.fetch_format = fetch_format

        self.with_meta = with_meta
        self.with_full_image = with_full_image
        self.full_image_shape = full_image_shape
        if full_image_shape[0] == 0:
            self.full_image_shape = patch_shape
        self.with_neighbor_patches = with_neighbor_patches

        self.tokenizer = tokenizer
        self.format_args = format_args

        # 1) iterate all image files
        files = []
        for dirpath, _, filenames in os.walk(self.root):
            for fname in sorted(filenames):
                if fname.lower().endswith(self.exts):
                    files.append(os.path.join(dirpath, fname))

        if not files:
            raise FileNotFoundError(f"No images found under {self.root} with extensions {self.exts}")

        # 2) Construct the patches index: (file, row, col)
        self.file_metas = {}
        self.index: List[Tuple[str, int, int]] = []
        if num_images is not None and num_images > 0:
            files = files[:num_images]
        for i, fp in enumerate(files):
            img_file_meta = _load_image_metadata_from_json(fp)
            img_file_meta['path'] = fp
            img_file_meta['img_no'] = i
            self.file_metas[fp] = img_file_meta
            try:
                with Image.open(fp) as im:
                    w0, h0 = im.size  # PIL: (width, height)
                rows = (h0 // self.h)
                cols = (w0 // self.w)
                if rows > 0 and cols > 0:
                    for r in range(rows):
                        for c in range(cols):
                            self.index.append((fp, r, c))
            except Exception:
                print("Errors occur when process file: " + fp)
                continue
        if not self.index:
            raise RuntimeError("No valid patches can be extracted with given patch_shape from images in the folder.")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        fp, r, c = self.index[idx]
        meta = self.file_metas[fp]
        # (H, W, 3) uint8
        rgb = _load_imgpath_np(fp)
        patch = rgb[r * self.h: (r + 1) * self.h, c * self.w: (c + 1) * self.w, :]
        patch_tensor = torch.tensor(patch, dtype=torch.uint8)
        rt_results = [patch_tensor]
        if self.with_full_image:
            rgb_resized = _load_imgpath_np(fp, self.full_image_shape)
            img_tensor = torch.tensor(rgb_resized, dtype=torch.uint8)
            rt_results.append(img_tensor)
        if self.with_neighbor_patches:
            # TODO: get up / left  patch tensors
            pass

        if self.fetch_format == 'ids':
            rt_ids = []
            for t in rt_results:
                ids = patch_to_token_ids(t, self.tokenizer, self.format_args)
                rt_ids.append(ids)
            rt_results = rt_ids
        elif self.fetch_format == 'json':
            rt_jsons = []
            for t in rt_results:
                jsonls = patch_to_jsonl_record(t)
                rt_jsons.append(jsonls)
            rt_results = rt_jsons

        if self.with_meta:
            rt_results.append(meta)
        return tuple(rt_results)


# endregion


def load_data_format_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        configs = yaml.safe_load(f)
        return argparse.Namespace(**configs)


def get_formatted_jsonl_data_iterator(jsonl_data_paths, tokenizer, format_args):
    if isinstance(jsonl_data_paths, str):
        jsonl_data_paths = jsonl_data_paths.split(",")
    dataset = TokenizedJsonlDataset(jsonl_data_paths, tokenizer, format_args, shuffle=True)
    for data_item in dataset:
        yield data_item


def get_dataset_iterator(dataset):
    for data_item in dataset:
        yield data_item


class TokenizedJsonlDataset(Dataset):
    def __init__(self, jsonl_paths, tokenizer, format_args,
                 shuffle=True,
                 full_image_shape=(32, 32),
                 with_full_img=True,
                 with_neighbor_patches=False):
        if isinstance(jsonl_paths, str):
            paths = [jsonl_paths]
        else:
            paths = jsonl_paths
        self.samples = []
        self.full_img_tokenized_ids = {}
        self.full_image_shape = full_image_shape
        self.tokenizer = tokenizer
        self.format_args = format_args
        self.with_full_img = with_full_img
        self.with_neighbor_patches = with_neighbor_patches

        for path in paths:
            print(path)
            domain = next(
                DOMAINS[key] for key in ['scid', 'kodak', 'div2k', 'bracs', 'merge', 'div'] if key in path
            )
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    meta = {k: v for k, v in rec.items() if k not in ["R", "G", "B"]}
                    meta['domain'] = domain
                    self.samples.append((rec, meta))

                    if self.with_full_img:
                        imgpath = meta.get('path', '')
                        if imgpath and imgpath not in self.full_img_tokenized_ids:
                            full_img_ids = imgpath_to_token_ids(imgpath, tokenizer, self.full_image_shape, format_args,
                                                                meta)
                            self.full_img_tokenized_ids[imgpath] = full_img_ids
                    if self.with_neighbor_patches:
                        # TODO: 处理 neighbor patches
                        pass
        if shuffle:
            random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rec, meta = self.samples[idx]
        standard_meta = {
            "domain": "",
            "path": "",
        }
        standard_meta = {k: meta.get(k, v) for k, v in standard_meta.items()}
        self.format_args.domain = standard_meta['domain']

        patch_tids = data_formator.read_rec(rec, self.tokenizer, self.format_args)
        img_tids = None
        if self.with_full_img and standard_meta['path']:
            img_tids = self.full_img_tokenized_ids[standard_meta['path']]

        nei_patches_tids = None
        if self.with_neighbor_patches:
            # TODO
            pass
        return patch_tids, nei_patches_tids, img_tids, standard_meta

class PixelJsonlDataset(Dataset):
    """Return (prompt_ids, pixel_values, nei_patches, img_ids, meta) for tokenizer-free pixels."""
    def __init__(self, jsonl_paths, tokenizer, format_args,
                 shuffle=True, full_image_shape=(32,32), with_full_img=True, with_neighbor_patches=False):
        if isinstance(jsonl_paths, str):
            paths = [jsonl_paths]
        else:
            paths = jsonl_paths

        # path = Path(jsonl_path).expanduser().resolve()
        # current_dir = Path(os.getcwd())
        # base_name = path.stem

        # fixed_path = path.with_name(base_name + "_fixed" + path.suffix)      # 修复文件仍放在原目录
        # log_path   = current_dir / f"{base_name}_load_log.txt"              # 日志

        self.samples = []
        self.full_img_tokenized_ids = {}
        self.full_image_shape = full_image_shape
        self.tokenizer = tokenizer
        self.format_args = format_args
        self.with_full_img = with_full_img
        self.with_neighbor_patches = with_neighbor_patches

        fixed_lines = []
        log_lines = []
        line_num = 0
        fixed_count = 0
        error_count = 0

        # log_lines.append(f"=== JSONL 加载日志 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        # log_lines.append(f"文件: {path}\n")
        # log_lines.append(f"大小: {path.stat().st_size / (1024*1024):.2f} MB\n\n")

        for path in paths:
            domain = next(DOMAINS[k] for k in ['scid','kodak','div2k','bracs','mix', 'merge'] if k in path)

            line_num = 0
            error_count = 0
            

            with open(path, "r", encoding="utf-8") as f:
                for line in f:

                    line_num += 1
                    original_line = line

                    line = line.strip()
                    if not line:
                        continue
                    # rec = json.loads(line)
                    # === 增强调试信息 ===
                    try:
                        # 先尝试直接解析
                        rec = json.loads(line)
                    except json.JSONDecodeError as e:

                        # log_entry = self._make_log_entry(path.name, line_num, e, original_line)
                        # log_lines.append(log_entry)
                        # print(log_entry)   # 同时打印到控制台

                        print(f"\n{'='*80}")
                        print(f"❌ JSON 解析失败！ File: {path}")
                        print(f"   Line {line_num} (char {e.pos})")
                        print(f"   Error: {e}")
                        print(f"   {'-'*60}")
                        
                        # 显示问题位置上下文
                        snippet_start = max(0, e.pos - 120)
                        snippet_end = min(len(original_line), e.pos + 120)
                        print("   错误位置上下文:")
                        print(repr(original_line[snippet_start:snippet_end]))
                        print("   " + " " * (e.pos - snippet_start) + "^" + " <-- 这里")
                        
                        # 自动尝试修复并显示修复结果
                        cleaned = self._auto_fix_json_line(original_line)
                        print(f"   尝试自动修复后:")
                        print(repr(cleaned[:500] + "..." if len(cleaned)>500 else cleaned))
                        
                        try:
                            rec = json.loads(cleaned)
                            print(f"   ✅ 自动修复成功！(Line {line_num})")
                            log_lines.append(f"   ✅ 自动修复成功 (Line {line_num})\n")
                        except Exception as fix_e:
                            print(f"   ❌ 自动修复仍失败: {fix_e}")
                            error_count += 1
                            continue  # 跳过此行
                        
                        print(f"{'='*80}\n")
                    
                    meta = {k:v for k,v in rec.items() if k not in ["R","G","B"]}
                    meta["domain"] = domain
                    self.samples.append((rec, meta))

                    # with open(log_path, "w", encoding="utf-8") as f:
                    #     f.writelines(log_lines)
                    # print(f"📋 已生成详细日志 → {log_path}\n")

                    if self.with_full_img:
                        imgpath = meta.get("path","")
                        if imgpath and imgpath not in self.full_img_tokenized_ids:
                            full_img_ids = imgpath_to_token_ids(imgpath, tokenizer, self.full_image_shape, format_args, meta)
                            self.full_img_tokenized_ids[imgpath] = full_img_ids

        if shuffle:
            random.shuffle(self.samples)

    def _auto_fix_json_line(self, line: str) -> str:
        """自动修复常见 JSONL 问题"""
        # 1. 清理 CRLF / CR
        line = line.replace("\r\n", "\n").replace("\r", "\n").strip()
        
        # 2. 移除 trailing comma
        line = re.sub(r",\s*([\]}])", r"\1", line)
        
        # 3. 修复对象字段间缺少逗号（最常见错误）
        line = re.sub(r'([\]}])\s*"(\w+)"\s*:', r'\1, "\2":', line)
        
        # 4. 修复路径中的单反斜杠
        line = re.sub(r'\\(?!\\)', r'\\\\', line)
        
        # 5. 移除可能的 BOM
        line = line.lstrip('\ufeff')
        
        return line

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rec, meta = self.samples[idx]
        standard_meta = {"domain": meta.get("domain",""), "path": meta.get("path","")}
        self.format_args.domain = standard_meta["domain"]

        prompt_ids = data_formator.build_prompt_only_ids(rec, self.tokenizer, self.format_args)

        if getattr(self.format_args, "color_space", "rgb") != "rgb":
            raise ValueError("PixelJsonlDataset currently supports color_space='rgb' only")
        pixel_values = data_formator.extract_pixel_values_rgb(rec)

        img_tids = None
        if self.with_full_img and standard_meta["path"]:
            img_tids = self.full_img_tokenized_ids[standard_meta["path"]]

        nei = None
        return prompt_ids, pixel_values, nei, img_tids, standard_meta

def get_formatted_jsonl_data_loader(
        jsonl_data_paths,
        tokenizer,
        format_args,
        batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=None,
):
    if isinstance(jsonl_data_paths, str):
        jsonl_data_paths = jsonl_data_paths.split(",")
    dataset = TokenizedJsonlDataset(jsonl_data_paths, tokenizer, format_args)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

