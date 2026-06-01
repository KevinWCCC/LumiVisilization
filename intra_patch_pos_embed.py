import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional
import os

@dataclass
class IntraPatchPosConfig:
    patch_h: int                 # e.g. 16
    patch_w: int                 # e.g. 16
    use_channel_emb: bool = False # 是否区分 R/G/B
    init_std: float = 0.01       # embedding 初始化 std
    init_scale: float = 1e-3     # gate β 的初始值
    use_layernorm: bool = False  # 是否对位置向量做 LayerNorm

class IntraPatchPositionEmbedding(nn.Module):
    """
    Intra-patch position embedding for rasterized RGB token sequence.

    Token order assumption:
        for each pixel (row-major):
            R -> G -> B

    For token index t:
        pixel_idx = t // 3
        channel   = t % 3
        row = pixel_idx // patch_w
        col = pixel_idx %  patch_w
    """

    def __init__(
        self,
        hidden_size: int,
        dtype: torch.dtype,
        device: torch.device,
        cfg: IntraPatchPosConfig,
    ):
        super().__init__()
        self.cfg = cfg
        self.hidden_size = hidden_size
        self.patch_h = cfg.patch_h
        self.patch_w = cfg.patch_w

        # --- Embeddings ---
        self.row_emb = nn.Embedding(cfg.patch_h, hidden_size)
        self.col_emb = nn.Embedding(cfg.patch_w, hidden_size)

        if cfg.use_channel_emb:
            self.ch_emb = nn.Embedding(3, hidden_size)
        else:
            self.ch_emb = None

        # --- Optional normalization ---
        if cfg.use_layernorm:
            self.pos_norm = nn.LayerNorm(hidden_size)
        else:
            self.pos_norm = nn.Identity()

        # --- Learnable gate (stability) ---
        # final pos = beta * (row + col + ch)
        self.beta = nn.Parameter(
            torch.tensor(cfg.init_scale, device=device, dtype=dtype)
        )

        # init
        nn.init.normal_(self.row_emb.weight, mean=0.0, std=cfg.init_std)
        nn.init.normal_(self.col_emb.weight, mean=0.0, std=cfg.init_std)
        if self.ch_emb is not None:
            nn.init.normal_(self.ch_emb.weight, mean=0.0, std=cfg.init_std)

        if cfg.use_layernorm:
            nn.init.ones_(self.pos_norm.weight)
            nn.init.zeros_(self.pos_norm.bias)

        self.to(device=device, dtype=dtype)

    def forward(self, token_idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_idx: LongTensor of shape [T] or [B, T]
                       token indices inside ONE patch
                       range: 0 ... patch_h * patch_w * 3 - 1

        Returns:
            pos_emb: Tensor of shape [T, H] or [B, T, H]
        """
        orig_shape = token_idx.shape
        t = token_idx.reshape(-1)  # [N]

        # --- decode token index ---
        pixel_idx = t // 3
        ch_idx = t % 3

        row_idx = pixel_idx // self.patch_w
        col_idx = pixel_idx % self.patch_w

        # safety clamp (avoid out-of-bound if malformed input)
        row_idx = row_idx.clamp(0, self.patch_h - 1)
        col_idx = col_idx.clamp(0, self.patch_w - 1)

        # --- lookup embeddings ---
        pos = self.row_emb(row_idx) + self.col_emb(col_idx)

        if self.ch_emb is not None:
            pos = pos + self.ch_emb(ch_idx)

        # optional normalization
        pos = self.pos_norm(pos)

        # gate scaling
        pos = pos * self.beta
        # 对比norm_lize 是否有收益
        return pos.reshape(*orig_shape, self.hidden_size)


def save_intra_pos(intra_pos: IntraPatchPositionEmbedding, savepath: str):
    os.makedirs(os.path.dirname(savepath) or ".", exist_ok=True)
    ckpt = {
        "state_dict": intra_pos.state_dict(),
        "cfg": intra_pos.cfg.__dict__,
        "hidden_size": intra_pos.hidden_size,
    }
    torch.save(ckpt, savepath)
    print(f"[IntraPos] Saved to {savepath}")

def load_intra_pos(savepath: str, model, device: str="cuda"):
    ckpt = torch.load(savepath, map_location="cpu")

    emb = model.get_input_embeddings()
    target_device = emb.weight.device
    target_dtype = emb.weight.dtype
    H = int(ckpt["hidden_size"])

    cfg = IntraPatchPosConfig(**ckpt["cfg"])
    intra_pos = IntraPatchPositionEmbedding(
        hidden_size=H,
        dtype=target_dtype,
        device=target_device,
        cfg=cfg
    )
    intra_pos.load_state_dict(ckpt["state_dict"])
    intra_pos.to(device=target_device, dtype=target_dtype).eval()
    return intra_pos