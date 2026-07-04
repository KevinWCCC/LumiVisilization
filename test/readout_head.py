import os
import torch
import torch.nn as nn
from dataclasses import dataclass
import math

@dataclass
class ReadoutConfig:
    hidden_size: int
    num_classes: int = 256
    layer_norm: bool = True
    mid_size: int = 1024
    in_mult : int = 1

class NumericReadoutHead(nn.Module):
    """
    Lightweight readout head mapping LLM hidden states to a num_classes-way
    distribution over numeric pixel values (default 256: 0..255).

    结构：
      - 如果 cfg.mid_size <= 0:
          [可选 LayerNorm] → Linear(hidden_size → num_classes)
      - 如果 cfg.mid_size > 0:
          [可选 LayerNorm] → Linear(hidden_size → mid_size)
                           → GELU
                           → Linear(mid_size → num_classes)
    """

    def __init__(self, cfg: ReadoutConfig):
        super().__init__()
        self.cfg = cfg

        in_size = int(cfg.hidden_size) * int(getattr(cfg, "in_mult", 1))

        layers = []
        if cfg.layer_norm:
            layers.append(nn.LayerNorm(in_size))

        if cfg.mid_size is not None and cfg.mid_size > 0:
            layers.append(nn.Linear(in_size, cfg.mid_size, bias=True))
            layers.append(nn.GELU())
            layers.append(nn.Linear(cfg.mid_size, cfg.num_classes, bias=True))
        else:
            layers.append(nn.Linear(in_size, cfg.num_classes, bias=True))

        self.net = nn.Sequential(*layers)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        h: [..., hidden_size]
        returns logits: [..., num_classes]
        """
        return self.net(h)

    def save(self, path: str):
        ckpt = {
            "state_dict": self.state_dict(),
            "cfg": self.cfg.__dict__,
        }
        torch.save(ckpt, path)
        print(f"Save Heading at {path}")


    @staticmethod
    def load(path: str, device: str = "cpu") -> "NumericReadoutHead":
        ckpt = torch.load(path, map_location=device)
        cfg = ReadoutConfig(**ckpt["cfg"])
        model = NumericReadoutHead(cfg)
        model.load_state_dict(ckpt["state_dict"])
        model.to(device)
        return model

from dataclasses import dataclass
import torch
import torch.nn as nn

@dataclass
class PixelInputEmbeddingConfig:
    hidden_size: int
    layer_norm: bool = False
    mid_size: int = 4096   # 0 => 单层线性；>0 => 两层 MLP

class PixelInputEmbedding(nn.Module):
    """Map pixel values to LLM input embeddings.

    pixels: [...], values in [0,255] (uint8/long). returns [..., hidden_size]
    """
    def __init__(self, cfg: PixelInputEmbeddingConfig):
        super().__init__()
        self.cfg = cfg
        # 用线性层替代嵌入层

        # in_dim = 1
        # in_dim = 4
        in_dim = 7
        
        if cfg.mid_size is not None and cfg.mid_size > 0:
            self.net = nn.Sequential(
                nn.Linear(in_dim, cfg.mid_size, bias=True),
                nn.GELU(),
                nn.Linear(cfg.mid_size, cfg.hidden_size, bias=True),
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(in_dim, cfg.hidden_size, bias=True),
            )

        # init (std~0.02)
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
                torch.nn.init.zeros_(m.bias)
        if cfg.layer_norm:
            self.ln = nn.LayerNorm(cfg.hidden_size)
        else:
            self.ln = None
        if self.ln is not None:
            nn.init.ones_(self.ln.weight)
            nn.init.zeros_(self.ln.bias)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:

        input_pixel = pixel_values.to(torch.float32)
        normalized_value = input_pixel / 255.0
        square_value = normalized_value * normalized_value
        ang = 2.0 * math.pi * input_pixel / 255.0
        sin_value = torch.sin(ang)
        cos_value = torch.cos(ang)

        # 动态推导形状（支持 [B, T] 或 [T]）
        if input_pixel.dim() == 1:
            input_pixel = input_pixel.unsqueeze(0)          # [T] → [1, T]
            B, T = 1, input_pixel.shape[1]
        else:
            B, T = input_pixel.shape[:2]          # [B, T, ...] 或 [B, T]

        # 3)  [..., 1]
        # feats = torch.stack([normalized_value], dim = -1)

        # 3)  [...,4]
        # feats = torch.stack([normalized_value, square_value, sin_value, cos_value], dim = -1)

        # # 3) 0 = R, 1 = G, 2 = B
        ch = (torch.arange(T, device = input_pixel.device) % 3).view(1, T).expand(B, T)  # [B , T]
        ch_oh = torch.nn.functional.one_hot(ch, num_classes = 3).to(torch.float32)  # [B, T ,3]

        val_feats = torch.stack([normalized_value, square_value, sin_value, cos_value], dim = -1)  # [B, T, 4]
        feats = torch.cat([val_feats, ch_oh], dim = -1)  # [B, T, 7]
        
        # 1. 输入统一为float32，支持 [B, T] 或 [T]
        input_pixel = pixel_values.to(torch.float32)
        if input_pixel.dim() == 1:
            input_pixel = input_pixel.unsqueeze(0)  # [T] → [1, T]
        B, T = input_pixel.shape[ : 2]

        # 2. 基础归一化（min-max）
        normalized_value = input_pixel / 255.0                    # [B, T] ∈ [0, 1]

        # 3. 简单中性化（centering）——核心修改
        centered_value = normalized_value - 0.5                   # [B, T] ∈ [-0.5, 0.5]
        square_value = normalized_value * normalized_value        # 原平方（保留原分布）
        centered_square = square_value - (1.0 / 3.0)              # 理论均值1/3 → 中心化

        # 4. 三角特征（天然零均值，无需额外中性化）
        ang = 2.0 * math.pi * input_pixel / 255.0
        sin_value = torch.sin(ang)
        cos_value = torch.cos(ang)

        # 5. 通道one-hot（RGB索引 0 = R, 1 = G, 2 = B）
        ch = (torch.arange(T, device=input_pixel.device) % 3).view(1, T).expand(B, T)
        ch_oh = torch.nn.functional.one_hot(ch, num_classes=3).to(torch.float32)  # [B, T, 3]

        # 6. 拼接所有特征（中心化后）
        val_feats = torch.stack([
            centered_value,      # 中性化数值
            centered_square,     # 中性化平方
            sin_value,           # 周期
            cos_value
        ], dim = -1)  # [B, T, 4]

        feats = torch.cat([val_feats, ch_oh], dim=-1)  # [B, T, 7]

        out = self.net(feats)
        if self.ln is not None:
            out = self.ln(out)
        return out

    def save(self, path: str):
        """保存模型的状态字典和配置"""
        ckpt = {
            "state_dict": self.state_dict(),
            "cfg": self.cfg.__dict__,
        }
        torch.save(ckpt, path)
        print(f"Save PE at {path}")

    @staticmethod
    def load(path: str, device: str = "cpu") -> "PixelInputEmbedding":
        """加载保存的模型"""
        ckpt = torch.load(path, map_location=device)
        cfg = PixelInputEmbeddingConfig(**ckpt["cfg"])
        model = PixelInputEmbedding(cfg)
        model.load_state_dict(ckpt["state_dict"])
        model.to(device)
        return model