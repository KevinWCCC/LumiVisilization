import os
import torch
import torch.nn as nn
from dataclasses import dataclass, asdict
from typing import Optional

@dataclass
class AmortizedPrefixConfig:
    length: int = 16
    hidden_dim: int = 512
    use_layernorm: bool = True

class AmortizedPrefix(nn.Module):
    """
    输入: 
        - 输入序列的 token ids (或它们的 embeddings)
        - attention_mask 用来做加权平均
    输出:
        - 每个样本自己的 soft prefix: [B, P, H]

    设计: 
      sequence_rep = 平均池化(non-pad token embeddings)
      tau_b = MLP(sequence_rep) -> reshape 到 [B, P, H]
    """
    def __init__(
        self, 
        hidden_size: int,          # LLM hidden size H
        dtype: torch.dtype,
        device: torch.device,
        cfg: Optional[AmortizedPrefixConfig] = None,
    ):
        super().__init__()
        if cfg is None:
            cfg = AmortizedPrefixConfig()
        self.cfg = cfg
        self.length = cfg.length
        self.hidden_size = hidden_size
        
        # 一个小 MLP，把 [H] 映射到 [P*H]
        layers = [
            nn.Linear(hidden_size, cfg.hidden_dim, bias=True),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.length * hidden_size, bias=True),
        ]
        if cfg.use_layernorm:
            # 最后对 prefix 做一下 LN，稳定一点
            self.ln = nn.LayerNorm(hidden_size).to(device=device, dtype=dtype)
        else:
            self.ln = None
        
        self.mlp = nn.Sequential(*layers).to(device=device, dtype=dtype)

    def forward(
        self, 
        input_ids: torch.Tensor,      # [B, L]
        attention_mask: torch.Tensor, # [B, L] 1=valid, 0=pad
        emb_layer: nn.Embedding,      # LLM 的 embedding 层
    ) -> torch.Tensor:
        """
        返回:
            tau_b: [B, P, H]
        """
        device = emb_layer.weight.device
        dtype = emb_layer.weight.dtype
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        # [B, L, H]
        tok_embs = emb_layer(input_ids)
        
        # mask 掉 pad，做加权平均池化得到一个 [B, H] 的 patch 表征
        # avoid divide-by-zero
        attn = attention_mask.unsqueeze(-1).to(tok_embs.dtype)  # [B,L,1]
        summed = (tok_embs * attn).sum(dim=1)                   # [B,H]
        denom = attn.sum(dim=1).clamp_min(1e-6)                 # [B,1]
        seq_rep = summed / denom                                # [B,H]

        # MLP 投影成 [B, P*H]
        seq_rep = seq_rep.to(dtype)
        out = self.mlp(seq_rep)                                 # [B, P*H]
        tau_b = out.view(-1, self.length, self.hidden_size)     # [B, P, H]

        if self.ln is not None:
            # 对每个 prefix token 做 LN
            B, P, H = tau_b.shape
            tau_b = self.ln(tau_b.view(B * P, H)).view(B, P, H)

        return tau_b

def save_amortized_prefix(
    module: AmortizedPrefix,
    cfg: AmortizedPrefixConfig,
    hidden_size: int,
    savepath: str,
):
    """
    保存 amortized prefix 网络的 checkpoint。
    内容包括:
      - state_dict: 网络权重
      - cfg: 配置 (length, hidden_dim, use_layernorm, ...)
      - hidden_size: 方便恢复时检查维度
    """
    os.makedirs(os.path.dirname(savepath) or ".", exist_ok=True)
    ckpt = {
        "state_dict": module.state_dict(),
        "cfg": asdict(cfg),
        "hidden_size": hidden_size,
    }
    torch.save(ckpt, savepath)
    print(f"[AmortizedPrefix] Saved checkpoint to {savepath}")


def load_amortized_prefix(
    ckpt_path: str,
    model,              # LLM，用来取 hidden_size / dtype / device
    device: torch.device,
) -> AmortizedPrefix:
    """
    从 ckpt 加载 amortized prefix 网络。
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg_dict = ckpt["cfg"]
    hidden_size = ckpt.get("hidden_size", model.config.hidden_size)
    cfg = AmortizedPrefixConfig(**cfg_dict)

    emb = model.get_input_embeddings()
    target_device = emb.weight.device
    target_dtype = emb.weight.dtype
    prefix_net = AmortizedPrefix(
        hidden_size=hidden_size,
        dtype=target_dtype,
        device=target_device,
        cfg=cfg,
    )

    prefix_net.load_state_dict(ckpt["state_dict"])
    prefix_net.to(device=target_device, dtype=target_dtype)
    prefix_net.eval()
    print(f"[AmortizedPrefix] Loaded from {ckpt_path} (P={cfg.length}, H={hidden_size})")
    return prefix_net