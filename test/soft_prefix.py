import os
import torch
import torch.nn as nn
from dataclasses import dataclass

from typing import List, Dict, Tuple


# ----------------------------
# Soft Prefix τ
# ----------------------------

@dataclass
class PrefixConfig:
    length: int = 16           # number of virtual tokens
    init_std: float = 0.02     # Gaussian init std
    init_text: str = "Pixels:" # (optional) initialize τ from mean embedding of this text


class SoftPrefix(nn.Module):
    """
    Learnable sequence of embeddings [P, H] to prepend to token embeddings.
    τ is kept in the SAME dtype/device as the model's embedding table to avoid dtype mismatch.
    """
    def __init__(self, hidden_size: int, dtype: torch.dtype, device: torch.device, cfg: PrefixConfig):
        super().__init__()
        self.cfg = cfg
        self.prefix = nn.Parameter(torch.empty(cfg.length, hidden_size, dtype=dtype, device=device)) # [P, H]
        nn.init.normal_(self.prefix, mean=0.0, std=cfg.init_std)

    @torch.no_grad()
    def init_from_text(self, text: str, tokenizer, model):
        if not text:
            return
        emb = model.get_input_embeddings()
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) == 0:
            return
        vecs = emb.weight[torch.tensor(ids, device=emb.weight.device, dtype=torch.long)]  # [K, H]
        mean_vec = vecs.mean(dim=0, keepdim=True)                                         # [1, H]
        self.prefix.copy_(mean_vec.repeat(self.prefix.shape[0], 1))

    def forward(self) -> torch.Tensor:
        return self.prefix  # [P, H]

# ----------------------------
# Prefix utils
# ----------------------------
def load_prefix(prefix_ckpt_path: str, device: torch.device, model) -> Tuple[torch.Tensor, int]:
    """
    Load soft prefix τ [P,H] from checkpoint and cast to model embedding dtype/device.
    """
    ckpt = torch.load(prefix_ckpt_path, map_location="cpu")
    tau = ckpt["prefix"]                      # [P,H]
    emb = model.get_input_embeddings()
    tau = tau.to(device=device, dtype=emb.weight.dtype)
    P = tau.size(0)
    return tau, P

def save_prefix_ckpt(
    savepath: str,
    tau_tensor: torch.Tensor,  # [P, H]
):
    os.makedirs(os.path.dirname(savepath) or ".", exist_ok=True)
    ckpt = {"prefix": tau_tensor.cpu()}
    torch.save(ckpt, savepath)
    print(f"Saved prefix checkpoint to {savepath}")
