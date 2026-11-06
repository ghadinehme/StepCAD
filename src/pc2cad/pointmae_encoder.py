"""Port of Point-MAE's encoder tower (Group + Encoder + 12 Transformer blocks)
with torch_cluster replacing pointnet2_ops / knn_cuda.

Architecture (matches the MAE_encoder state_dict from pretrain.pth):
  - Group     : FPS to num_group centers + KNN-K neighbors per center,
                centered on each center and concatenated -> (B, G, K, 3)
  - Encoder   : mini-PointNet per patch -> (B, G, 384) patch features
  - pos_embed : MLP(3 -> 128 -> 384) from centers
  - blocks    : 12 x standard pre-norm Transformer blocks (embed=384,
                heads=6, mlp_ratio=4, qkv_bias=False)
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from torch_cluster import fps as _fps, knn as _knn
except ImportError:                       # pure-torch fallback (same contract as torch_cluster.fps)
    def _fps(x, batch, ratio=0.5, random_start=True):
        out = []
        for b in torch.unique(batch):
            idx = (batch == b).nonzero(as_tuple=True)[0]
            p = x[idx]
            m = max(1, int(math.ceil(ratio * len(idx))))
            sel = torch.empty(m, dtype=torch.long, device=x.device)
            sel[0] = torch.randint(len(idx), (1,), device=x.device) if random_start else 0
            d = ((p - p[sel[0]]) ** 2).sum(-1)
            for i in range(1, m):
                sel[i] = d.argmax()
                d = torch.minimum(d, ((p - p[sel[i]]) ** 2).sum(-1))
            out.append(idx[sel])
        return torch.cat(out)
    _knn = None


# ---------- Group (FPS + KNN) ---------------------------------------------

class PointMAEGroup(nn.Module):
    def __init__(self, num_group: int = 64, group_size: int = 32):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size

    @staticmethod
    def _fps(xyz: torch.Tensor, num_group: int) -> torch.Tensor:
        """xyz: [B, N, 3] -> centers: [B, num_group, 3] via torch_cluster FPS."""
        B, N, _ = xyz.shape
        flat = xyz.reshape(-1, 3).contiguous()
        batch = torch.arange(B, device=xyz.device).repeat_interleave(N)
        ratio = float(num_group) / float(N) + 1e-4
        sel = _fps(flat, batch, ratio=ratio, random_start=True)
        sel_batch = batch[sel]
        counts = torch.bincount(sel_batch, minlength=B)
        # pick exactly num_group per sample (ragged edges trimmed)
        starts = torch.zeros(B, dtype=torch.long, device=xyz.device)
        starts[1:] = counts.cumsum(0)[:-1]
        within = torch.arange(num_group, device=xyz.device).unsqueeze(0) + starts.unsqueeze(1)
        sel = sel[within]
        local = sel - torch.arange(B, device=xyz.device).unsqueeze(1) * N
        centers = xyz.gather(1, local.unsqueeze(-1).expand(-1, -1, 3))
        return centers

    @staticmethod
    def _knn(centers: torch.Tensor, xyz: torch.Tensor, k: int) -> torch.Tensor:
        """Return indices [B, G, K] of K nearest xyz points for each center.
        Uses torch.cdist + topk instead of torch_cluster.knn because the
        latter sometimes returns fewer than k pairs per query when the
        input has coincident points / exact ties, breaking reshape."""
        # centers: [B, G, 3], xyz: [B, N, 3]
        d = torch.cdist(centers, xyz)       # [B, G, N]
        idx = d.topk(k, largest=False, dim=-1).indices  # [B, G, K]
        return idx

    def forward(self, xyz: torch.Tensor):
        """xyz: [B, N, 3] -> (neighborhood [B, G, K, 3], center [B, G, 3])."""
        B, N, _ = xyz.shape
        centers = self._fps(xyz, self.num_group)          # [B, G, 3]
        idx = self._knn(centers, xyz, self.group_size)    # [B, G, K]
        batch_idx = torch.arange(B, device=xyz.device).view(B, 1, 1).expand(-1, self.num_group, self.group_size)
        nbh = xyz[batch_idx, idx]                         # [B, G, K, 3]
        nbh = nbh - centers.unsqueeze(2)                  # center-relative
        return nbh, centers


# ---------- Encoder (per-patch mini-PointNet) -----------------------------

class PointMAEPatchEncoder(nn.Module):
    """Matches Point_MAE.Encoder: mini-PointNet on each patch -> embed_dim."""

    def __init__(self, encoder_channel: int = 384):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1),
        )

    def forward(self, point_groups: torch.Tensor):
        # point_groups: [B, G, K, 3]
        bs, g, k, _ = point_groups.shape
        x = point_groups.reshape(bs * g, k, 3)
        x = self.first_conv(x.transpose(2, 1))            # BG 256 K
        g_feat = x.max(dim=2, keepdim=True).values         # BG 256 1
        x = torch.cat([g_feat.expand(-1, -1, k), x], dim=1)  # BG 512 K
        x = self.second_conv(x)                            # BG C K
        g_feat = x.max(dim=2).values                        # BG C
        return g_feat.reshape(bs, g, self.encoder_channel)


# ---------- Transformer block (matches timm's Block w/ qkv_bias=False) -----

class PointMAEAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 6, attn_drop: float = 0., proj_drop: float = 0.):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # Use SDPA for speed + bfloat16
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        return self.proj_drop(out)


class PointMAEMlp(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor):
        return self.fc2(self.act(self.fc1(x)))


class PointMAEBlock(nn.Module):
    def __init__(self, dim: int = 384, num_heads: int = 6, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = PointMAEAttention(dim, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = PointMAEMlp(dim, int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor, pos: torch.Tensor):
        # Point-MAE adds pos embedding inside the block before the residual:
        #   x = x + drop_path(attn(norm1(x + pos)))
        # We'll match the canonical Point-MAE formulation exactly.
        x = x + self.attn(self.norm1(x + pos))
        x = x + self.mlp(self.norm2(x))
        return x


class PointMAEEncoder(nn.Module):
    """Drop-in PC encoder. Input: (B, N, 3) points. Output: (B, num_group,
    embed_dim) patch tokens. The wrapper in model_legacy_norm will project
    embed_dim -> LLM hidden_dim.
    """

    def __init__(self,
                 num_group: int = 64,
                 group_size: int = 32,
                 embed_dim: int = 384,
                 depth: int = 12,
                 num_heads: int = 6,
                 mlp_ratio: float = 4.0):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.embed_dim = embed_dim
        self.tokens_per_pc = num_group  # exposed to collate

        self.group_divider = PointMAEGroup(num_group, group_size)
        self.encoder = PointMAEPatchEncoder(embed_dim)

        # Position embedding (centers [B, G, 3] -> tokens [B, G, embed_dim])
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, embed_dim),
        )

        # Transformer stack
        self.blocks = nn.ModuleList([
            PointMAEBlock(embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, xyz: torch.Tensor):
        # xyz expected in [-1, 1]^3 (our normalized dataset)
        if xyz.shape[-1] > 3:
            xyz = xyz[..., :3]
        neighborhood, center = self.group_divider(xyz)
        tokens = self.encoder(neighborhood)                  # [B, G, C]
        pos = self.pos_embed(center)                         # [B, G, C]
        x = tokens
        for blk in self.blocks:
            x = blk(x, pos)
        return self.norm(x)                                  # [B, G, C]


def load_pretrained(model: PointMAEEncoder, ckpt_path: str, verbose: bool = True) -> int:
    """Load weights from Point-MAE's `base_model` state dict (prefix
    module.MAE_encoder.*). Returns number of tensors successfully loaded."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd_src = ck.get("base_model", ck)
    # Strip "module." prefix
    clean = {k[len("module."):]: v for k, v in sd_src.items() if k.startswith("module.")}
    # Keep only MAE_encoder.* and strip that prefix
    pref = "MAE_encoder."
    enc = {k[len(pref):]: v for k, v in clean.items() if k.startswith(pref)}
    # `blocks.blocks.N.*` -> `blocks.N.*`
    enc2 = {}
    for k, v in enc.items():
        if k.startswith("blocks.blocks."):
            enc2[k.replace("blocks.blocks.", "blocks.", 1)] = v
        else:
            enc2[k] = v
    # Load (non-strict; model has no mask token / cls).
    result = model.load_state_dict(enc2, strict=False)
    if verbose:
        print(f"  [PointMAE] loaded {len(enc2) - len(result.unexpected_keys)} tensors; "
              f"missing={len(result.missing_keys)} unexpected={len(result.unexpected_keys)}",
              flush=True)
        if result.missing_keys:
            print(f"    missing (first 5): {result.missing_keys[:5]}", flush=True)
    return len(enc2) - len(result.unexpected_keys)


class PointMAEForLLM(nn.Module):
    """PointMAE + linear adapter to LLM hidden dim. Exposes tokens_per_pc
    so the collate/model layout can size the pc_pad block correctly.
    """

    def __init__(self,
                 llm_hidden_dim: int,
                 num_group: int = 64,
                 group_size: int = 32,
                 embed_dim: int = 384,
                 depth: int = 12,
                 num_heads: int = 6,
                 mlp_ratio: float = 4.0,
                 pretrained_ckpt: str = None,
                 freeze_encoder: bool = False):
        super().__init__()
        self.encoder = PointMAEEncoder(
            num_group=num_group, group_size=group_size,
            embed_dim=embed_dim, depth=depth,
            num_heads=num_heads, mlp_ratio=mlp_ratio,
        )
        if pretrained_ckpt and os.path.exists(pretrained_ckpt):
            load_pretrained(self.encoder, pretrained_ckpt)
        elif pretrained_ckpt:
            print(f"  [PointMAE] WARNING: pretrained ckpt not found at {pretrained_ckpt}",
                  flush=True)
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
        # Adapter to LLM dim
        self.adapter = nn.Sequential(
            nn.Linear(embed_dim, llm_hidden_dim),
            nn.LayerNorm(llm_hidden_dim),
        )
        self.tokens_per_pc = num_group
        # Optional second, finer Point-MAE on the same cloud (POINTMAE_FINE="GxK", e.g. "128x16"):
        # its G tokens follow the coarse ones, tagged by a learned type embedding.
        fine = os.environ.get("POINTMAE_FINE", "")
        if fine:
            fg, fk = (int(v) for v in fine.lower().split("x"))
            self.fine = PointMAEEncoder(num_group=fg, group_size=fk, embed_dim=embed_dim, depth=depth,
                                        num_heads=num_heads, mlp_ratio=mlp_ratio)
            if pretrained_ckpt and os.path.exists(pretrained_ckpt):
                load_pretrained(self.fine, pretrained_ckpt)
            self.fine_adapter = nn.Sequential(nn.Linear(embed_dim, llm_hidden_dim), nn.LayerNorm(llm_hidden_dim))
            self.fine_type = nn.Parameter(torch.zeros(llm_hidden_dim))
            self.tokens_per_pc = num_group + fg

    @torch.no_grad()
    def init_fine_from_coarse(self):
        """Start the fine encoder / adapter from the (fine-tuned) coarse ones: the patch PointNet
        and transformer are agnostic to group count and size."""
        self.fine.load_state_dict(self.encoder.state_dict())
        self.fine_adapter.load_state_dict(self.adapter.state_dict())

    def forward(self, xyz: torch.Tensor):
        out = self.adapter(self.encoder(xyz))      # [B, G, llm_hidden_dim]
        if hasattr(self, "fine"):
            out = torch.cat([out, self.fine_adapter(self.fine(xyz)) + self.fine_type], dim=1)
        return out
