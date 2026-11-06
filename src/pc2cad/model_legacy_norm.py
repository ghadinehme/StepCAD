"""
PC2CAD Model: Point cloud -> first CAD operation.

Architecture:
  1.  FPS downsample 10K -> N points
  2.  Fourier positional encoding of (x,y,z) coords
  3.  Linear projection  ->  Qwen hidden dim (point tokens)
  4.  Center/scale  ->  small MLP  ->  1 conditioning token
  5.  Qwen2.5-Coder generates first-step CadQuery code

Float handling (quantized bins):
  - Each unsigned float in [0, 10) quantized to 1024 bins
  - Bin tokens <|f0|> ... <|f1023|> added to vocab
  - Negative sign stays as text token before bin token
  - Pure classification via standard CE loss (no regression head)
  - Max quantization error: 0.005 (bin width 0.00977)
"""

import math
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer


# Two-range non-uniform binning for pre-normalized data:
#   [0.0, 1.0) : 1000 bins, width 0.001    (fine — most values live here)
#   [1.0, 3.0) : 200 bins,  width 0.01     (coarse — outliers / extrude sizes)
# Total: 1200 float-bin tokens. Samples with |x| >= 3.0 are dropped.
NUM_BINS_FINE = 1000
FINE_WIDTH = 0.001
NUM_BINS_COARSE = 200
COARSE_WIDTH = 0.01
NUM_FLOAT_BINS = NUM_BINS_FINE + NUM_BINS_COARSE          # 1200
FLOAT_MAX_VAL = 1.0 + NUM_BINS_COARSE * COARSE_WIDTH      # 3.0
# Kept for backwards-compat with any code that still reads this constant.
# (The actual width is range-dependent now, see float_to_bin / bin_to_float.)
FLOAT_BIN_WIDTH = FINE_WIDTH


def float_to_bin(x):
    """Unsigned float in [0, FLOAT_MAX_VAL) -> bin id in [0, NUM_FLOAT_BINS).
    Two-range mapping: fine bins under 1.0, coarser bins [1.0, 3.0)."""
    if x < 1.0:
        b = int(x / FINE_WIDTH)
        if b < 0: b = 0
        if b >= NUM_BINS_FINE: b = NUM_BINS_FINE - 1
        return b
    b = NUM_BINS_FINE + int((x - 1.0) / COARSE_WIDTH)
    if b >= NUM_FLOAT_BINS: b = NUM_FLOAT_BINS - 1
    return b


def bin_to_float(b):
    """Bin id -> left-edge float value. Matches the inverse of float_to_bin
    up to within one bin width (which is what we want for eval decode)."""
    if b < NUM_BINS_FINE:
        return b * FINE_WIDTH
    return 1.0 + (b - NUM_BINS_FINE) * COARSE_WIDTH


def make_float_token(b):
    """Bin index -> token string."""
    return f"<|f{b}|>"


# ---------------------------------------------------------------------------
# Furthest-Point Sampling (batch, GPU)
# ---------------------------------------------------------------------------

def furthest_point_sampling(xyz, npoint):
    B, N, _ = xyz.shape
    device = xyz.device
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, device=device)
    farthest = torch.randint(0, N, (B,), device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[torch.arange(B, device=device), farthest].unsqueeze(1)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        distance = torch.minimum(distance, dist)
        farthest = distance.argmax(dim=-1)
    return centroids


def fps_subsample(xyz, npoint):
    idx = furthest_point_sampling(xyz[:, :, :3], npoint)
    batch_idx = torch.arange(xyz.shape[0], device=xyz.device).unsqueeze(1).expand_as(idx)
    return xyz[batch_idx, idx]


# ---------------------------------------------------------------------------
# Fourier Positional Encoding
# ---------------------------------------------------------------------------

class FourierPE(nn.Module):
    def __init__(self, num_bands=64, max_freq=10.0):
        super().__init__()
        self.num_bands = num_bands
        freqs = 2.0 * math.pi * torch.logspace(0, math.log10(max_freq),
                                                 num_bands, base=10.0)
        self.register_buffer("freqs", freqs)

    @property
    def output_dim(self):
        return 3 * 2 * self.num_bands

    def forward(self, xyz):
        proj = xyz.unsqueeze(-1) * self.freqs
        return torch.cat([proj.sin(), proj.cos()], dim=-1).flatten(-2)


# ---------------------------------------------------------------------------
# Point Cloud Encoder
# ---------------------------------------------------------------------------

class PointCloudEncoder(nn.Module):
    def __init__(self, num_points=2048, num_bands=64,
                 max_freq=10.0, hidden_dim=2048):
        super().__init__()
        self.num_points = num_points
        self.fourier_pe = FourierPE(num_bands=num_bands, max_freq=max_freq)
        self.proj = nn.Linear(self.fourier_pe.output_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, xyz):
        pe = self.fourier_pe(xyz[:, :, :3])
        tokens = self.norm(self.proj(pe))
        return tokens


# ---------------------------------------------------------------------------
# Geometry Conditioning
# ---------------------------------------------------------------------------

class GeomCondEncoder(nn.Module):
    def __init__(self, hidden_dim=2048):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4, 256),
            nn.GELU(),
            nn.Linear(256, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, geom):
        return self.mlp(geom).unsqueeze(1)


# ---------------------------------------------------------------------------
# PC2CAD Model
# ---------------------------------------------------------------------------

class PC2CADModel(nn.Module):
    """
    Point cloud + center/scale -> first CAD operation code.
    Floats quantized into 1024 bin tokens. Pure CE loss.

    Input sequence to LLM:
        [point_tokens(N) | geom_token(1) | text+bin_tokens(...)]
    """
    def __init__(self, qwen_model="Qwen/Qwen2.5-Coder-3B",
                 num_points=2048, num_bands=64,
                 max_freq=10.0, full_finetune=True,
                 pc_encoder_kind="baseline", num_latents=256,
                 utonia_grid_size: float = 0.01,
                 utonia_pe_kind: str = "fourier"):
        super().__init__()

        self.llm = AutoModelForCausalLM.from_pretrained(
            qwen_model, torch_dtype=torch.bfloat16,
            attn_implementation="sdpa")
        self.tokenizer = AutoTokenizer.from_pretrained(qwen_model)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        hidden_dim = self.llm.config.hidden_size

        # Special tokens: pc_pad, geom_pad, and 1024 float bin tokens
        float_tokens = [make_float_token(i) for i in range(NUM_FLOAT_BINS)]
        special_tokens = ["<|pc_pad|>", "<|geom_pad|>"] + float_tokens
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": special_tokens})
        self.llm.resize_token_embeddings(len(self.tokenizer))

        self.pc_pad_id = self.tokenizer.convert_tokens_to_ids("<|pc_pad|>")
        self.geom_pad_id = self.tokenizer.convert_tokens_to_ids("<|geom_pad|>")

        # Store bin token id range for decoding
        self.float_bin_start = self.tokenizer.convert_tokens_to_ids("<|f0|>")
        self.float_bin_end = self.float_bin_start + NUM_FLOAT_BINS

        self.num_points = num_points
        self.pc_encoder_kind = pc_encoder_kind

        # Point cloud encoder: per-point baseline, Perceiver, or pretrained
        # Point-MAE (with torch_cluster-ported FPS/KNN grouping).
        if pc_encoder_kind == "perceiver":
            from pc2cad.pc_encoders import PerceiverEncoder
            _enc = PerceiverEncoder(
                hidden_dim=hidden_dim, num_latents=num_latents,
                num_heads=8, num_self_layers=2,
                num_bands=num_bands, max_freq=max_freq, use_normals=False)
            class _Single(nn.Module):
                def __init__(self, e):
                    super().__init__()
                    self.e = e
                    self.tokens_per_pc = e.tokens_per_pc
                def forward(self, pc):
                    return self.e._encode_one(pc)
            self.pc_encoder = _Single(_enc)
            self.tokens_per_pc = num_latents
        elif pc_encoder_kind == "pointmae":
            from pc2cad.pointmae_encoder import PointMAEForLLM
            self.pc_encoder = PointMAEForLLM(
                llm_hidden_dim=hidden_dim,
                num_group=num_latents,   # reuse --num_latents as group count
                group_size=32,
                embed_dim=384, depth=12, num_heads=6, mlp_ratio=4.0,
                pretrained_ckpt="/home/ubuntu/pc2cad/point_mae_pretrain.pth",
                freeze_encoder=False,
            )
            self.tokens_per_pc = self.pc_encoder.tokens_per_pc
        elif pc_encoder_kind == "utonia":
            from pc2cad.utonia_encoder import UtoniaEncoderForLLM
            self.pc_encoder = UtoniaEncoderForLLM(
                llm_hidden_dim=hidden_dim,
                tokens_per_pc=num_latents,
                grid_size=utonia_grid_size,
                pe_kind=utonia_pe_kind,
                num_heads=8,
            )
            self.tokens_per_pc = self.pc_encoder.tokens_per_pc
        else:
            self.pc_encoder = PointCloudEncoder(
                num_points=num_points, num_bands=num_bands,
                max_freq=max_freq, hidden_dim=hidden_dim)
            self.tokens_per_pc = num_points

        # Geometry conditioning
        self.geom_encoder = GeomCondEncoder(hidden_dim=hidden_dim)

        # Freeze / unfreeze
        if not full_finetune:
            for param in self.llm.parameters():
                param.requires_grad = False
            for layer in self.llm.model.layers[-4:]:
                for param in layer.parameters():
                    param.requires_grad = True
            for param in self.llm.lm_head.parameters():
                param.requires_grad = True

    def forward(self, input_ids, attention_mask, labels, pc_embeds, geom_embeds):
        embeds = self.llm.get_input_embeddings()(input_ids)

        for i in range(input_ids.shape[0]):
            pc_pos = (input_ids[i] == self.pc_pad_id).nonzero(as_tuple=True)[0]
            n = min(len(pc_pos), pc_embeds.shape[1])
            if n > 0:
                embeds[i, pc_pos[:n]] = pc_embeds[i, :n].to(embeds.dtype)
            geom_pos = (input_ids[i] == self.geom_pad_id).nonzero(as_tuple=True)[0]
            if len(geom_pos) > 0:
                embeds[i, geom_pos[0]] = geom_embeds[i, 0].to(embeds.dtype)

        return self.llm(
            inputs_embeds=embeds,
            attention_mask=attention_mask,
            labels=labels,
        )

    def decode_token_id(self, token_id):
        """Decode a token id; if it's a float bin, return the float string."""
        if self.float_bin_start <= token_id < self.float_bin_end:
            b = token_id - self.float_bin_start
            return f"{bin_to_float(b):.3f}"
        return self.tokenizer.decode([token_id])

    def trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_pc2cad(qwen_model="Qwen/Qwen2.5-Coder-3B",
                 num_points=2048, num_bands=64, max_freq=10.0,
                 full_finetune=True, pc_encoder_kind="baseline",
                 num_latents=256,
                 utonia_grid_size: float = 0.01,
                 utonia_pe_kind: str = "fourier"):
    model = PC2CADModel(
        qwen_model=qwen_model,
        num_points=num_points,
        num_bands=num_bands,
        max_freq=max_freq,
        full_finetune=full_finetune,
        pc_encoder_kind=pc_encoder_kind,
        num_latents=num_latents,
        utonia_grid_size=utonia_grid_size,
        utonia_pe_kind=utonia_pe_kind,
    )
    total = sum(p.numel() for p in model.parameters()) / 1e6
    trainable = model.trainable_parameters() / 1e6
    vocab = len(model.tokenizer)
    print(f"  PC2CAD: Total={total:.1f}M  Trainable={trainable:.1f}M  "
          f"Vocab={vocab} ({NUM_FLOAT_BINS} float bins)", flush=True)
    return model
