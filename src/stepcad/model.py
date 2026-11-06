"""Π-StepCAD policy: π(a_t | Q_{t-1}, Q_m, a_<t).

Input sequence to the LLM (placeholders are replaced by embeddings):

    [<pc>×64 target] [<pc>×64 current] [<geom>] text-prompt  action  <eos>

  * target and current point clouds share one Point-MAE encoder (ShapeNet
    pretrained weights, 64 groups, k=32, 12 layers) and one adapter;
  * a *residual* signal -- each group centre's distance to the other cloud
    -- is added to Point-MAE's position embedding through a small MLP whose
    output layer starts at zero, so the pretrained encoder is unchanged at
    step 0 and learns to use "what is missing / what is excess";
  * the geometry token encodes the current solid's bbox centre and log
    scale in the target frame (zeros and log-scale -5 for the empty state);
  * point and geometry tokens enter at the scale of the LLM's own token
    embeddings (a learnable gain, initialised to their RMS).  A LayerNorm
    output is ~50x larger than a Qwen embedding; a shakedown run showed the
    LLM then learns to stop attending to the point block (0.08% of
    attention) and the encoder's gradient vanishes -- the policy ignores
    the shape entirely;
  * floats are plain digits: Qwen2.5's tokenizer already splits numbers
    one digit per token, so no bins and no range limit.

Weights: bare ``Qwen/Qwen2.5-Coder-1.5B`` + the official Point-MAE
``pretrain.pth``.  A missing Point-MAE checkpoint is an error, not a
warning -- training silently from random encoder weights is the failure
mode this guards against.
"""
from __future__ import annotations

import math
import os

import torch
import torch.nn as nn

from pc2cad.pointmae_encoder import PointMAEEncoder, load_pretrained

PC_PAD = '<|pc_pad|>'
GEOM_PAD = '<|geom_pad|>'
DONE = '<|done|>'
TOKENS_PER_PC = 64


class ResidualPointMAE(nn.Module):
    """Point-MAE whose position embedding also sees a per-centre residual."""

    def __init__(self, ckpt, num_group=TOKENS_PER_PC, group_size=32):
        super().__init__()
        self.enc = PointMAEEncoder(num_group=num_group, group_size=group_size,
                                   embed_dim=384, depth=12, num_heads=6)
        if not ckpt or not os.path.exists(ckpt):
            raise FileNotFoundError('Point-MAE checkpoint not found: %r' % ckpt)
        n = load_pretrained(self.enc, ckpt)
        if n < 150:
            raise RuntimeError('Point-MAE loaded only %d tensors' % n)
        self.n_loaded = n
        self.res_embed = nn.Sequential(nn.Linear(1, 128), nn.GELU(),
                                       nn.Linear(128, 384))
        nn.init.zeros_(self.res_embed[2].weight)
        nn.init.zeros_(self.res_embed[2].bias)
        # STEPCAD_GEOFEAT=1: metric features per patch -- Fourier features of
        # the patch centre and its bounding box, added to the output through
        # a zero-initialised projection.  Point-MAE encodes local shape well
        # but positions only coarsely; CAD literals need them to ~1e-3
        self.geofeat = os.environ.get('STEPCAD_GEOFEAT', '0') == '1'
        self.register_buffer('freqs', 2 * math.pi * torch.logspace(0, math.log10(32.0), 16))
        self.geo_proj = nn.Linear(9 + 9 * 2 * 16, 384)
        nn.init.zeros_(self.geo_proj.weight)
        nn.init.zeros_(self.geo_proj.bias)
        if not self.geofeat:
            for q in self.geo_proj.parameters():
                q.requires_grad_(False)
        # STEPCAD_RESFEAT=1: the residual per point, not per centre -- every
        # patch summarises how far its 32 points are from the other cloud
        # (mean, max, share beyond 4 thresholds), so a patch that is half
        # built reads differently from one that is missing or complete.  The
        # single centre distance above cannot tell these apart.  Zero-init.
        self.resfeat = os.environ.get('STEPCAD_RESFEAT', '0') == '1'
        self.register_buffer('res_tau', torch.tensor([0.01, 0.02, 0.05, 0.1]))
        self.res_proj = nn.Sequential(nn.Linear(6, 128), nn.GELU(), nn.Linear(128, 384))
        nn.init.zeros_(self.res_proj[2].weight)
        nn.init.zeros_(self.res_proj[2].bias)
        if not self.resfeat:
            for q in self.res_proj.parameters():
                q.requires_grad_(False)
        # STEPCAD_NORMALS=1: clouds carry unit normals (xyz + n).  Point-MAE
        # was pretrained on xyz only, so normals enter through zero-init
        # branches: per point, a 1x1 conv beside the first PointNet layer;
        # per patch, the normal second moment (dominant face directions,
        # which CAD workplane normals are made of).  Both use n n^T, which is
        # the same for n and -n: mesh normals of real targets may be flipped.
        self.normals = os.environ.get('STEPCAD_NORMALS', '0') == '1'
        self.n_conv = nn.Conv1d(6, 128, 1)
        nn.init.zeros_(self.n_conv.weight)
        nn.init.zeros_(self.n_conv.bias)
        self.n_proj = nn.Sequential(nn.Linear(6, 128), nn.GELU(), nn.Linear(128, 384))
        nn.init.zeros_(self.n_proj[2].weight)
        nn.init.zeros_(self.n_proj[2].bias)
        if not self.normals:
            for q in list(self.n_conv.parameters()) + list(self.n_proj.parameters()):
                q.requires_grad_(False)

    def forward(self, xyz, other=None, use_residual=True, other_empty=None):
        e = self.enc
        nn_feat = None
        if self.normals and xyz.shape[-1] >= 6:
            gd = e.group_divider
            pts = xyz[..., :3].contiguous()
            center = gd._fps(pts, gd.num_group)
            idx = gd._knn(center, pts, gd.group_size)
            bi = torch.arange(xyz.shape[0], device=xyz.device).view(-1, 1, 1).expand_as(idx)
            neighborhood = pts[bi, idx] - center.unsqueeze(2)
            n = xyz[bi, idx][..., 3:6].float()                              # [B, G, K, 3]
            nn_feat = torch.stack([n[..., 0] * n[..., 0], n[..., 1] * n[..., 1], n[..., 2] * n[..., 2],
                                   n[..., 0] * n[..., 1], n[..., 0] * n[..., 2], n[..., 1] * n[..., 2]], -1)
        else:
            neighborhood, center = e.group_divider(xyz[..., :3].contiguous())
        # The mini-PointNet has BatchNorm.  Training runs one shape per
        # forward (micro-batch 1), so BN normalises with that shape's own
        # statistics; its running averages -- updated from single shapes and
        # from the all-zero empty-state cloud -- are badly off, and eval mode
        # used them, so every evaluation saw features training never
        # produced.  Each shape is now always normalised with its own
        # statistics (see StepPolicy.train), one shape at a time.
        enc_one = e.encoder if nn_feat is None else self._patch_encode
        if nn_feat is None:
            parts = [(neighborhood[b:b + 1],) for b in range(neighborhood.shape[0])]
        else:
            parts = [(neighborhood[b:b + 1], nn_feat[b:b + 1]) for b in range(neighborhood.shape[0])]
        tokens = torch.cat([enc_one(*p) for p in parts])
        pos = e.pos_embed(center)
        if use_residual and other is not None:
            d = torch.cdist(center.float(), other[..., :3].float()).min(-1).values
            pos = pos + self.res_embed(d.unsqueeze(-1).to(pos.dtype))
        x = tokens
        for blk in e.blocks:
            x = blk(x, pos)
        x = e.norm(x)
        if self.geofeat:
            g = torch.cat([center, center + neighborhood.amin(2), center + neighborhood.amax(2)], -1).float()
            ang = g[..., None] * self.freqs                                   # [B, G, 9, F]
            f = torch.cat([g, ang.sin().flatten(-2), ang.cos().flatten(-2)], -1)
            x = x + self.geo_proj(f).to(x.dtype)
        if nn_feat is not None:
            x = x + self.n_proj(nn_feat.mean(2)).to(x.dtype)            # per-patch normal moment
        if self.resfeat:
            B, G, K, _ = neighborhood.shape
            if other is None:      # empty current state: everything is missing
                f = torch.ones(B, G, 6, device=x.device)
            else:
                p = (center[:, :, None] + neighborhood).reshape(B, G * K, 3).float()
                d = torch.cdist(p, other[..., :3].float()).min(-1).values.reshape(B, G, K)
                f = torch.cat([d.mean(-1, keepdim=True), d.amax(-1, keepdim=True),
                               (d[..., None] > self.res_tau).float().mean(2)], -1)
                if other_empty is not None:   # the zero placeholder cloud is not a build
                    f = torch.where(other_empty[:, None, None], torch.ones_like(f), f)
            x = x + self.res_proj(f).to(x.dtype)
        return x


    def _patch_encode(self, nb, nf):
        """Point-MAE's mini-PointNet with the zero-init normal branch added
        to its first layer (same weights and output as the original when the
        branch is zero)."""
        enc = self.enc.encoder
        bs, g, k, _ = nb.shape
        h = enc.first_conv[0](nb.reshape(bs * g, k, 3).transpose(2, 1))
        h = h + self.n_conv(nf.reshape(bs * g, k, 6).transpose(2, 1).to(h.dtype))
        for m in enc.first_conv[1:]:
            h = m(h)
        gf = h.max(dim=2, keepdim=True).values
        h = enc.second_conv(torch.cat([gf.expand(-1, -1, k), h], dim=1))
        return h.max(dim=2).values.reshape(bs, g, enc.encoder_channel)


class FourierPointEncoder(nn.Module):
    """STEPCAD_ENC=fourier: CAD-Recode-style point tokens instead of Point-MAE.

    n_tokens farthest-point-sampled points, one token each, carrying the
    exact coordinates through multi-frequency Fourier features (CAD literals
    need ~1e-3; Point-MAE's 64 patch tokens encode positions only coarsely),
    plus sign-invariant normal moments (n n^T) and the distance to the other
    cloud (the residual), through a small MLP.  Same call signature and
    output width as ResidualPointMAE, so the adapter / cross-attention stay.
    """

    def __init__(self, n_tokens=256, n_freq=16, width=384):
        super().__init__()
        self.n_tokens = n_tokens
        self.register_buffer('freqs', math.pi * torch.logspace(0, math.log10(256.0), n_freq))
        d_in = 3 + 3 * 2 * n_freq + 6 + 1 + 2 * 8
        self.register_buffer('dfreqs', math.pi * torch.logspace(0, math.log10(64.0), 8))
        self.mlp = nn.Sequential(nn.Linear(d_in, width), nn.GELU(), nn.Linear(width, width),
                                 nn.GELU(), nn.Linear(width, width), nn.LayerNorm(width))

    def _fps(self, xyz):
        from pc2cad.pointmae_encoder import PointMAEGroup
        return PointMAEGroup._fps(xyz, self.n_tokens)

    def forward(self, xyz, other=None, use_residual=True, other_empty=None):
        pts = xyz[..., :3].contiguous().float()
        B = pts.shape[0]
        c = self._fps(pts)                                                   # [B, T, 3]
        self.last_c = c
        idx = torch.cdist(c, pts).argmin(-1)                                 # FPS points' rows
        sel = torch.gather(xyz.float(), 1, idx[..., None].expand(-1, -1, xyz.shape[-1]))
        ang = c[..., None] * self.freqs                                       # [B, T, 3, F]
        feats = [c, ang.sin().flatten(-2), ang.cos().flatten(-2)]
        if xyz.shape[-1] >= 6:
            n = sel[..., 3:6]
            feats.append(torch.stack([n[..., 0] ** 2, n[..., 1] ** 2, n[..., 2] ** 2, n[..., 0] * n[..., 1],
                                      n[..., 0] * n[..., 2], n[..., 1] * n[..., 2]], -1))
        else:
            feats.append(torch.zeros(B, c.shape[1], 6, device=c.device))
        if other is not None and use_residual:
            d = torch.cdist(c, other[..., :3].float()).min(-1).values[..., None]
            if other_empty is not None:
                d = torch.where(other_empty[:, None, None], torch.ones_like(d), d)
        else:
            d = torch.ones(B, c.shape[1], 1, device=c.device)
        dang = d * self.dfreqs
        feats += [d, dang.sin(), dang.cos()]
        return self.mlp(torch.cat(feats, -1).to(self.mlp[0].weight.dtype))


class ImageOverlayEncoder(nn.Module):
    """STEPCAD_ENC=image: CADENA-style overlay of target and current build.

    Both clouds are splatted into 8 orthographic views (6 axis-aligned, 2
    isometric) at 224 px: green = target, red = current build, each coded by
    depth (near = bright), blue = the target's surface shading (|n . view|).
    Where the build is missing the pixel is green only, where it is excess
    red only -- the discrepancy the step policy must act on is explicit.
    DINOv2-small encodes each view; its 16x16 patch tokens are pooled 2x2 to
    64 per view, plus a learned view embedding -> 512 tokens of width 384.
    """

    VIEWS = [(0, 0), (90, 0), (180, 0), (270, 0), (0, 90), (0, -90), (45, 35.26), (225, -35.26)]

    def __init__(self, res=224, grid=112, width=384):
        super().__init__()
        from transformers import AutoModel
        self.dino = AutoModel.from_pretrained('facebook/dinov2-small')
        self.res, self.grid = res, grid
        R = []
        for az, el in self.VIEWS:
            a, e = math.radians(az), math.radians(el)
            Rz = torch.tensor([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1.0]])
            Rx = torch.tensor([[1.0, 0, 0], [0, math.cos(e), -math.sin(e)], [0, math.sin(e), math.cos(e)]])
            base = torch.tensor([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]])      # view along +y, z up
            R.append(Rx @ base @ Rz)
        self.register_buffer('R', torch.stack(R))                               # [V, 3, 3]
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.view_emb = nn.Parameter(torch.zeros(len(self.VIEWS), 1, width))
        self.proj = nn.Linear(self.dino.config.hidden_size, width)

    def _splat(self, pts, nrm, valid):
        """pts [B, N, 3] -> depth image [B, V, g, g] and shading [B, V, g, g]."""
        B, N, _ = pts.shape
        V, g = self.R.shape[0], self.grid
        P = torch.einsum('vij,bnj->bvni', self.R, pts.float())                  # x right, y up, z toward viewer
        px = ((P[..., 0] / 1.1 + 1) * 0.5 * (g - 1)).round().long().clamp(0, g - 1)
        py = ((1 - (P[..., 1] / 1.1 + 1) * 0.5) * (g - 1)).round().long().clamp(0, g - 1)
        depth = ((P[..., 2] / 1.1 + 1) * 0.5).clamp(0, 1)                         # 1 = nearest
        if nrm is not None:
            vd = self.R[:, 2]                                                    # view direction per view
            shade = torch.einsum('vj,bnj->bvn', vd, nrm.float()).abs()
        else:
            shade = torch.ones_like(depth)
        key = (depth * 1023).round().long() * 1024 + (shade * 1023).round().long()
        key = torch.where(valid[:, None, :], key, torch.full_like(key, -1))
        flat = (py * g + px)
        img = torch.full((B, V, g * g), -1, dtype=torch.long, device=pts.device)
        img = img.scatter_reduce(2, flat, key, reduce='amax')
        hit = img >= 0
        d = torch.where(hit, (img // 1024).float() / 1023, torch.zeros_like(img, dtype=torch.float))
        s = torch.where(hit, (img % 1024).float() / 1023, torch.zeros_like(img, dtype=torch.float))
        return d.view(B, V, g, g), s.view(B, V, g, g)

    def forward(self, target, current, current_empty):
        import torch.nn.functional as F
        B = target.shape[0]
        tn = target[..., 3:6] if target.shape[-1] >= 6 else None
        cn = current[..., 3:6] if current.shape[-1] >= 6 else None
        ones = torch.ones(target.shape[:2], dtype=torch.bool, device=target.device)
        td, ts = self._splat(target[..., :3], tn, ones)
        cvalid = (~current_empty)[:, None].expand(-1, current.shape[1])
        cd, _ = self._splat(current[..., :3], cn, cvalid)
        # dilate the splats so the 2048-point surfaces read as solid regions
        dil = lambda x: F.max_pool2d(x.flatten(0, 1)[:, None], 3, 1, 1).view_as(x)
        td, ts, cd = dil(td), dil(ts), dil(cd)
        img = torch.stack([cd, td, ts], 2).flatten(0, 1)                         # [B*V, 3, g, g]
        img = F.interpolate(img, size=(self.res, self.res), mode='bilinear', align_corners=False)
        img = (img - self.mean) / self.std
        h = self.dino(pixel_values=img.to(self.dino.dtype)).last_hidden_state[:, 1:]   # patch tokens
        n = int(round(h.shape[1] ** 0.5))
        h = h.transpose(1, 2).reshape(h.shape[0], -1, n, n)
        h = F.avg_pool2d(h.float(), 2).flatten(2).transpose(1, 2)               # [B*V, 64, C]
        z = self.proj(h.to(self.proj.weight.dtype)).view(B, len(self.VIEWS), -1, self.proj.out_features)
        z = z + self.view_emb[None]
        return z.flatten(1, 2)                                                   # [B, 512, width]

    def render(self, target, current, current_empty):
        """The overlay images themselves ([B, V, 3, g, g] in [0, 1]) for inspection."""
        import torch.nn.functional as F
        tn = target[..., 3:6] if target.shape[-1] >= 6 else None
        ones = torch.ones(target.shape[:2], dtype=torch.bool, device=target.device)
        td, ts = self._splat(target[..., :3], tn, ones)
        cd, _ = self._splat(current[..., :3], None, (~current_empty)[:, None].expand(-1, current.shape[1]))
        dil = lambda x: F.max_pool2d(x.flatten(0, 1)[:, None], 3, 1, 1).view_as(x)
        return torch.stack([dil(cd), dil(td), dil(ts)], 2)


class GeomToken(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(4, 256), nn.GELU(),
                                 nn.Linear(256, hidden), nn.LayerNorm(hidden))

    def forward(self, g):
        return self.mlp(g).unsqueeze(1)


class GatedXAttn(nn.Module):
    """Flamingo-style gated cross-attention from the text stream to the
    point tokens: h <- h + tanh(g) * Attn(LN(h), LN(z)).  The gate starts at
    zero, so at initialisation the shape input does not perturb the LLM at
    all and there is nothing for it to learn to suppress; the gate opens
    only as far as the shape lowers the loss.  (With the point tokens only
    in the input sequence, every shakedown LLM learned to stop attending to
    them within a few hundred steps and never recovered.)"""

    def __init__(self, H, heads=12, gate0=0.0):
        super().__init__()
        self.ln_q = nn.LayerNorm(H)
        self.ln_kv = nn.LayerNorm(H)
        self.attn = nn.MultiheadAttention(H, heads, batch_first=True)
        # gate0 > 0 starts slightly open: with a closed gate the attention
        # weights get no gradient and, in a 1.5k-step probe, the gate only
        # random-walked -- the shape never reached the LLM
        self.gate = nn.Parameter(torch.tensor(math.atanh(gate0)))
        # STEPCAD_XATTN_MODE=zeroproj: no gate (fixed at 1); the attention's
        # output projection starts at zero instead (LoRA / ControlNet style).
        # The LLM is still unperturbed at step 0, but the projection gets a
        # gradient immediately and there is no gate for the optimiser to
        # shut -- in an open-gate probe the gates were driven from 0.25 to
        # ~0.01 within 500 steps while the shape signal kept growing
        self.zeroproj = os.environ.get('STEPCAD_XATTN_MODE', 'gate') == 'zeroproj'
        if self.zeroproj:
            nn.init.zeros_(self.attn.out_proj.weight)
            nn.init.zeros_(self.attn.out_proj.bias)
            self.gate.requires_grad_(False)

    def forward(self, h, z, mask=None):
        kv = self.ln_kv(z).to(h.dtype)
        a, _ = self.attn(self.ln_q(h), kv, kv, key_padding_mask=mask, need_weights=False)
        if self.zeroproj:
            return h + a
        return h + torch.tanh(self.gate).to(h.dtype) * a


class StepPolicy(nn.Module):
    def __init__(self, qwen='Qwen/Qwen2.5-Coder-1.5B', pointmae_ckpt=None,
                 use_residual=True, grad_ckpt=False, dtype=torch.float32,
                 xattn_every=None):
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer
        # fp32 master weights by default; compute runs under bf16 autocast
        self.llm = AutoModelForCausalLM.from_pretrained(
            qwen, torch_dtype=dtype, attn_implementation='sdpa')
        self.tok = AutoTokenizer.from_pretrained(qwen)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.add_special_tokens({'additional_special_tokens':
                                     [PC_PAD, GEOM_PAD, DONE]})
        self.llm.resize_token_embeddings(len(self.tok))
        self.pc_id = self.tok.convert_tokens_to_ids(PC_PAD)
        self.geom_id = self.tok.convert_tokens_to_ids(GEOM_PAD)
        self.done_id = self.tok.convert_tokens_to_ids(DONE)
        if grad_ckpt:
            # non-reentrant: the reentrant variant breaks DDP gradient sync
            self.llm.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={'use_reentrant': False})
        H = self.llm.config.hidden_size
        self.pc = ResidualPointMAE(pointmae_ckpt)
        # STEPCAD_ENC=fourier: CAD-Recode-style per-point tokens replace
        # Point-MAE (kept loaded but frozen, so checkpoints stay compatible)
        self.fourier = os.environ.get('STEPCAD_ENC', 'pointmae') == 'fourier'
        self.image = os.environ.get('STEPCAD_ENC', 'pointmae') == 'image'
        self.fpe = FourierPointEncoder()
        for q in (self.pc if self.fourier else self.fpe).parameters():
            q.requires_grad_(False)
        if self.image:
            # STEPCAD_ENC=image: target/current overlay views through DINOv2
            # replace both point encoders (kept loaded, frozen)
            self.img = ImageOverlayEncoder()
            for q in list(self.pc.parameters()) + list(self.fpe.parameters()):
                q.requires_grad_(False)
        self.adapter = nn.Sequential(nn.Linear(384, H), nn.GELU(),
                                     nn.Linear(H, H), nn.LayerNorm(H))
        self.geom = GeomToken(H)
        with torch.no_grad():
            rms = float(self.llm.get_input_embeddings().weight.float().pow(2).mean().sqrt())
        # gains in log space around the embedding RMS: Adam moves a
        # parameter by ~lr per step whatever its size, so a raw 0.02 gain
        # could undo itself within a few hundred steps
        # STEPCAD_PC_SCALE=ln keeps the LayerNorm scale (rms 1, ~50x a text
        # embedding), as the original pc2cad encoder did
        if os.environ.get('STEPCAD_PC_SCALE', 'text') == 'ln':
            rms = 1.0
        self.register_buffer('emb_rms', torch.tensor(rms), persistent=True)
        self.pc_log_gain = nn.Parameter(torch.zeros(()))
        self.geom_log_gain = nn.Parameter(torch.zeros(()))
        # auxiliary read-out: pooled point tokens -> the cloud's own min,
        # max and spread.  Keeps the tokens informative while the LLM is
        # still learning to read them; without it a shakedown run's adapter
        # collapsed to near-identical tokens for every shape (spread across
        # samples 0.63 -> 0.09) and the LLM stopped attending to them.
        self.aux = nn.Linear(H, 9)
        self.aux_w = 0.0
        self._aux_loss = None
        # gated cross-attention after every xattn_every-th decoder layer
        # (STEPCAD_XATTN, default 4; 0 disables)
        if xattn_every is None:
            xattn_every = int(os.environ.get('STEPCAD_XATTN', '4'))
        layers = self.llm.model.layers
        self.xattn_layers = list(range(xattn_every - 1, len(layers), xattn_every)) if xattn_every else []
        gate0 = float(os.environ.get('STEPCAD_XATTN_GATE0', '0'))
        self.xattn = nn.ModuleList([GatedXAttn(H, gate0=gate0) for _ in self.xattn_layers])
        self._xz, self._xmask = None, None
        for blk, li in zip(self.xattn, self.xattn_layers):
            layers[li].register_forward_hook(self._make_hook(blk))
        self.use_residual = use_residual
        self.disc = os.environ.get("STEPCAD_DISC", "0") == "1"
        # STEPCAD_DISCAUX=1 (Fourier tokens): every target token predicts whether
        # its point is missing from the build, every current token whether its
        # point is excess -- the encoder must represent the discrepancy itself
        self.discaux = os.environ.get('STEPCAD_DISCAUX', '0') == '1'
        self.disc_head = nn.Linear(self.llm.config.hidden_size, 2)
        if not self.discaux:
            for q in self.disc_head.parameters():
                q.requires_grad_(False)

    def load_state_dict(self, sd, strict=True, **kw):
        # checkpoints from before the per-point residual have no res_proj /
        # res_tau; take this module's zero-initialised values for exactly
        # those keys, so the old model loads unchanged
        sd = dict(sd)
        w = sd.get('pc.res_proj.2.weight')
        if w is not None and not self.pc.resfeat and float(w.abs().sum()) > 0:
            raise RuntimeError('checkpoint was trained with STEPCAD_RESFEAT=1; set it to evaluate')
        w = sd.get('pc.n_proj.2.weight')
        if w is not None and not self.pc.normals and float(w.abs().sum()) > 0:
            raise RuntimeError('checkpoint was trained with STEPCAD_NORMALS=1; set it to evaluate')
        for k, v in self.state_dict().items():
            if k not in sd and ('pc.res_proj.' in k or k == 'pc.res_tau'
                                or 'pc.n_conv.' in k or 'pc.n_proj.' in k or k.startswith('fpe.')
                                or k.startswith('img.') or k.startswith('disc_head.')):
                sd[k] = v
        return super().load_state_dict(sd, strict=strict, **kw)

    def train(self, mode=True):
        """BatchNorm in the point encoder always uses per-shape batch
        statistics (training behaviour) and never updates its running
        averages, in every mode -- see ResidualPointMAE.forward."""
        super().train(mode)
        for m in self.pc.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.train(True)
                m.momentum = 0.0
        return self

    def _make_hook(self, blk):
        def hook(module, inp, out):
            if self._xz is None:
                return out
            h = out[0] if isinstance(out, tuple) else out
            z, m = self._xz, self._xmask
            if z.shape[0] != h.shape[0]:            # generate(num_return_sequences=n)
                r = h.shape[0] // z.shape[0]
                z, m = z.repeat_interleave(r, 0), m.repeat_interleave(r, 0)
            h = blk(h, z, m)
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        return hook

    def _gain(self, log_g):
        return self.emb_rms * log_g.exp()

    @staticmethod
    def _describe(xyz):
        x = xyz[..., :3].float()
        return torch.cat([x.amin(1), x.amax(1), x.std(1)], -1)      # [B, 9]

    def _penc(self, *a, **k):
        return (self.fpe if self.fourier else self.pc)(*a, **k)

    @staticmethod
    def _disc(a, b, b_empty, n=1024, k=2.5, min_pts=16):
        """Points of a farther than tau from b, resampled to n -> (cloud,
        none) where none marks rows with fewer than min_pts such points.  An
        empty b (b_empty) makes all of a the region."""
        d = torch.cdist(a[..., :3].float(), b[..., :3].float()).min(-1).values
        # threshold: k x b's median point spacing (a fixed 0.02 was below the
        # spacing of these clouds and flagged most points as different)
        bb = torch.cdist(b[..., :3].float(), b[..., :3].float())
        bb = bb + torch.eye(bb.shape[-1], device=bb.device) * 1e6
        tau = k * bb.min(-1).values.median(-1).values                     # [B]
        w = (d > tau[:, None]).float()
        if b_empty is not None:
            w = torch.where(b_empty[:, None], torch.ones_like(w), w)
        none = w.sum(1) < min_pts
        w = torch.where(none[:, None], torch.ones_like(w), w)
        idx = torch.multinomial(w, n, replacement=True)
        return torch.gather(a, 1, idx[..., None].expand(-1, -1, a.shape[-1])), none

    def _disc_aux(self, at, ac, target, current, current_empty, k=2.5):
        """Per-token missing / excess classification (BCE, positives up-weighted)."""
        import torch.nn.functional as F
        ct, cc = self._tok_pos_t, self._tok_pos_c
        def tau(x):
            bb = torch.cdist(x[..., :3].float(), x[..., :3].float())
            bb = bb + torch.eye(bb.shape[-1], device=bb.device) * 1e6
            return k * bb.min(-1).values.median(-1).values
        miss = torch.cdist(ct, current[..., :3].float()).min(-1).values > tau(current)[:, None]
        miss = miss | current_empty[:, None]                                  # empty build: everything missing
        exc = torch.cdist(cc, target[..., :3].float()).min(-1).values > tau(target)[:, None]
        lt = self.disc_head(at.float())[..., 0]
        lc = self.disc_head(ac.float())[..., 1]
        pw = torch.tensor(4.0, device=at.device)
        l1 = F.binary_cross_entropy_with_logits(lt, miss.float(), pos_weight=pw)
        full = (~current_empty).float()[:, None].expand_as(lc)
        l2 = (F.binary_cross_entropy_with_logits(lc, exc.float(), pos_weight=pw, reduction='none') * full).sum() / full.sum().clamp(min=1)
        return l1 + l2

    def encode_clouds(self, target, current, current_empty):
        """target, current: [B, N, 3]; current_empty: [B] bool."""
        if self.image:
            import torch.nn.functional as F
            z = self.adapter(self.img(target, current, current_empty))
            self._aux_loss = F.mse_loss(self.aux(z.float().mean(1)), self._describe(target))
            self._last_pool = z.detach().float().mean(1)
            self._xz = z
            self._xmask = torch.zeros(z.shape[:2], dtype=torch.bool, device=z.device)
            d = z[:, :1] * self._gain(self.pc_log_gain)                          # no in-sequence tokens (INSEQ=0)
            return d, d
        at = self.adapter(self._penc(target, current, self.use_residual, other_empty=current_empty))
        self._tok_pos_t = getattr(self.fpe, 'last_c', None)
        ac = self.adapter(self._penc(current, target, self.use_residual))
        self._tok_pos_c = getattr(self.fpe, 'last_c', None)
        # auxiliary descriptor regression (current only where it exists)
        import torch.nn.functional as F
        full = (~current_empty).float()
        lt = F.mse_loss(self.aux(at.float().mean(1)), self._describe(target))
        lc = (F.mse_loss(self.aux(ac.float().mean(1)), self._describe(current),
                         reduction='none').mean(-1) * full).sum() / full.sum().clamp(min=1)
        self._aux_loss = lt + lc
        if self.discaux and self.fourier:
            self._aux_loss = self._aux_loss + self._disc_aux(at, ac, target, current, current_empty)
        self._last_pool = at.detach().float().mean(1)     # for the token-diversity health metric
        # cross-attention keys: the un-scaled tokens; empty current masked
        self._xz = torch.cat([at, ac], 1)
        self._xmask = torch.cat([torch.zeros_like(current_empty)[:, None].expand(-1, at.shape[1]),
                                 current_empty[:, None].expand(-1, ac.shape[1])], 1)
        if self.disc:
            # STEPCAD_DISC=1: two more clouds, each 1024 points drawn only from
            # where target and build disagree -- what is missing (target far
            # from the build) and what is excess (build far from the target).
            # A small missing feature is a handful of the 1024 input points;
            # here it is all of them.  Extra cross-attention keys, masked when
            # the region is empty.
            miss, miss_none = self._disc(target, current, current_empty)
            exc, exc_none = self._disc(current, target, None)
            exc_none = exc_none | current_empty
            am = self.adapter(self._penc(miss, None, False))
            ae = self.adapter(self._penc(exc, None, False))
            self._xz = torch.cat([self._xz, am, ae], 1)
            self._xmask = torch.cat([self._xmask, miss_none[:, None].expand(-1, am.shape[1]),
                                     exc_none[:, None].expand(-1, ae.shape[1])], 1)
        zt = at * self._gain(self.pc_log_gain)
        zc = ac * self._gain(self.pc_log_gain)
        # the empty state has no geometry: zero its tokens rather than
        # encoding a degenerate cloud
        zc = zc * (~current_empty).to(zc.dtype)[:, None, None]
        return zt, zc

    def embed(self, input_ids, target, current, current_empty, geom):
        emb = self.llm.get_input_embeddings()(input_ids)
        zt, zc = self.encode_clouds(target, current, current_empty)
        pcs = torch.cat([zt, zc], 1).to(emb.dtype)            # [B, 128, H]
        g = (self.geom(geom.to(torch.float32)) * self._gain(self.geom_log_gain)).to(emb.dtype)  # [B, 1, H]
        out = emb.clone()
        for i in range(input_ids.shape[0]):
            pp = (input_ids[i] == self.pc_id).nonzero(as_tuple=True)[0]
            out[i, pp] = pcs[i, :len(pp)]
            gp = (input_ids[i] == self.geom_id).nonzero(as_tuple=True)[0]
            if len(gp):
                out[i, gp[0]] = g[i, 0]
        return out

    def forward(self, input_ids, attention_mask, labels, target, current,
                current_empty, geom):
        """Causal-LM loss computed on the labelled positions only.

        The full logits tensor (sequence x 152k vocabulary, fp32) costs
        ~5 GB per 4k-token sample, yet only the ~100-300 action tokens carry
        a loss.  Running the decoder, then the output head on just those
        positions, cuts that memory by more than an order of magnitude.
        """
        import torch.nn.functional as F
        from types import SimpleNamespace
        emb = self.embed(input_ids, target, current, current_empty, geom)
        h = self.llm.model(inputs_embeds=emb, attention_mask=attention_mask,
                           use_cache=False).last_hidden_state
        y = labels[:, 1:]
        m = y != -100
        hm, ym = h[:, :-1][m], y[m]
        # the output head in chunks, each recomputed in backward: a 4k-token
        # label over a 152k vocabulary is ~2.5 GB of fp32 logits otherwise
        from torch.utils.checkpoint import checkpoint
        head = lambda hc, yc: F.cross_entropy(self.llm.lm_head(hc).float(), yc, reduction='sum')
        C = 1024
        tot = sum(checkpoint(head, hm[i:i + C], ym[i:i + C], use_reentrant=False)
                  if self.training else head(hm[i:i + C], ym[i:i + C])
                  for i in range(0, len(ym), C))
        ce = tot / max(1, len(ym))
        aux = self._aux_loss
        return SimpleNamespace(loss=ce + self.aux_w * aux, ce=ce, aux=aux,
                               n_tokens=int(m.sum()))

    def param_groups(self, lr_llm, lr_pc, lr_proj=None, wd=0.01, lr_xattn=None, wd_xattn=0.0):
        """llm: the decoder; pc: the pretrained Point-MAE; proj: the new
        layers between them (adapter, geometry MLP, residual MLP, gains)."""
        lr_proj = lr_proj or lr_llm
        llm = [p for p in self.llm.parameters() if p.requires_grad]
        pc = [p for p in self.pc.enc.parameters() if p.requires_grad]
        if self.image:
            pc += list(self.img.dino.parameters())
        geo = [q for q in self.pc.geo_proj.parameters() if q.requires_grad]
        geo += [q for q in self.pc.res_proj.parameters() if q.requires_grad]
        geo += [q for q in list(self.pc.n_conv.parameters()) + list(self.pc.n_proj.parameters()) if q.requires_grad]
        gains = [self.pc_log_gain, self.geom_log_gain]
        proj = (list(self.adapter.parameters()) + list(self.geom.parameters()) + list(self.aux.parameters())
                + [p for p in self.disc_head.parameters() if p.requires_grad]
                + [p for p in self.pc.res_embed.parameters() if p.requires_grad] + geo
                + [p for p in self.fpe.parameters() if p.requires_grad]
                + ([self.img.view_emb] + list(self.img.proj.parameters()) if self.image else []))
        groups = [{'params': llm, 'lr': lr_llm, 'weight_decay': wd, 'name': 'llm'},
                  {'params': pc, 'lr': lr_pc, 'weight_decay': wd, 'name': 'pc'},
                  {'params': proj, 'lr': lr_proj, 'weight_decay': wd, 'name': 'proj'},
                  # log-gains: <= ~e^1 drift per 10k steps
                  {'params': gains, 'lr': lr_proj * 0.1, 'weight_decay': 0.0, 'name': 'gain'},
                  {'params': [p for p in self.xattn.parameters() if p.requires_grad], 'lr': lr_xattn or lr_proj,
                   'weight_decay': wd_xattn, 'name': 'xattn'}]
        ids = [id(p) for g in groups for p in g['params']]
        want = {id(p) for p in self.parameters() if p.requires_grad}
        assert len(ids) == len(set(ids)) and set(ids) == want, 'param groups do not partition the model'
        return groups
