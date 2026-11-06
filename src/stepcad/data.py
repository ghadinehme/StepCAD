"""Streaming state-action samples for the step policy.

Each parquet row is one *state* of one model: the solid after action k,
its 2048-point FPS cloud (xyz + normals), bbox, and the action text.  A
training sample is a transition:

    target   Q_m      cloud of the target state (final, or a short prefix
                      during warm-up)
    current  Q_{t-1}  cloud of the state before action t (empty at t = 0)
    context  a_<t     the actions so far, as text
    hint     h_t      a 3D point on the geometry action t adds or removes
    label    a_t      action t's text -- or <|done|> once the target is built

Row groups are aligned to whole models by the writer, so the loader
streams row groups (each worker its own disjoint slice, per rank) and
shuffles transitions through a buffer.  A model split across row groups
(older shards) is skipped rather than half-read.
"""
from __future__ import annotations

import glob
import math
import os
import random

import numpy as np
import torch

N_INPUT = 1024          # points fed to Point-MAE (random subset of 2048)
HINT_TAU = None         # distance separating new geometry from old; None = adaptive


def disc_tau(xyz, k=1.5):
    """Threshold for "this point is not on that shape": k x the cloud's median
    nearest-neighbour spacing.  A fixed 0.02 sat below the sampling spacing of
    a 2048-point cloud (median 0.042), so 65% of points looked changed between
    consecutive states and hints / discrepancy regions were mostly noise."""
    from scipy.spatial import cKDTree
    d, _ = cKDTree(xyz).query(xyz, k=2)
    return k * float(np.median(d[:, 1]))
MAX_REACH = 2.0         # skip transitions whose current state reaches past this
EMPTY_LOGSCALE = -5.0


def _cloud(row):
    a = np.frombuffer(row['pointcloud'], dtype=np.float32)
    return a.reshape(row['pc_num_points'], row['pc_dims'])


def _geom(bmin, bmax):
    if bmin is None:
        return np.array([0, 0, 0, EMPTY_LOGSCALE], np.float32)
    lo, hi = np.asarray(bmin, np.float32), np.asarray(bmax, np.float32)
    c = (lo + hi) / 2
    s = float(max(hi - lo)) / 2
    return np.array([c[0], c[1], c[2], math.log(max(s, 1e-4))], np.float32)


def hint_point(new_xyz, old_xyz, rng, tau=HINT_TAU):
    """A point on the geometry the action adds or removes, near the seam.

    Points of the new state farther than tau from the old state belong to
    the new geometry; prefer those in (tau, 3 tau] -- the band next to the
    existing solid -- so the hint says "extend here".
    """
    if old_xyz is None:
        return new_xyz[rng.integers(len(new_xyz))]
    from scipy.spatial import cKDTree
    tau = tau or disc_tau(old_xyz)
    d, _ = cKDTree(old_xyz).query(new_xyz)
    band = np.nonzero((d > tau) & (d <= 3 * tau))[0]
    if len(band):
        return new_xyz[band[rng.integers(len(band))]]
    far = np.nonzero(d > tau)[0]
    if len(far):
        return new_xyz[far[rng.integers(len(far))]]
    return new_xyz[rng.integers(len(new_xyz))]


def _fnum(x):
    r = round(float(x), 3)
    if r == 0:
        return '0'
    return str(int(r)) if r == int(r) else repr(r)


def fmt_pt(p):
    """3-decimal point, same literal style as the programs."""
    return '(%s)' % ', '.join(_fnum(x) for x in p[:3])


HINT_DROP = 0.2         # share of hints replaced by an uninformative one


def _train_hint(clouds, k, t, rng):
    """Hint for transition t towards target k, as inference will see it.

    Inference always shows a hint; once the build matches the target it is
    a random target point.  The stop transition gets exactly that, so a
    missing hint never means "stop", and HINT_DROP of the action
    transitions get it too, so the model must read the clouds rather than
    lean on the hint.
    """
    tgt = clouds[k][:, :3]
    if t > k or rng.random() < HINT_DROP:
        return tgt[rng.integers(len(tgt))]
    return hint_point(clouds[t][:, :3], clouds[t - 1][:, :3] if t > 0 else None, rng)


GAINW = os.environ.get('STEPCAD_GAINW', '0') == '1'


def _fscore(a, b, tau=0.08):
    """Point F-score of cloud a against cloud b at tau (0.08 tracks the IoU
    gain of a step with Spearman 0.75 on held-out models)."""
    from scipy.spatial import cKDTree
    da, _ = cKDTree(b).query(a)
    db, _ = cKDTree(a).query(b)
    p, q = (da < tau).mean(), (db < tau).mean()
    return 2 * p * q / (p + q + 1e-9)


def gain_weight(g):
    """STEPCAD_GAINW=1: loss weight of a step from its shape-gain proxy.  Half
    the true later steps add < 0.02 IoU (fillets / chamfers ~0.002) and the
    policy learned to propose those touch-ups; the steps that build the shape
    are what the closed loop needs."""
    return 0.2 if g < 0.01 else (1.5 if g >= 0.1 else 1.0)


def transitions(model_rows, stage, rng, stop_prob=1.0):
    """Yield raw transitions of one model (rows sorted by op_index)."""
    T = len(model_rows)
    if stage == 'single':
        # whole program from the target cloud alone (pc2cad's objective):
        # no hint, no program text, nothing to lean on but the cloud
        yield {'target': _cloud(model_rows[-1]), 'current': None,
               'geom': _geom(None, None), 'context': [],
               'label': '\n'.join(r['code'] for r in model_rows),
               'hint': None, 't': 0, 'model': model_rows[0]['model']}
        return
    if stage == 'sweep':
        # one transition per model per pass: a pass covers every model of
        # the corpus once (2.3M models ~ 2.6M transitions) instead of every
        # step of every model (18.6M); later passes draw other steps
        k = T - 1
        t = int(rng.integers(0, T + 1))                 # t = T is the stop transition
        clouds = [_cloud(r) for r in model_rows]
        cur = model_rows[t - 1] if t > 0 else None
        yield {'target': clouds[k], 'current': clouds[t - 1] if t > 0 else None,
               'geom': _geom(cur['bbox_min'], cur['bbox_max']) if cur else _geom(None, None),
               'context': [model_rows[j]['code'] for j in range(t)],
               'label': model_rows[t]['code'] if t <= k else None,
               'hint': _train_hint(clouds, k, t, rng), 't': t, 'model': model_rows[0]['model'], 'w': 1.0}
        return
    if stage == 'warmup':
        targets = [k for k in (0, 1) if k < T]
    else:
        targets = [T - 1]
    clouds = [_cloud(r) for r in model_rows]
    fs = None
    if GAINW:
        fs = {}
    for k in targets:
        if fs is not None:
            tk = clouds[k][:, :3]
            fk = [0.0] + [_fscore(clouds[j][:, :3], tk) for j in range(k + 1)]   # fk[j+1]: after op j
        for t in range(k + 2):          # t = k + 1 is the stop transition
            if t == k + 1 and rng.random() > stop_prob:
                continue
            cur = model_rows[t - 1] if t > 0 else None
            yield {
                'target': clouds[k],
                'current': clouds[t - 1] if t > 0 else None,
                'geom': _geom(cur['bbox_min'], cur['bbox_max']) if cur else _geom(None, None),
                'context': [model_rows[j]['code'] for j in range(t)],
                'label': model_rows[t]['code'] if t <= k else None,
                'hint': _train_hint(clouds, k, t, rng),
                't': t, 'model': model_rows[0]['model'],
                'w': gain_weight(fk[t + 1] - fk[t]) if (fs is not None and t <= k) else 1.0,
            }


def recovery_transitions(rows, rec):
    """Off-trajectory transitions: perturbed prefix state -> true next action."""
    import json
    target = _cloud(rows[-1])
    for r in rec:
        ctx = json.loads(r['context_code']) if r.get('context_code') else []
        yield {'target': target, 'current': _cloud(r),
               'geom': _geom(r['bbox_min'], r['bbox_max']),
               'context': ctx, 'label': r['code'],
               # the hint still points at what the true action adds
               'hint': hint_point(_cloud(rows[r['op_index']])[:, :3], _cloud(r)[:, :3],
                                  np.random.default_rng(len(ctx))),
               't': r['op_index'], 'model': rows[0]['model'], 'recovery': True}


class StepStream(torch.utils.data.IterableDataset):
    COLS = ['model', 'op_index', 'num_ops', 'code', 'pointcloud',
            'pc_num_points', 'pc_dims', 'bbox_min', 'bbox_max']

    def __init__(self, files, stage='main', rank=0, world=1, seed=0,
                 buffer=4096, epochs=10_000, val=False, use_recovery=True,
                 start=0, epoch_samples=0, order_seed=None):
        super().__init__()
        # a fixed group order (order_seed) plus the number of transitions
        # already consumed (start) lets a resumed run pick up where the
        # previous one left off instead of re-reading the start of the pass
        self.start, self.epoch_samples, self.order_seed = start, epoch_samples, order_seed
        self.use_recovery = use_recovery and not val
        self.files = sorted(files)
        self.stage, self.rank, self.world = stage, rank, world
        self.seed, self.buffer, self.epochs, self.val = seed, buffer, epochs, val
        import pyarrow.parquet as pq
        self.groups = []
        for f in self.files:
            n = pq.ParquetFile(f).num_row_groups
            self.groups += [(f, g) for g in range(n)]

    def _my_groups(self, epoch):
        info = torch.utils.data.get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        g = list(self.groups)
        if not self.val:
            base = self.seed if self.order_seed is None else self.order_seed
            random.Random(base + epoch).shuffle(g)
        k = self.rank * nw + wid
        return g[k::self.world * nw]

    def _cols(self, f):
        import pyarrow.parquet as pq
        names = set(pq.ParquetFile(f).schema_arrow.names)
        return [c for c in self.COLS + ['is_recovery', 'context_code'] if c in names]

    def __iter__(self):
        import json
        import pyarrow.parquet as pq
        info = torch.utils.data.get_worker_info()
        wid = info.id if info else 0
        rng = np.random.default_rng(self.seed * 1000 + self.rank * 97 + wid)
        buf = []
        ep0, skip = 0, 0.0
        if not self.val and self.epoch_samples and self.start:
            pos = self.start / self.epoch_samples
            ep0, skip = int(pos), pos - int(pos)
        for ep in range(ep0, 1 if self.val else self.epochs):
            mine = self._my_groups(ep)
            if ep == ep0 and skip:
                mine = mine[int(skip * len(mine)):]
            for f, g in mine:
                t = pq.ParquetFile(f).read_row_group(g, columns=self._cols(f)).to_pylist()
                by = {}
                for r in t:
                    by.setdefault(r['model'], []).append(r)
                for allrows in by.values():
                    rows = [r for r in allrows if not r.get('is_recovery')]
                    rec = [r for r in allrows if r.get('is_recovery')]
                    rows.sort(key=lambda r: r['op_index'])
                    if not rows or [r['op_index'] for r in rows] != list(range(rows[0]['num_ops'])):
                        continue                    # split across row groups
                    samples = list(transitions(rows, self.stage, rng,
                                               stop_prob=0.5 if self.stage == 'warmup' else 1.0))
                    if self.stage == 'main' and self.use_recovery:
                        samples += list(recovery_transitions(rows, rec))
                    elif self.stage == 'sweep' and self.use_recovery and rec and rng.random() < 0.15:
                        samples += list(recovery_transitions(rows, [rec[int(rng.integers(len(rec)))]]))
                    for s in samples:
                        # intermediate states may overshoot [-1, 1] (extrude
                        # long, cut back later); ~1% reach far beyond it, out
                        # of Point-MAE's input range -- skip those
                        if s['current'] is not None and np.abs(s['current'][:, :3]).max() > MAX_REACH:
                            continue
                        if self.val:
                            yield s
                            continue
                        buf.append(s)
                        if len(buf) >= self.buffer:
                            yield buf.pop(int(rng.integers(len(buf))))
        while buf:
            yield buf.pop(int(rng.integers(len(buf))))


# ---------------------------------------------------------------------
#  collate: text layout + tensors
# ---------------------------------------------------------------------

#: sequence layout, fixed per run through the environment so training,
#: rollout and the probes always agree:
#:   bare    [pc x64 target][pc x64 current][geom] prompt action
#:   prefix  labelled blocks after a short text header
#:   late    prompt [pc x64 target][pc x64 current][geom] action  -- the
#:           point tokens sit right before the generated action
LAYOUT = os.environ.get('STEPCAD_LAYOUT', 'bare')


def head_ids(tok, pc_id, geom_id, k=64):
    """Token ids of the point/geometry block for the current LAYOUT.

    Measured on an untrained model, the share of attention the action
    positions give the point blocks is ~20-25% for bare, ~1% for prefix
    (the text header draws it away).
    """
    if os.environ.get('STEPCAD_INSEQ', '1') == '0':
        # shape reaches the LLM only through cross-attention: with the point
        # tokens also in the sequence, the LLM's urge to suppress them
        # pushed the encoder to constant output and starved the
        # cross-attention path (token diversity 0.12 -> 0.07 in 250 steps)
        return [geom_id]
    if LAYOUT == 'prefix':
        enc = lambda s: tok.encode(s, add_special_tokens=False)
        return (enc('# CAD reconstruction\n# target shape:\n') + [pc_id] * k
                + enc('\n# current shape:\n') + [pc_id] * k
                + enc('\n# current bbox:') + [geom_id])
    return [pc_id] * (2 * k) + [geom_id]


def build_ids(head, prompt):
    """Input ids before the action: point block and prompt in LAYOUT order."""
    return prompt + head if LAYOUT == 'late' else head + prompt


TBOX = os.environ.get('STEPCAD_TBOX', '1') == '1'


def target_box(cloud):
    """The target's bounding box from its cloud, for the prompt."""
    if cloud is None or not TBOX:
        return None
    x = np.asarray(cloud)[:, :3]
    return x.min(0), x.max(0)


BOXES = os.environ.get('STEPCAD_BOXES', '0') == '1'


def discrepancy_boxes(target, current, tau=None):
    """STEPCAD_BOXES=1: bounding boxes of the largest region the build is
    missing (target points > tau from the current cloud) and the largest
    excess region (current points > tau from the target), for the prompt.
    The oracle diagnosis showed the policy picks the right kind of operation
    but places it wrong; these give it coordinates to copy, as the target
    bbox does for the first operation."""
    if not BOXES or current is None:
        return None
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    t, c = np.asarray(target)[:, :3], np.asarray(current)[:, :3]
    out = []
    for name, src, ref in (('missing', t, c), ('excess', c, t)):
        d, _ = cKDTree(ref).query(src)
        P = src[d > (tau or disc_tau(ref))]
        if len(P) < 6:
            continue
        _, nb = cKDTree(P).query(P, k=min(7, len(P)))
        rows = np.repeat(np.arange(len(P)), nb.shape[1] - 1)
        g = coo_matrix((np.ones(len(rows)), (rows, nb[:, 1:].ravel())), shape=(len(P), len(P)))
        _, lab = connected_components(g, directed=False)
        big = P[lab == np.argmax(np.bincount(lab))]
        if len(big) >= 6:
            out.append((name, big.min(0), big.max(0)))
    return out


def prompt_text(context, t, hint, use_text=True, use_hint=True, tbox=None, boxes=None):
    lines = []
    if tbox is not None:
        # exact extents as digits the LLM can copy, while the cross-attention
        # learns the finer geometry
        lines.append('# target bbox: min %s max %s' % (fmt_pt(tbox[0]), fmt_pt(tbox[1])))
    for name, lo, hi in boxes or ():
        lines.append('# %s region: min %s max %s' % (name, fmt_pt(lo), fmt_pt(hi)))
    if use_hint and hint is not None:
        lines.append('# hint: %s' % fmt_pt(hint))
    if use_text:
        lines.append('# program so far:')
        lines.append('\n'.join(context) if context else '# (empty)')
    lines.append('# Generate CAD operation %d for this shape:' % (t + 1))
    return '\n' + '\n'.join(lines) + '\n'


NORMALS = os.environ.get('STEPCAD_NORMALS', '0') == '1'
MAX_ACTION = int(os.environ.get('STEPCAD_MAX_ACTION', '100000'))   # label tokens; 1536 = decode cap
NPTS = int(os.environ.get('STEPCAD_NPTS', '1024'))      # points per cloud fed to the encoder (image: 2048)
PC_DIM = 6 if NORMALS else 3


def subsample(cloud, n, rng):
    """n points of a cloud; with STEPCAD_NORMALS=1 xyz + unit normal."""
    if cloud is None:
        return np.zeros((n, PC_DIM), np.float32)
    idx = rng.choice(len(cloud), size=n, replace=len(cloud) < n)
    if PC_DIM > cloud.shape[1]:
        raise ValueError('cloud has no normals but STEPCAD_NORMALS=1')
    return cloud[idx, :PC_DIM].astype(np.float32)


class Collate:
    def __init__(self, tok, pc_id, geom_id, done_token, max_len=4096,
                 use_text=True, use_hint=True, tokens_per_pc=64, seed=0):
        self.tok, self.pc_id, self.geom_id = tok, pc_id, geom_id
        self.done = done_token
        self.max_len, self.use_text, self.use_hint = max_len, use_text, use_hint
        self.k = tokens_per_pc
        self.rng = np.random.default_rng(seed)

    def __call__(self, batch):
        ids, labs, tg, cu, em, ge, ws = [], [], [], [], [], [], []
        eos = self.tok.eos_token_id
        head = head_ids(self.tok, self.pc_id, self.geom_id, self.k)
        for s in batch:
            target = (s['label'] + '\n') if s['label'] is not None else self.done
            y = self.tok.encode(target, add_special_tokens=False) + [eos]
            if len(y) > MAX_ACTION:
                continue                 # giant spline actions (2.5% > 1536 tokens): past the decode cap
            ctx = list(s['context'])
            boxes = discrepancy_boxes(s['target'], s['current'])
            while True:
                p = self.tok.encode(prompt_text(ctx, s['t'], s['hint'],
                                                self.use_text, self.use_hint,
                                                tbox=target_box(s['target']), boxes=boxes),
                                    add_special_tokens=False)
                if len(head) + len(p) + len(y) <= self.max_len or not ctx:
                    break
                ctx = ctx[1:]            # drop the oldest actions first
            x = build_ids(head, p) + y
            if len(x) > self.max_len:
                continue                 # a single action longer than the window
            ids.append(x)
            labs.append([-100] * (len(head) + len(p)) + y)
            tg.append(subsample(s['target'], NPTS, self.rng))
            cu.append(subsample(s['current'], NPTS, self.rng))
            em.append(s['current'] is None)
            ge.append(s['geom'])
            ws.append(float(s.get('w', 1.0)))
        if not ids:
            return None
        L = max(len(x) for x in ids)
        pad = self.tok.pad_token_id
        return {
            'input_ids': torch.tensor([x + [pad] * (L - len(x)) for x in ids]),
            'attention_mask': torch.tensor([[1] * len(x) + [0] * (L - len(x)) for x in ids]),
            'labels': torch.tensor([y + [-100] * (L - len(y)) for y in labs]),
            'target': torch.from_numpy(np.stack(tg)),
            'current': torch.from_numpy(np.stack(cu)),
            'current_empty': torch.tensor(em),
            'geom': torch.from_numpy(np.stack(ge)),
            'weight': torch.tensor(ws, dtype=torch.float32),
        }
