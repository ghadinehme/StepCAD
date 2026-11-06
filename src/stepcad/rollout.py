"""Closed-loop Π-StepCAD inference with best-of-N per step, and evaluation.

For every target mesh:

    S_0 = empty
    for t = 1 .. T_max:
        hints  <- discrepancy regions between target and current shape
        sample N candidate actions (split across the hints, T = temperature)
        execute every candidate on S_{t-1}; score IoU(S_t, target)
        keep the best; stop if its IoU gain < 0.001, or all candidates are
        <|done|> / invalid
    return the best prefix  (IoU never decreases)

Execution and scoring run in a CPU process pool; generation runs on the
GPU.  One process per GPU (torchrun), each taking a disjoint slice of the
eval set.  The final program of every mesh is scored with the full
verified metrics (stepcad.metrics).

    torchrun --nproc_per_node 8 -m stepcad.rollout --ckpt main/final.pt \
        --evalset eval_dev_1400.parquet --n 16 --out results/policy_bo16
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import time

import numpy as np

HEADER = 'import cadquery as cq\nimport math\n'

# ---------------------------------------------------------------------
#  CPU side: execute a program, score it, sample its cloud
# ---------------------------------------------------------------------

_TARGETS = {}


def _target(path):
    if path not in _TARGETS:
        import trimesh
        _TARGETS[path] = trimesh.load(path, force='mesh')
        if len(_TARGETS) > 64:
            _TARGETS.pop(next(iter(_TARGETS)))
    return _TARGETS[path]


def _worker_init():
    from OCP.OSD import OSD_ThreadPool
    try:
        OSD_ThreadPool.DefaultPool_s().Init(1)
    except Exception:
        pass
    import cadquery  # noqa: F401  warm the import


def exec_and_score(actions, target_path, fast=True, cloud=True, seed=0):
    """Run HEADER + actions; return validity, IoU vs target, next-state cloud."""
    import cadquery as cq
    from stepcad import metrics as M
    out = {'ok': False}
    try:
        ns = {'cq': cq, 'math': math}
        exec(HEADER + '\n'.join(actions), ns)
        res = ns.get('result')
        ok, why = M.valid_solid(res)
        if not ok:
            out['err'] = why
            return out
        tm = _target(target_path)
        pm = M.solid_to_mesh(res)
        if fast:
            out['iou'], _ = (M.iou(pm, tm, n=150_000))
        else:
            out.update(M.score(res, tm))
        if cloud:
            import trimesh
            from stepcad.geomutil import fps
            pts, fid = trimesh.sample.sample_surface(pm, 30000, seed=seed)
            q, nq = fps(pts, np.asarray(pm.face_normals)[fid], 2048, seed=seed)
            out['cloud'] = np.concatenate([q, nq], 1).astype(np.float32).tobytes()   # xyz + normal
        bb = res.val().BoundingBox()
        out['bbox'] = [[bb.xmin, bb.ymin, bb.zmin], [bb.xmax, bb.ymax, bb.zmax]]
        out['ok'] = True
    except Exception as exc:
        out['err'] = type(exc).__name__ + ': ' + str(exc)[:120]
    return out


def snap_candidate(actions, cand, target_path, n_init=4, n_iter=3, seed=0):
    """Per-step parameter snapping: Bayesian optimisation of the numeric
    literals of one candidate action (origins, sizes, depths) against the
    target, prefix fixed.  The policy often proposes the right operation a
    little off; this turns those near-misses into gains.  Returns
    (action, fast_iou) or None if no improvement was found."""
    from stepcad import refine as RF
    rng = np.random.default_rng(seed)
    score = RF.Scorer(_target(target_path))
    base = score(actions + [cand])
    if base < -0.5:
        return None
    out = RF.edit_continuous(actions + [cand], len(actions), score, rng, n_init=n_init, n_iter=n_iter, batch=3)
    if out is None:
        return None
    return out[-1], score(out)


# ---------------------------------------------------------------------
#  hints at inference: discrepancy components, walked to the seam
# ---------------------------------------------------------------------

def discrepancy_hints(target_xyz, current_xyz, k_max=4, tau=None, rng=None):
    """Up to k_max hint points, one per coherent region the build is missing.

    Target points farther than tau from the current cloud are candidates;
    a 6-NN graph over them is split into connected components, largest
    first; each component's representative (its point farthest from the
    current build) is walked greedily toward the build until within 2 tau.
    Excess regions (current points far from target) are added after the
    missing ones.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    rng = rng or np.random.default_rng(0)
    if current_xyz is None:
        return [target_xyz[rng.integers(len(target_xyz))]]
    hints = []
    for src, ref in ((target_xyz, current_xyz), (current_xyz, target_xyz)):
        if tau is None:
            from stepcad.data import disc_tau
            tau = disc_tau(current_xyz)
        d, _ = cKDTree(ref).query(src)
        cand = np.nonzero(d > tau)[0]
        if len(cand) < 6:
            continue
        P = src[cand]
        tree = cKDTree(P)
        _, nb = tree.query(P, k=min(7, len(P)))
        rows = np.repeat(np.arange(len(P)), nb.shape[1] - 1)
        g = coo_matrix((np.ones(len(rows)), (rows, nb[:, 1:].ravel())), shape=(len(P), len(P)))
        n, lab = connected_components(g, directed=False)
        sizes = np.bincount(lab)
        for c in np.argsort(-sizes):
            if sizes[c] < 6 or len(hints) >= k_max:
                break
            idx = np.nonzero(lab == c)[0]
            cur = idx[np.argmax(d[cand][idx])]
            # greedy walk toward the build
            for _ in range(2 * len(idx)):
                if d[cand][cur] <= 2 * tau:
                    break
                nxt = nb[cur, 1:]
                nxt = nxt[np.isin(nxt, idx)]
                if not len(nxt):
                    break
                best = nxt[np.argmin(d[cand][nxt])]
                if d[cand][best] >= d[cand][cur]:
                    break
                cur = best
            hints.append(P[cur])
        if len(hints) >= k_max:
            break
    if not hints:
        hints = [target_xyz[rng.integers(len(target_xyz))]]
    return hints


# ---------------------------------------------------------------------
#  GPU side
# ---------------------------------------------------------------------

def alpha_wrap(mesh, alpha_fraction=1e-3, offset_fraction=1e-4):
    """CGAL alpha_wrap_3 via pymeshlab, as CADBench's geometry.alpha_wrap does."""
    import pymeshlab as pml
    import trimesh
    ms = pml.MeshSet()
    ms.add_mesh(pml.Mesh(vertex_matrix=np.ascontiguousarray(mesh.vertices, dtype=np.float64),
                         face_matrix=np.ascontiguousarray(mesh.faces, dtype=np.int32)))
    for kw in ({'alpha_fraction': alpha_fraction, 'offset_fraction': offset_fraction},
               {'alpha_relative': alpha_fraction, 'offset_relative': offset_fraction}, {}):
        try:
            ms.apply_filter('generate_alpha_wrap', **kw)
            cm = ms.current_mesh()
            return trimesh.Trimesh(cm.vertex_matrix(), cm.face_matrix(), process=True)
        except Exception:
            continue
    return mesh


def target_cloud(stl_bytes, workdir, key):
    """Normalise the target into the eval frame, save it, sample its cloud."""
    import trimesh
    from stepcad.geomutil import fps
    from stepcad import metrics as M
    m = M.load_mesh(stl_bytes)
    mn, _, _ = M.normalise_target(m)
    path = os.path.join(workdir, '%s.stl' % key)
    # TARGET_WRAP=1: score against CADBench's alpha-wrapped target (closed outer shell, alpha 1e-3 and
    # offset 1e-4 of the bbox diagonal), so an open or soup-like input mesh gets a meaningful IoU
    (alpha_wrap(mn) if os.environ.get('TARGET_WRAP') == '1' and not mn.is_watertight else mn).export(path)
    pts, fid = trimesh.sample.sample_surface(mn, 30000, seed=0)
    q, nq = fps(pts, np.asarray(mn.face_normals)[fid], 2048, seed=0)
    return path, np.concatenate([q, nq], 1).astype(np.float32)          # xyz + normal


def generate(model, tok, target, current, geom, contexts, hints, ts, n_per,
             temperature, max_new, use_text, use_hint, dev):
    """Decoded actions per prompt, generated in chunks of at most
    STEPCAD_GEN_MAX sequences (default 32): 129 long candidates in one batch
    ran out of GPU memory and took down the rank's remaining meshes.  An OOM
    halves the chunk and retries."""
    import torch
    cap = max(1, int(os.environ.get('STEPCAD_GEN_MAX', '32')))
    res = [[] for _ in contexts]
    left = n_per
    chunk = max(1, cap // max(1, len(contexts)))
    while left > 0:
        k = min(chunk, left)
        try:
            part = _generate_once(model, tok, target, current, geom, contexts, hints, ts, k,
                                  temperature, max_new, use_text, use_hint, dev)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if chunk == 1:
                raise
            chunk = max(1, chunk // 2)
            continue
        for g, p in zip(res, part):
            g.extend(p)
        left -= k
    return res


def _generate_once(model, tok, target, current, geom, contexts, hints, ts, n_per,
                   temperature, max_new, use_text, use_hint, dev):
    """One batched generate call; returns list of decoded actions per prompt."""
    import torch
    from stepcad.data import build_ids, discrepancy_boxes, head_ids, prompt_text, subsample, target_box
    from stepcad.model import DONE
    rng = np.random.default_rng(int(time.time() * 1e3) % (2 ** 31))
    head = head_ids(tok, model.pc_id, model.geom_id)
    ids, tg, cu, em, ge = [], [], [], [], []
    for i in range(len(contexts)):
        p = tok.encode(prompt_text(contexts[i], ts[i], hints[i], use_text, use_hint,
                                   tbox=target_box(target[i]),
                                   boxes=discrepancy_boxes(target[i], current[i])),
                       add_special_tokens=False)
        ids.append(build_ids(head, p))
        from stepcad.data import NPTS
        tg.append(subsample(target[i], NPTS, rng))
        cu.append(subsample(current[i], NPTS, rng))
        em.append(current[i] is None)
        ge.append(geom[i])
    L = max(len(x) for x in ids)
    pad = tok.pad_token_id
    # left padding for generation
    input_ids = torch.tensor([[pad] * (L - len(x)) + x for x in ids], device=dev)
    attn = torch.tensor([[0] * (L - len(x)) + [1] * len(x) for x in ids], device=dev)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        emb = model.embed(input_ids, torch.from_numpy(np.stack(tg)).to(dev),
                          torch.from_numpy(np.stack(cu)).to(dev),
                          torch.tensor(em, device=dev),
                          torch.from_numpy(np.stack(ge)).to(dev))
        out = model.llm.generate(inputs_embeds=emb, attention_mask=attn,
                                 do_sample=temperature > 0, temperature=max(temperature, 1e-5),
                                 top_p=1.0, max_new_tokens=max_new,
                                 num_return_sequences=n_per,
                                 eos_token_id=tok.eos_token_id, pad_token_id=pad)
    texts = tok.batch_decode(out, skip_special_tokens=False)
    res = []
    for i in range(len(contexts)):
        group = []
        for j in range(n_per):
            t = texts[i * n_per + j]
            t = t.replace(tok.pad_token, '').split(tok.eos_token)[0].strip()
            group.append(None if t.startswith(DONE) or not t else t)
        res.append(group)
    return res


def rollout_one(model, tok, pool, key, tpath, tcloud, a, dev):
    """Closed loop: per step, N candidates (sampled, plus optional greedy),
    each executed and scored against the target; continue from the best.

    --n       sampled candidates per step (split across the hints); 0 = none
    --greedy  greedy candidates per step (one per hint, first hints first)
    --patience  steps the loop may continue without improving IoU before it
              stops; the best prefix seen is returned, so a detour that does
              not pay off costs nothing but time
    """
    from stepcad.data import _geom
    actions, cur_cloud, cur_bbox, cur_iou = [], None, None, 0.0
    best_iou, best_actions = 0.0, []
    stale = 0
    trace = []
    for t in range(a.t_max):
        hints = (discrepancy_hints(tcloud[:, :3], None if cur_cloud is None else cur_cloud[:, :3],
                                   k_max=a.k_hints) if not a.no_hint else [None])
        geom = _geom(*cur_bbox) if cur_bbox else _geom(None, None)
        flat = []
        n_t = a.n0 if (t == 0 and a.n0) else a.n   # the first action carries most of the IoU
        if n_t > 0:
            sh = hints[:n_t]                      # N < #hints: N hints, one sample each
            per = max(1, n_t // len(sh))
            cands = generate(model, tok, [tcloud] * len(sh), [cur_cloud] * len(sh),
                             [geom] * len(sh), [actions] * len(sh), sh,
                             [t] * len(sh), per, a.temperature, a.max_new,
                             not a.no_text, not a.no_hint, dev)
            flat += [c for g in cands for c in g if c is not None]
        if a.greedy > 0:
            gh = hints[:a.greedy]
            cands = generate(model, tok, [tcloud] * len(gh), [cur_cloud] * len(gh),
                             [geom] * len(gh), [actions] * len(gh), gh,
                             [t] * len(gh), 1, 0.0, a.max_new,
                             not a.no_text, not a.no_hint, dev)
            flat = [c for g in cands for c in g if c is not None] + flat
        flat = list(dict.fromkeys(flat))          # de-duplicate identical actions
        if not flat:
            trace.append({'t': t, 'stop': 'all_done'})
            break
        futs = [pool.schedule(exec_and_score, args=(actions + [c], tpath),
                              kwargs={'seed': t}, timeout=a.exec_timeout) for c in flat]
        best = None
        n_valid = 0
        for c, f in zip(flat, futs):
            try:
                r = f.result()
            except Exception:
                continue
            if r.get('ok'):
                n_valid += 1
                if best is None or r['iou'] > best[1]['iou']:
                    best = (c, r)
        if best is None:
            trace.append({'t': t, 'stop': 'all_invalid', 'n': len(flat)})
            break
        snapped = False
        if a.snap_k > 0:
            # snap the top-k valid candidates in parallel, keep the best result
            ranked = []
            for c, f in zip(flat, futs):
                try:
                    r = f.result()
                except Exception:
                    continue
                if r.get('ok'):
                    ranked.append((r['iou'], c))
            ranked.sort(reverse=True)
            sf = [pool.schedule(snap_candidate, args=(actions, c, tpath),
                                kwargs={'n_init': a.snap_init, 'n_iter': a.snap_iters, 'seed': t * 31 + i},
                                timeout=a.snap_timeout) for i, (_, c) in enumerate(ranked[:a.snap_k])]
            for f in sf:
                try:
                    out = f.result()
                except Exception:
                    out = None
                if out is None:
                    continue
                try:
                    r = pool.schedule(exec_and_score, args=(actions + [out[0]], tpath), kwargs={'seed': t},
                                      timeout=a.exec_timeout).result()
                except Exception:
                    continue                       # a slow or failing snapped candidate is just skipped
                if r.get('ok') and r['iou'] > best[1]['iou'] + 1e-4:
                    best, snapped = (out[0], r), True
        gain = best[1]['iou'] - cur_iou
        trace.append({'t': t, 'iou': round(best[1]['iou'], 4), 'gain': round(gain, 4),
                      'n_cand': len(flat), 'n_valid': n_valid, 'snapped': snapped})
        if gain < a.min_gain:
            stale += 1
            if stale > a.patience:
                trace[-1]['stop'] = 'plateau'
                break
        else:
            stale = 0
        actions.append(best[0])
        cur_iou = best[1]['iou']
        cur_cloud = np.frombuffer(best[1]['cloud'], np.float32).reshape(-1, 6)
        cur_bbox = best[1]['bbox']
        if cur_iou > best_iou + 1e-9:
            best_iou, best_actions = cur_iou, list(actions)
    return best_actions, trace


def main():
    import pyarrow.parquet as pq
    import torch
    import torch.distributed as dist
    from pebble import ProcessPool
    from stepcad.model import StepPolicy
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--evalset', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--qwen', default='Qwen/Qwen2.5-Coder-1.5B')
    ap.add_argument('--pointmae', default=os.environ.get('POINTMAE_CKPT', 'point_mae_pretrain.pth'))
    ap.add_argument('--n', type=int, default=16, help='candidates per step')
    ap.add_argument('--n0', type=int, default=0, help='candidates at the first step (0 = --n)')
    ap.add_argument('--snap_k', type=int, default=0, help='snap the top-k valid candidates per step (0 = off)')
    ap.add_argument('--snap_init', type=int, default=4)
    ap.add_argument('--snap_iters', type=int, default=3)
    ap.add_argument('--snap_timeout', type=float, default=120)
    ap.add_argument('--k_hints', type=int, default=4)
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--t_max', type=int, default=20)
    ap.add_argument('--min_gain', type=float, default=1e-3)
    ap.add_argument('--greedy', type=int, default=0, help='greedy candidates per step (one per hint)')
    ap.add_argument('--patience', type=int, default=0, help='non-improving steps allowed before stopping')
    ap.add_argument('--max_new', type=int, default=768)
    ap.add_argument('--exec_timeout', type=float, default=60)
    ap.add_argument('--cpus', type=int, default=0, help='pool size per GPU (0 = auto)')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--no_text', action='store_true')
    ap.add_argument('--no_hint', action='store_true')
    ap.add_argument('--no_residual', action='store_true')
    a = ap.parse_args()

    dist.init_process_group('nccl')
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local)
    dev = torch.device('cuda', local)
    os.makedirs(a.out, exist_ok=True)
    work = os.path.join(a.out, 'targets')
    os.makedirs(work, exist_ok=True)

    model = StepPolicy(a.qwen, a.pointmae, use_residual=not a.no_residual,
                       dtype=torch.bfloat16)
    sd = torch.load(a.ckpt, map_location='cpu', weights_only=False)
    model.load_state_dict(sd['model'] if 'model' in sd else sd)
    model.to(dev).eval()
    tok = model.tok

    t = pq.read_table(a.evalset)
    rows = t.to_pylist()
    if a.limit:
        rows = rows[:a.limit]
    mine = rows[rank::world]
    ncpu = a.cpus or max(4, (os.cpu_count() - 2 * world) // world)
    pool = ProcessPool(max_workers=ncpu, initializer=_worker_init)
    outp = open(os.path.join(a.out, 'rank%d.jsonl' % rank), 'a')
    done = set()
    try:
        done = {json.loads(l)['key'] for l in open(os.path.join(a.out, 'rank%d.jsonl' % rank))}
    except Exception:
        pass
    for r in mine:
        key = '%s__%s' % (r['group'], r['file_id'])
        if key in done:
            continue
        t0 = time.time()
        tpath, tcloud = target_cloud(r['stl'], work, key)
        actions, trace = rollout_one(model, tok, pool, key, tpath, tcloud, a, dev)
        final = {'valid': False, 'iou': 0.0}
        if actions:
            fut = pool.schedule(exec_and_score, args=(actions, tpath),
                                kwargs={'fast': False, 'cloud': False}, timeout=600)
            try:
                final = fut.result()
                final.pop('bbox', None)
            except Exception as exc:
                final = {'valid': False, 'iou': 0.0, 'err': repr(exc)[:80]}
        rec = {'key': key, 'group': r['group'], 'bench': r['bench'], 'file_id': r['file_id'],
               'program': HEADER + '\n'.join(actions) + '\nfinal_result = result\n',
               'n_actions': len(actions), 'trace': trace, 'final': final,
               'seconds': round(time.time() - t0, 1)}
        outp.write(json.dumps(rec) + '\n')
        outp.flush()
    pool.close()
    dist.barrier()
    if rank == 0:
        summarize(a.out)
    dist.destroy_process_group()


def summarize(out_dir):
    import collections
    import glob
    from stepcad import metrics as M
    recs = []
    for f in glob.glob(os.path.join(out_dir, 'rank*.jsonl')):
        recs += [json.loads(l) for l in open(f)]
    by = collections.defaultdict(list)
    for r in recs:
        fz = r['final']
        row = {'valid': bool(fz.get('valid', fz.get('ok', False))), 'iou': fz.get('iou', 0.0) or 0.0,
               'cd_8192': fz.get('cd_8192'), 'cd_30000': fz.get('cd_30000')}
        if not row['valid']:
            row['iou'] = 0.0
        by[r['group']].append(row)
        by['ALL'].append(row)
    table = {g: M.summarise(v) for g, v in sorted(by.items())}
    json.dump(table, open(os.path.join(out_dir, 'summary.json'), 'w'), indent=1)
    print('%-10s %5s %8s %8s %10s %10s %7s' % ('group', 'n', 'IoU', 'IoU(all)', 'CD8k med', 'CD30k med', 'VSR'))
    for g, s in table.items():
        print('%-10s %5d %8.3f %8.3f %10s %10s %7.2f' % (
            g, s['n'], s['iou_valid_mean'], s['iou_all_mean'],
            '%.5f' % s['cd_8192_median'] if s.get('cd_8192_median') is not None else '-',
            '%.5f' % s['cd_30000_median'] if s.get('cd_30000_median') is not None else '-',
            s['vsr']))


if __name__ == '__main__':
    main()
