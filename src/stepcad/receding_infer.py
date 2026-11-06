"""Receding-horizon rollout of the StepCAD policy.

    state (actions so far, current geometry)
    loop:
        sample --n remaining programs  pi(rest | target, current, code)
        execute each full program (actions + rest), IoU vs target
        the previous best plan's remainder is carried over as a candidate,
        so the best plan's IoU never decreases
        commit the first action of the best plan, execute it -> new state
    until the best plan is fully committed or --t_max steps

Step 0 is the single-shot model (best-of---n0); the record keeps that
number ('iou_t0') next to the final one so the gain of re-planning is
measured on the same meshes.  Output is rollout-format (evaltab.py).

    torchrun --nproc_per_node 4 -m stepcad.receding_infer --ckpt ckpt.pt \
        --evalset eval_dev_140.parquet --out res --n0 128 --n 32
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from stepcad.rollout import HEADER, _worker_init, exec_and_score, target_cloud


def split_actions(text):
    """Generated remainder -> list of actions (an action starts at a line
    'result = ...'; continuation lines join it; stops at final_result).
    A program in another style (e.g. the StepCAD checkpoint's) is kept
    whole as a single action: it can be scored but not re-planned."""
    body = []
    for ln in text.split('\n'):
        s = ln.strip()
        if s.startswith('final_result'):
            break
        if s.startswith('import ') or not s:
            continue
        body.append(ln)
    if not body:
        return []
    if not body[0].lstrip().startswith(('result =', 'result=')):
        # v1 / StepCAD style (named planes and sketches): one action per
        # statement run that updates result, the same split the v1 chains
        # were built with (stepcad.actions)
        from stepcad.actions import split_program
        return split_program('\n'.join(body)) or ['\n'.join(body)]
    acts = []
    for ln in body:
        if ln.startswith(('result =', 'result=')):
            acts.append(ln)
        else:
            acts[-1] += '\n' + ln
    return acts


def main():
    import pyarrow.parquet as pq
    import torch
    import torch.distributed as dist
    from pebble import ProcessPool
    from pc2cad.model_legacy_norm import build_pc2cad
    from infer_and_eval_norm_fast import decode_bin_tokens, trim_runaway_loops
    from stepcad.context import encode_context
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--evalset', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--qwen', default='Qwen/Qwen2.5-Coder-1.5B')
    ap.add_argument('--n0', type=int, default=64, help='samples at step 0 (single shot)')
    ap.add_argument('--n', type=int, default=32, help='samples per re-plan')
    ap.add_argument('--t_max', type=int, default=8, help='max committed actions (re-plans)')
    ap.add_argument('--temperature', type=float, default=0.5)
    ap.add_argument('--top_p', type=float, default=0.95)
    ap.add_argument('--max_new', type=int, default=3000)
    ap.add_argument('--gen_chunk', type=int, default=32)
    ap.add_argument('--exec_timeout', type=float, default=60)
    ap.add_argument('--cpus', type=int, default=0)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--greedy', action='store_true', help='the policy alone: one greedy sample per state (no best-of-N)')
    ap.add_argument('--bn_batch', action='store_true',
                    help='Point-MAE BatchNorm on batch statistics, as in training at batch 1 (no running-stat update)')
    a = ap.parse_args()

    dist.init_process_group('nccl')
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local)
    dev = torch.device('cuda', local)
    torch.manual_seed(a.seed + rank)
    rng = np.random.default_rng(a.seed + rank)
    work = os.path.join(a.out, 'targets')
    os.makedirs(work, exist_ok=True)

    model = build_pc2cad(qwen_model=a.qwen, num_points=1024, pc_encoder_kind='pointmae', num_latents=64,
                         full_finetune=True).to(dev).eval()
    ck = torch.load(a.ckpt, map_location='cpu', weights_only=False)
    model.load_state_dict(ck['model'] if 'model' in ck else ck, strict=False)
    tok, nt = model.tokenizer, model.tokens_per_pc
    if a.bn_batch:
        for m in model.pc_encoder.modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                m.train(); m.momentum = 0.0          # batch stats, running stats untouched

    def sub1024(c):
        idx = rng.choice(len(c), 1024, replace=len(c) < 1024)
        return torch.from_numpy(np.ascontiguousarray(c[idx, :3])).float().to(dev)

    @torch.no_grad()
    def sample(tc, cc, prefix, n):
        """n decoded remainders for one state (prefix '' = step 0)."""
        ids = encode_context(tok, nt, model.pc_pad_id, model.geom_pad_id, prefix)
        out = []
        with torch.autocast('cuda', dtype=torch.bfloat16):
            pcs = torch.stack([tc] + ([cc] if prefix else []))
            e = model.pc_encoder(pcs)
            pe = e.reshape(1, -1, e.shape[-1])                       # [1, nt or 2nt, H]
            inp = torch.tensor([ids], device=dev)
            emb = model.llm.get_input_embeddings()(inp)
            pos = (inp[0] == model.pc_pad_id).nonzero(as_tuple=True)[0]
            emb[0, pos] = pe[0, :len(pos)].to(emb.dtype)
            g = (inp[0] == model.geom_pad_id).nonzero(as_tuple=True)[0][0]
            emb[0, g] = model.geom_encoder(torch.zeros(1, 4, device=dev))[0, 0].to(emb.dtype)
            left, chunk = n, a.gen_chunk
            while left > 0:
                k = min(chunk, left)
                try:
                    kw = dict(do_sample=False, num_return_sequences=1) if a.greedy else \
                        dict(do_sample=True, temperature=a.temperature, top_p=a.top_p, num_return_sequences=k)
                    gen = model.llm.generate(inputs_embeds=emb, attention_mask=torch.ones_like(inp),
                                             max_new_tokens=a.max_new, repetition_penalty=1.1,
                                             pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id, **kw)
                    if a.greedy:
                        k = left                     # one deterministic sample covers the request
                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if chunk == 1:
                        break
                    chunk = max(1, chunk // 2)
                    continue
                for row in gen.cpu().tolist():
                    if tok.eos_token_id in row:
                        row = row[:row.index(tok.eos_token_id)]
                    out.append(trim_runaway_loops(decode_bin_tokens(tok.decode(row, skip_special_tokens=False))))
                left -= k
        return out

    t = pq.read_table(a.evalset)
    part = os.environ.get('INFER_PART', '')          # "i/n": this box takes rows i, i+n, ... (split a big set over boxes)
    if part:
        i, n = (int(x) for x in part.split('/'))
        t = t.take(list(range(i, t.num_rows, n)))
    rows = t.to_pylist()
    if a.limit:
        rows = rows[:a.limit]
    mine = rows[rank::world]
    pool = ProcessPool(max_workers=a.cpus or max(4, (os.cpu_count() - 2 * world) // world), initializer=_worker_init)
    path = os.path.join(a.out, 'rank%d.jsonl' % rank)
    done = set()
    if os.path.exists(path):
        done = {json.loads(l)['key'] for l in open(path)}
    outp = open(path, 'a')

    def run(jobs, cloud=False):
        futs = [pool.schedule(exec_and_score, args=(j, tpath), kwargs={'cloud': cloud, 'seed': len(j)},
                              timeout=a.exec_timeout) for j in jobs]
        res = []
        for f in futs:
            try:
                res.append(f.result())
            except Exception as exc:
                res.append({'ok': False, 'err': type(exc).__name__})
        return res

    for r in mine:
        key = '%s__%s' % (r['group'], r['file_id'])
        if key in done:
            continue
        t0 = time.time()
        tpath, tcloud = target_cloud(r['stl'], work, key)
        tc = sub1024(tcloud)
        actions, cc = [], None
        best_iou, best_prog, plan = -1.0, [], None     # plan = remainder of the best program
        trace, iou_t0 = [], None
        for t in range(a.t_max + 1):
            prefix = '' if t == 0 else HEADER + ''.join(x + '\n' for x in actions)
            rests = [split_actions(x) for x in sample(tc, cc, prefix, a.n0 if t == 0 else a.n)]
            cands = list({tuple(x): x for x in rests if x}.values())
            res = run([actions + c for c in cands])
            scored = [(x['iou'], c) for x, c in zip(res, cands) if x.get('ok')]
            if plan is not None:
                scored.append((best_iou, plan))                 # carry-over
            trace.append({'t': t, 'n_cand': len(cands), 'n_valid': len(scored) - (plan is not None),
                          'step_best': round(max([s for s, _ in scored], default=0.0), 4),
                          'seconds': round(time.time() - t0, 1)})
            if not scored:
                break
            s, c = max(scored, key=lambda z: z[0])
            if s > best_iou:
                best_iou, best_prog = s, actions + c
            trace[-1]['carried'] = c is plan
            if t == 0:
                iou_t0 = s
            if len(c) <= 1 or t == a.t_max:
                break
            # commit the first action of the best plan, re-plan from there
            st = run([actions + [c[0]]], cloud=True)[0]
            if not st.get('ok'):
                trace[-1]['commit_err'] = st.get('err', '')[:80]
                break
            actions = actions + [c[0]]
            cc = sub1024(np.frombuffer(st['cloud'], np.float32).reshape(-1, 6))
            plan = c[1:]
        final = {'valid': False, 'iou': 0.0}
        if best_prog:
            try:
                final = pool.schedule(exec_and_score, args=(best_prog, tpath),
                                      kwargs={'fast': False, 'cloud': False}, timeout=600).result()
                final.pop('bbox', None)
            except Exception as exc:
                final = {'valid': False, 'iou': 0.0, 'err': repr(exc)[:80]}
        rec = {'key': key, 'group': r['group'], 'bench': r.get('bench', 'S'), 'file_id': r['file_id'],
               'program': HEADER + '\n'.join(best_prog) + '\nfinal_result = result\n',
               'n_actions': len(best_prog), 'trace': trace, 'final': final,
               'iou_t0': iou_t0, 'iou_fast': best_iou, 'committed': len(actions),
               'seconds': round(time.time() - t0, 1)}
        outp.write(json.dumps(rec) + '\n')
        outp.flush()
    pool.close()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
