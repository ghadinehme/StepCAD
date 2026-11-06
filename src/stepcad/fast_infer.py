"""Latency-bounded receding-horizon inference + refinement (one shape at a time).

Same policy loop as receding_infer.py, engineered for wall-clock time per shape:
  * generation runs on vLLM (prompt embeddings: point tokens + geometry token + code prefix,
    built by the PC2CAD encoders on the same GPU), all samples of a state in one request;
  * candidate programs execute in a process pool with a short timeout;
  * re-planning stops when the policy share of the time budget is spent;
  * refinement runs on the top-R distinct programs in parallel (one process each, aligned
    objective: alignment-invariant volumetric overlap with the input mesh), and the best one is returned.

Driven by ../../infer.py, which sets the time budget for the chosen mode.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HEADER = 'import cadquery as cq\nimport math\n'


def _refine_job(args):
    """Aligned-objective refinement of one seed program (runs in its own process)."""
    acts, target_path, budget_s, seed = args
    import time
    import trimesh
    from stepcad.refine import refine, Scorer
    tm = trimesh.load(target_path, force='mesh')
    if budget_s <= 1:
        s = Scorer(tm)(acts)
        return s, acts, 0
    # refine() stops after max_passes even with budget left; chained restarts from the best program
    # kept improving the 700-set score (0.730 -> 0.742 -> 0.746), so restart until the budget is spent
    t_end, evals, best, stall, k = time.time() + budget_s, 0, None, 0, 0
    while True:
        left = t_end - time.time()
        if left < 5 or stall >= 2:
            break
        new, s, ev = refine(acts, tm, K=2, rng_seed=seed + 101 * k, budget_s=left)
        evals += ev
        k += 1
        if best is None or s > best[0] + 1e-4:
            best, stall, acts = (s, new), 0, new
        else:
            stall += 1
    return (best[0], best[1], evals) if best else (Scorer(tm)(acts), acts, evals)


def _refine_main(job_path, out_path, mem_gb=0):
    """Subprocess entry: run one refinement job, write (score, program) to out_path as soon as it ends.
    Progress is also written after every improvement, so a hard kill at the deadline keeps the best so far."""
    import pickle, threading
    acts, target_path, budget_s, seed = pickle.load(open(job_path, 'rb'))
    if mem_gb:                       # RSS watchdog (RLIMIT_AS breaks the CAD libraries' imports)
        def watch():
            page = os.sysconf('SC_PAGE_SIZE')
            while True:
                try:
                    if int(open('/proc/self/statm').read().split()[1]) * page > mem_gb << 30:
                        os._exit(3)  # best program so far is already on disk
                except Exception:
                    pass
                time.sleep(0.5)
        threading.Thread(target=watch, daemon=True).start()

    def save(sc, prog):
        tmp = out_path + '.tmp'
        pickle.dump((sc, prog), open(tmp, 'wb'))
        os.replace(tmp, out_path)
    import time, trimesh
    from stepcad.refine import refine, Scorer
    tm = trimesh.load(target_path, force='mesh')
    try:
        save(Scorer(tm)(acts), acts)
    except Exception:
        pass
    if budget_s <= 1:
        return
    t_end, best, stall, k = time.time() + budget_s, None, 0, 0
    while True:
        left = t_end - time.time()
        if left < 5 or stall >= 2:
            break
        new, sc, _ = refine(acts, tm, K=2, rng_seed=seed + 101 * k, budget_s=left,
                            progress=lambda pr, v: save(v, list(pr)) if best is None or v > best[0] else None)
        k += 1
        if best is None or sc > best[0] + 1e-4:
            best, stall, acts = (sc, new), 0, new
            save(sc, new)
        else:
            stall += 1


def refine_shape(seeds, tpath, left, work, mem_gb):
    """Refine every seed in its own subprocess for `left` seconds (coreutils timeout -s KILL, self-capped
    RSS); returns ((score, program), timing).  Used by the single-box run and by the CPU refinement stage."""
    import pickle, shutil, subprocess, tempfile
    best, tm = (-1.0, []), {}
    if not seeds:
        return best, tm
    t0 = time.time()
    jd = tempfile.mkdtemp(dir=work)
    procs = []
    for i, sd in enumerate(seeds):
        jp, op = os.path.join(jd, 'j%d.pkl' % i), os.path.join(jd, 'o%d.pkl' % i)
        with open(jp, 'wb') as f:
            pickle.dump((sd, tpath, max(1.0, left - 6), i), f)
        procs.append((subprocess.Popen(['timeout', '-s', 'KILL', '%d' % max(2, int(left - 2)), sys.executable, '-c',
                                        'import sys; from stepcad.fast_infer import _refine_main; '
                                        '_refine_main(sys.argv[1], sys.argv[2], %d)' % mem_gb, jp, op],
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL), op))
    tm['spawned'] = time.time() - t0
    deadline = t0 + left + 2.0
    while time.time() < deadline and any(pr.poll() is None for pr, _ in procs):
        time.sleep(0.5)
    tm['waited'] = time.time() - t0
    for pr, _ in procs:
        if pr.poll() is None:
            pr.kill()
    for pr, _ in procs:
        try:
            pr.wait(timeout=5)
        except Exception:
            pass
    for _, op in procs:
        try:
            sc, prog = pickle.load(open(op, 'rb'))
            if sc > best[0]:
                best = (sc, prog)
        except Exception:
            pass
    shutil.rmtree(jd, ignore_errors=True)
    tm['cleaned'] = time.time() - t0
    return best, tm


def refine_stage(a):
    """CPU stage of a split run: read policy-stage seeds, refine `a.shapes` shapes at a time (each with its
    own R subprocesses and its own remaining budget), write final records in the usual format."""
    import pyarrow.parquet as pq
    from concurrent.futures import ThreadPoolExecutor
    from stepcad.rollout import target_cloud
    os.environ['REFINE_OBJ'] = 'aligned'
    os.environ.setdefault('REFINE_SNAP', 'final')
    seeds = {}
    for f in a.seeds:
        for l in open(f):
            r = json.loads(l)
            seeds[r['key']] = r
    stl = {'%s__%s' % (r['group'], r['file_id']): r['stl'] for r in pq.read_table(a.evalset, columns=['group', 'file_id', 'stl']).to_pylist()}
    work = os.path.join(a.out, 'targets')
    os.makedirs(work, exist_ok=True)
    path = os.path.join(a.out, 'rank0.jsonl')
    done = {json.loads(l)['key'] for l in open(path)} if os.path.exists(path) else set()
    todo = [k for k in sorted(seeds) if k in stl]
    if a.part:                                   # split the refinement over several CPU boxes
        i, n = (int(x) for x in a.part.split('/'))
        todo = todo[i::n]
    todo = [k for k in todo if k not in done]
    import threading
    lock = threading.Lock()
    outp = open(path, 'a')

    def one(key):
        r = seeds[key]
        tpath, _ = target_cloud(stl[key], work, key)
        t0 = time.time()
        # one policy stage can feed several versions: --total_budget / --R re-derive this version's share
        left = r['refine_budget'] if not a.total_budget else max(1.0, a.total_budget - r['policy_s'] - 3.0)
        best, tm = refine_shape(r['seeds'][:a.R], tpath, left, work, a.refine_mem_gb)
        if not best[1] and r['seeds']:
            best = (0.0, r['seeds'][0])
        rec = {'key': key, 'group': r['group'], 'bench': r.get('bench', 'S'), 'file_id': r['file_id'],
               'program': HEADER + '\n'.join(best[1]) + '\nfinal_result = result\n' if best[1] else '',
               'aligned': round(best[0], 4), 'policy_best_iou': r.get('policy_best_iou'), 'n_seeds': len(r['seeds']),
               'final': {'valid': bool(best[1]), 'iou': best[0]}, 'trace': r.get('trace', []),
               'policy_s': r['policy_s'], 'refine_s': round(time.time() - t0, 1),
               'seconds': round(r['policy_s'] + time.time() - t0, 1)}
        with lock:
            outp.write(json.dumps(rec) + '\n'); outp.flush()
        print('%s done: refine %.1fs' % (key, time.time() - t0), flush=True)

    with ThreadPoolExecutor(a.shapes) as ex:
        list(ex.map(one, todo))
    outp.close()


def export(a):
    import torch
    from pc2cad.model_legacy_norm import build_pc2cad
    model = build_pc2cad(qwen_model=a.qwen, num_points=1024, pc_encoder_kind='pointmae', num_latents=64,
                         full_finetune=True)
    ck = torch.load(a.ckpt, map_location='cpu', weights_only=False)
    model.load_state_dict(ck['model'] if 'model' in ck else ck, strict=False)
    model.llm.to(torch.bfloat16).save_pretrained(a.hf_dir, safe_serialization=True)
    model.tokenizer.save_pretrained(a.hf_dir)
    print('exported LLM to', a.hf_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--hf_dir', required=True)
    ap.add_argument('--export', action='store_true')
    ap.add_argument('--qwen', default='Qwen/Qwen2.5-Coder-1.5B')
    ap.add_argument('--evalset')
    ap.add_argument('--out')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--part', default='', help='i/n: take rows i, i+n, ...')
    ap.add_argument('--n0', type=int, default=64)
    ap.add_argument('--n', type=int, default=16)
    ap.add_argument('--t_max', type=int, default=12, help='max policy steps (re-plans)')
    ap.add_argument('--temperature', type=float, default=0.5)
    ap.add_argument('--top_p', type=float, default=0.95)
    ap.add_argument('--max_new', type=int, default=3000)
    ap.add_argument('--exec_timeout', type=float, default=5)
    ap.add_argument('--budget', type=float, default=180, help='wall seconds per shape (policy + refinement)')
    ap.add_argument('--policy_frac', type=float, default=0.5)
    ap.add_argument('--R', type=int, default=8, help='programs refined in parallel')
    ap.add_argument('--stage', default='both', choices=['both', 'policy', 'refine'],
                    help='policy: GPU stage writes seeds; refine: CPU stage refines --seeds files')
    ap.add_argument('--seeds', nargs='*', default=[], help='refine stage: policy-stage jsonl files')
    ap.add_argument('--shapes', type=int, default=8, help='refine stage: shapes refined concurrently')
    ap.add_argument('--total_budget', type=float, default=0, help='refine stage: per-shape total budget (policy time included)')
    ap.add_argument('--refine_mem_gb', type=int, default=8, help='resident-memory cap per refinement job')
    ap.add_argument('--one_seed', action='store_true',
                    help='refine only the single best policy program (R parallel chains with different random seeds)')
    ap.add_argument('--rollouts', type=int, default=0,
                    help='K > 0: K independent receding rollouts (1 sample per step each), best-of at the end')
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--gpu_mem', type=float, default=0.6)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    if a.export:
        return export(a)
    if a.stage == 'refine':
        return refine_stage(a)

    os.environ['REFINE_OBJ'] = 'aligned'            # inherited by the refinement workers
    from pebble import ProcessPool
    from stepcad.rollout import _worker_init, exec_and_score, target_cloud
    from stepcad.receding_infer import split_actions
    # pools before any CUDA initialisation (forked workers must not inherit a CUDA context)
    import multiprocessing as mp
    ctx = mp.get_context('spawn')                   # vLLM / CUDA threads make fork unsafe (pebble forks lazily)
    pool = ProcessPool(max_workers=a.workers, initializer=_worker_init, context=ctx)
    os.environ.setdefault('REFINE_SNAP', 'final')  # polish with measured target dimensions

    import torch
    import pyarrow.parquet as pq
    from vllm import LLM, SamplingParams
    from pc2cad.model_legacy_norm import build_pc2cad
    from stepcad.context import encode_context
    from infer_and_eval_norm_fast import decode_bin_tokens, trim_runaway_loops

    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    dev = 'cuda'
    ck = torch.load(a.ckpt, map_location='cpu', weights_only=False)
    if 'pc_encoder' in ck:
        # compact release weights: only the point-cloud / geometry encoders; tokenizer and token
        # embeddings come from the exported LLM directory that vLLM serves
        from transformers import AutoTokenizer
        from safetensors import safe_open
        from pc2cad.model_legacy_norm import GeomCondEncoder
        from pc2cad.pointmae_encoder import PointMAEForLLM
        c = ck['config']
        os.environ['POINTMAE_FINE'] = c.get('fine', '')
        pc_enc = PointMAEForLLM(llm_hidden_dim=c['hidden_dim'], num_group=c['num_group'], group_size=c['group_size'],
                                embed_dim=c['embed_dim'], depth=c['depth'], num_heads=c['num_heads'],
                                mlp_ratio=c['mlp_ratio'], pretrained_ckpt=None, freeze_encoder=False)
        pc_enc.load_state_dict(ck['pc_encoder'])
        geom = GeomCondEncoder(hidden_dim=c['hidden_dim'])
        geom.load_state_dict(ck['geom_encoder'])
        tok = AutoTokenizer.from_pretrained(a.hf_dir)
        nt = pc_enc.tokens_per_pc
        with safe_open(os.path.join(a.hf_dir, 'model.safetensors'), framework='pt') as f:
            E = f.get_tensor('model.embed_tokens.weight').to(dev, torch.bfloat16)
        pc_pad, geom_pad = tok.convert_tokens_to_ids('<|pc_pad|>'), tok.convert_tokens_to_ids('<|geom_pad|>')
        pc_enc, geom = pc_enc.to(dev).eval(), geom.to(dev).eval()
    else:
        model = build_pc2cad(qwen_model=a.qwen, num_points=1024, pc_encoder_kind='pointmae', num_latents=64,
                             full_finetune=True)
        model.load_state_dict(ck['model'] if 'model' in ck else ck, strict=False)
        tok, nt = model.tokenizer, model.tokens_per_pc
        pc_enc = model.pc_encoder.to(dev).eval()
        geom = model.geom_encoder.to(dev).eval()
        E = model.llm.get_input_embeddings().weight.detach().to(dev, torch.bfloat16)
        pc_pad, geom_pad = model.pc_pad_id, model.geom_pad_id
        del model
    del ck
    gvec = geom(torch.zeros(1, 4, device=dev))[0, 0].to(torch.bfloat16)
    llm = LLM(model=a.hf_dir, tokenizer=a.hf_dir, dtype='bfloat16', enable_prompt_embeds=True,
              gpu_memory_utilization=a.gpu_mem, max_model_len=a.max_new + 2600, seed=a.seed)
    print('vLLM ready', flush=True)

    def sub1024(c):
        idx = rng.choice(len(c), 1024, replace=len(c) < 1024)
        return torch.from_numpy(np.ascontiguousarray(c[idx, :3])).float().to(dev)

    @torch.no_grad()
    def sample(tc, cc, prefix, n, max_new):
        ids = encode_context(tok, nt, pc_pad, geom_pad, prefix)
        pcs = torch.stack([tc] + ([cc] if prefix else []))
        with torch.autocast('cuda', dtype=torch.bfloat16):
            pe = pc_enc(pcs).reshape(-1, E.shape[1]).to(torch.bfloat16)
        inp = torch.tensor(ids, device=dev)
        emb = E[inp].clone()
        pos = (inp == pc_pad).nonzero(as_tuple=True)[0]
        emb[pos] = pe[:len(pos)]
        emb[(inp == geom_pad).nonzero(as_tuple=True)[0][0]] = gvec
        sp = SamplingParams(n=n, temperature=a.temperature, top_p=a.top_p, repetition_penalty=1.1,
                            max_tokens=max_new, stop_token_ids=[tok.eos_token_id],
                            seed=int(rng.integers(1 << 30)))
        res = llm.generate([{'prompt_embeds': emb.cpu()}], sp, use_tqdm=False)[0]
        return [trim_runaway_loops(decode_bin_tokens(tok.decode(o.token_ids, skip_special_tokens=False)))
                for o in res.outputs]

    @torch.no_grad()
    def embed(tc, cc, prefix):
        ids = encode_context(tok, nt, pc_pad, geom_pad, prefix)
        pcs = torch.stack([tc] + ([cc] if prefix else []))
        with torch.autocast('cuda', dtype=torch.bfloat16):
            pe = pc_enc(pcs).reshape(-1, E.shape[1]).to(torch.bfloat16)
        inp = torch.tensor(ids, device=dev)
        emb = E[inp].clone()
        emb[(inp == pc_pad).nonzero(as_tuple=True)[0]] = pe[:int((inp == pc_pad).sum())]
        emb[(inp == geom_pad).nonzero(as_tuple=True)[0][0]] = gvec
        return emb.cpu()

    def sample_batch(prompts, max_new):
        """One sampled completion per (target cloud, current cloud, prefix) prompt, all in one vLLM call."""
        sp = SamplingParams(n=1, temperature=a.temperature, top_p=a.top_p, repetition_penalty=1.1,
                            max_tokens=max_new, stop_token_ids=[tok.eos_token_id], seed=int(rng.integers(1 << 30)))
        res = llm.generate([{'prompt_embeds': embed(*p)} for p in prompts], sp, use_tqdm=False)
        return [trim_runaway_loops(decode_bin_tokens(tok.decode(r.outputs[0].token_ids, skip_special_tokens=False)))
                for r in res]

    rows = pq.read_table(a.evalset).to_pylist()
    if a.part:
        i, n = (int(x) for x in a.part.split('/'))
        rows = rows[i::n]
    if a.limit:
        rows = rows[:a.limit]
    os.makedirs(a.out, exist_ok=True)
    work = os.path.join(a.out, 'targets')
    os.makedirs(work, exist_ok=True)
    path = os.path.join(a.out, 'rank0.jsonl')
    done = {json.loads(l)['key'] for l in open(path)} if os.path.exists(path) else set()
    outp = open(path, 'a')

    clock = {'deadline': float('inf'), 'tok_s': None, 'first': True}   # first step always runs in full     # policy-stage deadline and measured decode speed

    def run(jobs, tpath, cloud=False):
        dl = float('inf') if clock['first'] else clock['deadline']
        tmo = max(1.0, min(a.exec_timeout, dl - time.time()))
        futs = [pool.schedule(exec_and_score, args=(j, tpath), kwargs={'cloud': cloud, 'seed': len(j)},
                              timeout=tmo) for j in jobs]
        out = []
        for f in futs:
            try:
                out.append(f.result(timeout=None if dl == float('inf') else max(0.5, dl + 2 - time.time())))
            except Exception as exc:
                f.cancel()
                out.append({'ok': False, 'err': type(exc).__name__})
        return out

    def budget_tokens():
        """Generation length that fits in the time left before the policy deadline (decode speed measured
        on the previous call of this shape); the first call of a shape uses the full max_new."""
        left = clock['deadline'] - time.time()
        if clock['first'] or clock['tok_s'] is None or left == float('inf'):
            return a.max_new
        return int(max(64, min(a.max_new, 0.8 * left * clock['tok_s'])))

    def timed(fn, *args):
        t = time.time()
        outs = fn(*args)
        ntok = max((len(tok.encode(o)) for o in outs), default=0)   # longest sample sets the batch time
        if ntok > 50:
            clock['tok_s'] = ntok / max(time.time() - t, 1e-3)
        return outs

    for r in rows:
        key = '%s__%s' % (r['group'], r['file_id'])
        if key in done:
            continue
        t0 = time.time()
        p_deadline = t0 + a.policy_frac * a.budget
        clock['deadline'], clock['tok_s'], clock['first'] = p_deadline, None, True
        tpath, tcloud = target_cloud(r['stl'], work, key)
        tc = sub1024(tcloud)
        actions, cc, plan, best_iou = [], None, None, -1.0
        seen = {}                                   # full program (tuple) -> internal IoU
        trace = []
        if a.rollouts:
            # K independent rollouts: each trajectory samples its own remainder, commits its first action, re-plans
            trajs = [{'acts': [], 'cc': None, 'done': False} for _ in range(a.rollouts)]
            for t in range(a.t_max + 1):
                live = [tr for tr in trajs if not tr['done']]
                if not live or (t > 0 and time.time() > p_deadline):
                    break
                tg = time.time()
                if t == 0:
                    outs = timed(sample, tc, None, '', len(live), budget_tokens())
                else:
                    outs = timed(sample_batch, [(tc, tr['cc'], HEADER + ''.join(x + '\n' for x in tr['acts'])) for tr in live], budget_tokens())
                tg = time.time() - tg
                te = time.time()
                plans = [split_actions(o) for o in outs]
                res = run([tr['acts'] + p for tr, p in zip(live, plans) if p], tpath)
                k = 0
                for tr, p in zip(live, plans):
                    if not p:
                        tr['done'] = True
                        continue
                    x = res[k]; k += 1
                    if x.get('ok'):
                        seen[tuple(tr['acts'] + p)] = x['iou']
                        best_iou = max(best_iou, x['iou'])
                    if not x.get('ok') or len(p) <= 1 or t == a.t_max:
                        tr['done'] = True
                    else:
                        tr['next'] = p[0]
                step = [tr for tr in live if not tr['done']]
                commits = run([tr['acts'] + [tr['next']] for tr in step], tpath, cloud=True)
                for tr, st in zip(step, commits):
                    if st.get('ok'):
                        tr['acts'] = tr['acts'] + [tr['next']]
                        tr['cc'] = sub1024(np.frombuffer(st['cloud'], np.float32).reshape(-1, 6))
                    else:
                        tr['done'] = True
                trace.append({'t': t, 'n_cand': len(live), 'gen_s': round(tg, 1), 'exec_s': round(time.time() - te, 1)})
                clock['first'] = False
        for t in range(0 if not a.rollouts else 10 ** 9, a.t_max + 1):
            if t > 0 and time.time() > p_deadline:
                break
            prefix = '' if t == 0 else HEADER + ''.join(x + '\n' for x in actions)
            tg = time.time()
            rests = [split_actions(x) for x in timed(sample, tc, cc, prefix, a.n0 if t == 0 else a.n, budget_tokens())]
            tg = time.time() - tg
            cands = list({tuple(x): x for x in rests if x}.values())
            te = time.time()
            res = run([actions + c for c in cands], tpath)
            te = time.time() - te
            scored = [(x['iou'], c) for x, c in zip(res, cands) if x.get('ok')]
            for s, c in scored:
                seen[tuple(actions + c)] = s
            if plan is not None:
                scored.append((best_iou, plan))
            trace.append({'t': t, 'n_cand': len(cands), 'gen_s': round(tg, 1), 'exec_s': round(te, 1)})
            clock['first'] = False
            if not scored:
                break
            s, c = max(scored, key=lambda z: z[0])
            best_iou = max(best_iou, s)
            if len(c) <= 1 or t == a.t_max:
                break
            st = run([actions + [c[0]]], tpath, cloud=True)[0]
            if not st.get('ok'):
                break
            actions = actions + [c[0]]
            cc = sub1024(np.frombuffer(st['cloud'], np.float32).reshape(-1, 6))
            plan = c[1:]
        clock['deadline'] = float('inf')
        t_pol = time.time() - t0
        # refinement: top-R distinct programs, in parallel, until the shape's deadline
        seeds = [list(p) for p, _ in sorted(seen.items(), key=lambda z: -z[1])[:a.R]]
        if a.one_seed and seeds:
            seeds = [seeds[0]] * a.R          # same program, different refinement random seeds
        left = max(0.0, a.budget - (time.time() - t0) - 3.0)
        if a.stage == 'policy':           # split run: hand the seeds to the CPU refinement stage
            rec = {'key': key, 'group': r['group'], 'bench': r.get('bench', 'S'), 'file_id': r['file_id'],
                   'seeds': seeds, 'refine_budget': round(left, 1), 'policy_best_iou': round(best_iou, 4),
                   'trace': trace, 'policy_s': round(t_pol, 1)}
            outp.write(json.dumps(rec) + '\n'); outp.flush()
            print('%s policy %.1fs seeds %d' % (key, t_pol, len(seeds)), flush=True)
            continue
        best, tm = refine_shape(seeds, tpath, left, work, a.refine_mem_gb)
        if tm:
            trace.append({'refine_timing': {k: round(v + (time.time() - t0) - tm['cleaned'], 1) for k, v in tm.items()}})
        if not best[1] and seeds:
            best = (0.0, seeds[0])                  # every refinement missed the deadline: best policy program
        t_tot = time.time() - t0
        rec = {'key': key, 'group': r['group'], 'bench': r.get('bench', 'S'), 'file_id': r['file_id'],
               'program': HEADER + '\n'.join(best[1]) + '\nfinal_result = result\n' if best[1] else '',
               'aligned': round(best[0], 4), 'policy_best_iou': round(best_iou, 4), 'n_seeds': len(seeds),
               'final': {'valid': bool(best[1]), 'iou': best[0]},
               'trace': trace, 'policy_s': round(t_pol, 1), 'seconds': round(t_tot, 1)}
        outp.write(json.dumps(rec) + '\n')
        outp.flush()
        print('%s done: policy %.1fs, total %.1fs' % (key, t_pol, t_tot), flush=True)
    outp.close()
    import subprocess
    subprocess.run(['pkill', '-9', '-P', str(os.getpid())])   # vLLM engine core and pool workers (exact parent PID)
    os._exit(0)          # pebble can hang forever terminating a stuck worker at shutdown; results are flushed


if __name__ == '__main__':
    main()
