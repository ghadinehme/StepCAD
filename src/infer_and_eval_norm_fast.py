"""Parallel single-shot PC -> full CadQuery script eval.

Speedups over infer_and_eval_norm.py:
  1. CPU-pool parallel exec: all N candidate subprocesses run concurrently
     (plus GT), so the per-sample exec wall-time collapses from
     sum(exec_i) to max(exec_i). Biggest win when several candidates hit
     the timeout wall.
  2. DDP sharding: if launched under torchrun --nproc_per_node=K, each
     rank processes picks[rank::world_size]. Nearly linear scaling up to
     K=8 on this box.
  3. Per-rank CPU pool sized by --cpu_workers (default = num_candidates).
  4. Batched LLM generation via num_return_sequences (unchanged, but now
     it's the actual hot path).

Expected: ~30-50x vs serial single-GPU eval.

Usage (single GPU, parallel exec only):
  python infer_and_eval_norm_fast.py --ckpt ... --num_samples 200 \
         --num_candidates 8 --temperature 0.5 --cpu_workers 16

Usage (DDP 8 GPUs):
  torchrun --nproc_per_node=8 infer_and_eval_norm_fast.py \
      --ckpt ... --num_samples 2000 --num_candidates 8 \
      --temperature 0.5 --cpu_workers 12
"""

import argparse, os, sys, time, csv, re, glob
import numpy as np
import torch
import trimesh

from pc2cad.model_legacy_norm import (
    build_pc2cad, NUM_FLOAT_BINS, make_float_token,
    bin_to_float as _norm_bin_to_float, FLOAT_MAX_VAL,
)
from pc2cad.dataset_legacy_cumul_norm import (
    PC2CADSingleShotNormDataset as PC2CADSingleShotDataset,
)


def bin_to_float(b: int) -> float:
    return _norm_bin_to_float(b)


FLOAT_BIN_RE = re.compile(r"<\|f(\d+)\|>")


def decode_bin_tokens(text: str) -> str:
    def sub(m):
        v = bin_to_float(int(m.group(1)))
        return f"{v:.3f}"
    return FLOAT_BIN_RE.sub(sub, text)


def trim_runaway_loops(code: str, max_consecutive: int = 3) -> str:
    if not code:
        return code
    out, run, rc = [], None, 0
    for ln in code.split("\n"):
        if ln.strip() == (run or "").strip() and ln.strip():
            rc += 1
            if rc <= max_consecutive:
                out.append(ln)
        else:
            run, rc = ln, 1
            out.append(ln)
    return "\n".join(out)


# ---- cadquery subprocess worker -------------------------------------------

# Shared voxel-grid config for IoU.
#   Normalize mesh vertices by (gt_center, gt_half), voxelize at VOXEL_PITCH,
#   quantize voxel centers to integer keys with VOXEL_ORIGIN offset so all
#   meshes share the exact same grid. Compare keys with intersect1d/union1d.
VOXEL_PITCH = 0.03
VOXEL_ORIGIN = -1.3   # accommodates gen meshes up to ~30% outside [-1,1]
VOXEL_K = 90          # grid side in cells (VOXEL_ORIGIN..+VOXEL_ORIGIN * -1)


def _mesh_voxel_keys(mesh_verts, mesh_faces, center, half,
                     pitch=VOXEL_PITCH, origin=VOXEL_ORIGIN, K=VOXEL_K):
    import trimesh
    import numpy as np
    v = (np.asarray(mesh_verts, dtype=np.float64) - center) / half
    m = trimesh.Trimesh(vertices=v, faces=mesh_faces, process=False)
    try:
        vg = m.voxelized(pitch).fill()
    except Exception:
        return np.zeros(0, dtype=np.int64)
    pts = vg.points  # voxel centers
    if pts.size == 0:
        return np.zeros(0, dtype=np.int64)
    idx = np.round((pts - origin) / pitch).astype(np.int64)
    mask = ((idx >= 0).all(1) & (idx < K).all(1))
    idx = idx[mask]
    keys = idx[:, 0] * (K * K) + idx[:, 1] * K + idx[:, 2]
    return np.unique(keys).astype(np.int64)


def _persistent_worker(task_q, result_q):
    """Long-lived worker. Imports cadquery once, then processes tasks
    from task_q until a None sentinel is received. Each task is a dict
    with the same keys as the ephemeral worker above. This avoids the
    ~2-3s cadquery+OCP import cost per task.

    If task['ref_verts'] / task['ref_faces'] are provided, the worker
    also computes boolean-IoU (V_I / V_U via trimesh.intersection/.union
    using manifold3d) vs that reference mesh. Falls back to NaN on
    non-manifold failures or timeout."""
    import cadquery as cq
    import math, trimesh, os, signal
    import numpy as np
    from cadquery import exporters
    BOOL_TIMEOUT_S = 2.0
    while True:
        task = task_q.get()
        if task is None:
            return
        try:
            np.random.seed(task["seed"])
            ns = {"cq": cq, "math": math, "__name__": "__main__"}
            exec(task["code"], ns, ns)
            result = ns.get("result")
            if result is None:
                result_q.put(("err", "no result")); continue
            exporters.export(result, task["stl_path"],
                             tolerance=task["mesh_tol"])
            m = trimesh.load(task["stl_path"], force="mesh")
            if m is None or len(m.vertices) == 0:
                result_q.put(("err", "empty mesh")); continue
            pts, _ = trimesh.sample.sample_surface(m, task["num_pts"])
            out = {"pts": np.asarray(pts, dtype=np.float32)}
            if task["compute_voxels"]:
                out["vox_keys"] = _mesh_voxel_keys(
                    m.vertices, m.faces,
                    task["gt_center"], task["gt_half"])
            rv = task.get("ref_verts"); rf = task.get("ref_faces")
            if rv is not None and rf is not None:
                # Normalize gen mesh with same (center, half) transform as
                # the ref so they share the [-1,1] frame.
                c = np.asarray(task["gt_center"], dtype=np.float64)
                h = float(task["gt_half"])
                v_gen = (np.asarray(m.vertices, dtype=np.float64) - c) / h
                m_gen_n = trimesh.Trimesh(
                    vertices=v_gen, faces=m.faces, process=False)
                m_ref = trimesh.Trimesh(
                    vertices=rv, faces=rf, process=False)
                class _T(Exception): pass
                def _h(*_): raise _T()
                old = signal.signal(signal.SIGALRM, _h)
                signal.setitimer(signal.ITIMER_REAL, BOOL_TIMEOUT_S)
                try:
                    inter = m_ref.intersection(m_gen_n)
                    uni = m_ref.union(m_gen_n)
                    if inter is not None and uni is not None \
                            and float(uni.volume) > 0:
                        out["bool_iou"] = (max(0.0, float(inter.volume))
                                           / float(uni.volume))
                    else:
                        out["bool_iou"] = float("nan")
                except BaseException:
                    out["bool_iou"] = float("nan")
                finally:
                    signal.setitimer(signal.ITIMER_REAL, 0)
                    signal.signal(signal.SIGALRM, old)
            result_q.put(("ok", out))
        except BaseException as e:
            result_q.put(("err", f"{type(e).__name__}: {str(e)[:200]}"))


class PersistentPool:
    """Pool of long-lived cadquery worker procs. Each worker imports
    cadquery once at startup, then loops taking tasks from its queue.
    Per-task timeout is enforced by the main process: if a worker
    exceeds its deadline, we SIGKILL and respawn it.

    submit_batch(tasks) returns a list-of-same-length (status, payload)
    with per-task timeout honored.
    """

    def __init__(self, size, timeout_s):
        import multiprocessing as mp
        self.ctx = mp.get_context("spawn")
        self.size = size
        self.timeout_s = timeout_s
        self.workers = []
        for _ in range(size):
            self._spawn_one()

    def _spawn_one(self):
        task_q = self.ctx.Queue()
        result_q = self.ctx.Queue()
        p = self.ctx.Process(target=_persistent_worker,
                             args=(task_q, result_q))
        p.start()
        self.workers.append({
            "proc": p, "task_q": task_q, "result_q": result_q,
            "busy": False, "deadline": None, "task_idx": None,
        })

    def _replace(self, wi):
        w = self.workers[wi]
        try:
            w["proc"].terminate(); w["proc"].join(1)
            if w["proc"].is_alive():
                w["proc"].kill(); w["proc"].join(1)
        except Exception:
            pass
        task_q = self.ctx.Queue()
        result_q = self.ctx.Queue()
        p = self.ctx.Process(target=_persistent_worker,
                             args=(task_q, result_q))
        p.start()
        self.workers[wi] = {
            "proc": p, "task_q": task_q, "result_q": result_q,
            "busy": False, "deadline": None, "task_idx": None,
        }

    def submit_batch(self, tasks):
        import queue as _queue
        N = len(tasks)
        results = [None] * N
        pending = list(range(N))
        # Assign as many as possible to free workers.
        def assign():
            for wi, w in enumerate(self.workers):
                if not w["busy"] and pending:
                    ti = pending.pop(0)
                    t = dict(tasks[ti])
                    t["seed"] = hash(t.get("label", str(ti))) & 0xffffffff
                    w["task_q"].put(t)
                    w["busy"] = True
                    w["deadline"] = time.monotonic() + self.timeout_s
                    w["task_idx"] = ti
        assign()
        while any(w["busy"] for w in self.workers):
            time.sleep(0.02)
            now = time.monotonic()
            for wi, w in enumerate(self.workers):
                if not w["busy"]:
                    continue
                try:
                    status, payload = w["result_q"].get_nowait()
                    results[w["task_idx"]] = (status, payload)
                    w["busy"] = False
                    w["deadline"] = None
                    w["task_idx"] = None
                except _queue.Empty:
                    if now > w["deadline"]:
                        results[w["task_idx"]] = (
                            "err", f"TIMEOUT {self.timeout_s}s")
                        self._replace(wi)
            assign()
        return results

    def close(self):
        for w in self.workers:
            try:
                w["task_q"].put(None)
            except Exception:
                pass
        for w in self.workers:
            try:
                w["proc"].join(2)
                if w["proc"].is_alive():
                    w["proc"].terminate(); w["proc"].join(1)
                if w["proc"].is_alive():
                    w["proc"].kill()
            except Exception:
                pass


def chamfer(a, b):
    if a is None or b is None or len(a) == 0 or len(b) == 0:
        return float("nan")
    a_t = torch.from_numpy(np.asarray(a, dtype=np.float32)).cuda()
    b_t = torch.from_numpy(np.asarray(b, dtype=np.float32)).cuda()
    d = torch.cdist(a_t.unsqueeze(0), b_t.unsqueeze(0))
    return float((d.min(-1).values.mean() + d.min(-2).values.mean()) / 2)


# ---- main ------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--parquet_dir", default="/opt/dlami/nvme/parquets")
    p.add_argument("--val_folders", default="0090,0091,0092,0093,0094")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--num_samples", type=int, default=100)
    p.add_argument("--num_candidates", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--max_new_tokens", type=int, default=3000)
    p.add_argument("--num_input_points", type=int, default=1024)
    p.add_argument("--num_sample_pts", type=int, default=2048)
    p.add_argument("--mesh_tol", type=float, default=0.02)
    p.add_argument("--timeout_s", type=int, default=8)
    p.add_argument("--qwen_model", default="Qwen/Qwen2.5-Coder-1.5B")
    p.add_argument("--num_bands", type=int, default=64)
    p.add_argument("--max_freq", type=float, default=10.0)
    p.add_argument("--pc_encoder", default="pointmae",
                   choices=["baseline", "perceiver", "pointmae"])
    p.add_argument("--num_latents", type=int, default=64)
    p.add_argument("--cpu_workers", type=int, default=0,
                   help="Persistent worker procs per rank. Default 0 => "
                        "(sample_batch * (num_candidates + 1)). Capped at "
                        "24 to avoid thrashing.")
    p.add_argument("--sample_batch", type=int, default=4,
                   help="Number of samples batched into a single LLM "
                        "generate call per rank. Larger = better GPU "
                        "utilization; memory scales linearly.")
    p.add_argument("--compute_iou", action="store_true",
                   help="Compute voxel-IoU (union/intersection) between each "
                        "candidate and the GT mesh in normalized [-1,1]^3 "
                        "space at pitch 0.03.")
    p.add_argument("--save_artifacts", action="store_true",
                   help="Save .py/.stl/.ply for every candidate. Off by "
                        "default for thousand-sample runs to keep I/O sane.")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def init_dist():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        import torch.distributed as dist
        from datetime import timedelta
        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        lr = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(lr)
        # 6-hour timeout: previously the default 10 min watchdog triggered
        # SIGABRT on the fast ranks while slow ranks were still running
        # cadquery/IoU on their tail meshes (Phase 9/11 mode), losing 10-25
        # rows per eval. Long timeout + the per-rank done-marker protocol
        # in the eval scripts keep the run intact even with stragglers.
        dist.init_process_group(backend="nccl",
                                timeout=timedelta(hours=6))
        return True, rank, world, lr
    return False, 0, 1, 0


def main():
    args = parse_args()
    is_ddp, rank, world, lr = init_dist()
    device = torch.device(f"cuda:{lr}")
    is_main = rank == 0
    if is_main:
        os.makedirs(args.out_dir, exist_ok=True)
    if is_ddp:
        import torch.distributed as dist
        dist.barrier()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model = build_pc2cad(
        qwen_model=args.qwen_model,
        num_points=args.num_input_points,
        num_bands=args.num_bands, max_freq=args.max_freq,
        full_finetune=True,
        pc_encoder_kind=args.pc_encoder,
        num_latents=args.num_latents,
    ).to(device).eval()
    if is_main:
        print(f"[rank {rank}] Loading ckpt: {args.ckpt}", flush=True)
    ck = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ck["model"], strict=False)
    if is_main:
        print(f"  step {ck.get('step','?')}  "
              f"best_val {ck.get('best_val','?')}", flush=True)
    tok = model.tokenizer
    eos_id = tok.eos_token_id
    pad_id = tok.pad_token_id

    val_folders = args.val_folders.split(",")
    ds = PC2CADSingleShotDataset(
        args.parquet_dir, val_folders,
        num_points=args.num_input_points,
        max_target_floats=500)

    rng = np.random.RandomState(args.seed)
    picks = rng.choice(len(ds), size=min(args.num_samples, len(ds)),
                       replace=False)
    picks.sort()
    my_picks = [int(x) for i, x in enumerate(picks) if i % world == rank]
    if is_main:
        print(f"[rank {rank}] {len(my_picks)} of {len(picks)} samples "
              f"(world={world})", flush=True)

    prompt_text = "# Generate the full CadQuery script for this shape:\n"
    prompt_ids = tok.encode(prompt_text, add_special_tokens=False)
    sample_batch = max(1, args.sample_batch)
    N_cand = args.num_candidates
    # Default pool size scales with in-flight tasks from one generation
    # batch: sample_batch * (N_cand + 1). Cap at 24 to avoid CPU thrash.
    default_workers = min(24, sample_batch * (N_cand + 1))
    cpu_workers = args.cpu_workers or default_workers
    if is_main:
        print(f"[rank {rank}] sample_batch={sample_batch} "
              f"num_candidates={N_cand} pool_size={cpu_workers}", flush=True)

    pool = PersistentPool(size=cpu_workers, timeout_s=args.timeout_s)
    if is_main:
        print(f"[rank {rank}] persistent pool ready "
              f"({cpu_workers} workers)", flush=True)

    rows = []
    fails = []
    t0 = time.time()
    per_rank_csv = os.path.join(args.out_dir, f"rows_r{rank}.csv")
    per_rank_f = open(per_rank_csv, "w", newline="")
    per_rank_w = None

    def norm_fn(pts, gt_center, gt_half):
        return (pts - gt_center) / gt_half

    try:
        for batch_start in range(0, len(my_picks), sample_batch):
            batch_idx = my_picks[batch_start:batch_start + sample_batch]
            B = len(batch_idx)

            # Load all samples in batch.
            samples = [ds[int(i)] for i in batch_idx]
            gt_codes = [s["code"] for s in samples]
            pcs = torch.stack([s["pointcloud"] for s in samples]).to(device)
            geoms = torch.stack([s["geom_cond"] for s in samples]).to(device)

            # Per-sample normalization + reference point cloud.
            gt_centers, gt_halves, pc_refs, cd_noises = [], [], [], []
            for bi in range(B):
                raw_full = samples[bi]["raw_pointcloud_full"].cpu().numpy()
                c = (raw_full.min(0) + raw_full.max(0)) / 2.0
                h = max(float(np.abs(raw_full - c).max()), 1e-8)
                raw_n = (raw_full - c) / h
                K = args.num_sample_pts
                rng_i = np.random.RandomState(args.seed + int(batch_idx[bi]))
                id_ref = rng_i.choice(raw_n.shape[0], K, replace=(K > raw_n.shape[0]))
                id_noi = rng_i.choice(raw_n.shape[0], K, replace=(K > raw_n.shape[0]))
                pc_ref = raw_n[id_ref]
                cd_noises.append(chamfer(pc_ref, raw_n[id_noi]))
                gt_centers.append(c); gt_halves.append(h); pc_refs.append(pc_ref)

            # Batched LLM generation: embeds shape (B, L_ctx, H),
            # num_return_sequences=N_cand -> output (B*N_cand, L_out).
            nt = model.tokens_per_pc
            ctx_ids = ([model.pc_pad_id] * nt
                       + [model.geom_pad_id] + prompt_ids)
            input_ids = torch.tensor([ctx_ids] * B, dtype=torch.long,
                                     device=device)
            attn = torch.ones_like(input_ids)

            with torch.no_grad(), torch.amp.autocast(
                    "cuda", dtype=torch.bfloat16):
                pc_emb = model.pc_encoder(pcs)          # (B, nt, H)
                geom_emb = model.geom_encoder(geoms)    # (B, 1, H)
                embeds = model.llm.get_input_embeddings()(input_ids)
                dt = embeds.dtype
                embeds[:, :nt] = pc_emb.to(dt)
                embeds[:, nt] = geom_emb[:, 0].to(dt)
                gen_kwargs = dict(
                    inputs_embeds=embeds,
                    attention_mask=attn,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=pad_id,
                    eos_token_id=eos_id,
                    repetition_penalty=1.1,
                    # no_repeat_ngram_size removed: O(L^2) CPU cost per token.
                    # trim_runaway_loops post-process catches line loops.
                )
                if N_cand == 1 or args.temperature <= 0:
                    gen_kwargs.update(do_sample=False, num_beams=1)
                    num_ret = 1
                else:
                    gen_kwargs.update(
                        do_sample=True, temperature=args.temperature,
                        top_p=args.top_p, num_return_sequences=N_cand)
                    num_ret = N_cand
                gens = model.llm.generate(**gen_kwargs)
                # gens shape: (B*num_ret, L_out).
            gens_cpu = gens.cpu()
            del gens, embeds, pc_emb, geom_emb

            # Decode all candidates for all B samples.
            all_codes = [[] for _ in range(B)]
            for idx_seq in range(B * num_ret):
                bi = idx_seq // num_ret
                ids = gens_cpu[idx_seq].tolist()
                if eos_id in ids:
                    ids = ids[:ids.index(eos_id)]
                raw = tok.decode(ids, skip_special_tokens=False)
                code = trim_runaway_loops(decode_bin_tokens(raw),
                                          max_consecutive=3)
                all_codes[bi].append(code)

            # Build tasks: B * (num_ret gen + 1 gt).
            tmp_dir = f"/tmp/_eval_r{rank}"
            os.makedirs(tmp_dir, exist_ok=True)
            tasks = []
            task_map = []   # (sample_bi, ci or -1 for GT)
            stems = []
            for bi in range(B):
                si_local = batch_start + bi
                stem = f"sample_r{rank}_{si_local:04d}"
                stems.append(stem)
                for ci, code in enumerate(all_codes[bi]):
                    stl = (os.path.join(args.out_dir,
                                        f"{stem}_c{ci}_gen.stl")
                           if args.save_artifacts
                           else os.path.join(tmp_dir, f"b{bi}_c{ci}.stl"))
                    tasks.append({
                        "code": code, "stl_path": stl,
                        "mesh_tol": args.mesh_tol,
                        "num_pts": args.num_sample_pts,
                        "gt_center": gt_centers[bi],
                        "gt_half": gt_halves[bi],
                        "compute_voxels": args.compute_iou,
                        "label": f"{stem}_c{ci}",
                    })
                    task_map.append((bi, ci))
                gt_stl = (os.path.join(args.out_dir, f"{stem}_gt.stl")
                          if args.save_artifacts
                          else os.path.join(tmp_dir, f"b{bi}_gt.stl"))
                tasks.append({
                    "code": gt_codes[bi], "stl_path": gt_stl,
                    "mesh_tol": args.mesh_tol,
                    "num_pts": args.num_sample_pts,
                    "gt_center": gt_centers[bi], "gt_half": gt_halves[bi],
                    "compute_voxels": args.compute_iou,
                    "label": f"{stem}_gt",
                })
                task_map.append((bi, -1))

            results = pool.submit_batch(tasks)

            # Demux by sample: per-sample lists of (status, payload) for
            # candidates, plus a GT entry.
            per_sample = [{"gen": [None] * num_ret, "gt": None}
                          for _ in range(B)]
            for ri, (bi, ci) in enumerate(task_map):
                if ci == -1:
                    per_sample[bi]["gt"] = results[ri]
                else:
                    per_sample[bi]["gen"][ci] = results[ri]

            for bi in range(B):
                stem = stems[bi]
                gt_center = gt_centers[bi]; gt_half = gt_halves[bi]
                pc_ref = pc_refs[bi]; cd_noise = cd_noises[bi]
                gt_status, gt_payload = per_sample[bi]["gt"]
                if gt_status == "ok":
                    gt_pts_n = (gt_payload["pts"] - gt_center) / gt_half
                    gt_cd = chamfer(gt_pts_n, pc_ref)
                    gt_vox = gt_payload.get("vox_keys")
                else:
                    gt_cd = float("nan"); gt_vox = None
                    fails.append(("gt_exec", str(gt_payload)))

                best = None; best_cd = float("inf"); best_iou = float("nan")
                ok_count = 0; any_ok = False
                cand_ious = []
                for ci in range(num_ret):
                    status, payload = per_sample[bi]["gen"][ci]
                    if status != "ok":
                        fails.append(("exec", str(payload)))
                        cand_ious.append(float("nan"))
                        continue
                    ok_count += 1; any_ok = True
                    pts_n = (payload["pts"] - gt_center) / gt_half
                    cd = chamfer(pts_n, pc_ref)
                    iou = float("nan")
                    if args.compute_iou and gt_vox is not None:
                        gen_vox = payload.get("vox_keys")
                        if gen_vox is not None and gen_vox.size and gt_vox.size:
                            inter = np.intersect1d(
                                gen_vox, gt_vox, assume_unique=True).size
                            uni = gen_vox.size + gt_vox.size - inter
                            iou = inter / max(uni, 1)
                        elif (gen_vox is not None and gen_vox.size == 0
                              and gt_vox.size == 0):
                            iou = 1.0
                        else:
                            iou = 0.0
                    cand_ious.append(iou)
                    if not np.isnan(cd) and cd < best_cd:
                        best_cd = cd; best = ci; best_iou = iou

                if args.save_artifacts:
                    with open(os.path.join(args.out_dir,
                                           f"{stem}_gt.py"), "w") as f:
                        f.write(gt_codes[bi] if gt_codes[bi].endswith("\n")
                                else gt_codes[bi] + "\n")
                    if best is not None:
                        with open(os.path.join(
                                args.out_dir, f"{stem}_gen.py"), "w") as f:
                            code = all_codes[bi][best]
                            f.write(code if code.endswith("\n")
                                    else code + "\n")
                    for ci, code in enumerate(all_codes[bi]):
                        with open(os.path.join(
                                args.out_dir,
                                f"{stem}_c{ci}_gen.py"), "w") as f:
                            f.write(code if code.endswith("\n")
                                    else code + "\n")
                    try:
                        trimesh.PointCloud(pc_ref).export(
                            os.path.join(args.out_dir, f"{stem}_pc.ply"))
                    except Exception:
                        pass

                best_iou_any = (
                    max([v for v in cand_ious if not np.isnan(v)],
                        default=float("nan"))
                    if args.compute_iou else float("nan"))
                row = {
                    "sample": stem,
                    "pq_idx": int(ds.pairs[int(batch_idx[bi])]),
                    "N": num_ret,
                    "best_cd": best_cd if best is not None else float("nan"),
                    "best_iou": best_iou,
                    "max_iou": best_iou_any,
                    "best_ok": best is not None,
                    "any_ok": any_ok, "ok_count": ok_count,
                    "gt_cd": gt_cd, "cd_sampling_noise": cd_noise,
                }
                rows.append(row)
                if per_rank_w is None:
                    per_rank_w = csv.DictWriter(per_rank_f,
                                                fieldnames=list(row.keys()))
                    per_rank_w.writeheader()
                per_rank_w.writerow(row); per_rank_f.flush()

            # Clean tmp STLs if not saving artifacts.
            if not args.save_artifacts:
                for fn in os.listdir(tmp_dir):
                    try: os.unlink(os.path.join(tmp_dir, fn))
                    except Exception: pass

            done = batch_start + B
            if is_main:
                elapsed = time.time() - t0
                rate = done / max(elapsed, 1e-6)
                eta = (len(my_picks) - done) / max(rate, 1e-6)
                last = rows[-1]
                bc = (f"{last['best_cd']:.4f}"
                      if last['best_ok'] else "inf")
                iou_s = (f" iou={last['best_iou']:.3f}"
                         f"/max={last['max_iou']:.3f}"
                         if args.compute_iou else "")
                print(f"[rank {rank}] [{done}/{len(my_picks)}] "
                      f"best={bc}{iou_s} gt={last['gt_cd']:.4f} "
                      f"noise={last['cd_sampling_noise']:.4f} "
                      f"ok={last['ok_count']}/{num_ret} "
                      f"elapsed={elapsed:.0f}s eta={eta:.0f}s",
                      flush=True)
    finally:
        pool.close()
        try: per_rank_f.close()
        except Exception: pass

    if is_ddp:
        import torch.distributed as dist
        try:
            dist.barrier()
        except Exception as e:
            print(f"[rank {rank}] barrier failed: {e}", flush=True)

    if not is_main:
        return

    # Aggregate from per-rank CSVs on disk (robust to NCCL hangs).
    rows = []
    for r_i in range(world):
        p = os.path.join(args.out_dir, f"rows_r{r_i}.csv")
        if not os.path.exists(p):
            continue
        for row in csv.DictReader(open(p)):
            row["N"] = int(row["N"])
            row["pq_idx"] = int(row.get("pq_idx", 0))
            for k in ("best_cd", "best_iou", "max_iou",
                      "gt_cd", "cd_sampling_noise"):
                v = row.get(k, "")
                row[k] = float(v) if v not in ("", "nan") else float("nan")
            for k in ("best_ok", "any_ok"):
                row[k] = row[k].lower() in ("true", "1")
            row["ok_count"] = int(row["ok_count"])
            rows.append(row)

    csv_path = os.path.join(args.out_dir, "cd_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    def stat(col, subset):
        v = [r[col] for r in subset if not np.isnan(r[col])]
        if not v: return "-"
        return (f"mean={np.mean(v):.4f} median={np.median(v):.4f} "
                f"p25={np.percentile(v,25):.4f} p75={np.percentile(v,75):.4f} "
                f"p90={np.percentile(v,90):.4f}")
    ok = [r for r in rows if r["best_ok"]]
    print(f"\n=== Single-shot CD eval ({len(rows)} samples, "
          f"N={args.num_candidates}, world={world}) ===")
    print(f"  any-ok rate: {sum(r['any_ok'] for r in rows)}/{len(rows)}")
    print(f"  best_cd (pred->ref)  : {stat('best_cd', ok)}")
    print(f"  gt_cd   (ceiling)    : {stat('gt_cd', rows)}")
    print(f"  cd_sampling_noise    : {stat('cd_sampling_noise', rows)}")
    if args.compute_iou:
        print(f"  best_iou (best-CD pick): {stat('best_iou', ok)}")
        print(f"  max_iou  (any cand)    : {stat('max_iou', ok)}")
    print(f"  wall time: {time.time()-t0:.0f}s")
    print(f"  csv: {csv_path}")

    from collections import Counter
    agg = {}
    for stage, msg in fails:
        agg.setdefault(stage, Counter())[msg] += 1
    print(f"\n  Failures: total={len(fails)}")
    for stage, c in agg.items():
        print(f"    {stage}: {sum(c.values())}")
        for msg, n in c.most_common(5):
            print(f"       [{n:>3}] {msg}")


if __name__ == "__main__":
    main()
