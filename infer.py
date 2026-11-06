#!/usr/bin/env python3
"""Mesh -> CadQuery program, locally on one GPU.

    python infer.py --input part.stl --out out/ --mode 3min
    python infer.py --input meshes/ --out out/ --mode 10min --gpu 1 --cpus 40

A policy proposes programs (128 independent receding-horizon rollouts on the GPU), the best ones are
refined against the input mesh on the CPU, and the best refined program is written in the canonical
program format.  --mode sets the time budget per shape (policy + refinement):

    1min    60 s  : ~36 s policy, 12 programs refined
    3min   180 s  : ~108 s policy, 12 programs refined
    10min  600 s  : ~108 s policy, 20 programs refined

Output, per input mesh <name>:
    <name>.py    the program (`result` / `final_result` is the solid), in the input's normalised frame:
                 x' = (x - centre) / scale, with the bbox centre and the largest bbox half-extent as scale
    <name>.json  centre and scale (to map the solid back), seconds taken
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(HERE, 'src')
HF_REPO = 'DeCoDELab/StepCAD'
MESH_EXT = ('.stl', '.obj', '.ply', '.off', '.glb', '.gltf')
MODES = {
    '1min': dict(budget=60, policy_frac=0.6, R=12),
    '3min': dict(budget=180, policy_frac=0.6, R=12),
    '10min': dict(budget=600, policy_frac=0.18, R=20),
}


def meshes(inp):
    if os.path.isdir(inp):
        return sorted(p for p in glob.glob(os.path.join(inp, '*')) if p.lower().endswith(MESH_EXT))
    return [inp]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--input', required=True, help='a mesh file or a directory of meshes')
    ap.add_argument('--out', required=True)
    ap.add_argument('--mode', choices=sorted(MODES), default='3min')
    ap.add_argument('--weights', default=None,
                    help='directory with encoders.pt and llm/ (default: ./weights, else downloaded from %s)' % HF_REPO)
    ap.add_argument('--gpu', default='0')
    ap.add_argument('--cpus', type=int, default=min(os.cpu_count(), 40), help='CPU cores to use')
    ap.add_argument('--gpu_mem', type=float, default=0.8, help='fraction of GPU memory for vLLM')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    if a.weights is None:
        a.weights = os.path.join(HERE, 'weights')
        if not os.path.exists(os.path.join(a.weights, 'encoders.pt')):
            from huggingface_hub import snapshot_download
            a.weights = snapshot_download(HF_REPO)
    sys.path.insert(0, PKG)
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    import trimesh
    from stepcad.canon import canonical

    files = meshes(a.input)
    if not files:
        sys.exit('no mesh found at %s' % a.input)
    os.makedirs(a.out, exist_ok=True)
    work = os.path.join(a.out, '.work')
    os.makedirs(work, exist_ok=True)
    rows, frame = [], {}
    for p in files:
        name = os.path.splitext(os.path.basename(p))[0].replace('__', '_')
        if os.path.exists(os.path.join(a.out, name + '.py')):
            continue
        m = trimesh.load(p, force='mesh')
        lo, hi = m.bounds
        frame[name] = {'centre': ((lo + hi) / 2).tolist(), 'scale': float(max(hi - lo) / 2) or 1.0}
        rows.append({'group': 'mesh', 'file_id': name, 'stl': trimesh.exchange.stl.export_stl(m)})
    if rows:
        evalset = os.path.join(work, 'input.parquet')
        pq.write_table(pa.Table.from_pylist(rows), evalset)
        mo = MODES[a.mode]
        cmd = [sys.executable, '-u', '-m', 'stepcad.fast_infer', '--stage', 'both',
               '--ckpt', os.path.join(a.weights, 'encoders.pt'), '--hf_dir', os.path.join(a.weights, 'llm'),
               '--evalset', evalset, '--out', work, '--rollouts', '128', '--n0', '128', '--n', '16', '--t_max', '12',
               '--budget', str(mo['budget']), '--policy_frac', str(mo['policy_frac']), '--R', str(mo['R']),
               '--workers', str(max(4, a.cpus - 2)), '--gpu_mem', str(a.gpu_mem), '--seed', str(a.seed)]
        env = dict(os.environ, PYTHONPATH=PKG + os.pathsep + os.environ.get('PYTHONPATH', ''),
                   CUDA_VISIBLE_DEVICES=a.gpu, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                   POINTMAE_CKPT='/nonexistent', TOKENIZERS_PARALLELISM='false')
        print('%d mesh(es), mode %s (%d s per shape)' % (len(rows), a.mode, mo['budget']), flush=True)
        subprocess.run(cmd, env=env, check=False)
    res = os.path.join(work, 'rank0.jsonl')
    n = 0
    for l in open(res) if os.path.exists(res) else []:
        r = json.loads(l)
        name = r['file_id']
        if name not in frame or not r.get('program'):
            continue
        code = canonical(r['program'], check=True)          # one program format; kept only if the same solid
        with open(os.path.join(a.out, name + '.py'), 'w') as f:
            f.write(code)
        json.dump({**frame[name], 'seconds': r.get('seconds'), 'mode': a.mode},
                  open(os.path.join(a.out, name + '.json'), 'w'), indent=1)
        n += 1
    print('wrote %d program(s) to %s' % (n, a.out))


if __name__ == '__main__':
    main()
