"""Point sampling shared by data generation and inference.

Farthest-point sampling in position *and* normal space (weight 0.35 on the
normal): every surface is visited before any is sampled twice, so small
features keep points.  Identical to the sampler used to build the corpus.
"""
import numpy as np


def fps(points, normals=None, n=2048, lam=0.35, seed=0):
    P = np.asarray(points, dtype=np.float32)
    F = P if normals is None else np.concatenate([P, lam * np.asarray(normals, np.float32)], 1)
    if len(F) <= n:
        return P, (np.asarray(normals, np.float32) if normals is not None else None)
    idx = np.zeros(n, dtype=np.int64)
    idx[0] = seed % len(F)
    d = np.linalg.norm(F - F[idx[0]], axis=1)
    for k in range(1, n):
        idx[k] = int(np.argmax(d))
        d = np.minimum(d, np.linalg.norm(F - F[idx[k]], axis=1))
    return P[idx], (np.asarray(normals, np.float32)[idx] if normals is not None else None)


def sample_cloud(mesh, n=2048, dense=30000, seed=0):
    import trimesh
    pts, fid = trimesh.sample.sample_surface(mesh, dense, seed=seed)
    q, qn = fps(pts, np.asarray(mesh.face_normals)[fid], n, seed=seed)
    return q.astype(np.float32), qn
