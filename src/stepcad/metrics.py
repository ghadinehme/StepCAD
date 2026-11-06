"""Verified reconstruction metrics: IoU, Chamfer distance, valid-shape rate.

Definitions (as in the paper):

    IoU = V(S_hat ∩ S) / V(S_hat ∪ S)                         (volumetric)
    CD  = 1/2 ( mean_{x in P_hat} min_{y in P} ||x - y||^2
              + mean_{y in P} min_{x in P_hat} ||y - x||^2 )   (squared, L2)
    VSR = fraction of programs that execute and give a valid solid

Protocol choices, all deliberate and all stricter than a re-normalising
evaluator:

  * **Fixed frame.**  The target mesh is normalised once -- bounding-box
    centre to the origin, largest half-extent to 1 -- and the prediction
    is scored *as produced* in that frame.  A program that gets the shape
    right but places or sizes it wrongly is penalised, not forgiven.
  * **IoU, two independent estimators.**  When both meshes are watertight,
    volumes come from exact mesh booleans (manifold3d).  Otherwise -- and
    Objaverse targets often are not watertight -- occupancy is estimated
    by the generalised winding number on a stratified sample of the union
    bounding box (libigl), which is defined for open and self-intersecting
    meshes.  ``selftest`` checks both against closed-form answers and
    against each other.
  * **CD at two sample counts** (8k and 30k by default) with a fixed seed,
    so convergence is visible in every report.
  * **Valid shape** = executes within the time limit, exactly one solid,
    OCCT ``isValid``, positive volume, and a watertight tessellation.

An invalid prediction scores IoU = 0 and CD = None; reports give both the
mean over valid predictions and the mean with invalid counted as 0.
"""
from __future__ import annotations

import math

import numpy as np

CD_POINTS = (8192, 30000)
IOU_MC_POINTS = 1_000_000


# ---------------------------------------------------------------------
#  frames and meshes
# ---------------------------------------------------------------------

def normalise_target(mesh):
    """Return (mesh_in_eval_frame, centre, scale).

    centre = bbox centre, scale = largest half-extent; the mesh is mapped
    to (v - centre) / scale so it spans [-1, 1] on its longest axis.
    """
    import trimesh
    m = mesh.copy()
    lo, hi = m.bounds
    centre = (lo + hi) / 2.0
    scale = float(max(hi - lo) / 2.0) or 1.0
    m.apply_translation(-centre)
    m.apply_scale(1.0 / scale)
    return m, centre, scale


def load_mesh(data, file_type='stl'):
    import io
    import trimesh
    m = trimesh.load(io.BytesIO(data), file_type=file_type, force='mesh',
                     process=True)
    return m


def solid_to_mesh(solid, tol=1e-3, ang=0.1):
    """cq.Workplane / cq.Shape -> trimesh (fine tessellation)."""
    import trimesh
    shp = solid.val() if hasattr(solid, 'val') else solid
    v, t = shp.tessellate(tol, ang)
    V = np.array([[p.x, p.y, p.z] for p in v], dtype=np.float64)
    return trimesh.Trimesh(V, np.asarray(t, dtype=np.int64), process=True)


def valid_solid(solid):
    """(ok, reason) for the VSR criterion."""
    try:
        wp = solid
        solids = wp.solids().vals() if hasattr(wp, 'solids') else [wp]
        if len(solids) != 1:
            return False, 'solids=%d' % len(solids)
        s = solids[0]
        if not s.isValid():
            return False, 'occt invalid'
        if s.Volume() <= 1e-9:
            return False, 'zero volume'
        # validity is a B-rep property.  A watertight *tessellation* used to
        # be required too, but OCCT leaves cracks between spline / loft /
        # revolved-spline faces: 15 of 17 freeform ground-truth models failed
        # it, so every freeform candidate was rejected in search and scoring.
        # Non-watertight meshes are scored by winding-number MC (see iou).
        return True, ''
    except Exception as exc:
        return False, 'error %s' % type(exc).__name__


# ---------------------------------------------------------------------
#  IoU
# ---------------------------------------------------------------------

def _to_manifold(m):
    import manifold3d as m3d
    mesh = m3d.Mesh(vert_properties=np.asarray(m.vertices, np.float32),
                    tri_verts=np.asarray(m.faces, np.uint32))
    return m3d.Manifold(mesh)


def iou_exact(a, b):
    """Exact IoU from mesh booleans; both meshes must be watertight."""
    A, B = _to_manifold(a), _to_manifold(b)
    if A.status().name != 'NoError' or B.status().name != 'NoError':
        raise ValueError('not a manifold')
    vi = (A ^ B).volume()
    va, vb = A.volume(), B.volume()
    vu = va + vb - vi
    return float(vi / vu) if vu > 0 else 0.0


def _inside(mesh, pts):
    """Occupancy of points; winding number when available."""
    try:
        import igl
        V = np.ascontiguousarray(mesh.vertices, np.float64)
        Q = np.ascontiguousarray(pts, np.float64)
        w, last = None, None
        # the binding was renamed across libigl releases
        for name in ('fast_winding_number_for_meshes', 'fast_winding_number',
                     'winding_number'):
            f = getattr(igl, name, None)
            if f is None:
                continue
            for ft in (np.int64, np.int32):
                try:
                    w = f(V, np.ascontiguousarray(mesh.faces, ft), Q)
                    break
                except Exception as exc:     # wrong dtype or signature
                    last = exc
            if w is not None:
                break
        if w is None:
            raise RuntimeError('no usable libigl winding number: %r' % last)
        return np.abs(np.asarray(w).reshape(-1)) > 0.5
    except ImportError:
        if not mesh.is_watertight:
            raise RuntimeError('non-watertight mesh needs libigl')
        return mesh.contains(pts)


def iou_mc(a, b, n=IOU_MC_POINTS, seed=0):
    """Monte-Carlo IoU on a stratified sample of the union bounding box.

    Standard error is about sqrt(IoU (1 - IoU) / n_union); n = 1e6 keeps
    it below 1e-3 for any part occupying a few per cent of its box.
    """
    lo = np.minimum(a.bounds[0], b.bounds[0])
    hi = np.maximum(a.bounds[1], b.bounds[1])
    k = int(round(n ** (1 / 3)))
    rng = np.random.default_rng(seed)
    g = (np.indices((k, k, k)).reshape(3, -1).T + rng.random((k ** 3, 3))) / k
    pts = lo + g * (hi - lo)
    ia, ib = _inside(a, pts), _inside(b, pts)
    u = np.logical_or(ia, ib).sum()
    return float(np.logical_and(ia, ib).sum() / u) if u else 0.0


def iou(pred, target, n=IOU_MC_POINTS):
    """-> (iou, method).  Exact when possible, winding-number MC otherwise."""
    if pred.is_watertight and target.is_watertight:
        try:
            return iou_exact(pred, target), 'exact'
        except Exception:
            pass
    return iou_mc(pred, target, n=n), 'winding_mc'


# ---------------------------------------------------------------------
#  Chamfer distance
# ---------------------------------------------------------------------

def chamfer(pred, target, n=8192, seed=0):
    """Squared-L2 Chamfer distance on n area-uniform surface samples each."""
    import trimesh
    from scipy.spatial import cKDTree
    pa, _ = trimesh.sample.sample_surface(pred, n, seed=seed)
    pb, _ = trimesh.sample.sample_surface(target, n, seed=seed + 1)
    da, _ = cKDTree(pb).query(pa)
    db, _ = cKDTree(pa).query(pb)
    return float(0.5 * (np.mean(da ** 2) + np.mean(db ** 2)))


# ---------------------------------------------------------------------
#  one prediction
# ---------------------------------------------------------------------

def score(pred_solid, target_mesh_eval, cd_points=CD_POINTS):
    """All metrics for one prediction already in the eval frame."""
    ok, why = valid_solid(pred_solid) if pred_solid is not None else (False, 'no solid')
    out = {'valid': ok, 'invalid_reason': why, 'iou': 0.0, 'iou_method': None}
    if not ok:
        for n in cd_points:
            out['cd_%d' % n] = None
        return out
    pm = solid_to_mesh(pred_solid)
    out['iou'], out['iou_method'] = iou(pm, target_mesh_eval)
    for n in cd_points:
        out['cd_%d' % n] = chamfer(pm, target_mesh_eval, n=n)
    return out


def summarise(rows):
    """Mean IoU (valid-only and invalid=0), median+mean CD, VSR."""
    n = len(rows)
    v = [r for r in rows if r['valid']]
    out = {'n': n, 'vsr': 100.0 * len(v) / max(1, n),
           'iou_valid_mean': float(np.mean([r['iou'] for r in v])) if v else 0.0,
           'iou_all_mean': float(np.sum([r['iou'] for r in v]) / max(1, n))}
    for k in [k for k in rows[0] if k.startswith('cd_')] if rows else []:
        xs = [r[k] for r in v if r[k] is not None]
        out[k + '_mean'] = float(np.mean(xs)) if xs else None
        out[k + '_median'] = float(np.median(xs)) if xs else None
    return out


# ---------------------------------------------------------------------
#  self-test: closed-form answers and estimator agreement
# ---------------------------------------------------------------------

def selftest(verbose=True):
    import trimesh
    res = []

    def check(name, got, want, tol):
        ok = abs(got - want) <= tol
        res.append((name, ok, got, want))
        if verbose:
            print('%-48s %s  got %.6f  want %.6f' % (name, 'PASS' if ok else 'FAIL', got, want))

    # two unit cubes offset by d along x: IoU = (1-d)/(1+d)
    for d in (0.0, 0.25, 0.5):
        a = trimesh.creation.box((1, 1, 1))
        b = trimesh.creation.box((1, 1, 1)); b.apply_translation((d, 0, 0))
        want = (1 - d) / (1 + d)
        check('cube offset %.2f exact' % d, iou_exact(a, b), want, 1e-6)
        check('cube offset %.2f winding-MC' % d, iou_mc(a, b, n=400_000), want, 4e-3)
    # nested spheres r=1 and r=0.5: IoU = 1/8 (tessellation error ~1e-3)
    a = trimesh.creation.icosphere(subdivisions=5, radius=1.0)
    b = trimesh.creation.icosphere(subdivisions=5, radius=0.5)
    check('nested spheres exact', iou_exact(a, b), (b.volume / a.volume), 1e-6)
    check('nested spheres exact vs analytic 1/8', iou_exact(a, b), 0.125, 2e-3)
    check('nested spheres winding-MC', iou_mc(a, b, n=400_000), 0.125, 4e-3)
    # disjoint: IoU 0
    b = trimesh.creation.box((1, 1, 1)); b.apply_translation((3, 0, 0))
    check('disjoint exact', iou_exact(trimesh.creation.box((1, 1, 1)), b), 0.0, 1e-9)
    # symmetry
    a = trimesh.creation.box((1, 2, 1)); b = trimesh.creation.cylinder(0.6, 2.0)
    check('symmetry iou(a,b)-iou(b,a)', iou_exact(a, b) - iou_exact(b, a), 0.0, 1e-9)
    # open target with a small hole (1% of the surface removed): the
    # winding number bridges it; a whole missing side would not be bridged
    a = trimesh.creation.box((1, 1, 1)).subdivide().subdivide().subdivide()
    o = a.copy()
    c = o.triangles_center
    hole = np.linalg.norm(c - np.array([0.5, 0.0, 0.0]), axis=1) < 0.06
    o.update_faces(~hole); o.remove_unreferenced_vertices()
    try:
        check('open-box (1%% hole, watertight=%s) winding-MC' % o.is_watertight,
              iou_mc(a, o, n=200_000), 1.0, 1e-2)
    except RuntimeError as exc:
        res.append(('open-box winding-MC', None, 0, 1))
        if verbose:
            print('%-48s SKIP  (%s)' % ('open-box winding-MC', exc))
    # Chamfer: identical -> 0; parallel unit squares offset d -> ~d^2
    # two independent samples of one surface: CD is the sampling noise
    # floor, ~ mean squared nearest-neighbour spacing ~ 0.3 * area / n
    a = trimesh.creation.box((1, 1, 1))
    floor8 = 0.3 * a.area / 8192
    check('chamfer identical (8k) ~ noise floor', chamfer(a, a, n=8192), floor8, floor8)
    check('chamfer identical (30k) < 8k floor / 3', chamfer(a, a, n=30000), 0.0, floor8 / 3)
    p = trimesh.Trimesh([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], [[0, 1, 2], [0, 2, 3]])
    q = p.copy(); q.apply_translation((0, 0, 0.1))
    check('chamfer parallel squares d=0.1', chamfer(p, q, n=30000), 0.01, 3e-4)
    # convergence 8k vs 30k on a nontrivial pair
    a = trimesh.creation.box((1, 1.5, 0.5)); b = trimesh.creation.cylinder(0.6, 1.0)
    c8, c30 = chamfer(a, b, n=8192), chamfer(a, b, n=30000)
    check('chamfer convergence |8k-30k|/30k', abs(c8 - c30) / c30, 0.0, 0.05)
    n_fail = sum(1 for r in res if r[1] is False)
    if verbose:
        print('%d checks, %d failed' % (len(res), n_fail))
    return n_fail == 0, res


if __name__ == '__main__':
    import sys
    ok, _ = selftest()
    sys.exit(0 if ok else 1)
