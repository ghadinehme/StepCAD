"""Geometry-aware program refinement (beam search over local edits).

Given a seed program p = (a_1..a_T) from the policy and the target mesh m:

    B_0 = {p}
    for t = 1..T:
        C = {p} ∪ { e_j(p, t) : p in B_{t-1}, j = 1..M }      (M = 4 edits)
        B_t = top-K of C by IoU(S(p'), m)                    (K = 2)
    return argmax_{p in B_T} IoU

Edits applied to operation a_t (paper appendix, parameters as specified):

  1. continuous   GP + expected-improvement Bayesian optimisation of a_t's
                  scalar literals in [v - r, v + r], r = min(max(0.2|v|,
                  0.005), 0.3); fillet/chamfer sizes in [0.01, 0.5];
                  6 random samples + 10 iterations x batch 4; kept only
                  if IoU improves
  2. sketch       section the target with a_t's working plane
                  (trimesh.section), largest closed loop, RDP-simplified,
                  fitted as a circle, a line+arc chain, or a polygon;
                  replaces a_t's sketch, keeping operation and depth
  3. skip         remove a_t; kept only if IoU improves
  4. complement   D = S(p) \\ m (manifold3d); planar faces of D clustered
     cut          by normal (2 deg) and offset (1e-3); clusters under
                  0.0025 area dropped; each cluster's loops -> sketch ->
                  cut-extrude, depth grid {0.05..2.0}; best appended;
                  at most two rounds per program

Programs are lists of self-contained actions (``result = result.op(...)``)
as produced by the policy; every candidate is executed from text.
"""
from __future__ import annotations

import math
import os
import re
import time

import numpy as np

HEADER = 'import cadquery as cq\nimport math\n'
DEPTHS = (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60, 0.80, 1.20, 2.00)
_NUM = re.compile(r'(?<![\w.])-?\d+\.?\d*(?:[eE]-?\d+)?')
_PLANE = re.compile(r'cq\.Plane\(origin=\(([^)]*)\),\s*normal=\(([^)]*)\),\s*xDir=\(([^)]*)\)\)')
_OP3D = re.compile(r'\.(extrude|revolve|twistExtrude|cutBlind|cutThruAll)\(')


def _f(v):
    r = round(float(v), 3)
    if r == 0:
        return '0'
    return str(int(r)) if r == int(r) else repr(r)


# ---------------------------------------------------------------------
#  execution + scoring with a cache
# ---------------------------------------------------------------------

def watertight_proxy(mesh):
    """A hole-filled copy of a target that is not watertight, when that closes
    it: complement/add and the exact (manifold) IoU need a closed solid.
    (A voxel remesh was tried and dropped: 91 s per mesh and still open.)
    Search scoring only; final scores use the original target."""
    import trimesh
    if mesh.is_watertight:
        return mesh
    try:
        m = mesh.copy()
        trimesh.repair.fill_holes(m)
        if m.is_watertight:
            return m
    except Exception:
        pass
    return mesh


def _mass_normalise(mesh):
    """Copy translated to its centre of mass and scaled by 1 / radius of gyration (as CADBench align_shapes)."""
    m = mesh.copy()
    c = m.center_mass
    s = float(np.sqrt(abs(np.trace(m.moment_inertia) / m.mass)))
    m.apply_translation(-c)
    m.apply_scale(1.0 / s)
    return m


def _alpha_wrap(mesh):
    from stepcad.rollout import alpha_wrap
    return alpha_wrap(mesh)


class Scorer:
    """IoU of a candidate program against the target, built for speed.

    * prefix cache: actions are self-contained ``result = result.op(...)``
      lines, so the solid after any prefix is reused -- an edit to action t
      re-executes only actions t..T, not the whole program;
    * the target's manifold and volume are built once, not per candidate;
    * search-time tessellation is coarser (final scores are recomputed at
      full resolution by the caller).
    """

    def __init__(self, target_mesh, n_mc=60_000, tol=3e-3, ang=0.2, max_prefix=256):
        from stepcad import metrics as M
        target_mesh = watertight_proxy(target_mesh)
        self.m = target_mesh
        self.n = n_mc
        self.tol, self.ang = tol, ang
        self.cache = {}
        self.prefix = {}
        self.max_prefix = max_prefix
        self.evals = 0
        self.T, self.vt = None, None
        if target_mesh.is_watertight:
            try:
                self.T = M._to_manifold(target_mesh)
                self.vt = self.T.volume()
            except Exception:
                self.T = None
        # REFINE_OBJ=aligned: score like CADBench's Aligned IoU -- both shapes normalised by their own
        # centre of mass and radius of gyration (pose kept), target alpha-wrapped -- so refinement
        # optimises the benchmark's measure instead of fixed-frame IoU.  Edits still use the fixed frame.
        self.Tn = None
        if os.environ.get('REFINE_OBJ') == 'aligned':
            try:
                tw = target_mesh if target_mesh.is_watertight else _alpha_wrap(target_mesh)
                self.Tn = M._to_manifold(_mass_normalise(tw))
                self.vtn = self.Tn.volume()
            except Exception:
                self.Tn = None

    def solid(self, actions):
        """Solid after all actions.  The prefix cache keeps each prefix's whole
        namespace, not just `result`: in the v1 / StepCAD style later actions
        reuse earlier names (plane_k, sketch_k, loop variables), and running
        them with only `result` raised NameError, so every candidate -- the
        seed included -- scored -1 and refinement silently did nothing."""
        import cadquery as cq
        k = len(actions)
        while k > 0 and '\n'.join(actions[:k]) not in self.prefix:
            k -= 1
        ns = dict(self.prefix['\n'.join(actions[:k])]) if k else {'cq': cq, 'math': math, 'result': None}
        for i in range(k, len(actions)):
            exec(actions[i], ns)
            if len(self.prefix) < self.max_prefix:
                self.prefix['\n'.join(actions[:i + 1])] = dict(ns)
            ns = dict(ns)
        return ns['result']

    def _iou(self, mesh):
        from stepcad import metrics as M
        if self.Tn is not None and mesh.is_watertight:
            try:
                A = M._to_manifold(_mass_normalise(mesh))
                vi = (A ^ self.Tn).volume()
                vu = A.volume() + self.vtn - vi
                return float(vi / vu) if vu > 0 else 0.0
            except Exception:
                pass
        if self.T is not None and mesh.is_watertight:
            try:
                A = M._to_manifold(mesh)
                vi = (A ^ self.T).volume()
                vu = A.volume() + self.vt - vi
                return float(vi / vu) if vu > 0 else 0.0
            except Exception:
                pass
        return M.iou(mesh, self.m, n=self.n)[0]

    def _aligned(self, res):
        """The candidate meshed the way CADBench meshes it (cadquery's default STL export), alpha-wrapped
        when that mesh is not closed -- an open export makes the official wrap a thin shell and the
        score collapse (0.99 fixed-frame IoU -> 0.03) -- then mass-normalised IoU against the target."""
        import tempfile, trimesh
        import cadquery as cq
        from stepcad import metrics as M
        with tempfile.NamedTemporaryFile(suffix='.stl') as f:
            cq.exporters.export(res, f.name)
            mesh = trimesh.load(f.name, force='mesh')
        if not mesh.is_watertight:
            mesh = _alpha_wrap(mesh)
        A = M._to_manifold(_mass_normalise(mesh))
        vi = (A ^ self.Tn).volume()
        vu = A.volume() + self.vtn - vi
        return float(vi / vu) if vu > 0 else 0.0

    def __call__(self, actions):
        key = '\n'.join(actions)
        if key in self.cache:
            return self.cache[key]
        from stepcad import metrics as M
        s = -1.0
        try:
            res = self.solid(actions)
            ok, _ = M.valid_solid(res)
            if ok and self.Tn is not None:
                s = self._aligned(res)
            elif ok:
                s = self._iou(M.solid_to_mesh(res, tol=self.tol, ang=self.ang))
        except Exception:
            s = -1.0
        self.evals += 1
        self.cache[key] = s
        return s


# ---------------------------------------------------------------------
#  edit 1: continuous refinement (GP-EI)
# ---------------------------------------------------------------------

def _literals(action):
    """Optimisable numeric literals: (start, end, value, lo, hi).

    Directions (normal=..., xDir=...) and integer counts are excluded.
    """
    skip = []
    for m in _PLANE.finditer(action):
        skip.append((m.start(2), m.end(2)))
        skip.append((m.start(3), m.end(3)))
    for m in re.finditer(r'(range|polarArray|rarray)\([^)]*\)', action):
        skip.append((m.start(), m.end()))
    is_edge = bool(re.search(r'\.(fillet|chamfer)\(', action))
    out = []
    for m in _NUM.finditer(action):
        if any(a <= m.start() < b for a, b in skip):
            continue
        txt = m.group()
        v = float(txt)
        if '.' not in txt and abs(v) >= 2 and action[max(0, m.start() - 12):m.start()].find('angleDegrees') < 0:
            continue    # integer counts
        if is_edge and action[:m.start()].rfind('fillet(') > action[:m.start()].rfind(')') \
                or is_edge and action[:m.start()].rfind('chamfer(') > action[:m.start()].rfind(')'):
            lo, hi = 0.01, 0.5
        else:
            r = min(max(0.2 * abs(v), 0.005), 0.3)
            lo, hi = v - r, v + r
        out.append((m.start(), m.end(), v, lo, hi))
    return out


def _substitute(action, lits, x):
    parts, last = [], 0
    for (s, e, _, _, _), v in zip(lits, x):
        parts.append(action[last:s])
        parts.append(_f(v))
        last = e
    parts.append(action[last:])
    return ''.join(parts)


def edit_continuous(actions, t, score, rng, n_init=4, n_iter=5, batch=3,
                    max_dims=24):
    lits = _literals(actions[t])
    if not lits:
        return None
    if len(lits) > max_dims:     # optimise a random subset in very long sketches
        keep = sorted(rng.choice(len(lits), max_dims, replace=False))
        lits = [lits[k] for k in keep]
    lo = np.array([l[3] for l in lits]); hi = np.array([l[4] for l in lits])
    x0 = np.array([l[2] for l in lits])
    base = score(actions)

    def f(u):
        x = lo + u * (hi - lo)
        a = list(actions)
        a[t] = _substitute(actions[t], lits, x)
        return score(a), a

    U, Y, progs = [], [], []
    u0 = (x0 - lo) / np.maximum(hi - lo, 1e-12)
    U.append(u0); Y.append(base); progs.append(list(actions))
    for _ in range(n_init):
        u = rng.random(len(lits))
        y, a = f(u)
        U.append(u); Y.append(y); progs.append(a)
    from scipy.stats import norm
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
    for _ in range(n_iter):
        Ua, Ya = np.array(U), np.array(Y)
        valid = Ya > -0.5
        if valid.sum() < 2:
            u = rng.random((batch, len(lits)))
        else:
            gp = GaussianProcessRegressor(
                ConstantKernel(1.0) * Matern(length_scale=0.3, nu=2.5) + WhiteKernel(1e-5),
                normalize_y=True, n_restarts_optimizer=0)
            gp.fit(Ua[valid], Ya[valid])
            cand = np.clip(Ua[valid][np.argmax(Ya[valid])] + 0.15 * rng.standard_normal((1024, len(lits))), 0, 1)
            cand = np.vstack([cand, rng.random((1024, len(lits)))])
            mu, sd = gp.predict(cand, return_std=True)
            best = Ya[valid].max()
            z = (mu - best) / np.maximum(sd, 1e-9)
            ei = (mu - best) * norm.cdf(z) + sd * norm.pdf(z)
            u = cand[np.argsort(-ei)[:batch]]
        for uu in u:
            y, a = f(uu)
            U.append(uu); Y.append(y); progs.append(a)
    k = int(np.argmax(Y))
    return progs[k] if Y[k] > base + 1e-6 else None


# ---------------------------------------------------------------------
#  edit 6: snap literals to dimensions measured on the target
# ---------------------------------------------------------------------

def target_dims(mesh, n_slices=5):
    """Dimensions measured on the target: positions of axis-aligned planes, distances between
    parallel planes (thicknesses / depths / offsets), circle radii and centre coordinates from
    axis-aligned sections.  Returns (coords, radii) as sorted unique arrays."""
    pos = {0: [], 1: [], 2: []}
    try:
        for f, n in zip(mesh.facets, mesh.facets_normal):
            if mesh.area_faces[f].sum() < 1e-4 * mesh.area:
                continue
            ax = int(np.argmax(np.abs(n)))
            if abs(n[ax]) > 0.999:
                pos[ax].append(float(mesh.vertices[mesh.faces[f[0]][0]][ax]))
    except Exception:
        pass
    coords, radii = [], []
    for ax in range(3):
        v = np.unique(np.round(pos[ax], 4))
        coords += list(v)
        if len(v) <= 40:
            coords += [abs(a - b) for i, a in enumerate(v) for b in v[i + 1:]]
        lo, hi = mesh.bounds[0][ax], mesh.bounds[1][ax]
        for h in np.linspace(lo, hi, n_slices + 2)[1:-1]:
            try:
                n = np.zeros(3); n[ax] = 1.0
                o = np.zeros(3); o[ax] = h
                sec = mesh.section(plane_origin=o, plane_normal=n)
                if sec is None:
                    continue
                for P in sec.discrete:
                    Q = np.delete(np.asarray(P), ax, axis=1)
                    if len(Q) < 8:
                        continue
                    cx, cy, r, dev = _circle_fit(Q)
                    if r > 1e-3 and dev < 0.02 * r:
                        radii.append(r)
                        coords += [cx, cy]
            except Exception:
                continue
    c = np.unique(np.round([x for x in coords if 1e-4 < abs(x) < 3.0] + [-x for x in coords if 1e-4 < abs(x) < 3.0], 4))
    return c, np.unique(np.round(radii, 4))


def edit_snap(actions, t, score, dims, max_lits=30, k=2):
    """Coordinate descent over the literals of action t: try the k nearest measured target
    dimensions within 20% (min 0.03) of each value; keep a change only if the score improves.
    Catches the near-miss dimensions an exact fitter gets right."""
    coords, radii = dims
    lits = _literals(actions[t])[:max_lits]
    if not lits or len(coords) + len(radii) == 0:
        return None
    base = score(actions)
    cur = list(actions)
    vals = [l[2] for l in lits]
    improved = False
    for i, (st, en, v, _, _) in enumerate(lits):
        is_r = actions[t][max(0, st - 8):st].endswith('circle(')
        pool = radii if is_r and len(radii) else coords
        if not len(pool):
            continue
        tol = max(0.03, 0.2 * abs(v))
        near = pool[np.abs(pool - v) < tol]
        near = near[np.argsort(np.abs(near - v))][:k]
        for c in near:
            if abs(c - v) < 1e-4:
                continue
            x = list(vals); x[i] = float(c)
            a = list(cur); a[t] = _substitute(actions[t], lits, x)
            sc = score(a)
            if sc > base + 1e-5:
                base, cur, vals, improved = sc, a, x, True
    return cur if improved else None


# ---------------------------------------------------------------------
#  sketch fitting (shared by edits 2 and 4)
# ---------------------------------------------------------------------

def _circle_fit(P):
    A = np.c_[2 * P, np.ones(len(P))]
    b = (P ** 2).sum(1)
    c, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = c[0], c[1]
    r = math.sqrt(max(c[2] + cx * cx + cy * cy, 0))
    dev = np.abs(np.hypot(P[:, 0] - cx, P[:, 1] - cy) - r).max() if r > 0 else 1e9
    return cx, cy, r, dev


def sketch_from_loop(P, tol=0.004):
    """Closed 2D loop -> CadQuery sketch chain (no Workplane prefix).

    circle if the loop is round to 1% of its radius; otherwise the
    RDP-simplified polygon, with runs of >= 3 consecutive vertices that lie
    on one circle written as threePointArc.
    """
    from shapely.geometry import Polygon
    P = np.asarray(P, float)
    if len(P) > 2 and np.allclose(P[0], P[-1]):
        P = P[:-1]
    if len(P) < 3:
        return None
    cx, cy, r, dev = _circle_fit(P)
    if r > 0.005 and dev < 0.01 * r + 1e-4:
        return '.center(%s, %s).circle(%s)' % (_f(cx), _f(cy), _f(r))
    poly = Polygon(P)
    if not poly.is_valid or poly.area < 1e-5:
        poly = poly.buffer(0)
        if poly.geom_type != 'Polygon' or poly.area < 1e-5:
            return None
    S = np.asarray(poly.simplify(tol, preserve_topology=True).exterior.coords)[:-1]
    if len(S) < 3:
        return None
    # line + arc decomposition over the dense loop: walk the simplified
    # vertices; where the dense points between i and i+2 sit on a circle,
    # emit an arc through the middle vertex
    out = '.moveTo(%s, %s)' % (_f(S[0][0]), _f(S[0][1]))
    i, n = 0, len(S)
    while i < n:
        a, b = S[i], S[(i + 1) % n]
        c = S[(i + 2) % n] if i + 2 <= n else None
        if c is not None and i + 2 <= n:
            seg = np.array([a, b, c])
            cx, cy, r, dev = _circle_fit(seg)
            chord = np.linalg.norm(c - a)
            sag = np.abs(np.cross(b - a, c - a)) / max(chord, 1e-9)
            if r < 2.0 and sag > 2 * tol and chord < 1.8 * r:
                out += '.threePointArc((%s, %s), (%s, %s))' % (_f(b[0]), _f(b[1]), _f(c[0]), _f(c[1]))
                i += 2
                continue
        out += '.lineTo(%s, %s)' % (_f(b[0]), _f(b[1]))
        i += 1
    return out + '.close()'


def _plane_frame(origin, normal, xdir):
    o, n, x = (np.array(v, float) for v in (origin, normal, xdir))
    n /= np.linalg.norm(n); x /= np.linalg.norm(x)
    y = np.cross(n, x)
    return o, n, x, y


def _section_loops(mesh, o, n, x, y):
    sec = mesh.section(plane_origin=o, plane_normal=n)
    if sec is None:
        return []
    loops = []
    for ent in sec.discrete:
        P = np.asarray(ent)
        if len(P) < 4 or np.linalg.norm(P[0] - P[-1]) > 1e-4:
            continue                                    # open curve
        d = P - o
        loops.append(np.c_[d @ x, d @ y])
    return loops


# ---------------------------------------------------------------------
#  edit 2: sketch replacement
# ---------------------------------------------------------------------

def edit_sketch(actions, t, score, mesh):
    a = actions[t]
    pm = _PLANE.search(a)
    op = list(_OP3D.finditer(a))
    if not pm or not op:
        return None
    vec = lambda s: [float(v) for v in s.split(',')]
    o, n, x, y = _plane_frame(vec(pm.group(1)), vec(pm.group(2)), vec(pm.group(3)))
    loops = _section_loops(mesh, o, n, x, y)
    if not loops:
        return None
    from shapely.geometry import Polygon
    areas = [abs(Polygon(L).area) if len(L) >= 3 else 0 for L in loops]
    L = loops[int(np.argmax(areas))]
    sk = sketch_from_loop(L)
    if sk is None:
        return None
    start = pm.end() + (1 if a[pm.end():pm.end() + 1] == ')' else 0)   # past cq.Workplane(...)
    new = a[:start] + sk + a[op[-1].start():]
    b = list(actions); b[t] = new
    return b if score(b) > -0.5 else None


# ---------------------------------------------------------------------
#  edit 3: operation skipping
# ---------------------------------------------------------------------

def edit_skip(actions, t, score):
    if len(actions) <= 1 or t == 0:
        return None     # the first action creates the body; nothing to skip to
    b = actions[:t] + actions[t + 1:]
    return b if score(b) > score(actions) else None


# ---------------------------------------------------------------------
#  edit 4: complement cut
# ---------------------------------------------------------------------

def edit_complement(actions, score, mesh, max_clusters=4, mode='cut'):
    """mode 'cut': D = S(p) \\ m, cut-extrude into D (excess material);
    mode 'add': D = m \\ S(p), union-extrude into D (missing material)."""
    from stepcad import metrics as M
    if not mesh.is_watertight:
        return None
    try:
        G = M.solid_to_mesh(score.solid(actions))
        A, B = M._to_manifold(G), M._to_manifold(mesh)
        D = (A - B).to_mesh() if mode == 'cut' else (B - A).to_mesh()
    except Exception:
        return None
    import trimesh
    V = np.asarray(D.vert_properties)[:, :3]
    F = np.asarray(D.tri_verts)
    if len(F) == 0:
        return None
    dm = trimesh.Trimesh(V, F, process=True)
    nrm = dm.face_normals
    cen = dm.triangles_center
    area = dm.area_faces
    off = (nrm * cen).sum(1)
    used = np.zeros(len(nrm), bool)
    clusters = []
    cos2 = math.cos(math.radians(2.0))
    for i in np.argsort(-area):
        if used[i]:
            continue
        sel = (~used) & (nrm @ nrm[i] > cos2) & (np.abs(off - off[i]) < 1e-3)
        used |= sel
        if area[sel].sum() >= 0.0025:
            clusters.append((nrm[i], off[i], np.nonzero(sel)[0], area[sel].sum()))
    clusters.sort(key=lambda c: -c[3])
    base = score(actions)
    best = None
    for nvec, _, faces, _ in clusters[:max_clusters]:
        sub = dm.submesh([faces], append=True)
        o = sub.vertices.mean(0)
        n = -nvec / np.linalg.norm(nvec)           # cut goes into D
        x = np.cross(n, [0, 0, 1.0]) if abs(n[2]) < 0.9 else np.cross(n, [0, 1.0, 0])
        x /= np.linalg.norm(x)
        y = np.cross(n, x)
        try:
            outline = sub.outline()
            loops = [np.asarray(p) for p in outline.discrete]
        except Exception:
            continue
        loops = [np.c_[(P - o) @ x, (P - o) @ y] for P in loops if len(P) >= 4]
        if not loops:
            continue
        from shapely.geometry import Polygon
        L = max(loops, key=lambda L: abs(Polygon(L).area) if len(L) >= 3 else 0)
        sk = sketch_from_loop(L)
        if sk is None:
            continue
        for d in DEPTHS:
            act = ('result = result.%s(cq.Workplane(cq.Plane(origin=(%s, %s, %s), '
                   'normal=(%s, %s, %s), xDir=(%s, %s, %s)))%s.extrude(%s))'
                   % ('cut' if mode == 'cut' else 'union', _f(o[0]), _f(o[1]), _f(o[2]), _f(n[0]), _f(n[1]), _f(n[2]),
                      _f(x[0]), _f(x[1]), _f(x[2]), sk, _f(d)))
            cand = actions + [act]
            s = score(cand)
            if s > base + 1e-6 and (best is None or s > best[0]):
                best = (s, cand)
    return best[1] if best else None


# ---------------------------------------------------------------------
#  beam search
# ---------------------------------------------------------------------

def refine(actions, target_mesh, K=2, rng_seed=0, max_complement=4,
           edits=('complement', 'add', 'sketch', 'skip', 'continuous'), log=None,
           budget_s=None, stop_iou=0.985, max_passes=4, progress=None):
    """v4 schedule (same five edits, same budget as v3; REFINE_V=3 restores v3).

    v3 re-ran complement/add for every action position and beam entry --
    ~40 evaluations each, though they append at the end and do not depend on
    the position -- and on hard parts (~100 evaluations in the budget) the
    search rarely got past the first action.  Here each pass:
      1. complement / add, greedily on the best program, repeated while they
         gain (at most max_complement appended operations overall);
      2. a beam sweep over actions t = 1..T with the per-action edits in
         cost order (sketch, skip, continuous); continuous is a light BO
         (3 + 3x3 evaluations) on the first pass, the full one afterwards.
    progress(actions, score) is called on every improvement, so a caller
    whose worker is killed or times out keeps the best program found.
    """
    if os.environ.get('REFINE_V', '4') == '3':
        return refine_v3(actions, target_mesh, K=K, rng_seed=rng_seed, max_complement=max_complement,
                         edits=edits, log=log, budget_s=budget_s, stop_iou=stop_iou, max_passes=max_passes)
    rng = np.random.default_rng(rng_seed)
    snap_final = os.environ.get('REFINE_SNAP') == 'final'
    t_start = time.time()
    deadline = t_start + (0.85 if snap_final else 1.0) * budget_s if budget_s else float('inf')
    late = lambda: time.time() > deadline
    score = Scorer(target_mesh, max_prefix=64)
    if os.environ.get('REFINE_SNAP') == '1' and 'snap' not in edits:
        edits = tuple(edits) + ('snap',)
    dims = target_dims(score.m) if 'snap' in edits else None
    stats = {e: {'s': 0.0, 'calls': 0, 'wins': 0} for e in edits}
    best = [score(actions), list(actions)]

    def note(s_, a_):
        if s_ > best[0] + 1e-6:
            best[0], best[1] = s_, list(a_)
            if progress:
                try:
                    progress(best[1], best[0])
                except Exception:
                    pass

    def run(name, fn, parent):
        t0 = time.time()
        try:
            r = fn()
        except Exception:
            r = None              # one failing edit must not end the refinement
        st = stats[name]
        st['s'] += time.time() - t0
        st['calls'] += 1
        if r:
            sr = score(r)
            if sr > parent + 1e-4:
                st['wins'] += 1
            note(sr, r)
        return r

    ncomp = 0
    for npass in range(max_passes):
        if late() or best[0] >= stop_iou:
            break
        start = best[0]
        # 1. complement / add on the best program while they gain
        while ncomp < max_complement and not late() and best[0] < stop_iou:
            gained = False
            for e, mode in (('complement', 'cut'), ('add', 'add')):
                if e in edits and ncomp < max_complement and not late():
                    s0 = best[0]
                    r = run(e, lambda: edit_complement(best[1], score, score.m, mode=mode), s0)
                    if r and score(r) > s0 + 1e-4:
                        ncomp += 1
                        gained = True
            if not gained:
                break
        # 2. per-action beam sweep, cheap edits first
        light = dict(n_init=3, n_iter=3, batch=3) if npass == 0 else {}
        beam = [(best[0], [(a, i) for i, a in enumerate(best[1])])]
        T = len(best[1])
        for t in range(T):
            if late() or beam[0][0] >= stop_iou:
                break
            cands = list(beam)
            for s, prog in beam:
                pos = next((k for k, (_, i) in enumerate(prog) if i == t), None)
                if pos is None:
                    continue
                acts = [a for a, _ in prog]
                idx = [i for _, i in prog]
                for e in ('sketch', 'skip', 'snap', 'continuous'):
                    if e not in edits or late():
                        continue
                    if e == 'snap':
                        r = run(e, lambda: edit_snap(acts, pos, score, dims), s)
                        if r: cands.append((score(r), list(zip(r, idx))))
                    elif e == 'sketch':
                        r = run(e, lambda: edit_sketch(acts, pos, score, score.m), s)
                        if r: cands.append((score(r), list(zip(r, idx))))
                    elif e == 'skip':
                        r = run(e, lambda: edit_skip(acts, pos, score), s)
                        if r: cands.append((score(r), list(zip(r, idx[:pos] + idx[pos + 1:]))))
                    else:
                        r = run(e, lambda: edit_continuous(acts, pos, score, rng, **light), s)
                        if r: cands.append((score(r), list(zip(r, idx))))
            seen, uniq = set(), []
            for c in sorted(cands, key=lambda c: -c[0]):
                key = '\n'.join(a for a, _ in c[1])
                if key not in seen:
                    seen.add(key)
                    uniq.append(c)
            beam = uniq[:K]
            note(beam[0][0], [a for a, _ in beam[0][1]])
            if log:
                log({'pass': npass, 't': t, 'best': round(best[0], 4), 'evals': score.evals})
        if best[0] < start + 1e-3:
            break
    if snap_final:                     # polish: snap every action's literals to measured target dimensions
        dims_f = target_dims(score.m)
        end = t_start + budget_s if budget_s else float('inf')
        for t in range(len(best[1])):
            if time.time() > end:
                break
            try:
                r = edit_snap(best[1], t, score, dims_f)
            except Exception:
                r = None
            if r:
                note(score(r), r)
    if log:
        log({'edit_stats': {e: {'s': round(v['s'], 1), 'calls': v['calls'], 'wins': v['wins']}
                            for e, v in stats.items()}})
    return best[1], best[0], score.evals


def refine_v3(actions, target_mesh, K=2, rng_seed=0, max_complement=4,
           edits=('complement', 'add', 'sketch', 'skip', 'continuous'), log=None,
           budget_s=None, stop_iou=0.985, max_passes=4):
    """Beam search; with budget_s, stops editing at the deadline and
    returns the best program found so far (never worse than the seed).

    Edits run in order of measured yield per second (complement cut, its
    missing-material twin 'add', sketch replacement, then the
    many-evaluation continuous search), so a short budget is spent where
    gains are largest; the search stops early once the best program reaches
    stop_iou.  One pass sweeps t = 1..T; further passes re-sweep the
    improved program while the budget lasts and the last pass gained (a
    single pass used ~57 s of a 300 s budget on dev140).  Per-edit
    statistics (time spent, candidates that improved on their parent) go
    to log.
    """
    rng = np.random.default_rng(rng_seed)
    deadline = time.time() + budget_s if budget_s else float('inf')
    late = lambda: time.time() > deadline
    score = Scorer(target_mesh)
    stats = {e: {'s': 0.0, 'calls': 0, 'wins': 0} for e in edits}

    def run(name, fn, parent):
        t0 = time.time()
        try:
            r = fn()
        except Exception:
            r = None              # one failing edit must not end the refinement
        st = stats[name]
        st['s'] += time.time() - t0
        st['calls'] += 1
        if r and score(r) > parent + 1e-4:
            st['wins'] += 1
        return r

    best_s, best_acts = score(actions), list(actions)
    for npass in range(max_passes):
        if late() or best_s >= stop_iou:
            break
        start_s = best_s
        beam = [(best_s, [(a, i) for i, a in enumerate(best_acts)], 0)]
        T = len(best_acts)
        for t in range(T):
            if late() or beam[0][0] >= stop_iou:
                break
            cands = list(beam)
            for s, prog, ncomp in beam:
                pos = next((k for k, (_, i) in enumerate(prog) if i == t), None)
                acts = [a for a, _ in prog]
                idx = [i for _, i in prog]
                outs = []
                for e in edits:
                    if late():
                        break
                    if e in ('complement', 'add') and ncomp < max_complement:
                        mode = 'cut' if e == 'complement' else 'add'
                        r = run(e, lambda: edit_complement(acts, score, score.m, mode=mode), s)
                        if r: outs.append((r, idx + [None], ncomp + 1))
                    elif pos is not None and e == 'sketch':
                        r = run(e, lambda: edit_sketch(acts, pos, score, score.m), s)
                        if r: outs.append((r, idx, ncomp))
                    elif pos is not None and e == 'skip':
                        r = run(e, lambda: edit_skip(acts, pos, score), s)
                        if r: outs.append((r, idx[:pos] + idx[pos + 1:], ncomp))
                    elif pos is not None and e == 'continuous':
                        r = run(e, lambda: edit_continuous(acts, pos, score, rng), s)
                        if r: outs.append((r, idx, ncomp))
                for r, ix, nc in outs:
                    cands.append((score(r), list(zip(r, ix)), nc))
            # unique programs, top-K
            seen, uniq = set(), []
            for c in sorted(cands, key=lambda c: -c[0]):
                key = '\n'.join(a for a, _ in c[1])
                if key not in seen:
                    seen.add(key)
                    uniq.append(c)
            beam = uniq[:K]
            if log:
                log({'pass': npass, 't': t, 'best': round(beam[0][0], 4), 'evals': score.evals})
        top = max(beam, key=lambda c: c[0])
        if top[0] > best_s:
            best_s, best_acts = top[0], [a for a, _ in top[1]]
        if best_s < start_s + 1e-3:          # this pass found nothing worth another
            break
    if log:
        log({'edit_stats': {e: {'s': round(v['s'], 1), 'calls': v['calls'], 'wins': v['wins']}
                            for e, v in stats.items()}})
    return best_acts, best_s, score.evals
