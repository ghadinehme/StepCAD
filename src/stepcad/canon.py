"""One canonical CadQuery program format.

Canonical form: the two import lines, then one ``result = ...`` statement per operation, every
plane / sketch / profile written inline, and a closing ``final_result = result``:

    import cadquery as cq
    import math
    result = cq.Workplane(cq.Plane(origin=(0, 0, 0), normal=(0, 0, 1), xDir=(1, 0, 0))).circle(0.5).extrude(1)
    result = result.cut(cq.Workplane(cq.Plane(...)).rect(0.2, 0.2).extrude(1, both=True))
    final_result = result

Programs written with named intermediates (``plane_0 = cq.Plane(...)``, ``sketch_0 = sketch_0.lineTo(...)``,
``for i in range(3): ...``) are rewritten by inlining every intermediate into the ``result`` statement
that uses it, unrolling loops over constant ranges and folding constant arithmetic.  The rewrite only
re-expresses the same CadQuery calls in the same order, so the solid is unchanged; ``equivalent``
checks that by executing both programs.

    canonicalize(code) -> canonical code, or None if the program uses a construct not handled here
    canonical(code, check=True) -> canonical code if it is provably equivalent, else the input unchanged
"""
from __future__ import annotations

import ast
import copy
import math

HEADER = 'import cadquery as cq\nimport math\n'
_FOLD = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)


class _Unsupported(Exception):
    pass


def _num(v):
    """A folded float as a short literal: the fewest decimals that keep it within 1e-9 (relative)."""
    if isinstance(v, bool) or not isinstance(v, float):
        return v
    for nd in range(3, 13):
        r = round(v, nd)
        if abs(r - v) <= 1e-9 * max(1.0, abs(v)):
            return r + 0.0
    return v


class _Subst(ast.NodeTransformer):
    """Replace names bound to inlined expressions / loop constants, then fold constant arithmetic."""

    def __init__(self, env, rv):
        self.env, self.rv = env, rv

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load) and node.id in self.env:
            expr, ver = self.env[node.id]
            if ver is not None and ver != self.rv:      # captured an older `result`: inlining would change it
                raise _Unsupported('stale result capture')
            return copy.deepcopy(expr)
        return node

    def visit_BinOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, _FOLD) and all(isinstance(x, ast.Constant) and isinstance(x.value, (int, float))
                                              and not isinstance(x.value, bool) for x in (node.left, node.right)):
            try:
                v = eval(compile(ast.Expression(node), '<fold>', 'eval'), {})
            except Exception:
                return node
            return ast.copy_location(ast.Constant(_num(v)), node)
        return node

    def visit_UnaryOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant) \
                and isinstance(node.operand.value, (int, float)) and node.operand.value < 0:
            return ast.copy_location(ast.Constant(-node.operand.value), node)
        return node


class _IntLiterals(ast.NodeTransformer):
    """Whole-number floats as ints (``normal=(0, 0, 1)``, ``box(1.5, 2, 2)``), as the canonical data writes them."""

    def visit_Constant(self, node):
        if isinstance(node.value, float) and node.value.is_integer() and abs(node.value) < 1e15:
            return ast.copy_location(ast.Constant(int(node.value)), node)
        return node


def _uses(node, name):
    return any(isinstance(n, ast.Name) and n.id == name for n in ast.walk(node))


def _iter_values(it, env, rv):
    it = _Subst(env, rv).visit(copy.deepcopy(it))
    if isinstance(it, ast.Call) and isinstance(it.func, ast.Name) and it.func.id == 'range' and not it.keywords:
        args = [a.value if isinstance(a, ast.Constant) else None for a in it.args]
        if None in args or not all(isinstance(a, int) for a in args) or len(range(*args)) > 1000:
            raise _Unsupported('range')
        return list(range(*args))
    try:
        vals = ast.literal_eval(it)
    except Exception:
        raise _Unsupported('loop iterable')
    if not isinstance(vals, (list, tuple)):
        raise _Unsupported('loop iterable')
    return list(vals)


def _bind(target, value, env):
    """Loop variable(s) -> constants (tuple unpacking supported)."""
    if isinstance(target, ast.Name):
        env[target.id] = (ast.parse(repr(value), mode='eval').body, None)
    elif isinstance(target, (ast.Tuple, ast.List)) and isinstance(value, (list, tuple)) and len(value) == len(target.elts):
        for t, v in zip(target.elts, value):
            _bind(t, v, env)
    else:
        raise _Unsupported('loop target')


def _run(stmts, env, out, state):
    for s in stmts:
        if isinstance(s, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(s, ast.Pass):
            continue
        if isinstance(s, ast.Assign) and len(s.targets) == 1 and isinstance(s.targets[0], ast.Name):
            name = s.targets[0].id
            value = _Subst(env, state['rv']).visit(copy.deepcopy(s.value))
            if name == 'result':
                out.append('result = ' + ast.unparse(_IntLiterals().visit(value)))
                state['rv'] += 1
            elif name == 'final_result':
                if not (isinstance(s.value, ast.Name) and s.value.id == 'result'):
                    raise _Unsupported('final_result')
                state['final'] = True
            else:
                env[name] = (value, state['rv'] if _uses(value, 'result') else None)
        elif isinstance(s, ast.For) and not s.orelse:
            for v in _iter_values(s.iter, env, state['rv']):
                _bind(s.target, v, env)
                _run(s.body, env, out, state)
        else:
            raise _Unsupported(type(s).__name__)


def is_canonical(code):
    try:
        m = ast.parse(code)
    except SyntaxError:
        return False
    for s in m.body:
        if isinstance(s, (ast.Import, ast.ImportFrom)):
            continue
        if not (isinstance(s, ast.Assign) and len(s.targets) == 1 and isinstance(s.targets[0], ast.Name)
                and s.targets[0].id in ('result', 'final_result')):
            return False
    return True


def canonicalize(code):
    if is_canonical(code):
        return code
    import sys
    if sys.getrecursionlimit() < 20000:             # long .lineTo() chains nest deeply in the AST
        sys.setrecursionlimit(20000)
    try:
        m = ast.parse(code)
        out, state = [], {'rv': 0, 'final': False}
        _run(m.body, {}, out, state)
    except (_Unsupported, SyntaxError, RecursionError, ValueError):
        return None
    if not out:
        return None
    return HEADER + '\n'.join(out) + '\n' + ('final_result = result\n' if state['final'] else '')


# ---------------------------------------------------------------- equivalence

def _signature(code, timeout):
    """Execute in a subprocess; (volume, area, bbox, n_faces, n_edges, n_vertices) of the final solid."""
    import json, subprocess, sys
    src = ('import json, sys\nimport cadquery as cq\nns = {}\nexec(sys.stdin.read(), ns)\n'
           'r = ns.get("final_result", ns.get("result"))\n'
           'v = r.val() if hasattr(r, "val") else r\n'
           'try:\n    v = cq.Compound.makeCompound([s for s in r.vals() if isinstance(s, cq.Shape)]) if hasattr(r, "vals") and len(r.vals()) > 1 else v\n'
           'except Exception:\n    pass\n'
           'b = v.BoundingBox()\n'
           'print(json.dumps([v.Volume(), v.Area(), [b.xmin, b.ymin, b.zmin, b.xmax, b.ymax, b.zmax], '
           'len(v.Faces()), len(v.Edges()), len(v.Vertices())]))\n')
    try:
        p = subprocess.run([sys.executable, '-c', src], input=code, capture_output=True, text=True, timeout=timeout)
        return json.loads(p.stdout.strip().splitlines()[-1]) if p.returncode == 0 and p.stdout.strip() else None
    except Exception:
        return None


def equivalent(a, b, tol=1e-6, timeout=120):
    """Same solid: equal topology counts and volume / area / bounding box within `tol` (relative to size)."""
    sa, sb = _signature(a, timeout), _signature(b, timeout)
    if sa is None or sb is None:
        return False
    scale = max(1e-9, max(abs(x) for x in sa[2]))
    close = lambda x, y, s: abs(x - y) <= tol * max(s, abs(x), abs(y))
    return (sa[3:] == sb[3:] and close(sa[0], sb[0], scale ** 3) and close(sa[1], sb[1], scale ** 2)
            and all(close(x, y, scale) for x, y in zip(sa[2], sb[2])))


def canonical(code, check=True, timeout=120):
    """`code` in canonical form when that is provably the same solid, otherwise `code` unchanged."""
    c = canonicalize(code)
    if c is None or c == code:
        return code
    if check and not equivalent(code, c, timeout=timeout):
        return code
    return c
