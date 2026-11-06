"""Split a CadQuery program into actions: the unit a policy commits.

An action is the run of top-level statements up to and including one that
updates `result` -- a `result = ...` assignment, or a compound statement
(for / if / with) whose body assigns it.  Both program styles fall out:

    ARCADE         one `result = result.op(...)` statement per action
                   (multi-line statements stay whole)
    named-feature  plane_k = ... / sketch_k = ... / solid_k = ... lines fold
                   into the `result = result.cut(solid_k)` they feed; a
                   pattern loop that updates result is one action

so every action boundary leaves `result` defined, and executing a prefix of
actions yields that intermediate solid.  Trailing statements that never
update result (and `final_result = ...`) are dropped.  Imports and the
shebang are not part of any action.
"""
from __future__ import annotations

import ast


def _assigns_result(node):
    for n in ast.walk(node):
        if isinstance(n, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            tg = n.targets if isinstance(n, ast.Assign) else [n.target]
            for t in tg:
                for x in ast.walk(t):
                    if isinstance(x, ast.Name) and x.id == 'result':
                        return True
    return False


def split_program(text):
    """Program text -> list of action strings (exact source of each span).
    Falls back to line-based splitting when the text does not parse
    (a truncated generation): each `result =` line closes an action."""
    lines = text.split('\n')
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return _split_lines(lines)
    acts, start = [], None
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'final_result'
                                                for t in node.targets):
            break
        if start is None:
            start = node.lineno
        if _assigns_result(node):
            acts.append('\n'.join(lines[start - 1:node.end_lineno]))
            start = None
    return acts


def _split_lines(lines):
    acts, cur = [], []
    for ln in lines:
        s = ln.strip()
        if s.startswith('final_result'):
            break
        if not s or s.startswith(('import ', '#!')):
            continue
        cur.append(ln)
        if not ln[:1].isspace() and (s.startswith('result =') or s.startswith('result=')):
            acts.append('\n'.join(cur))
            cur = []
    return acts
