"""Plane cutting with watertight caps.

Fully vectorized and memory-light (about 1 GB peak for 15M triangles), unlike a
general boolean engine. When vertices sit on (or within a few float32 steps of)
the plane, the plane is nudged by a sub-micron amount so every new vertex gets
a distinct position; otherwise exported parts could weld into non-manifold
shapes when re-imported.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import mapbox_earcut as earcut

from .mesh import CHUNK, Mesh, report
from .weld import group_rows


def unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("plane normal must be non-zero")
    return v / n


def plane_basis(normal) -> tuple[np.ndarray, np.ndarray]:
    """(u, w) orthonormal so that (u, w, normal) is right-handed."""
    n = unit(normal)
    helper = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(helper, n)
    u /= np.linalg.norm(u)
    return u, np.cross(n, u)


def _signed_distance(mesh: Mesh, n: np.ndarray, c: float) -> np.ndarray:
    d = np.empty(mesh.n_vertices, dtype=np.float64)
    for s in range(0, mesh.n_vertices, CHUNK):
        d[s:s + CHUNK] = mesh.vertices[s:s + CHUNK].astype(np.float64) @ n - c
    return d


def _plane_epsilon(mesh: Mesh) -> float:
    """Distance below which a vertex counts as lying on the plane."""
    b = mesh.bounds
    scale = float(np.abs(b).max()) or 1.0
    ulp = float(np.spacing(np.float32(scale)))
    return max(8 * ulp, 1e-7 * float(np.linalg.norm(b[1] - b[0])))


def _nudge(d: np.ndarray, eps: float) -> np.ndarray:
    if not (np.abs(d) < eps).any():
        return d
    for k in (2.0, -2.0, 3.3, -3.3, 4.7, -4.7, 6.1, -6.1, 8.9, -8.9):
        if not (np.abs(d - k * eps) < eps).any():
            return d - k * eps
    return d - 11.3 * eps


def section_segments(mesh: Mesh, normal, origin) -> np.ndarray:
    """Cross-section outline as (k, 2, 3) float32 segments (live preview)."""
    n = unit(normal)
    c = float(n @ np.asarray(origin, dtype=np.float64))
    d = _signed_distance(mesh, n, c)
    pos = d >= 0
    s = pos[mesh.faces]
    cnt = s.sum(1)
    st = (cnt == 1) | (cnt == 2)
    f = mesh.faces[st]
    s = s[st]
    lonely = np.where(cnt[st] == 1, np.argmax(s, 1), np.argmin(s, 1))
    rot = (lonely[:, None] + np.arange(3)[None, :]) % 3
    t = np.take_along_axis(f, rot, 1)
    v = mesh.vertices.astype(np.float64)

    def point(a, b):
        da, db = d[a], d[b]
        w = (da / (da - db))[:, None]
        return v[a] + w * (v[b] - v[a])

    return np.stack([point(t[:, 0], t[:, 1]), point(t[:, 0], t[:, 2])], 1).astype(np.float32)


@dataclass
class CutResult:
    positive: Mesh  # side the normal points to
    negative: Mesh
    loops: int = 0
    open_loops: int = 0  # >0 means the model was not closed where it was cut
    warnings: list = field(default_factory=list)


def plane_cut(mesh: Mesh, normal, origin, progress=None) -> CutResult:
    n = unit(normal)
    c = float(n @ np.asarray(origin, dtype=np.float64))
    V, F = mesh.vertices, mesh.faces
    nv = np.int64(mesh.n_vertices)
    report(progress, 0.05, "Classifying")
    d = _signed_distance(mesh, n, c)
    eps = _plane_epsilon(mesh)
    d = _nudge(d, eps)
    if d.max() < 4 * eps or d.min() > -4 * eps:
        # Plane only grazes the model: nothing to cut off.
        empty = Mesh(np.zeros((0, 3)), np.zeros((0, 3)))
        return CutResult(empty, mesh) if d.max() < 4 * eps else CutResult(mesh, empty)
    pos = d >= 0
    s = pos[F]
    cnt = s.sum(1, dtype=np.int8)
    del s
    full_pos = cnt == 3
    full_neg = cnt == 0
    st = np.nonzero(~(full_pos | full_neg))[0]
    if len(st) == 0:
        empty = Mesh(np.zeros((0, 3)), np.zeros((0, 3)))
        return (CutResult(mesh, empty) if full_pos.all() else CutResult(empty, mesh))

    report(progress, 0.25, "Splitting triangles")
    fs = F[st]
    ss = pos[fs]
    lonely_pos = cnt[st] == 1
    del cnt
    li = np.where(lonely_pos, np.argmax(ss, 1), np.argmin(ss, 1))
    rot = (li[:, None] + np.arange(3)[None, :]) % 3
    tri = np.take_along_axis(fs, rot, 1).astype(np.int64)
    a, b, cc = tri[:, 0], tri[:, 1], tri[:, 2]
    k = len(st)

    def ekey(x, y):
        return np.minimum(x, y) * nv + np.maximum(x, y)

    keys = np.concatenate([ekey(a, b), ekey(a, cc)])
    uk, inv = np.unique(keys, return_inverse=True)
    lo, hi = uk // nv, uk % nv
    dl, dh = d[lo], d[hi]
    t = (dl / (dl - dh))[:, None]
    vlo = V[lo].astype(np.float64)
    newpts = (vlo + t * (V[hi].astype(np.float64) - vlo)).astype(np.float32)
    del d, vlo
    # Neighbouring edges can round to the same float32 point: merge those
    # (an edge collapse) so re-importing the STL can't pinch the surface.
    first, ginv = group_rows(len(newpts), lambda ix: newpts[ix].astype(np.float64))
    newid = nv + ginv[inv.ravel()]
    newpts = newpts[first]
    p, q = newid[:k], newid[k:]

    lonely_tri = np.stack([a, p, q], 1)
    quad1 = np.stack([p, b, cc], 1)
    quad2 = np.stack([p, cc, q], 1)
    lp, ln = lonely_pos, ~lonely_pos
    pos_split = _drop_degenerate(np.concatenate([lonely_tri[lp], quad1[ln], quad2[ln]]))
    neg_split = _drop_degenerate(np.concatenate([lonely_tri[ln], quad1[lp], quad2[lp]]))

    report(progress, 0.45, "Building caps")
    # Directed cap edges for the positive part's cap (they follow its winding).
    seg_from = np.where(lonely_pos, q, p) - nv
    seg_to = np.where(lonely_pos, p, q) - nv
    real = seg_from != seg_to
    cap, loops, open_loops, warnings = _cap(newpts.astype(np.float64), seg_from[real], seg_to[real], n)
    cap = cap + nv

    report(progress, 0.7, "Assembling parts")
    allv = np.concatenate([V, newpts])
    pos_faces = np.concatenate([F[full_pos], pos_split, cap])
    neg_faces = np.concatenate([F[full_neg], neg_split, cap[:, ::-1]])
    del full_pos, full_neg
    positive = Mesh(allv, pos_faces).compact()
    report(progress, 0.85, "Assembling parts")
    negative = Mesh(allv, neg_faces).compact()
    if open_loops:
        warnings.append("open_cut")
    report(progress, 1.0, "Done")
    return CutResult(positive, negative, loops, open_loops, warnings)


def _drop_degenerate(f: np.ndarray) -> np.ndarray:
    return f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 2] != f[:, 0])]


def _trace_loops(seg_from: np.ndarray, seg_to: np.ndarray, n_points: int):
    nxt = np.full(n_points, -1, dtype=np.int64)
    nxt[seg_from] = seg_to
    has_in = np.zeros(n_points, dtype=bool)
    has_in[seg_to] = True
    loops, open_count = [], 0
    nxt_l = nxt.tolist()
    vis = [False] * n_points
    # Open chains first (start where nothing comes in), then closed loops.
    starts = seg_from[~has_in[seg_from]].tolist() + seg_from.tolist()
    for s0 in starts:
        if vis[s0]:
            continue
        loop = []
        cur = s0
        while cur != -1 and not vis[cur]:
            vis[cur] = True
            loop.append(cur)
            cur = nxt_l[cur]
        if cur != s0:
            open_count += 1
        if len(loop) >= 3:
            loops.append(np.array(loop, dtype=np.int64))
    return loops, open_count


def _signed_area(xy: np.ndarray) -> float:
    x, y = xy[:, 0], xy[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def _point_in_polygon(pt, poly) -> bool:
    x, y = pt
    xi, yi = poly[:, 0], poly[:, 1]
    xj, yj = np.roll(xi, 1), np.roll(yi, 1)
    cross = ((yi > y) != (yj > y))
    with np.errstate(divide="ignore", invalid="ignore"):
        xint = (xj - xi) * (y - yi) / (yj - yi) + xi
    return bool(np.count_nonzero(cross & (x < xint)) % 2)


def _cap(points: np.ndarray, seg_from, seg_to, normal):
    """Triangulate cut loops. Returns (faces (t, 3) local ids, loops, open, warnings)."""
    warnings = []
    loops, open_loops = _trace_loops(seg_from, seg_to, len(points))
    if not loops:
        return np.zeros((0, 3), dtype=np.int64), 0, open_loops, warnings
    u, w = plane_basis(normal)
    xy_all = np.stack([points @ u, points @ w], 1)
    polys = [xy_all[lp] for lp in loops]
    areas = np.array([_signed_area(p) for p in polys])
    outer_sign = np.sign(areas[np.argmax(np.abs(areas))]) or 1.0
    is_outer = np.sign(areas) == outer_sign
    outer_ids = np.nonzero(is_outer)[0]
    bbox = np.array([[p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()] for p in polys])
    holes_of = {i: [] for i in outer_ids}
    by_size = outer_ids[np.argsort(np.abs(areas[outer_ids]))]
    for h in np.nonzero(~is_outer)[0]:
        pt = polys[h][0]
        for o in by_size:
            bb = bbox[o]
            if bb[0] <= pt[0] <= bb[2] and bb[1] <= pt[1] <= bb[3] and _point_in_polygon(pt, polys[o]):
                holes_of[o].append(h)
                break
        else:
            warnings.append("orphan_hole")
    faces = []
    for o in outer_ids:
        rings = [o] + holes_of[o]
        ring_xy = np.concatenate([polys[r] for r in rings])
        ring_ids = np.concatenate([loops[r] for r in rings])
        ends = np.cumsum([len(polys[r]) for r in rings]).astype(np.uint32)
        tri = earcut.triangulate_float64(ring_xy, ends).reshape(-1, 3)
        expected = len(ring_ids) + 2 * (len(rings) - 1) - 2
        if len(tri) < expected:
            warnings.append("cap_fallback")
            tri = _fan(len(polys[o]))
            ring_ids = loops[o]
            ring_xy = polys[o]
        ta = ring_xy[tri]
        sa = ((ta[:, 1, 0] - ta[:, 0, 0]) * (ta[:, 2, 1] - ta[:, 0, 1])
              - (ta[:, 2, 0] - ta[:, 0, 0]) * (ta[:, 1, 1] - ta[:, 0, 1]))
        wrong = np.sign(sa) == -outer_sign
        tri[wrong] = tri[wrong][:, ::-1]
        faces.append(ring_ids[tri])
    return np.concatenate(faces).astype(np.int64), len(loops), open_loops, warnings


def _fan(k: int) -> np.ndarray:
    i = np.arange(1, k - 1)
    return np.stack([np.zeros_like(i), i, i + 1], 1)
