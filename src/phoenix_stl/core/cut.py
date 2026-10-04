"""Plane cutting with watertight caps.

Fully vectorized and memory-light (about 1.4 GB peak for 15M triangles), unlike
a general boolean engine.

Vertices within a few float32 steps of the plane are treated as lying on it:
they are snapped onto the plane and shared by both parts, so no near-duplicate
points are created (those would break the cap triangulation and could weld
into non-manifold shapes when the exported STL is re-imported). Faces lying in
the plane go to the side their material is on. Each part's cap is built from
that part's own open edges on the plane.
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


def _empty() -> Mesh:
    return Mesh(np.zeros((0, 3)), np.zeros((0, 3)))


def _rotate_to(f: np.ndarray, first: np.ndarray) -> np.ndarray:
    """Cyclically rotate each face (keeps winding) so column `first` comes first."""
    rot = (first[:, None] + np.arange(3)[None, :]) % 3
    return np.take_along_axis(f, rot, 1)


def plane_cut(mesh: Mesh, normal, origin, progress=None) -> CutResult:
    n = unit(normal)
    c = float(n @ np.asarray(origin, dtype=np.float64))
    V, F = mesh.vertices, mesh.faces
    nv = np.int64(mesh.n_vertices)
    report(progress, 0.05, "Classifying")
    d = _signed_distance(mesh, n, c)
    eps = _plane_epsilon(mesh)
    if d.max() <= eps:
        return CutResult(_empty(), mesh)
    if d.min() >= -eps:
        return CutResult(mesh, _empty())
    side = np.zeros(mesh.n_vertices, dtype=np.int8)
    side[d > eps] = 1
    side[d < -eps] = -1
    fs = side[F]
    npos = (fs > 0).sum(1, dtype=np.int8)
    nneg = (fs < 0).sum(1, dtype=np.int8)
    del fs
    straddle = (npos > 0) & (nneg > 0)
    pos_f = (npos > 0) & (nneg == 0)
    neg_f = (nneg > 0) & (npos == 0)
    flat = np.nonzero(~(straddle | pos_f | neg_f))[0]
    if len(flat):
        # Face lying in the plane: its normal points away from its material.
        t = V[F[flat]].astype(np.float64)
        up = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]) @ n > 0
        neg_f[flat[up]] = True
        pos_f[flat[~up]] = True
    del npos, nneg

    report(progress, 0.25, "Splitting triangles")
    st = F[straddle].astype(np.int64)
    ss = side[st]
    has_zero = (ss == 0).any(1)
    # Case 1: no vertex on the plane -> one vertex alone on its side.
    t1, s1 = st[~has_zero], ss[~has_zero]
    lonely_pos = s1.sum(1) < 0  # (+,-,-)
    t1 = _rotate_to(t1, np.where(lonely_pos, np.argmax(s1, 1), np.argmin(s1, 1)))
    # Case 2: one vertex on the plane, the other two on opposite sides.
    t2, s2 = st[has_zero], ss[has_zero]
    t2 = _rotate_to(t2, np.argmin(np.abs(s2), 1))
    del st, ss

    def ekey(x, y):
        return np.minimum(x, y) * nv + np.maximum(x, y)

    k1 = len(t1)
    keys = np.concatenate([ekey(t1[:, 0], t1[:, 1]), ekey(t1[:, 0], t1[:, 2]), ekey(t2[:, 1], t2[:, 2])])
    uk, inv = np.unique(keys, return_inverse=True)
    lo, hi = uk // nv, uk % nv
    dl, dh = d[lo], d[hi]
    w = (dl / (dl - dh))[:, None]
    vlo = V[lo].astype(np.float64)
    pts = vlo + w * (V[hi].astype(np.float64) - vlo)
    pts -= ((pts @ n) - c)[:, None] * n  # exactly onto the plane
    newpts = pts.astype(np.float32)
    del vlo, pts
    # Neighbouring edges can still round to the same float32 point: merge them.
    first, ginv = group_rows(len(newpts), lambda ix: newpts[ix].astype(np.float64))
    newid = nv + ginv[inv.ravel()]
    newpts = newpts[first]
    p1, q1, p2 = newid[:k1], newid[k1:2 * k1], newid[2 * k1:]

    a, b, cc = t1[:, 0], t1[:, 1], t1[:, 2]
    lonely_tri = np.stack([a, p1, q1], 1)
    quad1 = np.stack([p1, b, cc], 1)
    quad2 = np.stack([p1, cc, q1], 1)
    lp, ln = lonely_pos, ~lonely_pos
    z, ea, eb = t2[:, 0], t2[:, 1], t2[:, 2]
    half_a = np.stack([z, ea, p2], 1)  # on the side of ea
    half_b = np.stack([z, p2, eb], 1)  # on the side of eb
    a_pos = side[ea] > 0
    pos_split = _drop_degenerate(np.concatenate([lonely_tri[lp], quad1[ln], quad2[ln],
                                                 half_a[a_pos], half_b[~a_pos]]))
    neg_split = _drop_degenerate(np.concatenate([lonely_tri[ln], quad1[lp], quad2[lp],
                                                 half_a[~a_pos], half_b[a_pos]]))
    del t1, t2

    report(progress, 0.45, "Assembling parts")
    allv = np.concatenate([V, newpts])
    on_plane = np.zeros(len(allv), dtype=bool)
    zero = np.nonzero(side == 0)[0]
    on_plane[zero] = True
    on_plane[mesh.n_vertices:] = True
    if len(zero):
        allv[zero] = (V[zero].astype(np.float64) - d[zero, None] * n).astype(np.float32)
    del d, side
    pos_faces = np.concatenate([F[pos_f], pos_split])
    neg_faces = np.concatenate([F[neg_f], neg_split])
    del pos_f, neg_f

    report(progress, 0.6, "Building caps")
    warnings = []
    cap_p, loops, open_p = _cap(allv, pos_faces, on_plane, n, warnings)
    cap_n, _, open_n = _cap(allv, neg_faces, on_plane, n, warnings)
    report(progress, 0.75, "Assembling parts")
    positive = Mesh(allv, np.concatenate([pos_faces, cap_p])).compact()
    report(progress, 0.88, "Assembling parts")
    negative = Mesh(allv, np.concatenate([neg_faces, cap_n])).compact()
    open_loops = max(open_p, open_n)
    if open_loops:
        warnings.append("open_cut")
    report(progress, 1.0, "Done")
    return CutResult(positive, negative, loops, open_loops, sorted(set(warnings)))


def _drop_degenerate(f: np.ndarray) -> np.ndarray:
    return f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 2] != f[:, 0])]


def _open_edges_on_plane(faces: np.ndarray, on_plane: np.ndarray):
    """Directed half-edges (a, b) of `faces` that lie on the plane and are used once."""
    cand = faces[on_plane[faces].sum(1) >= 2]
    if not len(cand):
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    a = cand.ravel().astype(np.int64)
    b = np.roll(cand, -1, axis=1).ravel().astype(np.int64)
    keep = on_plane[a] & on_plane[b]
    a, b = a[keep], b[keep]
    big = np.int64(on_plane.size)
    key = np.minimum(a, b) * big + np.maximum(a, b)
    _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    once = cnt[inv.ravel()] == 1
    return a[once], b[once]


def _trace_loops(seg_from: np.ndarray, seg_to: np.ndarray):
    """Chain directed segments into loops; returns (loops, number of open chains)."""
    out: dict[int, list[int]] = {}
    has_in = set(seg_to.tolist())
    for x, y in zip(seg_from.tolist(), seg_to.tolist()):
        out.setdefault(x, []).append(y)
    loops, open_count = [], 0
    starts = [x for x in seg_from.tolist() if x not in has_in] + seg_from.tolist()
    for s0 in starts:
        if not out.get(s0):
            continue
        loop = [s0]
        cur = out[s0].pop()
        while cur != s0:
            nxt = out.get(cur)
            if not nxt:
                open_count += 1
                break
            loop.append(cur)
            cur = nxt.pop()
        loops += _split_pinched(loop)
    return [np.array(lp, dtype=np.int64) for lp in loops], open_count


def _split_pinched(loop: list) -> list:
    """Split a loop that passes a vertex twice into simple loops."""
    out, stack, where = [], [], {}
    for v in loop:
        if v in where:
            i = where[v]
            sub = stack[i:]
            for x in sub[1:]:
                del where[x]
            del stack[i + 1:]
            if len(sub) >= 3:
                out.append(sub)
        else:
            where[v] = len(stack)
            stack.append(v)
    if len(stack) >= 3:
        out.append(stack)
    return out


def _collinear_mask(xy: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """True for ring points lying exactly on the segment between their neighbours."""
    drop = np.zeros(len(xy), dtype=bool)
    start = 0
    for end in ends.tolist():
        r = xy[start:end]
        if len(r) > 3:
            prev, nxt = np.roll(r, 1, axis=0), np.roll(r, -1, axis=0)
            e1, e2 = r - prev, nxt - r
            cross = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
            scale = np.linalg.norm(e1, axis=1) * np.linalg.norm(e2, axis=1)
            straight = (np.abs(cross) <= 1e-12 * scale) & ((e1 * e2).sum(1) > 0)
            if (~straight).sum() >= 3:
                drop[start:end] = straight
        start = end
    return drop


def _triangulate_rings(xy: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """earcut on rings (outer + holes) that uses every ring point.

    earcut is unreliable with exactly collinear points (box sides, CAD models),
    so those are taken out first and put back afterwards.
    """
    drop = _collinear_mask(xy, ends)
    keep = np.nonzero(~drop)[0]
    starts = np.concatenate([[0], ends[:-1]]).astype(np.int64)
    red_ends = np.array([np.count_nonzero(~drop[s:e]) for s, e in zip(starts, ends)], dtype=np.int64).cumsum()
    tri = earcut.triangulate_float64(xy[keep], red_ends.astype(np.uint32)).reshape(-1, 3).astype(np.int64)
    if not len(tri):
        return tri
    return _restore_dropped(keep[tri], ends)


def _restore_dropped(tri: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Put back ring points earcut skipped (collinear ones, e.g. on box sides).

    Each run of skipped points lies on a polygon edge (a, b) earcut kept; the
    triangle on that edge is replaced by a fan through the run, so every
    boundary point is used and the cap stays watertight.
    """
    used = np.zeros(int(ends[-1]), dtype=bool)
    used[tri.ravel()] = True
    if used.all():
        return tri
    tris = [tuple(t) for t in tri.tolist()]
    start = 0
    for end in ends.tolist():
        ring = list(range(start, end))
        start = end
        if all(used[ring]):
            continue
        k0 = next((i for i, v in enumerate(ring) if used[v]), None)
        if k0 is None:
            continue
        ring = ring[k0:] + ring[:k0]  # begin at a used point
        i = 0
        while i < len(ring):
            j = i + 1
            while j < len(ring) and not used[ring[j]]:
                j += 1
            if j > i + 1:
                a, run, b = ring[i], ring[i + 1:j], ring[j % len(ring)]
                for ti, (x, y, z) in enumerate(tris):
                    cyc = [(x, y, z), (y, z, x), (z, x, y)]
                    hit = next((c for c in cyc if {c[0], c[1]} == {a, b}), None)
                    if hit is not None:
                        path = [a, *run, b] if hit[0] == a else [b, *run[::-1], a]
                        tris[ti:ti + 1] = [(path[m], path[m + 1], hit[2]) for m in range(len(path) - 1)]
                        break
            i = j
    return np.array(tris, dtype=np.int64)


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


def _cap(verts: np.ndarray, faces: np.ndarray, on_plane: np.ndarray, normal, warnings: list):
    """Faces that close one part's open edges on the plane: (faces, loops, open chains)."""
    a, b = _open_edges_on_plane(faces, on_plane)
    if not len(a):
        return np.zeros((0, 3), dtype=np.int64), 0, 0
    loops, open_loops = _trace_loops(b, a)  # the cap runs opposite to the part's edges
    if not loops:
        return np.zeros((0, 3), dtype=np.int64), 0, open_loops
    u, w = plane_basis(normal)
    polys = [np.stack([verts[lp] @ u, verts[lp] @ w], 1).astype(np.float64) for lp in loops]
    areas = np.array([_signed_area(p) for p in polys])
    outer_sign = np.sign(areas[np.argmax(np.abs(areas))]) or 1.0
    is_outer = np.sign(areas) == outer_sign
    outer_ids = np.nonzero(is_outer)[0]
    bbox = np.array([[p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()] for p in polys])
    holes_of = {i: [] for i in outer_ids}
    by_size = outer_ids[np.argsort(np.abs(areas[outer_ids]))]
    for h in np.nonzero(~is_outer)[0]:
        pt = polys[h].mean(0) if len(polys[h]) < 3 else polys[h][0]
        for o in by_size:
            bb = bbox[o]
            if bb[0] <= pt[0] <= bb[2] and bb[1] <= pt[1] <= bb[3] and _point_in_polygon(pt, polys[o]):
                holes_of[o].append(h)
                break
        else:
            warnings.append("orphan_hole")
    out = []
    for o in outer_ids:
        rings = [o] + holes_of[o]
        ring_xy = np.concatenate([polys[r] for r in rings])
        ring_ids = np.concatenate([loops[r] for r in rings])
        ends = np.cumsum([len(polys[r]) for r in rings]).astype(np.uint32)
        expected = len(ring_ids) + 2 * (len(rings) - 1) - 2
        tri = _triangulate_rings(ring_xy, ends)
        if len(tri) != expected:
            warnings.append("cap_fallback")
            k = len(polys[o])
            i = np.arange(1, k - 1)
            tri = np.stack([np.zeros_like(i), i, i + 1], 1)
            ring_ids, ring_xy = loops[o], polys[o]
        ta = ring_xy[tri]
        sa = ((ta[:, 1, 0] - ta[:, 0, 0]) * (ta[:, 2, 1] - ta[:, 0, 1])
              - (ta[:, 2, 0] - ta[:, 0, 0]) * (ta[:, 1, 1] - ta[:, 0, 1]))
        wrong = np.sign(sa) == -outer_sign
        tri[wrong] = tri[wrong][:, ::-1]
        out.append(ring_ids[tri])
    return np.concatenate(out), len(loops), open_loops
