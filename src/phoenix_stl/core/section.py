"""Cross-sections as closed 2D outlines, joint-face detection, solid envelope.

Used by the connector tools: the outline of a cut face tells where pins fit,
and the envelope (a hollow part with its cavities filled) tells where a solid
boss may be added inside a hollow model.
"""
from __future__ import annotations

import math

import manifold3d as m3d
import numpy as np

from .cut import _signed_distance, unit
from .mesh import CHUNK, Mesh
from .surface_cut import Frame


# -- outlines -------------------------------------------------------------------------

def section_loops(mesh: Mesh, normal, origin) -> list[np.ndarray]:
    """Closed outlines (each (m, 3) float64) where the plane cuts the mesh.

    Segments are linked through the mesh edges they cross, so loops close
    exactly; chains left open by holes in the mesh are dropped.
    """
    n = unit(normal)
    if mesh.n_faces == 0:
        return []
    d = _signed_distance(mesh, n, float(n @ np.asarray(origin, dtype=np.float64)))
    return _loops(mesh.vertices, mesh.faces, d)


def _loops(vertices: np.ndarray, faces: np.ndarray, d: np.ndarray) -> list[np.ndarray]:
    """Outlines where the per-vertex signed distance d crosses zero (faces may be a subset)."""
    if not len(faces):
        return []
    used = np.unique(faces)
    span = float(np.ptp(d[used])) or 1.0
    tiny = 1e-9 * span
    for _ in range(4):  # keep vertices off the plane so every crossing is a clean edge point
        if not (np.abs(d[used]) < tiny).any():
            break
        d = d - 3 * tiny
    s = d[faces] >= 0
    cnt = s.sum(1)
    st = (cnt == 1) | (cnt == 2)
    if not st.any():
        return []
    f = faces[st].astype(np.int64)
    s = s[st]
    lonely = np.where(cnt[st] == 1, np.argmax(s, 1), np.argmin(s, 1))
    rot = (lonely[:, None] + np.arange(3)[None, :]) % 3
    t = np.take_along_axis(f, rot, 1)
    nv = len(vertices)
    e1 = np.sort(t[:, [0, 1]], 1)
    e2 = np.sort(t[:, [0, 2]], 1)
    k1 = e1[:, 0] * nv + e1[:, 1]
    k2 = e2[:, 0] * nv + e2[:, 1]
    keys, inv = np.unique(np.concatenate([k1, k2]), return_inverse=True)
    m = len(f)
    a_node, b_node = inv[:m], inv[m:]
    # point of each node (edge crossing)
    eu, ev = keys // nv, keys % nv
    du, dv = d[eu], d[ev]
    w = (du / (du - dv))[:, None]
    pu = vertices[eu].astype(np.float64)
    pts = pu + w * (vertices[ev].astype(np.float64) - pu)
    return [pts[loop] for loop in _cycles(len(keys), a_node, b_node)]


def _cycles(n_nodes: int, a: np.ndarray, b: np.ndarray) -> list[np.ndarray]:
    """Closed cycles of an undirected graph where nodes have degree 2."""
    ends = np.concatenate([a, b])
    segs = np.concatenate([np.arange(len(a)), np.arange(len(a))])
    order = np.argsort(ends, kind="stable")
    ends, segs = ends[order], segs[order]
    deg = np.bincount(ends, minlength=n_nodes)
    start = np.concatenate([[0], np.cumsum(deg)])
    used = np.zeros(len(a), dtype=bool)
    bad = deg != 2
    loops = []
    for s0 in range(len(a)):
        if used[s0] or bad[a[s0]] or bad[b[s0]]:
            continue
        loop = [a[s0]]
        seg, node = s0, b[s0]
        used[seg] = True
        ok = True
        while node != loop[0]:
            if bad[node]:
                ok = False
                break
            loop.append(node)
            i0 = start[node]
            nxt = segs[i0] if segs[i0] != seg else segs[i0 + 1]
            if used[nxt]:
                ok = False
                break
            used[nxt] = True
            node = b[nxt] if a[nxt] == node else a[nxt]
            seg = nxt
        if ok and len(loop) >= 3:
            loops.append(np.asarray(loop))
    return loops


def to_cross_section(loops3d, frame: Frame, simplify: float = 0.0) -> m3d.CrossSection:
    """Outlines in the frame's (u, v) plane as a manifold3d CrossSection (even-odd fill)."""
    polys = []
    for loop in loops3d:
        xy = frame.to_local(loop)[:, :2]
        keep = np.r_[True, np.any(np.abs(np.diff(xy, axis=0)) > 1e-12, axis=1)]
        xy = xy[keep]
        if len(xy) >= 3:
            polys.append(np.ascontiguousarray(xy, dtype=np.float64))
    if not polys:
        return m3d.CrossSection()
    cs = m3d.CrossSection(polys, m3d.FillRule.EvenOdd)
    return cs.simplify(simplify) if simplify > 0 else cs


def polygons(cs: m3d.CrossSection) -> list[np.ndarray]:
    return [np.asarray(p, dtype=np.float64) for p in cs.to_polygons()]


def cross_section_of(polys) -> m3d.CrossSection:
    polys = [np.ascontiguousarray(p, dtype=np.float64) for p in polys if len(p) >= 3]
    return m3d.CrossSection(polys, m3d.FillRule.EvenOdd) if polys else m3d.CrossSection()


def fill_holes(cs: m3d.CrossSection) -> m3d.CrossSection:
    """The outline with every hole filled (outer contours only)."""
    outer = [p for p in polygons(cs) if _area(p) > 0]
    return m3d.CrossSection(outer, m3d.FillRule.Positive) if outer else m3d.CrossSection()


def _area(p: np.ndarray) -> float:
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def section_stack(mesh: Mesh, normal, origin, offsets) -> list[list[np.ndarray]]:
    """section_loops at several offsets along the normal.

    Distances are computed once; each level then only looks at the faces it
    crosses, so many levels cost little more than one.
    """
    n = unit(normal)
    o = np.asarray(origin, dtype=np.float64)
    offsets = list(offsets)
    if mesh.n_faces == 0 or not offsets:
        return [[] for _ in offsets]
    d = _signed_distance(mesh, n, float(n @ o))
    fmin = np.empty(mesh.n_faces, dtype=np.float64)
    fmax = np.empty(mesh.n_faces, dtype=np.float64)
    for s0 in range(0, mesh.n_faces, CHUNK):
        df = d[mesh.faces[s0:s0 + CHUNK]]
        fmin[s0:s0 + CHUNK] = df.min(1)
        fmax[s0:s0 + CHUNK] = df.max(1)
    out = []
    for t in offsets:
        pad = 1e-6 * (1.0 + abs(t))
        crossing = np.nonzero((fmin <= t + pad) & (fmax >= t - pad))[0]
        out.append(_loops(mesh.vertices, mesh.faces[crossing], d - t))
    return out


# -- joint detection -------------------------------------------------------------------------

def _face_planes(mesh: Mesh, box=None):
    """Unit normals, areas, centroids of faces (optionally only those with centroid in box)."""
    ns, areas, cents = [], [], []
    v = mesh.vertices
    for s in range(0, mesh.n_faces, CHUNK):
        tri = v[mesh.faces[s:s + CHUNK]].astype(np.float64)
        c = tri.mean(1)
        if box is not None:
            inside = np.all((c >= box[0]) & (c <= box[1]), axis=1)
            tri, c = tri[inside], c[inside]
        cr = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        a = np.linalg.norm(cr, axis=1)
        ok = a > 0
        ns.append((cr[ok] / a[ok, None]).astype(np.float32))
        areas.append(0.5 * a[ok])
        cents.append(c[ok].astype(np.float32))
    if not ns:
        return np.zeros((0, 3), np.float32), np.zeros(0), np.zeros((0, 3), np.float32)
    return np.concatenate(ns), np.concatenate(areas), np.concatenate(cents)


def _cluster(n, a, c, seed_n, seed_d, cos_tol, tol, mask=None):
    member = (n @ seed_n.astype(np.float32) > cos_tol) & (np.abs(c @ seed_n.astype(np.float32) - seed_d) < tol)
    if mask is not None:
        member &= mask
    if not member.any():
        return None
    w = a[member]
    nm = (n[member].astype(np.float64) * w[:, None]).sum(0)
    nm /= np.linalg.norm(nm)
    cm = (c[member].astype(np.float64) * w[:, None]).sum(0) / w.sum()
    return {"normal": nm, "d": float(cm @ nm), "center": cm, "area": float(w.sum()), "member": member}


def planar_clusters(mesh: Mesh, box=None, k: int = 8, angle_deg: float = 1.0, tol: float | None = None):
    """Largest flat regions of a mesh: [{normal, d, center, area}], biggest first.

    Faces are first binned by plane (normal and offset) and the bins ranked by
    total area, so a flat face made of many thin triangles (a cut cap) is not
    outranked by a few big facets; each bin then seeds an exact cluster.
    """
    n, a, c = _face_planes(mesh, box)
    if not len(a):
        return []
    if tol is None:
        tol = 1e-4 * float(np.linalg.norm(mesh.size)) + 0.01
    cos_tol = math.cos(math.radians(angle_deg))
    d = np.einsum("ij,ij->i", c.astype(np.float64), n.astype(np.float64))
    q = np.empty((len(a), 4), dtype=np.int64)
    q[:, :3] = np.round(n.astype(np.float64) * (1.0 / math.radians(max(angle_deg, 0.25))))
    q[:, 3] = np.round(d / (2 * tol))
    _, inv = np.unique(q, axis=0, return_inverse=True)
    inv = inv.ravel()
    bin_area = np.bincount(inv, weights=a)
    taken = np.zeros(len(a), dtype=bool)
    out = []
    for b in np.argsort(bin_area)[::-1][:4 * k]:
        sel = np.nonzero((inv == b) & ~taken)[0]
        if not len(sel):
            continue
        i = sel[np.argmax(a[sel])]
        cl = _cluster(n, a, c, n[i].astype(np.float64), float(d[i]), cos_tol, tol, ~taken)
        if cl is None:
            continue
        taken |= cl.pop("member")
        out.append(cl)
        if len(out) >= 2 * k:
            break
    out.sort(key=lambda x: -x["area"])
    return out[:k]


def find_joint(a: Mesh, b: Mesh, angle_deg: float = 1.0):
    """Shared flat face of two touching parts.

    Returns {normal (pointing out of a, into b), origin, area} or None.
    """
    ba, bb = a.bounds, b.bounds
    size = float(max(np.linalg.norm(a.size), np.linalg.norm(b.size)))
    tol = 1e-4 * size + 0.02
    lo = np.maximum(ba[0], bb[0]) - tol
    hi = np.minimum(ba[1], bb[1]) + tol
    if np.any(lo > hi):
        return None
    box = np.array([lo, hi])
    ca = planar_clusters(a, box, angle_deg=angle_deg, tol=tol)
    cb = planar_clusters(b, box, angle_deg=angle_deg, tol=tol)
    cos_tol = math.cos(math.radians(angle_deg))
    best = None
    for x in ca:
        for y in cb:
            if x["normal"] @ y["normal"] < -cos_tol and abs(x["d"] + y["d"]) < tol:
                score = min(x["area"], y["area"])
                if best is None or score > best[0]:
                    best = (score, x, y)
    if best is None:
        return None
    _, x, y = best
    nrm = x["normal"] - y["normal"]
    nrm /= np.linalg.norm(nrm)
    d = 0.5 * (x["center"] @ nrm + y["center"] @ nrm)
    origin = x["center"] + (d - x["center"] @ nrm) * nrm
    return {"normal": nrm, "origin": origin, "area": best[0]}


def flat_face_at(mesh: Mesh, point, normal, angle_deg: float = 4.0):
    """The flat region through a clicked point: {normal (outward), origin, area} or None."""
    n0 = unit(normal)
    p = np.asarray(point, float)
    n, a, c = _face_planes(mesh)
    size = float(np.linalg.norm(mesh.size)) or 1.0
    cl = _cluster(n, a, c, n0, float(p @ n0), math.cos(math.radians(angle_deg)), 2e-3 * size + 0.05)
    if cl is None:
        return None
    # refine with the measured plane and a tighter tolerance
    cl2 = _cluster(n, a, c, cl["normal"], cl["d"], math.cos(math.radians(1.0)), 1e-4 * size + 0.01) or cl
    nrm = cl2["normal"]
    origin = p + (cl2["d"] - p @ nrm) * nrm
    return {"normal": nrm, "origin": origin, "area": cl2["area"]}
