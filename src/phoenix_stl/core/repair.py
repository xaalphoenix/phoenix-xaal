"""Basic automatic repair: weld, clean, orient, fill holes, drop debris.

One topology pass decides which steps are needed, so a clean model costs
little more than an analysis. Self-intersections, non-manifold edges and voxel
rebuild come in phase 5.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import mapbox_earcut as earcut
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .analyze import Topology, boundary_half_edges, degenerate_mask, duplicate_mask, edge_info
from .mesh import CHUNK, Mesh, report
from .weld import weld_mesh

MAX_NESTING_TESTS = 2000


@dataclass
class RepairOptions:
    merge_distance: float = 0.0  # mm; 0 = only exactly equal vertices
    remove_degenerate: bool = True
    remove_duplicates: bool = True
    remove_debris: bool = True
    debris_ratio: float = 0.0025  # shells smaller than this fraction of the model diagonal
    fix_orientation: bool = True
    fill_holes: bool = True
    max_hole_edges: int = 0  # 0 = fill every hole

    @classmethod
    def from_dict(cls, d: dict | None) -> "RepairOptions":
        d = d or {}
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict:
        return asdict(self)


def repair(mesh: Mesh, opts: RepairOptions | None = None, progress=None):
    """Returns (repaired Mesh, log dict of what changed)."""
    opts = opts or RepairOptions()
    log = {}

    def sub(a, b):
        return lambda f, m: report(progress, a + (b - a) * f, m)

    if opts.merge_distance > 0:
        before = mesh.n_vertices
        mesh = weld_mesh(mesh, opts.merge_distance, sub(0.0, 0.1))
        log["merged_vertices"] = before - mesh.n_vertices
    topo = Topology.of(mesh, sub(0.1, 0.35))

    drop = np.zeros(mesh.n_faces, dtype=bool)
    if opts.remove_degenerate:
        bad = degenerate_mask(mesh, topo.areas)
        log["removed_degenerate"] = int(bad.sum())
        drop |= bad
    if opts.remove_duplicates:
        dup = duplicate_mask(mesh, topo.edges) & ~drop
        log["removed_duplicates"] = int(dup.sum())
        drop |= dup
    if drop.any():
        mesh = Mesh(mesh.vertices, mesh.faces[~drop]).compact()
        topo = Topology.of(mesh, sub(0.35, 0.5))
    report(progress, 0.5, "Removing floating debris")

    if opts.remove_debris and topo.n_shells > 1:
        mesh, removed = _remove_debris(mesh, topo, opts.debris_ratio)
        log["removed_debris_shells"] = removed
        if removed:
            topo = Topology.of(mesh, sub(0.5, 0.6))
    if opts.fix_orientation and mesh.n_faces:
        report(progress, 0.6, "Fixing normals")
        mesh, flipped = _fix_orientation(mesh, topo, sub(0.6, 0.85))
        log["flipped_faces"] = flipped
    if opts.fill_holes and mesh.n_faces and (topo.edges.count == 1).any():
        report(progress, 0.85, "Filling holes")
        mesh, filled, skipped = _fill_holes(mesh, topo, opts.max_hole_edges)
        log["filled_holes"] = filled
        if skipped:
            log["skipped_holes"] = skipped
    report(progress, 1.0, "Done")
    return mesh, log


def _shell_bounds(mesh: Mesh, fs: np.ndarray, n_sh: int):
    order = np.argsort(fs, kind="stable")
    starts = np.searchsorted(fs[order], np.arange(n_sh))
    lo = np.empty((n_sh, 3))
    hi = np.empty((n_sh, 3))
    for k in range(3):
        c = mesh.vertices[:, k][mesh.faces[order]]
        lo[:, k] = np.minimum.reduceat(c.min(1), starts)
        hi[:, k] = np.maximum.reduceat(c.max(1), starts)
    return lo, hi


def _remove_debris(mesh: Mesh, topo: Topology, ratio: float):
    lo, hi = _shell_bounds(mesh, topo.face_shell, topo.n_shells)
    diag = np.linalg.norm(hi - lo, axis=1)
    small = diag < ratio * float(np.linalg.norm(mesh.size))
    if not small.any() or small.all():
        return mesh, 0
    return Mesh(mesh.vertices, mesh.faces[~small[topo.face_shell]]).compact(), int(small.sum())


def _fix_orientation(mesh: Mesh, topo: Topology, progress=None):
    m = mesh.n_faces
    faces = mesh.faces
    vols = topo.vols
    flipped = 0
    e = topo.edges
    if ((e.count == 2) & ~e.consistent).any():
        report(progress, 0.1, "Making winding consistent")
        eh = edge_info(mesh, with_half_edges=True)
        two = eh.count == 2
        p = eh.start[two]
        f1 = eh.half_order[p] // 3
        f2 = eh.half_order[p + 1] // 3
        same = eh.consistent[two]
        del eh
        rows = np.concatenate([f1[same], f1[same] + m, f1[~same], f1[~same] + m])
        cols = np.concatenate([f2[same], f2[same] + m, f2[~same] + m, f2[~same]])
        g = coo_matrix((np.ones(len(rows), np.int8), (rows, cols)), shape=(2 * m, 2 * m))
        _, lab = connected_components(g, directed=False)
        flip = lab[:m] > lab[m:]
        if flip.any():
            faces = faces.copy()
            faces[flip] = faces[flip][:, ::-1]
            vols = np.where(flip, -vols, vols)
            flipped += int(flip.sum())
    report(progress, 0.7, "Orienting shells outward")
    fs, n_sh = topo.face_shell, topo.n_shells
    vol = np.bincount(fs, weights=vols, minlength=n_sh)
    depth = _nesting_depth(Mesh(mesh.vertices, faces), fs, n_sh, vol)
    wrong = ((vol < 0) == (depth % 2 == 0)) & (depth >= 0)  # depth -1: unknown, leave
    if wrong.any():
        fl = wrong[fs]
        if faces is mesh.faces:
            faces = faces.copy()
        faces[fl] = faces[fl][:, ::-1]
        flipped += int(fl.sum())
    return (Mesh(mesh.vertices, faces) if flipped else mesh), flipped


def _nesting_depth(mesh: Mesh, fs, n_sh, vol) -> np.ndarray:
    """How many other shells contain each shell (-1 when not determined)."""
    depth = np.zeros(n_sh, dtype=np.int64)
    if n_sh == 1:
        return depth
    lo, hi = _shell_bounds(mesh, fs, n_sh)
    absvol = np.abs(vol)
    order = np.argsort(fs, kind="stable")
    starts = np.searchsorted(fs[order], np.arange(n_sh + 1))
    tests = 0
    for i in range(n_sh):
        cand = np.nonzero(np.all(lo <= lo[i], 1) & np.all(hi >= hi[i], 1) & (absvol > absvol[i]))[0]
        if len(cand) == 0:
            continue
        if tests + len(cand) > MAX_NESTING_TESTS:
            depth[i] = -1
            continue
        pt = mesh.vertices[mesh.faces[order[starts[i]], 0]].astype(np.float64)
        for j in cand:
            tests += 1
            if _ray_parity(mesh.vertices, mesh.faces[order[starts[j]:starts[j + 1]]], pt):
                depth[i] += 1
    return depth


_RAY = np.array([0.5773502, 0.5773507, 0.5773497])
_RAY /= np.linalg.norm(_RAY)


def _ray_parity(verts, faces, origin) -> bool:
    """Odd number of ray hits = origin is inside the closed surface."""
    hits = 0
    for s in range(0, len(faces), CHUNK):
        t = verts[faces[s:s + CHUNK]].astype(np.float64)
        v0 = t[:, 0]
        e1 = t[:, 1] - v0
        e2 = t[:, 2] - v0
        pvec = np.cross(_RAY, e2)
        det = np.einsum("ij,ij->i", e1, pvec)
        ok = np.abs(det) > 1e-18
        inv = np.zeros_like(det)
        inv[ok] = 1.0 / det[ok]
        tvec = origin - v0
        u = np.einsum("ij,ij->i", tvec, pvec) * inv
        qvec = np.cross(tvec, e1)
        v = (qvec @ _RAY) * inv
        dist = np.einsum("ij,ij->i", e2, qvec) * inv
        hits += int(np.count_nonzero(ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (dist > 1e-9)))
    return hits % 2 == 1


def _fill_holes(mesh: Mesh, topo: Topology, max_edges: int):
    a, b = boundary_half_edges(mesh, topo.edges)
    if not len(a):
        return mesh, 0, 0
    nkey = np.int64(mesh.n_vertices)
    edge_keys = topo.edges.keys(mesh.n_vertices)  # sorted
    loops = _boundary_loops(a, b)
    verts = mesh.vertices.astype(np.float64)
    new_faces, new_verts = [], []
    nv = mesh.n_vertices
    filled = skipped = 0
    for loop in loops:
        if max_edges and len(loop) > max_edges:
            skipped += 1
            continue
        poly = loop[::-1]  # fill runs opposite to the boundary half-edges
        tri = _triangulate_loop(verts[poly])
        if tri is not None and _reuses_edges(poly[tri], poly, edge_keys, nkey):
            tri = None  # a diagonal already exists elsewhere: fan is always safe
        if tri is None:
            c = nv + len(new_verts)
            new_verts.append(verts[poly].mean(0))
            new_faces.append(np.stack([poly, np.roll(poly, -1), np.full(len(poly), c)], 1))
        else:
            new_faces.append(poly[tri])
        filled += 1
    if not new_faces:
        return mesh, 0, skipped
    V = mesh.vertices
    if new_verts:
        V = np.concatenate([V, np.array(new_verts, dtype=np.float32)])
    return Mesh(V, np.concatenate([mesh.faces, np.concatenate(new_faces)])), filled, skipped


def _reuses_edges(tris, poly, edge_keys, nkey) -> bool:
    e = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    k = np.minimum(e[:, 0], e[:, 1]) * nkey + np.maximum(e[:, 0], e[:, 1])
    b = np.minimum(poly, np.roll(poly, -1)) * nkey + np.maximum(poly, np.roll(poly, -1))
    k = np.setdiff1d(k, b)
    if not len(k) or not len(edge_keys):
        return False
    i = np.clip(np.searchsorted(edge_keys, k), 0, len(edge_keys) - 1)
    return bool(np.any(edge_keys[i] == k))


def _split_pinched(loop: list) -> list:
    """Split a loop that visits a vertex twice into simple loops."""
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


def _boundary_loops(a: np.ndarray, b: np.ndarray):
    out = {}
    for x, y in zip(a.tolist(), b.tolist()):
        out.setdefault(x, []).append(y)
    loops = []
    for start in a.tolist():
        if not out.get(start):
            continue
        loop = [start]
        cur = out[start].pop()
        while cur != start:
            nxts = out.get(cur)
            if not nxts:
                break  # broken chain; close it as is
            loop.append(cur)
            cur = nxts.pop()
        loops += [np.array(x, dtype=np.int64) for x in _split_pinched(loop)]
    return loops


def _triangulate_loop(pts: np.ndarray):
    """Triangles (local indices) following the loop order, or None to use a fan."""
    k = len(pts)
    if k == 3:
        return np.array([[0, 1, 2]])
    c = pts - pts.mean(0)
    try:
        _, _, vt = np.linalg.svd(c, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    xy = c @ vt[:2].T
    tri = earcut.triangulate_float64(xy, np.array([k], dtype=np.uint32)).reshape(-1, 3).astype(np.int64)
    if len(tri) != k - 2:
        return None
    x, y = xy[:, 0], xy[:, 1]
    poly_sign = np.sign(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))
    ta = xy[tri]
    sa = ((ta[:, 1, 0] - ta[:, 0, 0]) * (ta[:, 2, 1] - ta[:, 0, 1])
          - (ta[:, 2, 0] - ta[:, 0, 0]) * (ta[:, 1, 1] - ta[:, 0, 1]))
    wrong = np.sign(sa) == -poly_sign
    tri[wrong] = tri[wrong][:, ::-1]
    return tri
