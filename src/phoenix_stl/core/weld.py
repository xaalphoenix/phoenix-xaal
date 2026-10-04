"""Vertex welding: turn a triangle soup into an indexed mesh.

np.unique(axis=0) on 45M xyz rows is slow and memory hungry, so each point is
hashed to 64 bits (a float64 dot product, then bit mixing), the hash is packed
together with the point index into one uint64, and that single array is
sorted. Every point is then verified against its group's representative;
groups hit by a hash collision are regrouped exactly, so two different points
are never merged.
"""
from __future__ import annotations

import numpy as np

from .mesh import CHUNK, Mesh, report

_COEF = np.array([1.0, 3.141592653589793, 2.718281828459045, 1.4142135623730951])
_MIX = np.uint64(0x9E3779B185EBCA87)


def _canon(points: np.ndarray, tol: float) -> np.ndarray:
    """Canonical form used for both hashing and equality: exact or quantized."""
    if tol > 0:
        return np.floor(points.astype(np.float64) / tol + 0.5)
    return points.astype(np.float64) + 0.0  # +0.0 maps -0.0 to 0.0


def group_rows(n: int, rows_at, progress=None):
    """Exact grouping of n rows; rows_at(index_or_slice) -> (k, d) float64.

    Returns (first, inverse): first[g] is a row index representing group g,
    inverse[i] is the group of row i.
    """
    if n == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    bits = max(1, int(n - 1).bit_length())
    if bits > 40:
        raise ValueError("too many rows to group")
    ubits = np.uint64(bits)
    packed = np.empty(n, dtype=np.uint64)
    for s in range(0, n, CHUNK):
        e = min(s + CHUNK, n)
        r = rows_at(slice(s, e))
        h = (r @ _COEF[:r.shape[1]]).view(np.uint64)
        h *= _MIX
        h ^= h >> np.uint64(29)
        h >>= ubits
        h <<= ubits
        h |= np.arange(s, e, dtype=np.uint64)
        packed[s:e] = h
        report(progress, 0.25 * e / n, "hashing")
    report(progress, 0.25, "sorting")
    packed.sort()
    report(progress, 0.45, "grouping")
    order = (packed & np.uint64((1 << bits) - 1)).astype(np.int64)
    packed >>= ubits
    starts = np.empty(n, dtype=bool)
    starts[0] = True
    np.not_equal(packed[1:], packed[:-1], out=starts[1:])
    del packed
    group = np.cumsum(starts, dtype=np.int64)
    group -= 1
    first = order[starts]
    del starts
    inverse = np.empty(n, dtype=np.int64)
    inverse[order] = group
    del order, group
    report(progress, 0.7, "verifying")

    rep = rows_at(first)
    bad_groups = []
    for s in range(0, n, CHUNK):
        e = min(s + CHUNK, n)
        g = inverse[s:e]
        ok = np.all(rows_at(slice(s, e)) == rep[g], axis=1)
        if not ok.all():
            bad_groups.append(np.unique(g[~ok]))
        report(progress, 0.7 + 0.3 * e / n, "verifying")
    del rep
    if bad_groups:
        groups = np.unique(np.concatenate(bad_groups))
        members = np.nonzero(np.isin(inverse, groups))[0]
        _, sub = np.unique(rows_at(members), axis=0, return_inverse=True)
        inverse[members] = len(first) + sub.ravel()
        used = np.zeros(int(inverse.max()) + 1, dtype=bool)
        used[inverse] = True
        remap = np.cumsum(used, dtype=np.int64) - 1
        first_all = np.full(len(used), -1, dtype=np.int64)
        first_all[inverse[::-1]] = np.arange(n - 1, -1, -1)  # lowest index wins
        first = first_all[used]
        inverse = remap[inverse]
    return first, inverse


def weld_points(points: np.ndarray, tol: float = 0.0, progress=None):
    """Weld (n, 3) points. Returns (unique (u, 3) float32, inverse (n,) int32)."""
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    first, inverse = group_rows(len(points), lambda ix: _canon(points[ix], tol), progress)
    return points[first], inverse.astype(np.int32)


def weld_soup(tris: np.ndarray, tol: float = 0.0, progress=None) -> Mesh:
    """(m, 3, 3) triangle soup -> indexed Mesh."""
    verts, inv = weld_points(np.asarray(tris, dtype=np.float32).reshape(-1, 3), tol, progress)
    return Mesh(verts, inv.reshape(-1, 3))


def weld_mesh(mesh: Mesh, tol: float, progress=None) -> Mesh:
    """Merge vertices of an indexed mesh that are within tol of each other."""
    verts, inv = weld_points(mesh.vertices, tol, progress)
    return Mesh(verts, inv[mesh.faces])
