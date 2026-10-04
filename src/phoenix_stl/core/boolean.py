"""Boolean operations through manifold3d (robust, always-closed results).

manifold3d holds the GIL and needs roughly 500 bytes per triangle, so these
run only in the engine process, after a memory check.
"""
from __future__ import annotations

import manifold3d as m3d
import numpy as np

from .mesh import Mesh

# Parts that only touch (e.g. two halves of a cut) are not fused by an exact
# union: the shared faces survive as zero-thickness leftovers. Growing each
# part by this relative amount (2 micrometres on a 200 mm part, far below a
# 14 micrometre printer pixel) makes them overlap so they fuse cleanly.
FUSE_GROWTH = 1e-5


class NotSolid(Exception):
    """A mesh is not a closed, consistently oriented solid (repair it first)."""


def to_manifold(mesh: Mesh) -> m3d.Manifold:
    # Fresh owned copies: manifold3d rejects views into other buffers.
    man = m3d.Manifold(m3d.Mesh(vert_properties=np.array(mesh.vertices, dtype=np.float32, order="C"),
                                tri_verts=np.array(mesh.faces, dtype=np.uint32, order="C")))
    if man.status() != m3d.Error.NoError:
        raise NotSolid(str(man.status()))
    return man


def _is_manifold(mesh: Mesh) -> bool:
    from .analyze import edge_info
    f = mesh.faces
    if np.any((f[:, 0] == f[:, 1]) | (f[:, 1] == f[:, 2]) | (f[:, 2] == f[:, 0])):
        return False
    return bool(np.all(edge_info(mesh).count == 2))


def _extra_copies(v: np.ndarray) -> np.ndarray:
    """Indices of float32 vertices at the same position as an earlier one."""
    b = np.ascontiguousarray(v + np.float32(0.0)).view(np.uint32).astype(np.uint64)  # +0.0: -0.0 equals 0.0
    with np.errstate(over="ignore"):
        key = ((b[:, 0] << np.uint64(32)) | b[:, 1]) * np.uint64(0x9E3779B97F4A7C15) ^ b[:, 2] * np.uint64(
            0xC2B2AE3D27D4EB4F)
    order = np.argsort(key)
    hit = np.nonzero(key[order][1:] == key[order][:-1])[0]
    if not len(hit):
        return np.zeros(0, dtype=np.int64)
    # only the few vertices that share a key are compared exactly
    cand = np.sort(order[np.unique(np.r_[hit, hit + 1])])
    _, first, inv = np.unique(v[cand] + np.float32(0.0), axis=0, return_index=True, return_inverse=True)
    return cand[np.arange(len(cand)) != first[inv.ravel()]]


def _separate_collapsed(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Nudge vertices that landed on the same float32 position apart (by ~0.1 um).

    The boolean keeps them as different points; merging them (as reading the
    STL back would) breaks the surface, so each extra copy moves a hair towards
    the middle of its own neighbours instead.
    """
    v = v.astype(np.float32).copy()
    scale = float(np.abs(v).max()) or 1.0
    step = max(1e-5, 16.0 * float(np.spacing(np.float32(scale))))
    for _ in range(4):
        extra = _extra_copies(v)
        if not len(extra):
            break
        nsum = np.zeros((len(v), 3))
        ncnt = np.zeros(len(v))
        for a, b in ((0, 1), (1, 2), (2, 0), (1, 0), (2, 1), (0, 2)):
            np.add.at(nsum, f[:, a], v[f[:, b]])
            np.add.at(ncnt, f[:, a], 1)
        towards = nsum[extra] / np.maximum(ncnt[extra], 1)[:, None] - v[extra]
        towards /= np.maximum(np.linalg.norm(towards, axis=1, keepdims=True), 1e-30)
        v[extra] = v[extra] + (step * towards).astype(np.float32)
        step *= 2
    return v


def from_manifold(man: m3d.Manifold) -> Mesh:
    """Mesh of a manifold, keeping its topology exactly.

    Its vertices are distinct points; any two that round to the same STL
    float are moved apart by a fraction of a micrometre rather than merged, so
    the exported file reads back as the same closed surface.
    """
    mg = man.to_mesh()
    v = np.asarray(mg.vert_properties)[:, :3]
    f = np.asarray(mg.tri_verts)
    return Mesh(_separate_collapsed(v, f), f)


def _grown(man: m3d.Manifold) -> m3d.Manifold:
    c = tuple(np.asarray(man.bounding_box(), float).reshape(2, 3).mean(0))
    return man.translate(tuple(-x for x in c)).scale((1 + FUSE_GROWTH,) * 3).translate(c)


def union(meshes: list[Mesh], fuse_touching: bool = True) -> Mesh:
    mans = [to_manifold(m) for m in meshes if m.n_faces]
    if not mans:
        return Mesh(np.zeros((0, 3)), np.zeros((0, 3)))
    if fuse_touching and len(mans) > 1:
        mans = [_grown(m) for m in mans]
    return from_manifold(m3d.Manifold.batch_boolean(mans, m3d.OpType.Add))


def difference(a: Mesh, cutters: list[Mesh]) -> Mesh:
    man = to_manifold(a)
    cut = [to_manifold(c) for c in cutters if c.n_faces]
    if cut:
        man = man - m3d.Manifold.batch_boolean(cut, m3d.OpType.Add)
    return from_manifold(man)


def intersection(a: Mesh, b: Mesh) -> Mesh:
    return from_manifold(to_manifold(a) ^ to_manifold(b))
