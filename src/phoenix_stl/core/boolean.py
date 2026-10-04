"""Boolean operations through manifold3d (robust, always-closed results).

manifold3d holds the GIL and needs roughly 500 bytes per triangle, so these
run only in the engine process, after a memory check.
"""
from __future__ import annotations

import manifold3d as m3d
import numpy as np

from .mesh import Mesh
from .weld import weld_mesh

# Parts that only touch (e.g. two halves of a cut) are not fused by an exact
# union: the shared faces survive as zero-thickness leftovers. Growing each
# part by this relative amount (2 micrometres on a 200 mm part, far below a
# 14 micrometre printer pixel) makes them overlap so they fuse cleanly.
FUSE_GROWTH = 1e-5


class NotSolid(Exception):
    """A mesh is not a closed, consistently oriented solid (repair it first)."""


def to_manifold(mesh: Mesh) -> m3d.Manifold:
    man = m3d.Manifold(m3d.Mesh(vert_properties=np.ascontiguousarray(mesh.vertices, dtype=np.float32),
                                tri_verts=np.ascontiguousarray(mesh.faces, dtype=np.uint32)))
    if man.status() != m3d.Error.NoError:
        raise NotSolid(str(man.status()))
    return man


def from_manifold(man: m3d.Manifold) -> Mesh:
    mg = man.to_mesh()
    # Output can repeat a position across internal runs; weld so exported
    # files re-import with the same topology.
    return weld_mesh(Mesh(np.asarray(mg.vert_properties)[:, :3], np.asarray(mg.tri_verts)), 0.0)


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
