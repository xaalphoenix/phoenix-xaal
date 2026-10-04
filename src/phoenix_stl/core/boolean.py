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
    # Fresh owned copies: manifold3d rejects views into other buffers.
    man = m3d.Manifold(m3d.Mesh(vert_properties=np.array(mesh.vertices, dtype=np.float32, order="C"),
                                tri_verts=np.array(mesh.faces, dtype=np.uint32, order="C")))
    if man.status() != m3d.Error.NoError:
        raise NotSolid(str(man.status()))
    return man


def _raw_mesh(man: m3d.Manifold) -> tuple[Mesh, int]:
    mg = man.to_mesh()
    v = np.asarray(mg.vert_properties)[:, :3]
    # Output can repeat a position across internal runs; weld so exported
    # files re-import with the same topology.
    return weld_mesh(Mesh(v, np.asarray(mg.tri_verts)), 0.0), len(v)


def _is_manifold(mesh: Mesh) -> bool:
    from .analyze import edge_info
    f = mesh.faces
    if np.any((f[:, 0] == f[:, 1]) | (f[:, 1] == f[:, 2]) | (f[:, 2] == f[:, 0])):
        return False
    return bool(np.all(edge_info(mesh).count == 2))


def from_manifold(man: m3d.Manifold) -> Mesh:
    """Mesh of a manifold.

    Distinct points that an STL float cannot tell apart would merge into broken
    edges; if that happens, sub-micron features are simplified away (moving the
    surface by a few micrometres at most, far below a printer pixel) and the
    conversion is retried.
    """
    mesh, n_raw = _raw_mesh(man)
    if mesh.n_vertices == n_raw or _is_manifold(mesh):
        return mesh
    tol = 0.002
    while tol <= 0.06:
        mesh, _ = _raw_mesh(man.simplify(tol))
        if _is_manifold(mesh):
            break
        tol *= 3
    return mesh


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
