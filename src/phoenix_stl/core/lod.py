"""Light display copy of a heavy mesh (vertex clustering, O(n)).

Only used for drawing and live previews; every edit runs on the full mesh.
"""
from __future__ import annotations

import numpy as np

from .mesh import Mesh, report
from .weld import group_rows

SAMPLE = 200_000


def _approx_area(mesh: Mesh) -> float:
    m = mesh.n_faces
    if m <= SAMPLE:
        return mesh.area()
    idx = np.random.default_rng(0).choice(m, SAMPLE, replace=False)
    t = mesh.vertices[mesh.faces[idx]].astype(np.float64)
    a = 0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1)
    return float(a.sum()) * m / SAMPLE


def cluster(mesh: Mesh, cell: float) -> Mesh:
    lo = mesh.vertices.min(0)
    q = np.floor((mesh.vertices - lo) / np.float32(cell)).astype(np.float64)
    _, inv = group_rows(len(q), lambda ix: q[ix])
    n = int(inv.max()) + 1
    cnt = np.bincount(inv, minlength=n).astype(np.float64)
    verts = np.stack([np.bincount(inv, weights=mesh.vertices[:, k], minlength=n) for k in range(3)], 1)
    verts /= cnt[:, None]
    f = inv[mesh.faces]
    ok = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 2] != f[:, 0])
    return Mesh(verts, f[ok]).compact()


def display_mesh(mesh: Mesh, target_faces: int, progress=None) -> Mesh:
    if mesh.n_faces <= target_faces:
        return mesh
    report(progress, 0.1, "Building preview")
    cell = float(np.sqrt(2.0 * _approx_area(mesh) / target_faces))
    out = mesh
    for _ in range(4):
        out = cluster(mesh, cell)
        if out.n_faces <= 1.25 * target_faces:
            break
        cell *= float(np.sqrt(out.n_faces / target_faces))
    report(progress, 1.0, "Building preview")
    return out
