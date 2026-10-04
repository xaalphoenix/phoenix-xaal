"""Rigid/affine transforms of meshes (4x4 matrices, column-vector convention)."""
from __future__ import annotations

import numpy as np

from .mesh import CHUNK, Mesh


def translation(t) -> np.ndarray:
    m = np.eye(4)
    m[:3, 3] = t
    return m


def rotation_about(center, rot3: np.ndarray) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = rot3
    return translation(center) @ m @ translation(-np.asarray(center, float))


def scale_about(center, s) -> np.ndarray:
    m = np.diag([*np.broadcast_to(np.asarray(s, float), 3), 1.0])
    return translation(center) @ m @ translation(-np.asarray(center, float))


def rotation_to(a, b) -> np.ndarray:
    """3x3 rotation taking unit direction a onto unit direction b."""
    a = np.asarray(a, float) / np.linalg.norm(a)
    b = np.asarray(b, float) / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-12:
        if c > 0:
            return np.eye(3)
        # 180 degrees: rotate about any axis perpendicular to a
        axis = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0.0, 1.0, 0.0])
        axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


def transform_points(points: np.ndarray, m: np.ndarray) -> np.ndarray:
    out = np.empty((len(points), 3), dtype=np.float32)
    r, t = m[:3, :3], m[:3, 3]
    for s in range(0, len(points), CHUNK):
        out[s:s + CHUNK] = points[s:s + CHUNK].astype(np.float64) @ r.T + t
    return out


def transform_mesh(mesh: Mesh, m: np.ndarray) -> Mesh:
    """Apply m; a mirroring matrix (negative determinant) keeps faces outward."""
    m = np.asarray(m, dtype=np.float64)
    faces = mesh.faces[:, ::-1] if np.linalg.det(m[:3, :3]) < 0 else mesh.faces
    return Mesh(transform_points(mesh.vertices, m), faces)


def is_identity(m, tol: float = 1e-9) -> bool:
    return bool(np.allclose(np.asarray(m, float), np.eye(4), atol=tol))
