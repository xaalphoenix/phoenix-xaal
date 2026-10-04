"""Indexed triangle mesh: the single data type every core module works on."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Process big arrays in blocks so temporary float64 copies stay small.
CHUNK = 2_000_000


def report(progress, frac: float, msg: str = "") -> None:
    if progress is not None:
        progress(min(max(frac, 0.0), 1.0), msg)


@dataclass
class Mesh:
    vertices: np.ndarray  # (N, 3) float32
    faces: np.ndarray  # (M, 3) int32

    def __post_init__(self) -> None:
        self.vertices = np.ascontiguousarray(self.vertices, dtype=np.float32).reshape(-1, 3)
        self.faces = np.ascontiguousarray(self.faces, dtype=np.int32).reshape(-1, 3)

    @property
    def n_vertices(self) -> int:
        return len(self.vertices)

    @property
    def n_faces(self) -> int:
        return len(self.faces)

    @property
    def bounds(self) -> np.ndarray:
        """(2, 3) array: [min, max]."""
        if self.n_vertices == 0:
            return np.zeros((2, 3), dtype=np.float64)
        return np.array([self.vertices.min(0), self.vertices.max(0)], dtype=np.float64)

    @property
    def size(self) -> np.ndarray:
        b = self.bounds
        return b[1] - b[0]

    def copy(self) -> "Mesh":
        return Mesh(self.vertices.copy(), self.faces.copy())

    def face_areas(self) -> np.ndarray:
        return face_metrics(self)[0]

    def area(self) -> float:
        return float(self.face_areas().sum())

    def face_volumes(self) -> np.ndarray:
        """Signed tetra volume of each face (sums to the mesh volume)."""
        return face_metrics(self)[1]

    def volume(self) -> float:
        return float(self.face_volumes().sum())

    def compact(self) -> "Mesh":
        """Drop vertices no face references."""
        used = np.zeros(self.n_vertices, dtype=bool)
        used[self.faces.ravel()] = True
        if used.all():
            return self
        remap = np.cumsum(used, dtype=np.int64) - 1
        return Mesh(self.vertices[used], remap[self.faces].astype(np.int32))

    @staticmethod
    def concatenate(meshes: list["Mesh"]) -> "Mesh":
        meshes = [m for m in meshes if m.n_faces]
        if not meshes:
            return Mesh(np.zeros((0, 3)), np.zeros((0, 3)))
        offsets = np.cumsum([0] + [m.n_vertices for m in meshes[:-1]])
        return Mesh(np.concatenate([m.vertices for m in meshes]),
                    np.concatenate([m.faces + o for m, o in zip(meshes, offsets)]))


def face_metrics(mesh: "Mesh"):
    """(areas, signed volumes) per face, computed on contiguous coordinate arrays."""
    m = mesh.n_faces
    areas = np.empty(m, dtype=np.float64)
    vols = np.empty(m, dtype=np.float64)
    if m == 0:
        return areas, vols
    c = mesh.vertices.astype(np.float64).mean(0)
    cols = [np.ascontiguousarray(mesh.vertices[:, k], dtype=np.float64) - c[k] for k in range(3)]
    for s in range(0, m, CHUNK):
        f = mesh.faces[s:s + CHUNK]
        i0, i1, i2 = f[:, 0], f[:, 1], f[:, 2]
        ax, ay, az = cols[0][i0], cols[1][i0], cols[2][i0]
        bx, by, bz = cols[0][i1], cols[1][i1], cols[2][i1]
        cx, cy, cz = cols[0][i2], cols[1][i2], cols[2][i2]
        # volume: a . (b x c) / 6
        vols[s:s + CHUNK] = (ax * (by * cz - bz * cy) + ay * (bz * cx - bx * cz) + az * (bx * cy - by * cx)) / 6.0
        ux, uy, uz = bx - ax, by - ay, bz - az
        vx, vy, vz = cx - ax, cy - ay, cz - az
        nx, ny, nz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
        areas[s:s + CHUNK] = 0.5 * np.sqrt(nx * nx + ny * ny + nz * nz)
    return areas, vols
